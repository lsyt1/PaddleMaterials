# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Future-free LiFlow trajectory generation.

``IntegratorPredictor`` composes a propagator and a corrector velocity-field
model (both LiFlow wrappers) and integrates a trajectory from an initial
structure with no reference to future frames or labels.  It intentionally uses
composition rather than inheriting ``BasePredictor``, which manages a single
model.  Both models are configured on the same execution backend and run in eval
mode.

Generation is performed entirely outside the model: each MD step integrates the
flow field over ``flow_steps`` substeps with an Euler or Heun solver, and applies
a mass-weighted centroid correction.  The public ``run`` signature accepts only
the initial geometry, temperature and seed --- it has no ``end_positions``
argument.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import paddle
from ase import data as ase_data

from ppmat.models import build_model
from ppmat.models.liflow.geometry import get_neighbor_list_batch
from ppmat.utils import logger
from ppmat.utils import save_load

DEFAULT_CUTOFF = 5.0


def _element_indices(atomic_numbers: np.ndarray) -> np.ndarray:
    """Map atomic numbers to 0-based element indices (periodic-table order)."""
    return np.asarray(atomic_numbers, dtype=np.int64) - 1


class IntegratorPredictor:
    """Generate a future-free trajectory by integrating two LiFlow models."""

    def __init__(
        self,
        propagator_model: Optional[paddle.nn.Layer] = None,
        corrector_model: Optional[paddle.nn.Layer] = None,
        *,
        propagator_config: Optional[str] = None,
        propagator_checkpoint: Optional[str] = None,
        corrector_config: Optional[str] = None,
        corrector_checkpoint: Optional[str] = None,
        cutoff: float = DEFAULT_CUTOFF,
        periodic: bool = True,
        ref_temp: float = 1000.0,
    ):
        if propagator_model is None:
            propagator_model = self._load_model(
                propagator_config, propagator_checkpoint, "propagator"
            )
        if corrector_model is None:
            corrector_model = self._load_model(
                corrector_config, corrector_checkpoint, "corrector"
            )

        self.propagator = propagator_model
        self.corrector = corrector_model
        self.cutoff = float(cutoff)
        self.periodic = bool(periodic)
        self.ref_temp = float(ref_temp)

        # Reuse the shared backend configuration on both models so devices and
        # execution backends stay consistent.
        from ppmat.utils.execution import configure_execution_backend

        backend = "eager"
        for model in (self.propagator, self.corrector):
            configure_execution_backend(
                model,
                backend,
                world_size=paddle.distributed.get_world_size(),
                owner="IntegratorPredictor",
            )
            model.eval()

    @staticmethod
    def _load_model(
        config_path: Optional[str],
        checkpoint_path: Optional[str],
        role: str,
    ) -> paddle.nn.Layer:
        if config_path is None or checkpoint_path is None:
            raise ValueError(
                f"{role}: both config_path and checkpoint_path are required when "
                "no model instance is provided."
            )
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        model_config = config.get("Model", None)
        assert model_config is not None, "Model config must be provided."
        model = build_model(model_config)
        logger.info(f"Loading {role} weights from {checkpoint_path} with strict=True")
        save_load.load_pretrain(model, checkpoint_path, strict=True)
        return model

    # -- velocity field --------------------------------------------------------

    def _velocity_field(
        self,
        condition_positions: np.ndarray,
        flow_positions: np.ndarray,
        node_time: float,
        atomic_numbers: np.ndarray,
        lattice: np.ndarray,
        temperature: float,
        model,
    ) -> np.ndarray:
        """Return the ``[N, 3]`` velocity of the system at node_time.

        The neighbourhood graph is built over the ``N`` system nodes from the
        current flow geometry; both the condition and the flow positions share
        this node indexing.  The model returns one velocity per node.
        """
        assert condition_positions.shape == flow_positions.shape
        num_atoms = condition_positions.shape[0]
        elements = _element_indices(atomic_numbers)
        edge_index, shifts = get_neighbor_list_batch(
            [flow_positions], lattice, cutoff=self.cutoff, periodic=self.periodic
        )
        batch_data = {
            "condition_positions": paddle.to_tensor(condition_positions.astype(np.float32)),
            "flow_positions": paddle.to_tensor(flow_positions.astype(np.float32)),
            "edge_index": paddle.to_tensor(edge_index.astype(np.int64)),
            "shifts": paddle.to_tensor(shifts.astype(np.float32)),
            "elements": paddle.to_tensor(elements.astype(np.int64)),
            "node_time": paddle.full([num_atoms], node_time, dtype="float32"),
            "node_temperature": paddle.full([num_atoms], temperature, dtype="float32"),
        }
        with paddle.no_grad():
            velocity = model.predict(batch_data)["velocity"]
        return velocity.numpy().astype(np.float32)

    def _integrate_step(
        self,
        condition: np.ndarray,
        flow_steps: int,
        solver: str,
        atomic_numbers: np.ndarray,
        lattice: np.ndarray,
        temperature: float,
        *,
        corrector_every: int,
    ) -> np.ndarray:
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        dt = 1.0 / int(flow_steps)
        flow = np.array(condition, dtype=np.float32)
        t = 0.0
        for _ in range(int(flow_steps)):
            velocity = self._velocity_field(
                condition, flow, t, atomic_numbers, lattice, temperature, self.propagator
            )
            if solver == "euler":
                flow = flow + velocity * dt
            else:  # heun
                predictor_positions = flow + velocity * dt
                next_velocity = self._velocity_field(
                    condition,
                    predictor_positions,
                    t + dt,
                    atomic_numbers,
                    lattice,
                    temperature,
                    self.propagator,
                )
                flow = flow + 0.5 * (velocity + next_velocity) * dt
            t += dt
        if corrector_every:
            flow = self._refine_prediction(
                flow, atomic_numbers, lattice, temperature
            )
        return flow

    def _refine_prediction(
        self,
        flow: np.ndarray,
        atomic_numbers: np.ndarray,
        lattice: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Refine the integrated frame with the corrector velocity field."""
        for t in (0.5, 0.0):
            velocity = self._velocity_field(
                flow, flow, t, atomic_numbers, lattice, temperature, self.corrector
            )
            flow = flow + velocity
        return flow

    @staticmethod
    def _apply_centroid_correction(
        positions: np.ndarray, atomic_numbers: np.ndarray
    ) -> np.ndarray:
        """Remove the mass-weighted center-of-mass drift."""
        masses = ase_data.atomic_masses[atomic_numbers].astype(np.float32)
        total_mass = masses.sum()
        centroid = (positions * masses[:, None]).sum(axis=0) / total_mass
        return positions - centroid[None, :]

    def run(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        lattice: np.ndarray,
        temperature: float,
        *,
        steps: int = 25,
        flow_steps: int = 10,
        solver: str = "euler",
        corrector_every: int = 1,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate ``[generated_steps + 1, N, 3]`` trajectory from initial state."""
        positions = np.asarray(positions, dtype=np.float32)
        atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)
        lattice = np.asarray(lattice, dtype=np.float32)
        trajectory = [positions.copy()]
        current = positions.copy()
        for step in range(int(steps)):
            flow = self._integrate_step(
                current,
                flow_steps=flow_steps,
                solver=solver,
                atomic_numbers=atomic_numbers,
                lattice=lattice,
                temperature=temperature,
                corrector_every=corrector_every,
            )
            if not np.isfinite(flow).all():
                raise ValueError(
                    "Trajectory generation diverged (non-finite positions)."
                )
            current = self._apply_centroid_correction(flow, atomic_numbers)
            trajectory.append(current.copy())
        return np.stack(trajectory, axis=0)