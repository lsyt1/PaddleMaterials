# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a future-free LiFlow trajectory from an initial structure.

Input is an ``.npz`` containing ``positions``, ``atomic_numbers`` and
``lattice``; the target ``temperature`` comes from the inference config (or a
``--temperature`` override).  Outputs are ``trajectory.npy``,
``trajectory.xyz`` and ``run_metadata.json`` under ``--output_dir``.
"""

import argparse
import json
import os
import os.path as osp
from datetime import datetime

import numpy as np
from omegaconf import OmegaConf


def _write_xyz(path: str, trajectory: np.ndarray, atomic_numbers: np.ndarray):
    """Write the trajectory using ASE's xyz writer."""
    from ase import Atoms
    from ase.io import write as ase_write

    frames = [
        Atoms(numbers=atomic_numbers, positions=positions, pbc=False)
        for positions in trajectory
    ]
    ase_write(path, frames, format="xyz")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument(
        "--output_dir", "-o", type=str, default="output/liflow_prediction"
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--flow_steps", type=int, default=None)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)
    predict_cfg = config.get("Predict", {}) or {}
    overrides = {
        "temperature": args.temperature,
        "steps": args.steps,
        "flow_steps": args.flow_steps,
        "solver": args.solver,
        "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is None:
            overrides[key] = predict_cfg.get(key, default_by_key(key))
    temperature = float(overrides["temperature"])

    # load the initial structure
    data = np.load(args.input)
    positions = np.asarray(data["positions"], dtype=np.float32)
    atomic_numbers = np.asarray(data["atomic_numbers"], dtype=np.int64)
    lattice = np.asarray(data["lattice"], dtype=np.float32)

    from ppmat.predictor import IntegratorPredictor

    predictor = IntegratorPredictor(
        propagator_config=predict_cfg["propagator_config"],
        propagator_checkpoint=predict_cfg["propagator_checkpoint"],
        corrector_config=predict_cfg["corrector_config"],
        corrector_checkpoint=predict_cfg["corrector_checkpoint"],
        cutoff=predict_cfg.get("cutoff", 5.0),
        periodic=predict_cfg.get("periodic", True),
    )

    trajectory = predictor.run(
        positions,
        atomic_numbers,
        lattice,
        temperature,
        steps=int(overrides["steps"]),
        flow_steps=int(overrides["flow_steps"]),
        solver=overrides["solver"],
        corrector_every=int(predict_cfg.get("corrector_every", 1)),
        seed=int(overrides["seed"]),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(osp.join(args.output_dir, "trajectory.npy"), trajectory)
    _write_xyz(osp.join(args.output_dir, "trajectory.xyz"), trajectory, atomic_numbers)
    metadata = {
        "input": args.input,
        "temperature": temperature,
        "steps": int(overrides["steps"]),
        "flow_steps": int(overrides["flow_steps"]),
        "solver": overrides["solver"],
        "seed": int(overrides["seed"]),
        "trajectory_shape": list(trajectory.shape),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    with open(osp.join(args.output_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved trajectory to {osp.join(args.output_dir, 'trajectory.npy')}")


def default_by_key(key: str):
    return {
        "temperature": 800.0,
        "steps": 25,
        "flow_steps": 10,
        "solver": "euler",
        "seed": 42,
    }[key]


if __name__ == "__main__":
    main()
