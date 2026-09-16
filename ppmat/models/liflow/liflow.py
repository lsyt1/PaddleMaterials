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

"""PaddleMaterials standard wrapper for the LiFlow velocity field.

Training follows reference ``FlowModule.training_step`` (commit e6fc475): given an
endpoint pair ``(start, end)``, a per-node interpolant ``x_t = (1-t)*source + t*end``
is built with ``source = start + prior``, the network is evaluated at
``(start, x_t)`` with the flow time ``t`` and temperature broadcast per node, and
the node-MSE against the precomputed ``velocity`` label is returned.  The label
``velocity = end - source`` is produced by the Dataset so the public Trainer can
recompute the same Metric under the shared ``velocity`` key.

``predict()`` is the future-free path: it consumes an already prepared
``condition_positions`` / ``flow_positions`` pair (plus graph fields and per-node
time/temperature) and returns the raw ``[N, 3]`` velocity, routed through the same
single ``runtime_boundary("forward")`` execution boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import paddle
from paddle import nn

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.liflow.dual_painn import DualPaiNN


def _as_tensor(value: Any) -> paddle.Tensor:
    """Coerce a numpy array (or request field) into a float32/int64 tensor."""
    if isinstance(value, paddle.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return paddle.to_tensor(value)
    return value


def _velocity_loss(prediction: paddle.Tensor, label: paddle.Tensor) -> paddle.Tensor:
    """Sum over the xyz dimensions and average over the nodes (reference loss)."""
    return paddle.mean(paddle.sum((prediction - label) ** 2, axis=-1))


class LiFlow(RuntimeMixin, nn.Layer):
    """Flow-matching model wrapper exposing the standard PaddleMaterials protocol."""

    def __init__(
        self,
        num_features: int = 64,
        num_radial_basis: int = 20,
        num_layers: int = 3,
        num_elements: int = 77,
        r_max: float = 5.0,
        r_offset: float = 0.5,
        ref_temp: float = 1000.0,
        prediction_mode: str = "velocity",
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        if prediction_mode not in {"velocity", "data"}:
            raise ValueError("prediction_mode must be 'velocity' or 'data'")
        self.prediction_mode = prediction_mode
        self.ref_temp = float(ref_temp)
        self.network = DualPaiNN(
            num_features=num_features,
            num_radial_basis=num_radial_basis,
            num_layers=num_layers,
            num_elements=num_elements,
            r_max=r_max,
            r_offset=r_offset,
            ref_temp=ref_temp,
        )

    @runtime_boundary("forward")
    def _runtime_forward(
        self,
        condition_positions: paddle.Tensor,
        flow_positions: paddle.Tensor,
        edge_index: paddle.Tensor,
        shifts: paddle.Tensor,
        elements: paddle.Tensor,
        node_time: paddle.Tensor,
        node_temperature: paddle.Tensor,
    ) -> paddle.Tensor:
        """Compute the ``[N, 3]`` velocity field on an already prepared graph."""
        return self.network(
            condition_positions,
            flow_positions,
            edge_index,
            shifts,
            elements,
            node_time,
            node_temperature,
        )

    def forward(
        self,
        batch_data: dict[str, Any],
        return_loss: bool = True,
        return_prediction: bool = True,
    ) -> dict[str, dict[str, Any]]:
        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."

        start_positions = _as_tensor(batch_data["start_positions"])
        end_positions = _as_tensor(batch_data["end_positions"])
        prior = _as_tensor(batch_data["prior"])

        # time / temperature are per-graph and expanded to per-node via batch index
        flow_time = _as_tensor(batch_data["flow_time"]).reshape([-1])
        temperature = _as_tensor(batch_data["temperature"]).reshape([-1])
        if "batch_index" in batch_data:
            batch_index = _as_tensor(batch_data["batch_index"]).astype("int64")
        else:
            batch_index = paddle.zeros([start_positions.shape[0]], dtype="int64")

        # Interpolate x_t = (1-t)*source + t*end ; source = start + prior
        source = start_positions + prior
        t_node = flow_time[batch_index][:, None]
        flow_positions = (1.0 - t_node) * source + t_node * end_positions
        node_time = flow_time[batch_index]
        node_temperature = temperature[batch_index]

        edge_index = _as_tensor(batch_data["edge_index"]).astype("int64")
        shifts = _as_tensor(batch_data["shifts"])
        elements = _as_tensor(batch_data["elements"]).astype("int64")

        prediction = self._runtime_forward(
            start_positions,
            flow_positions,
            edge_index,
            shifts,
            elements,
            node_time,
            node_temperature,
        )

        result = {"loss_dict": {}, "pred_dict": {}}
        if return_prediction:
            result["pred_dict"]["velocity"] = prediction
        if return_loss:
            label = _as_tensor(batch_data["velocity"])
            loss_velocity = _velocity_loss(prediction, label)
            result["loss_dict"]["velocity"] = loss_velocity
            result["loss_dict"]["loss"] = loss_velocity
        return result

    @paddle.no_grad()
    def predict(self, batch_data: dict[str, Any]) -> dict[str, paddle.Tensor]:
        """Return the raw ``[N, 3]`` velocity without any future endpoint."""
        condition_positions = _as_tensor(batch_data["condition_positions"])
        flow_positions = _as_tensor(batch_data["flow_positions"])
        edge_index = _as_tensor(batch_data["edge_index"]).astype("int64")
        shifts = _as_tensor(batch_data["shifts"])
        elements = _as_tensor(batch_data["elements"]).astype("int64")
        node_time = _as_tensor(batch_data["node_time"]).reshape([-1])
        node_temperature = _as_tensor(batch_data["node_temperature"]).reshape([-1])
        velocity = self._runtime_forward(
            condition_positions,
            flow_positions,
            edge_index,
            shifts,
            elements,
            node_time,
            node_temperature,
        )
        return {"velocity": velocity}
