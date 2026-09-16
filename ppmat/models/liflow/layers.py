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

"""A faithful PaddlePort of the reference LiFlow equivariant layers.

The translation mirrors ``learningmatter-mit/liflow`` (reference commit
``e6fc475361d046865f12cae1aee11c4f56c48d87``) ``liflow/model/layers.py``.
Parameter and buffer names, tensor shapes and computation order are preserved so
that converted checkpoints load with ``strict_weights=True``.
"""

from __future__ import annotations

import math

import paddle
import paddle.nn as nn

from ppmat.utils.scatter import scatter_sum

__all__ = [
    "GaussianFourierBasis",
    "BesselBasis",
    "CosineCutoff",
    "DualMessageBlock",
    "UpdateBlock",
    "GatedEquivariantBlock",
]


# Adapted from yang-song/score_sde_pytorch
class GaussianFourierBasis(nn.Layer):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, num_basis: int):
        super().__init__()
        assert num_basis % 2 == 0
        self.num_basis = num_basis
        freqs = paddle.randn([num_basis // 2]) * 2 * math.pi
        self.register_buffer(name="freqs", tensor=freqs)

    def forward(self, x):
        args = self.freqs * x[..., None]
        emb = paddle.concat((paddle.sin(args), paddle.cos(args)), axis=-1)
        return emb


class BesselBasis(nn.Layer):
    def __init__(self, num_basis: int, r_max: float):
        super().__init__()
        self.num_basis = num_basis
        freqs = paddle.arange(1, num_basis + 1, dtype=paddle.float32) * math.pi / r_max
        prefactor = paddle.to_tensor(math.sqrt(2.0 / r_max), dtype=paddle.float32)
        self.register_buffer(name="freqs", tensor=freqs)
        self.register_buffer(name="prefactor", tensor=prefactor)

    def forward(self, x):
        args = self.freqs * x[..., None]
        rbf = self.prefactor * paddle.sin(args) / x[..., None]
        return rbf


class CosineCutoff(nn.Layer):
    def __init__(self, r_max: float):
        super().__init__()
        self.register_buffer(
            name="r_max", tensor=paddle.to_tensor(r_max, dtype=paddle.float32)
        )

    def forward(self, x):
        x_cut = 0.5 * (1.0 + paddle.cos(x * math.pi / self.r_max))
        x_cut = x_cut * (x < self.r_max).astype("float32")
        return x_cut


class DualMessageBlock(nn.Layer):
    def __init__(self, num_features: int, num_radial_basis: int):
        super().__init__()
        self.num_features = num_features

        self.mlp_phi = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 4),
        )
        self.linear_W = nn.Linear(num_radial_basis, num_features * 4)

    def forward(
        self,
        s,  # [n_nodes, 1, n_feats]
        v,  # [n_nodes, 3, n_feats]
        radial_embeddings_1,  # [n_edges, 1, num_radial_basis]
        radial_embeddings_2,  # [n_edges, 1, num_radial_basis]
        f_cut_1,  # [n_edges, 1]
        f_cut_2,  # [n_edges, 1]
        unit_vectors_1,  # [n_edges, 3]
        unit_vectors_2,  # [n_edges, 3]
        edge_index,  # [2, n_edges]
    ):
        idx_i, idx_j = edge_index[0], edge_index[1]
        n_nodes = s.shape[0]
        phi = self.mlp_phi(s)
        W = (
            self.linear_W(radial_embeddings_1) * f_cut_1[..., None]
            + self.linear_W(radial_embeddings_2) * f_cut_2[..., None]
        )
        x = phi[idx_j] * W
        x_s, x_vv, x_vs_1, x_vs_2 = paddle.chunk(x, 4, axis=-1)
        ds = scatter_sum(x_s, idx_i, dim=0, dim_size=n_nodes)
        x_v = (
            v[idx_j] * x_vv
            + x_vs_1 * unit_vectors_1[..., None]
            + x_vs_2 * unit_vectors_2[..., None]
        )
        dv = scatter_sum(x_v, idx_i, dim=0, dim_size=n_nodes)
        return s + ds, v + dv


class UpdateBlock(nn.Layer):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.mlp_a = nn.Sequential(
            nn.Linear(num_features * 2, num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features * 3),
        )
        self.linear_UV = nn.Linear(num_features, num_features * 2, bias_attr=False)

    def forward(self, s, v):
        U_v, V_v = paddle.chunk(self.linear_UV(v), 2, axis=-1)
        a = self.mlp_a(
            paddle.concat(
                (s, V_v.norm(p=2, axis=-2, keepdim=True)),
                axis=-1,
            )
        )
        a_vv, a_sv, a_ss = paddle.chunk(a, 3, axis=-1)
        dv = a_vv * U_v
        ds = a_ss + a_sv * paddle.sum(U_v * V_v, axis=-2, keepdim=True)
        return s + ds, v + dv


class GatedEquivariantBlock(nn.Layer):
    """Modified gated equivariant block to output a single vector.

    See PaiNN paper Fig. 3 or the schnetpack.nn.equivariant module.
    """

    def __init__(
        self,
        num_scalar_inputs: int,
        num_vector_inputs: int,
    ):
        super().__init__()
        self.num_scalar_inputs = num_scalar_inputs
        self.num_vector_inputs = num_vector_inputs
        self.linear_v = nn.Linear(num_vector_inputs, 2, bias_attr=False)
        self.mlp_s = nn.Sequential(
            nn.Linear(num_scalar_inputs + 1, num_scalar_inputs + 1),
            nn.SiLU(),
            nn.Linear(num_scalar_inputs + 1, 1),
        )

    def forward(self, s, v):
        W_v1, W_v2 = paddle.chunk(self.linear_v(v), 2, axis=-1)
        s_out = self.mlp_s(
            paddle.concat((s, W_v2.norm(axis=-2, keepdim=True)), axis=-1)
        )
        v_out = W_v1 * s_out
        return v_out
