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

"""Geometry helpers for the LiFlow Paddle port.

The tensor routines mirror ``liflow/utils/geometry.py`` at the reference commit
``e6fc475361d046865f12cae1aee11c4f56c48d87``.  The neighbor list uses the ASE
dependency already carried by PaddleMaterials and is normalized with
``np.unique`` so that it matches the reference ``vesin.NeighborList(full_list=True)``
output that the fixtures were generated with.
"""

from __future__ import annotations

import numpy as np
import paddle

from ppmat.utils.scatter import scatter_sum  # noqa: F401  (re-exported for callers)

__all__ = [
    "get_unit_vectors_and_lengths",
    "get_neighbor_list_batch",
    "unwrap_trajectory",
]


def get_unit_vectors_and_lengths(
    positions,  # [n_nodes, 3]
    edge_index,  # [2, n_edges]
    shifts,  # [n_edges, 3]
    eps: float = 1e-8,
):
    """Return per-edge unit vectors and lengths for a single configuration."""
    idx_i, idx_j = edge_index[0], edge_index[1]  # [n_edges]
    vectors = positions[idx_j] - positions[idx_i] + shifts  # [n_edges, 3]
    lengths = paddle.linalg.norm(vectors, axis=-1, keepdim=True)  # [n_edges, 1]
    unit_vectors = vectors / (lengths + eps)
    return unit_vectors, lengths


def get_neighbor_list_batch(
    positions_batch,  # [n_configs, n_atoms, 3]
    lattice,  # [3, 3]
    cutoff: float = 5.0,
    periodic: bool = True,
):
    """Construct a neighbor list for a batch as the union of each config's list.

    Each config contributes its directed neighbor pairs; duplicates across the
    configuration endpoints are removed with ``np.unique`` so the result is
    reproducible and matches the reference ``vesin``-based list.
    """
    from ase import Atoms
    from ase import neighborlist

    edge_attrs_all = []
    for positions in positions_batch:
        atoms = Atoms(
            numbers=np.ones(positions.shape[0], dtype=np.int64),
            positions=np.asarray(positions, dtype=np.float64),
            cell=np.asarray(lattice, dtype=np.float64),
            pbc=periodic,
        )
        src, dst, image = neighborlist.neighbor_list(
            "ijS", atoms, cutoff, self_interaction=False
        )
        # image: integer cell images [E, 3]; combine into the canonical triplet.
        edge_attrs_all.append(
            np.hstack((src[:, None], dst[:, None], image)).astype(np.int64)
        )
    edge_attrs = np.unique(np.vstack(edge_attrs_all), axis=0)
    edge_index = edge_attrs[:, :2].T
    shifts = edge_attrs[:, 2:] @ lattice  # convert images to Cartesian
    return edge_index, shifts


def unwrap_trajectory(
    positions,  # [n_frames, n_atoms, 3]
    lattice,  # [3, 3]
):
    """Unwrap a periodic trajectory; frames must jump less than half a box.

    Kept as a NumPy routine because it runs on generated trajectories outside the
    compiled tensor boundary.
    """
    positions = np.asarray(positions)
    frac_diff = np.diff(positions, axis=0) @ np.linalg.inv(lattice)
    frac_diff_unwrap = frac_diff - np.floor(frac_diff + 0.5)
    diff_unwrap = frac_diff_unwrap @ lattice
    return np.cumsum(
        np.concatenate((positions[0, None], diff_unwrap), axis=0), axis=0
    )