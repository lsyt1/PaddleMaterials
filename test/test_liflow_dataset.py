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

"""LiFlow dataset and collation tests.

Task 6 covers the variable-size trajectory ``LiFlowCollator``.  Later tasks
extend this module with the deterministic cached ``LiFlowDataset`` and the
four stochastic priors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppmat.datasets.collate_fn import LiFlowCollator
from ppmat.datasets.liflow_dataset import LiFlowDataset
from ppmat.models.liflow.prior import AdaptiveMaxwellBoltzmannPrior


@pytest.fixture()
def mini_dataset_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "liflow" / "dataset_mini"


def make_sample(num_atoms, edge_index, temperature=800.0, flow_time=0.5):
    rng = np.random.default_rng(0)
    start = rng.normal(size=(num_atoms, 3)).astype(np.float32)
    end = rng.normal(size=(num_atoms, 3)).astype(np.float32)
    prior = rng.normal(size=(num_atoms, 3)).astype(np.float32)
    return {
        "start_positions": start,
        "end_positions": end,
        "prior": prior,
        "velocity": (end - start - prior).astype(np.float32),
        "displacement": rng.normal(size=(num_atoms, 3)).astype(np.float32),
        "elements": np.arange(num_atoms, dtype=np.int64) % 8,
        "atomic_numbers": (np.arange(num_atoms) + 1).astype(np.int64),
        "edge_index": np.asarray(edge_index, dtype=np.int64),
        "shifts": rng.normal(size=(edge_index.shape[1], 3)).astype(np.float32),
        "temperature": temperature,
        "flow_time": flow_time,
        "lattice": np.eye(3) * 10.0,
        "name": "sample",
    }


def test_liflow_collator_offsets_edges():
    first = make_sample(num_atoms=2, edge_index=np.array([[0, 1], [1, 0]]))
    second = make_sample(num_atoms=3, edge_index=np.array([[0, 2], [2, 0]]))
    batch = LiFlowCollator()([first, second])

    np.testing.assert_array_equal(batch["num_atoms"], np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(
        batch["batch_index"], np.array([0, 0, 1, 1, 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        batch["edge_index"][:, 2:], np.array([[2, 4], [4, 2]], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        batch["edge_index"][:, :2], np.array([[0, 1], [1, 0]], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        batch["temperature"], np.array([800.0, 800.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        batch["flow_time"], np.array([0.5, 0.5], dtype=np.float32)
    )
    assert batch["start_positions"].shape == (5, 3)
    assert batch["shifts"].shape == (4, 3)
    assert batch["lattice"].shape == (2, 3, 3)
    assert batch["elements"].dtype == np.int64


def test_liflow_collator_node_fields_preserve_dtype():
    first = make_sample(num_atoms=2, edge_index=np.array([[0, 1], [1, 0]]))
    second = make_sample(num_atoms=2, edge_index=np.array([[0, 1], [1, 0]]))
    batch = LiFlowCollator()([first, second])
    for field in ("elements", "atomic_numbers"):
        assert batch[field].dtype == np.int64
        assert batch[field].shape[0] == 4


def test_liflow_collator_rejects_empty_batch():
    try:
        LiFlowCollator()([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty batch must raise ValueError")


def test_liflow_collator_rejects_empty_node():
    empty = make_sample(num_atoms=0, edge_index=np.zeros((2, 0), dtype=np.int64))
    try:
        LiFlowCollator()([empty])
    except ValueError:
        pass
    else:
        raise AssertionError("zero-node sample must raise ValueError")


def test_liflow_collator_rejects_inconsistent_fields():
    first = make_sample(num_atoms=2, edge_index=np.array([[0, 1], [1, 0]]))
    second = make_sample(num_atoms=2, edge_index=np.array([[0, 1], [1, 0]]))
    del second["velocity"]
    try:
        LiFlowCollator()([first, second])
    except ValueError:
        pass
    else:
        raise AssertionError("samples with different fields must raise ValueError")


# -- Task 7: deterministic cached dataset and seeded priors -------------------


def test_validation_sample_is_deterministic(mini_dataset_path):
    first = LiFlowDataset(path=mini_dataset_path, split="val", seed=42)[0]
    second = LiFlowDataset(path=mini_dataset_path, split="val", seed=42)[0]
    for key in (
        "start_positions",
        "end_positions",
        "prior",
        "flow_time",
        "edge_index",
        "shifts",
    ):
        np.testing.assert_array_equal(first[key], second[key])


def test_train_validation_test_names_are_disjoint(mini_dataset_path):
    splits = {
        name: set(LiFlowDataset(path=mini_dataset_path, split=name).sample_ids)
        for name in ("train", "val", "test")
    }
    assert splits["train"].isdisjoint(splits["val"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["val"].isdisjoint(splits["test"])


def test_cache_reused_without_rebuilding(mini_dataset_path, monkeypatch):
    LiFlowDataset(path=mini_dataset_path, split="val", overwrite=True)
    monkeypatch.setattr(
        LiFlowDataset,
        "_build_cache",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rebuilt")),
    )
    # Cache already matches the frozen split + config, so it must not rebuild.
    LiFlowDataset(path=mini_dataset_path, split="val", overwrite=False)


def test_adaptive_maxwell_boltzmann_prior_is_seeded():
    first = AdaptiveMaxwellBoltzmannPrior(scale=[[1.0, 10.0], [0.316, 3.16]], seed=42)
    second = AdaptiveMaxwellBoltzmannPrior(scale=[[1.0, 10.0], [0.316, 3.16]], seed=42)
    kwargs = {
        "temperature": 800.0,
        "atomic_numbers": np.array([3, 8], dtype=np.int64),
        "masses": np.array([6.94, 15.999], dtype=np.float64),
        "scale_Li_index": 1,
        "scale_frame_index": 0,
        "shape": (2, 3),
    }
    np.testing.assert_array_equal(first.sample(**kwargs), second.sample(**kwargs))
