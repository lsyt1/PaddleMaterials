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

"""Model package contract for the two registered LiFlow packages.

These tests exercise the exact registry path used by
``build_model_from_name`` (zip -> resolve package dir -> load YAML -> build
model -> strict load ``best.pdparams``).  ``download.get_weights_path_from_url``
is pointed at the local ``artifacts/`` tree so no network or real BCE download
is required on Windows/CI.
"""

from __future__ import annotations

from pathlib import Path

import paddle
import pytest

import ppmat.models as models_mod
from ppmat.models import build_model_from_name

MODEL_NAMES = ("liflow_universal_propagator", "liflow_universal_corrector")


@pytest.fixture(scope="module")
def artifacts_root() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts"


@pytest.fixture
def local_package_server(monkeypatch, artifacts_root: Path):
    class LocalPackageServer:
        def checkpoint(self, model_name: str) -> str:
            return str(artifacts_root / model_name / "checkpoints" / "best.pdparams")

    monkeypatch.setattr(
        models_mod.download,
        "get_weights_path_from_url",
        lambda _: str(artifacts_root),
    )
    return LocalPackageServer()


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_liflow_package_contract(local_package_server, model_name):
    model, config = build_model_from_name(model_name, strict_weights=True)
    assert type(model).__name__ == "LiFlow"
    assert config["Model"]["__class_name__"] == "LiFlow"
    state = paddle.load(local_package_server.checkpoint(model_name))
    assert set(model.state_dict()) == set(state)
    for name, expected in model.state_dict().items():
        assert tuple(state[name].shape) == tuple(expected.shape)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_liflow_package_arch_matches_config(local_package_server, model_name):
    model, config = build_model_from_name(model_name, strict_weights=True)
    params = config["Model"]["__init_params__"]
    assert model.network.num_features == params["num_features"]
    assert model.network.num_layers == params["num_layers"]
    assert model.network.atom_embedding.weight.shape[0] == params["num_elements"]


def test_liflow_registry_urls_are_consistent():
    for name in MODEL_NAMES:
        url = models_mod.MODEL_REGISTRY[name]
        assert url.startswith("https://paddle-org.bj.bcebos.com/paddlematerials/")
        assert url.endswith(f"/{name}.zip")
