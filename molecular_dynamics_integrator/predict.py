# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from typing import Optional

import numpy as np
import paddle
from omegaconf import OmegaConf

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.liflow_dataset import LiFlowDataset
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.utils import logger
from ppmat.utils import save_load


class IntegratorPredictor:
    """Molecular dynamics integrator predictor for LiFlow-style models.

    Supports two initialization modes:

    1. **Automatic Model Loading**
       Specify ``model_name`` (and optional ``weights_name``) to download and load
       pretrained weights from ``MODEL_REGISTRY``.

    2. **Custom Model Loading**
       Provide ``config_path`` and ``checkpoint_path`` to load a local checkpoint.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        if model_name is None:
            assert (
                config_path is not None and checkpoint_path is not None
            ), "config_path and checkpoint_path must be provided when model_name is None."

            logger.info(f"Loading model from {config_path} and {checkpoint_path}.")
            config = OmegaConf.load(config_path)
            config = OmegaConf.to_container(config, resolve=True)
            model_config = config.get("Model", None)
            assert model_config is not None, "Model config must be provided."
            model = build_model(model_config)
            save_load.load_pretrain(model, checkpoint_path)
        else:
            logger.info("Since model_name is given, downloading it...")
            model, config = build_model_from_name(model_name, weights_name)

        self.model = model
        self.config = config
        self.model.eval()

        predict_config = config.get("Predict") or {}
        self.predict_config = predict_config
        self.eval_with_no_grad = predict_config.get("eval_with_no_grad", True)

    def from_dataset_sample(
        self,
        data_path: Optional[str] = None,
        index_file: Optional[str] = None,
        sample_index: Optional[int] = None,
    ):
        """Run prediction on one trajectory pair from LiFlowDataset."""
        path = (
            data_path
            or self.predict_config.get("path")
            or self.predict_config.get("data_path")
        )
        index_file = index_file or self.predict_config.get("index_file")
        if path is None or index_file is None:
            raise ValueError("path/data_path and index_file must be provided.")

        if sample_index is None:
            sample_index = int(self.predict_config.get("sample_index", 0))

        dataset = LiFlowDataset(
            path=path,
            index_file=index_file,
            time_delay_steps=self.predict_config.get("time_delay_steps", 100),
            prior_scale_li=self.predict_config.get("prior_scale_li", (1.0, 10.0)),
            prior_scale_frame=self.predict_config.get(
                "prior_scale_frame", (0.316, 3.16)
            ),
            seed=self.predict_config.get("seed", 42),
            random_time=False,
        )
        if not 0 <= sample_index < len(dataset):
            raise IndexError(
                f"sample_index {sample_index} outside dataset of size {len(dataset)}"
            )

        sample = dataset[sample_index]
        batch = DefaultCollator()([sample])
        for key, value in list(batch.items()):
            if isinstance(value, np.ndarray):
                batch[key] = paddle.to_tensor(value)

        if self.eval_with_no_grad:
            with paddle.no_grad():
                prediction = self.model.predict(batch)
        else:
            prediction = self.model.predict(batch)

        velocity = prediction["velocity"].numpy()
        target = prediction["target"].numpy()
        mse = float(((velocity - target) ** 2).sum(axis=-1).mean())
        result = {
            "name": sample["name"],
            "frame_start": int(sample["frame_start"]),
            "frame_end": int(sample["frame_end"]),
            "num_atoms": len(sample["elements"]),
            "mse": mse,
            "velocity": velocity,
            "target": target,
        }
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Pretrained model name registered in MODEL_REGISTRY.",
    )
    parser.add_argument(
        "--weights_name",
        type=str,
        default=None,
        help="Weights name, e.g., best.pdparams, latest.pdparams.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override the trajectory data directory.",
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default=None,
        help="Override the trajectory index CSV.",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Sample index in the dataset.",
    )
    args = parser.parse_args()

    predictor = IntegratorPredictor(
        model_name=args.model_name,
        weights_name=args.weights_name,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
    )
    result = predictor.from_dataset_sample(
        data_path=args.data_path,
        index_file=args.index_file,
        sample_index=args.sample_index,
    )
    print(
        f"sample={result['name']} frames={result['frame_start']}->"
        f"{result['frame_end']} atoms={result['num_atoms']} mse={result['mse']:.8f}"
    )
    print("predicted_velocity:")
    print(result["velocity"])
