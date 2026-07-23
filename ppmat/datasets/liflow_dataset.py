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
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from paddle.io import Dataset
from pymatgen.core.periodic_table import Element

from ppmat.datasets.custom_data_type import ConcatNumpyWarper


class LiFlowDataset(Dataset):
    """LiFlow trajectory-pair dataset following the reference data semantics.

    Each item samples two frames separated by ``time_delay_steps`` from the CSV
    interval. The first endpoint receives the temperature- and mass-conditioned
    Gaussian prior used by LiFlow's propagator, while ``target`` is the physical
    endpoint displacement. Element IDs come directly from the dataset's stable
    ``element_index.npy`` lookup table.
    """

    def __init__(
        self,
        data_path: str,
        index_file: str,
        time_delay_steps: int = 100,
        prior_scale_li: tuple[float, ...] = (1.0, 10.0),
        prior_scale_frame: tuple[float, ...] = (0.316, 3.16),
        seed: int = 42,
        random_time: bool = True,
        in_memory: bool = False,
    ):
        super().__init__()
        self.root = Path(data_path)
        index_path = Path(index_file)
        if not index_path.is_absolute():
            index_path = self.root / index_path
        self.rows = self._read_index(index_path)
        self.time_delay_steps = int(time_delay_steps)
        self.prior_scale_li = np.asarray(prior_scale_li, dtype=np.float32)
        self.prior_scale_frame = np.asarray(prior_scale_frame, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.random_time = random_time
        self.in_memory = in_memory

        self.element_index = np.load(self.root / "element_index.npy")
        self.atomic_numbers = np.load(
            self.root / "atomic_numbers.npy", allow_pickle=True
        ).item()
        self.lattice = np.load(self.root / "lattice.npy", allow_pickle=True).item()
        self.positions = {}
        for temp in sorted({row["temp"] for row in self.rows}):
            archive = np.load(self.root / f"positions_{temp}K.npz")
            self.positions[temp] = dict(archive) if in_memory else archive

    @staticmethod
    def _read_index(path: Path) -> list[dict]:
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        required = {"name", "temp", "t_start", "t_end"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"LiFlow index must contain columns {sorted(required)}")
        parsed = []
        for row in rows:
            item = dict(row)
            item["temp"] = int(float(row["temp"]))
            item["t_start"] = int(row["t_start"])
            item["t_end"] = int(row["t_end"])
            item["prior_Li"] = int(row.get("prior_Li") or 0)
            item["prior_frame"] = int(row.get("prior_frame") or 0)
            parsed.append(item)
        return parsed

    def __len__(self):
        return len(self.rows)

    def _masses(self, atomic_numbers: np.ndarray) -> np.ndarray:
        return np.asarray(
            [float(Element.from_Z(int(number)).atomic_mass) for number in atomic_numbers],
            dtype=np.float32,
        )

    def __getitem__(self, idx):
        row = self.rows[idx]
        last_start = row["t_end"] - self.time_delay_steps - 1
        if last_start < row["t_start"]:
            raise ValueError(
                f"Trajectory interval for {row['name']} is shorter than "
                f"time_delay_steps={self.time_delay_steps}."
            )
        if self.random_time:
            start = int(self.rng.integers(row["t_start"], last_start + 1))
            flow_time = float(self.rng.random())
        else:
            start = row["t_start"]
            flow_time = 0.5
        end = start + self.time_delay_steps

        atomic_numbers = np.asarray(
            self.atomic_numbers[row["name"]], dtype=np.int64
        )
        elements = np.asarray(self.element_index[atomic_numbers], dtype=np.int64)
        trajectory = self.positions[row["temp"]][row["name"]]
        positions_1 = np.asarray(trajectory[start], dtype=np.float32)
        positions_2 = np.asarray(trajectory[end], dtype=np.float32)
        masses = self._masses(atomic_numbers)

        li_scale = self.prior_scale_li[row["prior_Li"]]
        frame_scale = self.prior_scale_frame[row["prior_frame"]]
        prefactor = np.where(atomic_numbers == 3, li_scale, frame_scale)
        scale = prefactor * np.sqrt(8.617333262e-5 * row["temp"] / masses)
        prior = self.rng.normal(scale=scale[:, None], size=positions_1.shape).astype(
            np.float32
        )
        source = positions_1 + prior
        target = positions_2 - source

        return {
            "positions_1": ConcatNumpyWarper(positions_1),
            "positions_2": ConcatNumpyWarper(positions_2),
            "prior": ConcatNumpyWarper(prior),
            "target": ConcatNumpyWarper(target.astype(np.float32)),
            "time": np.asarray(flow_time, dtype=np.float32),
            "temp": np.asarray(row["temp"], dtype=np.float32),
            "elements": ConcatNumpyWarper(elements),
            "atomic_numbers": ConcatNumpyWarper(atomic_numbers),
            "lattice": np.asarray(self.lattice[row["name"]], dtype=np.float32),
            "num_atoms": np.asarray(len(elements), dtype=np.int64),
            "name": row["name"],
            "frame_start": np.asarray(start, dtype=np.int64),
            "frame_end": np.asarray(end, dtype=np.int64),
        }
