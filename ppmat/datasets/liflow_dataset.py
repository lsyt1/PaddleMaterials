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

from __future__ import absolute_import
from __future__ import annotations

import csv
import os.path as osp
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from paddle.io import Dataset
from pymatgen.core.periodic_table import Element

from ppmat.datasets.custom_data_type import ConcatNumpyWarper
from ppmat.utils import logger


class LiFlowDataset(Dataset):
    """LiFlow Dataset Handler

    Utilities for loading and processing the LiFlow crystalline-transport trajectory
    dataset used by the LiFlow flow-matching models.

    **Dataset Overview**
    - **Source**: https://zenodo.org/records/14889658
      (DOI: https://doi.org/10.5281/zenodo.14889658)
    - **Reference code**: https://github.com/learningmatter-mit/liflow
    - Includes the universal MLIP set and LGPS. LPS data are available from the
      original authors upon request.

    **Data Format**
    After extracting the Zenodo archive into ``path``, the directory should contain:

    | File | Description |
    |------|-------------|
    | `element_index.npy` | Element lookup table, shape `[n_elements]` |
    | `atomic_numbers.npy` | Dict of atomic numbers keyed by structure name |
    | `lattice.npy` | Dict of lattice matrices keyed by structure name |
    | `positions_{temp}K.npz` | Trajectory positions keyed by structure name |
    | `{train,test}_{temp}K.csv` | Trajectory index CSV files |

    The CSV index contains columns such as `name`, `temp`, `t_start`, `t_end`,
    `prior_Li`, and `prior_frame`.

    Each sample returns a trajectory pair separated by ``time_delay_steps``. The first
    endpoint receives a temperature- and mass-conditioned Gaussian prior used by the
    LiFlow propagator, and ``target`` is the displacement from the noised source to the
    physical endpoint.

    Args:
        path (str): Root directory of the LiFlow dataset. If the directory does not
            exist, a download hint for the Zenodo record is raised.
        index_file (str): Index CSV file name or absolute path. Relative paths are
            resolved under ``path``. Defaults to ``train_800K.csv``.
        time_delay_steps (int, optional): Frame gap between the two endpoints.
            Defaults to 100.
        prior_scale_li (Tuple[float, ...] or list, optional): Prior scale for Li atoms
            indexed by ``prior_Li``. Defaults to ``(1.0, 10.0)``.
        prior_scale_frame (Tuple[float, ...] or list, optional): Prior scale for frame
            atoms indexed by ``prior_frame``. Defaults to ``(0.316, 3.16)``.
        seed (int, optional): Random seed used for frame sampling and prior noise.
            Defaults to 42.
        random_time (bool, optional): Whether to randomly sample the start frame and
            flow time. Defaults to True.
        in_memory (bool, optional): Whether to load trajectory archives fully into
            memory. Defaults to False.
        **kwargs: Reserved for compatibility with other dataset configs.
    """

    name = "liflow"
    # Official dataset record. The archive is multi-part on Zenodo and is not yet
    # mirrored as a single BCE zip; download from the URL below when path is missing.
    url = "https://zenodo.org/records/14889658"
    doi = "https://doi.org/10.5281/zenodo.14889658"

    def __init__(
        self,
        path: str = "./data/liflow",
        index_file: str = "train_800K.csv",
        time_delay_steps: int = 100,
        prior_scale_li: Union[Tuple[float, ...], List[float]] = (1.0, 10.0),
        prior_scale_frame: Union[Tuple[float, ...], List[float]] = (0.316, 3.16),
        seed: int = 42,
        random_time: bool = True,
        in_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        if not osp.exists(path):
            raise FileNotFoundError(
                f"LiFlow dataset path not found: {path}. "
                f"Please download the official data from {self.url} "
                f"(DOI: {self.doi}), merge all data.tar.gz.part.* files, and extract "
                f"them into '{path}'."
            )

        self.path = path
        index_path = index_file
        if not osp.isabs(index_path):
            index_path = osp.join(path, index_file)
        if not osp.exists(index_path):
            raise FileNotFoundError(f"LiFlow index file not found: {index_path}")

        self.index_path = index_path
        self.time_delay_steps = int(time_delay_steps)
        self.prior_scale_li = np.asarray(prior_scale_li, dtype=np.float32)
        self.prior_scale_frame = np.asarray(prior_scale_frame, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.random_time = random_time
        self.in_memory = in_memory

        self.row_data = self.read_data(index_path)
        self.num_samples = len(self.row_data)
        logger.info(f"Load {self.num_samples} samples from {index_path}")

        self.element_index, self.atomic_numbers, self.lattice, self.positions = (
            self.read_structure_data(path, self.row_data)
        )

    def read_data(self, index_path: str) -> List[Dict]:
        """Read trajectory index rows from a CSV file.

        Args:
            index_path (str): Path to the index CSV.

        Returns:
            List[Dict]: Parsed trajectory index rows.
        """
        with open(index_path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        required = {"name", "temp", "t_start", "t_end"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(
                f"LiFlow index must contain columns {sorted(required)}, got "
                f"{list(rows[0].keys()) if rows else []}"
            )

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

    def read_structure_data(self, path: str, rows: List[Dict]):
        """Load element tables, lattices, and trajectory archives.

        Args:
            path (str): Dataset root directory.
            rows (List[Dict]): Parsed index rows used to decide which temperatures to
                load.

        Returns:
            Tuple: ``(element_index, atomic_numbers, lattice, positions)``.
        """
        element_index = np.load(osp.join(path, "element_index.npy"))
        atomic_numbers = np.load(
            osp.join(path, "atomic_numbers.npy"), allow_pickle=True
        ).item()
        lattice = np.load(osp.join(path, "lattice.npy"), allow_pickle=True).item()

        positions = {}
        for temp in sorted({row["temp"] for row in rows}):
            archive_path = osp.join(path, f"positions_{temp}K.npz")
            if not osp.exists(archive_path):
                raise FileNotFoundError(f"Trajectory archive not found: {archive_path}")
            archive = np.load(archive_path)
            positions[temp] = dict(archive) if self.in_memory else archive
            logger.info(f"Load trajectory archive from {archive_path}")
        return element_index, atomic_numbers, lattice, positions

    def __len__(self):
        return self.num_samples

    def _masses(self, atomic_numbers: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                float(Element.from_Z(int(number)).atomic_mass)
                for number in atomic_numbers
            ],
            dtype=np.float32,
        )

    def __getitem__(self, idx: int):
        """Build one noised trajectory-pair sample.

        Args:
            idx (int): Sample index.

        Returns:
            Dict: Model input fields for one LiFlow training / prediction sample.
        """
        row = self.row_data[idx]
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

        atomic_numbers = np.asarray(self.atomic_numbers[row["name"]], dtype=np.int64)
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

        data = {
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
        return data
