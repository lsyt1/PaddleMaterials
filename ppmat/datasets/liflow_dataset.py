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

"""Deterministic, cached LiFlow trajectory dataset.

Local-path mode mirrors the reference ``TimeDelayedPairDataset`` at commit
``e6fc475361d046865f12cae1aee11c4f56c48d87``: it reads the universal data
directory (``element_index.npy``, ``atomic_numbers.npy``, ``positions_{T}K.npz``,
``lattice.npy``) together with a frozen ``{split}.csv`` index, builds a
variable-size endpoint sample consumed by ``LiFlowCollator``, and caches every
sample to disk.  A cache is reused only when the frozen split index, the build
configuration and the sample count are all unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import data as ase_data

from ppmat.models.liflow.geometry import get_neighbor_list_batch
from ppmat.models.liflow.prior import AdaptiveMaxwellBoltzmannPrior
from ppmat.models.liflow.prior import Prior

# Validation expands each sample over these fixed flow times (11-point grid).
VALIDATION_TIMES = np.linspace(0.0, 1.0, 11, dtype=np.float32)
_VALIDATION_N = int(VALIDATION_TIMES.shape[0])

_INDEX_COLUMNS = {
    "name",
    "temp",
    "t_start",
    "t_end",
    "prior_Li",
    "prior_frame",
    "comp",
}


class LiFlowDataset:
    """Variable-size time-delayed trajectory dataset with deterministic caching."""

    DATASET_VERSION = 1

    def __init__(
        self,
        path: str,
        split: str = "train",
        *,
        time_delay_steps: int = 100,
        cutoff: float = 5.0,
        periodic: bool = True,
        neighbor_list_both_ends: bool = False,
        prior: Prior | None = None,
        seed: int = 42,
        in_memory: bool = False,
        overwrite: bool = False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}.")
        self.path = Path(path)
        self.split = split
        self.time_delay_steps = int(time_delay_steps)
        self.cutoff = float(cutoff)
        self.periodic = bool(periodic)
        self.neighbor_list_both_ends = bool(neighbor_list_both_ends)
        self.prior = prior or AdaptiveMaxwellBoltzmannPrior(seed=seed)
        self.seed = int(seed)
        self.in_memory = bool(in_memory)
        self.rng = np.random.default_rng(self.seed)
        self.atomic_masses = ase_data.atomic_masses
        self._from_cache = False

        self._load_index()
        self._load_raw_data()

        if overwrite or not self._cache_is_valid():
            self._build_cache()

    # -- data loading ------------------------------------------------------

    def _load_index(self):
        csv_path = self.path / f"{self.split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing frozen split index: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = _INDEX_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing index columns: {sorted(missing)}")
        self._df = df.reset_index(drop=True)
        self.sample_ids = [str(name) for name in df["name"]]

    def _load_raw_data(self):
        self.element_index = np.load(self.path / "element_index.npy")
        with open(self.path / "atomic_numbers.npy", "rb") as handle:
            self.atomic_numbers = np.load(handle, allow_pickle=True).item()
        self._positions: dict[float, dict[str, np.ndarray]] = {}
        for temp in set(self._df["temp"]):
            blob = np.load(self.path / f"positions_{int(temp)}K.npz")
            self._positions[float(temp)] = dict(blob) if self.in_memory else blob
        with open(self.path / "lattice.npy", "rb") as handle:
            self.lattice = np.load(handle, allow_pickle=True).item()

    # -- dataset protocol --------------------------------------------------

    def __len__(self) -> int:
        if self.split == "val":
            return len(self._df) * _VALIDATION_N
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        idx = int(idx)
        if self._from_cache:
            return self._load_sample_from_cache(idx)
        return self._build_sample(idx)

    # -- sample construction -----------------------------------------------

    def _resolve_index(self, idx: int) -> tuple[int, int | None]:
        """Map an external index to ``(row_index, time_index)``.

        Validation expands each row over the 11 fixed ``flow_time`` values; the
        other splits map one-to-one and use a stochastic flow time.
        """
        if self.split == "val":
            return idx // _VALIDATION_N, idx % _VALIDATION_N
        return idx, None

    def _build_sample(self, idx: int) -> dict[str, Any]:
        row_idx, time_idx = self._resolve_index(idx)

        # The endpoint pair is fixed per row: derive a row-local seed so all 11
        # validation times share the same trajectory window.
        row_rng = np.random.default_rng(self.seed + row_idx)
        row = self._df.iloc[row_idx]
        name = str(row["name"])
        temp = float(row["temp"])
        atomic_numbers = np.asarray(self.atomic_numbers[name], dtype=np.int64)
        elements = self.element_index[atomic_numbers].astype(np.int64)
        masses = self.atomic_masses[atomic_numbers]
        positions = self._positions[temp][name]
        lattice = np.asarray(self.lattice[name], dtype=np.float64)

        t_start = int(row["t_start"])
        t_end = int(row["t_end"])
        if time_idx is not None:
            flow_time = VALIDATION_TIMES[time_idx]
        else:
            flow_time = self.rng.uniform(0.0, 1.0)
        start_time = int(row_rng.integers(t_start, t_end - self.time_delay_steps + 1))
        end_time = start_time + self.time_delay_steps

        start_positions = np.asarray(positions[start_time], dtype=np.float32)
        end_positions = np.asarray(positions[end_time], dtype=np.float32)

        nl_positions = (
            [start_positions, end_positions]
            if self.neighbor_list_both_ends
            else [start_positions]
        )
        edge_index, shifts = get_neighbor_list_batch(
            positions_batch=nl_positions,
            lattice=lattice,
            cutoff=self.cutoff,
            periodic=self.periodic,
        )

        prior = self.prior.sample(
            temperature=temp,
            atomic_numbers=atomic_numbers,
            masses=masses,
            scale_Li_index=int(row["prior_Li"]),
            scale_frame_index=int(row["prior_frame"]),
            shape=start_positions.shape,
        ).astype(np.float32)
        displacement = (end_positions - start_positions).astype(np.float32)
        velocity = (end_positions - (start_positions + prior)).astype(np.float32)

        return {
            "start_positions": start_positions,
            "end_positions": end_positions,
            "prior": prior,
            "velocity": velocity,
            "displacement": displacement,
            "elements": elements,
            "atomic_numbers": atomic_numbers,
            "edge_index": edge_index.astype(np.int64),
            "shifts": shifts.astype(np.float32),
            "temperature": np.asarray(temp, dtype=np.float32),
            "flow_time": np.asarray(flow_time, dtype=np.float32),
            "lattice": lattice.astype(np.float32),
            "name": name,
        }

    # -- caching -----------------------------------------------------------

    def _cache_dir(self) -> Path:
        return self.path / ".liflow_cache" / self.split

    def _source_index_sha256(self) -> str:
        csv_path = self.path / f"{self.split}.csv"
        return hashlib.sha256(csv_path.read_bytes()).hexdigest()

    def _manifest_config(self) -> dict[str, Any]:
        return {
            "dataset_version": self.DATASET_VERSION,
            "split": self.split,
            "time_delay_steps": self.time_delay_steps,
            "cutoff": self.cutoff,
            "periodic": self.periodic,
            "neighbor_list_both_ends": self.neighbor_list_both_ends,
            "source_index_sha256": self._source_index_sha256(),
        }

    def _cache_is_valid(self) -> bool:
        cd = self._cache_dir()
        manifest_path = cd / "manifest.json"
        flag = cd / "completed.flag"
        if not (manifest_path.exists() and flag.exists()):
            return False
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            return False
        if manifest != self._manifest_config():
            return False
        if len(list(cd.glob("sample_*.npz"))) != len(self):
            return False
        return True

    def _build_cache(self) -> None:
        cd = self._cache_dir()
        cd.mkdir(parents=True, exist_ok=True)
        for stale in list(cd.glob("sample_*.npz")):
            stale.unlink()
        (cd / "manifest.json").unlink(missing_ok=True)
        (cd / "completed.flag").unlink(missing_ok=True)
        for i in range(len(self)):
            sample = self._build_sample(i)
            np.savez(cd / f"sample_{i}.npz", **_encode(sample))
        (cd / "manifest.json").write_text(
            json.dumps(self._manifest_config(), sort_keys=True)
        )
        (cd / "completed.flag").write_text("done", encoding="utf-8")
        self._from_cache = True

    def _load_sample_from_cache(self, idx: int) -> dict[str, Any]:
        cd = self._cache_dir()
        blob = np.load(cd / f"sample_{idx}.npz")
        sample: dict[str, Any] = {}
        for key in blob.files:
            value = blob[key]
            if value.dtype.kind in "SU" or value.dtype.kind == "O":
                sample[key] = str(value.item())
            else:
                sample[key] = value
        return sample


def _encode(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    """Serialize a sample as an ``np.savez``-compatible mapping."""
    encoded: dict[str, np.ndarray] = {}
    for key, value in sample.items():
        if isinstance(value, str):
            encoded[key] = np.asarray([value])
        else:
            encoded[key] = np.asarray(value)
    return encoded
