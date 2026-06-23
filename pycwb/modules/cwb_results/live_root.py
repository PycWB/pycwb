"""Utilities for reading cWB ``liveTime`` ROOT files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


SECONDS_PER_JULIAN_YEAR = 365.25 * 24 * 3600


@dataclass(frozen=True)
class LiveRootSummary:
    entries: int
    total_live_seconds: float
    total_live_years: float
    nominal_seconds: float
    nominal_years: float
    loss_seconds: float
    loss_years: float
    live_min: float
    live_max: float
    live_eq_nominal_count: int
    live_lt_nominal_count: int
    live_eq_1200_count: int
    live_lt_1200_count: int
    live_zero_count: int
    unique_live_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "entries": self.entries,
            "total_live_seconds": self.total_live_seconds,
            "total_live_years": self.total_live_years,
            "nominal_seconds": self.nominal_seconds,
            "nominal_years": self.nominal_years,
            "loss_seconds": self.loss_seconds,
            "loss_years": self.loss_years,
            "live_min": self.live_min,
            "live_max": self.live_max,
            "live_eq_nominal_count": self.live_eq_nominal_count,
            "live_lt_nominal_count": self.live_lt_nominal_count,
            "live_eq_1200_count": self.live_eq_1200_count,
            "live_lt_1200_count": self.live_lt_1200_count,
            "live_zero_count": self.live_zero_count,
            "unique_live_count": self.unique_live_count,
        }


class CwbLiveRoot:
    """Chunked reader for cWB merged ``liveTime`` ROOT trees.

    cWB stores one row per job/lag/superlag in a ``liveTime`` tree. The
    canonical livetime is the scalar ``live`` branch. The ``start`` and
    ``stop`` vector branches hold per-detector analysis windows, while
    ``lag`` and ``slag`` hold regular-lag and superlag metadata.
    """

    required_branches = ("live", "start", "stop")

    def __init__(self, root_file: str | Path, tree_name: str = "liveTime"):
        self.root_file = Path(root_file)
        self.tree_name = tree_name

    def _open_tree(self):
        try:
            import uproot
        except ImportError as exc:
            raise ImportError(
                "uproot is required to read cWB live ROOT files: "
                "pip install uproot awkward"
            ) from exc

        return uproot.open(f"{self.root_file}:{self.tree_name}")

    @property
    def entries(self) -> int:
        with self._open_tree() as tree:
            return int(tree.num_entries)

    def typenames(self) -> dict[str, str]:
        with self._open_tree() as tree:
            return dict(tree.typenames())

    def sample(self, n: int = 5) -> dict[str, np.ndarray]:
        branches = ["run", "gps", "live", "lag", "slag", "start", "stop"]
        with self._open_tree() as tree:
            available = [branch for branch in branches if branch in tree.keys()]
            arrays = tree.arrays(
                available, entry_start=0, entry_stop=n, library="ak",
            )
            return self._to_numpy_dict(arrays, available)

    def summary(
        self,
        *,
        step_size: str | int = "100 MB",
        year_seconds: float = SECONDS_PER_JULIAN_YEAR,
    ) -> LiveRootSummary:
        total_live = 0.0
        nominal = 0.0
        entries = 0
        live_eq_nominal_count = 0
        live_lt_nominal_count = 0
        live_eq_1200_count = 0
        live_lt_1200_count = 0
        live_zero_count = 0
        live_min = np.inf
        live_max = -np.inf
        unique_live: set[float] = set()

        with self._open_tree() as tree:
            self._require_branches(tree)
            for arrays in tree.iterate(
                self.required_branches,
                library="ak",
                step_size=step_size,
            ):
                arrays = self._to_numpy_dict(arrays, self.required_branches)
                live = arrays["live"].astype(np.float64)
                duration = self._nominal_duration(arrays["start"], arrays["stop"])

                total_live += float(live.sum())
                nominal += float(duration.sum())
                entries += int(live.size)
                live_min = min(live_min, float(live.min()))
                live_max = max(live_max, float(live.max()))
                unique_live.update(float(value) for value in np.unique(live))
                live_eq_nominal_count += int(np.count_nonzero(np.isclose(live, duration)))
                live_lt_nominal_count += int(np.count_nonzero(live < duration - 1e-9))
                live_eq_1200_count += int(np.count_nonzero(np.isclose(live, 1200.0)))
                live_lt_1200_count += int(np.count_nonzero(live < 1200.0 - 1e-9))
                live_zero_count += int(np.count_nonzero(np.isclose(live, 0.0)))

        loss = nominal - total_live
        return LiveRootSummary(
            entries=entries,
            total_live_seconds=total_live,
            total_live_years=total_live / year_seconds,
            nominal_seconds=nominal,
            nominal_years=nominal / year_seconds,
            loss_seconds=loss,
            loss_years=loss / year_seconds,
            live_min=live_min,
            live_max=live_max,
            live_eq_nominal_count=live_eq_nominal_count,
            live_lt_nominal_count=live_lt_nominal_count,
            live_eq_1200_count=live_eq_1200_count,
            live_lt_1200_count=live_lt_1200_count,
            live_zero_count=live_zero_count,
            unique_live_count=len(unique_live),
        )

    def slag_summary(
        self,
        *,
        by: Literal["pair", "id"] = "pair",
        step_size: str | int = "100 MB",
        year_seconds: float = SECONDS_PER_JULIAN_YEAR,
    ) -> list[dict[str, Any]]:
        groups: dict[Any, list[float | int]] = {}

        with self._open_tree() as tree:
            self._require_branches(tree, extra=("slag",))
            for arrays in tree.iterate(
                ["live", "slag", "start", "stop"],
                library="ak",
                step_size=step_size,
            ):
                arrays = self._to_numpy_dict(
                    arrays, ("live", "slag", "start", "stop"),
                )
                live = arrays["live"].astype(np.float64)
                duration = self._nominal_duration(arrays["start"], arrays["stop"])
                keys = self._slag_keys(arrays["slag"], by)

                for key in np.unique(keys, axis=0 if by == "pair" else None):
                    mask = self._key_mask(keys, key, by)
                    group_key = tuple(int(v) for v in key) if by == "pair" else int(key)
                    rec = groups.setdefault(group_key, [0, 0.0, 0.0, 0, 0])
                    rec[0] += int(np.count_nonzero(mask))
                    rec[1] += float(live[mask].sum())
                    rec[2] += float(duration[mask].sum())
                    rec[3] += int(np.count_nonzero(live[mask] < duration[mask] - 1e-9))
                    rec[4] += int(np.count_nonzero(np.isclose(live[mask], 0.0)))

        result = []
        for key, rec in groups.items():
            rows, live_sum, nominal_sum, reduced_rows, zero_rows = rec
            result.append({
                "key": key,
                "rows": rows,
                "live_seconds": live_sum,
                "live_years": live_sum / year_seconds,
                "nominal_seconds": nominal_sum,
                "nominal_years": nominal_sum / year_seconds,
                "loss_seconds": nominal_sum - live_sum,
                "loss_years": (nominal_sum - live_sum) / year_seconds,
                "reduced_rows": reduced_rows,
                "zero_rows": zero_rows,
            })

        if by == "id":
            return sorted(result, key=lambda row: row["key"])
        return sorted(result, key=lambda row: row["live_seconds"], reverse=True)

    @staticmethod
    def _require_branches(tree, extra: tuple[str, ...] = ()) -> None:
        missing = [
            branch for branch in (*CwbLiveRoot.required_branches, *extra)
            if branch not in tree.keys()
        ]
        if missing:
            raise KeyError(f"Missing required liveTime branch(es): {missing}")

    @staticmethod
    def _to_numpy_dict(arrays, branches) -> dict[str, np.ndarray]:
        try:
            import awkward as ak
        except ImportError as exc:
            raise ImportError(
                "awkward is required to read cWB live ROOT files: "
                "pip install uproot awkward"
            ) from exc

        return {
            branch: np.asarray(ak.to_numpy(arrays[branch]))
            for branch in branches
        }

    @staticmethod
    def _nominal_duration(start: np.ndarray, stop: np.ndarray) -> np.ndarray:
        durations = []
        for ifo_idx in range(start.shape[1]):
            duration = stop[:, ifo_idx] - start[:, ifo_idx]
            durations.append(
                np.where(
                    (start[:, ifo_idx] > 0) & (duration > 0),
                    duration,
                    np.nan,
                )
            )
        return np.nanmin(np.vstack(durations), axis=0)

    @staticmethod
    def _slag_keys(slag: np.ndarray, by: Literal["pair", "id"]) -> np.ndarray:
        if by == "pair":
            return np.rint(slag[:, :2]).astype(np.int64)
        if by == "id":
            return np.rint(slag[:, 2]).astype(np.int64)
        raise ValueError("by must be 'pair' or 'id'")

    @staticmethod
    def _key_mask(keys: np.ndarray, key: np.ndarray | np.integer, by: str) -> np.ndarray:
        if by == "pair":
            return (keys[:, 0] == key[0]) & (keys[:, 1] == key[1])
        return keys == key
