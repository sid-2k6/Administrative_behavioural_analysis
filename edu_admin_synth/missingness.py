"""
missingness.py
==============

Realistic missing-data injection.

Missingness is **not** completely at random (MCAR).  Instead it is
workload-dependent (MNAR): the busier an administrator is in a given month,
the more likely their documentation / survey / log entries are incomplete.
This mirrors the real phenomenon where overloaded staff under-report.

Base rates follow the research design:

    surveys           ~ 5%
    administrative    ~ 2%
    text              ~ 1%

and are scaled up with workload via ``missing_workload_sensitivity``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .temporal import MonthlyState

WorkloadMap = Dict[Tuple[str, int], float]


def build_workload_map(states: Iterable[MonthlyState]) -> WorkloadMap:
    """Map each ``(admin_id, month_step)`` to its latent workload value."""
    return {(s.admin_id, s.month_step): s.latent["workload"] for s in states}


class MissingnessInjector:
    """Applies workload-driven (MNAR) missingness to a modality table."""

    def __init__(self, config: Config, rng: np.random.Generator,
                 workload_map: WorkloadMap) -> None:
        self._config = config
        self._rng = rng
        self._workload = workload_map

    # ------------------------------------------------------------------
    def _row_probability(self, workload: float, base_rate: float) -> float:
        """Scale the base missingness rate by (workload-centred) sensitivity."""
        sensitivity = self._config.missing_workload_sensitivity
        factor = 1.0 + sensitivity * (workload - 0.5)
        return float(np.clip(base_rate * max(0.2, factor), 0.0, 0.6))

    # ------------------------------------------------------------------
    def inject(
        self,
        df: pd.DataFrame,
        base_rate: float,
        protected: Set[str],
    ) -> pd.DataFrame:
        """Return a copy of ``df`` with MNAR missing values inserted.

        Parameters
        ----------
        df:
            Modality table containing ``admin_id`` and ``month_step`` columns.
        base_rate:
            Baseline per-cell missingness probability at average workload.
        protected:
            Column names that must never be set to missing (keys / identifiers).
        """
        if df.empty:
            return df.copy()

        out = df.copy()
        eligible = [c for c in out.columns if c not in protected]
        if not eligible:
            return out

        # Per-row missingness probability from workload.
        default_wl = 0.5
        probs = np.array(
            [
                self._row_probability(
                    self._workload.get((r.admin_id, int(r.month_step)), default_wl),
                    base_rate,
                )
                for r in out.itertuples(index=False)
            ]
        )

        n_rows = len(out)
        for col in eligible:
            draws = self._rng.random(n_rows)
            mask = draws < probs
            if mask.any():
                out.loc[mask, col] = np.nan
        return out
