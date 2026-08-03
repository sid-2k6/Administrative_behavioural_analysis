"""
labels.py
=========

Output-label generator.

Five continuous scores (0-100) and one categorical ``behaviour_category`` are
derived **from the latent state**, never randomly assigned:

    leadership_score          <- leadership
    communication_score       <- communication
    administrative_efficiency <- task_completion, responsiveness, attendance
    stress_score              <- stress
    engagement_score          <- engagement

The ``behaviour_category`` is produced by a two-stage, latent-driven ranking
that also *guarantees* the requested class imbalance (Highly Effective 20%,
Effective 35%, Average 25%, Overloaded 10%, Needs Improvement 10%):

1. The records with the strongest "overload signature" (high workload + high
   stress) are labelled *Overloaded* (top 10%).
2. The remaining records are ranked by an effectiveness composite and split
   into Highly Effective / Effective / Average / Needs Improvement using the
   renormalised target proportions.

Because the ranking is per administrator-month, a normally *Effective*
administrator can legitimately have an *Overloaded* month, which is exactly
the behaviour we want.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import Config
from .temporal import MonthlyState

# Target proportions for the categorical label.
_CATEGORY_TARGETS = {
    "Highly Effective": 0.20,
    "Effective": 0.35,
    "Average": 0.25,
    "Overloaded": 0.10,
    "Needs Improvement": 0.10,
}


class LabelGenerator:
    """Derives continuous scores and the behaviour category for all records."""

    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

    # ------------------------------------------------------------------
    def _score(self, value: float) -> float:
        """Map a latent 0..1 trait to a noisy 0..100 score."""
        noisy = 100.0 * value + self._rng.normal(0, self._config.label_noise_std)
        return round(float(np.clip(noisy, 0.0, 100.0)), 2)

    # ------------------------------------------------------------------
    def _continuous_scores(self, state: MonthlyState) -> Dict[str, float]:
        """Compute the five continuous output scores for one record."""
        lat = state.latent
        efficiency_latent = (
            0.5 * lat["task_completion"]
            + 0.3 * lat["responsiveness"]
            + 0.2 * lat["attendance"]
        )
        return {
            "leadership_score": self._score(lat["leadership"]),
            "communication_score": self._score(lat["communication"]),
            "administrative_efficiency": self._score(efficiency_latent),
            "stress_score": self._score(lat["stress"]),
            "engagement_score": self._score(lat["engagement"]),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _largest_remainder_counts(n: int, proportions: Dict[str, float]) -> Dict[str, int]:
        """Convert proportions into integer counts summing exactly to ``n``."""
        raw = {k: v * n for k, v in proportions.items()}
        counts = {k: int(np.floor(v)) for k, v in raw.items()}
        remainder = n - sum(counts.values())
        frac_order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
        for i in range(remainder):
            counts[frac_order[i]] += 1
        return counts

    # ------------------------------------------------------------------
    def _assign_categories(self, states: List[MonthlyState]) -> List[str]:
        """Assign a behaviour category to every state, honouring class balance."""
        n = len(states)
        overload_index = np.array(
            [
                0.45 * s.latent["workload"]
                + 0.45 * s.latent["stress"]
                - 0.10 * s.latent["task_completion"]
                + self._rng.normal(0, 0.02)
                for s in states
            ]
        )
        effectiveness = np.array(
            [
                0.30 * s.latent["leadership"]
                + 0.25 * s.latent["communication"]
                + 0.20 * s.latent["task_completion"]
                + 0.15 * s.latent["responsiveness"]
                + 0.10 * s.latent["engagement"]
                - 0.10 * s.latent["stress"]
                + self._rng.normal(0, 0.02)
                for s in states
            ]
        )

        categories: List[str] = [""] * n
        counts = self._largest_remainder_counts(n, _CATEGORY_TARGETS)

        # --- Stage 1: Overloaded = strongest overload signature ----------
        n_overloaded = counts["Overloaded"]
        overloaded_idx = set(np.argsort(-overload_index)[:n_overloaded].tolist())
        for i in overloaded_idx:
            categories[i] = "Overloaded"

        # --- Stage 2: rank the remainder by effectiveness ----------------
        remaining = [i for i in range(n) if i not in overloaded_idx]
        remaining_sorted = sorted(remaining, key=lambda i: effectiveness[i], reverse=True)

        ordered_labels = ["Highly Effective", "Effective", "Average", "Needs Improvement"]
        pos = 0
        for label in ordered_labels:
            take = counts[label]
            for i in remaining_sorted[pos:pos + take]:
                categories[i] = label
            pos += take
        # Any residual (rounding) -> Average.
        for i in remaining_sorted[pos:]:
            categories[i] = "Average"

        return categories

    # ------------------------------------------------------------------
    def generate(self, states: List[MonthlyState]) -> List[Dict[str, object]]:
        """Generate the full label table (one row per administrator-month)."""
        categories = self._assign_categories(states)
        rows: List[Dict[str, object]] = []
        for state, category in zip(states, categories):
            row: Dict[str, object] = {
                "admin_id": state.admin_id,
                "month": state.month_label,
                "month_step": state.month_step,
            }
            row.update(self._continuous_scores(state))
            row["behaviour_category"] = category
            rows.append(row)
        return rows
