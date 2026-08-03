"""
survey.py
=========

Modality 3 - teacher survey responses.

For every administrator-month a panel of teachers (default 20) rates the
administrator on 14 Likert items (1 = Strongly Disagree ... 5 = Strongly
Agree).  Each item's latent mean is a weighted function of the administrator's
latent traits, so surveys correlate strongly with communication quality,
leadership, responsiveness and (inversely) stress.

Realistic human variation is layered on top:

* per-teacher rating bias (some teachers are systematically lenient/harsh),
* per-response measurement noise,
* occasional *contrarian* teachers who deviate sharply from the consensus.

Both the raw individual responses and the monthly aggregates (mean / median /
std per item) are produced.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import (
    LIKERT_MAX,
    LIKERT_MIN,
    SURVEY_AGG_FUNCTIONS,
    SURVEY_ITEMS,
    Config,
)
from .config import NEGATIVE_TRAITS
from .temporal import MonthlyState


class SurveyGenerator:
    """Generates individual + aggregated teacher survey data."""

    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._items = list(SURVEY_ITEMS.keys())

    # ------------------------------------------------------------------
    def _item_latent(self, item: str, latent: Dict[str, float]) -> float:
        """Compute an item's underlying quality on a 0..1 scale.

        Negative-direction traits (stress, workload) contribute through
        ``(1 - trait)`` so that, e.g., higher stress lowers ``manages_workload``.
        """
        score = 0.0
        for trait, weight in SURVEY_ITEMS[item].items():
            value = latent[trait]
            if trait in NEGATIVE_TRAITS or weight < 0:
                # A negative weight expresses an inverse relationship.
                contribution = abs(weight) * (1.0 - value)
            else:
                contribution = weight * value
            score += contribution
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _item_mean_likert(self, item_latent: float) -> float:
        """Map a 0..1 item quality onto the 1..5 Likert mean."""
        return LIKERT_MIN + (LIKERT_MAX - LIKERT_MIN) * item_latent

    # ------------------------------------------------------------------
    def _sample_response(self, mean_likert: float, teacher_bias: float,
                         contrarian: bool) -> int:
        """Draw a single teacher's ordinal rating for one item."""
        if contrarian:
            # Contrarian teachers pull toward the opposite pole with extra noise.
            opposite = (LIKERT_MIN + LIKERT_MAX) - mean_likert
            raw = 0.5 * opposite + 0.5 * mean_likert + self._rng.normal(0, 1.1)
        else:
            raw = mean_likert + teacher_bias + self._rng.normal(
                0, self._config.survey_teacher_noise
            )
        return int(np.clip(round(raw), LIKERT_MIN, LIKERT_MAX))

    # ------------------------------------------------------------------
    def generate_for_state(
        self, state: MonthlyState
    ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        """Generate individual responses and the aggregate for one admin-month.

        Returns
        -------
        (individual_rows, aggregate_row)
            ``individual_rows`` is one dict per teacher; ``aggregate_row`` holds
            the per-item mean/median/std plus response count.
        """
        rng = self._rng
        latent = state.latent

        # Pre-compute each item's Likert mean once per admin-month.
        item_means = {
            item: self._item_mean_likert(self._item_latent(item, latent))
            for item in self._items
        }

        individual_rows: List[Dict[str, object]] = []
        # Collect responses per item for aggregation.
        responses: Dict[str, List[int]] = {item: [] for item in self._items}

        for t in range(self._config.teachers_per_admin):
            teacher_bias = float(rng.normal(0, 0.35))
            is_contrarian = rng.random() < self._config.contrarian_teacher_prob

            row: Dict[str, object] = {
                "admin_id": state.admin_id,
                "month": state.month_label,
                "month_step": state.month_step,
                "teacher_id": f"T{state.admin_id}_{state.month_step:02d}_{t+1:02d}",
            }
            for item in self._items:
                value = self._sample_response(
                    item_means[item], teacher_bias, is_contrarian
                )
                row[item] = value
                responses[item].append(value)
            individual_rows.append(row)

        # --- Aggregate ----------------------------------------------------
        aggregate: Dict[str, object] = {
            "admin_id": state.admin_id,
            "month": state.month_label,
            "month_step": state.month_step,
            "n_respondents": self._config.teachers_per_admin,
        }
        for item in self._items:
            arr = np.asarray(responses[item], dtype=float)
            for func in SURVEY_AGG_FUNCTIONS:
                if func == "mean":
                    aggregate[f"{item}_mean"] = round(float(arr.mean()), 3)
                elif func == "median":
                    aggregate[f"{item}_median"] = float(np.median(arr))
                elif func == "std":
                    aggregate[f"{item}_std"] = round(float(arr.std(ddof=0)), 3)

        return individual_rows, aggregate

    # ------------------------------------------------------------------
    def generate(
        self, states: List[MonthlyState]
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        """Generate survey data for all administrator-months.

        Returns
        -------
        (all_individual_rows, all_aggregate_rows)
        """
        all_individual: List[Dict[str, object]] = []
        all_aggregate: List[Dict[str, object]] = []
        for state in states:
            individual, aggregate = self.generate_for_state(state)
            all_individual.extend(individual)
            all_aggregate.append(aggregate)
        return all_individual, all_aggregate
