"""
temporal.py
===========

Temporal behaviour simulator.

Administrators do **not** behave independently each month.  Their observable
behaviour is produced from a slowly-evolving latent state built from:

    latent_state(t) = clip( baseline
                            + AR(1) drift wander(t)
                            + seasonal / event deltas(t) )

* ``baseline``  - the administrator's personal profile-driven set point.
* ``wander``    - a persistent AR(1) random walk giving smooth month-to-month
                  autocorrelation (a Markov / smooth state-transition process).
* ``deltas``    - additive shifts from the active calendar & stochastic events.

This yields realistic, autocorrelated trajectories with occasional event-driven
excursions but no unrealistic jumps (everything is clipped to [0, 1]).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .config import (
    CALENDAR_EVENTS,
    EVENT_LIBRARY,
    LATENT_TRAITS,
    Config,
)
from .profiles import Administrator


@dataclass
class MonthlyState:
    """The fully-resolved latent state of one administrator in one month.

    This is the single source of truth consumed by every modality generator,
    which guarantees cross-modal consistency: text, logs and surveys are all
    derived from the *same* latent vector.
    """

    admin: Administrator
    month_step: int          # 1-based simulation step
    calendar_month: int      # 1..12 calendar month
    month_label: str         # e.g. "2024-03"
    events: List[str]        # active event names this month
    intensity: float         # activity-volume multiplier from events
    latent: Dict[str, float] = field(default_factory=dict)

    @property
    def admin_id(self) -> str:
        return self.admin.admin_id


class TemporalSimulator:
    """Generates smooth, event-aware latent trajectories per administrator."""

    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._month_labels = config.month_labels()

    # ------------------------------------------------------------------
    def _resolve_events(self, admin: Administrator, month_step: int) -> List[str]:
        """Determine which events are active for an administrator this month."""
        cal_month = self._config.calendar_month_index(month_step)
        events: List[str] = [CALENDAR_EVENTS[cal_month]]

        # Institution-specific accreditation visit.
        if admin.accreditation_month == month_step:
            events.append("Accreditation Visit")

        # Occasional extra stochastic event during otherwise-routine months.
        if events == ["Routine"] and self._rng.random() < self._config.random_event_prob:
            extra = self._rng.choice(
                ["Faculty Recruitment", "Budget Planning", "Accreditation Visit"]
            )
            events = [str(extra)]

        return events

    # ------------------------------------------------------------------
    def _combine_event_effects(self, events: List[str]) -> tuple[Dict[str, float], float]:
        """Aggregate additive trait deltas and a combined intensity multiplier."""
        deltas: Dict[str, float] = {t: 0.0 for t in LATENT_TRAITS}
        intensity = 1.0
        for name in events:
            spec = EVENT_LIBRARY[name]
            for trait, delta in spec.trait_delta.items():
                deltas[trait] += delta
            intensity *= spec.intensity_multiplier
        # Prevent extreme compounded intensities.
        intensity = float(np.clip(intensity, 0.6, 1.9))
        return deltas, intensity

    # ------------------------------------------------------------------
    def simulate_admin(self, admin: Administrator) -> List[MonthlyState]:
        """Produce the month-by-month latent trajectory for one administrator.

        The trajectory combines three persistent components so that
        consecutive months are smoothly autocorrelated (a Markov-like process):

        * ``wander``          - an AR(1) random walk around the baseline.
        * ``smoothed_delta``  - event effects passed through an exponential
                                smoother so they ramp up and decay across
                                months instead of snapping on/off (which would
                                otherwise create unrealistic month-to-month
                                oscillation).
        """
        wander: Dict[str, float] = {t: 0.0 for t in LATENT_TRAITS}
        smoothed_delta: Dict[str, float] = {t: 0.0 for t in LATENT_TRAITS}
        carryover = self._config.event_carryover
        states: List[MonthlyState] = []

        for step in range(1, self._config.n_months + 1):
            # --- Advance the AR(1) wander term ---------------------------
            for trait in LATENT_TRAITS:
                innovation = self._rng.normal(0.0, self._config.drift_std)
                wander[trait] = (
                    self._config.drift_persistence * wander[trait] + innovation
                )

            # --- Resolve events & their effects --------------------------
            events = self._resolve_events(admin, step)
            raw_deltas, intensity = self._combine_event_effects(events)

            # Exponentially smooth the event deltas so effects persist and
            # decay gradually rather than reversing abruptly each month.
            for trait in LATENT_TRAITS:
                smoothed_delta[trait] = (
                    carryover * smoothed_delta[trait]
                    + (1.0 - carryover) * raw_deltas[trait]
                )

            # --- Compose the clipped latent state ------------------------
            latent: Dict[str, float] = {}
            for trait in LATENT_TRAITS:
                value = admin.baseline[trait] + wander[trait] + smoothed_delta[trait]
                latent[trait] = float(np.clip(value, 0.0, 1.0))

            # Workload directly suppresses responsiveness (busy people are
            # slower) - a deterministic, interpretable coupling on top of the
            # profile correlations.
            overload_penalty = 0.18 * max(0.0, latent["workload"] - 0.7)
            latent["responsiveness"] = float(
                np.clip(latent["responsiveness"] - overload_penalty, 0.0, 1.0)
            )

            states.append(
                MonthlyState(
                    admin=admin,
                    month_step=step,
                    calendar_month=self._config.calendar_month_index(step),
                    month_label=self._month_labels[step - 1],
                    events=events,
                    intensity=intensity,
                    latent=latent,
                )
            )

        return states

    # ------------------------------------------------------------------
    def simulate(self, administrators: List[Administrator]) -> List[MonthlyState]:
        """Simulate every administrator, returning a flat list of states."""
        all_states: List[MonthlyState] = []
        for admin in administrators:
            all_states.extend(self.simulate_admin(admin))
        return all_states
