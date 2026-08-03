"""
admin_logs.py
=============

Modality 2 - administrative activity logs.

Twenty-three operational metrics are derived for every administrator-month
from the shared latent state.  None of the variables are sampled
independently: each distribution's parameters are functions of the latent
traits, so the resulting log table is internally correlated and consistent
with the other modalities.

Distribution choices follow the research design:

    counts            -> Poisson
    response time     -> Log-Normal
    percentages       -> Beta
    working hours     -> Normal
    completion time   -> Beta-derived
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import Config
from .temporal import MonthlyState


def _beta_from_mean(rng: np.random.Generator, mean: float, kappa: float) -> float:
    """Sample from a Beta distribution parameterised by mean and concentration.

    Parameters
    ----------
    mean:
        Desired mean in the open interval (0, 1).
    kappa:
        Concentration; larger values -> tighter distribution around ``mean``.
    """
    mean = float(np.clip(mean, 1e-3, 1 - 1e-3))
    alpha = mean * kappa
    beta = (1.0 - mean) * kappa
    return float(rng.beta(alpha, beta))


class AdminLogGenerator:
    """Generates the administrative-log modality from latent states."""

    def __init__(self, config: Config, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng

    # ------------------------------------------------------------------
    def generate_for_state(self, state: MonthlyState) -> Dict[str, object]:
        """Produce the full administrative-log record for one admin-month."""
        rng = self._rng
        cfg = self._config
        lat = state.latent
        intensity = state.intensity

        comm = lat["communication"]
        lead = lat["leadership"]
        task = lat["task_completion"]
        resp = lat["responsiveness"]
        attend = lat["attendance"]
        engage = lat["engagement"]
        stress = lat["stress"]
        workload = lat["workload"]

        # --- Meetings -----------------------------------------------------
        meetings_conducted = int(
            rng.poisson(cfg.base_meetings * intensity * (0.5 + 0.5 * engage + 0.3 * workload))
        )
        # Attended is a high fraction of a slightly larger pool, driven by attendance.
        attend_frac = _beta_from_mean(rng, 0.6 + 0.35 * attend, 40)
        meetings_attended = int(
            rng.poisson(cfg.base_meetings * intensity * attend_frac)
        )

        # --- Throughput ---------------------------------------------------
        reports_submitted = int(rng.poisson(cfg.base_reports * (0.4 + 0.9 * task)))
        approvals_processed = int(
            rng.poisson(cfg.base_approvals * intensity * (0.4 + 0.8 * resp))
        )
        leave_requests_processed = int(rng.poisson(4 + 8 * resp))

        # --- Backlog ------------------------------------------------------
        pending_tasks = int(
            rng.poisson(3 + 22 * (1 - task) * (0.5 + 0.7 * workload))
        )
        overdue_frac = _beta_from_mean(rng, 0.15 + 0.55 * (1 - task) * stress, 12)
        overdue_tasks = int(round(pending_tasks * overdue_frac))
        active_projects = int(rng.poisson(2 + 5 * engage))

        # --- Timeliness ---------------------------------------------------
        # Fast responders (high resp) -> low response hours (Log-Normal).
        resp_mu = np.log(6.0) + 2.2 * (1 - resp)
        average_response_hours = float(
            np.clip(rng.lognormal(resp_mu, cfg.response_hours_lognorm_sigma), 0.5, 240.0)
        )
        completion_fraction = _beta_from_mean(rng, 0.85 * task + 0.05, 20)
        average_task_completion_days = float(
            round(1.0 + cfg.task_completion_days_scale * (1 - completion_fraction), 2)
        )

        # --- Presence -----------------------------------------------------
        attendance_percentage = round(100 * _beta_from_mean(rng, 0.6 + 0.38 * attend, 60), 2)
        punctuality_percentage = round(
            100 * _beta_from_mean(rng, 0.55 + 0.4 * attend - 0.1 * stress, 55), 2
        )
        working_hours_per_week = round(
            float(rng.normal(cfg.base_working_hours * (0.75 + 0.5 * workload), 3.5)), 2
        )
        meetings_per_week = round(meetings_conducted / 4.33, 2)

        # --- Decision making ----------------------------------------------
        policy_decisions = int(rng.poisson(1.5 + 5 * lead))
        emergency_decisions = int(rng.poisson(0.4 + 4.5 * workload * stress))
        delegated_tasks = int(rng.poisson(2 + 9 * lead))

        # --- Engagement / interactions ------------------------------------
        office_visits = int(rng.poisson(8 + 16 * engage))
        faculty_interactions = int(rng.poisson((12 + 28 * comm * engage) * intensity))
        student_interactions = int(rng.poisson(6 + 24 * engage))
        budget_requests = int(
            rng.poisson(1 + 3 * workload + (2 if "Budget Planning" in state.events else 0))
        )
        complaint_resolutions = int(rng.poisson(1 + 6 * lead * resp))

        return {
            "admin_id": state.admin_id,
            "month": state.month_label,
            "month_step": state.month_step,
            "meetings_conducted": meetings_conducted,
            "meetings_attended": min(meetings_attended, meetings_conducted + 4),
            "reports_submitted": reports_submitted,
            "approvals_processed": approvals_processed,
            "leave_requests_processed": leave_requests_processed,
            "pending_tasks": pending_tasks,
            "overdue_tasks": overdue_tasks,
            "active_projects": active_projects,
            "average_response_hours": round(average_response_hours, 2),
            "average_task_completion_days": average_task_completion_days,
            "attendance_percentage": attendance_percentage,
            "punctuality_percentage": punctuality_percentage,
            "working_hours_per_week": working_hours_per_week,
            "meetings_per_week": meetings_per_week,
            "policy_decisions": policy_decisions,
            "emergency_decisions": emergency_decisions,
            "delegated_tasks": delegated_tasks,
            "office_visits": office_visits,
            "faculty_interactions": faculty_interactions,
            "student_interactions": student_interactions,
            "budget_requests": budget_requests,
            "complaint_resolutions": complaint_resolutions,
        }

    # ------------------------------------------------------------------
    def generate(self, states: List[MonthlyState]) -> List[Dict[str, object]]:
        """Generate administrative logs for all administrator-months."""
        return [self.generate_for_state(s) for s in states]
