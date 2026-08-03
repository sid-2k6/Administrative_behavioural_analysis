"""
metadata.py
===========

Explainability / documentation module.

Produces two machine- and human-readable artefacts describing the data
generating process:

* ``feature_dictionary.csv`` - one row per exported feature with its modality,
  inferred dtype, observed value range, generating distribution and a plain
  language description.
* ``metadata.json`` - the full generation configuration, latent-profile
  proportions, distributional assumptions and the documented correlation
  structure that the generator enforces.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    CALENDAR_EVENTS,
    COMMUNICATION_TYPES,
    EVENT_LIBRARY,
    PROFILES,
    SURVEY_ITEMS,
    TEXT_TOPICS,
    Config,
)

# Static descriptions & the distribution used to generate each base feature.
_FEATURE_DOCS: Dict[str, Dict[str, str]] = {
    # identifiers / demographics
    "admin_id": {"modality": "identifier", "distribution": "n/a", "description": "Stable administrator identifier."},
    "month": {"modality": "identifier", "distribution": "n/a", "description": "Calendar month label (YYYY-MM)."},
    "month_step": {"modality": "identifier", "distribution": "n/a", "description": "1-based simulation month index."},
    "name": {"modality": "demographic", "distribution": "faker", "description": "Synthetic administrator name."},
    "gender": {"modality": "demographic", "distribution": "categorical", "description": "Administrator gender."},
    "age": {"modality": "demographic", "distribution": "uniform-int", "description": "Administrator age in years."},
    "years_experience": {"modality": "demographic", "distribution": "uniform-int", "description": "Years of administrative experience."},
    "institution_type": {"modality": "demographic", "distribution": "categorical", "description": "Type of institution."},
    "department": {"modality": "demographic", "distribution": "categorical", "description": "Administrative department."},
    "designation": {"modality": "demographic", "distribution": "categorical", "description": "Administrative role/title."},
    # administrative logs
    "meetings_conducted": {"modality": "admin_log", "distribution": "Poisson", "description": "Meetings chaired this month; scales with engagement/workload/events."},
    "meetings_attended": {"modality": "admin_log", "distribution": "Poisson", "description": "Meetings attended; scales with attendance trait."},
    "reports_submitted": {"modality": "admin_log", "distribution": "Poisson", "description": "Reports submitted; scales with task completion."},
    "approvals_processed": {"modality": "admin_log", "distribution": "Poisson", "description": "Approvals processed; scales with responsiveness."},
    "leave_requests_processed": {"modality": "admin_log", "distribution": "Poisson", "description": "Leave requests handled; scales with responsiveness."},
    "pending_tasks": {"modality": "admin_log", "distribution": "Poisson", "description": "Backlog of open tasks; rises with low task completion and high workload."},
    "overdue_tasks": {"modality": "admin_log", "distribution": "Poisson x Beta", "description": "Subset of pending tasks past deadline; rises with stress and poor completion."},
    "active_projects": {"modality": "admin_log", "distribution": "Poisson", "description": "Concurrent active projects; scales with engagement."},
    "average_response_hours": {"modality": "admin_log", "distribution": "Log-Normal", "description": "Mean response latency; lower for responsive administrators."},
    "average_task_completion_days": {"modality": "admin_log", "distribution": "Beta-derived", "description": "Mean days to complete a task; lower for high task completion."},
    "attendance_percentage": {"modality": "admin_log", "distribution": "Beta", "description": "Percent of expected days present."},
    "punctuality_percentage": {"modality": "admin_log", "distribution": "Beta", "description": "Percent of on-time arrivals; reduced by stress."},
    "working_hours_per_week": {"modality": "admin_log", "distribution": "Normal", "description": "Average weekly working hours; scales with workload."},
    "meetings_per_week": {"modality": "admin_log", "distribution": "derived", "description": "Meetings conducted normalised to a weekly rate."},
    "policy_decisions": {"modality": "admin_log", "distribution": "Poisson", "description": "Policy decisions made; scales with leadership."},
    "emergency_decisions": {"modality": "admin_log", "distribution": "Poisson", "description": "Emergency decisions; rises with workload x stress."},
    "delegated_tasks": {"modality": "admin_log", "distribution": "Poisson", "description": "Tasks delegated; scales with leadership."},
    "office_visits": {"modality": "admin_log", "distribution": "Poisson", "description": "Office visits received; scales with engagement."},
    "faculty_interactions": {"modality": "admin_log", "distribution": "Poisson", "description": "Interactions with faculty; scales with communication x engagement."},
    "student_interactions": {"modality": "admin_log", "distribution": "Poisson", "description": "Interactions with students; scales with engagement."},
    "budget_requests": {"modality": "admin_log", "distribution": "Poisson", "description": "Budget requests raised; rises with workload and budget-planning events."},
    "complaint_resolutions": {"modality": "admin_log", "distribution": "Poisson", "description": "Complaints resolved; scales with leadership x responsiveness."},
    # text (aggregated)
    "text_document_count": {"modality": "text", "distribution": "Poisson", "description": "Number of documents authored this month."},
    "text_avg_word_count": {"modality": "text", "distribution": "Gamma", "description": "Mean document length in words (150-600)."},
    "text_avg_sentiment": {"modality": "text", "distribution": "Truncated-Normal", "description": "Mean document sentiment (-1..1)."},
    "text_avg_positive_ratio": {"modality": "text", "distribution": "derived", "description": "Mean ratio of positive lexicon words."},
    "text_avg_negative_ratio": {"modality": "text", "distribution": "derived", "description": "Mean ratio of negative lexicon words."},
    "text_avg_readability": {"modality": "text", "distribution": "derived", "description": "Mean Flesch reading-ease score (0..100)."},
    "text_dominant_type": {"modality": "text", "distribution": "categorical", "description": "Most frequent communication type."},
    "text_dominant_topic": {"modality": "text", "distribution": "categorical", "description": "Most frequent document topic."},
    "text_dominant_urgency": {"modality": "text", "distribution": "categorical", "description": "Most frequent urgency level."},
    "text_dominant_style": {"modality": "text", "distribution": "derived", "description": "Dominant writing-style descriptor."},
    # labels
    "leadership_score": {"modality": "label", "distribution": "latent+noise", "description": "Derived leadership score (0-100)."},
    "communication_score": {"modality": "label", "distribution": "latent+noise", "description": "Derived communication score (0-100)."},
    "administrative_efficiency": {"modality": "label", "distribution": "latent+noise", "description": "Derived efficiency score (0-100)."},
    "stress_score": {"modality": "label", "distribution": "latent+noise", "description": "Derived stress score (0-100)."},
    "engagement_score": {"modality": "label", "distribution": "latent+noise", "description": "Derived engagement score (0-100)."},
    "behaviour_category": {"modality": "label", "distribution": "latent-ranked", "description": "Derived 5-class behaviour category."},
    "n_respondents": {"modality": "survey", "distribution": "n/a", "description": "Number of teacher respondents for the month."},
}


class MetadataGenerator:
    """Builds the feature dictionary and the metadata JSON structure."""

    def __init__(self, config: Config) -> None:
        self._config = config

    # ------------------------------------------------------------------
    def _survey_feature_doc(self, column: str) -> Dict[str, str]:
        """Describe an aggregated survey column such as ``fairness_mean``."""
        for item in SURVEY_ITEMS:
            for agg in ("mean", "median", "std"):
                if column == f"{item}_{agg}":
                    drivers = ", ".join(SURVEY_ITEMS[item].keys())
                    return {
                        "modality": "survey",
                        "distribution": f"Likert(1-5) {agg}",
                        "description": (
                            f"{agg.title()} of teacher ratings for '{item}' "
                            f"(driven by latent: {drivers})."
                        ),
                    }
        return {}

    # ------------------------------------------------------------------
    def build_feature_dictionary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assemble one documented row per column of the fused dataset."""
        rows: List[Dict[str, object]] = []
        for col in df.columns:
            doc = _FEATURE_DOCS.get(col) or self._survey_feature_doc(col) or {
                "modality": "unknown",
                "distribution": "n/a",
                "description": "Auto-generated feature.",
            }
            series = df[col]
            if pd.api.types.is_numeric_dtype(series) and series.notna().any():
                vmin = round(float(np.nanmin(series.values)), 4)
                vmax = round(float(np.nanmax(series.values)), 4)
            else:
                vmin, vmax = "", ""
            rows.append(
                {
                    "feature": col,
                    "modality": doc["modality"],
                    "dtype": str(series.dtype),
                    "min_value": vmin,
                    "max_value": vmax,
                    "distribution": doc["distribution"],
                    "description": doc["description"],
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def build_metadata(self, df: pd.DataFrame) -> Dict[str, object]:
        """Produce the metadata JSON structure documenting the whole DGP."""
        cfg = self._config
        return {
            "dataset_name": "Synthetic Educational Administrative Behaviour",
            "research_context": (
                "Multimodal Representation Learning for Educational Administrative "
                "Behavior Analysis: A Transformer-Based Fusion Framework"
            ),
            "version": "1.0.0",
            "reproducibility": {
                "random_seed": cfg.random_seed,
                "note": "np.random.seed(42) and random.seed(42) fixed at entrypoint.",
            },
            "dimensions": {
                "n_administrators": cfg.n_administrators,
                "n_months": cfg.n_months,
                "teachers_per_admin": cfg.teachers_per_admin,
                "total_records": int(len(df)),
            },
            "modalities": ["text", "administrative_logs", "survey"],
            "latent_profiles": {
                p.name: {
                    "proportion": p.proportion,
                    "trait_means": p.trait_means,
                    "between_admin_std": p.trait_std,
                }
                for p in PROFILES
            },
            "temporal_model": {
                "type": "AR(1) latent random walk + seasonal/event deltas",
                "drift_std": cfg.drift_std,
                "drift_persistence": cfg.drift_persistence,
                "calendar_events": CALENDAR_EVENTS,
                "event_library": {
                    name: {
                        "trait_delta": spec.trait_delta,
                        "intensity_multiplier": spec.intensity_multiplier,
                    }
                    for name, spec in EVENT_LIBRARY.items()
                },
            },
            "distributions": {
                "meeting_counts": "Poisson",
                "workload / working_hours": "Normal",
                "response_time": "Log-Normal",
                "attendance / percentages": "Beta",
                "survey_ratings": "Ordinal Likert (1-5)",
                "sentiment": "Truncated Normal",
                "task_completion": "Beta",
                "word_count": "Gamma",
            },
            "correlation_assumptions": [
                "Lower response time -> higher communication/responsiveness ratings.",
                "Higher task completion -> higher administrative efficiency.",
                "Higher overdue tasks -> lower leadership perception.",
                "Higher workload -> more emergency decisions and slower responses.",
                "Higher attendance -> higher survey scores.",
                "Positive sentiment in text -> higher satisfaction ratings.",
                "Meeting/workload overload -> higher stress.",
                "Survey ratings strongly track latent communication quality.",
            ],
            "missingness": {
                "mechanism": "MNAR (increases with workload)",
                "base_rates": {
                    "survey": cfg.missing_rate_survey,
                    "administrative_logs": cfg.missing_rate_logs,
                    "text": cfg.missing_rate_text,
                },
                "workload_sensitivity": cfg.missing_workload_sensitivity,
            },
            "communication_types": list(COMMUNICATION_TYPES),
            "text_topics": list(TEXT_TOPICS),
        }
