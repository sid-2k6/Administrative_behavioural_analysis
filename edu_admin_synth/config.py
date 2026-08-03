"""
config.py
=========

Central, fully-typed configuration for the synthetic dataset generator.

Every tunable knob of the simulation lives here so that dataset size,
statistical distributions, latent-profile behaviour, temporal dynamics,
missingness and noise can be changed in one place without touching the
generation logic.

All numeric constants are expressed on interpretable scales and documented
inline so the configuration doubles as living documentation of the data
generating process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Latent trait vocabulary
# ---------------------------------------------------------------------------
# Every administrator is described by a vector of latent behavioural traits,
# each on a normalised 0..1 scale where (unless noted) higher == "better".
#
#   communication   quality/clarity of written & verbal communication
#   leadership      vision, fairness, delegation, conflict resolution
#   task_completion reliability at finishing work on time
#   responsiveness  speed of reply/approvals (higher == faster)
#   attendance      physical presence / punctuality tendency
#   sentiment       positivity of written communication
#   engagement      proactive involvement with faculty/students
#   stress          psychological load           (higher == worse)
#   workload        volume of work carried        (higher == busier)
LATENT_TRAITS: Tuple[str, ...] = (
    "communication",
    "leadership",
    "task_completion",
    "responsiveness",
    "attendance",
    "sentiment",
    "engagement",
    "stress",
    "workload",
)

# Traits where a *higher* latent value is undesirable.
NEGATIVE_TRAITS: Tuple[str, ...] = ("stress", "workload")


# ---------------------------------------------------------------------------
# Latent behavioural profiles (the hidden ground truth)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProfileSpec:
    """Distributional specification of a single hidden behavioural profile.

    Attributes
    ----------
    name:
        Human readable profile name (never exported to the dataset).
    proportion:
        Fraction of the administrator population belonging to this profile.
    trait_means:
        Mean of every latent trait for this profile (0..1 scale).
    trait_std:
        Between-administrator standard deviation applied when sampling each
        administrator's personal baseline from ``trait_means``.
    """

    name: str
    proportion: float
    trait_means: Dict[str, float]
    trait_std: float = 0.07


# The five hidden profiles.  Proportions intentionally reproduce the target
# class imbalance requested by the research design.
PROFILES: Tuple[ProfileSpec, ...] = (
    ProfileSpec(
        name="Highly Effective",
        proportion=0.20,
        trait_means={
            "communication": 0.88,
            "leadership": 0.87,
            "task_completion": 0.86,
            "responsiveness": 0.85,
            "attendance": 0.90,
            "sentiment": 0.82,
            "engagement": 0.86,
            "stress": 0.28,
            "workload": 0.60,
        },
    ),
    ProfileSpec(
        name="Effective",
        proportion=0.35,
        trait_means={
            "communication": 0.72,
            "leadership": 0.71,
            "task_completion": 0.73,
            "responsiveness": 0.70,
            "attendance": 0.80,
            "sentiment": 0.68,
            "engagement": 0.70,
            "stress": 0.38,
            "workload": 0.55,
        },
    ),
    ProfileSpec(
        name="Average",
        proportion=0.25,
        trait_means={
            "communication": 0.55,
            "leadership": 0.54,
            "task_completion": 0.56,
            "responsiveness": 0.53,
            "attendance": 0.68,
            "sentiment": 0.54,
            "engagement": 0.53,
            "stress": 0.48,
            "workload": 0.52,
        },
    ),
    ProfileSpec(
        name="Overloaded",
        proportion=0.10,
        # Competent but crushed by volume: decent leadership/communication,
        # very high workload & stress which erode responsiveness.
        trait_means={
            "communication": 0.62,
            "leadership": 0.63,
            "task_completion": 0.55,
            "responsiveness": 0.45,
            "attendance": 0.66,
            "sentiment": 0.50,
            "engagement": 0.60,
            "stress": 0.82,
            "workload": 0.90,
        },
        trait_std=0.06,
    ),
    ProfileSpec(
        name="Needs Improvement",
        proportion=0.10,
        trait_means={
            "communication": 0.36,
            "leadership": 0.35,
            "task_completion": 0.37,
            "responsiveness": 0.38,
            "attendance": 0.55,
            "sentiment": 0.40,
            "engagement": 0.38,
            "stress": 0.62,
            "workload": 0.50,
        },
    ),
)


# ---------------------------------------------------------------------------
# Academic calendar / temporal events
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EventSpec:
    """A calendar/administrative event and its effect on latent traits.

    ``trait_delta`` values are *additive* shifts applied to latent traits for
    the month in which the event is active (before clipping to [0, 1]).
    ``intensity_multiplier`` scales count-based activity (meetings, workload
    volume) in the administrative-log generator.
    """

    name: str
    trait_delta: Dict[str, float]
    intensity_multiplier: float = 1.0


# Library of possible events.  Effects are calibrated to be realistic and
# to *never* produce unrealistic jumps once combined with drift + clipping.
EVENT_LIBRARY: Dict[str, EventSpec] = {
    "Semester Start": EventSpec(
        "Semester Start",
        {"workload": 0.14, "stress": 0.10, "engagement": 0.05, "responsiveness": -0.03},
        intensity_multiplier=1.25,
    ),
    "Semester End": EventSpec(
        "Semester End",
        {"workload": 0.10, "stress": 0.08, "task_completion": -0.02},
        intensity_multiplier=1.15,
    ),
    "Accreditation Visit": EventSpec(
        "Accreditation Visit",
        {"workload": 0.20, "stress": 0.18, "responsiveness": -0.06, "communication": 0.03},
        intensity_multiplier=1.40,
    ),
    "Faculty Recruitment": EventSpec(
        "Faculty Recruitment",
        {"workload": 0.12, "stress": 0.07, "engagement": 0.04},
        intensity_multiplier=1.20,
    ),
    "Exam Period": EventSpec(
        "Exam Period",
        {"workload": 0.16, "stress": 0.14, "responsiveness": -0.05, "attendance": 0.03},
        intensity_multiplier=1.30,
    ),
    "Budget Planning": EventSpec(
        "Budget Planning",
        {"workload": 0.11, "stress": 0.09, "leadership": 0.02},
        intensity_multiplier=1.15,
    ),
    "Holiday Season": EventSpec(
        "Holiday Season",
        {"workload": -0.14, "stress": -0.10, "responsiveness": -0.02, "engagement": -0.04},
        intensity_multiplier=0.75,
    ),
    "Routine": EventSpec("Routine", {}, intensity_multiplier=1.0),
}

# Deterministic month -> baseline event mapping for a Jan..Dec academic year.
# (Month index is 1-based.)  Stochastic events (e.g. Accreditation) are layered
# on top of these by the temporal simulator.
CALENDAR_EVENTS: Dict[int, str] = {
    1: "Semester Start",
    2: "Routine",
    3: "Exam Period",
    4: "Budget Planning",
    5: "Semester End",
    6: "Faculty Recruitment",
    7: "Semester Start",
    8: "Routine",
    9: "Routine",
    10: "Budget Planning",
    11: "Exam Period",
    12: "Holiday Season",
}


# ---------------------------------------------------------------------------
# Communication (text) modality configuration
# ---------------------------------------------------------------------------
COMMUNICATION_TYPES: Tuple[str, ...] = (
    "Official Email",
    "Meeting Minutes",
    "Circular",
    "Notice",
    "Academic Report",
    "Faculty Communication",
)

TEXT_TOPICS: Tuple[str, ...] = (
    "curriculum development",
    "examination scheduling",
    "faculty performance review",
    "student welfare",
    "accreditation preparation",
    "budget allocation",
    "research initiatives",
    "admission process",
    "infrastructure upgrade",
    "professional development",
    "disciplinary policy",
    "community outreach",
    "digital learning",
    "quality assurance",
    "staff recruitment",
)

URGENCY_LEVELS: Tuple[str, ...] = ("Low", "Medium", "High", "Critical")

# Word-count Gamma distribution parameters (shape, scale) -> mean ~ shape*scale.
# Clipped to [WORD_COUNT_MIN, WORD_COUNT_MAX].
WORD_COUNT_GAMMA: Tuple[float, float] = (9.0, 38.0)
WORD_COUNT_MIN: int = 150
WORD_COUNT_MAX: int = 600


# ---------------------------------------------------------------------------
# Survey modality configuration
# ---------------------------------------------------------------------------
LIKERT_MIN: int = 1
LIKERT_MAX: int = 5

# The 14 survey items and the latent traits that drive each item's mean.
# Weights within each item sum to ~1.0.
SURVEY_ITEMS: Dict[str, Dict[str, float]] = {
    "communicates_clearly": {"communication": 1.0},
    "provides_feedback": {"communication": 0.6, "responsiveness": 0.4},
    "approachable": {"communication": 0.5, "sentiment": 0.5},
    "motivates_staff": {"leadership": 0.6, "engagement": 0.4},
    "supports_teachers": {"leadership": 0.5, "communication": 0.5},
    "encourages_innovation": {"leadership": 0.7, "engagement": 0.3},
    "fairness": {"leadership": 1.0},
    "transparency": {"communication": 0.5, "leadership": 0.5},
    "responds_quickly": {"responsiveness": 1.0},
    "manages_workload": {"task_completion": 0.6, "stress": -0.4},
    "resolves_conflicts": {"leadership": 0.7, "communication": 0.3},
    "organizes_effectively": {"task_completion": 1.0},
    "overall_satisfaction": {
        "leadership": 0.3,
        "communication": 0.3,
        "responsiveness": 0.2,
        "engagement": 0.2,
    },
    "overall_effectiveness": {
        "leadership": 0.35,
        "task_completion": 0.35,
        "communication": 0.3,
    },
}

SURVEY_AGG_FUNCTIONS: Tuple[str, ...] = ("mean", "median", "std")


# ---------------------------------------------------------------------------
# Master configuration object
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Top-level configuration controlling one full dataset generation run."""

    # -- Reproducibility ---------------------------------------------------
    random_seed: int = 42

    # -- Dataset dimensions ------------------------------------------------
    n_administrators: int = 300
    n_months: int = 12
    teachers_per_admin: int = 20  # survey respondents per administrator/month
    start_year: int = 2024
    start_month: int = 1  # calendar month the simulation starts on (1..12)

    # -- Temporal dynamics -------------------------------------------------
    drift_std: float = 0.03           # monthly AR(1) innovation std
    drift_persistence: float = 0.88   # AR(1) persistence of the wander term
    event_carryover: float = 0.55     # exponential-smoothing carryover of event effects
    accreditation_prob: float = 0.35  # chance an institution has an accreditation year
    random_event_prob: float = 0.12   # chance of an extra stochastic event in a routine month

    # -- Distribution scale constants (documented in metadata) -------------
    # Administrative-log base rates (per month) before latent modulation.
    base_meetings: float = 16.0
    base_reports: float = 10.0
    base_approvals: float = 28.0
    base_working_hours: float = 45.0
    response_hours_lognorm_sigma: float = 0.5
    task_completion_days_scale: float = 12.0

    # -- Missingness (MNAR: increases with workload) -----------------------
    missing_rate_survey: float = 0.05
    missing_rate_logs: float = 0.02
    missing_rate_text: float = 0.01
    missing_workload_sensitivity: float = 1.8  # multiplier applied at max workload

    # -- Noise -------------------------------------------------------------
    survey_teacher_noise: float = 0.55   # per-teacher opinion dispersion (Likert std-ish)
    label_noise_std: float = 2.0         # noise (0..100 scale) on output scores
    contrarian_teacher_prob: float = 0.06  # teachers who deviate strongly

    # -- Output ------------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path("output"))
    make_plots: bool = True

    # ---- Derived helpers -------------------------------------------------
    @property
    def total_records(self) -> int:
        """Total number of administrator-month records in the fused dataset."""
        return self.n_administrators * self.n_months

    def month_labels(self) -> List[str]:
        """Return human-readable ``YYYY-MM`` labels for each simulated month."""
        labels: List[str] = []
        year, month = self.start_year, self.start_month
        for _ in range(self.n_months):
            labels.append(f"{year:04d}-{month:02d}")
            month += 1
            if month > 12:
                month = 1
                year += 1
        return labels

    def calendar_month_index(self, month_step: int) -> int:
        """Map a 1-based simulation step to a 1-based calendar month (1..12)."""
        return ((self.start_month - 1 + (month_step - 1)) % 12) + 1

    def validate(self) -> None:
        """Sanity-check the configuration, raising ``ValueError`` on problems."""
        total = sum(p.proportion for p in PROFILES)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Profile proportions must sum to 1.0 (got {total}).")
        if self.n_administrators <= 0 or self.n_months <= 0:
            raise ValueError("n_administrators and n_months must be positive.")
        if self.teachers_per_admin <= 0:
            raise ValueError("teachers_per_admin must be positive.")


DEFAULT_CONFIG = Config()
