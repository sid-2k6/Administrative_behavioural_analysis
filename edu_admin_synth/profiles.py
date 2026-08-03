"""
profiles.py
===========

Latent behavioural profile generator.

Assigns every administrator to one of the five *hidden* profiles (Highly
Effective, Effective, Average, Overloaded, Needs Improvement) while exactly
honouring the target class proportions, then draws each administrator's
personal latent *baseline* trait vector from that profile's distribution.

The profile label is retained internally (for validation / diagnostics) but
is **never** written to any exported modality file.  Only labels *derived*
from the latent state are exported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from faker import Faker

from .config import PROFILES, Config, ProfileSpec

INSTITUTION_TYPES = (
    "Public University",
    "Private University",
    "Community College",
    "Technical Institute",
    "Autonomous College",
)

DEPARTMENTS = (
    "Academic Affairs",
    "Student Services",
    "Administration",
    "Examinations",
    "Research & Development",
    "Human Resources",
    "Finance",
    "Quality Assurance",
)

DESIGNATIONS = (
    "Principal",
    "Vice Principal",
    "Dean",
    "Head of Department",
    "Registrar",
    "Coordinator",
    "Director",
)


@dataclass
class Administrator:
    """A single simulated administrator with a hidden latent baseline.

    Attributes
    ----------
    admin_id:
        Stable identifier of the form ``A0001``.
    profile:
        The hidden :class:`ProfileSpec` (ground truth, not exported).
    baseline:
        Per-trait baseline latent values (0..1) sampled from the profile.
    accreditation_month:
        The 1-based simulation step during which this administrator's
        institution undergoes an accreditation visit, or ``None``.
    demographics:
        Faker-generated descriptive attributes (name, department, ...).
    """

    admin_id: str
    profile: ProfileSpec
    baseline: Dict[str, float]
    accreditation_month: int | None
    demographics: Dict[str, object] = field(default_factory=dict)


class ProfileGenerator:
    """Builds the administrator population and their latent baselines."""

    def __init__(self, config: Config, rng: np.random.Generator, faker: Faker) -> None:
        self._config = config
        self._rng = rng
        self._faker = faker

    # ------------------------------------------------------------------
    def _assign_profiles(self) -> List[ProfileSpec]:
        """Return a shuffled profile assignment honouring exact proportions.

        Using the largest-remainder method guarantees the realised class
        balance matches the configured proportions as closely as integer
        counts allow, rather than relying on random sampling variance.
        """
        n = self._config.n_administrators
        raw_counts = {p.name: p.proportion * n for p in PROFILES}
        floor_counts = {name: int(np.floor(v)) for name, v in raw_counts.items()}
        remainder = n - sum(floor_counts.values())

        # Distribute the leftover records to the profiles with the largest
        # fractional parts.
        fractional = sorted(
            ((raw_counts[p.name] - floor_counts[p.name], p.name) for p in PROFILES),
            reverse=True,
        )
        for i in range(remainder):
            floor_counts[fractional[i][1]] += 1

        assignment: List[ProfileSpec] = []
        by_name = {p.name: p for p in PROFILES}
        for name, count in floor_counts.items():
            assignment.extend([by_name[name]] * count)

        self._rng.shuffle(assignment)  # type: ignore[arg-type]
        return assignment

    # ------------------------------------------------------------------
    def _sample_baseline(self, profile: ProfileSpec) -> Dict[str, float]:
        """Draw one administrator's baseline latent vector from a profile.

        Each trait is a Normal draw around the profile mean with the profile's
        between-administrator std, clipped to the valid [0, 1] range.
        """
        baseline: Dict[str, float] = {}
        for trait, mean in profile.trait_means.items():
            value = self._rng.normal(mean, profile.trait_std)
            baseline[trait] = float(np.clip(value, 0.02, 0.98))
        return baseline

    # ------------------------------------------------------------------
    def _make_demographics(self) -> Dict[str, object]:
        """Generate plausible descriptive attributes via Faker."""
        gender = self._rng.choice(["Female", "Male"])
        name = (
            self._faker.name_female() if gender == "Female" else self._faker.name_male()
        )
        return {
            "name": name,
            "gender": str(gender),
            "age": int(self._rng.integers(34, 63)),
            "years_experience": int(self._rng.integers(3, 32)),
            "institution_type": str(self._rng.choice(INSTITUTION_TYPES)),
            "department": str(self._rng.choice(DEPARTMENTS)),
            "designation": str(self._rng.choice(DESIGNATIONS)),
        }

    # ------------------------------------------------------------------
    def generate(self) -> List[Administrator]:
        """Create the full administrator population.

        Returns
        -------
        list of :class:`Administrator`
            ``config.n_administrators`` administrators, each with a hidden
            profile, latent baseline, demographics and (possibly) an
            accreditation month.
        """
        profiles = self._assign_profiles()
        administrators: List[Administrator] = []

        for idx, profile in enumerate(profiles, start=1):
            admin_id = f"A{idx:04d}"
            baseline = self._sample_baseline(profile)

            accreditation_month: int | None = None
            if self._rng.random() < self._config.accreditation_prob:
                accreditation_month = int(
                    self._rng.integers(1, self._config.n_months + 1)
                )

            administrators.append(
                Administrator(
                    admin_id=admin_id,
                    profile=profile,
                    baseline=baseline,
                    accreditation_month=accreditation_month,
                    demographics=self._make_demographics(),
                )
            )

        return administrators
