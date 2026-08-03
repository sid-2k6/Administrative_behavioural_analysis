"""
pipeline.py
===========

End-to-end orchestration of the synthetic dataset generation pipeline.

    Config
      -> Profiles
      -> Temporal Simulator
      -> {Text, Administrative Logs, Survey}
      -> Labels
      -> Fusion
      -> Missingness / Noise
      -> Validation
      -> Metadata
      -> Export

The single public entry point is :func:`run_pipeline`, which is fully
reproducible: both ``numpy`` and ``random`` global seeds are fixed, a seeded
``numpy.random.Generator`` is threaded through every component, and Faker is
seeded as well.
"""

from __future__ import annotations

import random
from typing import List, Set

import numpy as np
import pandas as pd
from faker import Faker

from .admin_logs import AdminLogGenerator
from .config import Config, DEFAULT_CONFIG
from .export import Exporter
from .fusion import FusionModule
from .labels import LabelGenerator
from .metadata import MetadataGenerator
from .missingness import MissingnessInjector, build_workload_map
from .profiles import ProfileGenerator
from .survey import SurveyGenerator
from .temporal import TemporalSimulator
from .text_generator import TextGenerator
from .validation import Validator


def _log(msg: str, verbose: bool) -> None:
    """Minimal stdout progress logger."""
    if verbose:
        print(f"[edu_admin_synth] {msg}")


def run_pipeline(config: Config | None = None, verbose: bool = True) -> pd.DataFrame:
    """Run the full generation pipeline and export every artefact.

    Parameters
    ----------
    config:
        Optional :class:`~edu_admin_synth.config.Config`.  Defaults to the
        package default (300 administrators x 12 months = 3600 records).
    verbose:
        If ``True`` (default), print progress messages.

    Returns
    -------
    pandas.DataFrame
        The fused ``monthly_dataset`` (also written to disk).
    """
    config = config or DEFAULT_CONFIG
    config.validate()

    # --- Reproducibility --------------------------------------------------
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    Faker.seed(config.random_seed)
    rng = np.random.default_rng(config.random_seed)
    faker = Faker()

    # --- 1. Latent profiles ----------------------------------------------
    _log("Generating administrator population and latent profiles...", verbose)
    administrators = ProfileGenerator(config, rng, faker).generate()

    # --- 2. Temporal latent trajectories ---------------------------------
    _log("Simulating temporal latent trajectories...", verbose)
    states = TemporalSimulator(config, rng).simulate(administrators)

    # --- 3-5. Modalities --------------------------------------------------
    _log("Generating text modality...", verbose)
    documents = TextGenerator(config, rng).generate(states)

    _log("Generating administrative-log modality...", verbose)
    logs_rows = AdminLogGenerator(config, rng).generate(states)

    _log("Generating survey modality...", verbose)
    survey_individual_rows, survey_aggregate_rows = SurveyGenerator(config, rng).generate(states)

    # --- 6. Labels --------------------------------------------------------
    _log("Deriving output labels from latent state...", verbose)
    label_rows = LabelGenerator(config, rng).generate(states)

    # --- 7. Fusion --------------------------------------------------------
    _log("Fusing modalities into the monthly dataset...", verbose)
    monthly_dataset = FusionModule().fuse(
        administrators, logs_rows, survey_aggregate_rows, label_rows, documents
    )

    # Assemble raw modality frames.
    logs_df = pd.DataFrame(logs_rows)
    survey_individual_df = pd.DataFrame(survey_individual_rows)
    survey_aggregate_df = pd.DataFrame(survey_aggregate_rows)
    text_documents_df = Exporter.documents_to_frame(documents)

    # --- 8. Missingness (MNAR) -------------------------------------------
    _log("Injecting workload-driven missingness...", verbose)
    workload_map = build_workload_map(states)
    injector = MissingnessInjector(config, rng, workload_map)

    logs_df = injector.inject(
        logs_df, config.missing_rate_logs, protected={"admin_id", "month", "month_step"}
    )
    survey_individual_df = injector.inject(
        survey_individual_df,
        config.missing_rate_survey,
        protected={"admin_id", "month", "month_step", "teacher_id"},
    )
    text_documents_df = injector.inject(
        text_documents_df,
        config.missing_rate_text,
        protected={
            "document_id", "admin_id", "month", "month_step",
            "communication_type", "subject", "document_text",
        },
    )
    monthly_dataset = _inject_fused_missingness(
        monthly_dataset, injector, config, logs_df, survey_aggregate_df, text_documents_df
    )

    # --- 9. Validation ----------------------------------------------------
    _log("Running validation checks...", verbose)
    validator = Validator(config.output_dir, make_plots=config.make_plots)
    results = validator.run(monthly_dataset)

    # --- 10. Metadata -----------------------------------------------------
    _log("Building metadata and feature dictionary...", verbose)
    meta_gen = MetadataGenerator(config)
    metadata = meta_gen.build_metadata(monthly_dataset)
    feature_dictionary = meta_gen.build_feature_dictionary(monthly_dataset)

    # --- 11. Export -------------------------------------------------------
    _log("Exporting all artefacts...", verbose)
    exporter = Exporter(config.output_dir)
    exporter.export_modalities(
        logs_df, text_documents_df, survey_individual_df,
        survey_aggregate_df, monthly_dataset,
    )
    exporter.export_metadata(metadata, feature_dictionary)
    exporter.export_validation(results, monthly_dataset, validator)

    _log(
        f"Done. {len(monthly_dataset):,} records, {len(documents):,} documents, "
        f"{len(survey_individual_df):,} survey responses -> '{config.output_dir}/'.",
        verbose,
    )
    return monthly_dataset


def _inject_fused_missingness(
    monthly_dataset: pd.DataFrame,
    injector: MissingnessInjector,
    config: Config,
    logs_df: pd.DataFrame,
    survey_aggregate_df: pd.DataFrame,
    text_documents_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply modality-specific missingness rates to the fused dataset.

    Each modality's column-group in the fused table receives missingness at
    that modality's own base rate (logs 2%, survey 5%, text 1%), while all
    other columns are protected during that pass.
    """
    keys = {"admin_id", "month", "month_step"}
    all_cols: Set[str] = set(monthly_dataset.columns)

    log_cols = (set(logs_df.columns) & all_cols) - keys
    survey_cols = (set(survey_aggregate_df.columns) & all_cols) - keys
    text_cols = {c for c in all_cols if c.startswith("text_")}

    df = monthly_dataset
    df = injector.inject(df, config.missing_rate_logs, protected=all_cols - log_cols)
    df = injector.inject(df, config.missing_rate_survey, protected=all_cols - survey_cols)
    df = injector.inject(df, config.missing_rate_text, protected=all_cols - text_cols)
    return df
