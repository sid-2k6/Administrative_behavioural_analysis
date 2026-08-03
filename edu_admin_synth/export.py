"""
export.py
=========

Export module.

Writes every required artefact to the configured output directory:

    administrative_logs.csv
    text_documents.csv
    survey_responses.csv
    monthly_dataset.csv
    metadata.json
    feature_dictionary.csv
    validation_report.html
    correlation_matrix.csv
    summary_statistics.csv

plus the diagnostic plots (under ``output/plots``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .text_generator import TextDocument
from .validation import ValidationResults


class Exporter:
    """Serialises all generated data and reports to disk."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def path(self, filename: str) -> Path:
        """Return the absolute path of ``filename`` inside the output dir."""
        return self._output_dir / filename

    # ------------------------------------------------------------------
    def export_modalities(
        self,
        logs_df: pd.DataFrame,
        text_documents_df: pd.DataFrame,
        survey_individual_df: pd.DataFrame,
        survey_aggregate_df: pd.DataFrame,
        monthly_dataset: pd.DataFrame,
    ) -> None:
        """Write the raw modality tables and the fused monthly dataset."""
        logs_df.to_csv(self.path("administrative_logs.csv"), index=False)
        text_documents_df.to_csv(self.path("text_documents.csv"), index=False)
        survey_individual_df.to_csv(self.path("survey_responses.csv"), index=False)
        survey_aggregate_df.to_csv(self.path("survey_aggregates.csv"), index=False)
        monthly_dataset.to_csv(self.path("monthly_dataset.csv"), index=False)

    # ------------------------------------------------------------------
    def export_metadata(
        self, metadata: Dict[str, object], feature_dictionary: pd.DataFrame
    ) -> None:
        """Write ``metadata.json`` and ``feature_dictionary.csv``."""
        with self.path("metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, ensure_ascii=False)
        feature_dictionary.to_csv(self.path("feature_dictionary.csv"), index=False)

    # ------------------------------------------------------------------
    def export_validation(
        self, results: ValidationResults, monthly_dataset: pd.DataFrame, validator
    ) -> None:
        """Write validation CSVs and the HTML report."""
        results.correlation_matrix.to_csv(self.path("correlation_matrix.csv"))
        results.summary_statistics.to_csv(self.path("summary_statistics.csv"))
        results.missing_report.to_csv(self.path("missing_value_report.csv"))
        results.outlier_report.to_csv(self.path("outlier_report.csv"), index=False)
        results.class_balance.to_csv(self.path("class_balance.csv"))
        results.feature_importance.to_csv(self.path("feature_importance.csv"), index=False)
        results.temporal_consistency.to_csv(
            self.path("temporal_consistency.csv"), index=False
        )
        validator.write_html_report(
            monthly_dataset, results, self.path("validation_report.html")
        )

    # ------------------------------------------------------------------
    @staticmethod
    def documents_to_frame(documents: List[TextDocument]) -> pd.DataFrame:
        """Convert the list of :class:`TextDocument` into a DataFrame."""
        return pd.DataFrame([doc.__dict__ for doc in documents])
