"""
fusion.py
=========

Multimodal fusion module.

Aligns and merges the three independently-generated modalities (text,
administrative logs, survey aggregates) with the derived labels and static
administrator demographics on the ``(admin_id, month)`` key, producing the
unified ``monthly_dataset`` that downstream transformer-fusion / classical
models consume.

The text modality is many-documents-per-month, so it is first aggregated into
per-administrator-month features (counts, mean sentiment/readability, dominant
communication type / topic / urgency) before the join.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .profiles import Administrator
from .text_generator import TextDocument


def _mode_or_none(series: pd.Series) -> object:
    """Return the most frequent value in a Series (first on ties), else None."""
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else None


class FusionModule:
    """Aligns modalities into a single administrator-month table."""

    def __init__(self) -> None:
        self._key = ["admin_id", "month_step"]

    # ------------------------------------------------------------------
    def aggregate_text(self, documents: List[TextDocument]) -> pd.DataFrame:
        """Collapse the document-level text modality to admin-month features."""
        if not documents:
            return pd.DataFrame(
                columns=["admin_id", "month", "month_step", "text_document_count"]
            )
        df = pd.DataFrame([doc.__dict__ for doc in documents])
        grouped = df.groupby(self._key)
        agg = grouped.agg(
            month=("month", "first"),
            text_document_count=("document_id", "count"),
            text_avg_word_count=("word_count", "mean"),
            text_avg_sentiment=("sentiment_score", "mean"),
            text_avg_positive_ratio=("positive_word_ratio", "mean"),
            text_avg_negative_ratio=("negative_word_ratio", "mean"),
            text_avg_readability=("readability_score", "mean"),
            text_dominant_type=("communication_type", _mode_or_none),
            text_dominant_topic=("topic", _mode_or_none),
            text_dominant_urgency=("urgency_level", _mode_or_none),
            text_dominant_style=("writing_style", _mode_or_none),
        ).reset_index()
        # Round the numeric aggregates.
        for col in [
            "text_avg_word_count",
            "text_avg_sentiment",
            "text_avg_positive_ratio",
            "text_avg_negative_ratio",
            "text_avg_readability",
        ]:
            agg[col] = agg[col].round(4)
        return agg

    # ------------------------------------------------------------------
    def _demographics_frame(self, administrators: List[Administrator]) -> pd.DataFrame:
        """Build a static per-administrator demographics table."""
        rows: List[Dict[str, object]] = []
        for admin in administrators:
            row = {"admin_id": admin.admin_id}
            row.update(admin.demographics)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def fuse(
        self,
        administrators: List[Administrator],
        logs_rows: List[Dict[str, object]],
        survey_agg_rows: List[Dict[str, object]],
        label_rows: List[Dict[str, object]],
        documents: List[TextDocument],
    ) -> pd.DataFrame:
        """Merge all modalities and labels into the unified monthly dataset.

        Returns
        -------
        pandas.DataFrame
            One row per administrator-month with demographic, text, log, survey
            and label columns.  The hidden profile is intentionally absent.
        """
        logs_df = pd.DataFrame(logs_rows)
        survey_df = pd.DataFrame(survey_agg_rows)
        labels_df = pd.DataFrame(label_rows)
        text_df = self.aggregate_text(documents)
        demo_df = self._demographics_frame(administrators)

        # Start from logs (dense, one row per admin-month) and left-join the rest.
        merged = logs_df.merge(
            survey_df.drop(columns=["month"], errors="ignore"),
            on=self._key,
            how="left",
        )
        merged = merged.merge(
            labels_df.drop(columns=["month"], errors="ignore"),
            on=self._key,
            how="left",
        )
        merged = merged.merge(
            text_df.drop(columns=["month"], errors="ignore"),
            on=self._key,
            how="left",
        )
        merged = merged.merge(demo_df, on="admin_id", how="left")

        # Order columns: identifiers -> demographics -> text -> logs -> survey -> labels.
        id_cols = ["admin_id", "month", "month_step"]
        ordered = id_cols + [c for c in merged.columns if c not in id_cols]
        merged = merged[ordered].sort_values(self._key).reset_index(drop=True)
        return merged
