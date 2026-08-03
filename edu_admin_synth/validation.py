"""
validation.py
=============

Automated data-quality validation and reporting.

After generation the fused dataset is put through a battery of checks that a
publication-grade synthetic benchmark is expected to pass, and the results are
written out as CSV artefacts, distribution plots and a single self-contained
HTML report.

Checks performed
----------------
* Summary statistics
* Missing-value report
* Correlation matrix (+ heatmap)
* Distribution plots
* Class-balance report
* Outlier report (IQR rule)
* Duplicate detection
* Feature-importance proxy (RandomForest on the behaviour category)
* Temporal-consistency report (per-administrator lag-1 autocorrelation)
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend for server / sandbox environments
import matplotlib.pyplot as plt  # noqa: E402

try:  # Feature-importance proxy is optional if sklearn is unavailable.
    from sklearn.ensemble import RandomForestClassifier
    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    _HAVE_SKLEARN = False


@dataclass
class ValidationResults:
    """Container for all computed validation artefacts."""

    summary_statistics: pd.DataFrame
    missing_report: pd.DataFrame
    correlation_matrix: pd.DataFrame
    class_balance: pd.DataFrame
    outlier_report: pd.DataFrame
    duplicate_count: int
    feature_importance: pd.DataFrame
    temporal_consistency: pd.DataFrame
    plot_paths: List[Path] = field(default_factory=list)


class Validator:
    """Runs validation checks and emits reports for the fused dataset."""

    # Continuous label columns of primary research interest.
    _LABEL_SCORES = [
        "leadership_score",
        "communication_score",
        "administrative_efficiency",
        "stress_score",
        "engagement_score",
    ]

    def __init__(self, output_dir: Path, make_plots: bool = True) -> None:
        self._output_dir = Path(output_dir)
        self._plots_dir = self._output_dir / "plots"
        self._make_plots = make_plots

    # ------------------------------------------------------------------
    def _numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the numeric sub-frame (excluding the month-step index)."""
        num = df.select_dtypes(include=[np.number]).copy()
        return num.drop(columns=["month_step"], errors="ignore")

    # ------------------------------------------------------------------
    def _summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Descriptive statistics for every numeric column."""
        return self._numeric(df).describe().T

    # ------------------------------------------------------------------
    def _missing_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-column missing count and percentage."""
        total = len(df)
        missing = df.isna().sum()
        report = pd.DataFrame(
            {
                "missing_count": missing,
                "missing_pct": (missing / total * 100).round(3),
            }
        )
        return report.sort_values("missing_pct", ascending=False)

    # ------------------------------------------------------------------
    def _correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pearson correlation matrix over numeric columns."""
        return self._numeric(df).corr(numeric_only=True).round(4)

    # ------------------------------------------------------------------
    def _class_balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Frequency and percentage of each behaviour category."""
        if "behaviour_category" not in df.columns:
            return pd.DataFrame()
        counts = df["behaviour_category"].value_counts(dropna=False)
        return pd.DataFrame(
            {
                "count": counts,
                "percentage": (counts / len(df) * 100).round(2),
            }
        )

    # ------------------------------------------------------------------
    def _outlier_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count IQR-rule outliers per numeric column."""
        num = self._numeric(df)
        rows: List[Dict[str, object]] = []
        for col in num.columns:
            series = num[col].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((series < low) | (series > high)).sum())
            rows.append(
                {
                    "feature": col,
                    "outlier_count": n_out,
                    "outlier_pct": round(n_out / len(series) * 100, 3),
                    "lower_bound": round(float(low), 3),
                    "upper_bound": round(float(high), 3),
                }
            )
        return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)

    # ------------------------------------------------------------------
    def _duplicates(self, df: pd.DataFrame) -> int:
        """Number of duplicate ``(admin_id, month)`` records (should be 0)."""
        return int(df.duplicated(subset=["admin_id", "month"]).sum())

    # ------------------------------------------------------------------
    def _feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """RandomForest feature-importance proxy for the behaviour category.

        Falls back to absolute correlation with the leadership score when
        scikit-learn is unavailable.
        """
        num = self._numeric(df)
        # Exclude the direct label scores so importance reflects the modalities.
        feature_cols = [c for c in num.columns if c not in self._LABEL_SCORES]
        features = num[feature_cols].fillna(num[feature_cols].median())

        if _HAVE_SKLEARN and "behaviour_category" in df.columns:
            target = df["behaviour_category"].fillna("Average")
            model = RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1
            )
            model.fit(features, target)
            imp = pd.DataFrame(
                {"feature": feature_cols, "importance": model.feature_importances_}
            )
            return imp.sort_values("importance", ascending=False).reset_index(drop=True)

        # Fallback proxy.
        corr = (
            features.corrwith(num.get("leadership_score", pd.Series(dtype=float)))
            .abs()
            .fillna(0.0)
        )
        return (
            pd.DataFrame({"feature": corr.index, "importance": corr.values})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def _temporal_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean per-administrator lag-1 autocorrelation of key scores.

        Values well above zero confirm that behaviour evolves smoothly over
        time rather than being redrawn independently each month.
        """
        rows: List[Dict[str, object]] = []
        for score in self._LABEL_SCORES:
            if score not in df.columns:
                continue
            autocorrs: List[float] = []
            for _, group in df.sort_values("month_step").groupby("admin_id"):
                series = group[score].dropna()
                if len(series) > 2 and series.std() > 1e-6:
                    ac = series.autocorr(lag=1)
                    if not np.isnan(ac):
                        autocorrs.append(ac)
            rows.append(
                {
                    "score": score,
                    "mean_lag1_autocorr": round(float(np.mean(autocorrs)), 4)
                    if autocorrs
                    else np.nan,
                    "n_admins": len(autocorrs),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _make_distribution_plots(self, df: pd.DataFrame, corr: pd.DataFrame) -> List[Path]:
        """Render distribution / balance / correlation plots to PNG files."""
        if not self._make_plots:
            return []
        self._plots_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []

        # 1) Distributions of the five output scores.
        present_scores = [s for s in self._LABEL_SCORES if s in df.columns]
        if present_scores:
            fig, axes = plt.subplots(1, len(present_scores), figsize=(4 * len(present_scores), 3.2))
            axes = np.atleast_1d(axes)
            for ax, score in zip(axes, present_scores):
                df[score].dropna().hist(ax=ax, bins=25, color="#4C72B0", edgecolor="white")
                ax.set_title(score, fontsize=9)
                ax.tick_params(labelsize=7)
            fig.suptitle("Output score distributions")
            fig.tight_layout()
            p = self._plots_dir / "score_distributions.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            paths.append(p)

        # 2) Class balance bar chart.
        if "behaviour_category" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            df["behaviour_category"].value_counts().plot(kind="bar", ax=ax, color="#55A868")
            ax.set_title("Behaviour category class balance")
            ax.set_ylabel("count")
            ax.tick_params(axis="x", labelrotation=25, labelsize=8)
            fig.tight_layout()
            p = self._plots_dir / "class_balance.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            paths.append(p)

        # 3) Correlation heatmap.
        if not corr.empty:
            fig, ax = plt.subplots(figsize=(10, 9))
            im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.index)))
            ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
            ax.set_yticklabels(corr.index, fontsize=5)
            ax.set_title("Feature correlation matrix")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            p = self._plots_dir / "correlation_heatmap.png"
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths.append(p)

        # 4) Temporal trajectory sample (mean leadership over months).
        if "leadership_score" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 3.4))
            monthly = df.groupby("month_step")["leadership_score"].mean()
            monthly.plot(ax=ax, marker="o", color="#C44E52")
            ax.set_title("Mean leadership score over time")
            ax.set_xlabel("month step")
            ax.set_ylabel("mean score")
            fig.tight_layout()
            p = self._plots_dir / "temporal_leadership.png"
            fig.savefig(p, dpi=110)
            plt.close(fig)
            paths.append(p)

        return paths

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> ValidationResults:
        """Execute every validation check and return the collected results."""
        corr = self._correlation_matrix(df)
        results = ValidationResults(
            summary_statistics=self._summary_statistics(df),
            missing_report=self._missing_report(df),
            correlation_matrix=corr,
            class_balance=self._class_balance(df),
            outlier_report=self._outlier_report(df),
            duplicate_count=self._duplicates(df),
            feature_importance=self._feature_importance(df),
            temporal_consistency=self._temporal_consistency(df),
            plot_paths=self._make_distribution_plots(df, corr),
        )
        return results

    # ------------------------------------------------------------------
    def write_html_report(
        self, df: pd.DataFrame, results: ValidationResults, path: Path
    ) -> None:
        """Write a self-contained HTML validation report (plots embedded)."""

        def embed(img_path: Path) -> str:
            data = base64.b64encode(img_path.read_bytes()).decode("ascii")
            return f'<img src="data:image/png;base64,{data}" style="max-width:100%;"/>'

        def table(frame: pd.DataFrame, max_rows: int = 60) -> str:
            return frame.head(max_rows).to_html(classes="tbl", border=0)

        plots_html = "".join(
            f"<div class='plot'><h3>{p.stem.replace('_', ' ').title()}</h3>{embed(p)}</div>"
            for p in results.plot_paths
        )

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Synthetic Educational Administrative Behaviour - Validation Report</title>
<style>
 body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color:#222; }}
 h1 {{ color:#2b4c7e; }} h2 {{ color:#3a6ea5; border-bottom:2px solid #eee; padding-bottom:4px; }}
 .tbl {{ border-collapse: collapse; font-size: 12px; margin-bottom: 20px; }}
 .tbl td, .tbl th {{ border: 1px solid #ddd; padding: 4px 8px; text-align:right; }}
 .tbl th {{ background:#f2f6fb; }}
 .kpi {{ display:inline-block; background:#f2f6fb; border-radius:8px; padding:10px 16px;
         margin:6px; font-size:14px; }}
 .plot {{ margin: 16px 0; }}
</style></head><body>
<h1>Validation Report</h1>
<p>Multimodal Representation Learning for Educational Administrative Behaviour Analysis
&mdash; synthetic benchmark dataset.</p>
<div>
 <span class="kpi"><b>Records:</b> {len(df):,}</span>
 <span class="kpi"><b>Features:</b> {df.shape[1]}</span>
 <span class="kpi"><b>Duplicates:</b> {results.duplicate_count}</span>
 <span class="kpi"><b>Administrators:</b> {df['admin_id'].nunique()}</span>
 <span class="kpi"><b>Months:</b> {df['month_step'].nunique()}</span>
</div>

<h2>Class Balance</h2>{table(results.class_balance)}
<h2>Temporal Consistency (lag-1 autocorrelation)</h2>{table(results.temporal_consistency)}
<h2>Top Feature Importances</h2>{table(results.feature_importance, 25)}
<h2>Missing-Value Report</h2>{table(results.missing_report, 40)}
<h2>Outlier Report</h2>{table(results.outlier_report, 40)}
<h2>Summary Statistics</h2>{table(results.summary_statistics, 80)}
<h2>Diagnostic Plots</h2>{plots_html}
</body></html>"""

        path.write_text(html, encoding="utf-8")
