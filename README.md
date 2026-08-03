# Synthetic Multimodal Educational Administrative Behaviour Dataset

A research-grade generator that produces a **statistically grounded, internally
consistent, reproducible** synthetic multimodal dataset for the study:

> **Multimodal Representation Learning for Educational Administrative Behavior
> Analysis: A Transformer-Based Fusion Framework**

Each record is **one educational administrator during one month**. The dataset is
driven by *hidden* latent behavioural profiles and smooth temporal dynamics, from
which three correlated modalities (text, administrative logs, teacher surveys),
output labels, validation reports and metadata are derived.

---

## Quick start

```bash
pip install -r requirements.txt

# Default: 300 administrators x 12 months = 3600 records
python generate_dataset.py

# Custom size / location
python generate_dataset.py --administrators 100 --months 12 --teachers 20 \
    --seed 42 --output-dir output
```

All artefacts are written to `output/` (see below). Requires **Python 3.11+**.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--administrators` | 300 | Number of administrators |
| `--months` | 12 | Months per administrator |
| `--teachers` | 20 | Survey respondents per administrator/month |
| `--seed` | 42 | Master seed (full reproducibility) |
| `--output-dir` | `output` | Output directory |
| `--no-plots` | off | Skip diagnostic plots |
| `--quiet` | off | Suppress progress logging |

---

## Design overview

```
Config -> Profiles -> Temporal Simulator -> {Text, Logs, Survey}
       -> Labels -> Fusion -> Missingness/Noise -> Validation -> Metadata -> Export
```

* **Hidden profiles** (never exported): Highly Effective 20%, Effective 35%,
  Average 25%, Overloaded 10%, Needs Improvement 10%. Each defines distributions
  over nine latent traits (communication, leadership, task completion,
  responsiveness, attendance, sentiment, engagement, stress, workload).
* **Temporal model**: per-administrator baseline + an **AR(1) random walk** +
  **exponentially-smoothed seasonal/event effects** (Semester Start/End, Exam
  Period, Accreditation Visit, Faculty Recruitment, Budget Planning, Holiday
  Season). This yields smooth, autocorrelated trajectories without unrealistic
  month-to-month jumps.
* **Cross-modal consistency**: text, logs and surveys are all derived from the
  *same* monthly latent state, so they reinforce one another (e.g. positive
  emails -> fast responses -> high survey ratings -> high leadership score).
* **Realistic distributions**: Poisson (counts), Log-Normal (response time),
  Beta (percentages / task completion), Normal (working hours), Gamma (word
  count), Truncated-Normal (sentiment), Ordinal (Likert surveys).
* **Missingness**: MNAR — probability rises with workload (survey ~5%,
  logs ~2%, text ~1%).
* **Labels**: five 0–100 scores plus a 5-class `behaviour_category`, all derived
  from latent variables with the target class imbalance guaranteed.

---

## Modalities

* **Text** — unique educational communication (emails, minutes, circulars,
  notices, reports, faculty communication) with sentiment, positive/negative word
  ratios, urgency, topic, readability and profile-dependent writing style.
* **Administrative logs** — 23 operational metrics (meetings, approvals, backlog,
  response time, attendance, decisions, interactions, ...).
* **Survey** — 20 teachers rate 14 Likert items per administrator/month; both the
  raw responses and monthly aggregates (mean / median / std) are exported.

---

## Output files (`output/`)

| File | Contents |
|------|----------|
| `administrative_logs.csv` | Raw administrative-log modality |
| `text_documents.csv` | Document-level text modality |
| `survey_responses.csv` | Individual teacher responses |
| `survey_aggregates.csv` | Per administrator-month survey aggregates |
| `monthly_dataset.csv` | **Fused** multimodal dataset (one row per admin-month) |
| `metadata.json` | Full generation logic, distributions, correlation assumptions |
| `feature_dictionary.csv` | Per-feature modality, dtype, range, distribution, description |
| `correlation_matrix.csv` | Numeric feature correlation matrix |
| `summary_statistics.csv` | Descriptive statistics |
| `missing_value_report.csv` | Per-column missingness |
| `outlier_report.csv` | IQR-based outlier counts |
| `class_balance.csv` | Behaviour-category distribution |
| `feature_importance.csv` | RandomForest feature-importance proxy |
| `temporal_consistency.csv` | Per-administrator lag-1 autocorrelation |
| `validation_report.html` | Self-contained report (tables + embedded plots) |
| `plots/*.png` | Distribution, class-balance, correlation, temporal plots |

---

## Package layout

```
edu_admin_synth/
  config.py          # all tunable parameters, profiles, events, distributions
  profiles.py        # hidden profile assignment + latent baselines + demographics
  temporal.py        # AR(1) + smoothed-event latent trajectory simulator
  text_generator.py  # Modality 1: compositional educational text engine
  admin_logs.py      # Modality 2: 23 administrative-log metrics
  survey.py          # Modality 3: teacher Likert surveys + aggregates
  labels.py          # output scores + behaviour category (latent-driven)
  fusion.py          # multimodal alignment/merge
  missingness.py     # MNAR workload-driven missingness
  validation.py      # checks, plots, HTML report
  metadata.py        # metadata.json + feature dictionary
  export.py          # serialisation of all artefacts
  pipeline.py        # end-to-end orchestration (run_pipeline)
generate_dataset.py  # CLI entry point
```

## Reproducibility

`numpy` and `random` global seeds are fixed and a seeded
`numpy.random.Generator` (plus seeded Faker) is threaded through every
component, so identical inputs produce byte-identical outputs.
