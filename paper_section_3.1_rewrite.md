# 3.1 Simulation Framework and Benchmark Construction

> **Reviewer-response note (delete before submission).** This section is rewritten
> in the language of *design decisions*, not of observed summary statistics. Every
> quantity below (missingness rates, class balance, noise scales, seeds) is a
> parameter we *chose*, reconstructed verbatim from the generator configuration
> (`edu_admin_synth/config.py`, `profiles.py`, `temporal.py`, `survey.py`,
> `admin_logs.py`, `text_generator.py`, `labels.py`, `missingness.py`).

Because no public multimodal dataset couples free-text administrative communication,
operational activity logs, and teacher-survey ratings for the *same* administrators
over time, we constructed a controlled simulation benchmark. A simulation benchmark
has a decisive methodological advantage over a scraped corpus for this study: the
ground-truth generative process is known, so we can (i) guarantee that the
classification target is *not* a trivial function of any single observed feature,
(ii) inject realistic, mechanism-driven missingness, and (iii) evaluate whether the
model's temporal, fusion, and consistency modules recover structure we explicitly
built in. Every design parameter is stated below so the benchmark is fully
reproducible.

The simulation produces **1,500 administrators × 12 months = 18,000
administrator-month records**, each observed through four modalities and assigned a
single five-class behaviour label derived from a hidden latent state.

## 3.1.1 Generative Model

The data-generating process is a **latent-variable measurement model**. A hidden
behavioural state evolves per administrator over time; the four observable
modalities are *noisy, aggregated observations* of that state; and the label is a
deterministic ranking function of the latent state that no modality reports
directly. This structure is the reason the task is well-posed rather than circular
(see "Label rule" and §3.1.2).

### Administrator population

| Attribute | Design choice |
|---|---|
| Population | 1,500 administrators |
| Institution types | 5 (Public University, Private University, Community College, Technical Institute, Autonomous College), sampled uniformly |
| Departments | 8 (Academic Affairs, Student Services, Administration, Examinations, Research & Development, Human Resources, Finance, Quality Assurance), sampled uniformly |
| Designations | 7 (Principal, Vice Principal, Dean, HoD, Registrar, Coordinator, Director) |
| Demographic metadata | gender, age ∈ [34, 62], years experience ∈ [3, 31], plus the three categorical attributes above (Faker-generated; descriptive only, not used by the label) |
| Latent baseline | Each administrator is assigned one of five hidden *profiles* (see below) in exact proportions, then draws a personal 9-dimensional latent baseline **b**ₐ ~ 𝒩(profile mean, σ²) with σ = 0.07 (0.06 for Overloaded), clipped to [0.02, 0.98] |
| Accreditation year | With probability 0.35 an administrator's institution has one accreditation-visit month, drawn uniformly over the 12 months |

The five hidden profiles and their per-trait means (0–1 scale; higher = better
except **stress** and **workload** where higher = worse) are fixed design
parameters:

| Trait | Highly Eff. | Effective | Average | Overloaded | Needs Impr. |
|---|---|---|---|---|---|
| communication | 0.88 | 0.72 | 0.55 | 0.62 | 0.36 |
| leadership | 0.87 | 0.71 | 0.54 | 0.63 | 0.35 |
| task_completion | 0.86 | 0.73 | 0.56 | 0.55 | 0.37 |
| responsiveness | 0.85 | 0.70 | 0.53 | 0.45 | 0.38 |
| attendance | 0.90 | 0.80 | 0.68 | 0.66 | 0.55 |
| sentiment | 0.82 | 0.68 | 0.54 | 0.50 | 0.40 |
| engagement | 0.86 | 0.70 | 0.53 | 0.60 | 0.38 |
| stress | 0.28 | 0.38 | 0.48 | 0.82 | 0.62 |
| workload | 0.60 | 0.55 | 0.52 | 0.90 | 0.50 |
| **population share** | **20%** | **35%** | **25%** | **10%** | **10%** |

The "Overloaded" profile is deliberately *competent but crushed by volume* (decent
leadership/communication, very high workload and stress that erode responsiveness),
so that "Overloaded" is not simply a low-skill class — this forces the classifier to
distinguish overload from incompetence.

### Latent behavioural state (the variable the model must recover)

For each trait τ ∈ {communication, leadership, task_completion, responsiveness,
attendance, sentiment, engagement, stress, workload}, administrator a, and month t:

```
latentₐ,τ(t) = clip( bₐ,τ + wanderτ(t) + smoothed_deltaτ(t),  0, 1 )
```

with the two persistent components

```
wanderτ(t)         = φ · wanderτ(t−1) + ετ(t),        ετ ~ 𝒩(0, σ_drift²)      [AR(1)]
smoothed_deltaτ(t) = c · smoothed_deltaτ(t−1) + (1−c) · raw_deltaτ(t)         [event smoother]
```

and one deterministic coupling reflecting that busy administrators respond slower:

```
responsiveness(t) ← responsiveness(t) − 0.18 · max(0, workload(t) − 0.7)
```

Design constants: AR(1) persistence **φ = 0.88**, innovation std **σ_drift = 0.03**,
event carryover **c = 0.55**. `raw_deltaτ(t)` are additive shifts from the active
calendar/administrative events for that month.

### Temporal process

The month-to-month evolution is the AR(1) wander term above (persistence 0.88)
superimposed on the fixed baseline, plus **exponentially-smoothed event effects**
(carryover 0.55) so that events ramp up and decay rather than switching on/off. This
is precisely the autocorrelated, event-perturbed trajectory the temporal-memory
module is designed to recover. Events are:

- **Deterministic academic calendar** (fixed month→event map): Semester Start (Jan/Jul),
  Exam Period (Mar/Nov), Budget Planning (Apr/Oct), Semester End (May), Faculty
  Recruitment (Jun), Holiday Season (Dec), Routine otherwise.
- **Stochastic events**: an accreditation visit in the administrator's accreditation
  month; and, with probability 0.12, an extra event injected into an otherwise
  routine month.

Each event applies additive latent deltas (e.g. Accreditation Visit: workload +0.20,
stress +0.18, responsiveness −0.06, communication +0.03) and an activity-intensity
multiplier ∈ [0.6, 1.9] that scales count-based log volumes.

### Administrative modality (22 features)

Twenty-two operational metrics are generated per administrator-month; each
distribution's parameters are functions of the latent state (no feature is sampled
independently), following the design map:

| Family | Distribution | Example (parameterisation) |
|---|---|---|
| Counts | Poisson | `meetings_conducted ~ Poisson(16 · intensity · (0.5 + 0.5·engagement + 0.3·workload))` |
| Response time | Log-Normal | `avg_response_hours ~ LogNormal(ln 6 + 2.2·(1−responsiveness), 0.5)` |
| Percentages | Beta(mean, κ) | `attendance_% = 100 · Beta(mean = 0.6 + 0.38·attendance, κ = 60)` |
| Working hours | Normal | `hours/week ~ 𝒩(45·(0.75 + 0.5·workload), 3.5²)` |
| Completion time | Beta-derived | `avg_task_days = 1 + 12·(1 − Beta(mean = 0.85·task_completion + 0.05, κ = 20))` |

The 22 features are: meetings_conducted, meetings_attended, reports_submitted,
approvals_processed, leave_requests_processed, pending_tasks, overdue_tasks,
active_projects, average_response_hours, average_task_completion_days,
attendance_percentage, punctuality_percentage, working_hours_per_week,
meetings_per_week, policy_decisions, emergency_decisions, delegated_tasks,
office_visits, faculty_interactions, student_interactions, budget_requests,
complaint_resolutions.

### Survey modality (43 aggregate features)

For every administrator-month a panel of **20 teachers** rates the administrator on
**14 Likert items** (1–5). Each item j has a latent quality

```
item_latentⱼ = clip( Σ_τ wⱼτ · gτ ,  0, 1 ),   gτ = latentτ  (positive traits)
                                               gτ = 1 − latentτ (stress/workload)
mean_likertⱼ = 1 + 4 · item_latentⱼ
```

The 14 items and their trait weights are fixed (e.g. `communicates_clearly` =
1.0·communication; `manages_workload` = 0.6·task_completion − 0.4·stress;
`responds_quickly` = 1.0·responsiveness; `overall_effectiveness` = 0.35·leadership +
0.35·task_completion + 0.30·communication). Each teacher's rating adds a per-rater
bias ~ 𝒩(0, 0.35), per-response noise ~ 𝒩(0, 0.55), and with probability 0.06 the
teacher is *contrarian* (pulled toward the opposite pole with 𝒩(0, 1.1) noise). We
export the per-item **mean, median, std** over the 20 raters (14 × 3 = 42) plus the
respondent count → **43 aggregate features**. Aggregation over 20 raters is why the
survey is the lowest-noise observation of the latent state.

### Textual modality (35,201 documents)

Each administrator-month produces `Poisson(1.4 · intensity · (0.7 + 0.6·workload))`
documents, clipped to [1, 5] (≈1.96 documents/month on average → **35,201 documents**
for the full run). Each document samples one of **6 communication types**
(Official Email, Meeting Minutes, Circular, Notice, Academic Report, Faculty
Communication) and one of **15 topics** (curriculum development, examination
scheduling, … quality assurance, staff recruitment), both uniformly.

Documents are produced by a **compositional template-and-phrase-bank engine, not an
LLM**: greetings, sentence templates, sentiment-laden vocabulary and closings are
sampled with probabilities driven by the latent state. Four style variables are set
as explicit functions of the latent traits:

```
positivity    = 0.5·sentiment + 0.5·communication
formality     = 0.7·communication + 0.3·leadership
directiveness = 0.6·(1 − communication) + 0.4·stress
urgency       = 0.55·stress + 0.45·workload
```

Derived per-document features: `word_count ~ Gamma(shape 9, scale 38)` clipped to
[150, 600]; `sentiment_score = 0.6·(2·positivity − 1) + 0.4·(lexical balance) +
𝒩(0, 0.05)`; positive/negative word ratios from fixed 20-word lexicons; ordinal
`urgency_level` thresholded from `urgency + 𝒩(0, 0.05)` at 0.35 / 0.60 / 0.82
(Low/Medium/High/Critical); Flesch Reading-Ease `readability_score`. Thus urgency,
readability and negative-word ratio are all monotone functions of stress/workload
and communication, as required.

### Label rule (latent-derived — non-circular)

The five-class `behaviour_category` is a deterministic, class-balance-preserving
ranking of the **latent state only**. It is computed in two stages over all N
administrator-months. First, two latent composites (with small ±0.02 tie-breaking
noise) are formed:

```
overload_index_i = 0.45·workload_i + 0.45·stress_i − 0.10·task_completion_i + 𝒩(0,0.02)
effectiveness_i  = 0.30·leadership_i + 0.25·communication_i + 0.20·task_completion_i
                 + 0.15·responsiveness_i + 0.10·engagement_i − 0.10·stress_i + 𝒩(0,0.02)
```

- **Stage 1 (Overloaded).** The top **10%** of records by `overload_index` are
  labelled **Overloaded**.
- **Stage 2 (remainder).** The remaining 90% are ranked by `effectiveness` and split
  by quantile into **Highly Effective (top 20%)**, **Effective (next 35%)**,
  **Average (next 25%)**, **Needs Improvement (bottom 10%)**, using the exact target
  counts (largest-remainder rounding; any residual → Average).

Because the ranking is per administrator-month, a normally *Effective* administrator
can legitimately have an *Overloaded* month — labels vary within an administrator
over time. Crucially, **every input to the label is a latent trait; none of the 22
log features, 43 survey aggregates, or text features is used.** The survey/log/text
features are noisy, aggregated *observations* of the same latent traits, so they
correlate with the label (as they should) without the label being a function of any
observed feature. See §3.1.2.

> Five continuous scores (`leadership_score`, `communication_score`,
> `administrative_efficiency`, `stress_score`, `engagement_score`) are also derived
> near-deterministically from the latent traits (100·trait + 𝒩(0, 2)). Because these
> are essentially the label's own latent inputs rescaled to 0–100, they constitute
> label leakage and are **excluded from the model's feature set**; we retain only the
> genuinely observational modalities.

### Missingness (MNAR, per modality)

Missingness is **Missing-Not-At-Random**: it increases with latent workload,
mirroring the real phenomenon that overloaded staff under-report. For a record with
latent workload w, each eligible cell of a modality is dropped with probability

```
p = clip( base_rate · max(0.2, 1 + 1.8·(w − 0.5)),  0,  0.6 )
```

Identifier columns are protected. Design base rates: **survey 5%, administrative
logs 2%, text 1%**, with MNAR workload-sensitivity 1.8. These base rates and the
MNAR mechanism are the design parameters that deterministically produce the realised
missing-value counts of the released benchmark (verified by regenerating with seed 42):

- **62,862** missing cells in the fused `monthly_dataset.csv` (the modelling table),
- **10,273** missing cells in the raw `administrative_logs.csv`,
- **3,699** missing cells in the raw `text_documents.csv`.

All counts are reproducible bit-for-bit from the base rates above and the fixed
generation seed; they are chosen consequences of the design, not properties of a
found dataset.

### Class balance (a design parameter)

The 25 / 35 / 20 / 10 / 10 split (Average / Effective / Highly Effective /
Overloaded / Needs Improvement) is **chosen**, not observed. It is enforced twice —
once when assigning hidden profiles to administrators and again by the label
ranking — via largest-remainder rounding, so the realised balance matches the target
to within integer rounding. We chose a *deliberately imbalanced* distribution
(rather than uniform) because it reflects a realistic institutional population where
most administrators are effective/average and the operationally critical minority
classes (Overloaded, Needs Improvement) are rare — the regime where macro-averaged
metrics and the model's confidence gating matter most.

### Seeds

The entire pipeline is driven by a **single master seed (42)**: one NumPy
`Generator` seeded once produces every latent trajectory, modality draw, and
missingness mask, giving bit-for-bit reproducibility via
`python generate_dataset.py --administrators 1500 --months 12 --seed 42`. Model
training uses fixed seeds for NumPy, Python `random`, and PyTorch (CPU + CUDA), with
deterministic cuDNN; report the exact training seed value(s) used for the results in
Table X. *(Fill in your training seed here — it lives in the notebook, not the
generator repo.)*

## 3.1.2 Why the benchmark is well-posed (not circular)

A synthetic benchmark is only meaningful if the label cannot be recovered by
trivially inverting a function of a single observed feature. Our construction
guarantees this by design:

1. **The label depends only on the hidden latent state**, through the ranking rule
   above. No observed modality feature appears in the label computation.
2. **Each modality is a lossy observation** of that state: surveys average 20 noisy,
   biased raters; logs are Poisson/Beta/Log-Normal draws around latent-derived
   means; text features carry additive noise. Recovering the label therefore
   requires *denoising and integrating* the modalities to infer a variable never
   observed directly — exactly the problem EduFusionFormer targets.
3. The strongest SHAP predictors being survey aggregates (`responds_quickly_mean`,
   `manages_workload_mean`, `overall_effectiveness_mean`) is the *expected* signature
   of this design: surveys are the lowest-variance observation of the latent traits
   that dominate the effectiveness composite. It reflects informativeness, not
   leakage.

To quantify how much the multimodal architecture adds beyond the most informative
single modality, we report a **survey-only baseline** (§7). If it approaches the full
model, we report that honestly and frame the contribution as robustness to MNAR
missingness and cross-modal integration rather than as single-modality
insufficiency.

## Figure — Generative model schematic (to draw)

Render the following as a draw.io / Graphviz figure:

```
                         ┌─────────────────────────────────────────┐
                         │        Hidden latent state  z_a(t)        │
   profile baseline b_a ─►  9 traits · AR(1) φ=0.88 · event deltas  │
   + calendar/stochastic │  (communication, leadership, ...,        │
     events              │   stress, workload)                      │
                         └───────┬───────┬───────┬───────┬──────────┘
                                 │       │       │       │
                 (noisy obs.)    │       │       │       │  (ranking rule,
                                 ▼       ▼       ▼       │   latent-only)
                          ┌────────┐┌────────┐┌────────┐ ▼
                          │  Text  ││ Survey ││ Admin  │┌──────────────┐
                          │ 35,201 ││  43    ││ logs   ││  Label:      │
                          │  docs  ││ aggs   ││ 22 ftr ││  5 classes   │
                          └────────┘└────────┘└────────┘└──────────────┘
                          MNAR 1%   MNAR 5%   MNAR 2%   (25/35/20/10/10)

   Model observes {Text, Survey, Admin} across 12 months  ──►  predicts Label
   (must infer the unobserved latent state z to do so)
```
