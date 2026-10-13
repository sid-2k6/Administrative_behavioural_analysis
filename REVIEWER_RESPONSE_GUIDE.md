# EduFusionFormer — Reviewer Response Guide

What each reviewer comment actually asks for, and the specific edit required in the manuscript.

---

## 0. Before you edit anything: you have two different manuscripts

The repository contains **two `.docx` files with different table numbering**:

| File | Section 3.1 | Generative spec table | Architecture table | Hyperparameters table |
|---|---|---|---|---|
| `EduFusionFormers (1).docx` | older | Table 4 | Table 5 | **Table 7** |
| `EduFusionFormers.docx` | rewritten, adds `Figure 1a` schematic | Table 3a | Table 4 | **Table 6** |

**The reviewers read `EduFusionFormers (1).docx`.** Their comments cite "Table 7" for
hyperparameters, "Table 8" for the best checkpoint, "Table 9" for test performance, "Table 10"
for the ablation and "Table 11" for the baseline comparison — that is the `(1)` numbering.

Everything numeric is byte-identical between the two files; only §3.1 and the table numbers
differ. So decide which file is your master, and if you continue in `EduFusionFormers.docx`
translate every reviewer table reference down by one:

| Reviewer says | In `EduFusionFormers.docx` |
|---|---|
| Table 5 (Architecture Summary) | Table 4 |
| Table 6 (Parameter Distribution) | Table 5 |
| Table 7 (Hyperparameter Configuration) | Table 6 |
| Table 8 (Best Checkpoint) | Table 7 |
| Table 9 (Final Test Performance) | Table 8 |
| Table 10 (Ablation) | Table 9 |
| Table 11 (Baseline Comparison) | Table 10 |

Table numbers below use the **reviewers' numbering**. Also add a response-letter note telling
the editor you renumbered, or reviewers will think you ignored them.

---

## Reviewer 1 — results are entirely synthetic

**What they are saying.** Not "your method is wrong" but "your evidence does not support a
claim about the world". Training and testing on data you generated means the model is being
scored on its ability to invert your own generator. Since the label is a deterministic
function of the latent state (§3.1), high accuracy demonstrates that the modalities carry
recoverable information about that state — a property you built in. It says nothing about real
administrators.

**Your response is the right strategy.** Reframing as a methods-and-benchmark contribution is
the only route that does not require new data. Two things to make sure of:

1. **The title and abstract must carry the reframing.** A reader who only sees the abstract
   must not come away thinking this was validated on institutional data. Consider a title that
   signals it, e.g. *"... : A Framework and Simulation Benchmark for Multimodal Educational
   Administrative Behavior Analysis"*.
2. **Purge outcome language throughout.** Search for and rewrite every instance of
   *"proves"*, *"confirms the efficiency"*, *"demonstrates the effectiveness"*. §4.3 and §4.4
   are dense with these. Replace with *"under the stated simulation assumptions, the model
   recovers the simulated labelling structure with ..."*.

---

## Reviewer 2

### 1. No explicit research questions

**What they are saying.** The introduction motivates a *gap* ("nobody has applied multimodal
transformers to administrative behaviour") but never states a *hypothesis*. A gap is not a
contribution: you have to say what property you are testing and what result would count as
evidence for or against it.

**Edit.** Add a short block at the end of §1, immediately before the contributions list, and
then answer each RQ explicitly by name in §4 and §5.

> **RQ1.** Does hierarchical multimodal fusion outperform late-fusion and single-mechanism
> fusion baselines on the benchmark? *H1: EduFusionFormer achieves higher macro-F1 than
> BERT+MLP, MulT, GMU and Perceiver fusion under an identical administrator-level split.*
>
> **RQ2.** Does each proposed component contribute measurably? *H2: cumulative addition of
> DMIL, hierarchical cross-modal attention and temporal memory each increase macro-F1.*
>
> **RQ3.** Which feature groups drive predictions? *H3: SHAP attributions concentrate on
> survey-derived features rather than operational counts.*

**Risk.** RQ2 as stated is **not supported by your own Table 10** — see item 7. Either weaken
H2 to "non-decreasing" or report the component results as inconclusive.

---

### 2. Epoch budget contradicts itself

**What they are saying.** Three mutually exclusive statements: Table 7 says max 30 epochs,
§4.4 and Table 8 say the best checkpoint was epoch 34, Figure 3 plots past epoch 40. At least
one is false, and a reviewer cannot tell which, so none of the training detail is trustworthy.

**Edit.** Your response (max 50, patience 10, best at 34, stopped at 44) is internally
consistent — 34 + 10 = 44 checks out. Apply it in **four** places:

| Location | Change |
|---|---|
| Table 7, *Maximum Epochs* | `30` → `50` |
| Table 7, *Early Stopping Patience* | `5` → `10` |
| §3.5 body text | "patience of 5 epochs" → "patience of 10 epochs" |
| Figure 3 caption | state the axis runs to epoch 44 and that training stopped there |

Also add one sentence to §3.5: *"Training ran for 44 of a maximum 50 epochs; early stopping
triggered after 10 epochs without validation macro-F1 improvement, and the epoch-34 checkpoint
was retained."* That single sentence resolves all three contradictions at once.

> **This changes the code.** The training notebook was built against Table 7 as published
> (30 epochs, patience 5). It needs `MAX_EPOCHS = 50` and
> `EARLY_STOPPING_PATIENCE = 10`, and the Table 7 compliance assertions updated to match.

---

### 3. Performance values disagree between tables

**What they are saying.** Table 9 says accuracy 0.9030 / macro-F1 0.9057. Table 10 says 0.8933
/ 0.9017 for what is described as the same full model. The conclusion then mixes the two. A
reader cannot tell what the model actually scores.

**Edit.** Add a legend to Table 10 and a sentence in §4.6 stating that ablation configurations
were trained in a separate run, and fix the conclusion to quote the Table 9 figures.

**Risk — your response creates a new problem.** You say the ablation run used *a different
random seed*, but:

- Reviewer **7** asks for "identical data partitions and **comparable training conditions**"
  across ablation configurations. A different seed from the main run is arguably not
  comparable.
- Reviewer **9** asks for multi-seed robustness, and your response there says "all results
  come from single runs". Using the 0.40-point gap as "an estimate of run-to-run variation" is
  an estimate from **n = 2**, which is not an uncertainty estimate and a methods reviewer will
  say so.

**Stronger options, in order of preference:**

1. **Re-run the full model inside the ablation run** so Table 10's last row *is* Table 9's
   row. One extra training run and the contradiction disappears entirely.
2. Keep both, but report the gap as *"a single additional run differed by 0.40 points in
   macro-F1, which we report as an observation rather than an uncertainty estimate; see
   Limitations"*. Do not call it run-to-run variation.

---

### 4. Wrong baseline identified as strongest

**What they are saying.** You claim Perceiver Fusion is the strongest baseline and that you
beat it by ~5.9 points. But Table 11 shows BERT+MLP at macro-F1 0.8529, higher than
Perceiver's 0.8472. So BERT+MLP is the one to beat, and the real margin is
0.9057 − 0.8529 = **5.28 points**, not 5.85. They are checking your arithmetic against your
own table, and it failed.

**Edit.** Your correction is right. Change in §4.7: delete the sentence calling Perceiver "the
strongest performing model among reported baseline models", and replace the `5.85` claim with
`5.28`. The `5.11` accuracy figure (0.9030 − 0.8519) is already correct — keep it. Then check
the abstract for any surviving margin claim.

**Note.** `5.85` occurs once and `5.11` once in the current text, so this is a small edit — but
re-derive every improvement claim from the table rather than trusting the prose.

---

### 5. The modality-weight validation experiment is missing

**What they are saying.** This is the most serious methodological point. The paper's pitch is
that DMIL produces *meaningful* modality importances — and the simulation uniquely allows you
to verify that, because you know the ground-truth informativeness of each modality. The
reviewer looked for that experiment and it is not there. They also warn against the standard
fallacy: a learned softmax weight is not a measure of predictive contribution.

**Edit.** Removing the claim is the honest move given you did not run it. But do it
**everywhere** — this claim is load-bearing and appears in several places:

- Abstract: "the DMIL module learns sample-wise modality importance for adaptive emphasis" →
  keep as a mechanism description, but delete any suggestion the weights were validated or
  interpreted.
- Contribution bullet 2: reword from importance-estimation-as-insight to
  mechanism-plus-ablation-evidence.
- §3.4 / §3.5 closing sentences: "the outputs of attention and modality weights allow for
  analysis of the effect of the administrative, survey, text, temporal and consistency data" →
  this promises analysis you do not deliver. Change to *"are exposed by the implementation but
  are not analysed in this study"*.
- Add to Limitations: the operationalisation they asked for, as future work — vary
  modality-specific noise and missingness in the generator, then correlate learned weights
  against known informativeness.

**Risk.** Items 5 and 12 together remove *all* interpretability evidence. Combined with item 7
weakening the ablation, the paper's novelty rests almost entirely on RQ1 (baseline
comparison). Re-read your contributions list afterwards and make sure every bullet still
corresponds to evidence you actually present.

---

### 6. Does the temporal module really use longitudinal information?

**What they are saying.** They read Eq. 10 and Algorithm 1 closely and noticed that the
Temporal Behavioral Memory Encoder receives the *current* administrator-month embedding plus a
month index — not a sequence of that administrator's previous months. If so, it cannot learn
individual behavioural trajectories; it learns a month-of-year offset and a shared memory
bank. That is a much weaker claim than "models behaviour evolution".

**They are correct.** This is confirmed by the implementation: the module takes a
`(B, 256)` embedding and a scalar month index, adds a learned month embedding, and
cross-attends to 16 learnable memory tokens. No previous observation of the same administrator
is visible, and there is no temporal mask. Nothing in the architecture could capture a
trajectory.

**Edit.** Your response is the right one. Add to §3.4:

> The encoder receives the hierarchical cross-modal embedding for a single administrator-month
> (B × 256) together with that observation's month index. The index selects one of 12 learnable
> month embeddings, which is added to the embedding before it cross-attends over 16 learnable
> memory tokens with 8 heads. Observations from previous months of the same administrator are
> not supplied, and no temporal masking is applied. The module therefore captures
> month-of-year regularities and shared behavioural prototypes rather than
> administrator-specific trajectories.

Then remove trajectory language from the abstract ("encodes temporal information related to
monthly administrative behaviours" is acceptable; "learn temporal representations reflecting
behavior evolution" in §1 is not), contribution bullet 4, and §4.4.

**Worth considering.** If you want the longitudinal claim back, the fix is architectural, not
editorial: batch by administrator and feed all 12 months as a sequence with causal masking.
That is a genuinely stronger paper, but it is new experiments.

---

### 7. Ablation design is unclear and the conclusions do not follow

**What they are saying.** Two problems. First, §4.6 says components were added
"individually" but then reads the table as a cumulative progression — those are different
experiments with different interpretations. Second, and worse, your stated conclusions
contradict your own numbers:

- You write that DMIL "does not help improve performance" — but there is no BERT+MLP row in
  Table 10, so there is nothing to compare DMIL against. The claim is unsupported.
- You write that "the full model does not outperform the BERT+MLP baseline in all metrics" —
  but full-model macro-F1 (0.9017) is far above BERT+MLP (0.8529 in Table 11). The claim is
  simply false.

**Edit.**
1. State the design explicitly as cumulative and list the components in each row:

   | Configuration | Components |
   |---|---|
   | BERT + MLP (reference) | text encoder + MLP head |
   | + DMIL | above + dynamic modality importance + confidence gating |
   | + Hierarchical | above + hierarchical cross-modal transformer |
   | + Temporal | above + temporal behavioral memory |
   | Full EduFusionFormer | above + consistency learning + adaptive fusion |

2. **Add the BERT+MLP row to Table 10.** The reviewer explicitly asks for it and it is
   currently absent.
3. Delete both unsupported sentences and replace with claims that match the values.

**The real problem this exposes.** Read your own Table 10 as a cumulative sequence:

| Step | Macro-F1 | Gain |
|---|---|---|
| + DMIL | 0.8807 | — |
| + Hierarchical | 0.8878 | +0.0071 |
| + Temporal | 0.9016 | +0.0138 |
| Full (+ consistency + adaptive fusion) | 0.9017 | **+0.0001** |

Cross-Modal Consistency Learning and the Adaptive Fusion Aggregator — two of your five named
contributions — together add **one ten-thousandth** of macro-F1. From a single run that is
indistinguishable from noise. Be straightforward about it:

> The final configuration adds 0.0001 macro-F1 over the temporal configuration. On a single
> run this difference is not distinguishable from run-to-run variation, and we therefore do not
> claim a measurable benefit for the consistency and adaptive-fusion components on this
> benchmark.

Conceding this is far safer than letting a reviewer compute it. If you want these components
to survive as contributions, they need multi-seed evidence (item 9).

---

### 8. Baselines are under-described and under-powered

**What they are saying.** Two separate asks. (a) *Documentation*: for each baseline, what
inputs did it see, how was it adapted, how many parameters, what tuning, what budget?
Without this, a 5-point margin could just be a parameter-count or tuning asymmetry — your
model has 24.3M trainable parameters and a BERT+MLP baseline has far fewer. (b) *Missing
baselines*: a survey-only model and a plain tabular classifier. This is the sharpest version
of the question — if gradient boosting on the 43 survey features alone scores 0.89, the entire
multimodal architecture is unjustified.

**Edit.** Documenting the baselines (your response) is necessary but handles only (a).

**Risk.** Deferring (b) to limitations is the weakest response in your set. The survey-only
baseline is *cheap* — minutes on the existing features — and the SHAP analysis in §4.8 already
says survey features dominate, which is exactly why the reviewer suspects it. If you defer
this and the reviewer runs the mental arithmetic, the paper is in trouble.

**Strongly recommended:** actually run them. Add to Table 11:

- Survey-only (43 features) → gradient boosting, and an MLP
- Admin-only (22 features) → gradient boosting
- All tabular (71 features, no text) → gradient boosting
- Text-only → BERT + linear head

These share the existing administrator-level split, cost very little, and directly answer
"does the complexity pay for itself". Even if a tabular model comes close, reporting it
honestly is survivable; being caught not having checked is not.

---

### 9. No robustness across seeds or datasets

**What they are saying.** Every number is from one run, so you cannot distinguish a real
5.28-point margin from sampling noise. The clustering point is subtle and correct: your 18,000
records are 1,500 independent administrators × 12 correlated months, so the effective sample
size is closer to 1,500 than 18,000, and ordinary confidence intervals over records
understate uncertainty. They also want generator sensitivity analysis.

**Edit.** Declaring single runs and deferring the rest is defensible for a methods paper, but
the bare minimum is to state it *at every results table*, not once.

**Recommended minimum.** Three seeds for the full model and the strongest baseline, reported
as mean ± SD. This is the cheapest purchase of credibility available, and it also repairs
items 3 and 7 — you would then have a real variability estimate instead of an n=2 gap, and you
could say whether the +0.0001 full-vs-temporal difference is noise. For clustering, bootstrap
over `admin_id` rather than over rows.

---

### 10. Simulation assumptions are unjustified

**What they are saying.** Class proportions, five latent profiles, trait correlations, and the
effectiveness/overload formulas are all your design choices. Where do they come from? And a
pointed conceptual objection: you force "Highly Effective" and "Overloaded" to be mutually
exclusive, when in reality the most effective administrators are often the most overloaded.
So the label space may encode a false assumption.

**Edit.** Your new Protocol paragraph is the right move. Make sure it does three things:

1. Labels each parameter as *design choice*, *literature-informed* (with citation), or
   *arbitrary-but-documented*. Do not over-claim empirical grounding you do not have.
2. Addresses the exclusivity objection directly — the label rule ranks a composite, so an
   administrator high on both effectiveness and overload is forced into one bucket. Note that a
   multi-label or two-dimensional formulation would be more faithful, and list it as future
   work.
3. States plainly that performance measures recovery of the simulated structure and carries no
   validity for personnel assessment.

---

### 11. Insufficient detail for reproducibility

**What they are saying.** A list of specifics they could not find, plus one sharp
architectural question: Table 5 advertises auxiliary heads and a consistency module, but §3.5's
loss has only classification cross-entropy. So are those components supervised or not? If not,
calling the head "Multi-Task" is misleading.

**Edit.** Your response covers the list. Verified values to state:

| Item | Value |
|---|---|
| Split (administrators) | 1,050 / 225 / 225 |
| Split (records) | 12,600 / 2,700 / 2,700 |
| BERT frozen | embeddings + encoder layers 0–9 |
| BERT fine-tuned | encoder layers 10–11 + pooler |
| Text aggregation | documents concatenated per `admin_id` × `month_step` |
| Truncation | 256 word-pieces, `truncation=True`, pad to max length |
| Metadata | 4 categoricals ordinal-encoded + age + years_experience = 6 features |
| Seed | state it |

The frozen/fine-tuned split is **independently confirmed by Table 6**: bert-base-uncased has
109,482,240 parameters, and 2 encoder layers (2 × 7,087,872) plus the pooler (590,592) is
exactly 14,766,336 trainable, leaving exactly 94,715,904 frozen. Your Table 6 is self-
consistent — say so explicitly, it is good evidence of reproducibility.

**On the auxiliary heads.** State clearly that the engagement, workload and leadership heads
and the consistency module receive **no dedicated loss**, so the objective is single-task.
Then **rename the module** — "Multi-Task Classification Head" claims something you do not do.
"Classification Head with Auxiliary Outputs" is accurate. Also remove "Multi-Task" from the
abstract. Alternatively, drop the unsupervised auxiliary heads entirely; they consume
parameters and contribute nothing, and removing them is cleaner than defending them.

---

### 12. SHAP procedure undocumented

**What they are saying.** "We used SHAP" is not a method. Which explainer? What background
distribution? Which output — one class's logit, or the predicted class? How were the text and
month inputs handled while tabular features were perturbed? And the deep problem: SHAP on
*correlated* features is unreliable, and your 43 survey features include mean/median/std
triplets per item that are nearly collinear by construction. KernelExplainer assumes feature
independence, so attributions will be split arbitrarily among correlated siblings — which
directly undermines §4.8's ranking of `responds_quickly_mean` above
`manages_workload_mean`.

**Edit.** Document all of it in §4.8: explainer, background sample size and source, number of
explained instances, `nsamples`, the explained output, and — critically — how text and
`month_step` were held while tabular features were perturbed. If text was frozen at a reference
value, say so and note that the attributions are therefore conditional on that text.

Add a sentence acknowledging collinearity:

> Because the survey block contains mean, median and standard-deviation statistics for each
> item, many inputs are strongly correlated. KernelExplainer assumes feature independence, so
> attribution mass may be distributed arbitrarily among correlated features. Individual
> rankings within the survey block should therefore be read as group-level rather than
> feature-level evidence.

Also fix the typo in §4.8: `respondents_quickly_mean` → `responds_quickly_mean`.

---

### 13. Discussion/conclusion overclaim

**What they are saying.** Same as Reviewer 1, aimed at §5. The conclusion currently asserts the
framework "proved to be an efficient and interpretable approach" — but interpretability
analysis was removed (items 5, 12) and efficiency was never measured.

**Edit.** Rewrite §5 so every sentence is traceable to a result. Specifically:
- "proved to be an efficient and interpretable approach" → cannot survive; both halves are
  unsupported now.
- "An ablation study also confirmed the contribution of the proposed multimodal and temporal
  approaches" → must be softened per item 7.
- Fix the metric: §5 currently reports macro-F1 90.17% (a Table 10 number) alongside Table 9
  numbers. Use 0.9057 throughout.
- Add the personnel-decision risk paragraph: these categories are simulated constructs; using
  such a classifier for real evaluation, promotion or workload decisions would require
  construct validation, fairness auditing across protected attributes, and human oversight.

---

### 14. Citation–reference misalignment — **the reviewer is wrong, do not capitulate**

**What they are saying.** They believe in-text descriptions are offset by one from the
reference list, and cite reference 4 as the example: described as a retracted Transformer
study, but allegedly listing "a different architecture".

**I checked all 17 in-text citations against the reference list. Every one matches.**
Reference [4] is:

> Gong, Y., Wang, F., Zhang, Y. and Geng, J., 2025. **RETRACTED:** TGEL-transformer: Fusing
> educational theories with deep learning for interpretable student performance prediction.
> *PLoS ONE*, 20(6), p.e0327481.

The §1 text describes [4] as "a theory-guided Transformer solution for interpretable student
performance prediction... the supplied reference clearly indicates that the work was
retracted". That is an exact match — right authors, right architecture, right topic, and the
retraction is in the reference title.

Spot-checks of the rest: [1] Zheng AMTBA ✓, [2] Patidar Edulyze ✓, [3] Moon formative
assessment ✓, [5] Sridharan MS-DDTCN + Chameleon Swarm ✓, [6] Ouhaichi interview study ✓,
[7] Bird educational literature ✓, [8] Sun quantum learning paths ✓, [9] Wang adaptive
pedagogy ✓, [10] Boufaida TSA-GRU ✓, [11] Lin RT-DETR ✓, [12] Zhou smart classrooms ✓,
[13] Kui TSSeer ✓, [14] Lee & Zhai GPT-4V ✓, [15] Davalos reading assessment ✓, [16] Korban
classroom instruction ✓, [17] Guo teaching quality ✓. No off-by-one anywhere.

**Your drafted response concedes an error that does not exist, and then deletes a correct
sentence.** Worse, the response says "the sentence on a retracted study, which had no matching
reference, was removed" — but it *does* have a matching reference, and noting that a related
work was retracted is exactly the scholarly diligence reviewers should want.

**Recommended response instead:**

> We re-verified every in-text citation against the reference list and found no misalignment.
> Regarding reference [4], the listed source is Gong et al. (2025), "RETRACTED:
> TGEL-transformer: Fusing educational theories with deep learning for interpretable student
> performance prediction", PLoS ONE 20(6), e0327481 — a theory-guided Transformer for
> interpretable student performance prediction that is indeed marked as retracted, matching the
> in-text description. We have retained the note on its retraction status so that readers do not
> treat it as supporting evidence. We would be glad to correct any specific pairing the
> reviewer can point us to; it is possible a different manuscript version was consulted.

That last clause is the diplomatic exit. Note the repository holds two manuscript versions
with different table numbering, so a version mix-up is plausible — but the citations are
identical in both files, so it does not explain this comment.

---

## Additional inconsistencies the reviewers have not yet flagged

Reviewer 2 is clearly checking numbers against tables. These will be found on the next round,
so fix them now.

### A. Macro recall is reported three different ways

| Location | Value |
|---|---|
| §4.3 body text | 0.9129 |
| §4.7 body text | 0.9129 |
| **Table 11, proposed row** | **0.9192** |
| **Table 10, full model row** | **0.9192** |
| §4.6 body text ("Recall of 0.9172") | 0.9172 |
| §5 conclusion | 91.29% |

Three distinct values for one quantity, and §4.6's prose (0.9172) contradicts the table it
describes (0.9192). Pick the true value and propagate.

### B. Table 10's prose does not match Table 10

§4.6 states the full model scores "Recall of 0.9172" while the table says 0.9192. Same
sentence, same table, different number.

### C. The abstract uses future/conditional tense for completed work

"the ablation analysis **can be conducted** to evaluate the performance of its components" —
reads as though the ablation was never run. Change to past tense, or the reviewer will wonder
what else was not done.

### D. Warm-up steps change with the epoch budget

Warm-up is 10% of *total training steps*, and total steps depend on max epochs. Moving 30 → 50
changes the absolute warm-up step count. If you report it numerically anywhere, update it.

---

## Consistency sweep — do this last

Search the manuscript for each string and verify every occurrence:

| Find | Action |
|---|---|
| `5.85` | → `5.28` (item 4) |
| `0.9172` | → the corrected recall (item A) |
| `0.9129` / `91.29` | reconcile with Table 11's `0.9192` (item A) |
| `90.17` / `0.9017` | should be `0.9057` in abstract/conclusion (items 3, 13) |
| `Maximum Epochs` | → `50` (item 2) |
| `patience of 5` | → `patience of 10` (item 2) |
| `epoch 34` / `34th epoch` | keep, but add that max was 50 and stopping was at 44 |
| `Perceiver` | remove "strongest baseline" framing (item 4) |
| `Multi-Task` | rename head; remove from abstract (item 11) |
| `respondents_quickly_mean` | → `responds_quickly_mean` (item 12) |
| `proves` / `confirms` / `demonstrates the effectiveness` | soften (Reviewer 1, item 13) |
| `Table 4`–`Table 11` | renumber if editing `EduFusionFormers.docx` (§0) |

---

## Priority ordering

**Must fix — the paper fails without these**
- Item 2 (epoch contradiction), item 3 (conflicting metrics), item 4 (wrong baseline),
  item 7 (false ablation claims + missing BERT+MLP row), item A (recall inconsistency),
  Reviewer 1 + item 13 (synthetic-data framing)

**Should fix — cheap, and materially strengthens the paper**
- Item 8 survey-only and tabular baselines (hours of compute, removes the biggest
  "is this complexity justified?" objection)
- Item 9 three seeds (also repairs items 3 and 7)
- Item 11 rename the multi-task head
- Item 14 rebut rather than concede

**Honest deferral is acceptable**
- Item 5 (modality-weight validation), item 9 generator sensitivity, item 12 attention-map
  analysis, item 6 true sequence modelling — provided the claims come out of the abstract and
  contributions, not just into a limitations list

---

## Impact on the code in this repository

1. **Training notebook hyperparameters change.** `MAX_EPOCHS` 30 → 50 and
   `EARLY_STOPPING_PATIENCE` 5 → 10, plus the Table 7 compliance assertions.
2. **Item 8 needs new code** — survey-only, admin-only, tabular and text-only baselines on the
   existing administrator-level split.
3. **Item 9 needs a seed loop** around training, with bootstrap over `admin_id` rather than
   over records.
4. **Item 12 needs the SHAP configuration pinned down** — explainer, background size,
   `nsamples`, explained output, and the treatment of text and month inputs — because those
   choices must be reported in §4.8.
5. Items 6 and 11 need **no code change**: the implementation already matches the corrected
   description (current-observation-only temporal module, no auxiliary losses).
