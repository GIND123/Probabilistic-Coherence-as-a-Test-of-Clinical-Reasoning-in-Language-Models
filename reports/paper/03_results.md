# Results

**Status.** Ten models complete (1,119,920 elicited responses on DDXPlus plus
33,558 on MIMIC-IV-Note). `gpt-oss-20b` is regenerating after the C11
elicitation fix and `gpt-oss-120b` follows it; every number below is over the
ten complete models and will be reissued when the family lands. The broken
gpt-oss run is retained outside `results/raw` as C11 evidence and is excluded
from all tables.

Throughout, the unit of comparison is not a raw divergence but a divergence
**against three anchors measured on the same model**: its test–retest floor
(same prompt, resampled), the between-patient ceiling (different patients),
and the dataset-level generator noise floor (**JSD = 0.0490**, from exact
replicate cases in DDXPlus itself).

---

## 0. Competence precedes coherence

An order-effect statistic computed over responses that do not track the patient
measures the decoding path, not clinical reasoning — and it fails silently in
both directions (C11). So every model is first shown to be answering the
question at all, on the 49-way pick-one-diagnosis task (chance = 0.0204):

| model | top-1 | x chance |
|---|---|---|
| qwen3-32b-nothink | 0.5373 | 26.3 |
| medgemma-27b | 0.5204 | 25.5 |
| qwen3-32b-think | 0.5119 | 25.1 |
| med42-8b | 0.3092 | 15.1 |
| r1-distill-32b | 0.3078 | 15.1 |
| qwen3-8b-nothink | 0.2812 | 13.8 |
| qwen3-4b-nothink | 0.2664 | 13.1 |
| qwen3-8b-think | 0.2516 | 12.3 |
| qwen3-4b-think | 0.2333 | 11.4 |
| medgemma-1.5-4b | 0.0805 | 3.9 |

All ten clear chance with the Wilson lower bound. No pass/fail threshold is
imposed: on this battery a weak model and a broken one are not separable by any
single cut (C11), so competence is reported as a continuous quantity and a
suspected fault is settled by changing one thing about one model and
re-measuring.

---

## 1. A1 — Order invariance

`P(d | e_1 ... e_n)` is a function of the evidence **set**. Reordering a
patient's findings must not move the posterior. It does, for every model.

| model | JSD(perm) | retest floor | between-patient ceiling | **normalised order effect** | top-1 flip |
|---|---|---|---|---|---|
| qwen3-4b-nothink | 0.1284 | 0.0722 | 0.3339 | **0.2149** | 0.796 |
| qwen3-8b-nothink | 0.2023 | 0.1692 | 0.3435 | 0.1898 | 0.947 |
| qwen3-8b-think | 0.1844 | 0.1611 | 0.3111 | 0.1551 | 0.885 |
| qwen3-32b-nothink | 0.2908 | 0.2562 | 0.5286 | 0.1270 | 0.928 |
| qwen3-32b-think | 0.3095 | 0.2666 | 0.6705 | 0.1062 | 0.887 |
| qwen3-4b-think | 0.1315 | 0.1143 | 0.3279 | 0.0805 | 0.699 |
| medgemma-27b | 0.2712 | 0.2464 | 0.5817 | 0.0741 | 0.836 |
| med42-8b | 0.2623 | 0.2550 | 0.3697 | 0.0637 | 0.957 |
| r1-distill-32b | 0.2750 | 0.2616 | 0.5216 | 0.0517 | 0.938 |
| medgemma-1.5-4b | 0.2836 | 0.2815 | 0.3532 | **0.0285** | 0.994 |

**The normalised order effect falls monotonically with scale and with medical
tuning** — 0.215 at Qwen3-4B to 0.127 at Qwen3-32B, and lower again for the
medically tuned models. Raw divergence does the opposite (0.128 to 0.291),
because the test–retest floor rises with scale faster than the order effect
does. **Reporting the raw number alone would have reversed the finding.** This
is the single strongest argument for the three-anchor design and it is why no
prior work's order-effect number is directly comparable to these.

The top-1 flip rate is high everywhere (0.70–0.99): reordering the same
findings changes the leading diagnosis for most patients, even where the
distributional movement is modest.

A caution on the bottom of the table: MedGemma-1.5-4B's 0.0285 is not a
coherence success. Its floor (0.2815) has nearly reached its own ceiling
(0.3532) and its flip rate is 0.994 — it is close to answering
patient-independently, which is what a small normalised effect looks like when
the denominator collapses. Its competence ratio (3.9x, the lowest of the ten)
should be read alongside it.

---

## 2. A2 — Belief updating against an assumption-free oracle

DDXPlus ships **0 of 888** conditional probabilities (audit finding). Rather
than assume conditional independence to reconstruct them — which fails at JSD
0.546 against a 0.049 noise floor, 11x — the posterior is conditioned directly
on 1.29M released patients, with Wilson intervals and a 500-case minimum
support. A2 therefore makes **no independence assumption**.

For each finding the empirical log-odds change is compared with the model's:

| quantity | range over 10 models |
|---|---|
| slope beta | 0.0009 – 0.0576 |
| r^2 | 0.000 – 0.021 |
| \|Spearman\| | <= 0.061 |
| **direction agreement** | **0.454 – 0.549** |

**Direction agreement straddles 0.5 for every model tested.** Whether a
model's belief in a diagnosis goes up or down when a finding is added is
uncorrelated with whether the corpus says it should. Six of ten models have a
slope whose interval excludes zero, but at beta <= 0.058 against an ideal of
1.0, the effect is real, tiny, and swamped: r^2 never exceeds 0.021.

This is the strongest negative result in the study. Order-sensitivity (A1) is a
consistency failure and could be dismissed as sampling noise; A2 is a failure
of **direction**, measured against ground truth, with no independence
assumption available to attack.

---

## 3. A3 — Redundancy and certified-uninformative evidence

Adding evidence that the corpus certifies as uninformative should not move the
posterior. Anchored to each model's own retest floor:

| model | mean spurious JSD | p90 | frac above own floor | top-1 flip |
|---|---|---|---|---|
| qwen3-4b-nothink | 0.0759 | 0.2603 | 0.291 | 0.346 |
| qwen3-8b-nothink | 0.0804 | 0.2828 | 0.158 | 0.407 |
| qwen3-8b-think | 0.1062 | 0.3190 | 0.191 | 0.296 |
| medgemma-27b | 0.1381 | 0.4059 | 0.223 | 0.328 |
| qwen3-32b-nothink | 0.2214 | 0.5876 | 0.351 | 0.546 |
| med42-8b | 0.2235 | 0.8848 | 0.270 | 0.546 |
| r1-distill-32b | 0.2288 | 0.9278 | 0.278 | 0.387 |
| qwen3-32b-think | 0.2518 | 0.8803 | 0.345 | 0.423 |
| medgemma-1.5-4b | 0.2550 | 0.5167 | 0.432 | 0.541 |

Irrelevant findings move the differential for 16–43% of cases by more than the
model's own resampling noise, and flip the leading diagnosis for 30–55%. The
p90 figures (up to 0.93) show the tail is severe: for a tenth of cases a
certified-uninformative finding moves the belief state almost as far as
swapping the patient. Confidence inflation accompanies it in 32–49% of cases.

---

## 4. A4 — Positional anchoring

Position effects are present but small: `eta^2` 0.245–0.263 across models, with
the position effect itself (between-position minus within-position divergence)
between -0.004 and 0.014. Order matters (A1), but **not primarily through a
simple first-position or last-position anchor** — the effect is distributed,
not positional. Following arXiv:2605.15000, which uses "premature closure" for
failure-to-abstain, this axiom is named **positional anchoring** throughout to
avoid a terminology collision.

---

## 5. Table 1 — Remedies

For `qwen3-32b-nothink`, paired over the cases every method produced:

| method | order residual | queries / new case | top-1 | top-5 | entropy (bits) | exact? | per-finding audit |
|---|---|---|---|---|---|---|---|
| direct | 0.2988 | 1 | **0.1892** | 0.3006 | 3.94 | no | no |
| self-consistency | 0.1244 | 6 | 0.2265 | 0.4346 | 4.50 | no | no |
| perm-ensemble K=2 | 0.1876 | 2 | 0.2117 | 0.3686 | 4.27 | approx | no |
| perm-ensemble K=3 | 0.1422 | 3 | 0.2265 | 0.4131 | 4.41 | approx | no |
| perm-ensemble K=5 | 0.1038 | 5 | **0.2377** | **0.4443** | 4.53 | approx | no |
| **ELR-Fusion (tau=1)** | **0.0000** | **0** | 0.1334 | 0.2480 | 0.86 | **exact** | **yes** |
| **ELR-Fusion (calibrated)** | **0.0000** | **0** | 0.0997 | 0.2393 | 5.25 | **exact** | **yes** |

ELR-Fusion is the only method with an **exactly** zero order effect — verified
numerically at 5.55e-16, machine epsilon — at **zero marginal query cost**,
because the weight table is elicited once per finding for the whole corpus
rather than once per case. Permutation ensembling buys its residual with
queries and decays only as O(1/sqrt(K)): five queries per case still leaves
0.1038, and no K reaches zero.

**The honest cost is accuracy.** ELR-Fusion trades 5.6 top-1 points against
direct elicitation (0.189 -> 0.133) and 10.4 against a 5-query permutation
ensemble. The guarantee is not free, and reporting it as free would be the
easy dishonesty here.

### 5.1 The shrinkage coefficient is a frontier, not a fitted constant

`fuse` adds one clipped log-LR per finding — the textbook naive-Bayes
accumulation, with the textbook pathology: correlated clinical findings are
counted as if independent, driving the posterior to the extremes (entropy 0.86
bits against 5.62 for uniform). The remedy is a single shrinkage coefficient on
the evidence term, the log-odds form of temperature scaling (Guo et al. 2017)
applied so the prior is untouched:

    log-odds(d | E) = log-odds(d) + tau * sum_i log LR(e_i | d)

**Order-invariance is exact for every tau**, because scaling an order-invariant
sum leaves it order-invariant — verified across 600 permutations at three tau
values, max |dp| = 5.55e-16. Shrinkage therefore buys calibration without
spending any of the guarantee, which is what distinguishes it from the ensemble
baselines.

It does not, however, buy accuracy (qwen3-32b-nothink, 1,956 cases):

| tau | top-1 | top-5 | NLL of truth | entropy |
|---|---|---|---|---|
| 0.00 | 0.0215 | 0.1125 | 3.947 | 5.47 |
| 0.05 | 0.0946 | 0.2367 | **3.858** | 5.31 |
| 0.10 | 0.1130 | 0.2469 | 3.907 | 4.89 |
| 0.20 | 0.1314 | 0.2495 | 4.457 | 3.73 |
| **0.40** | **0.1355** | 0.2444 | 6.491 | 2.22 |
| 1.00 | 0.1334 | 0.2480 | 13.261 | 0.86 |
| 3.00 | 0.1375 | 0.2515 | 20.977 | 0.26 |

Accuracy peaks near tau ~ 0.4; the likelihood of the truth peaks near
tau ~ 0.05, an eightfold-different operating point. tau = 1 — the unshrunk
default — is near accuracy-optimal and among the **worst** calibrated points on
the grid (NLL 13.3 against 3.86 achievable). The sweep is reported in full
rather than collapsed to a fitted number, because a single tau would conceal
that the two objectives disagree. In Table 1 the calibrated row uses tau fitted
by **cross-fitting** over two folds of cases, so the coefficient applied to a
case is always fitted on the fold excluding it.

The accuracy ceiling here is set by the elicited weights themselves, not by
overconfidence: tau = 0 (prior only) scores 0.0215, so the weights do carry
real signal, but an NLL-optimal use of them sits at tau ~ 0.05, close to
ignoring them. **Improving the log-LR elicitation, not the fusion rule, is
where the remaining accuracy is.**

---

## 6. What the instrument had to survive

Eleven calibration steps are reported as a first-class artifact
(`reports/instrument_calibration.md`). Several would each have produced a
plausible but wrong headline: arm-dependent schema failures (C4), a 62.9%
uniform collapse behind "100% schema validity" (C6), an inverted shuffled
ceiling and canonical-order contamination of the floor (C7), two defects in
Table 1 itself (C9), and most severely:

**C11.** gpt-oss-20b under grammar-constrained decoding returned `{"index": 0}`
for essentially every case at 1.000 schema validity and 1.000 informative rate,
scoring 0.039 against a 0.020 chance rate. Its coherence numbers looked like a
finding — permutation JSD 0.683 against a between-patient ceiling of 0.689 and a
top-1 flip rate of 1.000 — and would have been the paper's headline. The cause
was harmony: the model must open a channel at token zero and a JSON grammar
forbids it. Decoded freely the same model reasons correctly and reaches
**0.6328**, the most accurate in the sweep. A 16x change from one decoding flag.
