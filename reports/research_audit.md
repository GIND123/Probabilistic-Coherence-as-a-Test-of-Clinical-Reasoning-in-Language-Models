# Research audit — state of the work against a MAKE submission

**Date:** 2026-09-17
**Scope:** the full local tree, both Hugging Face repositories
(`GOVINDFROM/codx-clinical-coherence`, `GOVINDFROM/codx-elr-fusion`), and every
numeric claim in `README.md`, `reports/paper/`, `reports/instrument_calibration.md`
and `reports/data_audit_report.md`.
**Method:** every headline number re-derived from the CSVs of record; new
analyses run from `results/analysed/a1_per_case.parquet` and from the raw
per-item parquets pulled off the Hub.

---

## 0. Bottom line

The measurement is sound and the numbers in the README are accurate — I checked
28 of them against the tables and all 28 reproduce. The problem is not the
data. It is three things:

1. **The published artifact is wrong.** The Hub dataset repo still ships the
   C11-broken GPT-OSS run as a result row, under the name
   `_gptoss20b_harmony_broken`, and does not contain the fixed run. Anyone
   downloading it today gets the numbers the calibration report calls an
   artifact of xgrammar.
2. **The paper under-reports its own evidence.** Table 1 is one model out of
   eleven that were computed. Four columns that are in the CSVs and that a
   reviewer will ask about are not in the paper — one of them (A3's
   `excess_over_floor`) undercuts a claim as currently phrased.
3. **The strongest result in the dataset is not in the paper.** Every model
   names the correct diagnosis 2.6×–5.0× more often when asked to pick one
   than its own stated posterior does — and its forced-choice pick agrees with
   the argmax of its own distribution on the same item only 3–12% of the time.
   This is label-free, exactly paired, survives every confound I could test,
   and it reframes the whole paper.

Item 3 is worth more than everything else in this document. Sections 5 and 7
develop it.

---

## 1. Inventory — what exists

| artifact | state |
|---|---|
| Source tree | 45 modules, complete after the repair in §2.1 |
| Elicited responses | 1,118,457 over 11 models + 1,463 over the LoRA config |
| Result tables | 11 CSVs, all 11 models, all four axioms |
| Result figures | 6 result + 7 audit, PNG and PDF |
| Data audit | 19 findings, 2 CRITICAL, in `reports/data_audit_report.md` |
| Instrument calibration | C1–C11 documented |
| Paper drafts | related work, methodology, results |
| LoRA adapter | trained, published, `eval_loss` 0.191 — **never evaluated downstream** |

The DDXPlus audit is genuinely strong and is the part of this work most likely
to be cited by people who do not care about coherence: the 25.8% replicate
disagreement (a hard ceiling on every DDXPlus differential-matching metric),
the 0-of-888 empty conditional-probability slots, and the 3,861 cases appearing
in more than one split. Those three stand on their own.

---

## 2. Release-integrity defects — fix before any submission

### 2.1 Six source modules were never committed (repaired)

`.gitignore` line 5 was a bare `data/`, which git matches at any depth. It
silently excluded `coherence/data/` — `battery.py`, `ddxplus.py`,
`empirical_oracle.py`, `kb_reconstruct.py`, `mimic.py` — from every commit in
the repository's history. `git log -- coherence/data` returns nothing.

`empirical_oracle.py` is the module the C2 contribution rests on, and the
README links to it directly. The repo as published would not run.

**Repaired.** Files restored from the Hub copy of `code/`, and the pattern
anchored to `/data/` with a comment saying why. Verify with
`git check-ignore -v coherence/data/empirical_oracle.py` (now returns nothing)
and commit the six files.

### 2.2 The Hub dataset repo publishes the broken GPT-OSS run

Last synced 2026-09-15. Its `reports/tables/*.csv` carry:

| | published on the Hub | current local |
|---|---|---|
| model key | `_gptoss20b_harmony_broken` | `gpt-oss-20b` |
| permutation JSD | 0.6827 | 0.2866 |
| normalised order effect | **0.5890** | **0.0211** |
| top-1 flip rate | 1.0000 | 0.9586 |
| competence top-1 | 0.0427 | 0.6532 |
| `beats_chance` | **true** | true |

The published artifact contradicts the paper on its most dramatic single
number, and it contradicts it in the direction the calibration report
specifically warns against.

Note the last row. **`beats_chance` returned true for the broken model.** At
n = 34,885 the Wilson lower bound on 0.0427 clears 1/49 comfortably. The only
binary gate the instrument defines passed the model C11 exists to catch. That
belongs in C11 as a stated limitation — it makes the "no threshold, use a
within-model comparison" argument much stronger, because it shows the
alternative genuinely fails rather than merely being inelegant.

### 2.3 Raw data published for 5 of 11 models

| published in full | 11 files each | `med42-8b`, `qwen3-4b-nothink`, `qwen3-4b-think`, `qwen3-8b-nothink`, `qwen3-8b-think` |
|---|---|---|
| partial | 1 file | `qwen3-32b-nothink` (9 a1_posterior chunks only) |
| absent | 0 files | `qwen3-32b-think`, `medgemma-1.5-4b`, `medgemma-27b`, `gpt-oss-20b`, `r1-distill-32b` |

Six of the eleven models in the results tables have no raw data behind them in
the release. Two of those six — `medgemma-27b` and `gpt-oss-20b` — carry
headline claims.

### 2.4 Other release gaps

- No `result_*` figures on the Hub; only the audit figures are there.
- `reports/tables/calibration_c6.json` is local-only.
- `coherence/analysis/figures_results.py` is local-only, so the published code
  cannot regenerate the published figures.
- No `LICENSE` file in the repository. MDPI will ask; the dataset card already
  declares CC-BY-4.0 and the model card Apache-2.0, so the repo should say so
  too.
- Both Hub repos are private. Fine for now, but the submission needs a plan:
  DDXPlus is CC-BY-4.0 and public, so the dataset repo can go public in full;
  the MIMIC guard in `hf_sync.py` is correct and worth describing in the paper
  as a governance measure.

---

## 3. Reporting defects in the paper drafts

Every number I checked is right. These are omissions and framing, which is a
different and more fixable problem — but two of them are the kind a reviewer
converts into a rejection.

### 3.1 A3: the mean spurious effect is below the model's own floor (serious)

`a3_main.csv` has a column `excess_over_floor`. It is **negative for 10 of 11
models**, ranging −0.108 to −0.014. Only `qwen3-4b-nothink` is positive
(+0.0037).

README §4.3 does not report this column. It says irrelevant findings move the
posterior "by more than the model's own resampling noise for 15.8%–44.9% of
cases" — which is true, it is a per-case fraction — but a reviewer who opens
the CSV will see that *on average* a certified-uninformative finding moves
belief **less** than resampling the same prompt does.

The claim survives, but only as a tail claim, and it has to be phrased as one:
the distribution is heavy-tailed (p90 up to 0.93) while the mean sits under
the floor. Say that explicitly and report the column. Reporting the fraction
while omitting the mean, when both are in the same file, is the single most
attackable thing in the current draft.

Also drop `ratio_to_truth` (2,431–8,168) if it ever reaches the paper. Its
denominator, `certified_true_jsd`, is ~3×10⁻⁵ — the ratio is an artifact of
dividing by approximately zero.

### 3.2 A4: η² is reported for 3 of 11 models as if it were 11, and the stated conclusion is contradicted by the same table

Two separate problems.

**η² is a NaN bug (fixed).** `a4_main.csv` has `eta_squared` for only
`medgemma-1.5-4b` (0.2454), `med42-8b` (0.2634) and `gpt-oss-20b` (0.2694).
The other eight are NaN. The README reports "η² = 0.245–0.269 over 600 cases"
and `03_results.md` "0.245–0.263 across models" — both present a range over
three models as a range over the model set.

Cause: [`a4_anchoring.py:124`](coherence/axioms/a4_anchoring.py#L124) used
`t["eta_squared"].mean()`. A case whose log-odds are constant across replicates
has `ss_total == 0` and yields NaN, and polars propagates a single NaN through
the whole column mean. Changed to `np.nanmean`. η² is recoverable for all
eleven models on the next analysis run.

**The A4 conclusion is contradicted by its own table.** The draft says
position effects are "present but small" and that the effect is "distributed
rather than positional". But `beta_position` — the directional reading, slope
of the target log-odds on position coded −1/0/+1 — has a bootstrap CI
excluding zero for **7 of 11 models**:

| model | β | 95% CI | reading |
|---|---|---|---|
| r1-distill-32b | −0.126 | [−0.221, −0.037] | primacy |
| qwen3-32b-nothink | −0.074 | [−0.148, −0.005] | primacy |
| qwen3-8b-think | −0.059 | [−0.118, −0.002] | primacy |
| qwen3-8b-nothink | −0.047 | [−0.082, −0.014] | primacy |
| qwen3-4b-think | −0.046 | [−0.081, −0.012] | primacy |
| qwen3-4b-nothink | −0.041 | [−0.070, −0.011] | primacy |
| **medgemma-27b** | **+0.059** | [+0.001, +0.121] | **recency** |

Every general-purpose and reasoning-distilled model anchors on the **first**
position. MedGemma-27B, alone, reverses to recency. That is a finding, and the
paper currently denies it.

The reason it was missed is itself worth a paragraph: `position_effect`
(between-position JSD minus within-position JSD, −0.004 to +0.014) is a
**magnitude** measure, and JSD is symmetric, so a consistent directional shift
in log-odds cancels out of it. The divergence reading is blind to the effect
the regression detects. Reporting both, and explaining why they disagree, is a
genuine methodological contribution — it generalises to anyone measuring
position bias with a symmetric divergence.

### 3.3 A1: three models' order effect is not distinguishable from their own floor

`ci_lo`/`ci_hi` on `order_effect` cover zero for:

| model | order effect | 95% CI |
|---|---|---|
| medgemma-1.5-4b | 0.0020 | [−0.0039, +0.0083] |
| gpt-oss-20b | 0.0034 | [−0.0015, +0.0082] |
| med42-8b | 0.0073 | [−0.0015, +0.0160] |

The README cautions about these rows on entropy-collapse grounds, which is the
right instinct, but it never says the plainest thing: for these three models
the order effect is not statistically distinguishable from resampling noise.
Say it. It costs nothing — "eight of eleven models show an order effect
significantly above their own floor" is a stronger and more credible sentence
than an unqualified eleven.

### 3.4 Stale numbers across documents

| quantity | README | `03_results.md` | `instrument_calibration.md` | CSV of record |
|---|---|---|---|---|
| gpt-oss free-decode top-1 | 0.6532 ✓ | 0.6328 | 0.6328 | **0.6532** |
| C11 fold change | 16.6× ✓ | 16× | 16× | 16.58× |
| models complete | 11 ✓ | "ten, gpt-oss regenerating" | — | 11 |
| A3 frac above floor | 15.8–44.9% ✓ | 16–43% (9 rows) | — | 15.79–44.88% |
| A4 η² | 0.245–0.269 | 0.245–0.263 | — | 3 of 11 non-NaN |
| gpt-oss broken top-1 | 0.0394 | 0.039 | 0.0394 *and* 0.0427 | 0.0427 |

`03_results.md` predates the C11 fix. It should be regenerated, not patched.
The 0.0394-vs-0.0427 split inside C11 is explicable (one is before the
reasoning-parser attempt, one after) but reads as an inconsistency; pick one
and footnote the other.

### 3.5 Response-count accounting

README: "1,153,478 elicited responses (1,119,920 DDXPlus + 33,558 MIMIC)" over
"eleven models". The log sum over eleven models is **1,118,457**. The stated
1,119,920 is reached only by including the 1,463 responses from the twelfth
configuration, `qwen3-8b-elr-lora`. `01_related_work` already says "12 model
configurations", so the two documents disagree. One footnote fixes it.

### 3.6 τ ≈ 0.4 is not the accuracy peak

README §4.6 bolds τ = 0.40 (top-1 0.1355) as the accuracy optimum, but its own
table shows τ = 3.00 at 0.1375. Across the full sweep the accuracy-optimal τ
is ≥ 2.0 for 7 of 11 models. The honest statement is that accuracy *plateaus*
above τ ≈ 0.4 (0.1339–0.1375, a 0.4-point spread) while NLL degrades
monotonically and catastrophically over the same range — which is a better
version of the argument you are making anyway.

---

## 4. Promised vs delivered — the 17 ablations

`02_methodology.md` §9 says "every row is a planned experiment, and
collectively they are the contribution rather than a defensive appendix".
Seven of seventeen have a table.

| # | ablation | data elicited | analysed | note |
|---|---|---|---|---|
| 1 | K sweep | ✅ | ✅ | `a1_k_sweep.csv`; converged (§5.4) |
| 2 | three anchors | ✅ | ✅ | `a1_main.csv` |
| 3 | evidence-set size | ✅ | ❌ | **computed in this audit** (§5.3) |
| 4 | decisive position | ✅ | ✅ | `a4_main.csv`, but see §3.2 |
| 5 | ELR vs baselines | ✅ | ⚠️ | 11 models computed, 1 reported; CoT / oracle / prior-only baselines missing |
| 6 | independence correction | ✅ | ✅ | audit finding 5 |
| 7 | LR format ×3 | ✅ 483 items × 3 formats × 11 models | ❌ | analysis hard-codes `"numeric"` |
| 8 | thinking vs non-thinking | ✅ | ✅ | implicit in model set |
| 9 | prior supplied/withheld | ✅ 4 models | ❌ | `a1_prior_supplied`, 2,000 items each |
| 10 | narrative vs bulleted | ✅ all 11 | ❌ | `a1_narrative`, 4,400–8,800 items each |
| 11 | distractor | ✅ 4 models | ❌ | `a1_distractor`, 1,770 items each |
| 12 | severity-weighted | ✅ | ✅ | `a1_severity.csv` |
| 13 | temperature sweep | ❌ | ❌ | never run |
| 14 | **MIMIC external validity** | ✅ 33,558 | ❌ | **no table, no figure, no number anywhere** |
| 15 | answer/belief dissociation | ✅ | ❌ | **computed in this audit** (§5.1) — pre-registered |
| 16 | canonical vs random | ✅ | ❌ | **computed in this audit** (§5.2) |
| 17 | **ELR + trained adapter** | ✅ 1,463 | ❌ | adapter published, never evaluated |

Two of these are structural, not cosmetic:

**Ablation 14.** The README abstract advertises "a held-out MIMIC-IV-Note
external-validity arm". 33,558 responses were elicited. There is no MIMIC
number in any table, figure or results section. Either it runs or the claim
comes out of the abstract — and it is the designated answer to the "DDXPlus is
synthetic" reviewer attack in the proposal's own defence table, so it should
run.

**Ablation 17.** The paper's subtitle is *Measurement and Neuro-Symbolic
Correction*. The correction's trained component exists as a 349 MB published
adapter whose card says it "targets that deficit" — and there is no
measurement of whether it does. `eval_loss = 0.191` is the only number. A
reviewer who reads the model card will ask, and the honest answer right now is
that nobody knows.

Also: the model card claims the three-format joint training "is also ablation
#7 of the paper". Ablation 7 is not in the paper. Fix the card or run the
ablation.

Minor: "K" means permutation-pool size in `a1_k_sweep.csv` and ensemble size
in `methods.csv`. They are different quantities with the same symbol in the
same paper. Rename one.

---

## 5. New results computed in this audit

All five are written to `reports/tables/audit_2026-09-17/`. None required
re-running a model.

### 5.1 The answer–belief gap — the strongest result in the dataset

Every model was asked, on the **same item, with the same prompt content**,
both "pick the diagnosis" (`a1_answer`) and "score all 49 for plausibility"
(`a1_posterior`). The two tasks share `item_id`, so the comparison is exactly
paired.

Aggregate, all 11 models:

| model | forced-choice top-1 | posterior top-1 | ratio |
|---|---|---|---|
| gpt-oss-20b | 0.6532 | 0.3861 | 1.69× |
| qwen3-32b-nothink | 0.5373 | 0.1892 | 2.84× |
| medgemma-27b | 0.5204 | 0.1043 | **4.99×** |
| qwen3-32b-think | 0.5119 | 0.1498 | 3.42× |
| med42-8b | 0.3092 | 0.0772 | 4.00× |
| r1-distill-32b | 0.3078 | 0.1125 | 2.74× |
| qwen3-8b-nothink | 0.2812 | 0.1017 | 2.76× |
| qwen3-4b-nothink | 0.2664 | 0.0568 | 4.69× |
| qwen3-8b-think | 0.2516 | 0.0997 | 2.52× |
| qwen3-4b-think | 0.2333 | 0.0611 | 3.82× |
| medgemma-1.5-4b | 0.0805 | 0.0184 | 4.37× |

**11 of 11.** Mean gap 23.6 points.

Item-level paired, for the five models whose raw data is on the Hub:

| model | n paired | answer | posterior | **self-agreement** | agree (informative) | median rank of own answer | uniform share |
|---|---|---|---|---|---|---|---|
| med42-8b | 35,208 | 0.3092 | 0.0794 | 0.1209 | 0.1211 | 3 | 0.003 |
| qwen3-8b-nothink | 35,208 | 0.2812 | 0.1090 | 0.0846 | 0.1054 | 3 | 0.198 |
| qwen3-8b-think | 35,208 | 0.2516 | 0.0974 | 0.0670 | 0.0876 | 3 | 0.243 |
| qwen3-4b-nothink | 35,205 | 0.2665 | 0.0591 | 0.0417 | 0.0685 | 4 | 0.391 |
| qwen3-4b-think | 32,696 | 0.2347 | 0.0488 | 0.0192 | 0.0342 | 3 | 0.439 |

*self-agreement* = the model's forced-choice pick equals the argmax of its own
posterior, same item.

McNemar on med42-8b: χ² = 7,131 (8,633 vs 543 discordant pairs).
On qwen3-8b-nothink: χ² = 3,777 (7,896 vs 1,833).

**Confounds ruled out.**

- *Uniform collapse.* med42-8b emits a uniform posterior on 0.3% of items and
  still shows a 3.89× gap with 12.1% self-agreement. The effect is largest, not
  smallest, on the model with essentially no collapse.
- *Ties at the top.* Tie rate is 7.8% (med42-8b), 15.7% (qwen3-8b-nothink),
  46.3% (qwen3-4b-nothink). Restricting to strictly unimodal posteriors:
  0.0812, 0.1300, 0.1336 — the gap persists everywhere, and random rather than
  index-order tie-breaking changes nothing (0.0793, 0.1287, 0.0946).
- *Ordering.* Paired within item, so ordering is held fixed by construction.

**Why it matters more than A1.** The median rank of the model's own pick
inside its own distribution is 3 or 4 of 49. Chance would be 25. So the
posterior *does* encode the answer — it just refuses to put it on top. This is
not a knowledge failure and not noise. It is a failure to express what the
model knows as a coherent distribution.

That reading unifies the paper's results rather than adding to them:

- **A2's null** (direction agreement 0.454–0.549, r² ≤ 0.021) stops looking
  like "models cannot reason about evidence" and starts looking like "the
  probabilistic interface is broken", which is consistent with the same models
  scoring 26× chance when asked for a label.
- **ELR-Fusion becomes the indicated treatment rather than a bolt-on remedy.**
  If the model can identify evidence and diagnoses but cannot arithmetise
  them, then moving the arithmetic to a symbolic layer is the structurally
  correct fix — which is exactly the neuro-symbolic architecture MAKE names as
  a frontier.
- It is **label-free** at the self-agreement level, which matches the paper's
  stated design philosophy better than any axiom except A1.

It also delivers ablation 15's pre-registered prediction in a stronger form
than predicted: the dissociation does not merely hold, it holds without needing
answer-level *stability* — only answer-level *accuracy*.

**Before this goes in the paper:** compute it for all 11 models (needs the raw
parquets for the six models in §2.3), and add a same-format control — elicit
"pick one" *from* the plausibility vector by asking the model to name its own
argmax — to separate "cannot express a distribution" from "two prompts, two
moods".

### 5.2 Ablation 16 — DDXPlus's released ordering is not neutral, and its direction flips with scale

C7 argued from design that the code-sorted canonical order is "a systematically
more coherent presentation" and rebuilt the battery around that reasoning. The
canonical arm was collected but never analysed. It confirms C7, with a twist.

| model | canonical retest | random retest | Δ floor | canonical-vs-random | within-random | Δ | 95% CI |
|---|---|---|---|---|---|---|---|
| qwen3-4b-nothink | 0.0770 | 0.0722 | +0.0048 | 0.1405 | 0.1284 | **+0.0121** | [+0.008, +0.016] |
| qwen3-4b-think | 0.1271 | 0.1143 | +0.0128 | 0.1401 | 0.1315 | **+0.0079** | [+0.004, +0.012] |
| qwen3-8b-nothink | 0.1722 | 0.1692 | +0.0030 | 0.2071 | 0.2023 | **+0.0048** | [+0.001, +0.008] |
| qwen3-8b-think | 0.1547 | 0.1611 | −0.0064 | 0.1832 | 0.1844 | −0.0012 | [−0.006, +0.003] |
| gpt-oss-20b | 0.2783 | 0.2831 | −0.0048 | 0.2869 | 0.2866 | +0.0005 | [−0.003, +0.004] |
| medgemma-1.5-4b | 0.2810 | 0.2815 | −0.0005 | 0.2803 | 0.2836 | −0.0033 | [−0.008, +0.001] |
| med42-8b | 0.2400 | 0.2550 | −0.0150 | 0.2579 | 0.2623 | −0.0044 | [−0.011, +0.002] |
| r1-distill-32b | 0.2445 | 0.2616 | −0.0171 | 0.2660 | 0.2750 | **−0.0091** | [−0.016, −0.002] |
| medgemma-27b | 0.2210 | 0.2464 | −0.0254 | 0.2592 | 0.2712 | **−0.0120** | [−0.017, −0.006] |
| qwen3-32b-think | 0.2489 | 0.2666 | −0.0177 | 0.2964 | 0.3095 | **−0.0130** | [−0.018, −0.008] |
| qwen3-32b-nothink | 0.2323 | 0.2562 | −0.0239 | 0.2763 | 0.2908 | **−0.0145** | [−0.019, −0.010] |

Read the Δ-floor column: for every model at 20B and above, the canonical
ordering produces a **lower test–retest floor** than a random ordering — the
model is measurably more self-consistent under DDXPlus's released grouping.
For the 4B models the effect reverses.

Two consequences worth stating:

- C7's fix is now justified by evidence rather than by argument. Had the
  canonical order stayed in the floor, the floor would have been depressed by
  up to 0.024 for the large models — inflating their order effect — while
  being *inflated* for the small ones. The contamination was not a constant
  offset; it had opposite signs at opposite ends of the scale ladder, which is
  precisely the pattern that would have manufactured a spurious scale trend.
- It is a warning to anyone else using DDXPlus: the released evidence ordering
  is not a neutral presentation, and its effect depends on model scale.

### 5.3 Ablation 3 — incoherence does not grow with evidence load

The obvious confound, and it is clean.

Regressing per-case order effect on number of findings (range 6–40), the slope
is indistinguishable from zero for **10 of 11 models**. The exception,
`qwen3-8b-nothink`, is +0.00089 per finding — 0.03 across the entire range.

| model | 6–10 | 11–15 | 16–20 | 21–25 | 26–40 | slope | sig |
|---|---|---|---|---|---|---|---|
| qwen3-4b-nothink | 0.0579 | 0.0603 | 0.0531 | 0.0392 | 0.0706 | +0.00018 | – |
| qwen3-8b-nothink | 0.0151 | 0.0431 | 0.0370 | 0.0443 | 0.0352 | +0.00089 | yes |
| qwen3-32b-nothink | 0.0238 | 0.0539 | 0.0385 | 0.0399 | 0.0285 | +0.00026 | – |
| medgemma-27b | 0.0264 | 0.0078 | 0.0255 | 0.0435 | 0.0146 | −0.00012 | – |
| gpt-oss-20b | 0.0025 | −0.0004 | 0.0094 | 0.0030 | 0.0000 | +0.00003 | – |

Full table in `ab03_evidence_load.csv`.

The anchored quantity is flat; the *raw* permutation JSD is not, and it moves
in **opposite directions by model family** — negative for general Qwen3
(r = −0.09 to −0.38), positive for the medically tuned models (MedGemma-27B
r = +0.26, Med42-8B r = +0.15). Another instance of the paper's central
methodological claim: the raw number trends, the anchored one does not.

### 5.4 The order-effect estimate is converged in K

Across K ∈ {2,3,5,10} the order effect moves by at most 0.010 (median 0.003).
K = 2 already gives the K = 10 answer to within 0.01 for every model. Worth one
sentence — it forecloses "you under-sampled the permutation space".

### 5.5 ELR-Fusion across all 11 models — and a correction to the paper's stated reason

The exactness claim generalises perfectly and the paper should say so:
**residual order sensitivity is 0.0000 for all 11 models × all 12 τ values,
132 of 132 configurations.** Table 1 currently demonstrates this on one model.

The accuracy cost, however, is **not** the uniform 5.6-point loss the paper
implies from `qwen3-32b-nothink`:

| model | direct | ELR | retention |
|---|---|---|---|
| medgemma-1.5-4b | 0.0184 | 0.0194 | **1.06×** |
| qwen3-32b-think | 0.1498 | 0.1575 | **1.05×** |
| r1-distill-32b | 0.1125 | 0.1002 | 0.89× |
| qwen3-4b-think | 0.0611 | 0.0485 | 0.79× |
| gpt-oss-20b | 0.3861 | 0.3007 | 0.78× |
| qwen3-32b-nothink | 0.1892 | 0.1334 | 0.71× |
| med42-8b | 0.0772 | 0.0455 | 0.59× |
| qwen3-4b-nothink | 0.0568 | 0.0328 | 0.58× |
| qwen3-8b-think | 0.0997 | 0.0383 | 0.38× |
| medgemma-27b | 0.1043 | 0.0332 | 0.32× |
| qwen3-8b-nothink | 0.1017 | 0.0307 | 0.30× |

Range 0.30×–1.06×, median 0.59×. **For two models ELR-Fusion is more accurate
than direct elicitation.** Reporting the distribution is both more honest and
more interesting than reporting one model's loss as the cost of the guarantee.

**A correction to the paper's explanation.** §4.6 concludes "improving the
log-LR elicitation, not the fusion rule, is where the remaining accuracy is".
I tested the natural version of that claim — that retention should improve with
model competence — and it does not: Pearson r between competence top-1 and
retention ratio is **−0.05** (n = 11), and −0.02 using each model's best τ.
GPT-OSS-20B has by far the best absolute fused accuracy (0.330 at τ = 3, top-5
0.678) but only mid-pack retention.

So the conclusion may still be right, but the competence-scaling argument does
not support it and should not be offered. What the data does support is that
retention is model-specific and unpredictable from competence — which is a
reason to report all eleven rows rather than to theorise from one.

Cross-model, the τ tension is universal and much sharper than §4.6 shows: the
accuracy-optimal τ is ≥ 2.0 for 7 of 11 models while the NLL-optimal τ is
≤ 0.10 for 8 of 11. A 20-fold-different operating point, in the same direction,
in nearly every model. That is a far stronger version of the argument for
reporting the frontier instead of a fitted constant.

One caveat to check before publishing GPT-OSS's fusion numbers: its
`elr_weights_numeric` schema validity is **0.371** and `elr_weights_ordinal`
**0.284** (`schema_validity.csv`). Its likelihood table is built from roughly a
third of the intended elicitations, yet it fuses best of any model. Either
that is robustness worth a sentence, or it is a coverage artifact worth
checking.

---

## 6. What is genuinely strong and should be defended as-is

- **The three-anchor design and the scale reversal.** Raw divergence rises
  0.128 → 0.291 across the Qwen3 ladder while the normalised effect falls
  0.215 → 0.127. This is the paper's best methodological argument and it is
  correctly reported.
- **The empirical-posterior oracle.** Conditioning on 1.29M patients with
  Wilson intervals and ≥500 support, rather than assuming conditional
  independence — with the independence assumption's failure quantified at JSD
  0.546 against a 0.049 floor. This removes the standard objection from the
  measurement itself. It is the cleanest contribution in the paper.
- **The generator noise floor (C3).** 25.8% of exact replicates disagree, mean
  JSD 0.0490. It applies to everyone using DDXPlus, not just this paper.
- **C11 as written.** The 16.6× swing from one decoding flag, with the harmony
  channel mechanism explained and two remedies tried and rejected on evidence,
  is exactly the register MAKE rewards. Add the `beats_chance` failure from
  §2.2 and it gets better.
- **The MIMIC governance guard.** `hf_sync.py`'s `FORBIDDEN` screen, with the
  deliberate asymmetry that MIMIC *code* ships and MIMIC *data* cannot, is
  well-reasoned and worth describing rather than just doing.

---

## 7. Recommended research story

The current framing is "four coherence axioms, three anchors, one remedy". It
is defensible but it leads with A1, which is the axiom a reviewer is most
likely to call known, and it buries A2's null and the answer–belief gap.

Lead instead with the dissociation, and make the axioms the diagnosis:

> **Claim.** Clinical language models can identify the right diagnosis and
> cannot represent it as a probability. The failure is in the probabilistic
> interface, not the clinical knowledge — and because it is an interface
> failure, a symbolic combination layer fixes it by construction.

The evidence assembles in one line each:

1. **They know.** 26× chance on 49-way forced choice; GPT-OSS-20B at 0.653.
2. **They cannot say it.** Their own forced-choice pick is the argmax of their
   own posterior on 3–12% of items, ranked 3rd of 49 — paired, label-free,
   11/11 models, χ² in the thousands. *(§5.1)*
3. **So the belief state does not track evidence.** A2: direction agreement
   0.454–0.549, r² ≤ 0.021 against an assumption-free oracle. Not a reasoning
   deficit — a representation deficit.
4. **And it is unstable in ways that are not artifacts.** A1 order effects
   above the model's own floor for 8 of 11, with the anchors reversing the
   scale trend; A4 primacy anchoring in 6 of 11 plus recency in MedGemma-27B;
   A3 tail movement on certified-irrelevant evidence. Not evidence load
   (§5.3), not permutation sampling (§5.4), not the released ordering (§5.2).
5. **Move the arithmetic out of the model.** ELR-Fusion: exactly zero order
   effect, 132/132 configurations, at zero marginal query cost, against
   ensembling that decays as O(1/√K) and never reaches zero. Accuracy
   retention 0.30×–1.06×, reported in full.
6. **The instrument had to survive eleven calibration steps to say any of
   this** — including one where the only binary gate passed a broken model.

This keeps every existing result, adds no compute, demotes nothing to an
appendix, and gives the paper a claim that is not "order effects exist". It
also makes the neuro-symbolic argument structural rather than rhetorical,
which is the MAKE fit.

**Retitle suggestion:** the current subtitle *Measurement and Neuro-Symbolic
Correction* promises a correction that is currently unmeasured (§4, ablation
17). Either run it or narrow the subtitle.

---

## 8. Action list

**Blocking — do not submit without these**

1. Re-sync both Hub repos. The published tables currently carry
   `_gptoss20b_harmony_broken` as a result. *(§2.2)*
2. Commit `coherence/data/` — six modules, never in any commit. *(§2.1)*
3. Report `excess_over_floor` in A3 and rephrase the claim as a tail claim.
   *(§3.1)*
4. Regenerate `03_results.md` from the current CSVs; it predates the C11 fix.
   *(§3.4)*
5. Either run the MIMIC arm or remove it from the abstract. *(§4)*

**High value, low cost — all computable from data already collected**

6. Report Table 1 over all 11 models; the exactness result is 132/132. *(§5.5)*
7. Add the answer–belief gap as a primary result. *(§5.1)*
8. Re-run analysis with the η² fix and report A4's directional finding —
   primacy in 6, recency in MedGemma-27B — alongside the divergence reading
   that is blind to it. *(§3.2)*
9. Add §5.2 (canonical ordering) and §5.3 (evidence load) as ablations 16 and
   3. Both are written; both currently have no home in the paper.
10. State the three non-significant A1 rows as non-significant. *(§3.3)*
11. Add ablations 7, 9, 10, 11 — the data is collected, the analysis is not
    written. Ablation 7 in particular is claimed by the model card.
12. Fix τ ≈ 0.4 to "plateaus above 0.4"; drop the competence-scaling
    explanation, which the data does not support. *(§3.6, §5.5)*
13. Add A3 and A4 figures — two of four axioms currently have none.

**Release hygiene**

14. Publish raw data for the six missing models. *(§2.3)*
15. Add `LICENSE`; push `figures_results.py`, the `result_*` figures and
    `calibration_c6.json`. *(§2.4)*
16. Reconcile the 1,118,457 / 1,119,920 / eleven-vs-twelve accounting. *(§3.5)*
17. Rename one of the two quantities called K. *(§4)*

**Decide**

18. Ablation 17 — evaluate the LoRA adapter through ELR-Fusion, or narrow the
    subtitle and re-scope the model card. As it stands the paper's named
    correction has no result. *(§4)*

---

## Appendix — verification performed

- 28 README numeric claims re-derived from `reports/tables/*.csv`; all 28
  reproduce to the stated precision. The reporting problems in §3 are
  omissions, not errors.
- Local source tree diffed against the Hub `code/` tree: 6 files missing
  locally (restored), 1 missing on the Hub, 3 with content drift.
- Local `reports/` diffed against Hub `reports/`: 13 files local-only, 10
  tables with content drift, all drift traced to the C11 re-run.
- Elicited-response counts recomputed from `logs/run_*.jsonl`, de-duplicating
  re-run tasks by taking the max per (model, task).
- New analyses run against `results/analysed/a1_per_case.parquet` (21,516 rows,
  11 models) and raw per-item parquets for 5 models pulled from the Hub.
- Confound tests for §5.1: uniform collapse, tie-at-top with index-order and
  random tie-breaking, and per-item pairing.

Tables written to `reports/tables/audit_2026-09-17/`:
`answer_belief_gap_paired.csv`, `answer_belief_gap_aggregate.csv`,
`ab16_canonical_vs_random.csv`, `ab03_evidence_load.csv`,
`elr_retention_all_models.csv`.

Code changed: `.gitignore` (anchor `/data/`),
`coherence/axioms/a4_anchoring.py` (nanmean for η²). No result table was
regenerated — the η² fix requires a re-run of `run_analysis`.
