# Methodology

*Probabilistic Coherence as a Test of Clinical Reasoning in Large Language
Models: Measurement and Neuro-Symbolic Correction*

---

## 1. The correctness criterion is a theorem

Bayesian belief revision over a fixed evidence set is order-invariant.
For evidence `E = {e_1 … e_n}` and hypothesis `d`,

```
P(d | e_1, …, e_n)  is a function of the SET E, not of any ordering of it.
```

This holds unconditionally: it requires no independence assumption, no
clinician panel, no LLM judge, no severity weighting and no threshold. If a
model's differential changes when the same clinician findings are presented in
a different sequence, that is a reasoning failure whose correct value is
exactly zero.

Two consequences shape the whole design.

**The primary measurement needs no labels.** A1 and A4 compare a model
against itself, so they run unchanged on real de-identified narratives where
no ground-truth posterior exists. The synthetic-data objection does not reach
the core measurement.

**The remedy can be proved rather than demonstrated.** A method that
aggregates per-finding evidence weights by summation inherits commutativity,
so its order effect is identically zero for every model, case and budget
(Section 6).

---

## 2. Why an exact zero is not enough: the three anchors

A bare divergence across permutations is uninterpretable, because sampling
alone produces divergence. Every figure in this paper is reported against
three anchors, two of which are measured per model.

**Anchor 1 — test–retest floor (per model).** The same ordering, resampled.
This is the model's own stochasticity under an identical prompt.

```
OrderEffect  =  JSD(across permutations)  −  JSD(same permutation, resampled)
```

**Anchor 2 — between-patient ceiling (per model).** Divergence between
posteriors for *different* patients. It sets the scale: a normalised order
effect of 1.0 would mean that reordering one patient's findings moves the
belief as far as substituting a different patient.

```
NormalisedOrderEffect  =  (JSD_perm − JSD_retest) / (JSD_between_patient − JSD_retest)
```

An earlier design used a *shuffled-evidence* arm as the ceiling, following the
proposal. Measurement showed this inverts: given incoherent findings a model
falls back to a default posterior, so two different nonsense inputs agree
**more** than two resamples of a real case (0.036 against a 0.059 floor on
`qwen3-4b-nothink`). The shuffled arm is retained and reported, but as a
diagnostic for default-answer behaviour, not as a ceiling.

**Anchor 3 — the generator's own replicate noise (dataset-level).** Measured
in the data audit: across 21,004 DDXPlus replicate pairs — patients identical
in evidence multiset, age, sex, initial evidence and true pathology — the
shipped differential differs for 25.8% of pairs, mean JSD 0.049, top-1
flipping in 0.48%. No released field explains the difference, so it is
irreducible. **No model can be scored against the reference more precisely
than the reference reproduces itself.**

Reporting all three is the methodological register this venue rewards, and
omitting the first is, in our judgement, the fastest route to rejection.

---

## 3. Data

### 3.1 DDXPlus (primary)

All 1,292,579 released patients across the three splits; 49 pathologies, 223
evidence questions resolving to 516 distinct evidence tokens, per-pathology
severity 1–5. CC-BY-4.0, ungated.

The full audit is in `reports/data_audit_report.md` (19 checks). Four findings
bear directly on the design:

| finding | consequence |
|---|---|
| `release_conditions.json` supplies conditional probabilities for **0 of 888** (pathology, evidence) slots | exact posteriors are not readable from the KB; see §4 |
| the differential is **stochastic** — 25.8% of replicates disagree | anchor 3; and a ceiling below 1.0 on any DDXPlus differential-matching metric |
| prior spans **252×** across pathologies | case sampling is stratified by pathology and severity |
| 1.34% exact-duplicate records, 8,705 records repeated across splits | irrelevant here (within-case, label-free) but reported for downstream users |

### 3.2 MIMIC-IV-Note (external validity)

800 discharge summaries, HPI section segmented into 6–20 findings, notes
dominated by de-identification placeholders excluded. A1 and A4 only, since
neither needs labels.

Because MIMIC has no fixed pathology list, the candidate set is elicited
**once per case from the canonical ordering and then held fixed** across every
permutation of that case. This keeps the metric identical to the DDXPlus arm
and avoids string-matching free-text diagnoses or invoking an LLM judge —
either of which would reintroduce the contestable ground truth the design
exists to avoid.

**Governance.** MIMIC is credentialed. Patient-derived text and per-case
results never leave the machine; the Hub sync hard-refuses any path matching
`mimic` or `physionet`, and only aggregate divergence statistics are
published. No API model is pointed at this arm.

---

## 4. Ground truth for belief updating, without an independence assumption

A2 and A3 need an absolute reference for how much a finding *should* move the
posterior. The released KB cannot supply it (§3.1).

**The reconstruction that fails, reported because the failure is
informative.** Per-finding conditionals `P(e | d)` estimate cleanly from the
corpus (binomial SE ≈ 3×10⁻⁴). Recombining them by naive Bayes reproduces the
shipped differentials at **mean JSD 0.546** — roughly 11× the generator's own
noise — with top-1 agreement 74.5%. Temperature scaling does not rescue it, so
the defect is structural, not calibration. The cause is measurable:
conditional independence given the pathology is violated, with **8.7% of
sampled finding pairs exceeding |φ| = 0.1** (mean |φ| 0.030, max 1.0).

**The reference we use instead.** The corpus is 1.29M draws from the
generator's joint distribution and `PATHOLOGY` is the sampled ground truth, so

```
P(d | E)  ≈  #{patients with pathology d whose findings ⊇ E}
             ────────────────────────────────────────────────
             #{patients whose findings ⊇ E}
```

is a consistent, **assumption-free** estimator of the generator's own
posterior. Its cost is variance, which is an exact binomial quantity used to
*select* items rather than being absorbed silently: every item carries Wilson
intervals, and items are admitted only above a 500-patient support.

Measured feasibility on evidence subsets drawn from real patients: median
matching support 119k at |E|=1, 21k at 2, 9.0k at 3, 5.1k at 4, 2.9k at 5,
1.6k at 6. **A2/A3 are absolute measures up to |E| ≈ 6**; A1/A4 need no
reference and run at any size, including full ~20-finding cases and MIMIC
narratives.

This matters beyond convenience. The standard objection to Bayesian clinical
baselines — "naive Bayes assumes independence, which is false clinically" —
does not reach our *measurement* at all. It reaches only the ELR-Fusion
*method*, where we quantify it and correct it (§6.3).

---

## 5. The instrument: four axioms

Each has an exact zero and needs no human annotation.

### A1 — Order invariance (load-bearing)

Present evidence set `E` in `K = 10` random permutations; elicit a posterior
after each. Metric: mean pairwise Jensen–Shannon divergence (base 2, so in
[0,1]), reported as `OrderEffect` against the three anchors of §2.

Four arms per case, 39,120 items over 1,956 cases:

| arm | n/case | purpose |
|---|---|---|
| permutation | 10 | the estimand |
| retest | 6 | test–retest floor, same *random* ordering resampled |
| canonical | 2 | DDXPlus's released code-sorted order, as its own condition |
| shuffled | 2 | diagnostic for default-answer behaviour |

**Why `canonical` is separated.** DDXPlus releases findings code-sorted, which
groups related questions (all pain items, then antecedents) and is a
systematically more coherent presentation. An earlier design used it as
`perms[0]` and anchored the retest arm on it; models proved measurably more
willing to commit under it (informative rate 67.5% vs 60.2%), putting floor
and numerator under different conditions. Separating it removes the confound
**and** converts it into a reportable result: does the released grouping help?

### A2 — Update fidelity

For each finding `e` added to context `C`, regress the model's implied change
in log-odds on the true change from the empirical oracle. `β < 1` is
conservatism (the classical human pattern, Edwards 1968); `β > 1` is
overreaction; `R²` is how much of the correct signal is tracked at all.

Two filters, both load-bearing. Pathologies whose true update is negligible
are dropped (they carry no magnitude signal; A3 tests null updates on items
built for it). Pathologies the oracle cannot estimate in *both* conditioning
sets are dropped — without this, zero-count pathologies clip to log-odds −9.2
and the slope measures the floor rather than the model. Regression is weighted
by oracle precision. **Ties are reported separately, not scored as
wrong-direction**: a model returning a quantised posterior leaves many
pathologies bit-identical across the two elicitations, and counting those as
errors reports anti-correlation where there is none.

### A3 — Redundancy insensitivity

Add a finding the *corpus certifies* as uninformative given the context —
empirical posterior with and without it differing by < 10⁻³ JSD at ≥ 500
patients of support. True update is zero; any movement is spurious. Reported
as **excess over the model's own test–retest floor**, not as a ratio to the
certified truth: the raw ratio produces impressive multipliers (1,340× on one
model) that are entirely sampling noise.

This catches the "more text ⇒ more confidence" pathology, which is invisible
to accuracy metrics and clinically dangerous. We report the entropy-decrease
rate as the directional reading.

### A4 — Positional anchoring

Identify each case's most decisive finding *by measurement* (the finding that
moves the true pathology's log-odds furthest from the prior), then vary only
its position — first, middle, last — permuting the remainder. Regress the
decisive pathology's log-odds on position coded −1/0/+1, within-case centred.
`β < 0` is anchoring on the first finding; `β > 0` is recency; 0 is coherent.
Also reported as η², the share of within-case variance position explains.

**Terminology.** We say *positional anchoring* throughout and avoid "premature
closure", which arXiv:2605.15000 uses for a different construct (failure to
abstain).

### Severity weighting

DDXPlus attaches severity 1–5 per pathology. Incoherence is reported twice:
unweighted, and as the rate at which reordering moves a **severity ≤ 2**
pathology in or out of the top-5. Incoherence that shuffles two benign
diagnoses is not the clinical event that dropping a pulmonary embolism is.

---

## 6. The method: ELR-Fusion

### 6.1 Construction

Elicit, for each finding `e_i` and pathology `d`, a likelihood ratio
conditioned on the finding and pathology **alone** — no accumulated narrative.
Combine symbolically:

```
log-odds(d | E)  =  log-odds(d)  +  Σ_i  log LR(e_i | d)
```

### 6.2 Three properties

1. **Order invariance is a theorem.** Addition commutes, so the result is a
   function of the evidence set. `OrderEffect = 0` exactly, for every model,
   case and budget. Verified numerically at machine epsilon: mean pairwise JSD
   5.2×10⁻¹⁷ over 200 permutations.

2. **It is auditable.** Every finding carries a signed numeric weight toward
   every hypothesis. The explanation *is* the computation — causability in
   Holzinger's sense, not post-hoc attribution over an opaque one.

3. **It decomposes the failure.** Coherent but less accurate than direct
   prompting ⇒ the deficit is in likelihood estimation. Coherent *and* more
   accurate ⇒ direct prompting was losing information to positional effects.
   Either is a clean, reportable result.

**Cost, reported honestly both ways.** The weight table is elicited
context-free, so it amortises across the corpus: a new case costs *n* table
lookups and zero new queries once the finding vocabulary is covered. The
unamortised cost is *n* queries. Both are in Table 1, because amortisation
only helps when the vocabulary is closed.

### 6.3 The honest caveat, quantified

Order invariance of the *true* posterior is unconditional. Naive summation of
log-LRs is exact only under conditional independence given the pathology,
which §4 shows is violated. We therefore (a) report the violation measured on
1.29M patients, (b) implement the exact second-order correction
`log P(a,b|d) − log P(a|d) − log P(b|d)` estimated from the corpus, and (c)
make the correction an ablation so its value is a number rather than a claim.
The correction preserves exact order invariance, being a sum over unordered
pairs.

This is stated in its own subsection rather than buried. Note again that the
*measurement* (A1) assumes nothing; only the method does.

### 6.4 The comparison that decides the paper

A competent reviewer will ask about permutation ensembling, which also
achieves order invariance — trivially, by averaging over K orderings. We build
the paper around it rather than wait to be asked.

| method | order-invariant | cost / new case | auditable per finding | scales with |
|---|---|---|---|---|
| direct prompting | no | 1× | no | — |
| CoT then posterior | no | 1× | no | — |
| self-consistency (k) | no | k× | no | k |
| permutation ensemble (K) | approximately, O(1/√K) | **K×** | no | K |
| **ELR-Fusion** | **exactly, any budget** | **n (amortised: 0)** | **yes** | n, not K |
| empirical oracle | by definition | — | — | upper bound |
| prior only | by definition | 0 | — | floor |

Measured on `qwen3-4b-nothink`, the ensemble residual decays as predicted —
0.0731, 0.0517, 0.0331 at K = 2, 3, 5 — while ELR-Fusion is 0.0000 throughout.
Ensemble sizes satisfy 2K ≤ 10 because the residual is measured between two
**disjoint** ensembles; an earlier version clamped K silently and printed the
same number twice.

All methods are scored on the **intersection of cases every method produced**,
so the comparison is paired. Unpaired, a method that fails on hard cases is
scored on an easier subset and looks better for it.

---

## 7. Models

A factorial design; each axis is a question, not a leaderboard row.

| axis | contrast | question |
|---|---|---|
| reasoning | Qwen3 thinking vs non-thinking, **identical weights** | what does test-time reasoning buy? |
| domain tuning | MedGemma-27B, Med42-8B vs Qwen3 | does medical tuning improve belief revision or only knowledge? |
| scale | Qwen3 4B/8B/32B; MedGemma 4B/27B; GPT-OSS 20B/120B | does coherence emerge with scale? |
| recipe | R1-Distill-32B vs Qwen3-32B | distilled vs native reasoning |

Twelve configurations. Two corrections to the original plan, made after
checking the Hub: **MedGemma 1.5 27B does not exist** (Google released
`medgemma-1.5-4b-it`; the 27B line stops at `medgemma-27b-text-it`), and
Qwen3-32B in bf16 does not fit, so the official AWQ checkpoint is used with
the thinking contrast on those same weights.

**Hardware.** One RTX PRO 5000 Blackwell, 48 GB (sm_120). 27B–32B run 4-bit,
which is the deployment regime the paper is about. `gpt-oss-120b` does not
fit — MXFP4 weights ≈ 63 GB — so it runs CPU-offloaded on the A1 core only and
is flagged as a reduced-scale arm in every table rather than silently omitted.

---

## 8. Elicitation, and why it had to be calibrated first

Ten calibration steps are documented in `reports/instrument_calibration.md`.
Four would each have produced a plausible but wrong headline number, and none
was visible to the proposal's "≥ 98% valid structured outputs" gate.

| # | defect | why it mattered |
|---|---|---|
| C4 | schema failure rate depended on the arm — 20% shuffled vs 10% retest | discarding failures biases the order effect *by the ordering condition itself* |
| C6 | **62.9% of responses were exactly uniform** at 100% "validity" | a uniform posterior carries no belief and is trivially order-invariant; it deflates both the effect and its normaliser |
| C7 | retest arm replicated DDXPlus's canonical order | floor and numerator under different conditions (§5, A1) |
| C9 | Table 1 ensembles clamped and unpaired | the central comparison was wrong in two ways |

**The elicitation we adopted.** A fixed, alphabetically ordered candidate list
identical in every prompt (so output-position bias is constant and cannot
masquerade as an order effect); grammar-constrained decoding pinned to
xgrammar with whitespace disabled; and **0–100 plausibility scores rather than
a normalised distribution**, normalised afterwards. The last change roughly
halves the uniform collapse at both model scales (4B 68.8% → 39.6%; 8B 51.2%
→ 23.4%) because a flat answer is the easiest way to satisfy a sum-to-one
constraint. Normalisation is monotone and preserves the expressed ordering.

**The gate we now report.** Schema validity is necessary but not sufficient.
The reported gate is the **informative rate** — the share of responses that
express any belief — broken out **by arm**. If it differed across arms, the
conditional estimand would be selection-biased and only the unconditional one
reportable. After the C7 fix it does not (permutation 58.5% vs retest 59.9%).

**Informativeness is a model property, not an instrument defect**, and is
reported as a result: 53.4% (qwen3-4b-think), 55.3% (qwen3-4b-nothink), 72.5%
(qwen3-8b-think), 75.5% (qwen3-8b-nothink), **99.8% (med42-8b)**. A medically
tuned 8B model expresses a belief on essentially every item under the identical
prompt that makes a general 4B model shrug on half.

The A1 estimand is therefore reported three ways: unconditional over all cases
(honest population figure, diluted by non-responses); conditional on the model
having expressed a belief, computed over informative responses within each arm
with ≥ 2 per arm; and the informative rate itself.

---

## 9. Ablations

Every row is a planned experiment, and collectively they are the contribution
rather than a defensive appendix. Full-battery ablations run on all
configurations; focused ablations run on a representative subset spanning the
scale and tuning axes, which is where the resources buy the most information.

| # | ablation | isolates | scope |
|---|---|---|---|
| 1 | K ∈ {2,3,5,10} | convergence of the order effect | all |
| 2 | retest floor, between-patient ceiling, generator floor | anchoring of every figure | all |
| 3 | evidence-set size | does incoherence grow with context load? | all |
| 4 | decisive-finding position | A4 | all |
| 5 | ELR vs all baselines incl. permutation ensemble | the central comparison | all |
| 6 | independence correction on/off | the honest caveat, quantified | analysis-side |
| 7 | LR format: numeric / log-odds / ordinal bands | robustness of the elicitation | all |
| 8 | thinking vs non-thinking, identical weights | what test-time reasoning buys | Qwen3 |
| 9 | prior supplied vs withheld | base-rate neglect | subset |
| 10 | narrative prose vs bulleted | presentation-format dependence | all |
| 11 | salient-but-irrelevant distractor | framing susceptibility — motivated by arXiv:2609.02797's order-of-magnitude finding | subset |
| 12 | severity-weighted vs unweighted | clinical consequence | all |
| 13 | temperature ∈ {0, 0.3, 0.7} | is incoherence just sampling noise? (anchor 1 already answers this) | subset |
| 14 | MIMIC real-narrative arm | external validity without labels | 3 models |
| 15 | does answer-level stability predict posterior-level coherence? | behaviour/belief dissociation | all |
| 16 | canonical vs random ordering | does DDXPlus's code-sorted grouping help? | all |
| 17 | ELR-Fusion with the trained LR adapter | does the accuracy deficit close? | Qwen3-8B |

**Ablation 15 is pre-registered with a prediction.** If top-1 is stable across
orderings while the full posterior is not, that is a dissociation between
behaviour and belief, and it is the more citable half of the paper. Wang
(2026) reports its answer-level shadow — 90% access, 60% retention, a 30%
dissociation — from rubric scoring on 50 cases; we measure both levels on the
same items, label-free, at 1,956 cases across 12 configurations. **We predict
the dissociation holds**, and our early data is consistent: top-1 flips in
70–96% of cases *while* the posterior-level order effect stays a modest
fraction of the between-patient scale.

---

## 10. What this design cannot fail to produce

Order effects either exist or they do not. Large effects are a strong finding.
Small effects in frontier models but large in deployable open models is a
*better* finding, because that is the deployment-relevant regime. Null
everywhere is a clean negative result — and with the three anchors in place, a
credible one, which is exactly what an unanchored divergence number could
never be.
