# Instrument calibration

Before any coherence number is trusted, the elicitation instrument itself has
to be shown not to fail differentially across the conditions being compared.
A schema-failure rate that depends on the evidence ordering would register as
an order effect that is an artefact of the harness, not a property of the
model. This note records the calibration runs and the decisions they forced.
All runs use `qwen3-4b-nothink`, the weakest model in the set and therefore
the binding constraint.

## C1 — JSON parse strictness

`json.loads` on the full completion rejected responses that were well-formed
but followed by whitespace padding. Replaced with `raw_decode` on the leading
object. Recovered 0% here, but removes a latent failure mode for models that
pad after a complete object.

## C2 — Grammar backend and whitespace

**Symptom.** 3.5% of completions never closed the JSON array: the model
emitted `\t\n` repeatedly between elements until `max_tokens`.

**Cause.** vLLM's `auto` structured-output backend does not honour
`disable_any_whitespace`. The grammar therefore permitted unbounded
whitespace inside the array.

**Fix.** Pin the backend to `xgrammar` with `disable_any_whitespace=True`.

**Effect.** Truncation failures 3.5% to 0%. Throughput also rose from
12.6 to 18.4 items/s, because the padding tokens were being generated and
then discarded.

## C3 — Output representation

| representation | valid | all-zero | mean entropy (bits) |
|---|---|---|---|
| probabilities summing to 1 | 93.0% | 7.0% | 4.57 |
| integer points out of 1000 | 87.8% | 12.3% | 4.72 |

Integer point allocation was tested on the hypothesis that models round small
probabilities to zero. It is **worse**, and was rejected.

## C4 — The all-zero collapse, and why it mattered

A residual 7-20% of completions were a valid JSON array of 49 zeros, which is
not a probability distribution. Critically, the rate was **not uniform across
conditions**:

| arm | failure rate |
|---|---|
| retest | 10.0% |
| permutation | 13.6% |
| shuffled | 20.0% |

and it varied with the number of findings (30.9% below 8 findings, 2.4% at
8-15, 20.6% above 32). Discarding those items would have biased the order
effect, because the discard rate itself depends on the ordering condition.

Three prompt variants were compared on 600 items stratified across all three
arms:

| variant | overall valid | permutation | retest | shuffled | entropy | mean max-prob |
|---|---|---|---|---|---|---|
| baseline | 85.2% | 85.0% | 90.0% | 80.5% | 5.01 | 0.121 |
| **+ positivity clause** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 5.08 | 0.105 |
| + positivity + rank-first | 100.0% | 100.0% | 100.0% | 100.0% | 5.23 | 0.092 |

**Adopted: the positivity clause.** It removes the differential failure
entirely at minimal cost to the elicited distribution. The rank-first variant
also reaches 100% but flattens the posterior further (entropy 5.23 of a 5.61
maximum), which is a larger intervention in the quantity being measured.

## C5 — Residual note on near-uniformity

Even at 100% validity, `qwen3-4b-nothink` returns posteriors averaging 5.08
bits of entropy against a 5.61-bit maximum over 49 candidates, and a mean
top-1 mass of 0.105. That is a property of the model, not of the instrument,
and it is reported as a result: a model whose posterior is near-uniform has
little belief to be incoherent about, so its order-effect magnitude must be
read against its entropy. The analysis reports both.

## Gate

The proposal's week-2 gate was ">= 98% valid structured outputs across all
models". The A1 harness meets it at 100% on the binding model. The gate is
re-checked per model at run time and recorded in `results/schema_validity.csv`;
any model falling below 98% is re-calibrated before its results are used.

## C6 — The uniform collapse, and why "100% schema validity" was a hollow gate

C4 closed the all-zero failure and the harness reported 100% valid structured
outputs on every task. That gate turned out to measure the wrong thing.

Of `qwen3-4b-nothink`'s 33,252 A1 responses, **62.9% were the same vector** —
exactly uniform, 1/49 on every pathology. A uniform posterior expresses no
belief, and it is trivially order-invariant, so those items deflate both the
order effect and the between-patient ceiling used to normalise it. The
positivity clause adopted in C4 had not removed the degenerate mode; it had
moved it from all-zeros (13%) to uniform (63%) and made it parse.

Three variants were compared on 500 permutation items at two model scales:

| model | variant | valid | **uniform** | **informative** | distinct values emitted |
|---|---|---|---|---|---|
| qwen3-4b | probabilities + positivity | 100% | 68.8% | 31.2% | 1.5 |
| qwen3-4b | + explicit "do not give equal probabilities" | 100% | 67.6% | 32.4% | 1.5 |
| qwen3-4b | **plausibility 0–100** | 100% | **39.6%** | **60.4%** | 3.2 |
| qwen3-8b | probabilities + positivity | 100% | 51.2% | 48.8% | 1.9 |
| qwen3-8b | + explicit "do not give equal probabilities" | 100% | 50.6% | 49.4% | 2.0 |
| qwen3-8b | **plausibility 0–100** | 100% | **23.4%** | **76.6%** | 3.5 |

Two things follow. Instructing the model not to answer uniformly does almost
nothing (68.8% to 67.6%). Changing what is *asked for* does: scoring each
diagnosis 0–100 for plausibility, with no sum-to-one constraint, roughly
halves the collapse at both scales. Normalising those scores afterwards is a
monotone transform and preserves the ordering the model expressed.

Collapse also falls with scale (4B 39.6% to 8B 23.4% under the adopted
format), so part of it is a genuine capability limit rather than an artefact.

**Adopted: plausibility 0–100.** And the gate is replaced. Schema validity is
necessary but not sufficient; the reported gate is now the **informative
rate**, and the A1 estimand is reported three ways:

* `order_effect` — unconditional, over all cases. The honest population
  figure, diluted by non-responses.
* `order_effect_informative` — over cases where the model expressed a belief
  in every arm. Order sensitivity *given* that a belief was formed.
* `informative_rate`, broken out **by arm**. If informativeness differs
  across arms, the conditional estimand is selection-biased and only the
  unconditional one is reportable. This is checked per model and is the same
  discipline C4 forced.

The between-patient ceiling is likewise computed over informative responses
only; uniform anchors would drag it towards zero and inflate the normalised
order effect.

## C7 — The canonical order was contaminating the test–retest floor

With the C6 fix in place, informativeness still differed **by arm**:
permutation 60.2%, retest 67.5%, shuffled 36.4%. Since the conditional
estimand is only reportable when informativeness is arm-independent, this
had to be explained before the sweep could proceed.

The cause was a design asymmetry, not a model property. DDXPlus releases each
patient's findings **code-sorted** (`E_38, E_52, E_65, …`), which groups
related questions — all the pain items together, then the antecedents. That
is a systematically more coherent presentation than a random ordering, and
models are measurably more willing to commit to a belief under it. The
battery used that canonical order as `perms[0]`, and the test–retest arm
replicated `perms[0]`. So the floor was measured under a *more coherent*
presentation than the permutation arm it is subtracted from, and part of the
headline order effect was canonical-versus-random rather than
order-versus-order.

**Fix.** The permutation arm is now K *random* orderings with the canonical
order excluded; the retest arm resamples one of those same random orderings,
so floor and numerator sit under matched conditions; and the released order
becomes its own `canonical` arm with 2 replicates. That arm answers "does the
released grouping help?" directly — `jsd_canonical_vs_random` and
`jsd_canonical_retest` — instead of contaminating the estimand.

Battery A1 grows from 33,252 to 39,120 items (K=10 permutations, 6 retests,
2 canonical, 2 shuffled per case).

## C8 — All-zero plausibility vectors are a belief, not a parse failure

Under the 0–100 plausibility format, `qwen3-4b-think` returned an all-zero
score vector on 7.4% of A1 items, which the parser rejected as invalid. That
was the wrong classification. An all-zero plausibility vector is well-formed
and expresses *no preference among the candidates*, which after normalisation
is exactly the uniform posterior — the same degenerate belief C6 already
accounts for. Rejecting it dropped the item entirely and unbalanced the
per-arm sample.

It is now parsed as uniform and counted as non-informative. Schema validity
returns to ~100% and the informative-rate gate does the work it was
introduced to do. The rate was in any case arm-balanced (permutation 92.7%,
retest 92.9%, canonical 93.6%, shuffled 90.6%), so it was not a threat to the
estimand — but the accounting should still be right.

Results already generated are recovered rather than re-run: the runner keeps
the raw text of every failure, and `posterior_matrix` re-parses those rows.

## C9 — Two defects in Table 1, the paper's central comparison

**Ensemble sizes were silently clamped.** The residual order sensitivity of a
K-permutation ensemble is measured between two *disjoint* ensembles, so it
needs 2K orderings. With a 10-permutation pool, `K=10` was clamped to 5, and
the K=5 and K=10 rows of Table 1 were the same number printed twice. Reported
sizes are now K ∈ {2, 3, 5}, all satisfying 2K ≤ 10. The residual falls
monotonically — 0.0731, 0.0517, 0.0331 — which is the O(1/√K) behaviour the
argument against permutation ensembling depends on.

**The comparison was unpaired.** Methods were scored on whatever cases they
happened to produce: `perm_ensemble_5` on 1,113 cases against `direct` on
1,955. A method that fails on hard cases is then scored on an easier subset
and looks better for it. Every method is now restricted to the intersection
of cases all of them produced.
