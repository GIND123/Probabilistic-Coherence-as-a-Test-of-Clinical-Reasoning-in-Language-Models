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
