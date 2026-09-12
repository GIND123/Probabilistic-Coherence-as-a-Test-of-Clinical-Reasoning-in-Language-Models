# Probabilistic Coherence as a Test of Clinical Reasoning in Language Models

**CoDx** — the Coherence in Differential diagnosis battery (the instrument)
**ELR-Fusion** — Elicited Likelihood-Ratio fusion (the method)

Bayesian belief updating over a fixed evidence set is order-invariant:
`P(d | e₁, e₂)` is a function of the *set* `{e₁, e₂}`, not the sequence. So if
a model's differential changes when the same findings are presented in a
different order, that is a reasoning failure with an exact zero — no
annotator, no LLM judge, no threshold. The primary measurement needs **no
labels at all**, which is what lets it run on real clinical narratives.

## Layout

| path | what |
|---|---|
| `coherence/data/` | DDXPlus ingest, KB parsing, the empirical-posterior oracle, battery construction, MIMIC extraction |
| `coherence/audit/` | the 19-check data audit and its figures |
| `coherence/elicit/` | prompts, schemas, clinical rendering, vLLM engine, model registry |
| `coherence/calibration/` | instrument calibration experiments |
| `coherence/axioms/` | A1–A4 analysers |
| `coherence/methods/` | ELR-Fusion and the baselines |
| `coherence/run/` | job generation and the resumable runners |
| `coherence/analysis/` | results tables and figures |
| `coherence/train/` | LoRA training for likelihood-ratio emission |
| `coherence/hub/` | Hugging Face backup |
| `reports/` | the data-audit report, instrument calibration, figures, tables |

## Running

```bash
bash scripts/setup_env.sh              # once
.venv/bin/python -m coherence.audit.run_audit
.venv/bin/python -m coherence.data.battery
bash scripts/run_all.sh                # sweep -> sync -> train -> analyse
```

Everything is resumable. The sweep skips any `(model, task)` whose parquet
already exists; within a task it resumes from the last completed chunk.

## Two findings from the data audit that matter beyond this paper

**1. DDXPlus's released knowledge base contains no conditional
probabilities.** All 888 `(pathology, evidence)` slots in
`release_conditions.json` are empty objects, so exact posteriors for
arbitrary evidence subsets cannot be read off it. Reconstructing them by
naive Bayes from corpus frequencies fails badly — JSD 0.546 against the
shipped differentials, roughly 11× the generator's own noise — and the
failure is the *recombination rule*, not the estimation: conditional
independence given the pathology is violated on 8.7% of finding pairs at
|φ| > 0.1. This project instead conditions directly on the 1.29M released
patients, which is assumption-free and stays usable to |E| ≈ 6.

**2. DDXPlus's ground-truth differential is stochastic.** Across 21,004
replicate pairs — patients identical in evidence multiset, age, sex, initial
evidence and true pathology — the shipped differential differs for **25.8%**,
mean JSD 0.049, with the top-1 diagnosis flipping in 0.48%. No released field
explains the difference. Any DDXPlus differential-matching metric therefore
has a ceiling below 1.0.

See [reports/data_audit_report.md](reports/data_audit_report.md).

## The instrument had to be calibrated before it could be trusted

Four defects were found and fixed, each of which would have produced a
plausible-looking but wrong headline number. They are documented in
[reports/instrument_calibration.md](reports/instrument_calibration.md):

* **Schema failure that depended on the condition.** Failure rates ran 20% on
  the shuffled arm against 10% on retest, so discarding failures would have
  biased the order effect by the ordering condition itself.
* **"100% schema validity" was hollow.** 62.9% of responses were *exactly
  uniform* — parseable, but carrying no belief and trivially order-invariant.
  Fixed by eliciting 0–100 plausibility instead of a normalised distribution,
  and by replacing the validity gate with an **informative-rate** gate
  reported per arm.
* **An inverted ceiling.** The shuffled-evidence arm sat *below* the retest
  floor, because models answer nonsense with a default posterior. Replaced
  with a between-patient ceiling; the shuffled arm is kept as a diagnostic.
* **A contaminated floor.** DDXPlus releases findings code-sorted, which
  groups related questions and makes models ~6 pp more willing to commit. The
  retest arm replicated that canonical order while the permutation arm did
  not. The canonical order is now its own arm.

## Hardware

One NVIDIA RTX PRO 5000 Blackwell, 48 GB (sm_120), 24 cores, 123 GB RAM.
27B–32B models run 4-bit (AWQ/GPTQ), which is the deployment regime the paper
is about. `gpt-oss-120b` does not fit — its MXFP4 weights are ~63 GB — so it
runs CPU-offloaded on the A1 core subset only and is flagged as a
reduced-scale arm in every table.

## Data governance

DDXPlus, MedQA and MedR-Bench are public. MIMIC-IV-Note is credentialed: it
stays on this machine, and `coherence.hub.hf_sync` hard-refuses any path
containing `mimic` or `physionet`. Only aggregate divergence statistics are
published. Frontier API models are never pointed at the MIMIC arm.

## Backup

`GOVINDFROM/codx-clinical-coherence` (dataset) — battery, KB artifacts,
reports, figures, results.
`GOVINDFROM/codx-elr-fusion` (model) — the LoRA adapter and its config.
