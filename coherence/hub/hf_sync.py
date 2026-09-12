"""Back everything up to the Hugging Face Hub.

Two repositories:

  <user>/codx-clinical-coherence   (dataset) -- the CoDx battery, the
      reconstructed knowledge-base artifacts, the data-audit report and
      figures, and every raw and analysed result file.
  <user>/codx-elr-fusion           (model)   -- the LoRA adapter that trains
      schema-compliant likelihood-ratio emission, with its training config.

Raw patient text never leaves the machine: DDXPlus is CC-BY-4.0 and public,
but the MIMIC arm is credentialed, so `sync_results` refuses to upload
anything under a path containing "mimic". That check is deliberate and
belt-and-braces -- the MIMIC pipeline writes only derived statistics anyway.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from coherence.config import (BUILD, FIGURES, HF_DATASET_REPO, HF_MODEL_REPO,
                              HF_TOKEN, HF_USER, KB, PARQUET, REPORTS, RESULTS,
                              ROOT)

FORBIDDEN = ("mimic", "physionet", "discharge.csv", "radiology.csv")

# Large derived artifacts that are regenerable but worth keeping with the
# release so a reader does not have to rebuild them.
CODE_PATTERNS = ["coherence/**", "scripts/**", "pyproject.toml", "README.md",
                 ".python-version"]


def _api() -> HfApi:
    if not HF_TOKEN:
        raise RuntimeError("no HF token; set `hf=` in .env")
    return HfApi(token=HF_TOKEN)


def repo_id(kind: str = "dataset") -> str:
    api = _api()
    user = HF_USER or api.whoami()["name"]
    return f"{user}/{HF_DATASET_REPO if kind == 'dataset' else HF_MODEL_REPO}"


def ensure_repo(kind: str = "dataset", private: bool = True) -> str:
    rid = repo_id(kind)
    create_repo(rid, repo_type=kind, private=private, exist_ok=True, token=HF_TOKEN)
    return rid


def _check_safe(paths: list[Path]) -> None:
    bad = [p for p in paths
           if any(f in str(p).lower() for f in FORBIDDEN)]
    if bad:
        raise RuntimeError(
            f"refusing to upload {len(bad)} credentialed-data path(s), e.g. {bad[0]}")


def upload_folder(local: Path, path_in_repo: str, kind: str = "dataset",
                  allow_patterns: list[str] | None = None,
                  message: str = "sync") -> str:
    local = Path(local)
    if not local.exists():
        return f"skip (missing): {local}"
    # Only the files that will actually be sent are screened, so a sibling
    # MIMIC directory under the same parent does not block an otherwise
    # legitimate upload -- but anything matching FORBIDDEN that WOULD be sent
    # still aborts the call.
    import fnmatch

    candidates = [q for q in local.rglob("*") if q.is_file()]
    if allow_patterns:
        candidates = [q for q in candidates
                      if any(fnmatch.fnmatch(q.name, pat) for pat in allow_patterns)]
    _check_safe(candidates)
    rid = ensure_repo(kind)
    _api().upload_folder(
        repo_id=rid, repo_type=kind, folder_path=str(local),
        path_in_repo=path_in_repo, allow_patterns=allow_patterns,
        commit_message=message,
    )
    return f"{rid}:{path_in_repo}"


def upload_file(local: Path, path_in_repo: str, kind: str = "dataset",
                message: str = "sync") -> str:
    local = Path(local)
    if not local.exists():
        return f"skip (missing): {local}"
    _check_safe([local])
    rid = ensure_repo(kind)
    _api().upload_file(repo_id=rid, repo_type=kind, path_or_fileobj=str(local),
                       path_in_repo=path_in_repo, commit_message=message)
    return f"{rid}:{path_in_repo}"


DATASET_CARD = """---
license: cc-by-4.0
task_categories:
  - text-classification
tags:
  - clinical-reasoning
  - uncertainty-quantification
  - probabilistic-coherence
  - differential-diagnosis
  - llm-evaluation
---

# CoDx — Coherence in Differential diagnosis

Artifacts for *Probabilistic Coherence as a Test of Clinical Reasoning in
Large Language Models: Measurement and Neuro-Symbolic Correction*.

The instrument tests four coherence axioms, each with an exact zero and none
requiring human annotation:

| axiom | property | needs ground truth |
|---|---|---|
| A1 | order invariance | no |
| A2 | update fidelity | yes (empirical-posterior oracle) |
| A3 | redundancy insensitivity | yes (empirical-posterior oracle) |
| A4 | positional anchoring | no |

## Contents

| path | what |
|---|---|
| `battery/` | the CoDx battery: cases and A1-A4 items, versioned |
| `kb/` | reconstructed likelihood tables and the packed evidence matrix |
| `reports/` | the data-audit report, instrument calibration, figures |
| `results/` | raw elicitation outputs and analysed results, per model |

## Two findings from the data audit that matter beyond this paper

1. **DDXPlus's released knowledge base contains no conditional
   probabilities.** All 888 (pathology, evidence) slots in
   `release_conditions.json` are empty objects. Exact posteriors for
   arbitrary evidence subsets cannot be read off it. This project works
   around that with an assumption-free empirical-posterior oracle built by
   conditioning on the 1.29M released patients.

2. **DDXPlus's ground-truth differential is stochastic.** Across 21,004
   replicate pairs -- patients identical in evidence multiset, age, sex,
   initial evidence and true pathology -- the shipped differential differs
   for 25.8%, with mean JSD 0.049 and a top-1 flip in 0.48%. No released
   field explains the difference. Any DDXPlus differential-matching metric
   therefore has a ceiling below 1.0.

See `reports/data_audit_report.md` for all 19 findings.
"""


MODEL_CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3-8B
library_name: peft
tags:
  - lora
  - clinical-reasoning
  - likelihood-ratio
  - probabilistic-coherence
---

# ELR-Fusion likelihood-ratio adapter

LoRA adapter that teaches a model to emit a well-formed, calibrated
log-likelihood-ratio vector for a single clinical finding against 49
candidate pathologies. It is the trained component of **ELR-Fusion**, which
combines per-finding evidence weights symbolically:

```
log-odds(d | E) = log-odds(d) + sum_i log LR(e_i | d)
```

Because the combination is a sum, the resulting posterior is a function of
the evidence *set*. Order invariance is therefore exact by construction, for
every model and every case — verified numerically at machine epsilon (mean
pairwise JSD 5.2e-17 over 200 permutations).

## Why it was trained

Zero-shot likelihood-ratio elicitation is ELR-Fusion's weak point. Measured
on `qwen3-4b-nothink`, zero-shot fusion is exactly order-invariant but *less
accurate* than direct prompting (top-1 0.033 vs 0.057) and markedly
overconfident (entropy 3.11 vs 4.49 bits) — the compounding effect of summing
log-LRs over correlated findings. This adapter targets that deficit.

## Targets and the split that matters

Targets come from a likelihood table reconstructed from all 1,292,579
released DDXPlus patients (binomial SE ~3e-4 per entry). The table estimates
the generator's *per-finding conditionals*; it is deliberately not used as a
posterior, because naive recombination of those weights reproduces the
shipped differentials poorly (see the data audit).

**The train/test split is by finding, not by example.** 103 of 516 findings
are held out entirely — the adapter never sees them in any format or
demographic framing — so the held-out loss measures generalisation to unseen
findings rather than memorisation.

## Training

Qwen3-8B, LoRA r=32 / alpha=64 / dropout=0.05 on all attention and MLP
projections (87.3M trainable, 1.05% of parameters). Loss is masked to the
assistant turn only, so it is not spent reproducing the 49-line candidate
list that prefixes every prompt. Three elicitation formats are trained
jointly — numeric LR, log-odds, and ordinal verbal bands — which is also
ablation #7 of the paper.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "{repo}")
tok = AutoTokenizer.from_pretrained("{repo}")
```

Prompt construction and the symbolic fusion step are in
`coherence/elicit/prompts.py` and `coherence/methods/elr_fusion.py` of the
companion dataset repository.
"""


SOURCE_SUFFIXES = {".py", ".sh", ".toml", ".md", ".cfg", ".yaml", ".yml", ".txt", ""}
CODE_IGNORE = ["**/__pycache__/**", "**/*.pyc", "**/.ipynb_checkpoints/**"]


def sync_code(message: str = "source code") -> str:
    """Push the source tree so the artifacts are reproducible.

    The FORBIDDEN screen is deliberately NOT applied by filename here: the
    MIMIC *code* (`data/mimic.py`, `run/mimic_runner.py`) must ship for the
    external-validity arm to be reproducible, and it contains no patient
    data. What is enforced instead is stricter in the way that matters --
    only source files are uploaded, so no data file can ride along under a
    code path.
    """
    import fnmatch

    def _ignored(rel: str) -> bool:
        return any(fnmatch.fnmatch(rel, pat) for pat in CODE_IGNORE)

    bad = []
    for q in ROOT.rglob("*"):
        if not q.is_file():
            continue
        rel = str(q.relative_to(ROOT))
        if _ignored(rel) or not any(fnmatch.fnmatch(rel, pat) for pat in CODE_PATTERNS):
            continue
        if q.suffix not in SOURCE_SUFFIXES:
            bad.append(q)
    if bad:
        raise RuntimeError(f"non-source file matched a code pattern: {bad[:3]}")
    rid = ensure_repo("dataset")
    _api().upload_folder(
        repo_id=rid, repo_type="dataset", folder_path=str(ROOT),
        path_in_repo="code", allow_patterns=CODE_PATTERNS,
        ignore_patterns=CODE_IGNORE,
        commit_message=message)
    return f"{rid}:code"


def sync_model(adapter_dir: Path | None = None, private: bool = True) -> str:
    """Push the trained LoRA adapter, excluding intermediate checkpoints."""
    from coherence.config import MODELS

    adapter_dir = Path(adapter_dir or (MODELS / "elr-lora-qwen3-8b"))
    if not (adapter_dir / "adapter_model.safetensors").exists():
        return f"skip (no adapter yet): {adapter_dir}"
    rid = ensure_repo("model", private=private)
    card = adapter_dir / "README.md"
    card.write_text(MODEL_CARD.replace("{repo}", rid))
    _api().upload_folder(
        repo_id=rid, repo_type="model", folder_path=str(adapter_dir),
        ignore_patterns=["checkpoint-*/**", "**/optimizer.pt", "**/scheduler.pt",
                         "**/rng_state*.pth", "**/trainer_state.json"],
        commit_message="ELR-Fusion likelihood-ratio adapter")
    return f"{rid}"


def sync_all(private: bool = True, include_raw: bool = True) -> dict:
    out = {}
    rid = ensure_repo("dataset", private=private)
    card = ROOT / "build" / "README_dataset.md"
    card.write_text(DATASET_CARD)
    out["card"] = upload_file(card, "README.md", message="dataset card")
    out["battery"] = upload_folder(BUILD / "battery", "battery",
                                   allow_patterns=["*.json"], message="CoDx battery")
    out["kb"] = upload_folder(KB, "kb", allow_patterns=["*.npz", "*.npy"],
                              message="reconstructed likelihood tables and "
                                      "packed evidence matrix")
    out["parquet"] = upload_folder(BUILD / "parquet", "ddxplus_parquet",
                                   allow_patterns=["*.parquet"],
                                   message="typed DDXPlus corpus (all three splits)")
    out["reports"] = upload_folder(REPORTS, "reports",
                                   allow_patterns=["*.md", "*.png", "*.pdf", "*.csv"],
                                   message="audit report and figures")
    out["train_data"] = upload_folder(BUILD / "train", "train_data",
                                      allow_patterns=["*.jsonl", "*.json"],
                                      message="likelihood-ratio SFT set")
    out["code"] = sync_code()
    if include_raw:
        out["results"] = upload_folder(RESULTS, "results",
                                       allow_patterns=["*.parquet", "*.json", "*.csv"],
                                       message="elicitation results")
    out["model"] = sync_model(private=private)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--no-raw", action="store_true")
    a = ap.parse_args()
    for k, v in sync_all(private=not a.public, include_raw=not a.no_raw).items():
        print(f"  {k:<10s} {v}")


if __name__ == "__main__":
    main()
