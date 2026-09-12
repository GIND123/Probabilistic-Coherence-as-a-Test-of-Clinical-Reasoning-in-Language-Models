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
                              HF_TOKEN, HF_USER, KB, REPORTS, RESULTS, ROOT)

FORBIDDEN = ("mimic", "physionet", "discharge.csv", "radiology.csv")


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
    _check_safe(list(local.rglob("*")))
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


def sync_all(private: bool = True, include_raw: bool = True) -> dict:
    out = {}
    rid = ensure_repo("dataset", private=private)
    card = ROOT / "build" / "README_dataset.md"
    card.write_text(DATASET_CARD)
    out["card"] = upload_file(card, "README.md", message="dataset card")
    out["battery"] = upload_folder(BUILD / "battery", "battery",
                                   allow_patterns=["*.json"], message="CoDx battery")
    out["kb"] = upload_folder(KB, "kb", allow_patterns=["*.npz"],
                              message="reconstructed likelihood tables")
    out["reports"] = upload_folder(REPORTS, "reports",
                                   allow_patterns=["*.md", "*.png", "*.pdf", "*.csv"],
                                   message="audit report and figures")
    if include_raw:
        out["results"] = upload_folder(RESULTS, "results",
                                       allow_patterns=["*.parquet", "*.json", "*.csv"],
                                       message="elicitation results")
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
