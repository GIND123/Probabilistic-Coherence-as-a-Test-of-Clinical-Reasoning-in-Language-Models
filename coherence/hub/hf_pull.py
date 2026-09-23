"""Fetch the released artifacts back out of the Hub, and check they are there.

`hf_sync` is the write half of the release; this is the read half. Without it
the artifacts are backed up but not *reusable*: a reader cloning the GitHub
repo has the code and the tables, and no way to get the 641 MB evidence matrix
or the 23 MB battery that the code expects to find under `build/`.

Two entry points:

  verify   assert every artifact the README and the analysis code depend on is
           present on the Hub, without downloading any of it. Cheap, and the
           thing to run before claiming a release is complete.
  pull     download a named group into the layout `coherence.config` expects,
           so that after pulling, the analysis runs unmodified.

Group names are stable and match the paths in the dataset repo. `pull --all`
is deliberately not the default: `kb` alone is 641 MB and most readers only
want `tables`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from coherence.config import (BUILD, HF_TOKEN, KB, PARQUET, REPORTS, RESULTS,
                              ROOT)

# group -> (path in the dataset repo, local destination, why a reader wants it)
GROUPS: dict[str, tuple[str, Path, str]] = {
    "battery":  ("battery", BUILD / "battery",
                 "the CoDx battery: every case and A1-A4 item, versioned"),
    "kb":       ("kb", KB,
                 "reconstructed likelihood table and packed evidence matrix "
                 "(641 MB; needed only to rebuild the oracle)"),
    "parquet":  ("ddxplus_parquet", PARQUET,
                 "the typed DDXPlus corpus, all three splits"),
    "reports":  ("reports", REPORTS,
                 "tables, figures, audit report, instrument calibration"),
    "results":  ("results", RESULTS,
                 "raw elicitation outputs and analysed results, per model"),
    "train":    ("train_data", BUILD / "train",
                 "the likelihood-ratio SFT set used to train the adapter"),
}

# What a complete release must contain. Checked by `verify`, which is the
# reason this list is explicit rather than derived from whatever happens to be
# on the Hub -- a missing file should fail the check, not shrink the list.
REQUIRED_DATASET = [
    "README.md",
    "battery/codx_battery_v1.json",
    "battery/distractors_v1.json",
    "kb/likelihood_table.npz",
    "kb/evidence_matrix.npy",
    "kb/labels.npy",
    "ddxplus_parquet/ddxplus_train.parquet",
    "ddxplus_parquet/ddxplus_validate.parquet",
    "ddxplus_parquet/ddxplus_test.parquet",
    "train_data/lr_sft_train.jsonl",
    "train_data/lr_sft_test.jsonl",
    "train_data/lr_sft_meta.json",
    "reports/data_audit_report.md",
    "reports/instrument_calibration.md",
]
REQUIRED_MODEL = [
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "final_metrics.json",
]

# Tables the README and writeup.tex cite by name. If one of these is missing
# from the release, a number in the paper has no published provenance.
REQUIRED_TABLES = [
    "a1_main.csv", "a2_main.csv", "a3_main.csv", "a4_main.csv",
    "a1_k_sweep.csv", "a1_severity.csv", "competence.csv", "methods.csv",
    "elr_tau_sweep.csv", "schema_validity.csv",
    "rev_canonical_baseline.csv", "rev_corpus_lr_ablation.csv",
    "rev_logprob_baseline.csv", "rev_a1_absolute_gaps.csv",
    "rev_a2_noise_ceiling.csv", "rev_generator_floor.csv",
]

# Every arm that ran the full battery must have raw results on the Hub.
# gpt-oss-120b is excluded: it was never run (see the registry note).
REQUIRED_RESULT_ARMS = [
    "qwen3-4b-nothink", "qwen3-4b-think", "qwen3-8b-nothink", "qwen3-8b-think",
    "qwen3-32b-nothink", "qwen3-32b-think", "medgemma-1.5-4b", "medgemma-27b",
    "med42-8b", "gpt-oss-20b", "r1-distill-32b", "qwen3-8b-elr-lora",
]


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=HF_TOKEN)


def verify(verbose: bool = True) -> tuple[bool, list[str]]:
    """Check the release is complete. Returns (ok, list of problems)."""
    from coherence.hub.hf_sync import repo_id

    api = _api()
    problems: list[str] = []

    def _check(kind: str, required: list[str]) -> list[str]:
        rid = repo_id(kind)
        try:
            files = set(api.list_repo_files(rid, repo_type=kind))
        except Exception as exc:
            problems.append(f"{kind} repo {rid}: {type(exc).__name__}: {exc}")
            return []
        missing = [f for f in required if f not in files]
        for f in missing:
            problems.append(f"{kind}:{rid} missing {f}")
        if verbose:
            print(f"  {kind:<8s} {rid}  {len(files)} files, "
                  f"{len(required) - len(missing)}/{len(required)} required present")
        return sorted(files)

    ds_files = _check("dataset", REQUIRED_DATASET)
    _check("model", REQUIRED_MODEL)

    if ds_files:
        have = set(ds_files)
        missing_tables = [t for t in REQUIRED_TABLES
                          if f"reports/tables/{t}" not in have]
        for t in missing_tables:
            problems.append(f"dataset missing cited table reports/tables/{t}")
        if verbose:
            print(f"  tables   {len(REQUIRED_TABLES) - len(missing_tables)}/"
                  f"{len(REQUIRED_TABLES)} cited tables present")

        arms_present = {f.split("/")[2] for f in have
                        if f.startswith("results/raw/") and f.count("/") >= 2}
        missing_arms = [a for a in REQUIRED_RESULT_ARMS if a not in arms_present]
        for a in missing_arms:
            problems.append(f"dataset missing raw results for arm {a}")
        if verbose:
            print(f"  results  {len(REQUIRED_RESULT_ARMS) - len(missing_arms)}/"
                  f"{len(REQUIRED_RESULT_ARMS)} battery arms present")

        # The guard in hf_sync should make a credentialed DATA path impossible;
        # assert it anyway, because that is the one failure that deleting the
        # file afterwards does not undo.
        #
        # The MIMIC *source* modules are a deliberate exception and must not
        # trip this: `sync_code` ships them so the external-validity arm is
        # reproducible, and they contain no patient data (only docstrings --
        # re-checked here by suffix, since the policy that makes them safe is
        # "source files only", not "files we have inspected once").
        from coherence.hub.hf_sync import FORBIDDEN, SOURCE_SUFFIXES
        leaked, exempt = [], []
        for f in have:
            if not any(x in f.lower() for x in FORBIDDEN):
                continue
            if f.startswith("code/") and Path(f).suffix in SOURCE_SUFFIXES:
                exempt.append(f)
            else:
                leaked.append(f)
        for f in leaked:
            problems.append(f"CREDENTIALED PATH ON HUB: {f}")
        if verbose:
            print(f"  guard    {len(leaked)} credentialed data path(s) on the Hub"
                  f"{'  <-- INVESTIGATE' if leaked else ''}"
                  f"  ({len(exempt)} MIMIC source module(s) shipped by design)")

    return not problems, problems


def pull(groups: list[str], revision: str | None = None) -> dict[str, str]:
    """Download the named groups into the layout `coherence.config` expects."""
    from huggingface_hub import snapshot_download

    from coherence.hub.hf_sync import repo_id

    rid = repo_id("dataset")
    out = {}
    for g in groups:
        if g not in GROUPS:
            raise KeyError(f"unknown group {g!r}; known: {sorted(GROUPS)}")
        remote, dest, _ = GROUPS[g]
        dest.mkdir(parents=True, exist_ok=True)
        # Download into a cache, then place the subtree at `dest` so the local
        # layout matches what the analysis imports -- callers should not have
        # to know the repo's internal directory names.
        path = snapshot_download(
            repo_id=rid, repo_type="dataset", revision=revision,
            allow_patterns=[f"{remote}/**"], token=HF_TOKEN,
            local_dir=str(BUILD / "_hub_pull"))
        src = Path(path) / remote
        moved = 0
        for f in src.rglob("*"):
            if not f.is_file():
                continue
            target = dest / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(f.read_bytes())
                moved += 1
        out[g] = f"{dest.relative_to(ROOT)}  ({moved} new file(s))"
    return out


def pull_adapter(dest: Path | None = None, revision: str | None = None) -> str:
    """Download the ELR-Fusion LoRA adapter."""
    from huggingface_hub import snapshot_download

    from coherence.config import MODELS
    from coherence.hub.hf_sync import repo_id

    dest = Path(dest or (MODELS / "elr-lora-qwen3-8b"))
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id("model"), repo_type="model",
                      revision=revision, token=HF_TOKEN, local_dir=str(dest))
    return str(dest.relative_to(ROOT))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="fetch released artifacts from the Hub, or verify they exist")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="check the release is complete (no download)")
    sub.add_parser("groups", help="list what can be pulled")

    p = sub.add_parser("pull", help="download artifact groups")
    p.add_argument("groups", nargs="*", default=[],
                   help=f"any of: {', '.join(GROUPS)}")
    p.add_argument("--all", action="store_true")
    p.add_argument("--adapter", action="store_true", help="also pull the LoRA adapter")
    p.add_argument("--revision", default=None,
                   help="pin to a dataset-repo revision (see ARTIFACTS.md)")

    a = ap.parse_args()

    if a.cmd == "groups":
        for g, (remote, dest, why) in GROUPS.items():
            print(f"  {g:<9s} {remote:<16s} -> {dest.relative_to(ROOT)}\n"
                  f"            {why}")
        return

    if a.cmd == "verify":
        ok, problems = verify()
        if ok:
            print("\nrelease is complete.")
            return
        print(f"\n{len(problems)} problem(s):")
        for pr in problems:
            print(f"  - {pr}")
        sys.exit(1)

    groups = list(GROUPS) if a.all else a.groups
    if not groups and not a.adapter:
        ap.error("name at least one group, or --all, or --adapter")
    for g, where in pull(groups, revision=a.revision).items():
        print(f"  {g:<9s} {where}")
    if a.adapter:
        print(f"  {'adapter':<9s} {pull_adapter(revision=a.revision)}")


if __name__ == "__main__":
    main()
