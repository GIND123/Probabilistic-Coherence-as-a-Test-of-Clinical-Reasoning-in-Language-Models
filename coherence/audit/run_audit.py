"""Run the full DDXPlus data audit and emit the report.

    .venv/bin/python -m coherence.audit.run_audit
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import polars as pl

from coherence.audit import checks as C
from coherence.audit import checks_dist as CD
from coherence.audit import checks_oracle as CO
from coherence.config import REPORTS, RESULTS, TABLES
from coherence.data.ddxplus import (kb_has_conditional_probabilities, load_all,
                                    load_kb, load_split)
from coherence.data.empirical_oracle import EmpiricalOracle
from coherence.data.kb_reconstruct import (LikelihoodTable, build_likelihood_table,
                                           decide_regime, test_conditional_independence,
                                           validate_against_shipped)

SEV_ORDER = {"critical": 0, "major": 1, "minor": 2, "info": 3}
SEV_BADGE = {"critical": "CRITICAL", "major": "MAJOR", "minor": "MINOR", "info": "INFO"}


def _provenance() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except Exception:
            return "n/a"
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": sh("git rev-parse --short HEAD"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": sh("nvidia-smi --query-gpu=name,memory.total,driver_version "
                  "--format=csv,noheader"),
        "polars": pl.__version__,
        "numpy": np.__version__,
    }


def run() -> list[C.Finding]:
    kb = load_kb()
    df = load_all()
    findings: list[C.Finding] = []

    # --- week-1 gate --------------------------------------------------
    has_probs, nonempty, total = kb_has_conditional_probabilities()
    findings.append(CO.check_kb_probability_release(nonempty, total))

    # --- corpus integrity ---------------------------------------------
    findings.append(C.check_schema_and_nulls(df))
    findings.append(C.check_duplicates(df))
    findings.append(C.check_demographics(df))
    findings.append(C.check_evidence_tokens(df, kb))

    # --- label integrity ----------------------------------------------
    findings.append(C.check_differential_wellformed(df))
    findings.append(C.check_true_pathology_in_differential(df))

    # --- knowledge-base consistency -----------------------------------
    findings.append(C.check_kb_coverage(df, kb))

    # --- determinism / irreducible noise ------------------------------
    det = C.check_generator_determinism(df, kb)
    findings.append(det)
    noise_floor = det.stats["jsd_mean"]

    # --- distributional -----------------------------------------------
    findings.append(CD.check_class_balance(df, kb))
    findings.append(CD.check_evidence_load(df))
    findings.append(CD.check_split_shift(df))
    findings.append(CD.check_severity_coverage(df, kb))
    findings.append(CD.check_sex_plausibility(df, kb))
    findings.append(CD.check_age_plausibility(df, kb))

    # --- ground-truth construction for A2/A3 --------------------------
    table = build_likelihood_table(df, kb)
    table.save()
    validation = validate_against_shipped(table, df, n_cases=20000)
    independence = test_conditional_independence(df, table, n_pairs=4000)
    findings.append(CO.check_naive_bayes_reconstruction(validation, noise_floor, independence))

    oracle = EmpiricalOracle.build(df, table.tokens, table.pathologies)
    findings.append(CO.check_empirical_oracle_feasibility(oracle, df))
    findings.append(CO.check_oracle_vs_shipped(oracle, df))

    regime = decide_regime(validation, noise_floor)
    findings.append(C.Finding(
        "design_decision",
        "info",
        "A2/A3 operate in the ABSOLUTE regime using the empirical-posterior oracle; "
        "naive-Bayes reconstruction is retained only as an approximate baseline "
        f"(it alone would have forced the {regime} regime)",
        detail=(
            "A1 (order invariance) and A4 (positional anchoring) require no ground truth and "
            "are unaffected by any of this. A2 (update fidelity) and A3 (redundancy "
            "insensitivity) use the empirical oracle with a 500-patient minimum support and "
            "Wilson intervals on every target, restricted to evidence subsets of size <= 6. "
            "Every divergence figure in the paper is reported against three anchors: the "
            "model's test-retest floor, the generator's own replicate noise floor measured "
            f"here ({noise_floor:.4f} JSD), and a shuffled-evidence ceiling."
        ),
        stats={"regime_naive_bayes_would_force": regime,
               "generator_noise_floor_jsd": noise_floor},
    ))
    return findings


def to_markdown(findings: list[C.Finding], prov: dict) -> str:
    order = sorted(findings, key=lambda f: (SEV_ORDER[f.severity], f.check))
    counts = {s: sum(1 for f in findings if f.severity == s) for s in C.SEVERITIES}
    L = []
    A = L.append
    A("# DDXPlus Data Audit Report")
    A("")
    A("**Project:** CoDx — Probabilistic Coherence as a Test of Clinical Reasoning in LLMs  ")
    A(f"**Generated:** {prov['generated_utc']}  ")
    A(f"**Corpus:** DDXPlus, all three released splits  ")
    A(f"**Hardware:** {prov['gpu']}")
    A("")
    A("## Summary")
    A("")
    A("| Severity | Count |")
    A("|---|---|")
    for s in C.SEVERITIES:
        A(f"| {SEV_BADGE[s]} | {counts[s]} |")
    A("")
    A("| # | Check | Severity | Finding |")
    A("|---|---|---|---|")
    for i, f in enumerate(order, 1):
        A(f"| {i} | `{f.check}` | {SEV_BADGE[f.severity]} | {f.headline} |")
    A("")
    A("---")
    A("")
    A("## Findings in detail")
    for i, f in enumerate(order, 1):
        A("")
        A(f"### {i}. `{f.check}` — {SEV_BADGE[f.severity]}")
        A("")
        A(f"{f.headline}")
        if f.detail:
            A("")
            A(f.detail)
        if f.stats:
            A("")
            A("```json")
            A(json.dumps(f.stats, indent=2, default=str)[:4000])
            A("```")
        if f.table is not None and f.table.height:
            A("")
            with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True,
                           tbl_hide_column_data_types=True, tbl_rows=25, tbl_width_chars=200):
                A(str(f.table.head(25)))
    A("")
    A("---")
    A("")
    A("## Provenance")
    A("")
    A("```json")
    A(json.dumps(prov, indent=2))
    A("```")
    return "\n".join(L)


def main() -> None:
    findings = run()
    prov = _provenance()
    md = to_markdown(findings, prov)
    out = REPORTS / "data_audit_report.md"
    out.write_text(md)
    (RESULTS / "audit_findings.json").write_text(json.dumps(
        [{"check": f.check, "severity": f.severity, "headline": f.headline,
          "detail": f.detail, "stats": f.stats} for f in findings],
        indent=2, default=str))
    for f in findings:
        if f.table is not None and f.table.height:
            f.table.write_csv(TABLES / f"audit_{f.check}.csv")
    print(f"wrote {out} ({len(md):,} chars), {len(findings)} findings")
    for f in sorted(findings, key=lambda x: SEV_ORDER[x.severity]):
        print(f"  [{f.severity:<8s}] {f.check}")


if __name__ == "__main__":
    main()
