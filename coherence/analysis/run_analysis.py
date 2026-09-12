"""Turn raw elicitation output into the paper's tables and figures.

    .venv/bin/python -m coherence.analysis.run_analysis

Runs on whatever models have completed; re-running after more models finish
simply widens the tables. Every divergence is reported against three anchors:
the model's own test-retest floor, the shuffled-evidence ceiling, and the
generator's replicate noise floor from the data audit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from coherence.axioms import a1_order, a2_update, a3_redundancy, a4_anchoring
from coherence.axioms.loading import (available_models, load_task,
                                      posterior_matrix, schema_validity)
from coherence.config import RESULTS, TABLES
from coherence.data.battery import Battery
from coherence.data.ddxplus import load_kb
from coherence.elicit.registry import DEFAULT_ORDER
from coherence.methods import baselines
from coherence.methods.elr_fusion import fuse, verify_invariance
from coherence.analysis.elr_build import (build_elicited_prior, build_weight_table,
                                          prior_for)

BATTERY = Path("build/battery/codx_battery_v1.json")
ANALYSED = RESULTS / "analysed"
ANALYSED.mkdir(parents=True, exist_ok=True)


def _order(models: list[str]) -> list[str]:
    rank = {m: i for i, m in enumerate(DEFAULT_ORDER)}
    return sorted(models, key=lambda m: rank.get(m, 999))


def _audit_noise_floor() -> float:
    p = RESULTS / "audit_findings.json"
    if not p.exists():
        return float("nan")
    for f in json.loads(p.read_text()):
        if f["check"] == "generator_determinism":
            return float(f["stats"]["jsd_mean"])
    return float("nan")


# --------------------------------------------------------------------------
def analyse_a1(models: list[str], b: Battery, kb) -> dict:
    rows, ksweep, sevrows, percase = [], [], [], []
    for m in models:
        d = load_task(m, "a1_posterior")
        if d is None:
            continue
        P, _ = posterior_matrix(d)
        r = a1_order.analyse(m, d, P)
        rows.append({
            "model": m, "n_cases": r.n_cases,
            "jsd_permutation": r.jsd_permutation, "jsd_retest_floor": r.jsd_retest,
            "jsd_shuffled_ceiling": r.jsd_shuffled,
            "order_effect": r.order_effect,
            "ci_lo": r.order_effect_ci[0], "ci_hi": r.order_effect_ci[1],
            "normalised_order_effect": r.normalised_order_effect,
            "top1_flip_rate": r.top1_flip_rate, "top5_jaccard": r.top5_jaccard,
            "mean_entropy_bits": r.mean_entropy,
        })
        ksweep.append(a1_order.k_sweep(m, d, P))
        sev = a1_order.severity_weighted(r.per_case, P, d, kb.severity, b.pathologies)
        sevrows.append({"model": m, **sev})
        percase.append(r.per_case.with_columns(pl.lit(m).alias("model")))
    return {
        "main": pl.DataFrame(rows),
        "k_sweep": pl.concat(ksweep) if ksweep else pl.DataFrame(),
        "severity": pl.DataFrame(sevrows),
        "per_case": pl.concat(percase, how="diagonal") if percase else pl.DataFrame(),
    }


def analyse_a2(models: list[str], b: Battery) -> dict:
    rows, strata = [], []
    for m in models:
        d = load_task(m, "a2_posterior")
        if d is None:
            continue
        P, _ = posterior_matrix(d)
        pts = a2_update.build_points(d, P, b.a2, b.pathologies)
        if pts.height < 100:
            continue
        r = a2_update.analyse(m, pts)
        rows.append({"model": m, "n_items": r.n_items, "n_points": r.n_points,
                     "beta": r.beta, "beta_lo": r.beta_ci[0], "beta_hi": r.beta_ci[1],
                     "intercept": r.intercept, "r2": r.r2, "spearman": r.spearman,
                     "mean_abs_error": r.mean_abs_error,
                     "direction_agreement": r.direction_agreement})
        s = a2_update.by_stratum(pts, "context_size").with_columns(pl.lit(m).alias("model"))
        strata.append(s)
    return {"main": pl.DataFrame(rows),
            "by_context_size": pl.concat(strata) if strata else pl.DataFrame()}


def analyse_a3(models: list[str], b: Battery) -> pl.DataFrame:
    rows = []
    for m in models:
        d = load_task(m, "a3_posterior")
        if d is None:
            continue
        P, _ = posterior_matrix(d)
        r = a3_redundancy.analyse(m, d, P, b.a3)
        rows.append({"model": m, "n_items": r.n_items,
                     "mean_spurious_jsd": r.mean_spurious_jsd,
                     "median_spurious_jsd": r.median_spurious_jsd,
                     "p90_spurious_jsd": r.p90_spurious_jsd,
                     "certified_true_jsd": r.certified_true_jsd,
                     "ratio_to_truth": r.ratio_to_truth,
                     "top1_flip_rate": r.top1_flip_rate,
                     "mean_entropy_change": r.mean_entropy_change,
                     "confidence_inflation_rate": r.confidence_inflation_rate})
    return pl.DataFrame(rows)


def analyse_a4(models: list[str], b: Battery) -> pl.DataFrame:
    rows = []
    for m in models:
        d = load_task(m, "a4_posterior")
        if d is None:
            continue
        P, _ = posterior_matrix(d)
        r = a4_anchoring.analyse(m, d, P, b.pathologies)
        rows.append({"model": m, "n_cases": r.n_cases,
                     "beta_position": r.beta_position,
                     "ci_lo": r.beta_ci[0], "ci_hi": r.beta_ci[1],
                     "eta_squared": r.eta_squared,
                     "jsd_between_positions": r.jsd_between_positions,
                     "jsd_within_position": r.jsd_within_position,
                     "position_effect": r.position_effect,
                     "top1_flip_rate": r.top1_flip_rate})
    return pl.DataFrame(rows)


def analyse_methods(models: list[str], b: Battery, oracle) -> pl.DataFrame:
    """Table 1: the central comparison, including permutation ensembling."""
    from coherence.metrics.divergence import jsd

    ci = {c.case_id: c for c in b.cases}
    uniform = np.full(len(b.pathologies), 1.0 / len(b.pathologies))
    rows = []
    for m in models:
        d = load_task(m, "a1_posterior")
        if d is None:
            continue
        P, _ = posterior_matrix(d)
        d = d.with_row_index("row")
        wt = build_weight_table(m, "numeric", b.pathologies)
        priors = build_elicited_prior(m, b.pathologies) or {}
        emp_prior = np.log(np.clip(oracle.prior.probs, 1e-9, None))

        per = {k: [] for k in ("direct", "self_consistency", "perm_ensemble_5",
                               "perm_ensemble_10", "elr_fusion")}
        inv = {k: [] for k in per}
        for case_id, sub in d.group_by("case_id", maintain_order=True):
            case_id = case_id[0] if isinstance(case_id, tuple) else case_id
            c = ci.get(case_id)
            if c is None:
                continue
            perm = P[sub.filter((pl.col("arm") == "permutation") & pl.col("valid"))["row"].to_numpy()]
            ret = P[sub.filter((pl.col("arm") == "retest") & pl.col("valid"))["row"].to_numpy()]
            if len(perm) < 4:
                continue
            half = len(perm) // 2
            per["direct"].append(perm[0])
            inv["direct"].append(jsd(perm[0], perm[1]))
            if len(ret) >= 2:
                per["self_consistency"].append(ret.mean(0))
                inv["self_consistency"].append(jsd(perm[0], ret.mean(0)))
            for k, key in ((min(5, half), "perm_ensemble_5"),
                           (min(10, half), "perm_ensemble_10")):
                if k >= 2:
                    a = perm[:k].mean(0)
                    bb = perm[half:half + k].mean(0)
                    per[key].append(a)
                    inv[key].append(jsd(a, bb))
            if wt is not None and wt.coverage() > 0:
                lp = np.log(np.clip(prior_for(priors, c.age, c.sex,
                                              np.exp(emp_prior)), 1e-9, None))
                f = fuse(lp, wt, c.evidences)
                per["elr_fusion"].append(f)
                inv["elr_fusion"].append(0.0)   # theorem; verified separately

        for key, mats in per.items():
            if not mats:
                continue
            rows.append({
                "model": m, "method": key, "n_cases": len(mats),
                "residual_order_sensitivity_jsd": float(np.mean(inv[key])),
                "mean_entropy_bits": float(np.mean(
                    [-(p[p > 0] * np.log2(p[p > 0])).sum() for p in mats])),
            })
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
def main() -> None:
    b = Battery.load(BATTERY)
    kb = load_kb()
    models = _order(available_models("a1_posterior"))
    if not models:
        print("no completed models yet")
        return
    print(f"analysing {len(models)} model(s): {', '.join(models)}")

    from coherence.data.empirical_oracle import EmpiricalOracle
    from coherence.data.kb_reconstruct import LikelihoodTable
    from coherence.data.ddxplus import load_all

    tbl = LikelihoodTable.load()
    oracle = EmpiricalOracle.build(load_all(), tbl.tokens, tbl.pathologies)

    out: dict[str, pl.DataFrame] = {}
    a1 = analyse_a1(models, b, kb)
    out["a1_main"] = a1["main"]
    out["a1_k_sweep"] = a1["k_sweep"]
    out["a1_severity"] = a1["severity"]
    a1["per_case"].write_parquet(ANALYSED / "a1_per_case.parquet")
    out["a2_main"] = analyse_a2(models, b)["main"]
    out["a3_main"] = analyse_a3(models, b)
    out["a4_main"] = analyse_a4(models, b)
    out["methods"] = analyse_methods(models, b, oracle)
    out["schema_validity"] = pl.DataFrame(
        [{"model": m, **schema_validity(m)} for m in models])

    nf = _audit_noise_floor()
    for name, t in out.items():
        if t is None or not t.height:
            continue
        t.write_csv(TABLES / f"{name}.csv")
        print(f"\n### {name}   (generator noise floor JSD = {nf:.4f})")
        with pl.Config(tbl_formatting="ASCII_MARKDOWN", tbl_hide_dataframe_shape=True,
                       tbl_hide_column_data_types=True, tbl_rows=40,
                       tbl_width_chars=250, float_precision=4):
            print(t)


if __name__ == "__main__":
    main()
