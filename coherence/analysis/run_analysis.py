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
from coherence.methods.elr_fusion import (evidence_logsum, fit_tau, fuse,
                                          fuse_tau, verify_invariance)
from coherence.analysis.competence import competence_table
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
            "jsd_shuffled_diagnostic": r.jsd_shuffled,
            "jsd_between_case_ceiling": r.jsd_between_case,
            "order_effect": r.order_effect,
            "ci_lo": r.order_effect_ci[0], "ci_hi": r.order_effect_ci[1],
            "informative_rate": r.informative_rate,
            "informative_perm": r.informative_rate_by_arm.get("permutation"),
            "informative_retest": r.informative_rate_by_arm.get("retest"),
            "informative_shuffled": r.informative_rate_by_arm.get("shuffled"),
            "n_cases_informative": r.n_cases_informative,
            "order_effect_informative": r.order_effect_informative,
            "oei_ci_lo": r.order_effect_informative_ci[0],
            "oei_ci_hi": r.order_effect_informative_ci[1],
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
                     "direction_agreement": r.direction_agreement,
                     "tie_rate": r.tie_rate})
        s = a2_update.by_stratum(pts, "context_size").with_columns(pl.lit(m).alias("model"))
        strata.append(s)
    return {"main": pl.DataFrame(rows),
            "by_context_size": pl.concat(strata) if strata else pl.DataFrame()}


def analyse_a3(models: list[str], b: Battery,
               floors: dict[str, float] | None = None) -> pl.DataFrame:
    floors = floors or {}
    rows = []
    for m in models:
        d = load_task(m, "a3_posterior")
        if d is None:
            continue
        P, _ = posterior_matrix(d)
        r = a3_redundancy.analyse(m, d, P, b.a3,
                                  retest_floor=floors.get(m, float("nan")))
        rows.append({"model": m, "n_items": r.n_items,
                     "mean_spurious_jsd": r.mean_spurious_jsd,
                     "median_spurious_jsd": r.median_spurious_jsd,
                     "p90_spurious_jsd": r.p90_spurious_jsd,
                     "certified_true_jsd": r.certified_true_jsd,
                     "ratio_to_truth": r.ratio_to_truth,
                     "retest_floor": r.retest_floor,
                     "excess_over_floor": r.excess_over_floor,
                     "frac_above_floor": r.frac_above_floor,
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


def pidx_of(b: Battery) -> dict:
    return {p: i for i, p in enumerate(b.pathologies)}


def _fold_of(case_id: str, n_folds: int = 2) -> int:
    """Stable fold assignment, independent of case order or run."""
    import hashlib

    return int(hashlib.sha1(str(case_id).encode()).hexdigest(), 16) % n_folds


def _fit_tau_folds(b: Battery, wt, priors, emp_prior, pidx, n_folds: int = 2):
    """Cross-fitted shrinkage coefficients, one per fold.

    The tau applied to a case comes from the fold that does NOT contain it, so
    no case contributes to its own coefficient. Returns {fold: tau} plus the
    full grid trace on all cases, which is what the ablation table reports.
    """
    if wt is None or wt.coverage() <= 0:
        return {}, []
    packs = {f: [] for f in range(n_folds)}
    allp = []
    for c in b.cases:
        ti = pidx.get(c.pathology)
        if ti is None:
            continue
        ssum = evidence_logsum(wt, list(c.evidences))
        if np.isscalar(ssum):
            continue
        lp = np.log(np.clip(prior_for(priors, c.age, c.sex,
                                      np.exp(emp_prior)), 1e-9, None))
        packs[_fold_of(c.case_id, n_folds)].append((lp, ssum, ti))
        allp.append((lp, ssum, ti))
    out = {}
    for f in range(n_folds):
        fit = [r for g in range(n_folds) if g != f for r in packs[g]]
        if not fit:
            continue
        out[f], _ = fit_tau([r[0] for r in fit], [r[1] for r in fit],
                            [r[2] for r in fit])
    trace = []
    if allp:
        _, trace = fit_tau([r[0] for r in allp], [r[1] for r in allp],
                           [r[2] for r in allp])
    return out, trace


def elr_tau_sweep(models: list[str], b: Battery, oracle) -> pl.DataFrame:
    """Accuracy / calibration along the shrinkage coefficient.

    Reported because tau is a frontier, not a free parameter with a right
    answer: order-invariance is exact at every tau, but accuracy peaks near
    tau ~ 0.4 while the likelihood of the truth peaks near tau ~ 0.05. Hiding
    that behind a single fitted number would overstate the method.
    """
    pidx = pidx_of(b)
    emp_prior = np.log(np.clip(oracle.prior.probs, 1e-9, None))
    grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
    rows = []
    for m in models:
        wt = build_weight_table(m, "numeric", b.pathologies)
        if wt is None or wt.coverage() <= 0:
            continue
        priors = build_elicited_prior(m, b.pathologies) or {}
        packs = []
        for c in b.cases:
            ti = pidx.get(c.pathology)
            if ti is None:
                continue
            ssum = evidence_logsum(wt, list(c.evidences))
            if np.isscalar(ssum):
                continue
            lp = np.log(np.clip(prior_for(priors, c.age, c.sex,
                                          np.exp(emp_prior)), 1e-9, None))
            packs.append((lp, ssum, ti))
        if not packs:
            continue
        tr = np.asarray([r[2] for r in packs])
        for t in grid:
            M = np.vstack([_softmax(lp + t * ss) for lp, ss, _ in packs])
            rows.append({
                "model": m, "tau": t, "n_cases": len(packs),
                "top1_accuracy": float(np.mean(M.argmax(1) == tr)),
                "top5_accuracy": float(np.mean(
                    [tr[i] in np.argsort(-M[i])[:5] for i in range(len(M))])),
                "nll_of_truth": float(np.mean(
                    [-np.log(max(M[i, tr[i]], 1e-12)) for i in range(len(M))])),
                "mean_entropy_bits": float(np.mean(
                    [-(q[q > 0] * np.log2(q[q > 0])).sum() for q in M])),
                "order_invariant": "exact",
            })
    return pl.DataFrame(rows)


def _softmax(lo: np.ndarray) -> np.ndarray:
    lo = lo - lo.max()
    p = np.exp(lo)
    return p / p.sum()


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

        # ---- shrinkage on the evidence term -------------------------------
        # tau is fitted by CROSS-FITTING over two folds of cases: the tau used
        # for a case is always fitted on the fold that excludes it. That keeps
        # every case in the paired comparison (a held-out split would shrink
        # the common set for every other method too) while never letting a
        # case inform its own coefficient.
        tau_fold, tau_trace = _fit_tau_folds(b, wt, priors, emp_prior, pidx_of(b))

        # Ensemble sizes must satisfy 2K <= n_perms, because the residual is
        # measured between two DISJOINT ensembles drawn from the same pool.
        # An earlier version asked for K=10 from a 10-permutation pool and
        # silently clamped it to 5, so the K=5 and K=10 rows of Table 1 were
        # the same number reported twice.
        ens_ks = (2, 3, 5)
        per = {k: [] for k in ["direct", "self_consistency", "elr_fusion",
                               "elr_fusion_calibrated"]
               + [f"perm_ensemble_{k}" for k in ens_ks]}
        inv = {k: [] for k in per}
        truth = {k: [] for k in per}     # index of the sampled true pathology
        ddx = {k: [] for k in per}       # the shipped differential, as a vector
        cases_seen = {k: set() for k in per}
        case_of = {k: [] for k in per}   # parallel to per[k], for pairing
        pidx = {p: i for i, p in enumerate(b.pathologies)}
        n_retest_used = 0
        for case_id, sub in d.group_by("case_id", maintain_order=True):
            case_id = case_id[0] if isinstance(case_id, tuple) else case_id
            c = ci.get(case_id)
            if c is None:
                continue
            perm = P[sub.filter((pl.col("arm") == "permutation") & pl.col("valid"))["row"].to_numpy()]
            ret = P[sub.filter((pl.col("arm") == "retest") & pl.col("valid"))["row"].to_numpy()]
            if len(perm) < 4:
                continue
            ti = pidx.get(c.pathology)
            if ti is None:
                continue
            dv = np.zeros(len(b.pathologies))
            for nm, pr in zip(c.ddx_names, c.ddx_probs):
                if nm in pidx:
                    dv[pidx[nm]] = pr

            def _rec(key, vec, residual):
                per[key].append(vec)
                inv[key].append(residual)
                truth[key].append(ti)
                ddx[key].append(dv)
                cases_seen[key].add(case_id)
                case_of[key].append(case_id)

            case_vecs: dict[str, tuple] = {}

            _rec("direct", perm[0], jsd(perm[0], perm[1]))
            if len(ret) >= 2:
                n_retest_used = max(n_retest_used, len(ret))
                half_r = len(ret) // 2
                _rec("self_consistency", ret.mean(0),
                     jsd(ret[:half_r].mean(0), ret[half_r:2 * half_r].mean(0)))
            for k in ens_ks:
                if 2 * k <= len(perm):
                    a = perm[:k].mean(0)
                    bb = perm[k:2 * k].mean(0)
                    _rec(f"perm_ensemble_{k}", a, jsd(a, bb))
            if wt is not None and wt.coverage() > 0:
                lp = np.log(np.clip(prior_for(priors, c.age, c.sex,
                                              np.exp(emp_prior)), 1e-9, None))
                f = fuse(lp, wt, c.evidences)
                _rec("elr_fusion", f, 0.0)   # theorem; verified separately
                # Same weights, same sum, one scalar different -- so the
                # residual is still exactly 0 by the same argument.
                t = tau_fold.get(_fold_of(case_id))
                if t is not None:
                    _rec("elr_fusion_calibrated",
                         fuse_tau(lp, wt, c.evidences, t), 0.0)

        # Restrict every method to the cases ALL of them produced, so the
        # comparison is paired. Without this, a method that fails on hard
        # cases is scored on an easier subset and looks better for it: the
        # unpaired run had perm_ensemble_5 on 1,113 cases against direct on
        # 1,955, and they are not comparable numbers.
        populated = [k for k, v in per.items() if v]
        common = set.intersection(*[cases_seen[k] for k in populated]) if populated else set()
        for key in populated:
            keep = [i for i, cid in enumerate(case_of[key]) if cid in common]
            if not keep:
                continue
            mats = [per[key][i] for i in keep]
            M = np.vstack(mats)
            tr = np.asarray([truth[key][i] for i in keep])
            dd = np.vstack([ddx[key][i] for i in keep])
            inv_k = [inv[key][i] for i in keep]
            top1 = float(np.mean(M.argmax(axis=1) == tr))
            top5 = float(np.mean([tr[i] in np.argsort(-M[i])[:5] for i in range(len(M))]))
            # Cost is the query count a method needs for ONE new case.
            # ELR-Fusion is reported twice: its weight table is elicited once
            # per finding for the whole corpus, so a new case costs zero new
            # queries -- but the unamortised figure is given too, because the
            # amortisation only helps when the finding vocabulary is closed.
            cost = {"direct": 1.0, "self_consistency": float(n_retest_used),
                    "elr_fusion": 0.0, "elr_fusion_calibrated": 0.0}.get(key)
            if cost is None:
                cost = float(key.rsplit("_", 1)[1])
            rows.append({
                "model": m, "method": key, "n_cases": len(mats),
                "residual_order_sensitivity_jsd": float(np.mean(inv_k)),
                "order_invariant": "exact" if key.startswith("elr_fusion")
                else ("approx" if key.startswith("perm_ensemble") else "no"),
                "queries_per_new_case": cost,
                "auditable_per_finding": key.startswith("elr_fusion"),
                "top1_accuracy": top1, "top5_accuracy": top5,
                "jsd_to_shipped_ddx": float(np.mean(
                    [jsd(dd[i], M[i]) for i in range(len(M))])) if dd is not None else None,
                "mean_entropy_bits": float(np.mean(
                    [-(p[p > 0] * np.log2(p[p > 0])).sum() for p in M])),
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
    floors = {r["model"]: r["jsd_retest_floor"]
              for r in out["a1_main"].iter_rows(named=True)} if out["a1_main"].height else {}
    out["a3_main"] = analyse_a3(models, b, floors)
    out["a4_main"] = analyse_a4(models, b)
    out["methods"] = analyse_methods(models, b, oracle)
    out["elr_tau_sweep"] = elr_tau_sweep(models, b, oracle)
    out["schema_validity"] = pl.DataFrame(
        [{"model": m, **schema_validity(m)} for m in models])
    # Competence gates coherence: an order-effect computed over responses that
    # do not track the patient is a measurement of noise, however clean its
    # schema validity looks. See coherence/analysis/competence.py.
    out["competence"] = competence_table(models, b)

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
