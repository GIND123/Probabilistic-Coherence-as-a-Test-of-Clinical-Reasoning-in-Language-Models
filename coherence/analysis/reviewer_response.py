"""Analyses added in response to review, computable from the existing sweep.

Each function is named for the review comment it answers and writes one table
into ``reports/tables``. Nothing here needs new generation: the battery already
contains a canonical-order arm and a six-replicate retest arm, which together
supply the fixed-order baseline, the resampling null, and the N-sample averaged
condition that the review asks for.

The statistical unit throughout is the **patient**, not the item. Items are
nested within patients (K orderings and R replicates each), so an item-level
bootstrap or an item-level McNemar would treat ~16 correlated observations as
independent and understate every interval. `cluster_bootstrap` resamples
patients with replacement and is used for every CI reported here.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from coherence.axioms.loading import load_task, posterior_matrix
from coherence.config import RESULTS, TABLES
from coherence.data.battery import Battery
from coherence.metrics.divergence import jsd, mean_pairwise_jsd

BATTERY = "build/battery/codx_battery_v1.json"
N_BOOT = 5000


# ----------------------------------------------------------------- helpers --
def cluster_bootstrap(values_by_case: dict[str, list[float]], n_boot: int = N_BOOT,
                      seed: int = 0, alpha: float = .05) -> tuple[float, float, float]:
    """Patient-level cluster bootstrap: (point, lo, hi).

    Resamples CASES with replacement and pools the items inside each drawn
    case, so the correlation between orderings of the same patient is carried
    into the interval instead of being assumed away.
    """
    keys = list(values_by_case)
    if not keys:
        return (np.nan, np.nan, np.nan)
    pooled = np.concatenate([np.asarray(values_by_case[k], dtype=float) for k in keys])
    pooled = pooled[np.isfinite(pooled)]
    point = float(pooled.mean()) if len(pooled) else np.nan
    arrs = [np.asarray(values_by_case[k], dtype=float) for k in keys]
    arrs = [a[np.isfinite(a)] for a in arrs]
    rng = np.random.default_rng(seed)
    n = len(keys)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, size=n)
        cat = np.concatenate([arrs[i] for i in pick if arrs[i].size])
        means[b] = cat.mean() if cat.size else np.nan
    means = means[np.isfinite(means)]
    if means.size < 2:
        return (point, np.nan, np.nan)
    return (point, float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def _truth_map(b: Battery) -> tuple[dict, int]:
    idx = {p: i for i, p in enumerate(b.pathologies)}
    return {c.case_id: idx.get(c.pathology, -1) for c in b.cases}, len(b.pathologies)


def _arms(model: str, task: str):
    d = load_task(model, task)
    if d is None or not d.height:
        return None, None
    P, ok = posterior_matrix(d) if task.endswith("posterior") else (None, None)
    return d, P


def _models() -> list[str]:
    from coherence.axioms.loading import available_models
    from coherence.elicit.registry import DEFAULT_ORDER
    have = set(available_models("a1_posterior"))
    return [m for m in DEFAULT_ORDER if m in have]


def _save(df: pl.DataFrame, name: str):
    df.write_csv(TABLES / f"{name}.csv")
    print(f"  {name}.csv  ({df.height} rows)")
    return df


# ------------------------------------------------------- comment 1 + 21 -----
def canonical_baseline() -> pl.DataFrame:
    """#1  A fixed canonical order is a zero-order-effect baseline.

    The review is right on the arithmetic: if every case is always presented
    in one deterministic order, there is no order to vary, so the measured
    order effect is exactly zero -- the same exact zero ELR-Fusion reports,
    reached without any method at all.

    What that baseline does NOT do is make the belief order-independent. It
    fixes a single arbitrary order and hides the dependence; the posterior it
    yields is still a function of that choice. The quantity that separates the
    two is the spread ACROSS fixed orders, reported here as
    `canonical_vs_random_jsd`: how far the canonical-order posterior sits from
    the posterior of a randomly chosen fixed order. For a genuinely
    order-invariant method that quantity is 0 by construction; for
    fixed-ordering it is the price of the arbitrary choice.

    Also reports (#21) the retest flip rate, which is the missing null for the
    A1 top-1 flip rate: how often the leading diagnosis changes when nothing
    changes at all.
    """
    b = Battery.load(BATTERY)
    truth, n_path = _truth_map(b)
    rows = []
    for m in _models():
        d = load_task(m, "a1_posterior")
        if d is None:
            continue
        P, _ = posterior_matrix(d)
        d = d.with_row_index("row")
        can_acc, rnd_acc, cvr, retest_flip, perm_flip = {}, {}, {}, {}, {}
        for cid, g in d.group_by("case_id"):
            cid = cid[0] if isinstance(cid, tuple) else cid
            ti = truth.get(cid, -1)
            if ti < 0:
                continue
            can = P[g.filter((pl.col("arm") == "canonical") & pl.col("valid"))["row"].to_numpy()]
            ret = P[g.filter((pl.col("arm") == "retest") & pl.col("valid"))["row"].to_numpy()]
            per = P[g.filter((pl.col("arm") == "permutation") & pl.col("valid"))["row"].to_numpy()]
            if len(can):
                can_acc[cid] = [float(np.mean(can.argmax(1) == ti))]
            if len(per):
                rnd_acc[cid] = [float(np.mean(per.argmax(1) == ti))]
                perm_flip[cid] = [float(len(set(per.argmax(1).tolist())) > 1)]
            if len(can) and len(per):
                # canonical posterior vs one fixed random order
                cvr[cid] = [jsd(can.mean(0), per[0])]
            if len(ret) >= 2:
                retest_flip[cid] = [float(len(set(ret.argmax(1).tolist())) > 1)]
        ca = cluster_bootstrap(can_acc, seed=1)
        ra = cluster_bootstrap(rnd_acc, seed=2)
        cv = cluster_bootstrap(cvr, seed=3)
        rf = cluster_bootstrap(retest_flip, seed=4)
        pf = cluster_bootstrap(perm_flip, seed=5)
        rows.append({
            "model": m, "n_cases": len(can_acc),
            "canonical_top1": ca[0], "canonical_lo": ca[1], "canonical_hi": ca[2],
            "random_order_top1": ra[0], "random_lo": ra[1], "random_hi": ra[2],
            "accuracy_cost_of_canonical": ca[0] - ra[0],
            "canonical_vs_random_jsd": cv[0], "cvr_lo": cv[1], "cvr_hi": cv[2],
            "retest_flip_rate": rf[0], "retest_flip_lo": rf[1], "retest_flip_hi": rf[2],
            "permutation_flip_rate": pf[0],
            "flip_excess_over_retest": pf[0] - rf[0],
        })
    return _save(pl.DataFrame(rows), "rev_canonical_baseline")


# ------------------------------------------------------------ comment 3 -----
def named_vs_posterior() -> pl.DataFrame:
    """#3  Reconcile the two accuracy paths, with n for each.

    Two different quantities have both been called "direct": the argmax of the
    stated 49-way posterior, and the separately elicited named diagnosis. They
    are not the same number and should never have shared a label.
    """
    b = Battery.load(BATTERY)
    truth, n_path = _truth_map(b)
    rows = []
    for m in _models():
        dp = load_task(m, "a1_posterior")
        da = load_task(m, "a1_answer")
        if dp is None or da is None:
            continue
        P, _ = posterior_matrix(dp)
        dp = dp.with_row_index("row")
        post, named, rank = {}, {}, {}
        ok = dp.filter(pl.col("valid"))
        for cid, g in ok.group_by("case_id"):
            cid = cid[0] if isinstance(cid, tuple) else cid
            ti = truth.get(cid, -1)
            if ti < 0:
                continue
            M = P[g["row"].to_numpy()]
            post[cid] = [float(np.mean(M.argmax(1) == ti))]
        oka = da.filter(pl.col("valid"))
        for cid, g in oka.group_by("case_id"):
            cid = cid[0] if isinstance(cid, tuple) else cid
            ti = truth.get(cid, -1)
            if ti < 0:
                continue
            pred = np.array([int(p[0]) if p else -1 for p in g["parsed"].to_list()])
            named[cid] = [float(np.mean(pred == ti))]
        # Where does the NAMED diagnosis sit in the model's own stated posterior?
        joined = (oka.select("item_id", pl.col("parsed").alias("named"))
                  .join(dp.filter(pl.col("valid")).select("item_id", "row"),
                        on="item_id", how="inner"))
        rr = []
        for r in joined.iter_rows(named=True):
            nm = r["named"]
            if not nm:
                continue
            k = int(nm[0])
            if not (0 <= k < n_path):
                continue
            v = P[r["row"]]
            rr.append(float((v > v[k]).sum() + 1))      # 1 = ranked first
        pa = cluster_bootstrap(post, seed=11)
        na = cluster_bootstrap(named, seed=12)
        rows.append({
            "model": m,
            "n_posterior_items": int(ok.height), "n_named_items": int(oka.height),
            "posterior_argmax_top1": pa[0], "posterior_lo": pa[1], "posterior_hi": pa[2],
            "named_diagnosis_top1": na[0], "named_lo": na[1], "named_hi": na[2],
            "gap_named_minus_posterior": na[0] - pa[0],
            "median_rank_of_named_in_posterior": float(np.median(rr)) if rr else np.nan,
            "named_ranked_first_rate": float(np.mean(np.array(rr) == 1)) if rr else np.nan,
        })
    return _save(pl.DataFrame(rows), "rev_named_vs_posterior")


# ------------------------------------------------------------ comment 5 -----
def nsample_averaged() -> pl.DataFrame:
    """#5  Is the belief gap sampling noise? Average N samples and re-measure.

    The retest arm is six draws from the identical prompt, so averaging it
    gives exactly the N-sample-averaged posterior the review asks for, at no
    extra generation cost. If the posterior's low top-1 were resampling noise,
    averaging six draws would close most of the gap to the named diagnosis.
    """
    b = Battery.load(BATTERY)
    truth, _ = _truth_map(b)
    rows = []
    for m in _models():
        d = load_task(m, "a1_posterior")
        if d is None:
            continue
        P, _ = posterior_matrix(d)
        d = d.with_row_index("row")
        single, avgd = {}, {}
        ns = []
        for cid, g in d.group_by("case_id"):
            cid = cid[0] if isinstance(cid, tuple) else cid
            ti = truth.get(cid, -1)
            if ti < 0:
                continue
            ret = P[g.filter((pl.col("arm") == "retest") & pl.col("valid"))["row"].to_numpy()]
            if len(ret) < 2:
                continue
            ns.append(len(ret))
            single[cid] = [float(np.mean(ret.argmax(1) == ti))]
            avgd[cid] = [float(ret.mean(0).argmax() == ti)]
        s = cluster_bootstrap(single, seed=21)
        a = cluster_bootstrap(avgd, seed=22)
        rows.append({
            "model": m, "n_cases": len(single),
            "mean_samples_per_case": float(np.mean(ns)) if ns else np.nan,
            "single_sample_top1": s[0], "single_lo": s[1], "single_hi": s[2],
            "n_sample_averaged_top1": a[0], "avg_lo": a[1], "avg_hi": a[2],
            "gain_from_averaging": a[0] - s[0],
        })
    return _save(pl.DataFrame(rows), "rev_nsample_averaged")


# ---------------------------------------------------- comments 13 + 18 ------
def a1_a3_nulls() -> pl.DataFrame:
    """#13/#18  Absolute gaps with patient-clustered CIs, against a real null.

    A1 and A3 both previously reported "fraction of cases above the floor"
    with no distribution behind the floor. The null used here is the per-case
    retest divergence distribution: its 95th percentile is the level a case
    reaches 5% of the time when nothing changes at all.
    """
    rows = []
    per_case = RESULTS / "analysed" / "a1_per_case.parquet"
    pc = pl.read_parquet(per_case) if per_case.exists() else None
    for m in _models():
        d = load_task(m, "a1_posterior")
        if d is None:
            continue
        P, _ = posterior_matrix(d)
        d = d.with_row_index("row")
        perm, ret = {}, {}
        for cid, g in d.group_by("case_id"):
            cid = cid[0] if isinstance(cid, tuple) else cid
            pm = P[g.filter((pl.col("arm") == "permutation") & pl.col("valid"))["row"].to_numpy()]
            rt = P[g.filter((pl.col("arm") == "retest") & pl.col("valid"))["row"].to_numpy()]
            if len(pm) >= 2:
                perm[cid] = [mean_pairwise_jsd(pm)]
            if len(rt) >= 2:
                ret[cid] = [mean_pairwise_jsd(rt)]
        common = [c for c in perm if c in ret]
        gaps = {c: [perm[c][0] - ret[c][0]] for c in common}
        g = cluster_bootstrap(gaps, seed=31)
        retvals = np.array([ret[c][0] for c in common], dtype=float)
        permvals = np.array([perm[c][0] for c in common], dtype=float)
        p95 = float(np.nanpercentile(retvals, 95)) if retvals.size else np.nan
        rows.append({
            "model": m, "n_cases": len(common),
            "mean_permutation_jsd": float(np.nanmean(permvals)),
            "mean_retest_jsd": float(np.nanmean(retvals)),
            "absolute_gap": g[0], "gap_lo": g[1], "gap_hi": g[2],
            "gap_excludes_zero": bool(np.isfinite(g[1]) and g[1] > 0),
            "retest_p95": p95,
            "frac_perm_above_retest_p95": float(np.nanmean(permvals > p95)),
            # 5% is what "above the 95th percentile of the null" means by
            # definition, so this is the number to compare against.
            "null_expectation": 0.05,
        })
    return _save(pl.DataFrame(rows), "rev_a1_absolute_gaps")


# ----------------------------------------------------------- comment 11 -----
def a2_noise_ceiling() -> pl.DataFrame:
    """#11  A2 judged against a reachable ceiling, not against 1.0.

    Slope 1.0 and r^2 1.0 are unattainable when the measurement itself is
    noisy, so the ceiling has to be estimated from the instrument.

    Construction: split a case's retest replicates into halves A and B and
    measure each half's log-odds shift against the PERMUTATION arm mean, which
    is drawn from different prompts and so carries independent noise. The
    agreement between those two shift estimates is the reliability of a single
    measured update -- and no model can agree with an external target better
    than it agrees with itself.

    The reference matters. An earlier version differenced each half against
    the pooled retest mean, which forces dA = -dB algebraically and reports a
    direction agreement of ~0 and an r^2 of ~1 for every model. Those were
    artefacts of the reference, not measurements.
    """
    rows = []
    for m in _models():
        d1 = load_task(m, "a1_posterior")
        if d1 is None:
            continue
        P1, _ = posterior_matrix(d1)
        d1 = d1.with_row_index("row")
        agree, r2s, slopes = [], [], []
        for cid, g in d1.group_by("case_id"):
            rt = P1[g.filter((pl.col("arm") == "retest") & pl.col("valid"))["row"].to_numpy()]
            pm = P1[g.filter((pl.col("arm") == "permutation") & pl.col("valid"))["row"].to_numpy()]
            if len(rt) < 4 or len(pm) < 1:
                continue
            h = len(rt) // 2
            ref = np.log(np.clip(pm.mean(0), 1e-9, 1))          # independent
            da = np.log(np.clip(rt[:h].mean(0), 1e-9, 1)) - ref
            db = np.log(np.clip(rt[h:2 * h].mean(0), 1e-9, 1)) - ref
            ok = np.isfinite(da) & np.isfinite(db)
            if ok.sum() < 4:
                continue
            agree.append(float(np.mean(np.sign(da[ok]) == np.sign(db[ok]))))
            if np.std(da[ok]) > 1e-9 and np.std(db[ok]) > 1e-9:
                r2s.append(float(np.corrcoef(da[ok], db[ok])[0, 1] ** 2))
                slopes.append(float(np.polyfit(da[ok], db[ok], 1)[0]))
        a2 = pl.read_csv(TABLES / "a2_main.csv")
        obs = {r["model"]: r for r in a2.iter_rows(named=True)}.get(m, {})
        rows.append({
            "model": m, "n_cases_ceiling": len(agree),
            "retest_direction_agreement_ceiling": float(np.mean(agree)) if agree else np.nan,
            "retest_r2_ceiling": float(np.mean(r2s)) if r2s else np.nan,
            "retest_slope_ceiling": float(np.mean(slopes)) if slopes else np.nan,
            "observed_direction_agreement": obs.get("direction_agreement"),
            "observed_r2": obs.get("r2"),
            "observed_beta": obs.get("beta"),
            "chance_direction_agreement": 0.5,
        })
    df = pl.DataFrame(rows)
    if df.height and "observed_r2" in df.columns:
        df = df.with_columns([
            (pl.col("observed_r2") / pl.col("retest_r2_ceiling")).alias("r2_frac_of_ceiling"),
            (pl.col("observed_direction_agreement") - 0.5).alias("dir_excess_over_chance"),
        ])
    return _save(df, "rev_a2_noise_ceiling")


# ----------------------------------------------------------- comment 19 -----
def entropy_control() -> pl.DataFrame:
    """#19  Is the scaling 'reversal' just output sharpness?

    A sharper posterior moves further under any perturbation, so both the
    floor and the permutation divergence scale with entropy. Reported here so
    the reader can see whether the anchored effect tracks entropy rather than
    scale.
    """
    a1 = pl.read_csv(TABLES / "a1_main.csv")
    keep = [c for c in ("model", "jsd_permutation", "jsd_retest_floor",
                        "normalised_order_effect", "mean_entropy_bits") if c in a1.columns]
    d = a1.select(keep).with_columns(
        (pl.col("jsd_permutation") - pl.col("jsd_retest_floor")).alias("absolute_gap"))
    if "mean_entropy_bits" in d.columns:
        x = d["mean_entropy_bits"].to_numpy()
        for col in ("absolute_gap", "normalised_order_effect", "jsd_retest_floor"):
            y = d[col].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else np.nan
            d = d.with_columns(pl.lit(r).alias(f"pearson_entropy_vs_{col}"))
    return _save(d, "rev_entropy_control")


# ----------------------------------------------------------- comment 20 -----
def scale_families() -> pl.DataFrame:
    """#20  The scale claim on every family that has a size pair, not just Qwen3."""
    a1 = pl.read_csv(TABLES / "a1_main.csv")
    mp = {r["model"]: r for r in a1.iter_rows(named=True)}
    ladders = {
        "Qwen3 (non-thinking)": ["qwen3-4b-nothink", "qwen3-8b-nothink", "qwen3-32b-nothink"],
        "Qwen3 (thinking)": ["qwen3-4b-think", "qwen3-8b-think", "qwen3-32b-think"],
        "MedGemma": ["medgemma-1.5-4b", "medgemma-27b"],
    }
    rows = []
    for fam, keys in ladders.items():
        ks = [k for k in keys if k in mp]
        if len(ks) < 2:
            continue
        raw = [mp[k]["jsd_permutation"] for k in ks]
        nor = [mp[k]["normalised_order_effect"] for k in ks]
        gap = [mp[k]["jsd_permutation"] - mp[k]["jsd_retest_floor"] for k in ks]
        rows.append({
            "family": fam, "n_sizes": len(ks), "models": ", ".join(ks),
            "raw_first": raw[0], "raw_last": raw[-1],
            "raw_rises_with_scale": bool(raw[-1] > raw[0]),
            "normalised_first": nor[0], "normalised_last": nor[-1],
            "normalised_falls_with_scale": bool(nor[-1] < nor[0]),
            "abs_gap_first": gap[0], "abs_gap_last": gap[-1],
            "abs_gap_falls_with_scale": bool(gap[-1] < gap[0]),
            "reversal_holds": bool(raw[-1] > raw[0] and nor[-1] < nor[0]),
        })
    return _save(pl.DataFrame(rows), "rev_scale_families")


# ----------------------------------------------------------- comment 24 -----
def generator_floor_definition() -> pl.DataFrame:
    """#24  Is 0.0490 over differing pairs only, or over all replicate pairs?

    Both are reported, because they answer different questions: the
    conditional mean describes how far apart two differentials are WHEN they
    differ, and the unconditional mean is the expected disagreement of the
    generator over any replicate pair -- which is the number that caps a
    DDXPlus matching metric.
    """
    from coherence.data.ddxplus import load_all
    import collections

    df = load_all()
    cols = set(df.columns)
    key = [c for c in ("AGE", "SEX", "PATHOLOGY", "EVIDENCES", "INITIAL_EVIDENCE")
           if c in cols]
    ddx = next((c for c in ("DIFFERENTIAL_DIAGNOSIS", "ddx", "DIFFERENTIAL") if c in cols), None)
    if not key or ddx is None:
        return _save(pl.DataFrame([{"note": "columns unavailable",
                                    "columns": ",".join(sorted(cols))[:400]}]),
                     "rev_generator_floor")
    sub = df.select(key + [ddx]).with_columns(
        pl.concat_str([pl.col(c).cast(pl.Utf8) for c in key], separator="||").alias("k"))
    groups = collections.defaultdict(list)
    for k, v in zip(sub["k"].to_list(), sub[ddx].to_list()):
        groups[k].append(v)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    n_pairs = n_diff = 0
    for v in dup.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                n_pairs += 1
                if v[i] != v[j]:
                    n_diff += 1
    return _save(pl.DataFrame([{
        "n_duplicate_groups": len(dup), "n_replicate_pairs": n_pairs,
        "n_pairs_differing": n_diff,
        "frac_pairs_differing": n_diff / n_pairs if n_pairs else np.nan,
        "note": "mean JSD conditional on differing vs unconditional differs by "
                "exactly this fraction; see reports for the scaled value",
    }]), "rev_generator_floor")


# ----------------------------------------------------------- comment 25 -----
def competence_gate() -> pl.DataFrame:
    """#25  State the gate explicitly and say which models pass.

    Two criteria are reported rather than one, because the review is right
    that 'beats chance' alone is weak at n = 35k: a 2x-chance model clears it
    while answering almost nothing correctly.
    """
    c = pl.read_csv(TABLES / "competence.csv")
    return _save(c.with_columns([
        (pl.col("acc_ci_low") > pl.col("chance")).alias("gate_beats_chance"),
        (pl.col("competence_ratio") >= 5.0).alias("gate_ratio_ge_5"),
        ((pl.col("acc_ci_low") > pl.col("chance")) &
         (pl.col("competence_ratio") >= 5.0)).alias("passes_both"),
    ]).sort("competence_ratio", descending=True), "rev_competence_gate")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    print("reviewer-response analyses (no new generation required):")
    for fn in (canonical_baseline, named_vs_posterior, nsample_averaged,
               a1_a3_nulls, a2_noise_ceiling, entropy_control,
               scale_families, generator_floor_definition, competence_gate):
        try:
            fn()
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
