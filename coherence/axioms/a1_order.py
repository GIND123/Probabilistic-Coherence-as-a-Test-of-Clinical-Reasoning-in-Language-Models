"""A1 -- order invariance.

Estimand
--------
    OrderEffect = JSD(across permutations) - JSD(same permutation, resampled)

The subtrahend is the model's test-retest floor. Reporting the raw
across-permutation divergence without it would attribute ordinary sampling
noise to order sensitivity. A shuffled-evidence ceiling anchors the other end
of the scale, and the generator's own replicate noise (measured in the data
audit at 0.049 JSD) anchors what the ground truth itself can distinguish.

Nothing here needs a label, which is what lets A1 run on real clinical
narratives as well as on DDXPlus.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from coherence.metrics.divergence import entropy, jsd, mean_pairwise_jsd, topk_jaccard


@dataclass
class A1Result:
    model_key: str
    n_cases: int
    jsd_permutation: float
    jsd_retest: float
    jsd_shuffled: float
    order_effect: float
    order_effect_ci: tuple[float, float]
    normalised_order_effect: float      # (perm - retest) / (shuffled - retest)
    top1_flip_rate: float
    top5_jaccard: float
    mean_entropy: float
    per_case: pl.DataFrame = field(repr=False, default=None)


def _pairwise(mat: np.ndarray) -> float:
    return mean_pairwise_jsd(mat) if len(mat) >= 2 else np.nan


def _bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0,
                  alpha: float = 0.05) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def per_case_table(d: pl.DataFrame, posteriors: np.ndarray,
                   k_perms: int | None = None) -> pl.DataFrame:
    """One row per case: the three arms plus the derived quantities."""
    d = d.with_row_index("row")
    rows = []
    for case_id, sub in d.group_by("case_id", maintain_order=True):
        case_id = case_id[0] if isinstance(case_id, tuple) else case_id
        arms = {}
        for arm in ("permutation", "retest", "shuffled"):
            s = sub.filter((pl.col("arm") == arm) & pl.col("valid"))
            if arm == "permutation" and k_perms is not None:
                s = s.filter(pl.col("perm_index") < k_perms)
            arms[arm] = posteriors[s["row"].to_numpy()]
        perm = arms["permutation"]
        if len(perm) < 2:
            continue
        top = perm.argmax(axis=1)
        jac = np.mean([topk_jaccard(perm[i], perm[j], 5)
                       for i in range(len(perm)) for j in range(i + 1, len(perm))])
        rows.append({
            "case_id": case_id,
            "n_perm": len(perm), "n_retest": len(arms["retest"]),
            "n_shuffled": len(arms["shuffled"]),
            "jsd_permutation": _pairwise(perm),
            "jsd_retest": _pairwise(arms["retest"]),
            "jsd_shuffled": _pairwise(arms["shuffled"]),
            "top1_flips": int(len(set(top.tolist())) > 1),
            "top1_modal_share": float(np.bincount(top).max() / len(top)),
            "top5_jaccard": float(jac),
            "mean_entropy": float(np.mean([entropy(p) for p in perm])),
            "severity": int(sub["severity"][0]) if "severity" in sub.columns else -1,
            "pathology": sub["pathology"][0] if "pathology" in sub.columns else "",
            "n_findings": int(sub["n_findings"][0]) if "n_findings" in sub.columns else -1,
        })
    t = pl.DataFrame(rows)
    return t.with_columns(
        (pl.col("jsd_permutation") - pl.col("jsd_retest")).alias("order_effect"))


def analyse(model_key: str, d: pl.DataFrame, posteriors: np.ndarray,
            k_perms: int | None = None, seed: int = 0) -> A1Result:
    t = per_case_table(d, posteriors, k_perms)
    oe = t["order_effect"].to_numpy()
    perm = float(np.nanmean(t["jsd_permutation"].to_numpy()))
    ret = float(np.nanmean(t["jsd_retest"].to_numpy()))
    shuf = float(np.nanmean(t["jsd_shuffled"].to_numpy()))
    denom = shuf - ret
    return A1Result(
        model_key=model_key, n_cases=t.height,
        jsd_permutation=perm, jsd_retest=ret, jsd_shuffled=shuf,
        order_effect=float(np.nanmean(oe)),
        order_effect_ci=_bootstrap_ci(oe, seed=seed),
        normalised_order_effect=float((perm - ret) / denom) if denom > 1e-9 else np.nan,
        top1_flip_rate=float(t["top1_flips"].mean()),
        top5_jaccard=float(t["top5_jaccard"].mean()),
        mean_entropy=float(t["mean_entropy"].mean()),
        per_case=t)


def k_sweep(model_key: str, d: pl.DataFrame, posteriors: np.ndarray,
            ks: tuple[int, ...] = (2, 3, 5, 10)) -> pl.DataFrame:
    """Ablation #1: does the order effect converge as K grows?"""
    rows = []
    for k in ks:
        r = analyse(model_key, d, posteriors, k_perms=k)
        rows.append({"model_key": model_key, "k": k,
                     "jsd_permutation": r.jsd_permutation,
                     "jsd_retest": r.jsd_retest,
                     "order_effect": r.order_effect,
                     "ci_lo": r.order_effect_ci[0], "ci_hi": r.order_effect_ci[1],
                     "top1_flip_rate": r.top1_flip_rate})
    return pl.DataFrame(rows)


def severity_weighted(t: pl.DataFrame, posteriors: np.ndarray, d: pl.DataFrame,
                      severity: dict[str, int], pathologies: list[str],
                      k: int = 5, high_severity_max: int = 2) -> dict:
    """Does reordering move a high-severity diagnosis in or out of the top-k?

    Incoherence that shuffles two benign diagnoses is not the same clinical
    event as incoherence that drops a pulmonary embolism out of the top five.
    """
    sev = np.array([severity.get(p, 5) for p in pathologies])
    high = sev <= high_severity_max
    d = d.with_row_index("row")
    flips, n = 0, 0
    for case_id, sub in d.group_by("case_id", maintain_order=True):
        s = sub.filter((pl.col("arm") == "permutation") & pl.col("valid"))
        P = posteriors[s["row"].to_numpy()]
        if len(P) < 2:
            continue
        n += 1
        sets = [frozenset(np.argsort(-p)[:k].tolist()) for p in P]
        union, inter = set().union(*sets), set(sets[0]).intersection(*sets)
        unstable = union - inter
        if any(high[i] for i in unstable):
            flips += 1
    return {"n_cases": n, "high_severity_topk_instability": flips / n if n else np.nan,
            "k": k, "high_severity_max": high_severity_max}
