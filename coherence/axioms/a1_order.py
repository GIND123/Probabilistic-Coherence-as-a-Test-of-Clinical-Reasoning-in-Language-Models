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

UNIFORM_ATOL = 1e-6


def is_informative(p: np.ndarray, atol: float = UNIFORM_ATOL) -> bool:
    """Does this response express a belief at all?

    An exactly-uniform posterior is parseable but carries no belief, and it is
    trivially order-invariant, so including it deflates both the order effect
    and the between-patient ceiling that normalises it. Calibration C6 found
    this on 68.8% of qwen3-4b-nothink responses under the original
    probability-elicitation prompt. Responses are therefore classified, and
    the order effect is reported twice: unconditionally, and conditional on
    the model having formed a belief in every arm of that case.
    """
    if p is None or not np.all(np.isfinite(p)):
        return False
    return not np.allclose(p, 1.0 / len(p), atol=atol)


@dataclass
class A1Result:
    model_key: str
    n_cases: int
    jsd_permutation: float
    jsd_retest: float
    jsd_shuffled: float
    jsd_between_case: float          # the interpretable ceiling
    order_effect: float
    order_effect_ci: tuple[float, float]
    normalised_order_effect: float      # (perm - retest) / (between_case - retest)
    top1_flip_rate: float
    top5_jaccard: float
    mean_entropy: float
    informative_rate: float               # share of responses expressing a belief
    informative_rate_by_arm: dict
    n_cases_informative: int
    order_effect_informative: float       # estimand on belief-forming cases only
    order_effect_informative_ci: tuple[float, float]
    per_case: pl.DataFrame = field(repr=False, default=None)


def _pairwise(mat: np.ndarray) -> float:
    return mean_pairwise_jsd(mat) if len(mat) >= 2 else np.nan


def between_case_ceiling(d: pl.DataFrame, posteriors: np.ndarray,
                         n_pairs: int = 20000, seed: int = 0) -> float:
    """Divergence between posteriors for DIFFERENT patients.

    This is the upper anchor. If reordering one patient's findings moves the
    posterior as far as swapping in an unrelated patient, the belief state is
    not a representation of that patient at all.

    It replaces the shuffled-evidence arm as the headline ceiling, because the
    shuffled arm measures something else and measures it in the wrong
    direction: given incoherent findings a model falls back to a default
    posterior, so two different nonsense inputs agree MORE than two resamples
    of a real case. Measured on qwen3-4b-nothink the shuffled arm sat at
    0.036 against a 0.059 retest floor, inverting the scale. The shuffled arm
    is retained and reported, but as a diagnostic for default-answer
    behaviour rather than as a ceiling.
    """
    rng = np.random.default_rng(seed)
    d = d.with_row_index("row")
    anchors = (d.filter((pl.col("arm") == "permutation") & (pl.col("perm_index") == 0)
                        & pl.col("valid"))["row"].to_numpy())
    if len(anchors) < 2:
        return float("nan")
    P = posteriors[anchors]
    # Uniform responses would drag the ceiling towards 0 and make the
    # normalised order effect look larger than it is.
    P = P[[is_informative(p) for p in P]]
    if len(P) < 2:
        return float("nan")
    i = rng.integers(0, len(P), n_pairs)
    j = rng.integers(0, len(P), n_pairs)
    keep = i != j
    return float(np.mean([jsd(P[a], P[b]) for a, b in zip(i[keep], j[keep])]))


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
        for arm in ("permutation", "retest", "shuffled", "canonical"):
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
        informative = {a: float(np.mean([is_informative(p) for p in M])) if len(M) else np.nan
                       for a, M in arms.items()}
        # Conditional estimand: recompute the divergences over the responses
        # that actually expressed a belief, rather than requiring every
        # response in the case to be informative. The conjunction over 10
        # permutations and 6 retests is far too strict -- at a 55% per-response
        # informative rate it retained 52 of 1,956 cases -- and the cases it
        # kept were exactly the ones the model found easiest, which is a
        # selection effect, not a cleaner measurement.
        inf_arms = {a: M[[is_informative(p) for p in M]] if len(M) else M
                    for a, M in arms.items()}
        enough = len(inf_arms["permutation"]) >= 2 and len(inf_arms["retest"]) >= 2
        rows.append({
            "case_id": case_id,
            "informative_perm": informative["permutation"],
            "informative_retest": informative["retest"],
            "all_informative": bool(
                informative["permutation"] == 1.0 and informative["retest"] == 1.0),
            "n_perm": len(perm), "n_retest": len(arms["retest"]),
            "n_perm_informative": len(inf_arms["permutation"]),
            "n_retest_informative": len(inf_arms["retest"]),
            "conditional_usable": bool(enough),
            "jsd_permutation_inf": _pairwise(inf_arms["permutation"]) if enough else np.nan,
            "jsd_retest_inf": _pairwise(inf_arms["retest"]) if enough else np.nan,
            "n_shuffled": len(arms["shuffled"]),
            "n_canonical": len(arms["canonical"]),
            "jsd_permutation": _pairwise(perm),
            "jsd_retest": _pairwise(arms["retest"]),
            "jsd_shuffled": _pairwise(arms["shuffled"]),
            "jsd_canonical_retest": _pairwise(arms["canonical"]),
            "jsd_canonical_vs_random": float(np.mean(
                [jsd(a, q) for a in arms["canonical"] for q in perm]))
            if len(arms["canonical"]) and len(perm) else np.nan,
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
        (pl.col("jsd_permutation") - pl.col("jsd_retest")).alias("order_effect"),
        (pl.col("jsd_permutation_inf") - pl.col("jsd_retest_inf")).alias("order_effect_inf"))


def _informative_rate_by_arm(d: pl.DataFrame, posteriors: np.ndarray) -> dict:
    """Per-arm informativeness. If this differs across arms, the conditional
    estimand is selection-biased and only the unconditional one is reportable."""
    d = d.with_row_index("row")
    out = {}
    for arm in ("permutation", "retest", "shuffled", "canonical"):
        rows = d.filter((pl.col("arm") == arm) & pl.col("valid"))["row"].to_numpy()
        if not len(rows):
            continue
        out[arm] = float(np.mean([is_informative(posteriors[r]) for r in rows]))
    return out


def analyse(model_key: str, d: pl.DataFrame, posteriors: np.ndarray,
            k_perms: int | None = None, seed: int = 0) -> A1Result:
    t = per_case_table(d, posteriors, k_perms)
    oe = t["order_effect"].to_numpy()
    perm = float(np.nanmean(t["jsd_permutation"].to_numpy()))
    ret = float(np.nanmean(t["jsd_retest"].to_numpy()))
    shuf = float(np.nanmean(t["jsd_shuffled"].to_numpy()))
    btw = between_case_ceiling(d, posteriors, seed=seed)
    denom = btw - ret
    ti = t.filter(pl.col("conditional_usable"))
    oei = ti["order_effect_inf"].to_numpy() if ti.height else np.array([])
    oei = oei[np.isfinite(oei)]
    by_arm = _informative_rate_by_arm(d, posteriors)
    return A1Result(
        model_key=model_key, n_cases=t.height,
        jsd_permutation=perm, jsd_retest=ret, jsd_shuffled=shuf,
        jsd_between_case=btw,
        order_effect=float(np.nanmean(oe)),
        order_effect_ci=_bootstrap_ci(oe, seed=seed),
        # 0 = indistinguishable from resampling the same prompt;
        # 1 = reordering one patient's findings moves the belief as far as
        #     substituting a different patient.
        normalised_order_effect=float((perm - ret) / denom) if denom > 1e-9 else np.nan,
        top1_flip_rate=float(t["top1_flips"].mean()),
        top5_jaccard=float(t["top5_jaccard"].mean()),
        mean_entropy=float(t["mean_entropy"].mean()),
        informative_rate=float(np.mean(list(by_arm.values()))) if by_arm else np.nan,
        informative_rate_by_arm=by_arm,
        n_cases_informative=int(ti.height),
        order_effect_informative=float(np.nanmean(oei)) if len(oei) else np.nan,
        order_effect_informative_ci=_bootstrap_ci(oei, seed=seed) if len(oei) > 1
        else (np.nan, np.nan),
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
