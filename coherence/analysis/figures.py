"""Result figures. Every divergence panel carries its anchors."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from coherence.config import FIGURES, RESULTS, TABLES
from coherence.audit.figures import ACCENT, GRID, INK, MUTED, SEQ, WARN, _save

FLOOR_C = "#4393c3"
CEIL_C = "#f4a582"


def _t(name: str) -> pl.DataFrame | None:
    p = TABLES / f"{name}.csv"
    return pl.read_csv(p) if p.exists() else None


def _noise_floor() -> float:
    p = RESULTS / "audit_findings.json"
    if not p.exists():
        return float("nan")
    for f in json.loads(p.read_text()):
        if f["check"] == "generator_determinism":
            return float(f["stats"]["jsd_mean"])
    return float("nan")


def fig_a1_main() -> None:
    """The headline: order effect against both anchors, per model."""
    t = _t("a1_main")
    if t is None or not t.height:
        return
    m = t["model"].to_list()
    y = np.arange(len(m))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 0.46 * len(m) + 2.4),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    # Left: the three arms on one scale.
    h = 0.26
    a1.barh(y - h, t["jsd_retest_floor"], height=h, color=FLOOR_C,
            label="test–retest floor (same order, resampled)")
    a1.barh(y, t["jsd_permutation"], height=h, color=ACCENT,
            label="across permutations")
    a1.barh(y + h, t["jsd_shuffled_ceiling"], height=h, color=CEIL_C,
            label="shuffled-evidence ceiling")
    nf = _noise_floor()
    if np.isfinite(nf):
        a1.axvline(nf, color=WARN, ls="--", lw=1.1)
        a1.text(nf, len(m) - 0.3, f"  generator's own\n  noise floor ({nf:.3f})",
                color=WARN, fontsize=7.5, va="top")
    a1.set_yticks(y); a1.set_yticklabels(m, fontsize=8); a1.invert_yaxis()
    a1.set_xlabel("Jensen–Shannon divergence (base 2)")
    a1.set_title("A1 — order invariance, anchored at both ends", loc="left")
    a1.legend(fontsize=7.5, loc="lower right")

    # Right: the estimand itself, with bootstrap CIs.
    oe = t["order_effect"].to_numpy()
    lo = oe - t["ci_lo"].to_numpy()
    hi = t["ci_hi"].to_numpy() - oe
    a2.errorbar(oe, y, xerr=[lo, hi], fmt="o", color=ACCENT, ms=5, lw=1.2, capsize=3)
    a2.axvline(0, color=MUTED, lw=1)
    a2.set_yticks(y); a2.set_yticklabels([]); a2.invert_yaxis()
    a2.set_xlabel("OrderEffect = JSD(permutations) − JSD(retest)")
    a2.set_title("Estimand with 95% bootstrap CI over cases", loc="left")
    _save(fig, "a1_order_effect")


def fig_a1_k_sweep() -> None:
    t = _t("a1_k_sweep")
    if t is None or not t.height:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for i, (mk, sub) in enumerate(t.group_by("model_key", maintain_order=True)):
        mk = mk[0] if isinstance(mk, tuple) else mk
        sub = sub.sort("k")
        ax.plot(sub["k"], sub["order_effect"], "o-", lw=1.6, ms=4,
                color=SEQ[i % len(SEQ)], label=mk)
        ax.fill_between(sub["k"], sub["ci_lo"], sub["ci_hi"],
                        color=SEQ[i % len(SEQ)], alpha=0.15)
    ax.set_xlabel("K permutations per case"); ax.set_ylabel("OrderEffect (JSD)")
    ax.set_title("Ablation 1 — does the order effect converge in K?", loc="left")
    ax.legend(fontsize=7.5)
    _save(fig, "a1_k_sweep")


def fig_a2() -> None:
    t = _t("a2_main")
    if t is None or not t.height:
        return
    m = t["model"].to_list(); y = np.arange(len(m))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 0.44 * len(m) + 2.2))
    b = t["beta"].to_numpy()
    a1.errorbar(b, y, xerr=[b - t["beta_lo"].to_numpy(), t["beta_hi"].to_numpy() - b],
                fmt="o", color=ACCENT, ms=5, lw=1.2, capsize=3)
    a1.axvline(1.0, color=WARN, ls="--", lw=1.1)
    a1.text(1.0, -0.7, " β=1: correct update magnitude", color=WARN, fontsize=7.5)
    a1.set_yticks(y); a1.set_yticklabels(m, fontsize=8); a1.invert_yaxis()
    a1.set_xlabel("β  (implied Δlog-odds regressed on true)")
    a1.set_title("A2 — update fidelity.  β<1 is conservatism", loc="left")
    a2.barh(y, t["r2"], color=ACCENT, height=0.6)
    a2.set_yticks(y); a2.set_yticklabels([]); a2.invert_yaxis()
    a2.set_xlabel("R²  (share of the true update signal tracked)")
    a2.set_title("How much of the correct signal is tracked at all", loc="left")
    _save(fig, "a2_update_fidelity")


def fig_a3() -> None:
    t = _t("a3_main")
    if t is None or not t.height:
        return
    m = t["model"].to_list(); y = np.arange(len(m))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 0.44 * len(m) + 2.2))
    a1.barh(y, t["mean_spurious_jsd"], color=ACCENT, height=0.6)
    ct = float(t["certified_true_jsd"][0])
    a1.axvline(ct, color=WARN, ls="--", lw=1.1)
    a1.text(ct, -0.7, f" true update ≈ {ct:.1e}", color=WARN, fontsize=7.5)
    a1.set_yticks(y); a1.set_yticklabels(m, fontsize=8); a1.invert_yaxis()
    a1.set_xlabel("spurious JSD on a certified-uninformative finding")
    a1.set_title("A3 — redundancy insensitivity", loc="left")
    a2.barh(y, t["confidence_inflation_rate"] * 100, color=CEIL_C, height=0.6)
    a2.axvline(50, color=MUTED, ls="--", lw=1)
    a2.text(50, -0.7, " 50% = no systematic inflation", color=MUTED, fontsize=7.5)
    a2.set_yticks(y); a2.set_yticklabels([]); a2.invert_yaxis()
    a2.set_xlabel("% of items where entropy DROPPED (more text ⇒ more confidence)")
    a2.set_title("The clinically dangerous direction", loc="left")
    _save(fig, "a3_redundancy")


def fig_a4() -> None:
    t = _t("a4_main")
    if t is None or not t.height:
        return
    m = t["model"].to_list(); y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(7.4, 0.44 * len(m) + 2.2))
    b = t["beta_position"].to_numpy()
    ax.errorbar(b, y, xerr=[b - t["ci_lo"].to_numpy(), t["ci_hi"].to_numpy() - b],
                fmt="o", color=ACCENT, ms=5, lw=1.2, capsize=3)
    ax.axvline(0, color=WARN, ls="--", lw=1.1)
    ax.set_yticks(y); ax.set_yticklabels(m, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("β  (log-odds of the true pathology vs decisive-finding position)")
    ax.set_title("A4 — positional anchoring.  β<0 anchoring on the first finding,\n"
                 "β>0 recency; 0 is the coherent value", loc="left")
    _save(fig, "a4_anchoring")


def fig_methods() -> None:
    t = _t("methods")
    if t is None or not t.height:
        return
    piv = t.pivot(on="method", index="model", values="residual_order_sensitivity_jsd")
    methods = [c for c in piv.columns if c != "model"]
    m = piv["model"].to_list(); y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(m) + 2.6))
    h = 0.8 / len(methods)
    for i, meth in enumerate(methods):
        v = np.nan_to_num(piv[meth].to_numpy(), nan=0.0)
        ax.barh(y + (i - len(methods) / 2) * h, np.maximum(v, 1e-6), height=h,
                color=SEQ[i % len(SEQ)], label=meth)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(m, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("residual order sensitivity (JSD, log scale) — lower is better")
    ax.set_title("Table 1 — ELR-Fusion is exactly invariant at any budget;\n"
                 "permutation ensembling is only approximately invariant, at cost K",
                 loc="left")
    ax.legend(fontsize=7.5, loc="lower right")
    _save(fig, "methods_comparison")


def main() -> None:
    for fn in (fig_a1_main, fig_a1_k_sweep, fig_a2, fig_a3, fig_a4, fig_methods):
        try:
            fn()
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"figures -> {FIGURES}")


if __name__ == "__main__":
    main()
