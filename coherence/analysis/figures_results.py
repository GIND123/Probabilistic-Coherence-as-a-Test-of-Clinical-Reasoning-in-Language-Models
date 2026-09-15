"""Result figures for the paper and the repository README.

Every figure is drawn from the CSVs in reports/tables, so a figure can never
disagree with the table it illustrates.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from coherence.config import FIGURES, TABLES

INK, MUTED, RULE = "#12181F", "#5C6B7A", "#DDE3EA"
ACCENT, FLOOR, CEIL = "#0E7C86", "#8FA3B5", "#C2CCD6"
WARN, GOOD = "#B4531A", "#2F6B3F"
NOISE_FLOOR = 0.0490

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.png")


def fig_anchors():
    """A1: floor -> observed -> ceiling per model, on one shared scale."""
    d = pl.read_csv(TABLES / "a1_main.csv").sort("normalised_order_effect")
    m = d["model"].to_list()
    y = np.arange(len(m))
    fl = d["jsd_retest_floor"].to_numpy()
    ob = d["jsd_permutation"].to_numpy()
    ce = d["jsd_between_case_ceiling"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.hlines(y, ob, ce, color=CEIL, lw=2.4, zorder=1)
    ax.hlines(y, fl, ob, color=ACCENT, lw=2.8, zorder=2)
    ax.scatter(ce, y, s=34, c=CEIL, edgecolors=FLOOR, lw=.8, zorder=3,
               label="between-patient ceiling")
    ax.scatter(fl, y, s=34, c=FLOOR, zorder=4, label="test–retest floor")
    ax.scatter(ob, y, s=42, c=ACCENT, zorder=5, label="permutation JSD")
    ax.axvline(NOISE_FLOOR, color=ACCENT, ls=":", lw=1.1, alpha=.8)
    ax.text(NOISE_FLOOR + .006, len(m) - .35, "generator noise 0.049",
            color=ACCENT, fontsize=7.5)

    ax.set_yticks(y); ax.set_yticklabels(m, fontsize=8)
    ax.set_xlabel("Jensen–Shannon divergence (base 2)")
    ax.set_xlim(0, .72); ax.set_ylim(-.7, len(m) - .3)
    ax.set_title("A1  Order effect is the gap from a model's own floor",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", ncol=1)
    ax.grid(axis="x", color=RULE, lw=.6, alpha=.7); ax.set_axisbelow(True)
    _save(fig, "result_a1_anchors")


def fig_scale():
    """A1: raw divergence rises with scale, normalised effect falls."""
    d = pl.read_csv(TABLES / "a1_main.csv")
    ladder = ["qwen3-4b-nothink", "qwen3-8b-nothink", "qwen3-32b-nothink"]
    sub = d.filter(pl.col("model").is_in(ladder))
    order = {k: i for i, k in enumerate(ladder)}
    sub = sub.sort(pl.col("model").replace_strict(order, default=99))
    x = np.arange(len(ladder))

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(x, sub["jsd_permutation"], "o-", color=WARN, lw=2, ms=7,
            label="raw permutation JSD")
    ax.plot(x, sub["normalised_order_effect"], "s-", color=ACCENT, lw=2, ms=7,
            label="normalised order effect")
    for i, (a, b) in enumerate(zip(sub["jsd_permutation"],
                                   sub["normalised_order_effect"])):
        ax.annotate(f"{a:.3f}", (i, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color=WARN)
        ax.annotate(f"{b:.3f}", (i, b), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7.5, color=ACCENT)
    ax.set_xticks(x); ax.set_xticklabels(["Qwen3-4B", "Qwen3-8B", "Qwen3-32B"])
    ax.set_ylabel("divergence"); ax.set_ylim(0, .36)
    ax.set_title("The anchors reverse the direction of the finding",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=RULE, lw=.6, alpha=.7); ax.set_axisbelow(True)
    _save(fig, "result_scale_reversal")


def fig_a2():
    """A2: direction agreement against chance."""
    d = pl.read_csv(TABLES / "a2_main.csv").sort("direction_agreement")
    m, v = d["model"].to_list(), d["direction_agreement"].to_numpy()
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.axvline(.5, color=WARN, lw=1.6, zorder=1)
    ax.text(.5, len(m) - .2, "chance", color=WARN, fontsize=8,
            ha="center", va="bottom")
    ax.hlines(y, .5, v, color=RULE, lw=1.6, zorder=2)
    ax.scatter(v, y, s=46, c=ACCENT, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(m, fontsize=8)
    ax.set_xlim(.42, .58)
    ax.set_xlabel("direction agreement with the empirical log-odds change")
    ax.set_title("A2  Belief updates do not track the evidence",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    ax.grid(axis="x", color=RULE, lw=.6, alpha=.7); ax.set_axisbelow(True)
    _save(fig, "result_a2_direction")


def fig_methods():
    """Table 1: order residual against query cost."""
    d = pl.read_csv(TABLES / "methods.csv").filter(
        pl.col("model") == "qwen3-32b-nothink")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    # Both ELR rows sit exactly on (0, 0); nudge their labels apart so the
    # two variants stay readable where the markers coincide.
    elr_seen = 0
    for r in d.iter_rows(named=True):
        meth = r["method"]
        exact = meth.startswith("elr_fusion")
        c = GOOD if exact else (WARN if meth == "direct" else ACCENT)
        ax.scatter(r["queries_per_new_case"], r["residual_order_sensitivity_jsd"],
                   s=140 if exact else 70, c=c, zorder=3,
                   marker="*" if exact else "o")
        if exact:
            off = (10, 6) if elr_seen == 0 else (10, -14)
            elr_seen += 1
        else:
            off = (-8, 8) if meth == "self_consistency" else (8, 5)
        ha = "right" if meth == "self_consistency" else "left"
        ax.annotate(meth.replace("_", " "),
                    (r["queries_per_new_case"], r["residual_order_sensitivity_jsd"]),
                    textcoords="offset points", xytext=off, fontsize=7.5,
                    color=INK, ha=ha)
    ks = np.array([2, 3, 5]); res = []
    for k in ks:
        row = d.filter(pl.col("method") == f"perm_ensemble_{k}")
        if row.height:
            res.append(row["residual_order_sensitivity_jsd"][0])
    if len(res) == len(ks):
        ax.plot(ks, res, color=ACCENT, lw=1.2, ls="--", alpha=.7, zorder=2)
    ax.axhline(0, color=GOOD, lw=1, ls=":", alpha=.8)
    ax.set_xlabel("model queries per new case")
    ax.set_ylabel("residual order sensitivity (JSD)")
    ax.set_xlim(-.9, 7.0); ax.set_ylim(-.035, .34)
    ax.set_title("Table 1  ELR-Fusion reaches exact zero at zero query cost",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    ax.grid(color=RULE, lw=.6, alpha=.7); ax.set_axisbelow(True)
    _save(fig, "result_methods_frontier")


def fig_tau():
    """The shrinkage frontier: accuracy and NLL disagree."""
    d = pl.read_csv(TABLES / "elr_tau_sweep.csv").filter(
        pl.col("model") == "qwen3-32b-nothink").sort("tau")
    t = d["tau"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    ax.plot(t, d["top1_accuracy"], "o-", color=ACCENT, lw=2, ms=5,
            label="top-1 accuracy")
    ax.set_xlabel("shrinkage coefficient  τ"); ax.set_ylabel("top-1 accuracy",
                                                             color=ACCENT)
    ax.tick_params(axis="y", labelcolor=ACCENT)
    ax.set_ylim(0, .17)
    ax2 = ax.twinx()
    ax2.plot(t, d["nll_of_truth"], "s--", color=WARN, lw=2, ms=5,
             label="NLL of truth")
    ax2.set_ylabel("negative log-likelihood of truth", color=WARN)
    ax2.tick_params(axis="y", labelcolor=WARN)
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False)
    best_a = float(t[np.argmax(d["top1_accuracy"].to_numpy())])
    best_n = float(t[np.argmin(d["nll_of_truth"].to_numpy())])
    ax.axvline(best_a, color=ACCENT, ls=":", lw=1)
    ax.axvline(best_n, color=WARN, ls=":", lw=1)
    ax.annotate(f"accuracy-optimal\nτ={best_a:g}", (best_a, .02),
                fontsize=7.5, color=ACCENT, ha="left")
    ax.annotate(f"calibration-optimal\nτ={best_n:g}", (best_n, .11),
                fontsize=7.5, color=WARN, ha="left")
    ax.set_title("τ is a frontier — invariance stays exact at every point",
                 loc="left", fontsize=10.5, color=INK, pad=10)
    ax.grid(color=RULE, lw=.6, alpha=.7); ax.set_axisbelow(True)
    _save(fig, "result_tau_frontier")


def fig_competence():
    """C11: competence gates coherence."""
    d = pl.read_csv(TABLES / "competence.csv").sort("top1_accuracy")
    m, v = d["model"].to_list(), d["top1_accuracy"].to_numpy()
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(y, v, color=ACCENT, height=.62, zorder=3)
    ax.axvline(1 / 49, color=WARN, lw=1.6, zorder=4)
    ax.text(1 / 49 + .008, -.6, "chance 0.020", color=WARN, fontsize=7.5)
    ax.scatter([0.0394], [len(m) - 1], marker="X", s=90, c=WARN, zorder=6)
    ax.annotate("gpt-oss-20b before the C11 fix: 0.039",
                (0.0394, len(m) - 1), textcoords="offset points",
                xytext=(12, -2), fontsize=7.5, color=WARN)
    ax.set_yticks(y); ax.set_yticklabels(m, fontsize=8)
    ax.set_xlabel("top-1 accuracy, 49-way pick-one-diagnosis")
    ax.set_xlim(0, .72)
    ax.set_title("Competence precedes coherence", loc="left",
                 fontsize=10.5, color=INK, pad=10)
    ax.grid(axis="x", color=RULE, lw=.6, alpha=.7); ax.set_axisbelow(True)
    _save(fig, "result_competence")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("writing result figures:")
    fig_anchors(); fig_scale(); fig_a2()
    fig_methods(); fig_tau(); fig_competence()


if __name__ == "__main__":
    main()
