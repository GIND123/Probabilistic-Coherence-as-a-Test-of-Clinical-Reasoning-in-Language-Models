"""Publication figures for the CoDx paper and the repository README.

House style
-----------
Every figure is drawn from the CSVs in ``reports/tables`` (or the per-case
parquet), so a figure can never disagree with the table it illustrates.

Colours come from the Okabe-Ito colourblind-safe palette, and model family is
encoded consistently across every panel: a reader who learns the colours in
Figure 1 can carry them through the paper. Figures are sized for a two-column
manuscript (``COL`` single column, ``WIDE`` full width) with type at 8-9 pt so
they stay legible at print size without rescaling.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from coherence.config import FIGURES, TABLES, RESULTS

# ---------------------------------------------------------------- palette ---
# Okabe-Ito: distinguishable under the common forms of colour vision deficiency
# and in greyscale print.
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERM, PURPLE, SKY = "#D55E00", "#CC79A7", "#56B4E9"
INK, MUTED, RULE = "#1A1A1A", "#6B7280", "#D8DEE5"
FLOOR_C, CEIL_C = "#9AA7B4", "#D3DAE2"
ACCENT = BLUE

NOISE_FLOOR = 0.0490
CHANCE = 1.0 / 49

COL, WIDE = 3.42, 7.0          # inches: single and double column

FAMILY = {
    "qwen3-4b-nothink": "Qwen3", "qwen3-4b-think": "Qwen3",
    "qwen3-8b-nothink": "Qwen3", "qwen3-8b-think": "Qwen3",
    "qwen3-32b-nothink": "Qwen3", "qwen3-32b-think": "Qwen3",
    "medgemma-1.5-4b": "Medical", "medgemma-27b": "Medical",
    "med42-8b": "Medical",
    "r1-distill-32b": "Reasoning-distilled", "gpt-oss-20b": "GPT-OSS",
}
FAM_C = {"Qwen3": BLUE, "Medical": GREEN,
         "Reasoning-distilled": PURPLE, "GPT-OSS": ORANGE}

# Short labels: the full keys are too wide for an axis and repeat the family.
SHORT = {
    "qwen3-4b-nothink": "Qwen3-4B", "qwen3-4b-think": "Qwen3-4B ᵗ",
    "qwen3-8b-nothink": "Qwen3-8B", "qwen3-8b-think": "Qwen3-8B ᵗ",
    "qwen3-32b-nothink": "Qwen3-32B", "qwen3-32b-think": "Qwen3-32B ᵗ",
    "medgemma-1.5-4b": "MedGemma-1.5-4B", "medgemma-27b": "MedGemma-27B",
    "med42-8b": "Med42-8B", "r1-distill-32b": "R1-Distill-32B",
    "gpt-oss-20b": "GPT-OSS-20B",
}

plt.rcParams.update({
    # Lato only: it ships weights 250-900, so the 600 used for titles resolves
    # directly. Listing heavier fallbacks here makes matplotlib probe each of
    # them for weight 600 and warn, even though Lato already matched.
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Nimbus Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .8, "xtick.major.width": .8, "ytick.major.width": .8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.frameon": False, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": .02,
    "savefig.facecolor": "white", "axes.titlepad": 8,
    "axes.titlelocation": "left", "axes.titleweight": 600,
})


def _c(m):
    return FAM_C[FAMILY.get(m, "Qwen3")]


def _grid(ax, axis="both"):
    ax.grid(axis=axis, color=RULE, lw=.55, alpha=.85)
    ax.set_axisbelow(True)


def _fam_legend(ax, **kw):
    ax.legend(handles=[Patch(facecolor=v, label=k) for k, v in FAM_C.items()],
              **kw)


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


def _a1():
    return pl.read_csv(TABLES / "a1_main.csv")


# ============================================================ Figure 1 =====
def fig1_anchors():
    """A1: floor -> observed -> ceiling, one shared scale."""
    d = _a1().sort("normalised_order_effect")
    m = d["model"].to_list()
    y = np.arange(len(m))
    fl, ob, ce = (d["jsd_retest_floor"].to_numpy(),
                  d["jsd_permutation"].to_numpy(),
                  d["jsd_between_case_ceiling"].to_numpy())

    fig, ax = plt.subplots(figsize=(WIDE, 3.7))
    ax.hlines(y, ob, ce, color=CEIL_C, lw=2.6, zorder=1)
    for i, mk in enumerate(m):
        ax.hlines(y[i], fl[i], ob[i], color=_c(mk), lw=3.4, zorder=2)
    ax.scatter(ce, y, s=26, c=CEIL_C, edgecolors=FLOOR_C, lw=.7, zorder=3)
    ax.scatter(fl, y, s=26, c="white", edgecolors=FLOOR_C, lw=1.4, zorder=4)
    ax.scatter(ob, y, s=40, c=[_c(k) for k in m], zorder=5)

    ax.axvline(NOISE_FLOOR, color=MUTED, ls=(0, (2, 2)), lw=.9)
    # Sits under the axis rather than in the plot body, where it used to
    # overlap the top row.
    ax.annotate("generator noise floor 0.049", xy=(NOISE_FLOOR, -.62),
                xytext=(NOISE_FLOOR + .03, -.62), fontsize=7, color=MUTED,
                va="center", arrowprops=dict(arrowstyle="-", color=MUTED,
                                             lw=.7))

    for i, mk in enumerate(m):
        ax.text(ce[i] + .014, y[i], f"{d['normalised_order_effect'][i]:.3f}",
                va="center", fontsize=7.2, color=_c(mk), fontweight=600)

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[k] for k in m])
    ax.set_xlabel("Jensen–Shannon divergence (bits)")
    ax.set_xlim(0, .78)
    ax.set_ylim(-1.0, len(m) - .3)
    ax.set_title("Order effect is the gap from a model's own test–retest floor")
    _grid(ax, "x")

    # Both legends go beneath the axes: the plot body is full of long ceiling
    # bars on the right, which is the only place they would otherwise fit.
    anchors = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=FLOOR_C, mew=1.4,
               ms=5, label="test–retest floor"),
        Line2D([], [], marker="o", ls="", color=INK, ms=6,
               label="permutation JSD"),
        Line2D([], [], marker="o", ls="", mfc=CEIL_C, mec=FLOOR_C, ms=5,
               label="between-patient ceiling"),
    ]
    leg = ax.legend(handles=anchors, loc="upper center",
                    bbox_to_anchor=(.5, -.13), ncol=3, columnspacing=2.2,
                    handletextpad=.5)
    ax.add_artist(leg)
    ax.legend(handles=[Patch(facecolor=v, label=k) for k, v in FAM_C.items()],
              loc="upper center", bbox_to_anchor=(.5, -.23), ncol=4,
              columnspacing=1.6, handletextpad=.5, handlelength=1.1)
    _save(fig, "fig01_a1_anchors")


# ============================================================ Figure 2 =====
def fig2_scale_reversal():
    """Raw divergence rises with scale; the normalised effect falls."""
    d = _a1()
    ladders = {
        "non-thinking": ["qwen3-4b-nothink", "qwen3-8b-nothink",
                         "qwen3-32b-nothink"],
        "thinking": ["qwen3-4b-think", "qwen3-8b-think", "qwen3-32b-think"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.0), sharex=True)
    x = np.arange(3)
    for ax, (nm, keys) in zip(axes, ladders.items()):
        sub = {r["model"]: r for r in d.iter_rows(named=True)}
        raw = [sub[k]["jsd_permutation"] for k in keys]
        nor = [sub[k]["normalised_order_effect"] for k in keys]
        ax.plot(x, raw, "o-", color=VERM, lw=1.8, ms=5.5,
                label="raw permutation JSD")
        ax.plot(x, nor, "s-", color=BLUE, lw=1.8, ms=5.5,
                label="normalised order effect")
        for i in range(3):
            ax.annotate(f"{raw[i]:.3f}", (i, raw[i]), xytext=(0, 7),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=VERM)
            ax.annotate(f"{nor[i]:.3f}", (i, nor[i]), xytext=(0, -13),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=BLUE)
        ax.set_xticks(x)
        ax.set_xticklabels(["4B", "8B", "32B"])
        ax.set_ylim(0, .38)
        ax.set_title(f"Qwen3, {nm}", fontsize=9)
        ax.set_xlabel("parameters")
        _grid(ax, "y")
    axes[0].set_ylabel("divergence (bits)")
    axes[0].legend(loc="upper left")
    fig.suptitle("The anchors reverse the direction of the scaling result",
                 x=.02, ha="left", fontsize=10, fontweight=600, y=1.03)
    _save(fig, "fig02_scale_reversal")


# ============================================================ Figure 3 =====
def fig3_percase_distribution():
    """Per-case order effect: the mean hides a long tail."""
    p = RESULTS / "analysed" / "a1_per_case.parquet"
    if not p.exists():
        return
    d = pl.read_parquet(p)
    order = _a1().sort("normalised_order_effect")["model"].to_list()
    fig, ax = plt.subplots(figsize=(WIDE, 3.9))
    for i, mk in enumerate(order):
        v = d.filter(pl.col("model") == mk)["order_effect"].to_numpy()
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        parts = ax.violinplot(v, positions=[i], orientation="horizontal",
                              widths=.82,
                              showextrema=False, showmedians=False)
        for b in parts["bodies"]:
            b.set_facecolor(_c(mk)); b.set_alpha(.45); b.set_edgecolor("none")
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.hlines(i, q1, q3, color=_c(mk), lw=3.2, zorder=3)
        ax.plot(med, i, "o", ms=4, color="white", mec=_c(mk), mew=1.3, zorder=4)
        ax.plot(np.percentile(v, 95), i, "|", ms=7, color=_c(mk), zorder=4)
    ax.axvline(0, color=MUTED, lw=.9, ls=(0, (2, 2)))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[k] for k in order])
    ax.set_xlabel("per-case order effect (JSD above the model's own floor)")
    ax.set_xlim(-.25, .85)
    ax.set_title("A mean order effect near zero still hides a heavy upper tail")
    ax.text(.84, -.55, "│ = 95th percentile", ha="right", fontsize=7,
            color=MUTED)
    _grid(ax, "x")
    _save(fig, "fig03_percase_distribution")


# ============================================================ Figure 4 =====
def fig4_k_sweep():
    """Order effect against the number of permutations sampled."""
    d = pl.read_csv(TABLES / "a1_k_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.0))
    for mk, g in d.group_by("model_key"):
        mk = mk[0] if isinstance(mk, tuple) else mk
        g = g.sort("k")
        k = g["k"].to_numpy()
        axes[0].plot(k, g["order_effect"], "o-", color=_c(mk), lw=1.4, ms=4,
                     alpha=.9)
        axes[0].fill_between(k, g["ci_lo"], g["ci_hi"], color=_c(mk),
                             alpha=.13, lw=0)
        axes[1].plot(k, g["top1_flip_rate"], "o-", color=_c(mk), lw=1.4, ms=4,
                     alpha=.9)
    axes[0].set_ylabel("order effect (JSD above floor)")
    axes[0].set_title("Estimate stabilises by K ≈ 5", fontsize=9)
    axes[1].set_ylabel("top-1 flip rate")
    axes[1].set_title("Flip rate keeps climbing with K", fontsize=9)
    for ax in axes:
        ax.set_xlabel("permutations sampled per case  (K)")
        ax.set_xticks([2, 3, 5, 10])
        _grid(ax)
    axes[1].set_ylim(0, 1.02)
    _fam_legend(axes[1], loc="lower right", ncol=1)
    fig.suptitle("More orderings do not reveal more divergence — they reveal "
                 "more flips", x=.02, ha="left", fontsize=10,
                 fontweight=600, y=1.04)
    _save(fig, "fig04_k_sweep")


# ============================================================ Figure 5 =====
def fig5_a2_direction():
    """A2: direction agreement against chance."""
    d = pl.read_csv(TABLES / "a2_main.csv").sort("direction_agreement")
    m = d["model"].to_list()
    v = d["direction_agreement"].to_numpy()
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(COL * 2, 3.6))
    ax.axvspan(.48, .52, color=VERM, alpha=.07, lw=0)
    ax.axvline(.5, color=VERM, lw=1.4, zorder=2)
    ax.text(.5, len(m) - .25, "chance", color=VERM, fontsize=7.5,
            ha="center", va="bottom", fontweight=600)
    ax.hlines(y, .5, v, color=RULE, lw=1.4, zorder=2)
    ax.scatter(v, y, s=44, c=[_c(k) for k in m], zorder=3)
    for i, mk in enumerate(m):
        ax.text(v[i] + (.004 if v[i] >= .5 else -.004), y[i], f"{v[i]:.3f}",
                va="center", ha="left" if v[i] >= .5 else "right",
                fontsize=7, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[k] for k in m])
    ax.set_xlim(.435, .575)
    ax.set_xlabel("agreement with the empirical log-odds direction")
    ax.set_title("A2  Belief updates do not track the evidence")
    _grid(ax, "x")
    _save(fig, "fig05_a2_direction")


# ============================================================ Figure 6 =====
def fig6_a2_slope():
    """A2 forest plot: slope against the ideal of 1."""
    d = pl.read_csv(TABLES / "a2_main.csv").sort("beta")
    m = d["model"].to_list()
    b = d["beta"].to_numpy()
    lo, hi = d["beta_lo"].to_numpy(), d["beta_hi"].to_numpy()
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(COL * 2, 3.6))
    ax.axvline(0, color=MUTED, lw=1.0, ls=(0, (2, 2)))
    ax.hlines(y, lo, hi, color=[_c(k) for k in m], lw=2.0, alpha=.75)
    ax.scatter(b, y, s=36, c=[_c(k) for k in m], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[k] for k in m])
    ax.set_xlabel("slope β  (95% CI)")
    ax.set_title("A2  Calibrated updating would give β = 1; the largest is 0.058")
    ax.annotate("perfect Bayesian updating: β = 1.0, off-scale by 17×",
                xy=(.99, .02), xycoords="axes fraction", ha="right",
                fontsize=7.2, color=VERM, fontweight=600)
    _grid(ax, "x")
    _save(fig, "fig06_a2_slope")


# ============================================================ Figure 7 =====
def fig7_a3_redundancy():
    """A3: how far certified-uninformative evidence moves belief."""
    d = pl.read_csv(TABLES / "a3_main.csv").sort("mean_spurious_jsd")
    m = d["model"].to_list()
    y = np.arange(len(m))
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.5), sharey=True)
    ax = axes[0]
    ax.hlines(y, d["median_spurious_jsd"], d["p90_spurious_jsd"],
              color=[_c(k) for k in m], lw=2.4, alpha=.45)
    ax.scatter(d["median_spurious_jsd"], y, s=26, c="white",
               edgecolors=[_c(k) for k in m], lw=1.3, zorder=3)
    ax.scatter(d["mean_spurious_jsd"], y, s=40, c=[_c(k) for k in m], zorder=4)
    ax.scatter(d["p90_spurious_jsd"], y, s=24, marker="|",
               c=[_c(k) for k in m], zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[k] for k in m])
    ax.set_xlabel("JSD induced by irrelevant evidence")
    ax.set_title("Median → p90 spread", fontsize=9)
    ax.set_xlim(0, 1.0)
    _grid(ax, "x")

    ax = axes[1]
    ax.barh(y - .19, d["frac_above_floor"], height=.36,
            color=[_c(k) for k in m], alpha=.9, label="above own noise floor")
    ax.barh(y + .19, d["top1_flip_rate"], height=.36,
            color=[_c(k) for k in m], alpha=.42, label="top-1 diagnosis flipped")
    ax.set_xlabel("fraction of cases")
    ax.set_title("Share of patients affected", fontsize=9)
    ax.set_xlim(0, .78)
    ax.legend(loc="lower right")
    _grid(ax, "x")
    fig.suptitle("A3  Evidence the corpus certifies as uninformative still "
                 "moves the differential", x=.02, ha="left", fontsize=10,
                 fontweight=600, y=1.02)
    _save(fig, "fig07_a3_redundancy")


# ============================================================ Figure 8 =====
def fig8_a4_position():
    """A4: position effects are present but small."""
    d = pl.read_csv(TABLES / "a4_main.csv").sort("position_effect")
    m = d["model"].to_list()
    y = np.arange(len(m))
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.3), sharey=True)
    ax = axes[0]
    ax.barh(y, d["position_effect"], color=[_c(k) for k in m], height=.62)
    ax.axvline(0, color=MUTED, lw=.9)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[k] for k in m])
    ax.set_xlabel("between-position − within-position JSD")
    ax.set_title("Position effect", fontsize=9)
    _grid(ax, "x")

    ax = axes[1]
    ax.scatter(d["jsd_within_position"], d["jsd_between_positions"],
               s=52, c=[_c(k) for k in m], zorder=3)
    lim = [0, max(d["jsd_between_positions"].max(),
                  d["jsd_within_position"].max()) * 1.12]
    ax.plot(lim, lim, ls=(0, (3, 3)), color=MUTED, lw=.9, zorder=1)
    ax.annotate("y = x: no positional\npreference", xy=(lim[1] * .62, lim[1] * .66),
                fontsize=7, color=MUTED, linespacing=1.3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("within-position JSD")
    ax.set_ylabel("between-position JSD")
    ax.set_title("Points sit on the diagonal", fontsize=9)
    _grid(ax)
    fig.suptitle("A4  Order matters, but not through a first- or last-position "
                 "anchor", x=.02, ha="left", fontsize=10,
                 fontweight=600, y=1.03)
    _save(fig, "fig08_a4_position")


# ============================================================ Figure 9 =====
def fig9_methods_pareto():
    """Table 1: order residual against query cost."""
    d = pl.read_csv(TABLES / "methods.csv").filter(
        pl.col("model") == "qwen3-32b-nothink")
    style = {
        "direct": (VERM, "o", "direct elicitation"),
        "self_consistency": (SKY, "o", "self-consistency"),
        "perm_ensemble_2": (BLUE, "o", "perm-ensemble K=2"),
        "perm_ensemble_3": (BLUE, "o", "perm-ensemble K=3"),
        "perm_ensemble_5": (BLUE, "o", "perm-ensemble K=5"),
        "elr_fusion": (GREEN, "*", "ELR-Fusion  τ=1"),
        "elr_fusion_calibrated": (GREEN, "*", "ELR-Fusion  calibrated"),
    }
    fig, ax = plt.subplots(figsize=(WIDE * .62, 3.6))
    ks, res = [], []
    seen_elr = 0
    for r in d.iter_rows(named=True):
        meth = r["method"]
        if meth not in style:
            continue
        c, mk, lab = style[meth]
        q = r["queries_per_new_case"]
        v = r["residual_order_sensitivity_jsd"]
        ax.scatter(q, v, s=210 if mk == "*" else 62, c=c, marker=mk,
                   zorder=4, edgecolors="white", lw=.6)
        if meth.startswith("perm_ensemble"):
            ks.append(q); res.append(v)
        if mk == "*":
            dy = 9 if seen_elr == 0 else -15
            seen_elr += 1
            ax.annotate(lab, (q, v), xytext=(11, dy),
                        textcoords="offset points", fontsize=7.4, color=c,
                        fontweight=600)
        else:
            # Explicit per-method placement: K=5 and self-consistency sit close
            # enough that a shared rule put one label beside the other's point.
            place = {
                "direct": ((9, 4), "left"),
                "perm_ensemble_2": ((9, 4), "left"),
                "perm_ensemble_3": ((-10, -4), "right"),
                "perm_ensemble_5": ((0, -15), "center"),
                "self_consistency": ((0, 11), "center"),
            }
            off, ha = place[meth]
            ax.annotate(lab, (q, v), xytext=off, textcoords="offset points",
                        fontsize=7.4, color=INK, ha=ha)
    o = np.argsort(ks)
    ax.plot(np.array(ks)[o], np.array(res)[o], color=BLUE, lw=1.1,
            ls=(0, (4, 3)), alpha=.7, zorder=2)
    ax.axhline(0, color=GREEN, lw=1.0, ls=(0, (2, 2)), alpha=.9, zorder=1)
    ax.annotate("exact zero — a theorem, at any budget", xy=(6.6, .008),
                fontsize=7.2, color=GREEN, ha="right", fontweight=600)
    ax.set_xlabel("model queries per new case")
    ax.set_ylabel("residual order sensitivity (JSD)")
    ax.set_xlim(-1.0, 7.6); ax.set_ylim(-.045, .345)
    ax.set_title("ELR-Fusion reaches exact zero at zero marginal cost")
    _grid(ax)
    _save(fig, "fig09_methods_pareto")


# =========================================================== Figure 10 =====
def fig10_tau_frontier():
    """The shrinkage frontier: accuracy and calibration disagree."""
    d = pl.read_csv(TABLES / "elr_tau_sweep.csv").filter(
        pl.col("model") == "qwen3-32b-nothink").sort("tau")
    t = d["tau"].to_numpy()
    acc = d["top1_accuracy"].to_numpy()
    nll = d["nll_of_truth"].to_numpy()
    fig, ax = plt.subplots(figsize=(WIDE * .62, 3.6))
    ln1, = ax.plot(t, acc, "o-", color=BLUE, lw=1.8, ms=4.5,
                   label="top-1 accuracy")
    ax.set_xlabel("shrinkage coefficient  τ")
    ax.set_ylabel("top-1 accuracy", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_ylim(0, .175)
    ax2 = ax.twinx()
    ln2, = ax2.plot(t, nll, "s--", color=VERM, lw=1.8, ms=4.5,
                    label="NLL of the true diagnosis")
    ax2.set_ylabel("negative log-likelihood of truth", color=VERM)
    ax2.tick_params(axis="y", colors=VERM)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(RULE)
    ax2.spines["top"].set_visible(False)

    ta, tn = float(t[np.argmax(acc)]), float(t[np.argmin(nll)])
    ax.axvline(ta, color=BLUE, ls=(0, (2, 2)), lw=.9)
    ax.axvline(tn, color=VERM, ls=(0, (2, 2)), lw=.9)
    ax.annotate(f"accuracy-optimal\nτ = {ta:g}", (ta, .162), xytext=(6, 0),
                textcoords="offset points", fontsize=7.2, color=BLUE,
                linespacing=1.3, va="top")
    ax.annotate(f"calibration-optimal\nτ = {tn:g}", (tn, .052), xytext=(6, 0),
                textcoords="offset points", fontsize=7.2, color=VERM,
                linespacing=1.3, va="top")
    ax.axvspan(tn, ta, color=MUTED, alpha=.06, lw=0)
    ax.legend(handles=[ln1, ln2], loc="lower right")
    ax.set_title("τ is a frontier — invariance stays exact at every point")
    _grid(ax)
    _save(fig, "fig10_tau_frontier")


# =========================================================== Figure 11 =====
def fig11_competence():
    """C11: competence gates coherence."""
    d = pl.read_csv(TABLES / "competence.csv").sort("top1_accuracy")
    m = d["model"].to_list()
    v = d["top1_accuracy"].to_numpy()
    lo, hi = d["acc_ci_low"].to_numpy(), d["acc_ci_high"].to_numpy()
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(WIDE * .62, 3.6))
    ax.barh(y, v, color=[_c(k) for k in m], height=.6, zorder=3)
    ax.hlines(y, lo, hi, color=INK, lw=1.0, zorder=4)
    ax.axvline(CHANCE, color=VERM, lw=1.3, zorder=5)
    ax.text(CHANCE + .012, -.85, f"chance {CHANCE:.3f}", color=VERM,
            fontsize=7.2, fontweight=600)
    i = m.index("gpt-oss-20b") if "gpt-oss-20b" in m else None
    if i is not None:
        ax.scatter([0.0394], [i], marker="X", s=62, c=VERM, zorder=7)
        ax.annotate("before the C11 fix: 0.039\n(schema validity was 1.000)",
                    (0.0394, i), xytext=(14, -2), textcoords="offset points",
                    fontsize=7.2, color=VERM, linespacing=1.3,
                    arrowprops=dict(arrowstyle="-", color=VERM, lw=.7))
    ax.set_yticks(y); ax.set_yticklabels([SHORT[k] for k in m])
    ax.set_xlabel("top-1 accuracy, 49-way pick-one-diagnosis")
    ax.set_xlim(0, .76)
    ax.set_title("Competence precedes coherence")
    _grid(ax, "x")
    _save(fig, "fig11_competence")


# =========================================================== Figure 12 =====
def fig12_summary_matrix():
    """All four axioms on one grid, every column oriented so higher = worse.

    A heat matrix is only readable if colour means one thing throughout. Two
    of these quantities run the other way in their raw form and are recoded:
    A2 direction agreement is best when it is FAR from 0.5, so it enters as
    closeness-to-chance; and competence is best when it is HIGH, so it is not
    placed on this scale at all -- it is drawn beside the matrix with its own
    axis, because colouring "most accurate" in the same red as "least
    coherent" is exactly the misreading this figure has to avoid.
    """
    a1 = _a1()
    a2 = pl.read_csv(TABLES / "a2_main.csv")
    a3 = pl.read_csv(TABLES / "a3_main.csv")
    cm = pl.read_csv(TABLES / "competence.csv")
    order = a1.sort("normalised_order_effect", descending=True)["model"].to_list()

    def col(df, name):
        mp = {r["model"]: r[name] for r in df.iter_rows(named=True)}
        return np.array([mp.get(k, np.nan) for k in order])

    # Every entry below: larger number = worse coherence.
    dir_dist = np.abs(col(a2, "direction_agreement") - .5)
    cols = {
        "A1 order\neffect": col(a1, "normalised_order_effect"),
        "A1 top-1\nflip rate": col(a1, "top1_flip_rate"),
        "A2 closeness\nto chance": 1.0 - dir_dist / np.nanmax(dir_dist),
        "A3 spurious\nJSD": col(a3, "mean_spurious_jsd"),
        "A3 cases\naffected": col(a3, "frac_above_floor"),
        "A3 top-1\nflip rate": col(a3, "top1_flip_rate"),
    }
    raw = list(cols.values())
    M = np.vstack([v / np.nanmax(np.abs(v)) for v in raw]).T

    fig, (ax, axc) = plt.subplots(
        1, 2, figsize=(WIDE, 4.3), sharey=True,
        gridspec_kw=dict(width_ratios=[6, 1.15], wspace=.06))

    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols.keys(), fontsize=7.4, linespacing=1.35)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[k] for k in order])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{raw[j][i]:.2f}", ha="center", va="center",
                    fontsize=6.9,
                    color="white" if M[i, j] > .70 or M[i, j] < .10 else INK)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.3)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Every model fails every coherence axiom; only the magnitude "
                 "varies", fontsize=9.5)

    # Competence sits on its own axis, deliberately not on the red scale.
    comp = col(cm, "top1_accuracy")
    axc.barh(np.arange(len(order)), comp, height=.62,
             color=[_c(k) for k in order], zorder=3)
    axc.axvline(CHANCE, color=VERM, lw=1.1, zorder=4)
    for i, v in enumerate(comp):
        if np.isfinite(v):
            axc.text(v + .015, i, f"{v:.2f}", va="center", fontsize=6.9,
                     color=INK)
    axc.set_xlim(0, .92)
    # No invert_yaxis(): the axis is shared with imshow, which already places
    # row 0 at the top. Inverting it here would silently reverse the sort.
    axc.set_xlabel("top-1 accuracy", fontsize=7.6)
    axc.set_title("competence\n(higher is better)", fontsize=8,
                  color=MUTED, loc="center")
    axc.grid(axis="x", color=RULE, lw=.55)
    axc.set_axisbelow(True)
    axc.tick_params(labelsize=7)

    cb = fig.colorbar(im, ax=axc, fraction=.09, pad=.42)
    cb.set_label("column-normalised severity  (higher = worse)", fontsize=7.2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    _save(fig, "fig12_summary_matrix")


# =========================================================== Figure 13 =====
def fig13_severity():
    """High-severity cases are not protected."""
    a1 = _a1()
    sv = pl.read_csv(TABLES / "a1_severity.csv")
    mp = {r["model"]: r["high_severity_topk_instability"]
          for r in sv.iter_rows(named=True)}
    d = a1.sort("normalised_order_effect")
    m = d["model"].to_list()
    y = np.arange(len(m))
    v = np.array([mp.get(k, np.nan) for k in m])
    fig, ax = plt.subplots(figsize=(WIDE * .62, 3.5))
    ax.barh(y, v, color=[_c(k) for k in m], height=.62, zorder=3)
    for i in range(len(m)):
        ax.text(v[i] - .02, y[i], f"{v[i]:.2f}", va="center", ha="right",
                fontsize=7.2, color="white", fontweight=600)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[k] for k in m])
    ax.set_xlabel("share of cases whose high-severity top-5 set changes")
    ax.set_xlim(0, 1.05)
    ax.set_title("Reordering destabilises the high-severity differential")
    _grid(ax, "x")
    _save(fig, "fig13_severity")


# =========================================================== Figure 14 =====
def fig14_accuracy_vs_coherence():
    """Competence and coherence are close to orthogonal."""
    a1 = _a1()
    cm = pl.read_csv(TABLES / "competence.csv")
    mp = {r["model"]: r["top1_accuracy"] for r in cm.iter_rows(named=True)}
    m = a1["model"].to_list()
    x = np.array([mp.get(k, np.nan) for k in m])
    yv = a1["normalised_order_effect"].to_numpy()
    fig, ax = plt.subplots(figsize=(WIDE * .62, 3.6))
    ax.scatter(x, yv, s=62, c=[_c(k) for k in m], zorder=3,
               edgecolors="white", lw=.6)
    for i, mk in enumerate(m):
        ax.annotate(SHORT[mk], (x[i], yv[i]), xytext=(6, 3),
                    textcoords="offset points", fontsize=6.8, color=MUTED)
    ok = np.isfinite(x) & np.isfinite(yv)
    if ok.sum() > 2:
        r = np.corrcoef(x[ok], yv[ok])[0, 1]
        ax.annotate(f"Pearson r = {r:+.2f}  (n = {int(ok.sum())})",
                    xy=(.97, .95), xycoords="axes fraction", ha="right",
                    fontsize=7.6, color=INK, fontweight=600)
    ax.set_xlabel("top-1 diagnostic accuracy")
    ax.set_ylabel("normalised order effect")
    ax.set_xlim(0, .72)
    ax.set_title("Being right and being coherent are largely separate axes")
    _grid(ax)
    _save(fig, "fig14_accuracy_vs_coherence")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("writing publication figures:")
    for fn in (fig1_anchors, fig2_scale_reversal, fig3_percase_distribution,
               fig4_k_sweep, fig5_a2_direction, fig6_a2_slope,
               fig7_a3_redundancy, fig8_a4_position, fig9_methods_pareto,
               fig10_tau_frontier, fig11_competence, fig12_summary_matrix,
               fig13_severity, fig14_accuracy_vs_coherence):
        try:
            fn()
        except Exception as e:                       # keep the suite going
            print(f"  [skip] {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
