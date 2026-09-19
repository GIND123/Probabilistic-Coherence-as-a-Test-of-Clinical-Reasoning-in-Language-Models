"""Explainer figures: the measurement pipeline, and one case carried through it.

Two figures aimed at a reader who has not seen the instrument before.

`fig_pipeline` is the data-and-prediction diagram: where the data comes from,
what is held fixed, what is varied, what the model is asked, and which
comparison each axiom makes. Everything on it is a real quantity from this
study, not a schematic placeholder.

`fig_worked_example` carries one patient through that pipeline. The case is
chosen for legibility, not for effect size: patient 09c27173dfc592e6 is a
textbook Guillain-Barre presentation, and the same seven findings in ten
different orders produce six different leading diagnoses.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coherence.analysis.figures_results import (BLUE, GREEN, INK, MUTED, ORANGE,
                                                PURPLE, RULE, VERM, WIDE, _save)
from coherence.axioms.loading import load_task, posterior_matrix
from coherence.data.battery import Battery
from coherence.metrics.divergence import mean_pairwise_jsd

BATTERY = "build/battery/codx_battery_v1.json"
CASE_ID = "09c27173dfc592e6"
MODEL = "qwen3-32b-nothink"

FINDINGS = [
    "Recently had a viral infection",
    "Shortness of breath, significant",
    "Weakness in facial muscles and/or eyes",
    "Weakness in both arms and/or both legs",
    "Numbness / tingling in the feet",
    "Weakness or paralysis on one side of the face",
    "No travel outside the country in 4 weeks",
]


def _box(ax, x, y, w, h, text, fc="white", ec=RULE, fs=7.4, weight=400,
         tc=INK, lw=1.0, pad=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={pad},rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, linespacing=1.45, fontweight=weight)


def _arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, color=color, lw=lw,
                                 shrinkA=2, shrinkB=2, zorder=1))


# ====================================================== pipeline diagram =====
def fig_pipeline():
    """The data-and-prediction diagram.

    Laid out on an explicit vertical budget. FancyBboxPatch adds its `pad` in
    DATA units on every side, so on a 0-1 axis a pad of 0.02 silently grows a
    box by 4% in each dimension -- enough to push it under the next section
    header. The pad here is 0.004 and every row has a stated gap.
    """
    fig, ax = plt.subplots(figsize=(WIDE, 6.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    PAD = .004

    def box(x, y, w, h, text, ec=RULE, fc="white", fs=7.2):
        _box(ax, x, y, w, h, text, fc=fc, ec=ec, fs=fs, pad=PAD)

    def header(y, text):
        ax.text(.01, y, text, fontsize=7.5, fontweight=600, color=INK, va="center")

    ax.text(.01, .975, "How one patient becomes a coherence measurement",
            fontsize=11, fontweight=600, color=INK, va="center")
    ax.text(.01, .942,
            "Every box is a measured quantity from this study. The evidence SET is "
            "held fixed throughout; only its ORDER varies.",
            fontsize=7.4, color=MUTED, va="center")

    # ---- row 1: corpus -> battery -> oracle  (y 0.845-0.905)
    box(.01, .845, .285, .060,
        "DDXPlus corpus\n1,292,579 patients\n49 pathologies · 223 findings",
        ec=BLUE, fc="#F3F7FB")
    box(.355, .845, .285, .060,
        "CoDx battery\n1,956 patients sampled\nmedian 19 findings each",
        ec=BLUE, fc="#F3F7FB")
    box(.705, .845, .285, .060,
        "Empirical oracle\nP(d | E) conditioned on the corpus\n≥500 support · Wilson CI",
        ec=BLUE, fc="#F3F7FB", fs=6.9)
    _arrow(ax, .300, .875, .350, .875, color=BLUE)
    _arrow(ax, .645, .875, .700, .875, color=BLUE)

    # ---- row 2: the four arms  (header 0.800, boxes 0.715-0.775)
    header(.800, "ONE PATIENT — the evidence set {e₁ … e₇} never changes")
    arms = [(.01, "permutation\nK = 10 random orders", ORANGE, "varies ORDER"),
            (.255, "retest\n6 replicates, one order", PURPLE, "varies NOTHING"),
            (.500, "canonical\n2 replicates, released order", GREEN, "fixed order"),
            (.745, "between-patient\ndifferent patients", MUTED, "varies PATIENT")]
    for x, label, col, sub in arms:
        box(x, .715, .225, .060, label, ec=col, fs=7.0)
        ax.text(x + .1125, .695, sub, ha="center", fontsize=6.5, color=col,
                fontweight=600)
    _arrow(ax, .4975, .843, .4975, .779, color=MUTED)

    # ---- row 3: three elicitation paths  (header 0.650, boxes 0.560-0.625)
    header(.650, "THREE WAYS TO ASK THE SAME MODEL")
    paths = [(.01, "stated posterior\n49 plausibility scores 0–100,\nnormalised afterwards",
              "top-1  0.023 – 0.355", VERM),
             (.355, "named diagnosis\na single index, 0–48",
              "top-1  0.081 – 0.651", BLUE),
             (.700, "token logprob\nscore all 49 names,\nno verbalised number",
              "top-1  0.200 (pilot)", GREEN)]
    for x, label, res, col in paths:
        box(x, .560, .290, .065, label, ec=col, fs=6.9)
        ax.text(x + .145, .540, res, ha="center", fontsize=6.7, color=col,
                fontweight=600)
    # A single stem rather than a fan: the fan's left branch crossed the
    # section header, and "every arm is asked all three ways" is already
    # carried by the header itself.
    _arrow(ax, .4975, .712, .4975, .628, color=MUTED, lw=.9)

    # ---- row 4: what each axiom compares  (header 0.490)
    header(.490, "WHAT EACH AXIOM COMPARES")
    rows = [("A1  order invariance",
             "JSD across the 10 orders   −   JSD across the 6 retests", ORANGE),
            ("A2  belief updating",
             "model Δlog-odds when a finding is added   vs   oracle Δlog-odds", BLUE),
            ("A3  redundancy",
             "movement caused by evidence the corpus certifies uninformative", GREEN),
            ("A4  positional anchoring",
             "between-position JSD   −   within-position JSD", PURPLE)]
    y = .443
    for name, comp, col in rows:
        ax.add_patch(FancyBboxPatch((.012, y - .013), .011, .026,
                                    boxstyle="round,pad=0.001",
                                    facecolor=col, edgecolor="none", zorder=3))
        ax.text(.035, y, name, fontsize=7.1, fontweight=600, color=INK,
                va="center")
        ax.text(.315, y, comp, fontsize=6.9, color=MUTED, va="center")
        y -= .042

    # ---- row 5: remedies  (header 0.230, boxes 0.130-0.195)
    header(.230, "REMEDIES COMPARED   (Qwen3-32B, paired over cases)")
    rem = [("direct", "1 query", "0.2988", VERM),
           ("perm-ensemble K = 5", "5 queries", "0.1038", BLUE),
           ("ELR-Fusion", "0 marginal", "0.0000", PURPLE),
           ("fixed canonical order", "1 query", "0.0000", GREEN)]
    x = .01
    for name, cost, resid, col in rem:
        box(x, .130, .225, .065, f"{name}\n{cost}\nresidual {resid}", ec=col, fs=6.9)
        x += .245
    ax.text(.01, .098,
            "residual = remaining order sensitivity in JSD; lower is better.",
            fontsize=6.6, color=MUTED, va="center")

    # ---- the finding that reframes the comparison
    ax.add_patch(FancyBboxPatch((.01, .012), .98, .066,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                facecolor="#FBF3EE", edgecolor=VERM, lw=1.0,
                                zorder=2))
    ax.text(.5, .045,
            "A fixed canonical order reaches the same exact zero at −0.012 to +0.021 top-1 cost, i.e. free.\n"
            "The separating quantity is not the residual but the spread ACROSS fixed orders: 0.120–0.258 JSD.",
            ha="center", va="center", fontsize=7.0, color=INK, zorder=3,
            linespacing=1.55)
    _save(fig, "fig15_pipeline")


# =================================================== worked example ==========
def fig_worked_example():
    b = Battery.load(BATTERY)
    paths = list(b.pathologies)
    pidx = {p: i for i, p in enumerate(paths)}
    case = {c.case_id: c for c in b.cases}[CASE_ID]
    ti = pidx[case.pathology]

    d = load_task(MODEL, "a1_posterior")
    P, _ = posterior_matrix(d)
    d = d.with_row_index("row")
    g = d.filter(pl.col("case_id") == CASE_ID)
    per = g.filter((pl.col("arm") == "permutation") & pl.col("valid")).sort("perm_index")
    ret = g.filter((pl.col("arm") == "retest") & pl.col("valid"))
    M = P[per["row"].to_numpy()]
    Mr = P[ret["row"].to_numpy()]

    fig = plt.figure(figsize=(WIDE, 6.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.30],
                          height_ratios=[1.32, 1.0], hspace=.60, wspace=.30)

    # ---- panel A: the fixed evidence set
    axA = fig.add_subplot(gs[0, 0]); axA.axis("off")
    axA.set_xlim(0, 1); axA.set_ylim(0, 1)
    axA.set_title("A   The evidence set (identical everywhere)",
                  fontsize=8.6, loc="left", pad=7)
    axA.text(0, .98, f"19-year-old female\ntrue diagnosis: {case.pathology}",
             fontsize=7.4, color=INK, fontweight=600, va="top", linespacing=1.5)
    for i, f in enumerate(FINDINGS):
        yy = .795 - i * .104
        axA.add_patch(FancyBboxPatch((0, yy - .045), .045, .072,
                                     boxstyle="round,pad=0.004",
                                     facecolor=ORANGE, edgecolor="none"))
        axA.text(.0225, yy - .009, f"e{i+1}", ha="center", va="center",
                 fontsize=6.6, color="white", fontweight=600)
        axA.text(.065, yy - .009, f, fontsize=7.1, color=INK, va="center")
    axA.text(0, -.055, "Textbook presentation: post-viral, ascending\n"
                        "bilateral weakness, facial weakness, distal\nparaesthesia.",
             fontsize=6.8, color=MUTED, va="top", linespacing=1.55)

    # ---- panel B: top-1 per ordering
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("B   Ten orders of that set — six different answers",
                  fontsize=8.6, loc="left", pad=7)
    top1 = M.argmax(1)
    conf = M.max(1)
    uniq = list(dict.fromkeys(top1.tolist()))
    pal = [GREEN if u == ti else c for u, c in
           zip(uniq, [VERM, BLUE, ORANGE, PURPLE, "#8C6D31", "#7A7A7A", MUTED])]
    cmap = {u: (GREEN if u == ti else pal[k]) for k, u in enumerate(uniq)}
    y = np.arange(len(M))
    axB.barh(y, conf, color=[cmap[t] for t in top1], height=.66, zorder=3)
    for k in range(len(M)):
        nm = paths[top1[k]]
        nm = nm if len(nm) <= 34 else nm[:32] + "…"
        axB.text(conf[k] + .015, y[k], f"{nm}  {conf[k]:.2f}",
                 va="center", fontsize=6.8,
                 color=cmap[top1[k]], fontweight=600 if top1[k] == ti else 400)
    axB.set_yticks(y); axB.set_yticklabels([f"order {k+1}" for k in y], fontsize=7)
    axB.set_ylim(len(M) - .35, -.75)      # headroom so row 1 clears the title
    axB.set_xlim(0, 1.52); axB.set_xticks([0, .25, .5, .75, 1.0])
    axB.set_xlabel("probability assigned to the leading diagnosis", fontsize=7.6)
    axB.grid(axis="x", color=RULE, lw=.55); axB.set_axisbelow(True)
    axB.spines["right"].set_visible(False); axB.spines["top"].set_visible(False)

    # ---- panel C: probability of the truth
    axC = fig.add_subplot(gs[1, 0])
    axC.set_title("C   Probability on the correct diagnosis",
                  fontsize=8.6, loc="left", pad=7)
    tp = M[:, ti]
    axC.plot(np.arange(1, len(tp) + 1), tp, "o-", color=GREEN, lw=1.7, ms=5,
             zorder=3, label="across 10 orders")
    rt = Mr[:, ti]
    axC.plot(np.arange(1, len(rt) + 1), rt, "s--", color=MUTED, lw=1.2, ms=4,
             zorder=2, label="across 6 retests (order fixed)")
    axC.set_xlabel("condition index", fontsize=7.6)
    axC.set_ylabel("P(Guillain-Barré)", fontsize=7.6)
    axC.set_ylim(-.05, 1.08)
    axC.legend(fontsize=6.8, loc="upper left")
    axC.grid(color=RULE, lw=.55); axC.set_axisbelow(True)
    axC.annotate(f"{tp.min():.3f} → {tp.max():.3f}\nfrom reordering alone",
                 xy=(.74, .46), xycoords="axes fraction", ha="center",
                 fontsize=7, color=GREEN, fontweight=600, linespacing=1.4)

    # ---- panel D: the dissociation
    axD = fig.add_subplot(gs[1, 1]); axD.axis("off")
    # Pin the frame: the rule lines below are drawn with plot(), which
    # autoscales and would otherwise collapse this panel's 0-1 coordinate
    # space onto the range of those rules, pushing every row into the title.
    axD.set_xlim(0, 1); axD.set_ylim(0, 1)
    axD.set_title("D   It names the diagnosis correctly every time",
                  fontsize=8.6, loc="left", pad=7)
    pj, rj = mean_pairwise_jsd(M), mean_pairwise_jsd(Mr)
    lines = [
        ("Named diagnosis (one index asked)", "Guillain-Barré 18 / 18", GREEN),
        ("Stated posterior ranks it first", f"{int((top1 == ti).sum())} / {len(M)} orders", VERM),
        ("Mean pairwise JSD across orders", f"{pj:.3f}", ORANGE),
        ("Mean pairwise JSD, retests (floor)", f"{rj:.3f}", MUTED),
        ("Order effect above this model's floor", f"{pj - rj:.3f}", INK),
    ]
    yy = .690
    for lab, val, col in lines:
        axD.text(0, yy, lab, fontsize=7.3, color=INK, va="center")
        axD.text(1.0, yy, val, fontsize=7.6, color=col, va="center",
                 ha="right", fontweight=600)
        axD.plot([0, 1.0], [yy - .068, yy - .068], color=RULE, lw=.7)
        yy -= .150
    axD.text(0, -.025,
             "The information is present: the model states it when asked for a\n"
             "name. What moves with order is the probability it puts on it.",
             fontsize=6.9, color=MUTED, va="top", linespacing=1.55)

    _save(fig, "fig16_worked_example")


def main():
    print("writing explainer figures:")
    fig_pipeline()
    fig_worked_example()


if __name__ == "__main__":
    main()
