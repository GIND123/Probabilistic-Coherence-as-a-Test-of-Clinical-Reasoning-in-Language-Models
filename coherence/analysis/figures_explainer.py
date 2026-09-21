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


def _vpt(fig, pts):
    """Points -> axes fraction, vertically. Layout is specified in points so
    that spacing survives a change of figure size or of installed font."""
    return pts / (fig.get_figheight() * 72.0)


def _hpt(fig, pts):
    return pts / (fig.get_figwidth() * 72.0)


def _extent(ax, t):
    """A drawn text's bounding box, in axes fraction."""
    bb = t.get_window_extent(ax.figure.canvas.get_renderer())
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(
        [[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x0, y0, x1, y1


def _fit(ax, t, maxw, floor=5.8):
    """Shrink a text until it is no wider than `maxw` (axes fraction).

    The first version of this figure sized every box by hand, which is only
    correct for the font metrics of the machine that drew it: under a
    substituted font the text ran outside its frame. Nothing here is sized by
    hand -- text is drawn, measured, and the box is fitted to it.
    """
    fig = ax.figure
    fig.canvas.draw()
    x0, _, x1, _ = _extent(ax, t)
    while x1 - x0 > maxw and t.get_fontsize() > floor:
        t.set_fontsize(t.get_fontsize() - 0.15)
        fig.canvas.draw()
        x0, _, x1, _ = _extent(ax, t)
    return t


def _stack(ax, cx, top, maxw, segs, lead=3.0):
    """A centred vertical stack of (text, size, colour, weight) segments."""
    fig = ax.figure
    ts, y = [], top
    for s, fs, col, wt in segs:
        t = ax.text(cx, y, s, ha="center", va="top", fontsize=fs, color=col,
                    fontweight=wt, linespacing=1.42, zorder=3, clip_on=False)
        _fit(ax, t, maxw)
        y = _extent(ax, t)[1] - _vpt(fig, lead)
        ts.append(t)
    return ts, y + _vpt(fig, lead)


def _cell_row(ax, top, xs, w, cells, colours, face="white", padx=7.0,
              pady=6.0, lead=3.0):
    """A row of boxes whose height is measured from their own contents.

    Every box in the row takes the height of the tallest, and a shorter stack
    is re-centred inside it, so the row reads as a row without any box
    cropping its own text.
    """
    fig = ax.figure
    built = []
    for x, segs in zip(xs, cells):
        ts, bot = _stack(ax, x + w / 2, top - _vpt(fig, pady),
                         w - 2 * _hpt(fig, padx), segs, lead)
        built.append((x, ts, bot))
    h = top - min(b for _, _, b in built) + _vpt(fig, pady)
    inner = top - h + _vpt(fig, pady)
    for (x, ts, bot), col in zip(built, colours):
        slack = (bot - inner) / 2.0
        if slack > 1e-4:
            for t in ts:
                t.set_y(t.get_position()[1] - slack)
        ax.add_patch(FancyBboxPatch((x, top - h), w, h,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor=face, edgecolor=col, lw=1.0,
                                    zorder=2, clip_on=False))
    return top - h


def _arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=8, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=1,
                                 clip_on=False))


def _rule(ax, x1, y1, x2, y2, color=MUTED, lw=1.0):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=1, clip_on=False,
            solid_capstyle="round")


def _fan(ax, src, bus_y, xs, dst_y, color=MUTED, lw=1.0):
    """One source, many targets: a drop, a horizontal bus, one head each.

    Connectors run in the gutters between boxes and never cross one, which the
    single stem in the first version did.
    """
    sx, sy = src
    _rule(ax, sx, sy, sx, bus_y, color, lw)
    _rule(ax, min(xs + [sx]), bus_y, max(xs + [sx]), bus_y, color, lw)
    for x in xs:
        _arrow(ax, x, bus_y, x, dst_y, color=color, lw=lw)


def _merge(ax, xs, src_y, bus_y, color=MUTED, lw=1.0):
    """Many sources, one bus: the collector under a row of arms."""
    for x in xs:
        _rule(ax, x, src_y, x, bus_y, color, lw)
    _rule(ax, min(xs), bus_y, max(xs), bus_y, color, lw)


# ====================================================== pipeline diagram =====
def fig_pipeline():
    """The data-and-prediction diagram.

    Laid out top-down from a cursor carried in points: each row is drawn,
    measured, and the cursor moves to its true bottom edge. No coordinate in
    this function is a guess about how tall a piece of text will render, which
    is what produced the overlaps in the first version.
    """
    fig, ax = plt.subplots(figsize=(WIDE, 6.95))
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_autoscale_on(False)          # plot() must not rescale the frame

    def V(pts):
        return _vpt(fig, pts)

    def header(top, text, note=None):
        t = ax.text(.01, top, text, fontsize=7.6, fontweight=600, color=INK,
                    va="top", zorder=3, clip_on=False)
        if note:
            ax.text(.99, top, note, fontsize=6.8, color=MUTED, ha="right",
                    va="top", zorder=3, clip_on=False)
        fig.canvas.draw()
        return _extent(ax, t)[1]

    # ------------------------------------------------------------- title ---
    t = ax.text(.01, .992, "How one patient becomes a coherence measurement",
                fontsize=11.5, fontweight=600, color=INK, va="top",
                clip_on=False)
    fig.canvas.draw()
    y = _extent(ax, t)[1] - V(5)
    t = ax.text(.01, y, "Every box is a measured quantity from this study. The "
                "evidence SET is held fixed throughout; only its ORDER varies.",
                fontsize=7.4, color=MUTED, va="top", clip_on=False)
    _fit(ax, t, .98)
    y = _extent(ax, t)[1] - V(16)

    # -------------------------------------- row 1: corpus, battery, oracle --
    xs1, w1 = [.01, .35, .69], .30
    top1 = y
    y = _cell_row(ax, y, xs1, w1, [
        [("DDXPlus corpus", 7.4, BLUE, 600),
         ("1,292,579 patients\n49 pathologies · 223 findings", 7.0, INK, 400)],
        [("CoDx battery", 7.4, BLUE, 600),
         ("1,956 patients sampled\nmedian 19 findings each", 7.0, INK, 400)],
        [("Empirical oracle", 7.4, BLUE, 600),
         ("P(d | E) from corpus counts\n≥ 500 support · Wilson CI",
          7.0, INK, 400)],
    ], [BLUE] * 3, face="#F2F7FB")
    mid = (top1 + y) / 2
    _arrow(ax, .314, mid, .346, mid, color=BLUE)
    _arrow(ax, .654, mid, .686, mid, color=BLUE)

    # ----------------------------------------------- row 2: the four arms ---
    hb = header(y - V(13),
                "ONE PATIENT — the evidence set never changes",
                "per patient: 10 orders · 6 retests · 2 canonical")
    xs2, w2 = [.01, .262, .514, .766], .224
    c2 = [x + w2 / 2 for x in xs2]
    bus = hb - V(11)
    top2 = bus - V(11)
    _fan(ax, (.50, y), bus, c2, top2)
    y = _cell_row(ax, top2, xs2, w2, [
        [("permutation", 7.2, ORANGE, 600),
         ("K = 10 random orders", 7.0, INK, 400),
         ("varies ORDER", 6.6, ORANGE, 600)],
        [("retest", 7.2, PURPLE, 600),
         ("6 replicates, one order", 7.0, INK, 400),
         ("varies NOTHING", 6.6, PURPLE, 600)],
        [("canonical", 7.2, GREEN, 600),
         ("2 replicates, released order", 7.0, INK, 400),
         ("fixed order", 6.6, GREEN, 600)],
        [("between-patient", 7.2, MUTED, 600),
         ("different patients", 7.0, INK, 400),
         ("varies PATIENT", 6.6, MUTED, 600)],
    ], [ORANGE, PURPLE, GREEN, MUTED])

    # --------------------------------- row 3: three ways to ask the model ---
    collect = y - V(9)
    _merge(ax, c2, y, collect)
    hb = header(collect - V(13), "THREE WAYS TO ASK THE SAME MODEL",
                "every arm is asked all three ways")
    xs3, w3 = [.01, .3483, .6867], .3033
    c3 = [x + w3 / 2 for x in xs3]
    bus = hb - V(11)
    top3 = bus - V(11)
    _fan(ax, (.50, collect), bus, c3, top3)
    y = _cell_row(ax, top3, xs3, w3, [
        [("stated posterior", 7.2, VERM, 600),
         ("49 plausibility ratings, 0–100\nnormalised to a simplex",
          7.0, INK, 400),
         ("top-1  0.023 – 0.355", 6.8, VERM, 600)],
        [("named diagnosis", 7.2, BLUE, 600),
         ("one label, asked on its own\nno distribution requested",
          7.0, INK, 400),
         ("top-1  0.081 – 0.651", 6.8, BLUE, 600)],
        [("token logprob", 7.2, GREEN, 600),
         ("all 49 names scored\nno verbalised number", 7.0, INK, 400),
         ("top-1  0.068 – 0.324", 6.8, GREEN, 600)],
    ], [VERM, BLUE, GREEN])
    t = ax.text(.01, y - V(7),
                "top-1 spans the 11 models: 35,208 A1 items for the first two "
                "paths, 1,956 cases for the third (length-normalised;\nsummed "
                "log-probability gives 0.020 – 0.511 on the same runs, so "
                "both scoring rules are reported).",
                fontsize=6.6, color=MUTED, va="top", linespacing=1.5,
                clip_on=False)
    _fit(ax, t, .98)
    fig.canvas.draw()
    y = _extent(ax, t)[1]

    # ---------------------------------------- row 4: what each axiom asks ---
    y = header(y - V(15), "WHAT EACH AXIOM COMPARES") - V(10)
    rows = [("A1  order invariance",
             "JSD across the 10 orders   −   JSD across the 6 retests",
             ORANGE),
            ("A2  belief updating",
             "model Δlog-odds when a finding is added   vs   oracle "
             "Δlog-odds", BLUE),
            ("A3  redundancy",
             "movement caused by evidence the corpus certifies uninformative",
             GREEN),
            ("A4  positional anchoring",
             "between-position JSD   −   within-position JSD", PURPLE)]
    for name, comp, col in rows:
        ax.add_patch(FancyBboxPatch((.012, y - V(4.6)), .0095, V(9.2),
                                    boxstyle="round,pad=0,rounding_size=0.003",
                                    facecolor=col, edgecolor="none", zorder=3,
                                    clip_on=False))
        ax.text(.033, y, name, fontsize=7.1, fontweight=600, color=INK,
                va="center", clip_on=False)
        _fit(ax, ax.text(.30, y, comp, fontsize=6.9, color=MUTED, va="center",
                         clip_on=False), .68)
        y -= V(15.5)
    y += V(15.5)

    # --------------------------------------------- row 5: the remedies ------
    y = header(y - V(16), "REMEDIES COMPARED",
               "Qwen3-32B, paired over 1,954 cases") - V(10)
    y = _cell_row(ax, y, xs2, w2, [
        [("direct", 7.2, VERM, 600),
         ("1 query\nresidual 0.2988", 7.0, INK, 400),
         ("top-1  0.189", 6.8, VERM, 600)],
        [("perm-ensemble K = 5", 7.2, BLUE, 600),
         ("5 queries\nresidual 0.1038", 7.0, INK, 400),
         ("top-1  0.238", 6.8, BLUE, 600)],
        [("ELR-Fusion", 7.2, PURPLE, 600),
         ("0 marginal, 497 fixed\nresidual 0.0000", 7.0, INK, 400),
         ("top-1  0.133", 6.8, PURPLE, 600)],
        [("fixed canonical order", 7.2, GREEN, 600),
         ("1 query\nresidual 0.0000", 7.0, INK, 400),
         ("top-1  0.192", 6.8, GREEN, 600)],
    ], [VERM, BLUE, PURPLE, GREEN])
    t = ax.text(.01, y - V(7),
                "residual = order sensitivity still present, in JSD; lower is "
                "better. ELR-Fusion's fixed cost is 497 elicitation queries "
                "per model, paid once.",
                fontsize=6.6, color=MUTED, va="top", clip_on=False)
    _fit(ax, t, .98)

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
