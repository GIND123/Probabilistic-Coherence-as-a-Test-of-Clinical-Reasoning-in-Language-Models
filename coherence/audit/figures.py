"""Audit figures. One file per figure, vector PDF plus PNG for the report."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from coherence.config import FIGURES, RESULTS
from coherence.data.ddxplus import load_all, load_kb

# A restrained, colour-blind-safe palette; severity uses a single hue ramp.
INK = "#1b1b1f"
MUTED = "#6b7280"
ACCENT = "#2166ac"
WARN = "#b2182b"
GRID = "#e5e7eb"
SEQ = ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f4a582"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  figure: {name}")


def fig_pathology_prior(df: pl.DataFrame, kb) -> None:
    c = (df.group_by("pathology").len().sort("len", descending=True)
         .with_columns(pl.col("pathology").replace_strict(kb.severity, default=3).alias("sev")))
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    y = np.arange(c.height)
    colors = [SEQ[min(int(s) - 1, 4)] for s in c["sev"]]
    ax.barh(y, c["len"].to_numpy(), color=colors, height=0.72)
    ax.set_yticks(y); ax.set_yticklabels(c["pathology"].to_list(), fontsize=7)
    ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel("patients (log scale)")
    ax.set_title("DDXPlus pathology prior spans 252x, bar colour = severity (1 = most severe)",
                 loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEQ[i]) for i in range(5)]
    ax.legend(handles, [f"severity {i+1}" for i in range(5)], fontsize=7,
              loc="lower right", ncol=1)
    _save(fig, "audit_pathology_prior")


def fig_generator_noise(findings: dict) -> None:
    s = findings["generator_determinism"]["stats"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    keys = ["jsd_median", "jsd_mean", "jsd_p90", "jsd_p99", "jsd_max"]
    labels = ["median", "mean", "p90", "p99", "max"]
    vals = [s[k] for k in keys]
    ax.bar(labels, vals, color=[MUTED, WARN, ACCENT, ACCENT, ACCENT], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f" {v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Jensen–Shannon divergence (base 2)")
    ax.set_title(
        f"Ground-truth noise floor: identical DDXPlus cases get different differentials\n"
        f"{s['n_replicate_pairs']:,} replicate pairs, {100*s['frac_pairs_disagreeing']:.1f}% "
        f"disagree, top-1 flips {100*s['top1_flip_rate']:.2f}%", loc="left")
    ax.set_ylim(0, max(vals) * 1.18)
    _save(fig, "audit_generator_noise_floor")


def fig_oracle_feasibility(findings: dict) -> None:
    s = findings["empirical_oracle_feasibility"]["stats"]["by_k"]
    ks = sorted(int(k) for k in s)
    frac = [s[str(k)]["frac_scorable"] * 100 for k in ks]
    med = [s[str(k)]["median_support"] for k in ks]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.4))
    a1.plot(ks, frac, "o-", color=ACCENT, lw=1.8, ms=5)
    a1.axhline(80, color=WARN, ls="--", lw=1)
    a1.text(ks[-1], 81, "80% design threshold", ha="right", fontsize=7.5, color=WARN)
    a1.set_xlabel("evidence set size |E|"); a1.set_ylabel("% of items scorable")
    a1.set_title("Assumption-free oracle stays usable to |E|≈6", loc="left")
    a2.plot(ks, med, "o-", color=ACCENT, lw=1.8, ms=5)
    a2.axhline(500, color=WARN, ls="--", lw=1)
    a2.text(ks[-1], 560, "500-patient minimum support", ha="right", fontsize=7.5, color=WARN)
    a2.set_yscale("log"); a2.set_xlabel("evidence set size |E|")
    a2.set_ylabel("median matching patients")
    a2.set_title("Support decays geometrically with |E|", loc="left")
    _save(fig, "audit_oracle_feasibility")


def fig_evidence_load(df: pl.DataFrame) -> None:
    k = df.select(pl.col("evidences").list.len().alias("n"))["n"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.hist(k, bins=np.arange(0, k.max() + 2) - 0.5, color=ACCENT, alpha=0.85,
            edgecolor="white", linewidth=0.3)
    q1, q3 = np.percentile(k, [25, 75]); fence = q3 + 1.5 * (q3 - q1)
    ax.axvline(np.median(k), color=INK, lw=1.2, label=f"median {np.median(k):.0f}")
    ax.axvline(fence, color=WARN, lw=1.2, ls="--",
               label=f"Tukey fence {fence:.0f} ({(k>fence).sum():,} cases)")
    ax.set_xlabel("findings per patient"); ax.set_ylabel("patients")
    ax.set_title("Evidence load sets the ELR-Fusion query budget (n queries per case)",
                 loc="left")
    ax.legend(fontsize=8)
    _save(fig, "audit_evidence_load")


def fig_ddx_shape(df: pl.DataFrame) -> None:
    s = df.select(pl.col("ddx_names").list.len().alias("k"))["k"].to_numpy()
    ent = df.select(
        pl.col("ddx_probs").map_elements(
            lambda p: float(-sum(x * np.log2(x) for x in p if x > 0)),
            return_dtype=pl.Float64).alias("h"))["h"].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.2))
    a1.hist(s, bins=np.arange(0, s.max() + 2) - 0.5, color=ACCENT, alpha=0.85,
            edgecolor="white", linewidth=0.3)
    a1.set_xlabel("pathologies in the differential"); a1.set_ylabel("patients")
    a1.set_title(f"Differential support (median {np.median(s):.0f} of 49)", loc="left")
    a2.hist(ent, bins=60, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.3)
    a2.axvline(ent.mean(), color=WARN, lw=1.2, label=f"mean {ent.mean():.2f} bits")
    a2.set_xlabel("entropy of the ground-truth posterior (bits)"); a2.set_ylabel("patients")
    a2.set_title("Targets are genuinely distributional, not one-hot", loc="left")
    a2.legend(fontsize=8)
    _save(fig, "audit_differential_shape")


def fig_independence(findings: dict) -> None:
    s = findings["naive_bayes_reconstruction"]["stats"]
    ind, val = s["independence"], s["validation"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.3))
    labels = ["mean |φ|", "median |φ|", "p95 |φ|", "max |φ|"]
    vals = [ind["phi_mean_abs"], ind["phi_median_abs"], ind["phi_p95_abs"], ind["phi_max_abs"]]
    a1.bar(labels, vals, color=[MUTED, MUTED, ACCENT, WARN], width=0.6)
    for i, v in enumerate(vals):
        a1.text(i, v, f" {v:.3f}", ha="center", va="bottom", fontsize=8)
    a1.set_ylabel("|φ| between finding pairs, within pathology")
    a1.set_title(f"Conditional independence is violated\n"
                 f"{100*ind['frac_pairs_abs_phi_gt_0.1']:.1f}% of "
                 f"{ind['n_pairs_tested']:,} pairs exceed |φ|=0.1", loc="left")
    bars = ["generator\nnoise floor", "naive-Bayes\nreconstruction"]
    vals2 = [s["generator_noise_floor_jsd"], val["jsd_mean"]]
    a2.bar(bars, vals2, color=[ACCENT, WARN], width=0.55)
    for i, v in enumerate(vals2):
        a2.text(i, v, f" {v:.3f}", ha="center", va="bottom", fontsize=8)
    a2.set_ylabel("mean JSD vs shipped differential")
    a2.set_title("Why naive Bayes is a baseline, not ground truth\n"
                 f"{vals2[1]/vals2[0]:.0f}x the generator's own noise", loc="left")
    _save(fig, "audit_independence_violation")


def fig_severity_and_splits(df: pl.DataFrame, kb) -> None:
    d = df.with_columns(pl.col("pathology").replace_strict(kb.severity, default=3).alias("sev"))
    g = d.group_by("sev").len().sort("sev")
    piv = (df.group_by("split", "pathology").len()
           .pivot(on="split", index="pathology", values="len").fill_null(0))
    cols = [c for c in piv.columns if c != "pathology"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.3))
    a1.bar([f"sev {int(r['sev'])}" for r in g.iter_rows(named=True)],
           g["len"].to_numpy(), color=[SEQ[min(int(r["sev"]) - 1, 4)] for r in g.iter_rows(named=True)],
           width=0.6)
    a1.set_ylabel("patients")
    a1.set_title("Severity coverage supports the risk-weighted analysis", loc="left")
    frac = {c: piv[c].to_numpy() / piv[c].sum() for c in cols}
    ref = frac["train"]
    for c in cols:
        if c == "train":
            continue
        a2.scatter(ref, frac[c], s=14, alpha=0.75, label=f"train vs {c}")
    lim = max(ref.max(), max(v.max() for v in frac.values())) * 1.05
    a2.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
    a2.set_xlabel("pathology share, train"); a2.set_ylabel("pathology share, held-out split")
    a2.set_title("Splits are distributionally matched (PSI < 0.013)", loc="left")
    a2.legend(fontsize=8)
    _save(fig, "audit_severity_and_splits")


def main() -> None:
    df = load_all()
    kb = load_kb()
    findings = {f["check"]: f for f in json.loads((RESULTS / "audit_findings.json").read_text())}
    fig_pathology_prior(df, kb)
    fig_generator_noise(findings)
    fig_oracle_feasibility(findings)
    fig_evidence_load(df)
    fig_ddx_shape(df)
    fig_independence(findings)
    fig_severity_and_splits(df, kb)
    print(f"figures -> {FIGURES}")


if __name__ == "__main__":
    main()
