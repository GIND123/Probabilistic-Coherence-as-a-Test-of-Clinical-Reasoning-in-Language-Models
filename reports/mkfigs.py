import sys, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1]
REP = sys.argv[2]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.linewidth": 0.6, "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black", "text.color": "black",
    "axes.labelcolor": "black", "savefig.bbox": "tight", "figure.dpi": 300,
    "axes.grid": False, "legend.frameon": False,
})

a1 = {r["model"]: r for r in csv.DictReader(open(f"{REP}/a1_main.csv"))}
cp = {r["model"]: r for r in csv.DictReader(open(f"{REP}/competence.csv"))}
meth = {}
for r in csv.DictReader(open(f"{REP}/methods.csv")):
    meth.setdefault(r["model"], {})[r["method"]] = r

label = {
    "qwen3-4b-nothink": "Qwen3 4B", "qwen3-4b-think": "Qwen3 4B T",
    "qwen3-8b-nothink": "Qwen3 8B", "qwen3-8b-think": "Qwen3 8B T",
    "qwen3-32b-nothink": "Qwen3 32B", "qwen3-32b-think": "Qwen3 32B T",
    "medgemma-1.5-4b": "MedGemma 4B", "medgemma-27b": "MedGemma 27B",
    "med42-8b": "Med42 8B", "gpt-oss-20b": "GPT OSS 20B",
    "r1-distill-32b": "R1 Distill 32B",
}

# ---------------- Figure 1: A1 anchors ----------------
order = sorted(a1, key=lambda m: -float(a1[m]["normalised_order_effect"]))
fig, ax = plt.subplots(figsize=(6.4, 2.45))
for i, m in enumerate(order):
    fl = float(a1[m]["jsd_retest_floor"])
    pm = float(a1[m]["jsd_permutation"])
    ce = float(a1[m]["jsd_between_case_ceiling"])
    ax.plot([fl, ce], [i, i], color="black", lw=0.6, zorder=1)
    ax.plot([fl], [i], marker="|", ms=7, color="black", mew=1.1, zorder=3)
    ax.plot([ce], [i], marker="|", ms=7, color="black", mew=1.1, zorder=3)
    ax.plot([pm], [i], marker="o", ms=4.2, mfc="black", mec="black", zorder=4)
ax.axvline(0.0490, color="black", lw=0.6, ls=(0, (4, 2)), zorder=0)
ax.annotate("generator noise floor 0.049", xy=(0.049, -0.55), xytext=(0.105, -0.78),
            fontsize=6.4, va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkA=0, shrinkB=1))
ax.plot([], [], marker="o", color="black", lw=0, ms=4.2, label="permutation divergence")
ax.plot([], [], marker="|", color="black", lw=0, ms=7, mew=1.1, label="own floor and ceiling")
ax.legend(fontsize=6.4, loc="lower right", handletextpad=0.4, borderpad=0.2)
ax.set_yticks(np.arange(len(order)))
ax.set_yticklabels([label[m] for m in order], fontsize=7)
ax.set_xlabel("Jensen Shannon divergence")
ax.set_xlim(0, 0.72)
ax.set_ylim(len(order) - 0.4, -1.35)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(f"{OUT}/fig1_anchors.pdf")
plt.close(fig)

# ---------------- Figure 2: answer and belief gap ----------------
fig, ax = plt.subplots(figsize=(3.15, 2.35))
o2 = sorted(a1, key=lambda m: float(cp[m]["top1_accuracy"]))
x = np.arange(len(o2))
w = 0.38
av = [float(cp[m]["top1_accuracy"]) for m in o2]
pv = [float(meth[m]["direct"]["top1_accuracy"]) for m in o2]
ax.barh(x - w / 2, av, w, color="black", edgecolor="black", lw=0.5, label="named diagnosis")
ax.barh(x + w / 2, pv, w, color="white", edgecolor="black", lw=0.5, hatch="////",
        label="stated posterior")
ax.axvline(1 / 49, color="black", lw=0.5, ls=":")
ax.annotate("chance", xy=(1 / 49, -1.2), xytext=(0.085, -1.2), fontsize=6.2, va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkA=0, shrinkB=1))
ax.set_yticks(x)
ax.set_yticklabels([label[m] for m in o2], fontsize=6.5)
ax.set_xlabel("top 1 accuracy", fontsize=7.5)
ax.set_xlim(0, 0.72)
ax.set_ylim(-1.75, len(o2) - 0.4)
ax.legend(fontsize=6.3, loc="lower right", handletextpad=0.4, borderpad=0.2)
ax.tick_params(labelsize=6.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(f"{OUT}/fig2_gap.pdf")
plt.close(fig)

# ---------------- Figure 3: remedies frontier ----------------
fig, ax = plt.subplots(figsize=(3.15, 2.35))
spec = [
    ("direct", "o", "black", "direct"),
    ("self_consistency", "s", "white", "self consistency"),
    ("perm_ensemble_2", "^", "white", "ensemble K = 2"),
    ("perm_ensemble_3", "^", "0.55", "ensemble K = 3"),
    ("perm_ensemble_5", "^", "black", "ensemble K = 5"),
    ("elr_fusion", "D", "black", "ELR Fusion"),
]
M = meth["qwen3-32b-nothink"]
for key, mkr, fc, nm in spec:
    r = M[key]
    ax.plot(float(r["queries_per_new_case"]), float(r["residual_order_sensitivity_jsd"]),
            marker=mkr, mfc=fc, mec="black", ms=5, mew=0.7, lw=0, label=nm)
xs = np.linspace(1.85, 5.4, 60)
c = float(M["perm_ensemble_2"]["residual_order_sensitivity_jsd"]) * np.sqrt(2)
ax.plot(xs, c / np.sqrt(xs), color="black", lw=0.6, ls=(0, (2, 2)))
ax.annotate(r"$O(1/\sqrt{K})$", xy=(4.2, c / np.sqrt(4.2)), xytext=(2.55, 0.048),
            fontsize=7.0,
            arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkA=1, shrinkB=2))
ax.set_xlabel("model queries per new case", fontsize=7.5)
ax.set_ylabel("residual order sensitivity", fontsize=7.5)
ax.set_xlim(-0.6, 7.4)
ax.set_ylim(-0.025, 0.345)
ax.legend(fontsize=6.0, loc="upper right", handletextpad=0.3, borderpad=0.2, labelspacing=0.28)
ax.tick_params(labelsize=6.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.savefig(f"{OUT}/fig3_methods.pdf")
plt.close(fig)

print("figures written to", OUT)
