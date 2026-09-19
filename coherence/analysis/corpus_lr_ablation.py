"""#2 and #9: separate the fusion rule from the weights, and bound the task.

Two questions the review raises share one piece of machinery.

#2 asks whether ELR-Fusion's accuracy cost comes from the recombination rule
(a log-linear sum, which is naive Bayes in log-odds form) or from the elicited
weights. Replacing the model-elicited log-LR table with one estimated from the
1.29M-patient corpus, and changing nothing else, separates the two.

#9 asks for an upper bound on the task so that "32x chance" can be read against
something. The corpus-derived table, fused over the same evidence with the same
rule, is that bound: it is what the log-linear family can do on these patients
when the weights are estimated from ground truth rather than elicited.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from coherence.analysis.elr_build import build_elicited_prior, build_weight_table, prior_for
from coherence.config import TABLES
from coherence.data.battery import Battery
from coherence.data.ddxplus import load_all
from coherence.data.empirical_oracle import EmpiricalOracle
from coherence.data.kb_reconstruct import LikelihoodTable
from coherence.methods.elr_fusion import LOG_LR_CLIP, TAU_GRID

BATTERY = "build/battery/codx_battery_v1.json"


def _softmax(lo):
    lo = lo - lo.max()
    p = np.exp(lo)
    return p / p.sum()


def corpus_log_lr(df, tokens, pathologies, alpha: float = 0.5):
    """log P(e|d) - log P(e|not d) per (finding, pathology), Laplace-smoothed."""
    y = df["pathology"].to_list()
    pidx = {p: i for i, p in enumerate(pathologies)}
    yi = np.array([pidx.get(p, -1) for p in y])
    ev = df["evidences"].to_list()
    tidx = {t: i for i, t in enumerate(tokens)}
    n_t, n_p = len(tokens), len(pathologies)
    pres = np.zeros((n_t, n_p))
    cnt = np.bincount(yi[yi >= 0], minlength=n_p).astype(float)
    for row_y, row_e in zip(yi, ev):
        if row_y < 0:
            continue
        for t in row_e:
            j = tidx.get(t)
            if j is not None:
                pres[j, row_y] += 1
    tot = cnt.sum()
    p_e_given_d = (pres + alpha) / (cnt + 2 * alpha)
    other = (pres.sum(1, keepdims=True) - pres + alpha) / ((tot - cnt) + 2 * alpha)
    return np.log(np.clip(p_e_given_d, 1e-9, 1)) - np.log(np.clip(other, 1e-9, 1))


def main():
    b = Battery.load(BATTERY)
    paths = list(b.pathologies)
    pidx = {p: i for i, p in enumerate(paths)}
    tbl = LikelihoodTable.load()
    df = load_all()
    oracle = EmpiricalOracle.build(df, tbl.tokens, tbl.pathologies)
    emp = np.log(np.clip(oracle.prior.probs, 1e-9, None))

    tokens = list(tbl.tokens)
    L = corpus_log_lr(df, tokens, paths)
    tpos = {t: i for i, t in enumerate(tokens)}

    cases = [c for c in b.cases if c.pathology in pidx]
    truth = np.array([pidx[c.pathology] for c in cases])

    # Corpus-derived sum over the evidence SET (order-invariant, same rule).
    sums = []
    for c in cases:
        s = np.zeros(len(paths))
        for t in c.evidences:
            j = tpos.get(t)
            if j is not None:
                s += np.clip(L[j], -LOG_LR_CLIP, LOG_LR_CLIP)
        sums.append(s)
    sums = np.vstack(sums)
    lp = np.vstack([np.log(np.clip(prior_for({}, c.age, c.sex, np.exp(emp)), 1e-9, None))
                    for c in cases])

    rows = []
    for tau in [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        M = np.vstack([_softmax(lp[i] + tau * sums[i]) for i in range(len(cases))])
        rows.append({
            "weights": "corpus-derived", "tau": tau, "n_cases": len(cases),
            "top1_accuracy": float(np.mean(M.argmax(1) == truth)),
            "top5_accuracy": float(np.mean([truth[i] in np.argsort(-M[i])[:5]
                                            for i in range(len(M))])),
            "nll_of_truth": float(np.mean([-np.log(max(M[i, truth[i]], 1e-12))
                                           for i in range(len(M))])),
            "mean_entropy_bits": float(np.mean(
                [-(q[q > 0] * np.log2(q[q > 0])).sum() for q in M])),
        })

    # Same rule, elicited weights, for the models that have a table.
    from coherence.axioms.loading import available_models
    from coherence.elicit.registry import DEFAULT_ORDER
    have = set(available_models("a1_posterior"))
    for m in [k for k in DEFAULT_ORDER if k in have]:
        wt = build_weight_table(m, "numeric", paths)
        if wt is None or wt.coverage() <= 0:
            continue
        priors = build_elicited_prior(m, paths) or {}
        widx = wt.index
        es = []
        for c in cases:
            s = np.zeros(len(paths))
            for t in c.evidences:
                i = widx.get(t)
                if i is not None and wt.valid[i]:
                    s += np.clip(wt.log_lr[i], -LOG_LR_CLIP, LOG_LR_CLIP)
            es.append(s)
        es = np.vstack(es)
        lpm = np.vstack([np.log(np.clip(prior_for(priors, c.age, c.sex, np.exp(emp)),
                                        1e-9, None)) for c in cases])
        best = None
        for tau in [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
            M = np.vstack([_softmax(lpm[i] + tau * es[i]) for i in range(len(cases))])
            acc = float(np.mean(M.argmax(1) == truth))
            nll = float(np.mean([-np.log(max(M[i, truth[i]], 1e-12)) for i in range(len(M))]))
            if best is None or nll < best["nll_of_truth"]:
                best = {"weights": f"elicited:{m}", "tau": tau, "n_cases": len(cases),
                        "top1_accuracy": acc, "top5_accuracy": float(np.mean(
                            [truth[i] in np.argsort(-M[i])[:5] for i in range(len(M))])),
                        "nll_of_truth": nll,
                        "mean_entropy_bits": float(np.mean(
                            [-(q[q > 0] * np.log2(q[q > 0])).sum() for q in M]))}
        rows.append(best)

    out = pl.DataFrame(rows)
    out.write_csv(TABLES / "rev_corpus_lr_ablation.csv")
    print(f"  rev_corpus_lr_ablation.csv ({out.height} rows)")
    print(out.filter(pl.col("weights") == "corpus-derived")
          .select("tau", "top1_accuracy", "top5_accuracy", "nll_of_truth"))
    best = out.filter(pl.col("weights") == "corpus-derived").sort("top1_accuracy",
                                                                 descending=True)
    print(f"\n  corpus-LR best top-1 = {best['top1_accuracy'][0]:.4f} "
          f"at tau={best['tau'][0]}  (this is the #9 log-linear upper bound)")


if __name__ == "__main__":
    main()
