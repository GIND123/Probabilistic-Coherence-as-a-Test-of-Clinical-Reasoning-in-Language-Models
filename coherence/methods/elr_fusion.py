"""ELR-Fusion: Elicited Likelihood-Ratio fusion.

Instead of asking a model for a posterior over an accumulated narrative, ask
it for one evidence weight per finding, each elicited in isolation, and
combine them symbolically:

    log-odds(d | E) = log-odds(d) + sum_i log LR(e_i | d)

Three properties follow.

1. **Order invariance is a theorem, not a measurement.** Addition commutes,
   so the fused posterior is a function of the *set* of findings. The order
   effect is exactly 0 for every model, every case, every budget. This is
   asserted by construction and verified numerically in `verify_invariance`.

2. **It is auditable.** Every finding carries a signed numeric weight towards
   every hypothesis. The explanation is the computation, not a post-hoc
   attribution over it.

3. **It decomposes the failure.** If fusion is coherent but less accurate
   than direct prompting, the deficit is in likelihood estimation. If it is
   both coherent and more accurate, direct prompting was losing information
   to positional effects.

The honest caveat, which the paper states in its own subsection rather than
burying: order invariance of the *true* posterior is unconditional, but naive
summation of log-LRs is exact only under conditional independence given the
pathology. The data audit measures that violation directly on 1.29M patients
(mean |phi| 0.030, 8.7% of finding pairs above 0.1), and
`pairwise_corrected` implements the correction whose value the independence
ablation reports. Note that the *measurement* -- A1 -- assumes nothing; only
the method does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOG_LR_CLIP = 7.0


@dataclass
class WeightTable:
    """Elicited log-likelihood-ratios: findings x pathologies."""

    findings: list[str]
    pathologies: list[str]
    log_lr: np.ndarray            # (F, D)
    valid: np.ndarray             # (F,) bool

    @property
    def index(self) -> dict[str, int]:
        return {f: i for i, f in enumerate(self.findings)}

    def coverage(self) -> float:
        return float(self.valid.mean())


def fuse(log_prior: np.ndarray, weights: WeightTable, findings: list[str],
         clip: float = LOG_LR_CLIP) -> np.ndarray:
    """Posterior over pathologies for an evidence SET.

    `findings` is a list only for convenience; the result is invariant to its
    order because the body is a sum.
    """
    idx = weights.index
    lo = log_prior.astype(np.float64).copy()
    for f in findings:
        i = idx.get(f)
        if i is None or not weights.valid[i]:
            continue
        lo = lo + np.clip(weights.log_lr[i], -clip, clip)
    lo -= lo.max()
    p = np.exp(lo)
    return p / p.sum()


def fuse_batch(log_prior: np.ndarray, weights: WeightTable,
               finding_sets: list[list[str]]) -> np.ndarray:
    return np.vstack([fuse(log_prior, weights, fs) for fs in finding_sets])


def verify_invariance(log_prior: np.ndarray, weights: WeightTable,
                      findings: list[str], n_perms: int = 50,
                      seed: int = 0) -> dict:
    """Numerical confirmation that the theorem holds in floating point.

    The guarantee is mathematical; this exists so the paper can report the
    realised numerical residual rather than asking a reviewer to take the
    algebra on faith.
    """
    from coherence.metrics.divergence import mean_pairwise_jsd

    rng = np.random.default_rng(seed)
    P = [fuse(log_prior, weights, list(rng.permutation(findings)))
         for _ in range(n_perms)]
    P = np.vstack(P)
    return {"n_perms": n_perms,
            "mean_pairwise_jsd": mean_pairwise_jsd(P),
            "max_abs_deviation": float(np.abs(P - P[0]).max())}


# --------------------------------------------------------------------------
# Independence correction (ablation #6)
# --------------------------------------------------------------------------
def pairwise_correction_terms(oracle, findings: list[str], pathologies: list[str],
                              min_support: int = 500) -> np.ndarray:
    """Second-order correction for correlated findings.

    Naive summation double-counts evidence shared between correlated
    findings. The exact second-order term for a pair (a, b) given pathology d
    is log P(a,b|d) - log P(a|d) - log P(b|d), estimated from the corpus.
    Returns a (D,) vector to add to the fused log-odds.
    """
    D = len(pathologies)
    corr = np.zeros(D)
    M, y = oracle.M, oracle.y
    ti = oracle.token_index
    idx = [ti[f] for f in findings if f in ti]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            ia, ib = idx[a], idx[b]
            for d in range(D):
                sel = y == d
                n_d = int(sel.sum())
                if n_d < min_support:
                    continue
                pa = M[sel, ia].mean()
                pb = M[sel, ib].mean()
                pab = (M[sel, ia] & M[sel, ib]).mean()
                if pa <= 0 or pb <= 0 or pab <= 0:
                    continue
                corr[d] += np.log(pab) - np.log(pa) - np.log(pb)
    return corr


def fuse_corrected(log_prior: np.ndarray, weights: WeightTable,
                   findings: list[str], correction: np.ndarray) -> np.ndarray:
    """Fusion with the pairwise correction. Still exactly order-invariant:
    the correction is a sum over unordered pairs."""
    idx = weights.index
    lo = log_prior.astype(np.float64).copy()
    for f in findings:
        i = idx.get(f)
        if i is not None and weights.valid[i]:
            lo = lo + np.clip(weights.log_lr[i], -LOG_LR_CLIP, LOG_LR_CLIP)
    lo = lo + correction
    lo -= lo.max()
    p = np.exp(lo)
    return p / p.sum()
