"""Baselines, including the one a reviewer will demand.

The comparison that decides the paper is against permutation ensembling,
which also achieves order invariance -- trivially, by averaging over K
orderings. ELR-Fusion has to win on grounds other than coherence:

    method                    order-invariant   cost        auditable
    direct prompting          no                1x          no
    CoT then posterior        no                1x          no
    self-consistency (k)      no                kx          no
    permutation ensemble (K)  approximately     Kx          no
    ELR-Fusion                exactly           n findings  yes
    exact oracle              yes (by defn)     0           n/a
    prior only                yes (by defn)     0           n/a
"""
from __future__ import annotations

import numpy as np


def direct(posteriors: np.ndarray, index: int = 0) -> np.ndarray:
    """A single elicitation at one ordering."""
    return posteriors[index]


def self_consistency(posteriors: np.ndarray) -> np.ndarray:
    """Mean of k samples at the SAME ordering. Reduces sampling noise only."""
    return posteriors.mean(axis=0)


def permutation_ensemble(posteriors: np.ndarray) -> np.ndarray:
    """Mean over K distinct orderings. Order-invariant in expectation.

    Invariance is approximate and improves as O(1/sqrt(K)); the residual at
    each K is what the cost comparison turns on, and it is measured rather
    than assumed.
    """
    return posteriors.mean(axis=0)


def prior_only(prior: np.ndarray) -> np.ndarray:
    """Floor: ignore the findings entirely."""
    return prior


def oracle(posterior: np.ndarray) -> np.ndarray:
    """Ceiling: the empirical-posterior oracle for the same evidence set."""
    return posterior


def ensemble_residual_curve(posteriors: np.ndarray, ks: tuple[int, ...],
                            n_draws: int = 200, seed: int = 0) -> dict[int, float]:
    """How far from invariant is a K-permutation ensemble, as a function of K?

    Two independent ensembles of size K are drawn from the same pool of
    orderings; their divergence is the residual order sensitivity that
    survives the ensembling. ELR-Fusion's residual is exactly 0 at every K.
    """
    from coherence.metrics.divergence import jsd

    rng = np.random.default_rng(seed)
    n = len(posteriors)
    out = {}
    for k in ks:
        if 2 * k > n:
            continue
        vals = []
        for _ in range(n_draws):
            idx = rng.permutation(n)
            a = posteriors[idx[:k]].mean(axis=0)
            b = posteriors[idx[k:2 * k]].mean(axis=0)
            vals.append(jsd(a, b))
        out[k] = float(np.mean(vals))
    return out
