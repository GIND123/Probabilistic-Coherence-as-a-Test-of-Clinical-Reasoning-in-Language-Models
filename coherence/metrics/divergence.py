"""Divergences between posteriors over pathologies.

All functions take dense, aligned probability vectors (numpy arrays over a
fixed pathology ordering) and return floats. JSD is reported base-2, so it
lives in [0, 1] and 0 means exact agreement.
"""
from __future__ import annotations

import itertools

import numpy as np

EPS = 1e-12


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, None)
    s = p.sum()
    if s <= 0:
        return np.full_like(p, 1.0 / len(p))
    return p / s


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = _norm(p), _norm(q)
    mask = p > EPS
    return float(np.sum(p[mask] * np.log2(p[mask] / np.clip(q[mask], EPS, None))))


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2, in [0, 1]. Squared JS distance."""
    p, q = _norm(p), _norm(q)
    m = 0.5 * (p + q)
    return float(np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0, 1.0))


def mean_pairwise_jsd(posteriors: np.ndarray) -> float:
    """Mean JSD over all unordered pairs of rows. Rows are posteriors."""
    k = len(posteriors)
    if k < 2:
        return 0.0
    return float(np.mean([jsd(posteriors[i], posteriors[j])
                          for i, j in itertools.combinations(range(k), 2)]))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(_norm(p) - _norm(q)).sum())


def topk_jaccard(p: np.ndarray, q: np.ndarray, k: int = 5) -> float:
    a = set(np.argsort(-_norm(p))[:k])
    b = set(np.argsort(-_norm(q))[:k])
    return len(a & b) / len(a | b)


def rank_correlation(p: np.ndarray, q: np.ndarray) -> float:
    from scipy.stats import kendalltau

    t = kendalltau(_norm(p), _norm(q)).statistic
    return float(t) if np.isfinite(t) else 0.0


def entropy(p: np.ndarray) -> float:
    p = _norm(p)
    m = p > EPS
    return float(-np.sum(p[m] * np.log2(p[m])))
