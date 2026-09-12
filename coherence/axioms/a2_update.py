"""A2 -- update fidelity.

For each finding e added to a context C, compare the model's implied change
in log-odds against the true change measured by the empirical-posterior
oracle. Regressing implied on true gives

    beta < 1  under-updating (conservatism; Edwards 1968)
    beta > 1  over-reaction
    R^2       how much of the correct update signal is tracked at all

The regression is weighted by the oracle's precision, because the target is
itself estimated: items whose oracle interval is wide should not pull the
slope as hard as items where the target is nailed down.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

LOGIT_FLOOR = 1e-4


def log_odds(p: np.ndarray, floor: float = LOGIT_FLOOR) -> np.ndarray:
    q = np.clip(p, floor, 1 - floor)
    return np.log(q / (1 - q))


@dataclass
class A2Result:
    model_key: str
    n_items: int
    n_points: int
    beta: float
    beta_ci: tuple[float, float]
    intercept: float
    r2: float
    spearman: float
    mean_abs_error: float
    direction_agreement: float


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Weighted least squares of y on x with intercept. Returns (b, a, r2)."""
    W = w / w.sum()
    mx, my = (W * x).sum(), (W * y).sum()
    cov = (W * (x - mx) * (y - my)).sum()
    vx = (W * (x - mx) ** 2).sum()
    b = cov / vx if vx > 0 else np.nan
    a = my - b * mx
    pred = a + b * x
    ss_res = (W * (y - pred) ** 2).sum()
    ss_tot = (W * (y - my) ** 2).sum()
    return float(b), float(a), float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def build_points(d: pl.DataFrame, posteriors: np.ndarray,
                 a2_items: list, pathologies: list[str],
                 min_true_delta: float = 0.05) -> pl.DataFrame:
    """One row per (item, pathology): true vs model implied delta log-odds.

    Pathologies whose true update is essentially nil are dropped: they carry
    no signal about update *magnitude* and would drag the slope towards zero
    while inflating n. A3 is the axiom that tests null updates, on items
    constructed for it.
    """
    d = d.with_row_index("row")
    by_id = {it.item_id: it for it in a2_items}
    idx = {}
    for r in d.iter_rows(named=True):
        a2_id, phase = r["a2_id"], r["phase"]
        if r["valid"]:
            idx[(a2_id, phase)] = r["row"]
    rows = []
    for a2_id, it in by_id.items():
        rb, ra = idx.get((a2_id, "before")), idx.get((a2_id, "after"))
        if rb is None or ra is None:
            continue
        model_delta = log_odds(posteriors[ra]) - log_odds(posteriors[rb])
        true_delta = np.asarray(it.true_delta_log_odds)
        se_b = np.asarray(it.se_before)
        se_a = np.asarray(it.se_after)
        pb = np.clip(np.asarray(it.true_posterior_before), LOGIT_FLOOR, 1 - LOGIT_FLOOR)
        pa = np.clip(np.asarray(it.true_posterior_after), LOGIT_FLOOR, 1 - LOGIT_FLOOR)
        # Delta-method SE of the difference of two log-odds.
        se_lo = np.sqrt((se_b / (pb * (1 - pb))) ** 2 + (se_a / (pa * (1 - pa))) ** 2)
        keep = np.abs(true_delta) >= min_true_delta
        for j in np.flatnonzero(keep):
            rows.append({
                "a2_id": a2_id, "pathology": pathologies[j],
                "true_delta": float(true_delta[j]),
                "model_delta": float(model_delta[j]),
                "se_true": float(se_lo[j]),
                "context_size": len(it.context),
                "support_before": it.support_before, "support_after": it.support_after,
            })
    return pl.DataFrame(rows)


def analyse(model_key: str, points: pl.DataFrame, seed: int = 0,
            n_boot: int = 2000) -> A2Result:
    from scipy.stats import spearmanr

    x = points["true_delta"].to_numpy()
    y = points["model_delta"].to_numpy()
    se = points["se_true"].to_numpy()
    w = 1.0 / np.clip(se, 1e-3, None) ** 2
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    b, a, r2 = _wls(x, y, w)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        bs.append(_wls(x[i], y[i], w[i])[0])
    bs = np.asarray(bs)
    return A2Result(
        model_key=model_key, n_items=points["a2_id"].n_unique(), n_points=int(len(x)),
        beta=b, beta_ci=(float(np.quantile(bs, .025)), float(np.quantile(bs, .975))),
        intercept=a, r2=r2,
        spearman=float(spearmanr(x, y).statistic),
        mean_abs_error=float(np.mean(np.abs(y - x))),
        direction_agreement=float(np.mean(np.sign(x) == np.sign(y))))


def by_stratum(points: pl.DataFrame, col: str) -> pl.DataFrame:
    out = []
    for key, sub in points.group_by(col, maintain_order=True):
        k = key[0] if isinstance(key, tuple) else key
        if sub.height < 50:
            continue
        r = analyse("", sub)
        out.append({col: k, "n": sub.height, "beta": r.beta, "r2": r.r2,
                    "direction_agreement": r.direction_agreement})
    return pl.DataFrame(out).sort(col)
