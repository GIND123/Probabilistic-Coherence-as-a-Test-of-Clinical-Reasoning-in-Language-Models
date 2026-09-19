"""A4 -- positional anchoring.

The decisive finding of a case is placed first, middle or last while every
other finding is permuted freely. A coherent reasoner shows no effect of the
decisive finding's position. Deviation is premature closure, or recency, with
a number attached.

Two readings are reported:
  * `beta_position`  slope of the decisive pathology's log-odds on position
    coded -1 / 0 / +1 (first / middle / last). Positive means recency, the
    model weights late findings more; negative means anchoring on the first.
  * `eta_squared`    the share of between-item variance in the posterior that
    position explains, which is the effect size a reviewer will ask for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from coherence.axioms.a2_update import log_odds
from coherence.metrics.divergence import jsd

POSITION_CODE = {"first": -1.0, "middle": 0.0, "last": 1.0}


@dataclass
class A4Result:
    model_key: str
    n_cases: int
    beta_position: float
    beta_ci: tuple[float, float]
    eta_squared: float
    jsd_between_positions: float
    jsd_within_position: float
    position_effect: float
    top1_flip_rate: float
    per_case: pl.DataFrame = None


def analyse(model_key: str, d: pl.DataFrame, posteriors: np.ndarray,
            pathologies: list[str], seed: int = 0, n_boot: int = 2000) -> A4Result:
    pi = {p: i for i, p in enumerate(pathologies)}
    d = d.with_row_index("row")
    rows, slopes_x, slopes_y = [], [], []
    jb, jw, flips = [], [], []
    for case_id, sub in d.group_by("case_id", maintain_order=True):
        case_id = case_id[0] if isinstance(case_id, tuple) else case_id
        sub = sub.filter(pl.col("valid"))
        if sub.height < 6:
            continue
        target = pi.get(sub["pathology"][0]) if "pathology" in sub.columns else None
        if target is None:
            continue
        P = posteriors[sub["row"].to_numpy()]
        pos = np.array([POSITION_CODE[p] for p in sub["position"].to_list()])
        lo = log_odds(P[:, target])
        # Between-position divergence, averaged over position pairs, against
        # the within-position (replicate) divergence -- the same floor logic
        # A1 uses, so the two axioms are on a comparable scale.
        groups = {g: P[pos == v] for g, v in POSITION_CODE.items()}
        b_pairs, w_pairs = [], []
        for g, M in groups.items():
            for i in range(len(M)):
                for j in range(i + 1, len(M)):
                    w_pairs.append(jsd(M[i], M[j]))
        gs = list(groups)
        for a in range(len(gs)):
            for b in range(a + 1, len(gs)):
                A, B = groups[gs[a]], groups[gs[b]]
                if len(A) and len(B):
                    b_pairs.append(float(np.mean([jsd(x, y) for x in A for y in B])))
        means = {g: lo[pos == v].mean() for g, v in POSITION_CODE.items() if (pos == v).any()}
        grand = lo.mean()
        ss_between = sum(((pos == POSITION_CODE[g]).sum()) * (m - grand) ** 2
                         for g, m in means.items())
        ss_total = float(((lo - grand) ** 2).sum())
        top = P.argmax(axis=1)
        modal_by_pos = {g: np.bincount(top[pos == v]).argmax()
                        for g, v in POSITION_CODE.items() if (pos == v).any()}
        rows.append({
            "case_id": case_id, "n": sub.height,
            "logodds_first": float(means.get("first", np.nan)),
            "logodds_middle": float(means.get("middle", np.nan)),
            "logodds_last": float(means.get("last", np.nan)),
            "eta_squared": float(ss_between / ss_total) if ss_total > 0 else np.nan,
            "jsd_between": float(np.mean(b_pairs)) if b_pairs else np.nan,
            "jsd_within": float(np.mean(w_pairs)) if w_pairs else np.nan,
            "top1_flip": int(len(set(modal_by_pos.values())) > 1),
            "decisiveness": float(sub["decisiveness"][0]),
            "severity": int(sub["severity"][0]) if "severity" in sub.columns else -1,
        })
        slopes_x.append(pos)
        slopes_y.append(lo)
        jb.append(rows[-1]["jsd_between"])
        jw.append(rows[-1]["jsd_within"])
        flips.append(rows[-1]["top1_flip"])

    t = pl.DataFrame(rows)
    X = np.concatenate(slopes_x)
    Y = np.concatenate(slopes_y)
    # Within-case centring, so between-case differences in baseline log-odds
    # cannot masquerade as a position effect.
    starts = np.cumsum([0] + [len(a) for a in slopes_x])
    Yc = Y.copy()
    for i in range(len(slopes_x)):
        s, e = starts[i], starts[i + 1]
        Yc[s:e] -= Yc[s:e].mean()
    b = float(np.polyfit(X, Yc, 1)[0])
    rng = np.random.default_rng(seed)
    n_cases = len(slopes_x)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_cases, n_cases)
        xs = np.concatenate([slopes_x[i] for i in pick])
        ys = np.concatenate([slopes_y[i] - slopes_y[i].mean() for i in pick])
        bs.append(np.polyfit(xs, ys, 1)[0])
    bs = np.asarray(bs)
    jbm, jwm = float(np.nanmean(jb)), float(np.nanmean(jw))
    return A4Result(
        model_key=model_key, n_cases=t.height, beta_position=b,
        beta_ci=(float(np.quantile(bs, .025)), float(np.quantile(bs, .975))),
        # nanmean, not mean: a case whose log-odds are constant across all
        # replicates has ss_total == 0 and yields NaN, and a single such case
        # propagated NaN through the whole model average -- which is why
        # eta_squared was reported for only 3 of 11 models.
        eta_squared=float(np.nanmean(t["eta_squared"].to_numpy())),
        jsd_between_positions=jbm, jsd_within_position=jwm,
        position_effect=jbm - jwm,
        top1_flip_rate=float(np.mean(flips)), per_case=t)
