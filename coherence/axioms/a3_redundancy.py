"""A3 -- redundancy insensitivity.

Each item adds a finding that the corpus certifies as uninformative given the
context: the empirical posterior with and without it differ by less than
1e-3 JSD at 500+ patients of support. The true update is therefore zero, and
any movement the model shows is spurious.

This catches the "more text implies more confidence" pathology, which is
invisible to accuracy metrics and clinically dangerous.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from coherence.metrics.divergence import entropy, jsd


@dataclass
class A3Result:
    model_key: str
    n_items: int
    mean_spurious_jsd: float
    median_spurious_jsd: float
    p90_spurious_jsd: float
    certified_true_jsd: float
    ratio_to_truth: float
    top1_flip_rate: float
    mean_entropy_change: float
    confidence_inflation_rate: float   # fraction where entropy DROPPED
    per_item: pl.DataFrame = None


def analyse(model_key: str, d: pl.DataFrame, posteriors: np.ndarray,
            a3_items: list) -> A3Result:
    d = d.with_row_index("row")
    idx = {(r["a3_id"], r["phase"]): r["row"] for r in d.iter_rows(named=True) if r["valid"]}
    cert = {it.item_id: it.certified_jsd for it in a3_items}
    rows = []
    for a3_id, c in cert.items():
        rw, ro = idx.get((a3_id, "without")), idx.get((a3_id, "with"))
        if rw is None or ro is None:
            continue
        pw, po = posteriors[rw], posteriors[ro]
        hw, ho = entropy(pw), entropy(po)
        rows.append({"a3_id": a3_id, "spurious_jsd": jsd(pw, po),
                     "certified_jsd": c,
                     "top1_flip": int(pw.argmax() != po.argmax()),
                     "entropy_change": ho - hw,
                     "max_prob_change": float(po.max() - pw.max())})
    t = pl.DataFrame(rows)
    s = t["spurious_jsd"].to_numpy()
    ct = float(t["certified_jsd"].mean())
    return A3Result(
        model_key=model_key, n_items=t.height,
        mean_spurious_jsd=float(s.mean()), median_spurious_jsd=float(np.median(s)),
        p90_spurious_jsd=float(np.percentile(s, 90)),
        certified_true_jsd=ct,
        ratio_to_truth=float(s.mean() / ct) if ct > 0 else np.inf,
        top1_flip_rate=float(t["top1_flip"].mean()),
        mean_entropy_change=float(t["entropy_change"].mean()),
        confidence_inflation_rate=float((t["entropy_change"] < 0).mean()),
        per_item=t)
