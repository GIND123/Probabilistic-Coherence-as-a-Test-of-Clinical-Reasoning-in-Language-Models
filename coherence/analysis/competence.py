"""Task competence: does the model answer the question at all?

Every gate the instrument had before this one -- schema validity, informative
rate, parse success -- is a statement about the SHAPE of a response, not its
content. A model can pass all of them while emitting numbers unrelated to the
patient in front of it. That is not hypothetical: gpt-oss-20b decoded under a
JSON grammar scored 1.000 schema validity and 1.000 informative rate while
answering `{"index": 0}` to essentially every case, for a top-1 accuracy of
0.039 against a 0.020 chance rate (see C11 in the calibration report).

An order-effect number computed over such responses is a measurement of noise,
and it looks perfectly reasonable: near-zero order effect is exactly what a
constant answer produces, and near-ceiling order effect is exactly what a
random one produces. Neither is a fact about clinical reasoning.

So competence is reported alongside coherence and gates it. The natural test
is the pick-one-diagnosis task, which has a ground-truth label and a known
chance rate of 1/49. `competence_ratio` is accuracy over chance; a model that
cannot beat chance has no belief state whose order-sensitivity is worth
reporting, and its coherence rows are flagged rather than silently ranked.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from coherence.axioms.loading import load_task
from coherence.data.battery import Battery

# Deliberately no pass/fail accuracy threshold. Measured on this battery, a
# genuinely weak model (MedGemma-1.5-4B: 0.081 accuracy, 3.9x chance, top-class
# share 0.467) and a broken one (gpt-oss-20b under a JSON grammar: 0.043, 2.1x
# chance, top-class share 0.459) are not separable by any single cut on
# accuracy or on prediction diversity. What distinguished them was a
# within-model comparison: fixing the decoding path moved the SAME model from
# 0.039 to 0.633 on the SAME items. So competence is reported as a continuous
# quantity, `beats_chance` is the only binary claim made (its Wilson lower
# bound clears 1/49), and a suspected elicitation fault is settled by changing
# one thing about that model and re-measuring -- not by a threshold.


def _case_truth(b: Battery) -> tuple[dict[str, int], int]:
    paths = list(b.pathologies)
    idx = {p: i for i, p in enumerate(paths)}
    c2p: dict[str, int] = {}
    for c in b.cases:
        cid = getattr(c, "case_id", None) if not isinstance(c, dict) else c.get("case_id")
        pa = getattr(c, "pathology", None) if not isinstance(c, dict) else c.get("pathology")
        if cid is not None:
            c2p[cid] = idx.get(pa, -1)
    return c2p, len(paths)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def model_competence(model_key: str, b: Battery) -> dict | None:
    """Top-1 accuracy on pick-one-diagnosis, with a Wilson interval."""
    d = load_task(model_key, "a1_answer")
    if d is None or not d.height:
        return None
    c2p, n_path = _case_truth(b)
    chance = 1.0 / n_path
    ok_rows = d.filter(pl.col("valid"))
    if not ok_rows.height:
        return None
    # load_task() already expands the JSON meta blob into columns.
    truth = np.array([c2p.get(c, -1) for c in ok_rows["case_id"].to_list()])
    pred = np.array([int(p[0]) if p else -1 for p in ok_rows["parsed"].to_list()])
    m = truth >= 0
    k = int((pred[m] == truth[m]).sum())
    n = int(m.sum())
    # A response set concentrated on one label is the signature of a model
    # emitting a constant rather than reading the case. Reported, not gated.
    counts = np.bincount(pred[m][pred[m] >= 0], minlength=n_path).astype(float)
    share = counts / max(counts.sum(), 1)
    nz = share[share > 0]
    pred_entropy = float(-(nz * np.log2(nz)).sum())
    top_share = float(share.max())
    acc = k / n if n else float("nan")
    lo, hi = _wilson(k, n)
    return {
        "model": model_key,
        "n_items": n,
        "answer_rate": float(d["valid"].mean()),
        "top1_accuracy": acc,
        "acc_ci_low": lo,
        "acc_ci_high": hi,
        "chance": chance,
        "competence_ratio": acc / chance if chance else float("nan"),
        "pred_entropy_bits": pred_entropy,
        "top_class_share": top_share,
        # The only binary claim: the interval clears chance, so the model has
        # a demonstrated dependence on the case. Everything above is continuous
        # and left to the reader.
        "beats_chance": bool(lo > chance),
    }


def competence_table(models: list[str], b: Battery) -> pl.DataFrame:
    rows = [r for r in (model_competence(m, b) for m in models) if r is not None]
    return pl.DataFrame(rows) if rows else pl.DataFrame()
