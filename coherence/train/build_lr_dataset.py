"""Training data for schema-compliant likelihood-ratio emission.

ELR-Fusion is only as good as the weights it is given, and the audit shows
zero-shot elicitation is the method's weak point. This builds a supervised
set that teaches a model to emit a well-formed, calibrated log-likelihood-
ratio vector for a single finding.

Targets come from the likelihood table reconstructed from the 1.29M released
patients. Note what this does and does not claim: the table is a precise
estimate of the generator's per-finding conditionals (binomial SE ~3e-4), and
it is *not* the generator's posterior -- naive recombination of these weights
was shown in the audit to reproduce shipped differentials poorly. So the
adapter is trained on quantities the corpus estimates well, and the fusion
step's independence error is handled separately by the correction ablation.

**Split is by finding, not by example.** A random example-level split would
leak: the same finding would appear in train and test with a different
demographic framing, and the held-out score would measure memorisation. The
held-out set is findings the adapter has never seen in any form.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coherence.config import BUILD, SEED
from coherence.data.ddxplus import load_kb
from coherence.data.kb_reconstruct import LikelihoodTable
from coherence.elicit.prompts import likelihood_ratio_prompt
from coherence.elicit.schema import ORDINAL_BANDS

OUT = BUILD / "train"
OUT.mkdir(parents=True, exist_ok=True)

AGE_BANDS = [(0, 4), (5, 14), (15, 29), (30, 44), (45, 59), (60, 74), (75, 100)]
FORMATS = ("numeric", "logodds", "ordinal")


def _to_ordinal(log_lr: float) -> str:
    """Snap a log-LR to the nearest verbal band."""
    best, bd = "uninformative", 1e9
    for name, v in ORDINAL_BANDS.items():
        d = abs(log_lr - v)
        if d < bd:
            best, bd = name, d
    return best


def _target(log_lr: np.ndarray, fmt: str) -> dict:
    v = np.clip(log_lr, -7.0, 7.0)
    if fmt == "logodds":
        return {"weights": [round(float(x), 3) for x in v]}
    if fmt == "numeric":
        return {"weights": [round(float(np.clip(np.exp(x), 1e-3, 1e3)), 4) for x in v]}
    return {"weights": [_to_ordinal(float(x)) for x in v]}


def build(seed: int = SEED, holdout_frac: float = 0.2,
          n_demo_variants: int = 4) -> dict:
    """Build the SFT set.

    `n_demo_variants` controls how many (age, sex) framings each
    (finding, format) pair gets. The point of varying demographics is to stop
    the adapter latching onto one surface framing, and a handful of variants
    does that; the original 14 produced 16,212 examples over only 413 train
    findings -- a 39x redundancy that tripled training time without adding
    signal.
    """
    kb = load_kb()
    tbl = LikelihoodTable.load()
    log_lr = tbl.log_lr()                       # (D, T)
    pathologies = tbl.pathologies
    rng = np.random.default_rng(seed)

    tokens = list(tbl.tokens)
    order = rng.permutation(len(tokens))
    n_hold = int(len(tokens) * holdout_frac)
    hold = {tokens[i] for i in order[:n_hold]}

    splits: dict[str, list[dict]] = {"train": [], "test": []}
    for ti, tok in enumerate(tokens):
        which = "test" if tok in hold else "train"
        col = log_lr[:, ti]
        # Skip findings the corpus barely observed: their weights are noise
        # and teaching them would teach noise.
        if tbl.counts[:, ti].sum() < 200:
            continue
        for fmt in FORMATS:
            tgt = json.dumps(_target(col, fmt), separators=(",", ":"))
            bands = [AGE_BANDS[i] for i in
                     rng.choice(len(AGE_BANDS), size=min(n_demo_variants, len(AGE_BANDS)),
                                replace=False)]
            for bi, (lo, hi) in enumerate(bands):
                for sex in (("M", "F")[bi % 2],):
                    age = int(rng.integers(lo, hi + 1))
                    msgs = likelihood_ratio_prompt(kb, pathologies, tok, age, sex, fmt)
                    splits[which].append({
                        "messages": msgs + [{"role": "assistant", "content": tgt}],
                        "finding": tok, "lr_format": fmt, "age": age, "sex": sex,
                    })
    for k, v in splits.items():
        rng.shuffle(v)
        p = OUT / f"lr_sft_{k}.jsonl"
        with p.open("w") as fh:
            for r in v:
                fh.write(json.dumps(r) + "\n")
    meta = {"n_train": len(splits["train"]), "n_test": len(splits["test"]),
            "n_demo_variants": n_demo_variants,
            "n_findings_total": len(tokens), "n_findings_held_out": len(hold),
            "held_out_findings": sorted(hold), "formats": list(FORMATS),
            "split_unit": "finding", "seed": seed}
    (OUT / "lr_sft_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps({k: v for k, v in m.items() if k != "held_out_findings"}, indent=2))
