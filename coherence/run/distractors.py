"""Salient-but-uninformative findings, for ablation 11.

A distractor has to be two things at once, and both are *measured* rather
than assumed:

* **uninformative** for this case — adding it leaves the empirical posterior
  essentially unmoved (JSD < 1e-3 at >= 500 patients of support), the same
  certification A3 uses;
* **salient** — it carries a large likelihood ratio for *some* pathology, so
  it reads as clinically meaningful rather than as filler.

Without the second condition the ablation would test tolerance to noise. With
it, it tests whether evidence that *looks* diagnostic but is not destabilises
belief — which is the clinically dangerous version, and the one
arXiv:2609.02797 found raises incoherence by an order of magnitude in the
general domain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from coherence.config import BUILD
from coherence.data.battery import Battery

CACHE = BUILD / "battery" / "distractors_v1.json"


def build_distractors(b: Battery, n_cases: int = 400, min_support: int = 500,
                      max_jsd: float = 1e-3) -> dict[str, str]:
    from coherence.data.ddxplus import load_all
    from coherence.data.empirical_oracle import EmpiricalOracle
    from coherence.data.kb_reconstruct import LikelihoodTable

    tbl = LikelihoodTable.load()
    oracle = EmpiricalOracle.build(load_all(), tbl.tokens, tbl.pathologies)
    # Salience = the largest absolute log-LR the finding carries for any
    # pathology, from the reconstructed table.
    salience = np.abs(tbl.log_lr()).max(axis=0)          # (T,)
    sal = {t: float(salience[i]) for i, t in enumerate(tbl.tokens)}

    out: dict[str, str] = {}
    for c in b.cases[:n_cases]:
        ctx = [t for t in c.evidences if t in oracle.token_index]
        if len(ctx) < 3:
            continue
        # Certify on a 3-finding sub-context so support stays usable.
        sub = ctx[:3]
        cands = oracle.redundancy_candidates(sub, min_support=min_support,
                                             max_jsd=max_jsd, limit=40)
        cands = [(t, j, n) for t, j, n in cands if t not in set(c.evidences)]
        if not cands:
            continue
        best = max(cands, key=lambda r: sal.get(r[0], 0.0))
        out[c.case_id] = best[0]
    return out


def load_or_build_distractors(b: Battery) -> dict[str, str]:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    d = build_distractors(b)
    CACHE.write_text(json.dumps(d))
    return d


if __name__ == "__main__":
    b = Battery.load(Path("build/battery/codx_battery_v1.json"))
    d = load_or_build_distractors(b)
    print(f"distractors for {len(d)} cases -> {CACHE}")
    from collections import Counter
    print("most common distractor tokens:", Counter(d.values()).most_common(5))
