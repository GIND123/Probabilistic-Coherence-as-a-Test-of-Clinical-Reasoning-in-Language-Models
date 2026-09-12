"""Calibration C6 -- posterior collapse to uniform.

Schema validity turned out to be a hollow gate. Every one of
`qwen3-4b-nothink`'s 33,252 A1 responses parsed, yet 62.9% of them were
*exactly* uniform (1/49 on every pathology). A uniform posterior carries no
belief, and it is trivially order-invariant, so including those items
deflates both the order effect and the between-patient ceiling used to
normalise it.

The collapse is partly self-inflicted: the positivity clause adopted in C4 to
stop an all-zero collapse pushed the model to a uniform one instead. This
module measures an INFORMATIVENESS criterion alongside validity, across
prompt variants and model scale, so the instrument can be fixed rather than
reported around.

    .venv/bin/python -m coherence.calibration.uniform_collapse \
        qwen3-4b-nothink qwen3-8b-nothink
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from coherence.config import TABLES
from coherence.data.battery import Battery
from coherence.data.ddxplus import load_kb
from coherence.elicit import prompts as P
from coherence.elicit import registry
from coherence.elicit.engine import Engine, GenConfig, parse_probabilities
from coherence.elicit.render import bulleted

BATTERY = Path("build/battery/codx_battery_v1.json")

VARIANTS: dict[str, tuple[str, float, float]] = {
    # name: (instruction tail, schema min, schema max)
    "positivity": (
        'Give the probability of each candidate diagnosis for this patient. '
        'Return a JSON object with a single key "probabilities" holding an array of '
        '{N} numbers in the same order as the list above. Every diagnosis must '
        'receive a strictly positive probability: use a small value such as 0.001 '
        'for diagnoses you consider very unlikely, never 0. The numbers must sum '
        'to 1.', 0.0, 1.0),
    "positivity_anti_uniform": (
        'Give the probability of each candidate diagnosis for this patient. '
        'Return a JSON object with a single key "probabilities" holding an array of '
        '{N} numbers in the same order as the list above. Every diagnosis must '
        'receive a strictly positive probability: use a small value such as 0.001 '
        'for diagnoses you consider very unlikely, never 0. Do NOT give every '
        'diagnosis the same probability: the findings above discriminate between '
        'these diagnoses and your answer must reflect that. The numbers must sum '
        'to 1.', 0.0, 1.0),
    "plausibility_0_100": (
        'Score each candidate diagnosis from 0 to 100 for how plausible it is for '
        'this patient, where 100 means highly likely and 1 means very unlikely. The '
        'scores do NOT need to sum to anything. Use the full range: the findings '
        'above discriminate between these diagnoses, so the scores must differ. '
        'Return a JSON object with a single key "probabilities" holding an array of '
        '{N} numbers in the same order as the list above.', 0.0, 100.0),
}


def _schema(n: int, lo: float, hi: float) -> dict:
    return {"type": "object",
            "properties": {"probabilities": {
                "type": "array",
                "items": {"type": "number", "minimum": lo, "maximum": hi},
                "minItems": n, "maxItems": n}},
            "required": ["probabilities"], "additionalProperties": False}


def _messages(kb, pathologies, case, order, tail) -> list[dict]:
    cand = "\n".join(f"{i}. {p}" for i, p in enumerate(pathologies))
    u = (f"Below is a fixed list of candidate diagnoses. It is the same list in the "
         f"same order for every case.\n\n{cand}\n\n"
         f"Patient: A {case.age}-year-old "
         f"{'man' if case.sex == 'M' else 'woman'} presenting with the following "
         f"findings.\n\n{bulleted(kb, order)}\n\n{tail.format(N=len(pathologies))}")
    return [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": u}]


def run(model_keys: list[str], n_items: int = 500, seed: int = 0) -> dict:
    b = Battery.load(BATTERY)
    kb = load_kb()
    paths = b.pathologies
    N = len(paths)
    ci = {c.case_id: c for c in b.cases}
    items = [it for it in b.a1 if it.arm == "permutation"][:n_items]
    uniform = np.full(N, 1.0 / N)

    out: dict[str, dict] = {}
    for mk in model_keys:
        eng = Engine(registry.get(mk), max_num_seqs=256)
        gen = GenConfig(temperature=0.7, top_p=0.95, max_tokens=1100)
        out[mk] = {}
        try:
            for name, (tail, lo, hi) in VARIANTS.items():
                prompts = [_messages(kb, paths, ci[it.case_id], it.evidence_order, tail)
                           for it in items]
                texts = eng.generate(prompts, gen, _schema(N, lo, hi))
                ok = uni = 0
                ents, mxs, nuniq = [], [], []
                for t in texts:
                    v = parse_probabilities(t, N)
                    if v is None:
                        continue
                    ok += 1
                    if np.allclose(v, uniform, atol=1e-6):
                        uni += 1
                    ents.append(float(-(v[v > 0] * np.log2(v[v > 0])).sum()))
                    mxs.append(float(v.max()))
                    nuniq.append(int(len(np.unique(np.round(v, 5)))))
                n = len(items)
                out[mk][name] = {
                    "valid_rate": round(ok / n, 4),
                    "uniform_rate": round(uni / n, 4),
                    "informative_rate": round((ok - uni) / n, 4),
                    "mean_entropy_bits": round(float(np.mean(ents)), 3),
                    "mean_max_prob": round(float(np.mean(mxs)), 4),
                    "mean_distinct_values": round(float(np.mean(nuniq)), 1),
                }
                print(f"{mk} | {name} -> {json.dumps(out[mk][name])}", flush=True)
        finally:
            eng.close()
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "calibration_c6.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run(sys.argv[1:] or ["qwen3-4b-nothink", "qwen3-8b-nothink"])
