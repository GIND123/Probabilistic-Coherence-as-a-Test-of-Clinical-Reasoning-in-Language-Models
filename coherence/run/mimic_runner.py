"""Two-stage A1 elicitation on real MIMIC narratives.

Stage 1  From the canonical ordering, elicit a candidate differential for the
         case: a free-text list of N plausible diagnoses.
Stage 2  For every ordering of that case -- permutations, retests, shuffled
         controls -- elicit a probability over that FIXED candidate list.

Holding the candidate list fixed is what makes the MIMIC arm measure the same
quantity as the DDXPlus arm. Without it, two orderings could produce different
diagnosis *vocabularies* and any comparison would need string matching or an
LLM judge, either of which reintroduces exactly the contestable ground truth
the whole design was built to avoid.

    .venv/bin/python -m coherence.run.mimic_runner --model qwen3-8b-nothink
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from coherence.config import RESULTS, SEED
from coherence.data.mimic import MimicBattery
from coherence.elicit import registry
from coherence.elicit.engine import Engine, GenConfig, parse_probabilities
from coherence.elicit.prompts import SYSTEM

RAW = RESULTS / "raw_mimic"
RAW.mkdir(parents=True, exist_ok=True)
N_CANDIDATES = 12
CHUNK = 1500

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {"diagnoses": {"type": "array", "items": {"type": "string"},
                                 "minItems": N_CANDIDATES, "maxItems": N_CANDIDATES}},
    "required": ["diagnoses"], "additionalProperties": False,
}


def posterior_schema(n: int) -> dict:
    return {"type": "object",
            "properties": {"probabilities": {"type": "array",
                                             "items": {"type": "number", "minimum": 0,
                                                       "maximum": 1},
                                             "minItems": n, "maxItems": n}},
            "required": ["probabilities"], "additionalProperties": False}


def _narrative(findings: list[str]) -> str:
    return "\n".join(f"- {f}" for f in findings)


def candidate_prompt(cc: str, findings: list[str]) -> list[dict]:
    u = (f"Chief complaint: {cc or 'not stated'}\n\n"
         f"History of present illness:\n\n{_narrative(findings)}\n\n"
         f"List exactly {N_CANDIDATES} distinct diagnoses that should be on this "
         f"patient's differential, most likely first. Use concise standard clinical "
         f"names. Return JSON with a single key \"diagnoses\" holding the list.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]


def posterior_prompt(cc: str, findings: list[str], candidates: list[str]) -> list[dict]:
    lst = "\n".join(f"{i}. {c}" for i, c in enumerate(candidates))
    u = (f"Below is a fixed list of candidate diagnoses for this patient.\n\n{lst}\n\n"
         f"Chief complaint: {cc or 'not stated'}\n\n"
         f"History of present illness:\n\n{_narrative(findings)}\n\n"
         f"Give the probability of each candidate diagnosis. Return a JSON object with "
         f"a single key \"probabilities\" holding an array of {len(candidates)} numbers "
         f"in the same order as the list above. Every diagnosis must receive a strictly "
         f"positive probability: use a small value such as 0.001 for diagnoses you "
         f"consider very unlikely, never 0. The numbers must sum to 1.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-num-seqs", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit-cases", type=int, default=0)
    args = ap.parse_args()

    spec = registry.get(args.model)
    b = MimicBattery.load()
    cases = {c.case_id: c for c in b.cases}
    if args.limit_cases:
        keep = list(cases)[:args.limit_cases]
        cases = {k: cases[k] for k in keep}
    items = [it for it in b.a1 if it["case_id"] in cases]

    out_dir = RAW / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / "candidates.json"
    post_path = out_dir / "a1_posterior.parquet"
    if post_path.exists():
        print(f"  [skip] {post_path} exists")
        return

    engine = Engine(spec, max_num_seqs=args.max_num_seqs)
    gen = GenConfig(temperature=args.temperature, top_p=0.95,
                    max_tokens=1200 if spec.thinking is not True else 3000)
    try:
        # ---- stage 1: one candidate list per case, from the canonical order
        if cand_path.exists():
            cands = json.loads(cand_path.read_text())
        else:
            ids = list(cases)
            prompts = [candidate_prompt(cases[i].chief_complaint, cases[i].findings)
                       for i in ids]
            texts = engine.generate(prompts, gen, CANDIDATE_SCHEMA)
            cands = {}
            for cid, t in zip(ids, texts):
                try:
                    ds = json.loads(t[t.find("{"):])["diagnoses"]
                    seen, uniq = set(), []
                    for x in ds:
                        k = str(x).strip().lower()
                        if k and k not in seen:
                            seen.add(k)
                            uniq.append(str(x).strip())
                    if len(uniq) >= 6:
                        cands[cid] = uniq
                except Exception:
                    continue
            cand_path.write_text(json.dumps(cands))
        print(f"  candidate lists: {len(cands)}/{len(cases)} cases")

        # ---- stage 2: posterior over the fixed list, for every ordering
        jobs = [it for it in items if it["case_id"] in cands]
        print(f"  posterior jobs: {len(jobs):,}")
        rows = []
        for ci in range(0, len(jobs), CHUNK):
            chunk = jobs[ci:ci + CHUNK]
            # Grammar arity varies with candidate-list length, so group by it.
            by_n: dict[int, list] = {}
            for it in chunk:
                by_n.setdefault(len(cands[it["case_id"]]), []).append(it)
            t0 = time.time()
            for n, group in by_n.items():
                prompts = [posterior_prompt(cases[it["case_id"]].chief_complaint,
                                            it["order"], cands[it["case_id"]])
                           for it in group]
                texts = engine.generate(prompts, gen, posterior_schema(n))
                for it, t in zip(group, texts):
                    v = parse_probabilities(t, n)
                    rows.append({"model_key": spec.key, "case_id": it["case_id"],
                                 "arm": it["arm"], "perm_index": it["perm_index"],
                                 "replicate": it["replicate"], "n_candidates": n,
                                 "valid": v is not None,
                                 "parsed": v.tolist() if v is not None else None})
            dt = time.time() - t0
            print(f"    {ci+len(chunk)}/{len(jobs)}  {dt:.0f}s  "
                  f"valid {100*np.mean([r['valid'] for r in rows[-len(chunk):]]):.1f}%")
        pl.DataFrame(rows, schema_overrides={"parsed": pl.List(pl.Float64),
                                             "case_id": pl.Utf8,
                                             "arm": pl.Utf8}).write_parquet(
            post_path, compression="zstd")
        print(f"  -> {post_path}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
