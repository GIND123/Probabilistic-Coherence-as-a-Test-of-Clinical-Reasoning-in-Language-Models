"""#6  Token-logprob baseline: is the deficit specific to verbalized probability?

Every number in A1-A3 comes from a distribution the model *writes out* -- 49
scores it has to name. That confounds two things: what the model's weights
encode about the patient, and whether it can express that as calibrated text.
All models here are open-weight, so the first can be measured directly.

Method. The prompt ends at "The single most likely diagnosis is:" and each of
the 49 candidate names is scored as a continuation by its summed token
log-probability, length-normalised. Softmaxing those 49 scores gives a
distribution that never passes through verbalized numbers. The shared prefix is
identical across the 49 candidates for a case, so prefix caching prefills it
once and the marginal cost is only the candidate tokens.

The comparison this licenses: if the logprob distribution ranks the true
pathology far better than the stated posterior does, the deficit is in
expression, not knowledge. If both are equally poor, it is knowledge.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from coherence.config import RESULTS, SEED
from coherence.data.battery import Battery
from coherence.data.ddxplus import load_kb
from coherence.elicit import prompts as P
from coherence.elicit.engine import Engine
from coherence.elicit.registry import get

RAW = RESULTS / "raw"
CHUNK_CASES = 120


def _stem(kb, pathologies, evidence, age, sex, fmt="bulleted") -> str:
    """The shared prefix: same clinical vignette, no candidate list.

    The 49-line candidate list is deliberately omitted. Including it would let
    position in that list influence the score, which is the positional confound
    the review raises in comment 8.
    """
    msgs = P.top_diagnosis_prompt(kb, pathologies, evidence, age, sex, fmt)
    user = msgs[-1]["content"]
    cut = user.find("Candidate diagnoses:")
    body = user[:cut].rstrip() if cut > 0 else user
    return (f"{body}\n\nThe single most likely diagnosis for this patient is:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--battery", default="build/battery/codx_battery_v1.json")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--cases", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    b = Battery.load(a.battery)
    kb = load_kb()
    paths = list(b.pathologies)
    spec = get(a.model)
    out_dir = RAW / spec.key
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "logprob_baseline.parquet"
    if final.exists():
        print(f"  [skip] already complete: {final}")
        return

    cases = b.cases if not a.cases else b.cases[: a.cases]
    print(f"model {spec.key}  cases={len(cases):,}  candidates={len(paths)}")

    engine = Engine(spec, max_num_seqs=a.max_num_seqs)
    tok = engine.tokenizer

    # Pre-tokenise the candidates once; token counts drive length normalisation.
    cand_tok = [tok(" " + p, add_special_tokens=False).input_ids for p in paths]
    cand_len = np.array([len(t) for t in cand_tok], dtype=float)

    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    shards = out_dir / "logprob_baseline.chunks"
    shards.mkdir(exist_ok=True)
    done = {int(p.stem) for p in shards.glob("*.parquet")}
    n_chunks = (len(cases) + CHUNK_CASES - 1) // CHUNK_CASES
    print(f"  {n_chunks} chunks ({len(done)} already done)")

    for ci in range(n_chunks):
        if ci in done:
            continue
        block = cases[ci * CHUNK_CASES:(ci + 1) * CHUNK_CASES]
        prompts, index = [], []
        for c in block:
            stem = _stem(kb, paths, list(c.evidences), c.age, c.sex)
            stem_ids = tok(stem, add_special_tokens=False).input_ids
            for j, ct in enumerate(cand_tok):
                prompts.append(stem_ids + ct)
                index.append((c.case_id, j, len(stem_ids), len(ct)))
        t0 = time.time()
        outs = engine.llm.generate(
            prompt_token_ids=prompts, sampling_params=sp) \
            if _accepts_token_ids(engine) else engine.llm.generate(
                [tok.decode(p) for p in prompts], sp)
        dt = time.time() - t0

        scores: dict[str, np.ndarray] = {}
        for (cid, j, n_stem, n_cand), o in zip(index, outs):
            lp = o.prompt_logprobs or []
            tail = lp[n_stem:n_stem + n_cand]
            tot = 0.0
            for pos in tail:
                if not pos:
                    continue
                tot += float(next(iter(pos.values())).logprob)
            scores.setdefault(cid, np.full(len(paths), np.nan))[j] = tot

        rows = []
        pidx = {p: i for i, p in enumerate(paths)}
        for c in block:
            v = scores.get(c.case_id)
            if v is None or not np.isfinite(v).all():
                continue
            norm = v / np.maximum(cand_len, 1.0)          # length-normalised
            for arr, tag in ((v, "sum"), (norm, "meanlen")):
                z = arr - arr.max()
                p = np.exp(z) / np.exp(z).sum()
                rows.append({
                    "model_key": spec.key, "task": "logprob_baseline",
                    "item_id": f"{c.case_id}:{tag}", "kind": "posterior",
                    "valid": True, "parsed": p.tolist(), "raw_text": None,
                    "meta": json.dumps({"case_id": c.case_id, "scoring": tag,
                                        "pathology": c.pathology,
                                        "truth_index": pidx.get(c.pathology, -1),
                                        "n_findings": len(c.evidences)}),
                })
        pl.DataFrame(rows, schema_overrides={"parsed": pl.List(pl.Float64),
                                             "raw_text": pl.Utf8,
                                             "item_id": pl.Utf8,
                                             "meta": pl.Utf8}) \
          .write_parquet(shards / f"{ci}.parquet", compression="zstd")
        print(f"    chunk {ci + 1}/{n_chunks}  {len(block)} cases  "
              f"{len(prompts):,} scored  {dt:.0f}s")

    parts = [pl.read_parquet(p) for p in sorted(shards.glob("*.parquet"),
                                                key=lambda q: int(q.stem))]
    if parts:
        pl.concat(parts, how="diagonal_relaxed").write_parquet(
            final, compression="zstd")
        print(f"  -> {final}")


def _accepts_token_ids(engine) -> bool:
    import inspect
    try:
        return "prompt_token_ids" in inspect.signature(engine.llm.generate).parameters
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
