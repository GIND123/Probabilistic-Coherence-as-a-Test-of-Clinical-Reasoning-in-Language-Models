"""Execute elicitation jobs for one model and checkpoint the results.

Usage
-----
    .venv/bin/python -m coherence.run.runner --model qwen3-8b-nothink \
        --tasks a1_posterior a4_posterior --battery build/battery/codx_battery_v1.json

Results land in results/raw/<model_key>/<task>.parquet. A task whose parquet
already exists is skipped, so the whole sweep is resumable at task
granularity. Within a task, jobs are chunked and appended, so an interrupted
task resumes from its last completed chunk rather than from zero.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from coherence.config import LOGS, RESULTS, SEED
from coherence.data.battery import Battery
from coherence.data.ddxplus import load_kb
from coherence.elicit import registry
from coherence.elicit.engine import (Engine, GenConfig, parse_index,
                                     parse_probabilities, parse_weights)
from coherence.run import tasks as T

RAW = RESULTS / "raw"
RAW.mkdir(parents=True, exist_ok=True)
CHUNK = 2000


def _shard_dir(model_key: str, task: str) -> Path:
    p = RAW / model_key / f"{task}.chunks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _final_path(model_key: str, task: str) -> Path:
    p = RAW / model_key
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{task}.parquet"


def _parse(kind: str, text: str, n: int, lr_format: str = "numeric"):
    if kind == "posterior":
        v = parse_probabilities(text, n)
        return (v.tolist() if v is not None else None), v is not None
    if kind == "weights":
        v = parse_weights(text, n, lr_format)
        return (v.tolist() if v is not None else None), v is not None
    i = parse_index(text, n)
    return ([float(i)] if i is not None else None), i is not None


def run_task(engine: Engine, jobs: list[T.Job], task: str, n_path: int,
             gen: GenConfig, model_key: str, save_raw_text: bool = False) -> Path:
    final = _final_path(model_key, task)
    if final.exists():
        print(f"  [skip] {task}: already complete ({final})")
        return final
    shards = _shard_dir(model_key, task)
    done = {int(p.stem) for p in shards.glob("*.parquet")}
    n_chunks = (len(jobs) + CHUNK - 1) // CHUNK
    print(f"  [{task}] {len(jobs):,} jobs in {n_chunks} chunks "
          f"({len(done)} already done)")
    for ci in range(n_chunks):
        if ci in done:
            continue
        chunk = jobs[ci * CHUNK:(ci + 1) * CHUNK]
        t0 = time.time()
        texts = engine.generate([j.messages for j in chunk], gen, chunk[0].schema)
        dt = time.time() - t0
        rows = []
        for j, txt in zip(chunk, texts):
            parsed, ok = _parse(j.kind, txt, n_path, j.meta.get("lr_format", "numeric"))
            row = {
                "model_key": model_key, "task": task, "item_id": j.item_id,
                "kind": j.kind, "valid": ok, "parsed": parsed,
                "meta": json.dumps(j.meta),
            }
            if save_raw_text or not ok:
                # Failures always keep their text so schema-compliance failures
                # can be audited rather than guessed at.
                row["raw_text"] = txt[:4000]
            else:
                row["raw_text"] = None
            rows.append(row)
        # `raw_text` is None on success and a string on failure. Without an
        # explicit dtype polars infers Null from a leading run of successes
        # and then fails on the first failure row.
        df = pl.DataFrame(rows, schema_overrides={"parsed": pl.List(pl.Float64),
                                                  "raw_text": pl.Utf8,
                                                  "item_id": pl.Utf8,
                                                  "meta": pl.Utf8})
        df.write_parquet(shards / f"{ci}.parquet", compression="zstd")
        vr = float(np.mean([r["valid"] for r in rows]))
        print(f"    chunk {ci+1}/{n_chunks}  {len(chunk)} jobs  {dt:.1f}s  "
              f"({len(chunk)/dt:.1f}/s)  valid {100*vr:.1f}%")
    parts = sorted(shards.glob("*.parquet"), key=lambda p: int(p.stem))
    # Shards written across a schema change can disagree on `raw_text`: a
    # chunk with no failures infers Null, one with failures infers String.
    # Cast on read so a resumed task can merge shards from both.
    frames = []
    for part in parts:
        f = pl.read_parquet(part)
        if "raw_text" in f.columns:
            f = f.with_columns(pl.col("raw_text").cast(pl.Utf8))
        frames.append(f)
    pl.concat(frames, how="diagonal_relaxed").write_parquet(final, compression="zstd")
    for p in parts:
        p.unlink()
    shards.rmdir()
    print(f"  [{task}] -> {final}")
    return final


def build_jobs(task: str, b: Battery, kb) -> list[T.Job]:
    if task.startswith("elr_weights_"):
        fmt = task.rsplit("_", 1)[1]
        return list(T.elr_weight_jobs(b, kb, formats=(fmt,)))
    if task == "a1_prior_supplied":
        from coherence.data.empirical_oracle import EmpiricalOracle
        from coherence.data.kb_reconstruct import LikelihoodTable
        from coherence.data.ddxplus import load_all
        tbl = LikelihoodTable.load()
        o = EmpiricalOracle.build(load_all(), tbl.tokens, tbl.pathologies)
        return list(T.prior_supplied_jobs(b, kb, o.prior.probs.tolist()))
    fn = T.TASK_BUILDERS.get(task)
    if fn is None:
        raise KeyError(f"unknown task {task!r}")
    return list(fn(b, kb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--battery", default="build/battery/codx_battery_v1.json")
    ap.add_argument("--tasks", nargs="+", default=T.CORE_TASKS)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap jobs per task")
    ap.add_argument("--save-raw-text", action="store_true")
    args = ap.parse_args()

    spec = registry.get(args.model)
    b = Battery.load(Path(args.battery))
    kb = load_kb()
    n_path = len(b.pathologies)

    print(f"model {spec.key}  ({spec.hf_id}, {spec.quantization or 'native'}, "
          f"{spec.reasoning_tag})")
    job_sets = {t: build_jobs(t, b, kb) for t in args.tasks}
    pending = [t for t in args.tasks if not _final_path(spec.key, t).exists()]
    if not pending:
        print("  nothing to do")
        return
    for t in args.tasks:
        print(f"  {t}: {len(job_sets[t]):,} jobs")

    engine = Engine(spec, max_num_seqs=args.max_num_seqs)
    # Reasoning models need headroom for the thinking block before the JSON.
    max_tokens = args.max_tokens if spec.thinking is not True else max(args.max_tokens, 3000)
    gen = GenConfig(temperature=args.temperature, top_p=args.top_p,
                    max_tokens=max_tokens, seed=None)
    log = LOGS / f"run_{spec.key}.jsonl"
    try:
        for t in args.tasks:
            jobs = job_sets[t]
            if args.limit:
                jobs = jobs[:args.limit]
            t0 = time.time()
            run_task(engine, jobs, t, n_path, gen, spec.key, args.save_raw_text)
            with log.open("a") as fh:
                fh.write(json.dumps({"model": spec.key, "task": t, "n": len(jobs),
                                     "seconds": time.time() - t0}) + "\n")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
