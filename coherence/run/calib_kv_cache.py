"""C11 -- is an fp8 KV cache measurably different from fp16?

Decode on a single card is KV-bound, not compute-bound, and
`gpu_memory_utilization` is already at the card's ceiling, so halving KV width
is the only way left to raise concurrency. That is a change to attention
numerics, so it has to be shown harmless before it is used for any model whose
numbers enter a table.

The test reuses the instrument's own noise floor rather than an arbitrary
tolerance. A model asked the identical question twice does not answer
identically (temperature 0.7); the test-retest arm measures exactly that
spread. So the question is not "does fp8 change the answer" -- it does, as does
asking twice -- but "does fp8 change it by more than asking twice does".

  d_kv      JSD(fp8 answer, fp16 answer) on the SAME item
  d_retest  JSD between fp16 replicates of the SAME prompt  (the floor)

fp8 is accepted when d_kv does not exceed d_retest, and when the headline A1
statistic recomputed under fp8 lands inside the fp16 bootstrap CI.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from coherence.axioms.a1_order import _bootstrap_ci
from coherence.config import REPORTS, RESULTS, SEED

RAW = RESULTS / "raw"
from coherence.elicit.engine import Engine, GenConfig
from coherence.elicit.registry import get
from coherence.metrics.divergence import jsd, mean_pairwise_jsd
from coherence.run import runner as R
from coherence.run import tasks as T
from coherence.data.battery import Battery
from coherence.data.ddxplus import load_kb


def _posteriors(df: pl.DataFrame) -> np.ndarray:
    return np.array([p for p in df["parsed"].to_list()], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-32b-nothink")
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--kv", default="fp8")
    a = ap.parse_args()

    base_path = RAW / a.model / "a1_posterior.parquet"
    if not base_path.exists():
        raise SystemExit(f"no fp16 baseline at {base_path}")
    base = pl.read_parquet(base_path)
    base = base.with_columns(
        pl.col("meta").str.json_path_match("$.case_id").alias("case_id"),
        pl.col("meta").str.json_path_match("$.arm").alias("arm"),
    )

    # Deterministic case subset, so the comparison is reproducible.
    cases = sorted(base["case_id"].unique().to_list())
    rng = np.random.default_rng(SEED)
    keep = set(rng.permutation(cases)[: a.cases].tolist())

    b = Battery.load("build/battery/codx_battery_v1.json")
    kb = load_kb()
    jobs = R.build_jobs("a1_posterior", b, kb)
    sel = [j for j in jobs if j.meta.get("case_id") in keep
           and j.meta.get("arm") in ("permutation", "retest")]
    print(f"model={a.model}  kv={a.kv}  cases={len(keep)}  items={len(sel):,}")

    spec = get(a.model)
    engine = Engine(spec, max_num_seqs=a.max_num_seqs)   # CODX_KV_DTYPE drives kv
    gen = GenConfig(seed=SEED)
    texts = engine.generate([j.messages for j in sel], gen, sel[0].schema)

    n_path = len(b.pathologies)
    rows = []
    for j, txt in zip(sel, texts):
        parsed, ok = R._parse(j.kind, txt, n_path)
        rows.append({"item_id": j.item_id, "valid": ok, "parsed": parsed,
                     "case_id": j.meta["case_id"], "arm": j.meta["arm"]})
    new = pl.DataFrame(rows, schema_overrides={"parsed": pl.List(pl.Float64)})
    # Kept out of results/raw: everything under that tree is swept up by
    # the analysis and by the Hub sync as if it were battery output.
    out_dir = RESULTS / "calib"; out_dir.mkdir(parents=True, exist_ok=True)
    new.write_parquet(out_dir / f"kv_{a.kv}_{a.model}.parquet", compression="zstd")

    # ---- (1) fp8-vs-fp16 displacement on identical items -----------------
    j16 = base.filter(pl.col("valid")).select("item_id", "parsed", "case_id", "arm")
    m = new.filter(pl.col("valid")).join(j16, on="item_id", suffix="_16")
    d_kv = np.array([jsd(np.array(p), np.array(q)) for p, q in
                     zip(m["parsed"].to_list(), m["parsed_16"].to_list())])

    # ---- (2) the instrument's own floor: fp16 retest replicates ----------
    floor = []
    for (cid,), g in j16.filter(pl.col("arm") == "retest").group_by(["case_id"]):
        if cid in keep and len(g) >= 2:
            floor.append(mean_pairwise_jsd(_posteriors(g)))
    d_retest = np.array(floor)

    # ---- (3) headline A1 statistic under each KV dtype --------------------
    def perm_effect(df: pl.DataFrame) -> np.ndarray:
        v = []
        for (cid,), g in df.filter(pl.col("arm") == "permutation").group_by(["case_id"]):
            if cid in keep and len(g) >= 2:
                v.append(mean_pairwise_jsd(_posteriors(g)))
        return np.array(v)

    e16, e8 = perm_effect(j16), perm_effect(new.filter(pl.col("valid")))
    ci16, ci8 = _bootstrap_ci(e16), _bootstrap_ci(e8)

    res = {
        "model": a.model, "kv_dtype": a.kv, "n_cases": len(keep),
        "n_items": len(sel), "valid_rate_fp8": float(new["valid"].mean()),
        "d_kv_mean": float(np.nanmean(d_kv)),
        "d_kv_median": float(np.nanmedian(d_kv)),
        "d_kv_p95": float(np.nanpercentile(d_kv, 95)),
        "d_retest_mean": float(np.nanmean(d_retest)),
        "d_retest_median": float(np.nanmedian(d_retest)),
        "d_retest_p95": float(np.nanpercentile(d_retest, 95)),
        "a1_effect_fp16": float(np.nanmean(e16)), "a1_ci_fp16": ci16,
        "a1_effect_fp8": float(np.nanmean(e8)), "a1_ci_fp8": ci8,
    }
    res["within_floor"] = bool(res["d_kv_mean"] <= res["d_retest_mean"])
    res["headline_overlaps"] = bool(ci8[0] <= res["a1_effect_fp16"] <= ci8[1]
                                    and ci16[0] <= res["a1_effect_fp8"] <= ci16[1])
    res["accept"] = bool(res["within_floor"] and res["headline_overlaps"])

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "calib_kv_cache.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("\nACCEPT fp8" if res["accept"] else "\nREJECT fp8 -- keep fp16")


if __name__ == "__main__":
    main()
