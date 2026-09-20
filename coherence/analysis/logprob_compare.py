"""#6  Token-logprob vs verbalized probability vs named diagnosis, all 11 models.

Three ways of asking the same weights the same question, on the same patients:

  stated posterior   49 plausibility scores, written out, then normalised
  named diagnosis    one index, 0-48
  token logprob      each of the 49 names scored as a continuation

The logprob path is reported under BOTH scoring rules, because they disagree
and choosing per model would be selection on the outcome. Summed log-probability
favours short diagnosis names; length-normalisation removes that bias but
discards genuine evidence that longer names are less likely a priori. Neither is
obviously correct, so both are given and the fixed-rule conclusion is stated for
each.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import polars as pl

from coherence.analysis.reviewer_response import cluster_bootstrap
from coherence.axioms.loading import load_task, posterior_matrix
from coherence.config import TABLES
from coherence.data.battery import Battery
from coherence.elicit.registry import DEFAULT_ORDER

BATTERY = "build/battery/codx_battery_v1.json"


def main() -> None:
    b = Battery.load(BATTERY)
    paths = list(b.pathologies)
    pidx = {p: i for i, p in enumerate(paths)}
    c2p = {c.case_id: pidx.get(c.pathology, -1) for c in b.cases}
    order = {k: i for i, k in enumerate(DEFAULT_ORDER)}

    rows = []
    for f in sorted(glob.glob("results/raw/*/logprob_baseline.parquet")):
        m = os.path.basename(os.path.dirname(f))
        d = pl.read_parquet(f).with_columns(
            pl.col("meta").str.json_path_match("$.scoring").alias("sc"),
            pl.col("meta").str.json_path_match("$.truth_index").cast(pl.Int64).alias("ti"),
            pl.col("meta").str.json_path_match("$.case_id").alias("cid"))
        r = {"model": m, "n_cases": d.filter(pl.col("sc") == "sum").height}
        for sc in ("sum", "meanlen"):
            g = d.filter(pl.col("sc") == sc)
            P = np.array(g["parsed"].to_list())
            t = g["ti"].to_numpy()
            cid = g["cid"].to_list()
            hit = {c: [float(P[i].argmax() == t[i])] for i, c in enumerate(cid)}
            pt, lo, hi = cluster_bootstrap(hit, seed=61)
            r[f"logprob_{sc}_top1"] = pt
            r[f"logprob_{sc}_lo"] = lo
            r[f"logprob_{sc}_hi"] = hi
            r[f"logprob_{sc}_top5"] = float(np.mean(
                [t[i] in np.argsort(-P[i])[:5] for i in range(len(P))]))

        keep = set(d["cid"].to_list())
        dp = load_task(m, "a1_posterior")
        if dp is not None:
            Pp, _ = posterior_matrix(dp)
            dp = dp.with_row_index("row")
            sub = dp.filter(pl.col("valid") & pl.col("case_id").is_in(list(keep)))
            M = Pp[sub["row"].to_numpy()]
            cids = sub["case_id"].to_list()
            hit = {}
            for i, c in enumerate(cids):
                ti = c2p.get(c, -1)
                if ti >= 0:
                    hit.setdefault(c, []).append(float(M[i].argmax() == ti))
            pt, lo, hi = cluster_bootstrap(hit, seed=62)
            r["stated_posterior_top1"], r["stated_lo"], r["stated_hi"] = pt, lo, hi
        da = load_task(m, "a1_answer")
        if da is not None:
            sa = da.filter(pl.col("valid") & pl.col("case_id").is_in(list(keep)))
            hit = {}
            for c, p in zip(sa["case_id"].to_list(), sa["parsed"].to_list()):
                ti = c2p.get(c, -1)
                if ti >= 0 and p:
                    hit.setdefault(c, []).append(float(int(p[0]) == ti))
            pt, lo, hi = cluster_bootstrap(hit, seed=63)
            r["named_top1"], r["named_lo"], r["named_hi"] = pt, lo, hi
        rows.append(r)

    out = pl.DataFrame(rows).with_columns([
        (pl.col("logprob_sum_top1") / pl.col("stated_posterior_top1")).alias("sum_over_stated"),
        (pl.col("logprob_meanlen_top1") / pl.col("stated_posterior_top1")).alias("meanlen_over_stated"),
        (pl.col("logprob_sum_top5") - pl.col("logprob_sum_top1")).alias("sum_top5_minus_top1"),
    ]).sort(pl.col("model").replace_strict(order, default=99))
    out.write_csv(TABLES / "rev_logprob_baseline.csv")
    print(f"  rev_logprob_baseline.csv ({out.height} models)")
    return out


if __name__ == "__main__":
    o = main()
    pl.Config.set_tbl_rows(14); pl.Config.set_tbl_cols(9)
    pl.Config.set_fmt_str_lengths(20)
    print(o.select("model", "logprob_sum_top1", "logprob_meanlen_top1",
                   "stated_posterior_top1", "named_top1",
                   "meanlen_over_stated", "sum_top5_minus_top1"))
    n_ml = int((o["logprob_meanlen_top1"] > o["stated_posterior_top1"]).sum())
    n_s = int((o["logprob_sum_top1"] > o["stated_posterior_top1"]).sum())
    n_nm = int((o["named_top1"] > o["stated_posterior_top1"]).sum())
    print(f"\n  length-normalised beats stated posterior : {n_ml}/{o.height}")
    print(f"  summed            beats stated posterior : {n_s}/{o.height}")
    print(f"  named             beats stated posterior : {n_nm}/{o.height}")
    print(f"  named beats BOTH logprob variants        : "
          f"{int(((o['named_top1']>o['logprob_sum_top1'])&(o['named_top1']>o['logprob_meanlen_top1'])).sum())}/{o.height}")
