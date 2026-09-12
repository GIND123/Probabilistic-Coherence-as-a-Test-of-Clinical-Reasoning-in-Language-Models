"""Data-audit checks for DDXPlus.

Each public function returns a `Finding`. Findings carry a severity so the
report can be sorted by how much each issue threatens the experimental
design, not by the order the checks happen to run in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from coherence.data.ddxplus import DDXPlusKB, split_token

SEVERITIES = ("critical", "major", "minor", "info")


@dataclass
class Finding:
    check: str
    severity: str
    headline: str
    detail: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    table: pl.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(self.severity)


def _pct(n: int, d: int) -> str:
    return f"{n} ({100 * n / d:.4f}%)" if d else "0"


# --------------------------------------------------------------------------
# A. Corpus integrity
# --------------------------------------------------------------------------
def check_schema_and_nulls(df: pl.DataFrame) -> Finding:
    nulls = {c: int(df[c].null_count()) for c in df.columns}
    bad = {c: n for c, n in nulls.items() if n}
    empty_ev = int(df.filter(pl.col("evidences").list.len() == 0).height)
    empty_ddx = int(df.filter(pl.col("ddx_names").list.len() == 0).height)
    sev = "critical" if bad or empty_ddx else "info"
    return Finding(
        "schema_and_nulls",
        sev,
        f"{df.height:,} records; null fields: {bad or 'none'}; "
        f"empty evidence lists: {empty_ev}; empty differentials: {empty_ddx}",
        stats={"n_rows": df.height, "nulls": nulls, "empty_evidences": empty_ev,
               "empty_differentials": empty_ddx},
    )


def check_duplicates(df: pl.DataFrame) -> Finding:
    keyed = df.with_columns(
        (pl.col("evidences").list.sort().list.join("|") + "##" + pl.col("age").cast(pl.Utf8)
         + "##" + pl.col("sex").cast(pl.Utf8) + "##" + pl.col("pathology")
         + "##" + pl.col("initial_evidence")).alias("k")
    )
    n_dup = keyed.height - keyed["k"].n_unique()
    g = keyed.group_by("k").agg(pl.col("split").n_unique().alias("ns"), pl.len().alias("n"))
    cross = g.filter(pl.col("ns") > 1)
    n_cross_records = int(keyed.join(cross.select("k"), on="k", how="semi").height)
    return Finding(
        "duplicates",
        "major",
        f"{_pct(n_dup, keyed.height)} of records are exact duplicates of another record; "
        f"{cross.height:,} distinct cases appear in more than one split "
        f"({n_cross_records:,} records), i.e. train/test leakage",
        detail=(
            "A case is keyed on (evidence multiset, age, sex, pathology, initial evidence). "
            "Cross-split repetition means any supervised model trained on DDXPlus train and "
            "scored on DDXPlus test has memorisation headroom. It does not affect the "
            "coherence measurements in this project, which are within-case and label-free, "
            "but it is reported because downstream users of the corpus are affected."
        ),
        stats={"n_exact_duplicate_records": n_dup,
               "n_cases_in_multiple_splits": int(cross.height),
               "n_records_in_multiple_splits": n_cross_records},
    )


def check_demographics(df: pl.DataFrame) -> Finding:
    age = df["age"].to_numpy()
    sexes = df["sex"].unique().to_list()
    out_of_range = int(((age < 0) | (age > 120)).sum())
    q = np.percentile(age, [0, 1, 25, 50, 75, 99, 100])
    return Finding(
        "demographics",
        "critical" if out_of_range or set(map(str, sexes)) - {"M", "F"} else "info",
        f"age range [{age.min()}, {age.max()}], median {np.median(age):.0f}; "
        f"sex values {sorted(map(str, sexes))}; out-of-range ages: {out_of_range}",
        stats={"age_quantiles": dict(zip(["min", "p1", "p25", "p50", "p75", "p99", "max"],
                                         [float(x) for x in q])),
               "sex_values": sorted(map(str, sexes)),
               "n_age_out_of_range": out_of_range},
    )


def check_evidence_tokens(df: pl.DataFrame, kb: DDXPlusKB) -> Finding:
    ex = df.select(pl.col("evidences").explode().alias("tok"))
    counts = ex.group_by("tok").len().sort("len", descending=True)
    unknown_code, bad_value, malformed = [], [], []
    for tok in counts["tok"].to_list():
        try:
            code, value = split_token(tok)
        except ValueError:
            malformed.append(tok)
            continue
        ev = kb.evidences.get(code)
        if ev is None:
            unknown_code.append(tok)
            continue
        if value is None:
            if not ev.is_binary:
                bad_value.append(tok)  # non-binary evidence with no value
        else:
            allowed = set(ev.possible_values) | set(ev.value_meaning)
            if allowed and value not in allowed:
                bad_value.append(tok)
    n_bad = len(unknown_code) + len(bad_value) + len(malformed)
    return Finding(
        "evidence_token_validity",
        "critical" if n_bad else "info",
        f"{counts.height:,} distinct evidence tokens over {len(kb.evidences)} question codes; "
        f"unknown codes: {len(unknown_code)}, invalid values: {len(bad_value)}, "
        f"malformed: {len(malformed)}",
        stats={"n_distinct_tokens": int(counts.height),
               "unknown_codes": unknown_code[:20], "invalid_values": bad_value[:20],
               "malformed": malformed[:20]},
        table=counts.head(30).rename({"len": "n"}),
    )


# --------------------------------------------------------------------------
# B. Label integrity
# --------------------------------------------------------------------------
def check_differential_wellformed(df: pl.DataFrame) -> Finding:
    s = df.select(
        pl.col("ddx_probs").list.sum().alias("total"),
        pl.col("ddx_probs").list.len().alias("k"),
        pl.col("ddx_names").list.len().alias("kn"),
    )
    tot = s["total"].to_numpy()
    off = int((np.abs(tot - 1.0) > 1e-6).sum())
    mism = int((s["k"] != s["kn"]).sum())
    monotone = df.select(
        pl.col("ddx_probs")
        .map_elements(lambda p: all(a >= b - 1e-12 for a, b in zip(p, p[1:])),
                      return_dtype=pl.Boolean)
        .alias("mono")
    )["mono"]
    n_nonmono = int((~monotone).sum())
    neg = int(df.select(pl.col("ddx_probs").list.min().alias("m"))["m"].to_numpy().min() < 0)
    return Finding(
        "differential_wellformed",
        "critical" if off or mism or neg else "info",
        f"differentials sum to 1 within 1e-6 for {_pct(df.height - off, df.height)}; "
        f"length mismatches: {mism}; non-monotone orderings: {n_nonmono}; negatives: {neg}",
        stats={"n_not_normalised": off, "n_length_mismatch": mism,
               "n_non_monotone": n_nonmono, "max_abs_dev_from_1": float(np.abs(tot - 1).max()),
               "k_quantiles": dict(zip(["min", "p25", "p50", "p75", "p95", "max"],
                                       [float(x) for x in np.percentile(s["k"].to_numpy(),
                                                                       [0, 25, 50, 75, 95, 100])]))},
    )


def check_true_pathology_in_differential(df: pl.DataFrame) -> Finding:
    d = df.with_columns(
        pl.col("ddx_names").list.contains(pl.col("pathology")).alias("in_ddx"),
    )
    n_missing = int((~d["in_ddx"]).sum())
    ranks = d.filter(pl.col("in_ddx")).select(
        pl.struct(["ddx_names", "pathology"])
        .map_elements(lambda r: r["ddx_names"].index(r["pathology"]) + 1, return_dtype=pl.Int32)
        .alias("rank")
    )["rank"].to_numpy()
    top1 = float((ranks == 1).mean())
    return Finding(
        "true_pathology_in_differential",
        "major" if n_missing else "info",
        f"ground-truth pathology absent from its own differential in "
        f"{_pct(n_missing, df.height)}; it is rank-1 in {100 * top1:.2f}% of the rest "
        f"(median rank {np.median(ranks):.0f})",
        detail=(
            "Rank-1 agreement well below 100% is expected and is not an error: the "
            "differential is the generator's posterior, while PATHOLOGY is the sampled "
            "ground truth. It does bound achievable top-1 accuracy for any method, and "
            "that bound belongs in the results table."
        ),
        stats={"n_truth_not_in_ddx": n_missing, "top1_rate": top1,
               "mean_rank": float(ranks.mean()), "median_rank": float(np.median(ranks)),
               "top5_rate": float((ranks <= 5).mean())},
    )


# --------------------------------------------------------------------------
# C. Knowledge-base consistency
# --------------------------------------------------------------------------
def check_kb_coverage(df: pl.DataFrame, kb: DDXPlusKB) -> Finding:
    obs = (
        df.select("pathology", "evidences").explode("evidences")
        .rename({"evidences": "tok"})
        .with_columns(pl.col("tok").str.extract(r"^(E_\d+)").alias("code"))
        .group_by("pathology", "code").len()
    )
    rows = []
    for r in obs.iter_rows(named=True):
        cond = kb.conditions.get(r["pathology"])
        if cond is None:
            rows.append((r["pathology"], r["code"], r["len"], "unknown_pathology"))
        elif r["code"] not in cond.allowed_evidences:
            rows.append((r["pathology"], r["code"], r["len"], "evidence_not_in_kb_for_pathology"))
    viol = pl.DataFrame(rows, schema=["pathology", "code", "n", "kind"], orient="row")
    seen_paths = set(df["pathology"].unique().to_list())
    unseen_paths = sorted(set(kb.pathologies) - seen_paths)
    seen_codes = set(obs["code"].unique().to_list())
    unseen_codes = sorted(set(kb.evidences) - seen_codes)
    return Finding(
        "kb_coverage",
        "major" if viol.height else "info",
        f"{viol.height:,} (pathology, evidence) pairs observed in data but not declared in the "
        f"knowledge base; {len(unseen_paths)} declared pathologies never observed; "
        f"{len(unseen_codes)} declared evidences never observed",
        detail=(
            "Declared-but-unobserved is benign. Observed-but-undeclared means the structural "
            "KB cannot be used as a hard support constraint when reconstructing likelihoods: "
            "zeroing out undeclared evidence would assign probability 0 to cases that the "
            "generator actually produced."
        ),
        stats={"n_violating_pairs": int(viol.height),
               "unseen_pathologies": unseen_paths, "unseen_evidence_codes": unseen_codes},
        table=viol.sort("n", descending=True).head(30) if viol.height else None,
    )


# --------------------------------------------------------------------------
# D. Determinism / irreducible ground-truth noise
# --------------------------------------------------------------------------
def check_generator_determinism(df: pl.DataFrame, kb: DDXPlusKB) -> Finding:
    """Are two identically-specified patients given the same differential?

    This is the single most consequential check in the audit: it sets an
    irreducible floor below which no coherence measurement is interpretable.
    """
    from coherence.metrics.divergence import jsd

    P = kb.pathologies
    idx = {p: i for i, p in enumerate(P)}

    keyed = df.with_columns(
        (pl.col("evidences").list.sort().list.join("|") + "##" + pl.col("age").cast(pl.Utf8)
         + "##" + pl.col("sex").cast(pl.Utf8) + "##" + pl.col("initial_evidence")
         + "##" + pl.col("pathology")).alias("k")
    )
    g = (keyed.group_by("k")
         .agg(pl.len().alias("n"), pl.col("ddx_names"), pl.col("ddx_probs"))
         .filter(pl.col("n") > 1))

    jsds, l1s, flips = [], [], 0
    for r in g.iter_rows(named=True):
        vs = []
        for names, probs in zip(r["ddx_names"], r["ddx_probs"]):
            v = np.zeros(len(P))
            for nm, p in zip(names, probs):
                v[idx[nm]] = p
            vs.append(v)
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                jsds.append(jsd(vs[i], vs[j]))
                l1s.append(float(np.abs(vs[i] - vs[j]).sum()))
                flips += int(vs[i].argmax() != vs[j].argmax())
    jsds = np.asarray(jsds)
    l1s = np.asarray(l1s)
    n = len(jsds)
    frac_disagree = float((jsds > 1e-12).mean()) if n else 0.0
    return Finding(
        "generator_determinism",
        "critical" if frac_disagree > 0.01 else "info",
        f"{n:,} replicate case pairs; the generator returns a DIFFERENT differential for "
        f"{100 * frac_disagree:.2f}% of them (mean JSD {jsds.mean():.5f}, "
        f"p90 {np.percentile(jsds, 90):.5f}, max {jsds.max():.5f}); the top-1 diagnosis "
        f"flips in {100 * flips / max(n, 1):.2f}% of replicate pairs",
        detail=(
            "Two patients identical in evidence multiset, age, sex, initial evidence and true "
            "pathology can receive materially different ground-truth differentials. No released "
            "field explains the difference, so it is irreducible from the corpus alone. "
            "Consequence for this project: the ground truth itself has an order-of-1e-2 JSD "
            "noise floor. Any model order-effect must be reported against it, alongside the "
            "model's own test-retest floor. Consequence for the wider literature: DDXPlus "
            "differential-matching metrics have a hard ceiling that is not 1.0, and papers "
            "reporting near-perfect differential reproduction should be read with that in mind."
        ),
        stats={"n_replicate_pairs": n, "frac_pairs_disagreeing": frac_disagree,
               "jsd_mean": float(jsds.mean()) if n else 0.0,
               "jsd_median": float(np.median(jsds)) if n else 0.0,
               "jsd_p90": float(np.percentile(jsds, 90)) if n else 0.0,
               "jsd_p99": float(np.percentile(jsds, 99)) if n else 0.0,
               "jsd_max": float(jsds.max()) if n else 0.0,
               "l1_mean": float(l1s.mean()) if n else 0.0,
               "top1_flip_rate": flips / max(n, 1)},
    )
