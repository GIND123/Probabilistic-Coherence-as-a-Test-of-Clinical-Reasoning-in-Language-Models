"""Distributional, clinical-plausibility and split-shift checks for DDXPlus."""
from __future__ import annotations

import numpy as np
import polars as pl

from coherence.audit.checks import Finding
from coherence.data.ddxplus import DDXPlusKB

# Evidence codes whose affirmative answer is biologically tied to one sex.
# Codes resolved by question text at audit time rather than hard-coded IDs.
_FEMALE_ONLY_PATTERNS = ("pregnan", "menstru", "vagina", "uterus", "period")
_MALE_ONLY_PATTERNS = ("penis", "scrotum", "testic", "prostate")


def check_class_balance(df: pl.DataFrame, kb: DDXPlusKB) -> Finding:
    c = df.group_by("pathology").len().sort("len", descending=True)
    n = df.height
    c = c.with_columns((pl.col("len") / n).alias("prior"),
                       pl.col("pathology").replace_strict(kb.severity, default=None)
                       .alias("severity"))
    p = c["prior"].to_numpy()
    imbalance = float(p.max() / p.min())
    ent = float(-(p * np.log2(p)).sum())
    sev = (c.group_by("severity").agg(pl.col("prior").sum().alias("mass"),
                                      pl.len().alias("n_pathologies"))
           .sort("severity"))
    return Finding(
        "class_balance",
        "major" if imbalance > 20 else "minor",
        f"{c.height} pathologies; prior range {p.min():.5f}-{p.max():.5f} "
        f"(imbalance ratio {imbalance:.1f}x); prior entropy {ent:.2f} bits of "
        f"{np.log2(c.height):.2f} max",
        detail=(
            "Case sampling for the coherence battery must be stratified by pathology and by "
            "severity, otherwise the headline order-effect number is dominated by the handful "
            "of common respiratory pathologies and the severity-weighted analysis is starved "
            "of high-severity cases."
        ),
        stats={"n_pathologies": int(c.height), "imbalance_ratio": imbalance,
               "prior_entropy_bits": ent,
               "severity_mass": {int(r["severity"]): float(r["mass"]) for r in sev.iter_rows(named=True)}},
        table=c.head(50),
    )


def check_evidence_load(df: pl.DataFrame) -> Finding:
    k = df.select(pl.col("evidences").list.len().alias("n_ev"))["n_ev"].to_numpy()
    q = np.percentile(k, [0, 1, 25, 50, 75, 99, 100])
    # Tukey fence on the upper tail.
    q1, q3 = np.percentile(k, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    n_out = int((k > fence).sum())
    return Finding(
        "evidence_load",
        "info",
        f"evidences per patient: median {np.median(k):.0f}, IQR {q1:.0f}-{q3:.0f}, "
        f"range {k.min()}-{k.max()}; {n_out:,} patients above the Tukey fence ({fence:.0f})",
        detail=(
            "This distribution sets the feasible range for the evidence-set-size ablation and "
            "bounds the cost of ELR-Fusion, which issues one query per finding."
        ),
        stats={"quantiles": dict(zip(["min", "p1", "p25", "p50", "p75", "p99", "max"],
                                     [float(x) for x in q])),
               "tukey_fence": float(fence), "n_above_fence": n_out,
               "mean": float(k.mean())},
    )


def check_split_shift(df: pl.DataFrame) -> Finding:
    """Population stability index between splits on the pathology marginal."""
    piv = (df.group_by("split", "pathology").len()
           .pivot(on="split", index="pathology", values="len").fill_null(0))
    cols = [c for c in piv.columns if c != "pathology"]
    mats = {c: piv[c].to_numpy().astype(float) for c in cols}
    mats = {c: v / v.sum() for c, v in mats.items()}
    eps = 1e-9
    psi = {}
    ref = "train"
    for c in cols:
        if c == ref:
            continue
        a, b = mats[ref] + eps, mats[c] + eps
        psi[f"{ref}->{c}"] = float(np.sum((b - a) * np.log(b / a)))
    worst = max(psi.values()) if psi else 0.0
    return Finding(
        "split_shift",
        "major" if worst > 0.25 else ("minor" if worst > 0.1 else "info"),
        f"pathology-marginal population stability index: "
        + ", ".join(f"{k}={v:.4f}" for k, v in psi.items())
        + "  (<0.1 negligible, 0.1-0.25 moderate, >0.25 material)",
        stats={"psi": psi, "split_sizes": {c: int(df.filter(pl.col("split") == c).height)
                                           for c in cols}},
        table=piv,
    )


def _codes_matching(kb: DDXPlusKB, patterns: tuple[str, ...]) -> list[str]:
    return [c for c, e in kb.evidences.items()
            if any(p in e.question_en.lower() for p in patterns)]


def check_sex_plausibility(df: pl.DataFrame, kb: DDXPlusKB) -> Finding:
    fem = _codes_matching(kb, _FEMALE_ONLY_PATTERNS)
    mal = _codes_matching(kb, _MALE_ONLY_PATTERNS)
    ex = (df.select("sex", "evidences").explode("evidences").rename({"evidences": "tok"})
          .with_columns(pl.col("tok").str.extract(r"^(E_\d+)").alias("code")))
    viol_f = ex.filter(pl.col("code").is_in(fem) & (pl.col("sex").cast(pl.Utf8) == "M"))
    viol_m = ex.filter(pl.col("code").is_in(mal) & (pl.col("sex").cast(pl.Utf8) == "F"))
    tbl = (pl.concat([viol_f.group_by("code").len().with_columns(pl.lit("M_with_female_specific").alias("kind")),
                      viol_m.group_by("code").len().with_columns(pl.lit("F_with_male_specific").alias("kind"))])
           .sort("len", descending=True))
    n = int(viol_f.height + viol_m.height)
    return Finding(
        "sex_plausibility",
        "major" if n else "info",
        f"{n:,} sex-implausible evidence assertions "
        f"({len(fem)} female-specific and {len(mal)} male-specific questions screened)",
        detail=(
            "Screened by question wording, so it is a lower bound and may contain false "
            "positives where a question merely mentions an anatomical term. Any true positives "
            "are generator artefacts and the affected cases are excluded from the clinical "
            "plausibility subset used for the clinician-review arm."
        ),
        stats={"n_violations": n, "female_specific_codes": fem, "male_specific_codes": mal,
               "example_questions": {c: kb.evidences[c].question_en for c in (fem + mal)[:12]}},
        table=tbl if tbl.height else None,
    )


def check_age_plausibility(df: pl.DataFrame, kb: DDXPlusKB) -> Finding:
    """Evidences asserted in patients too young for them to be possible."""
    smoke = [c for c, e in kb.evidences.items() if "smok" in e.question_en.lower()]
    occupational = [c for c, e in kb.evidences.items()
                    if "work" in e.question_en.lower() or "occupation" in e.question_en.lower()]
    travel = [c for c, e in kb.evidences.items() if "travel" in e.question_en.lower()]
    codes = sorted(set(smoke + occupational + travel))
    ex = (df.select("age", "evidences").explode("evidences").rename({"evidences": "tok"})
          .with_columns(pl.col("tok").str.extract(r"^(E_\d+)").alias("code"))
          .filter(pl.col("code").is_in(codes)))
    rows = (ex.group_by("code").agg(pl.len().alias("n"), pl.col("age").min().alias("min_age"),
                                    pl.col("age").quantile(0.01).alias("p01_age"))
            .sort("min_age"))
    n_child = int(ex.filter(pl.col("age") < 10).height)
    return Finding(
        "age_plausibility",
        "minor" if n_child else "info",
        f"{n_child:,} assertions of smoking / occupational-exposure / travel history in "
        f"patients under 10 years old across {len(codes)} screened questions",
        detail=(
            "Reported as a realism limitation of the synthetic generator, not as a defect "
            "that threatens the coherence measurement: order invariance holds regardless of "
            "whether the case is clinically plausible. It does matter for the MIMIC "
            "external-validity arm, which is the reason that arm exists."
        ),
        stats={"n_under_10": n_child, "screened_codes": codes},
        table=rows.with_columns(
            pl.col("code").replace_strict({c: kb.evidences[c].question_en for c in codes},
                                          default=None).alias("question")),
    )


def check_severity_coverage(df: pl.DataFrame, kb: DDXPlusKB) -> Finding:
    sev = kb.severity
    d = df.with_columns(pl.col("pathology").replace_strict(sev, default=None).alias("severity"))
    g = d.group_by("severity").len().sort("severity")
    hs = d.filter(pl.col("severity") <= 2)
    return Finding(
        "severity_coverage",
        "info",
        "severity distribution (1 = most severe): "
        + ", ".join(f"sev{r['severity']}={r['len']:,}" for r in g.iter_rows(named=True))
        + f"; {hs.height:,} cases at severity <= 2 are available for the "
          "severity-weighted incoherence analysis",
        stats={"counts": {int(r["severity"]): int(r["len"]) for r in g.iter_rows(named=True)},
               "n_high_severity": int(hs.height)},
        table=g,
    )
