"""Assemble the elicited weight table and the ELR-Fusion posteriors."""
from __future__ import annotations

import numpy as np
import polars as pl

from coherence.axioms.loading import load_task, posterior_matrix
from coherence.methods.elr_fusion import WeightTable


def build_weight_table(model_key: str, lr_format: str,
                       pathologies: list[str]) -> WeightTable | None:
    d = load_task(model_key, f"elr_weights_{lr_format}")
    if d is None:
        return None
    findings = d["finding"].to_list()
    D = len(pathologies)
    W = np.zeros((len(findings), D))
    valid = np.zeros(len(findings), dtype=bool)
    for i, (v, ok) in enumerate(zip(d["parsed"].to_list(), d["valid"].to_list())):
        if ok and v and len(v) == D:
            W[i] = v
            valid[i] = True
    return WeightTable(findings=findings, pathologies=pathologies, log_lr=W, valid=valid)


def build_elicited_prior(model_key: str, pathologies: list[str]) -> dict | None:
    """Demographic priors, keyed by (age_lo, age_hi, sex)."""
    d = load_task(model_key, "elr_prior")
    if d is None:
        return None
    out = {}
    for r in d.iter_rows(named=True):
        if r["valid"] and r["parsed"]:
            out[(int(r["age_lo"]), int(r["age_hi"]), r["sex"])] = np.asarray(r["parsed"])
    return out


def prior_for(priors: dict, age: int, sex: str, fallback: np.ndarray) -> np.ndarray:
    for (lo, hi, s), v in priors.items():
        if s == sex and lo <= age <= hi:
            return v
    return fallback
