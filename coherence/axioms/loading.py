"""Load raw elicitation results into aligned numpy arrays."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from coherence.config import RESULTS

RAW = RESULTS / "raw"


def load_task(model_key: str, task: str) -> pl.DataFrame | None:
    p = RAW / model_key / f"{task}.parquet"
    if not p.exists():
        return None
    d = pl.read_parquet(p)
    meta = pl.DataFrame([json.loads(m) for m in d["meta"].to_list()])
    return pl.concat([d.drop("meta"), meta], how="horizontal")


def available_models(task: str = "a1_posterior") -> list[str]:
    return sorted(p.parent.name for p in RAW.glob(f"*/{task}.parquet"))


def posterior_matrix(d: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (posteriors, valid_mask). Invalid rows are NaN-filled."""
    rows = d["parsed"].to_list()
    valid = d["valid"].to_numpy()
    n = len(next(r for r in rows if r))
    out = np.full((len(rows), n), np.nan)
    for i, r in enumerate(rows):
        if r:
            out[i] = r
    return out, valid


def schema_validity(model_key: str) -> dict[str, float]:
    out = {}
    for p in (RAW / model_key).glob("*.parquet"):
        d = pl.read_parquet(p, columns=["valid"])
        out[p.stem] = float(d["valid"].mean())
    return out
