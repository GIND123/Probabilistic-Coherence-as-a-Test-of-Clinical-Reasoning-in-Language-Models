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


def posterior_matrix(d: pl.DataFrame, repair: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return (posteriors, valid_mask). Unusable rows are NaN-filled.

    `repair` re-parses rows the runner marked invalid, using the raw text it
    kept for exactly that purpose. Results generated before all-zero
    plausibility vectors were reclassified as degenerate-but-valid are
    recovered here rather than re-generated.
    """
    from coherence.elicit.engine import parse_probabilities

    rows = d["parsed"].to_list()
    valid = d["valid"].to_numpy().copy()
    n = len(next(r for r in rows if r))
    out = np.full((len(rows), n), np.nan)
    raw = d["raw_text"].to_list() if "raw_text" in d.columns else [None] * len(rows)
    for i, r in enumerate(rows):
        if r:
            out[i] = r
        elif repair and raw[i]:
            v = parse_probabilities(raw[i], n)
            if v is not None:
                out[i] = v
                valid[i] = True
    return out, valid


def schema_validity(model_key: str) -> dict[str, float]:
    out = {}
    for p in (RAW / model_key).glob("*.parquet"):
        d = pl.read_parquet(p, columns=["valid"])
        out[p.stem] = float(d["valid"].mean())
    return out
