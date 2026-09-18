"""DDXPlus ingest, knowledge-base parsing, and evidence-code algebra.

The public DDXPlus release ships `release_conditions.json` with every
per-evidence slot as an empty dict -- the generator's conditional
probability tables are *not* released. Everything probabilistic in this
project therefore has to be reconstructed from the 1.3M released patients;
see `coherence.data.kb_reconstruct`.

This module owns the deterministic half: parsing, normalising and
validating the corpus and the structural knowledge base.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import polars as pl

from coherence.config import CONDITIONS_JSON, DDXPLUS_SPLITS, EVIDENCES_JSON, PARQUET

# An evidence token is either "E_53" (binary) or "E_55_@_V_29" / "E_56_@_6"
# (categorical / multi-choice with a value suffix).
EVIDENCE_TOKEN = re.compile(r"^(E_\d+)(?:_@_(.+))?$")


def split_token(token: str) -> tuple[str, str | None]:
    """Split an evidence token into (code, value) where value may be None."""
    m = EVIDENCE_TOKEN.match(token)
    if not m:
        raise ValueError(f"unparseable evidence token: {token!r}")
    return m.group(1), m.group(2)


@dataclass(frozen=True)
class Evidence:
    """One question in the DDXPlus evidence schema."""

    name: str
    question_en: str
    is_antecedent: bool
    data_type: str  # B binary, C categorical (ordered), M multi-choice
    default_value: str | int
    possible_values: tuple[str, ...]
    value_meaning: dict[str, str]

    @property
    def is_binary(self) -> bool:
        return self.data_type == "B"

    def render_value(self, value: str | None) -> str:
        if value is None:
            return "present"
        return self.value_meaning.get(value, str(value))


@dataclass(frozen=True)
class Condition:
    """One pathology, with the evidences the generator may emit for it."""

    name: str
    icd10: str
    severity: int
    symptoms: tuple[str, ...]
    antecedents: tuple[str, ...]

    @property
    def allowed_evidences(self) -> frozenset[str]:
        return frozenset(self.symptoms) | frozenset(self.antecedents)


@dataclass
class DDXPlusKB:
    """Structural knowledge base: which evidences each pathology may show."""

    conditions: dict[str, Condition]
    evidences: dict[str, Evidence]
    _by_icd: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_icd = {c.icd10: n for n, c in self.conditions.items()}

    @property
    def pathologies(self) -> list[str]:
        return sorted(self.conditions)

    @property
    def severity(self) -> dict[str, int]:
        return {n: c.severity for n, c in self.conditions.items()}

    def question(self, code: str) -> str:
        return self.evidences[code].question_en

    def describe(self, token: str) -> str:
        """Human-readable rendering of one evidence token, for prompting."""
        code, value = split_token(token)
        ev = self.evidences[code]
        if ev.is_binary:
            return ev.question_en.rstrip("?") + "? Yes"
        return f"{ev.question_en.rstrip('?')}? {ev.render_value(value)}"


@lru_cache(maxsize=1)
def load_kb(
    conditions_path: Path = CONDITIONS_JSON, evidences_path: Path = EVIDENCES_JSON
) -> DDXPlusKB:
    raw_c = json.loads(Path(conditions_path).read_text())
    raw_e = json.loads(Path(evidences_path).read_text())

    evidences = {
        name: Evidence(
            name=name,
            question_en=v["question_en"],
            is_antecedent=bool(v["is_antecedent"]),
            data_type=v["data_type"],
            default_value=v["default_value"],
            possible_values=tuple(str(x) for x in v.get("possible-values", [])),
            value_meaning={k: m["en"] for k, m in (v.get("value_meaning") or {}).items()},
        )
        for name, v in raw_e.items()
    }
    conditions = {
        name: Condition(
            name=v["condition_name"],
            icd10=v["icd10-id"],
            severity=int(v["severity"]),
            symptoms=tuple(v.get("symptoms", {})),
            antecedents=tuple(v.get("antecedents", {})),
        )
        for name, v in raw_c.items()
    }
    return DDXPlusKB(conditions=conditions, evidences=evidences)


def kb_has_conditional_probabilities(
    conditions_path: Path = CONDITIONS_JSON,
) -> tuple[bool, int, int]:
    """The week-1 gate. Returns (has_probs, n_nonempty_slots, n_slots)."""
    raw = json.loads(Path(conditions_path).read_text())
    total = nonempty = 0
    for cond in raw.values():
        for group in ("symptoms", "antecedents"):
            for meta in cond.get(group, {}).values():
                total += 1
                nonempty += bool(meta)
    return nonempty > 0, nonempty, total


def _parse_differential(s: str) -> list[tuple[str, float]]:
    return [(n, float(p)) for n, p in ast.literal_eval(s)]


def ingest_split(split: str, force: bool = False) -> Path:
    """Parse one raw CSV split into a typed parquet file. Returns its path."""
    out = PARQUET / f"ddxplus_{split}.parquet"
    if out.exists() and not force:
        return out
    src = DDXPLUS_SPLITS[split]
    df = pl.read_csv(src, infer_schema_length=0)
    df = df.with_columns(
        pl.col("AGE").cast(pl.Int32).alias("age"),
        pl.col("SEX").cast(pl.Categorical).alias("sex"),
        pl.col("PATHOLOGY").alias("pathology"),
        pl.col("INITIAL_EVIDENCE").alias("initial_evidence"),
        pl.col("EVIDENCES")
        .map_elements(ast.literal_eval, return_dtype=pl.List(pl.Utf8))
        .alias("evidences"),
        pl.col("DIFFERENTIAL_DIAGNOSIS")
        .map_elements(
            lambda s: [n for n, _ in _parse_differential(s)], return_dtype=pl.List(pl.Utf8)
        )
        .alias("ddx_names"),
        pl.col("DIFFERENTIAL_DIAGNOSIS")
        .map_elements(
            lambda s: [p for _, p in _parse_differential(s)], return_dtype=pl.List(pl.Float64)
        )
        .alias("ddx_probs"),
    ).select(
        "age", "sex", "pathology", "initial_evidence", "evidences", "ddx_names", "ddx_probs"
    )
    df = df.with_row_index("row_id").with_columns(pl.lit(split).alias("split"))
    df.write_parquet(out, compression="zstd")
    return out


def load_split(split: str) -> pl.DataFrame:
    return pl.read_parquet(ingest_split(split))


def load_all() -> pl.DataFrame:
    return pl.concat([load_split(s) for s in DDXPLUS_SPLITS], how="vertical")
