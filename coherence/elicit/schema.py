"""Structured-output schemas for posterior and likelihood-ratio elicitation.

Everything is decoded under a grammar constraint so that schema compliance is
not itself a confound: a model that fails to emit valid JSON under one
evidence ordering and succeeds under another would register as incoherent for
reasons that have nothing to do with belief revision.
"""
from __future__ import annotations


def posterior_schema(n_pathologies: int) -> dict:
    """A plausibility score for every candidate, in the fixed presented order.

    The range is 0-100, not 0-1: see calibration C6. Scores are normalised to
    a posterior at parse time, which is a monotone transform and preserves the
    ordering the model expressed.
    """
    return {
        "type": "object",
        "properties": {
            "probabilities": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 100},
                "minItems": n_pathologies,
                "maxItems": n_pathologies,
            }
        },
        "required": ["probabilities"],
        "additionalProperties": False,
    }


def likelihood_ratio_schema(n_pathologies: int, fmt: str = "numeric") -> dict:
    """One evidence weight per candidate for a single finding.

    `fmt` is ablation #7: the same quantity asked for three ways.
      numeric  -- the likelihood ratio itself, P(finding|disease)/P(finding|not disease)
      logodds  -- its natural logarithm, which is what the fusion actually sums
      ordinal  -- a verbal band, mapped to a log-LR by `ORDINAL_BANDS`
    """
    if fmt == "numeric":
        item = {"type": "number", "minimum": 0.001, "maximum": 1000.0}
    elif fmt == "logodds":
        item = {"type": "number", "minimum": -7.0, "maximum": 7.0}
    elif fmt == "ordinal":
        item = {"type": "string", "enum": list(ORDINAL_BANDS)}
    else:
        raise ValueError(fmt)
    return {
        "type": "object",
        "properties": {
            "weights": {"type": "array", "items": item,
                        "minItems": n_pathologies, "maxItems": n_pathologies}
        },
        "required": ["weights"],
        "additionalProperties": False,
    }


# Verbal bands mapped to log-likelihood-ratios. The mapping is the standard
# diagnostic-testing ladder (LR 10, 5, 2, 1, 0.5, 0.2, 0.1) so that the
# ordinal format is commensurable with the numeric one rather than being a
# separate scale.
ORDINAL_BANDS: dict[str, float] = {
    "strongly_argues_for": 2.302585,     # LR 10
    "moderately_argues_for": 1.609438,   # LR 5
    "weakly_argues_for": 0.693147,       # LR 2
    "uninformative": 0.0,                # LR 1
    "weakly_argues_against": -0.693147,  # LR 0.5
    "moderately_argues_against": -1.609438,
    "strongly_argues_against": -2.302585,
}


def top_diagnosis_schema(n_pathologies: int) -> dict:
    """Answer-level probe for ablation #15 (behaviour vs belief)."""
    return {
        "type": "object",
        "properties": {"index": {"type": "integer", "minimum": 0,
                                 "maximum": n_pathologies - 1}},
        "required": ["index"],
        "additionalProperties": False,
    }
