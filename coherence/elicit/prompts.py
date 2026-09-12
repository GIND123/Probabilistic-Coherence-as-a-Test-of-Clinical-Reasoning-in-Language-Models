"""Prompt construction.

Two invariants the experiment depends on:

1. The candidate pathology list is rendered in a FIXED order (the battery's
   `pathologies`, alphabetical) in every prompt. Only the evidence order
   varies between the members of an A1 family, so output-position bias is a
   constant and cannot be mistaken for an order effect.

2. Everything that does not depend on the case comes FIRST, so vLLM's prefix
   cache serves it once for the whole run rather than per item.
"""
from __future__ import annotations

from typing import Literal

from coherence.data.ddxplus import DDXPlusKB
from coherence.elicit import render

Format = Literal["narrative", "bulleted"]

SYSTEM = (
    "You are an experienced physician reasoning about a differential diagnosis. "
    "You reason from the findings you are given and you report calibrated "
    "probabilities. You never refuse to give a numeric answer."
)


def _candidate_block(pathologies: list[str]) -> str:
    lines = [f"{i}. {p}" for i, p in enumerate(pathologies)]
    return "\n".join(lines)


def render_findings(kb: DDXPlusKB, tokens: list[str], fmt: Format = "bulleted") -> str:
    """Render an ordered evidence list as clinical text.

    Both formats carry identical content in identical order; they differ only
    in presentation, which is what ablation #10 isolates.
    """
    if fmt == "bulleted":
        return render.bulleted(kb, tokens)
    return render.narrative(kb, tokens)


def demographics(age: int, sex: str) -> str:
    return f"A {age}-year-old {'man' if sex == 'M' else 'woman'}"


def posterior_prompt(kb: DDXPlusKB, pathologies: list[str], tokens: list[str],
                     age: int, sex: str, fmt: Format = "bulleted",
                     with_prior: bool = False,
                     prior: list[float] | None = None) -> list[dict]:
    """Elicit a full posterior over the fixed candidate list."""
    head = [
        "Below is a fixed list of candidate diagnoses. It is the same list in "
        "the same order for every case you will see.",
        "",
        _candidate_block(pathologies),
        "",
    ]
    if with_prior and prior is not None:
        # Ablation #9: base rates supplied vs withheld.
        head += [
            "Population base rates for these diagnoses, in the same order:",
            ", ".join(f"{p:.4f}" for p in prior),
            "",
        ]
    body = [
        f"Patient: {demographics(age, sex)} presenting with the following findings.",
        "",
        render_findings(kb, tokens, fmt),
        "",
        "Give the probability of each candidate diagnosis for this patient. "
        "Return a JSON object with a single key \"probabilities\" holding an array "
        f"of {len(pathologies)} numbers in the same order as the list above. "
        # Without the positivity clause, models collapse to an all-zero array
        # on 7-20% of items -- and at a rate that differs between evidence
        # orderings, which would have contaminated the order effect itself.
        # Instrument calibration (see reports/instrument_calibration.md) puts
        # schema validity at 100% in every arm with it.
        "Every diagnosis must receive a strictly positive probability: use a "
        "small value such as 0.001 for diagnoses you consider very unlikely, "
        "never 0. The numbers must sum to 1.",
    ]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(head + body)}]


def cot_posterior_prompt(kb: DDXPlusKB, pathologies: list[str], tokens: list[str],
                         age: int, sex: str, fmt: Format = "bulleted") -> list[dict]:
    """Free-text reasoning first, posterior second (the CoT baseline)."""
    msgs = posterior_prompt(kb, pathologies, tokens, age, sex, fmt)
    msgs[1]["content"] = msgs[1]["content"].replace(
        "Give the probability of each candidate diagnosis",
        "Think step by step about which diagnoses the findings support and which "
        "they argue against, then give the probability of each candidate diagnosis")
    return msgs


def likelihood_ratio_prompt(kb: DDXPlusKB, pathologies: list[str], finding: str,
                            age: int, sex: str, fmt: str = "numeric") -> list[dict]:
    """Elicit one finding's evidence weight against every candidate.

    Deliberately context-free: the finding is presented alone, with no other
    findings and no accumulated narrative. That is what makes the elicited
    quantity a per-finding likelihood ratio rather than a posterior, and it is
    why ELR-Fusion is order-invariant by construction -- there is no order for
    it to depend on.
    """
    desc = kb.describe(finding)
    head = [
        "Below is a fixed list of candidate diagnoses. It is the same list in "
        "the same order for every question you will see.",
        "",
        _candidate_block(pathologies),
        "",
    ]
    if fmt == "numeric":
        ask = (
            "For each candidate diagnosis, give the likelihood ratio for this single "
            "finding: the probability of observing the finding in a patient who has "
            "that diagnosis, divided by the probability of observing it in a patient "
            "who does not have that diagnosis. A value above 1 means the finding "
            "argues for the diagnosis; below 1, against it; exactly 1, uninformative. "
            f"Return JSON with a single key \"weights\" holding {len(pathologies)} "
            "numbers in the list order above.")
    elif fmt == "logodds":
        ask = (
            "For each candidate diagnosis, give the natural logarithm of the "
            "likelihood ratio for this single finding. Positive means the finding "
            "argues for the diagnosis, negative against, zero uninformative. "
            f"Return JSON with a single key \"weights\" holding {len(pathologies)} "
            "numbers in the list order above.")
    else:
        ask = (
            "For each candidate diagnosis, say how strongly this single finding "
            "argues for or against it, using exactly one of these labels: "
            "strongly_argues_for, moderately_argues_for, weakly_argues_for, "
            "uninformative, weakly_argues_against, moderately_argues_against, "
            "strongly_argues_against. "
            f"Return JSON with a single key \"weights\" holding {len(pathologies)} "
            "labels in the list order above.")
    body = [
        f"Consider {demographics(age, sex).lower()}. Consider ONLY this single finding, "
        "in isolation from any other information:",
        "",
        f"  {desc}",
        "",
        ask,
    ]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(head + body)}]


def top_diagnosis_prompt(kb: DDXPlusKB, pathologies: list[str], tokens: list[str],
                         age: int, sex: str, fmt: Format = "bulleted") -> list[dict]:
    """Answer-level elicitation, for the behaviour-vs-belief dissociation."""
    head = ["Candidate diagnoses:", "", _candidate_block(pathologies), ""]
    body = [
        f"Patient: {demographics(age, sex)} presenting with:",
        "",
        render_findings(kb, tokens, fmt),
        "",
        "Give the index of the single most likely diagnosis. Return JSON with one "
        "key \"index\".",
    ]
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(head + body)}]
