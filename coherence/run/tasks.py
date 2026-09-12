"""Turn the battery into concrete elicitation jobs.

A Job is one prompt plus everything needed to interpret its output. Jobs are
grouped into tasks; a task is the unit of checkpointing, so a run can be
interrupted and resumed at task granularity without repeating work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from coherence.data.battery import Battery
from coherence.data.ddxplus import DDXPlusKB
from coherence.elicit import prompts as P
from coherence.elicit.schema import (likelihood_ratio_schema, posterior_schema,
                                     top_diagnosis_schema)


@dataclass
class Job:
    task: str
    item_id: str
    messages: list[dict]
    schema: dict
    kind: str                      # "posterior" | "weights" | "index"
    meta: dict = field(default_factory=dict)


def _case_index(b: Battery) -> dict:
    return {c.case_id: c for c in b.cases}


# --------------------------------------------------------------------------
def a1_jobs(b: Battery, kb: DDXPlusKB, fmt: str = "bulleted") -> Iterator[Job]:
    """Order invariance: every permutation, retest and shuffled control."""
    ci = _case_index(b)
    sch = posterior_schema(len(b.pathologies))
    for it in b.a1:
        c = ci[it.case_id]
        yield Job(
            task="a1_posterior", item_id=it.item_id,
            messages=P.posterior_prompt(kb, b.pathologies, it.evidence_order,
                                        c.age, c.sex, fmt),
            schema=sch, kind="posterior",
            meta={"case_id": it.case_id, "arm": it.arm, "perm_index": it.perm_index,
                  "replicate": it.replicate, "format": fmt,
                  "n_findings": len(it.evidence_order), "pathology": c.pathology,
                  "severity": c.severity})


def a1_answer_jobs(b: Battery, kb: DDXPlusKB, fmt: str = "bulleted") -> Iterator[Job]:
    """Answer-level probe for the behaviour-vs-belief dissociation (#15)."""
    ci = _case_index(b)
    sch = top_diagnosis_schema(len(b.pathologies))
    for it in b.a1:
        if it.arm == "shuffled":
            continue
        c = ci[it.case_id]
        yield Job(
            task="a1_answer", item_id=it.item_id,
            messages=P.top_diagnosis_prompt(kb, b.pathologies, it.evidence_order,
                                            c.age, c.sex, fmt),
            schema=sch, kind="index",
            meta={"case_id": it.case_id, "arm": it.arm, "perm_index": it.perm_index,
                  "replicate": it.replicate, "format": fmt})


def a2_jobs(b: Battery, kb: DDXPlusKB, age: int = 45, sex: str = "F",
            fmt: str = "bulleted") -> Iterator[Job]:
    """Update fidelity: posterior before and after one added finding.

    Demographics are held FIXED across every A2 item, because the oracle
    target is a corpus-wide conditional that does not condition on age or sex.
    Letting demographics vary would compare the model against a target that
    ignores information the model was given.
    """
    sch = posterior_schema(len(b.pathologies))
    seen: set[tuple] = set()
    for it in b.a2:
        for phase, toks in (("before", it.context),
                            ("after", list(it.context) + [it.new_finding])):
            key = tuple(sorted(toks))
            uid = f"{it.item_id}:{phase}"
            yield Job(
                task="a2_posterior", item_id=uid,
                messages=P.posterior_prompt(kb, b.pathologies, list(toks), age, sex, fmt),
                schema=sch, kind="posterior",
                meta={"a2_id": it.item_id, "phase": phase, "format": fmt,
                      "context_size": len(it.context), "new_finding": it.new_finding,
                      "dedup_key": "|".join(sorted(toks))})
            seen.add(key)


def a3_jobs(b: Battery, kb: DDXPlusKB, age: int = 45, sex: str = "F",
            fmt: str = "bulleted") -> Iterator[Job]:
    """Redundancy insensitivity: posterior with and without a null finding."""
    sch = posterior_schema(len(b.pathologies))
    for it in b.a3:
        for phase, toks in (("without", it.context),
                            ("with", list(it.context) + [it.redundant_finding])):
            yield Job(
                task="a3_posterior", item_id=f"{it.item_id}:{phase}",
                messages=P.posterior_prompt(kb, b.pathologies, list(toks), age, sex, fmt),
                schema=sch, kind="posterior",
                meta={"a3_id": it.item_id, "phase": phase, "format": fmt,
                      "certified_jsd": it.certified_jsd,
                      "redundant_finding": it.redundant_finding})


def a4_jobs(b: Battery, kb: DDXPlusKB, fmt: str = "bulleted") -> Iterator[Job]:
    """Positional anchoring: decisive finding first, middle or last."""
    ci = _case_index(b)
    sch = posterior_schema(len(b.pathologies))
    for it in b.a4:
        c = ci[it.case_id]
        yield Job(
            task="a4_posterior", item_id=it.item_id,
            messages=P.posterior_prompt(kb, b.pathologies, it.evidence_order,
                                        c.age, c.sex, fmt),
            schema=sch, kind="posterior",
            meta={"case_id": it.case_id, "position": it.position,
                  "decisive_finding": it.decisive_finding,
                  "decisiveness": it.decisiveness, "replicate": it.replicate,
                  "format": fmt, "pathology": c.pathology, "severity": c.severity})


def elr_weight_jobs(b: Battery, kb: DDXPlusKB, formats: tuple[str, ...] =
                    ("numeric", "logodds", "ordinal"),
                    age: int = 45, sex: str = "F") -> Iterator[Job]:
    """Per-finding evidence weights: the ELR-Fusion elicitation.

    Deliberately context-free and therefore shared across every case that
    contains the finding. This is what makes ELR-Fusion cheap: the weight
    table is amortised over the corpus, so a new case costs n table lookups
    and zero new queries. Both the amortised and the unamortised cost are
    reported in the cost table.
    """
    tokens = sorted({t for c in b.cases for t in c.evidences})
    for fmt in formats:
        sch = likelihood_ratio_schema(len(b.pathologies), fmt)
        for t in tokens:
            yield Job(
                task=f"elr_weights_{fmt}", item_id=f"{fmt}:{t}",
                messages=P.likelihood_ratio_prompt(kb, b.pathologies, t, age, sex, fmt),
                schema=sch, kind="weights",
                meta={"finding": t, "lr_format": fmt})


def elr_prior_jobs(b: Battery, kb: DDXPlusKB) -> Iterator[Job]:
    """Demographic prior for ELR-Fusion: one query per (age band, sex)."""
    sch = posterior_schema(len(b.pathologies))
    bands = [(0, 4), (5, 14), (15, 29), (30, 44), (45, 59), (60, 74), (75, 120)]
    for lo, hi in bands:
        for sex in ("M", "F"):
            mid = (lo + hi) // 2
            yield Job(
                task="elr_prior", item_id=f"{lo}-{hi}:{sex}",
                messages=P.posterior_prompt(kb, b.pathologies, [], mid, sex, "bulleted"),
                schema=sch, kind="posterior",
                meta={"age_lo": lo, "age_hi": hi, "sex": sex})


def cot_jobs(b: Battery, kb: DDXPlusKB, n_perms: int = 3,
             fmt: str = "bulleted") -> Iterator[Job]:
    """Chain-of-thought baseline on a subset of A1 permutations."""
    ci = _case_index(b)
    sch = posterior_schema(len(b.pathologies))
    for it in b.a1:
        if it.arm != "permutation" or it.perm_index >= n_perms:
            continue
        c = ci[it.case_id]
        yield Job(
            task="cot_posterior", item_id=it.item_id,
            messages=P.cot_posterior_prompt(kb, b.pathologies, it.evidence_order,
                                            c.age, c.sex, fmt),
            schema=sch, kind="posterior",
            meta={"case_id": it.case_id, "perm_index": it.perm_index, "format": fmt})


def narrative_jobs(b: Battery, kb: DDXPlusKB, n_cases: int = 400,
                   n_perms: int = 5) -> Iterator[Job]:
    """Ablation #10: the A1 core repeated in narrative prose."""
    ci = _case_index(b)
    keep = {c.case_id for c in b.cases[:n_cases]}
    sch = posterior_schema(len(b.pathologies))
    for it in b.a1:
        if it.case_id not in keep or it.arm not in ("permutation", "retest"):
            continue
        if it.arm == "permutation" and it.perm_index >= n_perms:
            continue
        c = ci[it.case_id]
        yield Job(
            task="a1_narrative", item_id=it.item_id,
            messages=P.posterior_prompt(kb, b.pathologies, it.evidence_order,
                                        c.age, c.sex, "narrative"),
            schema=sch, kind="posterior",
            meta={"case_id": it.case_id, "arm": it.arm, "perm_index": it.perm_index,
                  "replicate": it.replicate, "format": "narrative"})


def prior_supplied_jobs(b: Battery, kb: DDXPlusKB, prior: list[float],
                        n_cases: int = 400, n_perms: int = 5) -> Iterator[Job]:
    """Ablation #9: base rates supplied instead of withheld."""
    ci = _case_index(b)
    keep = {c.case_id for c in b.cases[:n_cases]}
    sch = posterior_schema(len(b.pathologies))
    for it in b.a1:
        if it.case_id not in keep or it.arm != "permutation" or it.perm_index >= n_perms:
            continue
        c = ci[it.case_id]
        yield Job(
            task="a1_prior_supplied", item_id=it.item_id,
            messages=P.posterior_prompt(kb, b.pathologies, it.evidence_order,
                                        c.age, c.sex, "bulleted",
                                        with_prior=True, prior=prior),
            schema=sch, kind="posterior",
            meta={"case_id": it.case_id, "arm": it.arm, "perm_index": it.perm_index,
                  "format": "bulleted", "prior_supplied": True})


TASK_BUILDERS = {
    "a1_posterior": a1_jobs,
    "a1_answer": a1_answer_jobs,
    "a2_posterior": a2_jobs,
    "a3_posterior": a3_jobs,
    "a4_posterior": a4_jobs,
    "elr_prior": elr_prior_jobs,
    "cot_posterior": cot_jobs,
    "a1_narrative": narrative_jobs,
}

# Run order: cheapest and most load-bearing first, so an interrupted run
# still yields the headline A1 result.
CORE_TASKS = ["a1_posterior", "a4_posterior", "a2_posterior", "a3_posterior"]
METHOD_TASKS = ["elr_prior", "elr_weights_numeric", "elr_weights_logodds",
                "elr_weights_ordinal"]
ABLATION_TASKS = ["a1_answer", "cot_posterior", "a1_narrative"]
