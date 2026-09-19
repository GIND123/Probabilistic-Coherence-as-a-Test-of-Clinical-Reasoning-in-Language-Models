"""MIMIC-IV-Note arm: A1 and A4 on real clinical narratives.

Why this arm exists
-------------------
The obvious attack on any DDXPlus result is "it is synthetic". A1 and A4
require no ground truth -- they compare a model against itself -- so they run
unchanged on real de-identified discharge summaries. That is the whole point
of designing the primary metric to be label-free.

How the metric stays identical
------------------------------
MIMIC has no fixed pathology list, so the candidate set is elicited ONCE per
case from the canonical ordering and then held fixed across every permutation
of that case. Only the order of the narrative varies. No annotator, no LLM
judge, no string matching between free-text diagnoses -- the same clean
divergence metric as the DDXPlus arm.

Governance
----------
MIMIC-IV-Note is credentialed. Nothing derived from patient text is uploaded:
`coherence.hub.hf_sync` refuses any path containing "mimic" or "physionet",
and only aggregate divergence statistics leave this module. Frontier API
models are never pointed at this arm; it runs on locally-served open weights
only.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from coherence.config import BUILD, MIMIC_NOTE, SEED

csv.field_size_limit(min(sys.maxsize, 10 ** 9))

OUT = BUILD / "mimic"
OUT.mkdir(parents=True, exist_ok=True)

SECTION = re.compile(r"^([A-Z][A-Za-z /\-]{3,40}):", re.M)
DEID = re.compile(r"_{2,}")
# Sentence split that keeps clinical abbreviations ("q.d.", "Dr.") intact
# well enough for segmentation; findings are clauses, not a parse.
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass
class MimicCase:
    case_id: str
    note_id: str
    chief_complaint: str
    findings: list[str]
    n_deid: int
    n_chars: int


def _sections(text: str) -> dict[str, str]:
    out, last, pos = {}, None, 0
    for m in SECTION.finditer(text):
        if last is not None:
            out[last] = text[pos:m.start()].strip()
        last, pos = m.group(1).strip(), m.end()
    if last is not None:
        out[last] = text[pos:].strip()
    return out


def _segment(hpi: str) -> list[str]:
    body = re.sub(r"\s*\n\s*", " ", hpi).strip()
    parts = [p.strip() for p in SENT.split(body)]
    return [p for p in parts if 25 <= len(p) <= 400]


def extract_cases(limit: int = 4000, min_findings: int = 6, max_findings: int = 20,
                  max_deid_ratio: float = 0.02, seed: int = SEED,
                  scan_limit: int = 60000) -> list[MimicCase]:
    """Pull structured cases out of the discharge summaries.

    Heavily de-identified notes are dropped: a narrative that is mostly `___`
    placeholders gives the model little to be coherent about, and its order
    effect would be uninterpretable rather than interestingly small.
    """
    src = MIMIC_NOTE / "discharge.csv.gz"
    cases: list[MimicCase] = []
    with gzip.open(src, "rt") as fh:
        r = csv.DictReader(fh)
        for i, row in enumerate(r):
            if i >= scan_limit or len(cases) >= limit:
                break
            text = row["text"]
            secs = _sections(text)
            hpi = secs.get("History of Present Illness", "")
            cc = secs.get("Chief Complaint", "").strip()
            if not hpi or len(hpi) < 400:
                continue
            n_deid = len(DEID.findall(hpi))
            if n_deid / max(len(hpi.split()), 1) > max_deid_ratio:
                continue
            findings = _segment(hpi)
            if not (min_findings <= len(findings) <= max_findings):
                continue
            cases.append(MimicCase(
                case_id=hashlib.blake2b(row["note_id"].encode(), digest_size=8).hexdigest(),
                note_id=row["note_id"],
                chief_complaint=re.sub(r"\s+", " ", cc)[:200],
                findings=findings, n_deid=n_deid, n_chars=len(hpi)))
    return cases


@dataclass
class MimicBattery:
    cases: list[MimicCase]
    a1: list[dict]
    meta: dict = field(default_factory=dict)

    def save(self, path: Path | None = None) -> Path:
        path = path or (OUT / "mimic_battery_v1.json")
        path.write_text(json.dumps({
            "cases": [asdict(c) for c in self.cases], "a1": self.a1, "meta": self.meta}))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "MimicBattery":
        path = path or (OUT / "mimic_battery_v1.json")
        d = json.loads(path.read_text())
        return cls(cases=[MimicCase(**c) for c in d["cases"]], a1=d["a1"],
                   meta=d.get("meta", {}))


def build_battery(n_cases: int = 800, k_perms: int = 8, n_retest: int = 4,
                  n_shuffled: int = 2, seed: int = SEED) -> MimicBattery:
    cases = extract_cases(limit=n_cases, seed=seed)
    rng = random.Random(seed)
    pool = [f for c in cases for f in c.findings]
    items = []
    for c in cases:
        seen, perms = set(), [list(c.findings)]
        seen.add(tuple(c.findings))
        guard = 0
        while len(perms) < k_perms and guard < 200 * k_perms:
            guard += 1
            p = list(c.findings)
            rng.shuffle(p)
            if tuple(p) not in seen:
                seen.add(tuple(p))
                perms.append(p)
        for j, p in enumerate(perms):
            items.append({"case_id": c.case_id, "arm": "permutation", "perm_index": j,
                          "replicate": 0, "order": p})
        for r in range(n_retest):
            items.append({"case_id": c.case_id, "arm": "retest", "perm_index": 0,
                          "replicate": r, "order": list(perms[0])})
        for s in range(n_shuffled):
            items.append({"case_id": c.case_id, "arm": "shuffled", "perm_index": s,
                          "replicate": s,
                          "order": rng.sample(pool, k=len(c.findings))})
    return MimicBattery(
        cases=cases, a1=items,
        meta={"seed": seed, "k_perms": k_perms, "n_retest": n_retest,
              "n_cases": len(cases), "n_items": len(items),
              "source": "MIMIC-IV-Note 2.2 discharge summaries, HPI section",
              "governance": "credentialed; derived statistics only, never uploaded"})


if __name__ == "__main__":
    b = build_battery()
    p = b.save()
    print(json.dumps(b.meta, indent=2))
    print(f"-> {p}")
    c = b.cases[0]
    print(f"\nexample case: {len(c.findings)} findings, cc={c.chief_complaint!r}")
