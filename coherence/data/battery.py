"""Construction of the CoDx coherence battery.

Every item is built once, versioned, and reused across all models so that
model comparisons are paired. Items carry their own ground truth where one
exists (A2, A3) and carry none where none is needed (A1, A4).

Design commitments that the analysis depends on:

* Cases are stratified by pathology and severity. The pathology prior spans
  252x (audit: `class_balance`), so an unstratified sample would report an
  order effect for the handful of common respiratory presentations.
* The pathology list presented to the model is in a FIXED order for every
  item in an A1 family. Only the evidence order varies. Output-position bias
  is therefore held constant and cannot masquerade as an order effect; it is
  measured separately by the `list_order` ablation.
* A1 families carry three arms: K permutations (the estimand), R resamples of
  one fixed permutation (the test-retest floor), and evidence drawn from
  unrelated cases (the shuffled ceiling).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from coherence.config import BUILD, SEED
from coherence.data.ddxplus import DDXPlusKB, load_all, load_kb
from coherence.data.empirical_oracle import MIN_SUPPORT, EmpiricalOracle

BATTERY_DIR = BUILD / "battery"
BATTERY_DIR.mkdir(parents=True, exist_ok=True)

Format = Literal["narrative", "bulleted"]


@dataclass
class Case:
    """One patient presentation, independent of any ordering."""

    case_id: str
    pathology: str
    severity: int
    age: int
    sex: str
    evidences: list[str]          # canonical order as released
    initial_evidence: str
    ddx_names: list[str]
    ddx_probs: list[float]
    split: str


@dataclass
class A1Item:
    """One ordering of one case, tagged with the arm it belongs to."""

    item_id: str
    case_id: str
    arm: Literal["permutation", "retest", "shuffled", "canonical"]
    perm_index: int
    evidence_order: list[str]
    replicate: int = 0


@dataclass
class A2Item:
    """A context, one added finding, and the true change in log-odds."""

    item_id: str
    context: list[str]
    new_finding: str
    support_before: int
    support_after: int
    true_delta_log_odds: list[float]
    true_posterior_before: list[float]
    true_posterior_after: list[float]
    se_before: list[float]
    se_after: list[float]


@dataclass
class A3Item:
    """A context plus a finding certified to leave the posterior unmoved."""

    item_id: str
    context: list[str]
    redundant_finding: str
    certified_jsd: float
    support_context: int
    support_with_finding: int
    true_posterior: list[float]


@dataclass
class A4Item:
    """One case with a decisive finding placed at a controlled position."""

    item_id: str
    case_id: str
    decisive_finding: str
    position: Literal["first", "middle", "last"]
    decisiveness: float          # |delta log-odds| the finding actually carries
    evidence_order: list[str]
    replicate: int = 0


@dataclass
class Battery:
    version: str
    pathologies: list[str]
    cases: list[Case]
    a1: list[A1Item]
    a2: list[A2Item]
    a3: list[A3Item]
    a4: list[A4Item]
    meta: dict = field(default_factory=dict)

    def save(self, path: Path | None = None) -> Path:
        path = path or (BATTERY_DIR / f"codx_battery_{self.version}.json")
        path.write_text(json.dumps({
            "version": self.version, "pathologies": self.pathologies, "meta": self.meta,
            "cases": [asdict(c) for c in self.cases],
            "a1": [asdict(x) for x in self.a1], "a2": [asdict(x) for x in self.a2],
            "a3": [asdict(x) for x in self.a3], "a4": [asdict(x) for x in self.a4],
        }))
        return path

    @classmethod
    def load(cls, path: Path) -> "Battery":
        d = json.loads(Path(path).read_text())
        return cls(
            version=d["version"], pathologies=d["pathologies"], meta=d.get("meta", {}),
            cases=[Case(**c) for c in d["cases"]],
            a1=[A1Item(**x) for x in d["a1"]], a2=[A2Item(**x) for x in d["a2"]],
            a3=[A3Item(**x) for x in d["a3"]], a4=[A4Item(**x) for x in d["a4"]],
        )


def _hid(*parts) -> str:
    return hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).hexdigest()


# --------------------------------------------------------------------------
# Case sampling
# --------------------------------------------------------------------------
def sample_cases(df: pl.DataFrame, kb: DDXPlusKB, n_cases: int = 2000,
                 min_findings: int = 6, max_findings: int = 40,
                 seed: int = SEED) -> list[Case]:
    """Stratified by pathology, then by severity, with a findings-count filter."""
    sev = kb.severity
    pool = (df.filter(pl.col("split") == "test")
            .with_columns(pl.col("evidences").list.len().alias("n_ev"))
            .filter((pl.col("n_ev") >= min_findings) & (pl.col("n_ev") <= max_findings)))
    per = max(1, n_cases // len(kb.pathologies))
    rng = np.random.default_rng(seed)
    out: list[Case] = []
    for p in kb.pathologies:
        sub = pool.filter(pl.col("pathology") == p)
        if not sub.height:
            continue
        take = min(per, sub.height)
        idx = rng.choice(sub.height, size=take, replace=False)
        for i in idx:
            r = sub.row(int(i), named=True)
            out.append(Case(
                case_id=_hid(p, r["age"], str(r["sex"]), "|".join(r["evidences"])),
                pathology=p, severity=sev[p], age=int(r["age"]), sex=str(r["sex"]),
                evidences=list(r["evidences"]), initial_evidence=r["initial_evidence"],
                ddx_names=list(r["ddx_names"]), ddx_probs=[float(x) for x in r["ddx_probs"]],
                split=r["split"]))
    rng.shuffle(out)
    return out[:n_cases]


# --------------------------------------------------------------------------
# A1: order invariance
# --------------------------------------------------------------------------
def build_a1(cases: list[Case], k_perms: int = 10, n_retest: int = 6,
             n_shuffled: int = 2, n_canonical: int = 2, seed: int = SEED) -> list[A1Item]:
    """Build the four A1 arms.

    A design correction the instrument calibration forced. DDXPlus releases
    each patient's findings in a *code-sorted* order, which groups related
    questions together (all the pain items, then the antecedents) and is
    therefore a systematically more coherent presentation than a random one.
    An earlier version used that canonical order as the anchor for the
    test-retest arm, and models turned out measurably more willing to commit
    to a belief under it: informative rate 67.5% against 60.2% for random
    orderings. That put the floor and the numerator under different
    conditions, so part of the measured order effect was canonical-vs-random
    rather than order-vs-order.

    So: the permutation arm is K *random* orderings, the retest arm resamples
    one of those same random orderings, and the released order is pulled out
    into its own `canonical` arm, where "does the released grouping help?" is
    asked directly instead of contaminating the headline estimand.
    """
    rng = np.random.default_rng(seed + 1)
    all_ev = [e for c in cases for e in c.evidences]
    items: list[A1Item] = []
    for c in cases:
        ev = list(c.evidences)
        seen: set[tuple] = {tuple(ev)}
        perms: list[list[str]] = []
        guard = 0
        while len(perms) < k_perms and guard < 200 * k_perms:
            guard += 1
            p = list(rng.permutation(ev))
            if tuple(p) not in seen:
                seen.add(tuple(p))
                perms.append(p)
        if not perms:                 # too few findings to permute at all
            perms = [list(ev)]
        for j, p in enumerate(perms):
            items.append(A1Item(_hid(c.case_id, "perm", j), c.case_id, "permutation", j, p))
        # Test-retest floor: the SAME random ordering, resampled.
        for r in range(n_retest):
            items.append(A1Item(_hid(c.case_id, "retest", r), c.case_id, "retest", 0,
                                list(perms[0]), replicate=r))
        # The released code-sorted order, as its own condition.
        for r in range(n_canonical):
            items.append(A1Item(_hid(c.case_id, "canon", r), c.case_id, "canonical", 0,
                                list(ev), replicate=r))
        # Diagnostic, not a ceiling: models answer nonsense with a default
        # posterior, so two different nonsense inputs agree MORE than two
        # resamples of a real case.
        for sdx in range(n_shuffled):
            fake = list(rng.choice(all_ev, size=len(ev), replace=False))
            items.append(A1Item(_hid(c.case_id, "shuf", sdx), c.case_id, "shuffled",
                                sdx, fake, replicate=sdx))
    return items


# --------------------------------------------------------------------------
# A2: update fidelity
# --------------------------------------------------------------------------
def build_a2(oracle: EmpiricalOracle, cases: list[Case], n_items: int = 3000,
             ctx_sizes: tuple[int, ...] = (1, 2, 3, 4), min_support: int = MIN_SUPPORT,
             seed: int = SEED) -> list[A2Item]:
    rng = np.random.default_rng(seed + 2)
    items: list[A2Item] = []
    tries = 0
    while len(items) < n_items and tries < n_items * 40:
        tries += 1
        c = cases[rng.integers(len(cases))]
        ev = [t for t in c.evidences if t in oracle.token_index]
        k = int(rng.choice(ctx_sizes))
        if len(ev) < k + 1:
            continue
        pick = list(rng.choice(ev, size=k + 1, replace=False))
        ctx, new = pick[:k], pick[k]
        delta, before, after = oracle.delta_log_odds(ctx, new)
        if before.support < min_support or after.support < min_support:
            continue
        items.append(A2Item(
            item_id=_hid("a2", "|".join(sorted(ctx)), new),
            context=ctx, new_finding=new,
            support_before=before.support, support_after=after.support,
            true_delta_log_odds=[float(x) for x in delta],
            true_posterior_before=[float(x) for x in before.probs],
            true_posterior_after=[float(x) for x in after.probs],
            se_before=[float(x) for x in before.se], se_after=[float(x) for x in after.se]))
    # Deduplicate on item_id; identical (context, finding) pairs add no information.
    seen, uniq = set(), []
    for it in items:
        if it.item_id not in seen:
            seen.add(it.item_id)
            uniq.append(it)
    return uniq


# --------------------------------------------------------------------------
# A3: redundancy insensitivity
# --------------------------------------------------------------------------
def build_a3(oracle: EmpiricalOracle, cases: list[Case], n_items: int = 1500,
             ctx_sizes: tuple[int, ...] = (2, 3, 4), max_jsd: float = 1e-3,
             min_support: int = MIN_SUPPORT, seed: int = SEED) -> list[A3Item]:
    rng = np.random.default_rng(seed + 3)
    items: list[A3Item] = []
    tries = 0
    while len(items) < n_items and tries < n_items * 20:
        tries += 1
        c = cases[rng.integers(len(cases))]
        ev = [t for t in c.evidences if t in oracle.token_index]
        k = int(rng.choice(ctx_sizes))
        if len(ev) < k:
            continue
        ctx = list(rng.choice(ev, size=k, replace=False))
        base = oracle.posterior(ctx)
        if base.support < min_support:
            continue
        cands = oracle.redundancy_candidates(ctx, min_support=min_support,
                                             max_jsd=max_jsd, limit=3)
        for tok, d, n in cands:
            items.append(A3Item(
                item_id=_hid("a3", "|".join(sorted(ctx)), tok),
                context=ctx, redundant_finding=tok, certified_jsd=d,
                support_context=base.support, support_with_finding=n,
                true_posterior=[float(x) for x in base.probs]))
    seen, uniq = set(), []
    for it in items:
        if it.item_id not in seen:
            seen.add(it.item_id)
            uniq.append(it)
    return uniq[:n_items]


# --------------------------------------------------------------------------
# A4: positional anchoring
# --------------------------------------------------------------------------
def build_a4(oracle: EmpiricalOracle, cases: list[Case], n_cases: int = 600,
             n_replicates: int = 3, seed: int = SEED) -> list[A4Item]:
    """Identify each case's most decisive finding, then vary only its position."""
    rng = np.random.default_rng(seed + 4)
    items: list[A4Item] = []
    chosen = cases[:n_cases]
    for c in chosen:
        ev = [t for t in c.evidences if t in oracle.token_index]
        if len(ev) < 5:
            continue
        # Decisiveness: how far the finding alone moves log-odds of the true
        # pathology away from the prior. Measured, not assumed.
        di = oracle.pathology_index.get(c.pathology)
        if di is None:
            continue
        prior_lo = oracle.prior.log_odds()[di]
        best, best_val = None, -np.inf
        for t in ev:
            p = oracle.posterior([t])
            if p.support < MIN_SUPPORT:
                continue
            v = abs(float(p.log_odds()[di] - prior_lo))
            if v > best_val:
                best, best_val = t, v
        if best is None:
            continue
        rest = [t for t in ev if t != best]
        for pos in ("first", "middle", "last"):
            for r in range(n_replicates):
                order = list(rng.permutation(rest))
                mid = len(order) // 2
                if pos == "first":
                    seq = [best] + order
                elif pos == "last":
                    seq = order + [best]
                else:
                    seq = order[:mid] + [best] + order[mid:]
                items.append(A4Item(
                    item_id=_hid("a4", c.case_id, pos, r), case_id=c.case_id,
                    decisive_finding=best, position=pos, decisiveness=float(best_val),
                    evidence_order=seq, replicate=r))
    return items


# --------------------------------------------------------------------------
def build_battery(n_cases: int = 2000, k_perms: int = 10, n_retest: int = 6,
                  n_a2: int = 3000, n_a3: int = 1500, n_a4_cases: int = 600,
                  version: str = "v1", seed: int = SEED) -> Battery:
    kb = load_kb()
    df = load_all()
    from coherence.data.kb_reconstruct import LikelihoodTable

    tbl = LikelihoodTable.load()
    oracle = EmpiricalOracle.build(df, tbl.tokens, tbl.pathologies)

    cases = sample_cases(df, kb, n_cases=n_cases, seed=seed)
    a1 = build_a1(cases, k_perms=k_perms, n_retest=n_retest, seed=seed)
    a2 = build_a2(oracle, cases, n_items=n_a2, seed=seed)
    a3 = build_a3(oracle, cases, n_items=n_a3, seed=seed)
    a4 = build_a4(oracle, cases, n_cases=n_a4_cases, seed=seed)
    b = Battery(version=version, pathologies=kb.pathologies, cases=cases,
                a1=a1, a2=a2, a3=a3, a4=a4,
                meta={"seed": seed, "k_perms": k_perms, "n_retest": n_retest,
                      "min_support": MIN_SUPPORT,
                      "n_cases": len(cases), "n_a1": len(a1), "n_a2": len(a2),
                      "n_a3": len(a3), "n_a4": len(a4)})
    return b


if __name__ == "__main__":
    b = build_battery()
    p = b.save()
    print(f"battery -> {p}")
    print(json.dumps(b.meta, indent=2))
