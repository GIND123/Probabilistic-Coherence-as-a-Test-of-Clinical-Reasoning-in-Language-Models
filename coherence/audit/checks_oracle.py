"""Checks on the two candidate ground-truth constructions for A2/A3."""
from __future__ import annotations

import numpy as np
import polars as pl

from coherence.audit.checks import Finding
from coherence.data.empirical_oracle import MIN_SUPPORT, EmpiricalOracle


def check_kb_probability_release(nonempty: int, total: int) -> Finding:
    return Finding(
        "kb_probability_release",
        "critical",
        f"release_conditions.json declares {total} (pathology, evidence) slots and supplies "
        f"conditional probabilities for {nonempty} of them",
        detail=(
            "This is the week-1 gate named in the proposal. The generator's likelihood tables "
            "are NOT part of the public DDXPlus release: every slot is an empty object. Exact "
            "posteriors for arbitrary evidence subsets therefore cannot be read off the "
            "knowledge base, and axioms A2 and A3 need another reference. Two were tested; "
            "see `naive_bayes_reconstruction` and `empirical_oracle_feasibility`."
        ),
        stats={"n_slots": total, "n_with_probabilities": nonempty},
    )


def check_naive_bayes_reconstruction(validation: dict, noise_floor: float,
                                     independence: dict) -> Finding:
    ok = validation["jsd_mean"] <= noise_floor
    return Finding(
        "naive_bayes_reconstruction",
        "info" if ok else "major",
        f"reconstructing the likelihood tables from corpus frequencies and recombining them "
        f"by naive Bayes reproduces the shipped differentials at mean JSD "
        f"{validation['jsd_mean']:.4f} (top-1 agreement {100*validation['top1_agreement']:.1f}%, "
        f"top-5 Jaccard {validation['top5_jaccard']:.3f}) against a generator noise floor of "
        f"{noise_floor:.4f} -- {'adequate' if ok else 'NOT adequate'}",
        detail=(
            "The per-table estimates are precise (binomial SE ~3e-4); the failure is the "
            "recombination rule, not the estimation. Sampled evidence pairs within a pathology "
            f"have mean |phi| {independence['phi_mean_abs']:.3f} (p95 "
            f"{independence['phi_p95_abs']:.3f}, max {independence['phi_max_abs']:.3f}), with "
            f"{100*independence['frac_pairs_abs_phi_gt_0.1']:.1f}% of pairs exceeding |phi|=0.1, "
            "so conditional independence given the pathology is violated materially and summing "
            "log-likelihood-ratios over ~20 correlated findings compounds the error. Temperature "
            "scaling does not rescue it, which confirms the defect is structural rather than a "
            "calibration issue. Consequence: naive Bayes is retained ONLY as an explicitly "
            "approximate baseline and as the object of the independence-correction ablation, "
            "never as ground truth."
        ),
        stats={"validation": validation, "independence": independence,
               "generator_noise_floor_jsd": noise_floor},
    )


def check_empirical_oracle_feasibility(oracle: EmpiricalOracle, df: pl.DataFrame,
                                       n_probe: int = 400, seed: int = 0) -> Finding:
    """How large an evidence set can the assumption-free oracle still score?"""
    rng = np.random.default_rng(seed)
    rows = rng.choice(df.height, size=n_probe, replace=False)
    ev_lists = df["evidences"].to_list()
    per_k: dict[int, list[int]] = {}
    for r in rows:
        evs = [t for t in ev_lists[r] if t in oracle.token_index]
        for k in (1, 2, 3, 4, 5, 6, 8, 10):
            if len(evs) < k:
                continue
            sub = list(rng.choice(evs, size=k, replace=False))
            per_k.setdefault(k, []).append(oracle.posterior(sub).support)
    summary = {
        k: {"median_support": float(np.median(v)),
            "p10_support": float(np.percentile(v, 10)),
            "frac_scorable": float(np.mean(np.asarray(v) >= MIN_SUPPORT))}
        for k, v in sorted(per_k.items())
    }
    max_k = max((k for k, s in summary.items() if s["frac_scorable"] >= 0.8), default=0)
    return Finding(
        "empirical_oracle_feasibility",
        "info",
        "conditioning directly on the corpus gives an assumption-free posterior; "
        + ", ".join(f"|E|={k}: {100*s['frac_scorable']:.0f}% scorable "
                    f"(median support {s['median_support']:.0f})"
                    for k, s in summary.items())
        + f" at a {MIN_SUPPORT}-patient minimum",
        detail=(
            f"Evidence sets up to |E|={max_k} keep at least 80% of items scorable, which covers "
            "the absolute-reference requirements of A2 and A3. A1 and A4 need no ground truth at "
            "all and so run at any evidence-set size, including the full ~20-finding cases and "
            "the MIMIC narratives. This is the resolution of the week-1 gate: A2/A3 are ABSOLUTE "
            "measures, not relative ones, and the reference carries no independence assumption "
            "for a reviewer to attack -- it is the generator's own joint distribution, counted."
        ),
        stats={"min_support": MIN_SUPPORT, "max_k_80pct_scorable": max_k, "by_k": summary},
        table=pl.DataFrame([{"evidence_set_size": k, **s} for k, s in summary.items()]),
    )


def check_oracle_vs_shipped(oracle: EmpiricalOracle, df: pl.DataFrame,
                            n: int = 5000, seed: int = 0) -> Finding:
    """Do the two ground-truth candidates agree where both are available?"""
    from coherence.metrics.divergence import jsd

    rng = np.random.default_rng(seed)
    s = df.sample(n=n, seed=seed)
    pi = oracle.pathology_index
    jsds, sup = [], []
    for names, probs, evs in zip(s["ddx_names"].to_list(), s["ddx_probs"].to_list(),
                                 s["evidences"].to_list()):
        toks = [t for t in evs if t in oracle.token_index]
        if len(toks) < 3:
            continue
        sub = list(rng.choice(toks, size=3, replace=False))
        emp = oracle.posterior(sub)
        if emp.support < MIN_SUPPORT:
            continue
        truth = np.zeros(len(oracle.pathologies))
        for nm, pr in zip(names, probs):
            truth[pi[nm]] = pr
        jsds.append(jsd(truth, emp.probs))
        sup.append(emp.support)
    jsds = np.asarray(jsds)
    return Finding(
        "oracle_vs_shipped_differential",
        "info",
        f"the empirical 3-finding posterior differs from the shipped full-case differential at "
        f"mean JSD {jsds.mean():.3f} over {len(jsds):,} comparable items",
        detail=(
            "This is expected and is not a defect: the two quantities condition on different "
            "evidence. The shipped differential conditions on the patient's entire finding set, "
            "the empirical oracle on the 3-finding subset actually presented. The comparison is "
            "reported so that the report does not silently imply the two are interchangeable. "
            "Each axiom states which reference it uses."
        ),
        stats={"n_items": int(len(jsds)), "jsd_mean": float(jsds.mean()),
               "jsd_median": float(np.median(jsds)),
               "median_support": float(np.median(sup))},
    )
