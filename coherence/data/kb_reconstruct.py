"""Reconstruct DDXPlus's likelihood model from the released corpus.

The public `release_conditions.json` declares *which* evidences each
pathology may present but not *with what probability*: all 888 evidence
slots are empty. Exact posteriors for arbitrary evidence subsets -- which
axioms A2 (update fidelity) and A3 (redundancy insensitivity) need -- are
therefore not readable from the release.

They are, however, estimable. The corpus is 1.29M patients drawn from the
generator, ~26k per pathology, so the empirical conditional frequency of
each evidence token given each pathology is a Monte-Carlo estimate of the
generator's own table with a standard error near 3e-3. This module builds
that table, quantifies its uncertainty, tests the conditional-independence
assumption the estimate would be combined under, and -- critically --
validates the reconstruction by checking whether it reproduces the
differentials the generator actually shipped.

Everything downstream states which regime it is operating in:

  ABSOLUTE  reconstruction reproduces shipped differentials within the
            generator's own replicate noise -> A2/A3 are absolute measures
  RELATIVE  it does not -> A2/A3 become relative measures and A1/A4,
            which need no ground truth at all, carry the paper
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from coherence.config import KB as KB_DIR
from coherence.data.ddxplus import DDXPlusKB, load_kb

LAPLACE = 0.5  # Jeffreys prior; keeps log-odds finite for unobserved tokens


@dataclass
class LikelihoodTable:
    """P(evidence token | pathology), plus the marginals needed for Bayes."""

    pathologies: list[str]
    tokens: list[str]
    log_prior: np.ndarray          # (D,)
    p_e_given_d: np.ndarray        # (D, T)
    counts: np.ndarray             # (D, T) raw co-occurrence counts
    n_per_pathology: np.ndarray    # (D,)

    @property
    def token_index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.tokens)}

    @property
    def pathology_index(self) -> dict[str, int]:
        return {p: i for i, p in enumerate(self.pathologies)}

    def standard_error(self) -> np.ndarray:
        """Binomial SE of each P(e|d) estimate."""
        p = self.p_e_given_d
        return np.sqrt(p * (1 - p) / self.n_per_pathology[:, None])

    def log_lr(self) -> np.ndarray:
        """log P(e|d) - log P(e|not d), the per-finding evidence weight.

        This is the quantity ELR-Fusion elicits from a model; the table
        version is the oracle it is scored against.
        """
        p = np.clip(self.p_e_given_d, 1e-6, 1 - 1e-6)
        prior = np.exp(self.log_prior)[:, None]
        p_e = (p * prior).sum(axis=0, keepdims=True)        # (1, T)
        p_e_not_d = np.clip((p_e - p * prior) / np.clip(1 - prior, 1e-12, None),
                            1e-6, 1 - 1e-6)
        return np.log(p) - np.log(p_e_not_d)

    def posterior(self, tokens: list[str], present_only: bool = True) -> np.ndarray:
        """Naive-Bayes posterior over pathologies for an evidence subset.

        `present_only=True` scores only the asserted findings (the regime the
        shipped differentials appear to use); `False` additionally scores the
        absence of every unasserted token.
        """
        ti = self.token_index
        idx = [ti[t] for t in tokens if t in ti]
        lp = self.log_prior.copy()
        p = np.clip(self.p_e_given_d, 1e-6, 1 - 1e-6)
        if idx:
            lp = lp + np.log(p[:, idx]).sum(axis=1)
        if not present_only:
            mask = np.ones(len(self.tokens), dtype=bool)
            mask[idx] = False
            lp = lp + np.log(1 - p[:, mask]).sum(axis=1)
        lp -= lp.max()
        w = np.exp(lp)
        return w / w.sum()

    def save(self, path: Path | None = None) -> Path:
        path = path or (KB_DIR / "likelihood_table.npz")
        np.savez_compressed(
            path,
            pathologies=np.array(self.pathologies, dtype=object),
            tokens=np.array(self.tokens, dtype=object),
            log_prior=self.log_prior,
            p_e_given_d=self.p_e_given_d,
            counts=self.counts,
            n_per_pathology=self.n_per_pathology,
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "LikelihoodTable":
        path = path or (KB_DIR / "likelihood_table.npz")
        z = np.load(path, allow_pickle=True)
        return cls(
            pathologies=list(z["pathologies"]),
            tokens=list(z["tokens"]),
            log_prior=z["log_prior"],
            p_e_given_d=z["p_e_given_d"],
            counts=z["counts"],
            n_per_pathology=z["n_per_pathology"],
        )


def build_likelihood_table(df: pl.DataFrame, kb: DDXPlusKB | None = None) -> LikelihoodTable:
    """Estimate P(token | pathology) and P(pathology) from the corpus."""
    kb = kb or load_kb()
    pathologies = kb.pathologies
    pi = {p: i for i, p in enumerate(pathologies)}

    tokens = sorted(
        df.select(pl.col("evidences").explode().alias("t"))["t"].unique().to_list()
    )
    ti = {t: i for i, t in enumerate(tokens)}

    D, T = len(pathologies), len(tokens)
    counts = np.zeros((D, T), dtype=np.float64)
    n_per = np.zeros(D, dtype=np.float64)

    for path_name, evs in zip(df["pathology"].to_list(), df["evidences"].to_list()):
        d = pi[path_name]
        n_per[d] += 1
        for t in evs:
            counts[d, ti[t]] += 1

    p = (counts + LAPLACE) / (n_per[:, None] + 2 * LAPLACE)
    log_prior = np.log(n_per / n_per.sum())
    return LikelihoodTable(pathologies, tokens, log_prior, p, counts, n_per)


def validate_against_shipped(
    table: LikelihoodTable,
    df: pl.DataFrame,
    n_cases: int = 20000,
    seed: int = 0,
    present_only: bool = True,
) -> dict:
    """Does the reconstruction reproduce the generator's own differentials?"""
    from coherence.metrics.divergence import jsd, topk_jaccard

    rng = np.random.default_rng(seed)
    n = min(n_cases, df.height)
    sample = df.sample(n=n, seed=seed) if n < df.height else df
    pi = table.pathology_index

    jsds, top1, top5 = [], [], []
    for names, probs, evs in zip(sample["ddx_names"].to_list(),
                                 sample["ddx_probs"].to_list(),
                                 sample["evidences"].to_list()):
        truth = np.zeros(len(table.pathologies))
        for nm, pr in zip(names, probs):
            truth[pi[nm]] = pr
        pred = table.posterior(list(evs), present_only=present_only)
        jsds.append(jsd(truth, pred))
        top1.append(int(truth.argmax() == pred.argmax()))
        top5.append(topk_jaccard(truth, pred, 5))

    jsds = np.asarray(jsds)
    return {
        "n_cases": n,
        "present_only": present_only,
        "jsd_mean": float(jsds.mean()),
        "jsd_median": float(np.median(jsds)),
        "jsd_p90": float(np.percentile(jsds, 90)),
        "top1_agreement": float(np.mean(top1)),
        "top5_jaccard": float(np.mean(top5)),
    }


def test_conditional_independence(
    df: pl.DataFrame, table: LikelihoodTable, n_pairs: int = 4000, seed: int = 0,
    min_count: int = 200,
) -> dict:
    """How wrong is 'evidences are independent given the pathology'?

    ELR-Fusion sums log-likelihood-ratios, which is exact only under
    conditional independence. The proposal promises this error is quantified
    rather than assumed. This measures it directly: for sampled token pairs
    within each pathology, compare the observed joint against the product of
    marginals, and report the induced error in summed log-evidence.
    """
    rng = np.random.default_rng(seed)
    ti = table.token_index
    pi = table.pathology_index
    T = len(table.tokens)

    by_path: dict[int, list[list[int]]] = {}
    for path_name, evs in zip(df["pathology"].to_list(), df["evidences"].to_list()):
        by_path.setdefault(pi[path_name], []).append([ti[t] for t in evs if t in ti])

    phis, kl_terms, n_tested = [], [], 0
    for d, rows in by_path.items():
        n_d = len(rows)
        if n_d < min_count:
            continue
        marg = table.counts[d]
        cand = np.flatnonzero((marg >= min_count) & (marg <= n_d - min_count))
        if len(cand) < 2:
            continue
        present = np.zeros((n_d, len(cand)), dtype=bool)
        pos = {c: j for j, c in enumerate(cand)}
        for i, r in enumerate(rows):
            for t in r:
                j = pos.get(t)
                if j is not None:
                    present[i, j] = True
        k = min(n_pairs // max(len(by_path), 1), len(cand) * (len(cand) - 1) // 2)
        for _ in range(max(k, 1)):
            a, b = rng.choice(len(cand), size=2, replace=False)
            pa, pb = present[:, a], present[:, b]
            n11 = float((pa & pb).sum()); n10 = float((pa & ~pb).sum())
            n01 = float((~pa & pb).sum()); n00 = float((~pa & ~pb).sum())
            tot = n11 + n10 + n01 + n00
            num = n11 * n00 - n10 * n01
            den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
            if den <= 0:
                continue
            phis.append(float(num / den))  # phi coefficient == Pearson r for binaries
            # excess log-evidence from pretending independence, at the observed joint
            joint = np.array([n11, n10, n01, n00]) / tot
            indep = np.array([
                (n11 + n10) * (n11 + n01), (n11 + n10) * (n10 + n00),
                (n01 + n00) * (n11 + n01), (n01 + n00) * (n10 + n00)]) / tot ** 2
            m = joint > 0
            kl_terms.append(float(np.sum(joint[m] * np.log(joint[m] / np.clip(indep[m], 1e-12, None)))))
            n_tested += 1

    phis = np.asarray(phis)
    kl_terms = np.asarray(kl_terms)
    return {
        "n_pairs_tested": n_tested,
        "phi_mean_abs": float(np.abs(phis).mean()) if n_tested else 0.0,
        "phi_median_abs": float(np.median(np.abs(phis))) if n_tested else 0.0,
        "phi_p95_abs": float(np.percentile(np.abs(phis), 95)) if n_tested else 0.0,
        "phi_max_abs": float(np.abs(phis).max()) if n_tested else 0.0,
        "frac_pairs_abs_phi_gt_0.1": float((np.abs(phis) > 0.1).mean()) if n_tested else 0.0,
        "mutual_information_nats_mean": float(kl_terms.mean()) if n_tested else 0.0,
        "mutual_information_nats_p95": float(np.percentile(kl_terms, 95)) if n_tested else 0.0,
    }


def decide_regime(validation: dict, noise_floor_jsd: float) -> str:
    """ABSOLUTE if reconstruction error is within the generator's own noise."""
    return "ABSOLUTE" if validation["jsd_mean"] <= noise_floor_jsd else "RELATIVE"
