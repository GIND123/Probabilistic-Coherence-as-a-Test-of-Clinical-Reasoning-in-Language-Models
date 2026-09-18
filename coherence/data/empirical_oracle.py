"""Model-free ground-truth posteriors, estimated by conditioning on the corpus.

Motivation
----------
DDXPlus does not release the generator's conditional probability tables, and
reconstructing them by naive Bayes fails badly (see the data audit: JSD 0.55
against the shipped differentials, against a generator noise floor of 0.049).
The independence assumption, not the estimation, is what breaks.

So do not assume independence. The corpus is 1.29M draws from the generator's
joint distribution, and PATHOLOGY is the sampled ground truth. Therefore

    P(d | E) ~= #{patients with pathology d whose findings include E}
                -------------------------------------------------------
                #{patients whose findings include E}

is a consistent, assumption-free estimator of the generator's own posterior
for the evidence subset E. Its only cost is variance, which is a binomial
standard error that we can compute exactly and use to *select* test items
with enough support to be worth scoring.

This gives axioms A2 (update fidelity) and A3 (redundancy insensitivity) an
absolute reference that is order-invariant by construction -- it is a
function of the evidence *set* -- and that carries no independence
assumption for a reviewer to attack.

Feasibility (measured, see audit): for evidence subsets drawn from real
patients, median matching support is ~119k at |E|=1, ~21k at |E|=2, ~9.0k at
|E|=3, ~5.1k at |E|=4, ~2.9k at |E|=5 and ~1.6k at |E|=6. Items are filtered
on a minimum support so every reported quantity has a usable confidence
interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl

from coherence.config import KB as KB_DIR

MIN_SUPPORT = 500  # below this an item is not scored
Z = 1.959963984540054  # 95% normal quantile


@dataclass
class EmpiricalPosterior:
    """One conditional estimate P(. | E), with its support and uncertainty."""

    probs: np.ndarray        # (D,)
    support: int             # number of matching patients
    counts: np.ndarray       # (D,) per-pathology matching counts

    @property
    def se(self) -> np.ndarray:
        p = self.probs
        return np.sqrt(np.clip(p * (1 - p), 0, None) / max(self.support, 1))

    def wilson_ci(self) -> tuple[np.ndarray, np.ndarray]:
        """Wilson score interval; behaves at p near 0, which Wald does not."""
        n = max(self.support, 1)
        p = self.probs
        d = 1 + Z**2 / n
        c = (p + Z**2 / (2 * n)) / d
        h = Z * np.sqrt(np.clip(p * (1 - p) / n + Z**2 / (4 * n**2), 0, None)) / d
        return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)

    def log_odds(self, floor: float = 1e-4) -> np.ndarray:
        p = np.clip(self.probs, floor, 1 - floor)
        return np.log(p / (1 - p))


class EmpiricalOracle:
    """Conditioning engine over the packed evidence matrix."""

    def __init__(self, matrix: np.ndarray, labels: np.ndarray,
                 tokens: list[str], pathologies: list[str],
                 age: np.ndarray | None = None, sex: np.ndarray | None = None):
        self.M = matrix
        self.y = labels
        self.tokens = tokens
        self.pathologies = pathologies
        self.age = age
        self.sex = sex
        self.token_index = {t: i for i, t in enumerate(tokens)}
        self.pathology_index = {p: i for i, p in enumerate(pathologies)}
        self.D = len(pathologies)

    # -- construction -----------------------------------------------------
    @classmethod
    def build(cls, df: pl.DataFrame, tokens: list[str], pathologies: list[str],
              cache: Path | None = None) -> "EmpiricalOracle":
        cache = cache or KB_DIR
        mp, lp = cache / "evidence_matrix.npy", cache / "labels.npy"
        ti = {t: i for i, t in enumerate(tokens)}
        pi = {p: i for i, p in enumerate(pathologies)}
        if mp.exists() and lp.exists():
            M, y = np.load(mp), np.load(lp)
        else:
            M = np.zeros((df.height, len(tokens)), dtype=bool)
            y = np.empty(df.height, dtype=np.int32)
            for i, (pth, evs) in enumerate(zip(df["pathology"].to_list(),
                                               df["evidences"].to_list())):
                y[i] = pi[pth]
                for t in evs:
                    j = ti.get(t)
                    if j is not None:
                        M[i, j] = True
            np.save(mp, M)
            np.save(lp, y)
        age = df["age"].to_numpy()
        sex = (df["sex"].cast(pl.Utf8).to_numpy() == "M")
        return cls(M, y, tokens, pathologies, age, sex)

    @cached_property
    def prior(self) -> EmpiricalPosterior:
        counts = np.bincount(self.y, minlength=self.D).astype(np.float64)
        return EmpiricalPosterior(counts / counts.sum(), int(len(self.y)), counts)

    # -- conditioning -----------------------------------------------------
    def mask(self, tokens: list[str], sex: bool | None = None,
             age_band: tuple[int, int] | None = None) -> np.ndarray:
        m = np.ones(len(self.y), dtype=bool)
        idx = [self.token_index[t] for t in tokens if t in self.token_index]
        if idx:
            m &= self.M[:, idx].all(axis=1)
        if sex is not None and self.sex is not None:
            m &= (self.sex == sex)
        if age_band is not None and self.age is not None:
            m &= (self.age >= age_band[0]) & (self.age <= age_band[1])
        return m

    def posterior(self, tokens: list[str], **kw) -> EmpiricalPosterior:
        m = self.mask(tokens, **kw)
        n = int(m.sum())
        counts = np.bincount(self.y[m], minlength=self.D).astype(np.float64)
        probs = counts / n if n else np.full(self.D, 1.0 / self.D)
        return EmpiricalPosterior(probs, n, counts)

    def delta_log_odds(self, context: list[str], new_finding: str,
                       **kw) -> tuple[np.ndarray, EmpiricalPosterior, EmpiricalPosterior]:
        """True change in log-odds from adding one finding to a context.

        This is the A2 target: the model's implied update is regressed on it.
        """
        before = self.posterior(context, **kw)
        after = self.posterior(list(context) + [new_finding], **kw)
        return after.log_odds() - before.log_odds(), before, after

    def is_scorable(self, post: EmpiricalPosterior,
                    min_support: int = MIN_SUPPORT) -> bool:
        return post.support >= min_support

    # -- redundancy (A3) --------------------------------------------------
    def redundancy_candidates(self, context: list[str], min_support: int = MIN_SUPPORT,
                              max_jsd: float = 1e-3, limit: int = 50) -> list[tuple[str, float, int]]:
        """Findings whose addition provably does not move the posterior.

        A3 needs findings with a true posterior change of exactly zero. Rather
        than assuming which findings are uninformative, we measure it: return
        tokens whose conditional posterior is within `max_jsd` of the context
        posterior while retaining enough support to certify that.
        """
        from coherence.metrics.divergence import jsd

        base = self.posterior(context)
        if base.support < min_support:
            return []
        m = self.mask(context)
        sub = self.M[m]
        ysub = self.y[m]
        freq = sub.mean(axis=0)
        cand = np.flatnonzero(freq > 0)
        out = []
        ctx = set(context)
        for j in cand:
            tok = self.tokens[j]
            if tok in ctx:
                continue
            sel = sub[:, j]
            n = int(sel.sum())
            if n < min_support:
                continue
            counts = np.bincount(ysub[sel], minlength=self.D).astype(np.float64)
            d = jsd(base.probs, counts / n)
            if d <= max_jsd:
                out.append((tok, float(d), n))
        out.sort(key=lambda r: r[1])
        return out[:limit]
