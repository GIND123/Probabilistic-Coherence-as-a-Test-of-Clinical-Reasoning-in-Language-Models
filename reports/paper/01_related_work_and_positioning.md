# Related work and positioning

Searched September 2026. Every claim of novelty below is stated against a
specific paper, with the specific dimension on which we differ. Where a
neighbour is close, it is cited as support rather than argued away.

---

## 1. The nearest neighbours, and what separates us

### 1.1 Dutch Books for Language Models (arXiv:2609.02797, Sep 2026)

The closest methodological relative. Evaluates probabilistic coherence of LM
forecasts via de Finetti: the largest guaranteed arbitrage profit against
model-stated probabilities, computed by linear programming. Label-free.
Finds substantial incoherence, worse with richer logical relationships
between events, and — notably for us — that *irrelevant contextual detail can
raise incoherence by an order of magnitude*.

| | Dutch Books (2609.02797) | **CoDx (this work)** |
|---|---|---|
| Coherence type | **synchronic** — one assignment across related events | **diachronic** — invariance of belief *revision* under accumulating evidence |
| Formal basis | de Finetti / Dutch book | order-invariance of Bayesian updating over a fixed evidence set |
| Domain | stock-return events | clinical differential diagnosis |
| Noise separation | none reported | test–retest floor, between-patient ceiling, generator replicate floor |
| Remedy | none (speculates on training) | ELR-Fusion, invariant by construction |

The two are complementary and non-competing: a model can be Dutch-book
coherent at every instant and still let the *order* of evidence determine
where it lands. Their irrelevant-context finding independently motivates our
distractor ablation.

### 1.2 Measuring the Unmeasurable (Wang, medRxiv 2026.03.27.26349583)

The closest *clinical* relative, and the strongest motivation for this paper.
Three-condition ablation, **N = 50 case reports, 150 runs, one model**
(claude-sonnet-4). Introduces a 5+2 human-scored rubric and a 6-code failure
taxonomy. Documents **Convergence Regression**: models reach the correct
diagnosis at an intermediate stage and abandon it when later evidence
triggers pattern-matching — 90% access, 60% retention, a **30%
Access–Stability Dissociation invisible to single-shot evaluation**. Proposes
the SIPS scaffold, which closes the gap but *costs* top-1 accuracy (60% to
40%).

This is the qualitative observation our instrument formalises, and the
differences are the contribution:

* It varies **staging** (information arriving in tranches) with content
  changing between stages. We hold the evidence **set** exactly fixed and vary
  only its **sequence**, which is what makes the correct answer a theorem
  rather than a judgement.
* Its ground truth is a **human-scored rubric** — precisely the contestable
  criterion our design is built to avoid.
* 50 cases and one model, versus 1,956 cases across 12 model configurations.
* No noise floor, so the reported instability is not separated from
  resampling variance.
* Its Access–Stability Dissociation is the *answer-level* shadow of our
  ablation 15 (does top-1 stability predict posterior-level coherence?). We
  can measure both levels on the same items and report the dissociation
  quantitatively.

**We cite Wang as the motivating finding and position CoDx as the measurement
instrument it calls for.**

### 1.3 MedEinst (arXiv:2601.06636, ACL 2026)

Counterfactual benchmark, 5,383 paired cases over **49 diseases** — the
DDXPlus pathology set. Each pair is a control plus a "trap" whose
discriminative evidence is altered to flip the diagnosis; susceptibility is
the **Bias Trap Rate**. 17 LLMs evaluated. Proposes ECR-Agent (causal-graph
reasoning plus critic-driven memory).

MedEinst changes **what the evidence says**; we change **only the order in
which it is said**. Theirs requires constructing counterfactual traps and
knowing the flipped label; ours requires neither — the correctness criterion
is invariance, and its target value is zero by theorem. The two measure
orthogonal failure modes (shortcut reliance vs. sequence dependence) on the
same substrate, and MedEinst's ACL acceptance is evidence that DDXPlus-derived
instruments are taken seriously.

### 1.4 Premature Closure in Frontier LLMs (arXiv:2605.15000)

**Terminology collision to head off explicitly.** This paper uses "premature
closure" for *failure to abstain* — answering when clarification, escalation
or refusal was the safer act — measured as false-action rates on MedQA (500),
AfriMed-QA (490), HealthBench (861) and 191 physician-authored adversarial
queries. Baseline false-action rates 55–82%.

Our A4 uses "anchoring" in the **positional** sense: the effect of *where* a
decisive finding sits in the presentation on the final posterior. Different
construct, different measurement. The paper will say so in one sentence and
use *positional anchoring* throughout rather than the overloaded term.

### 1.5 Order effects and position bias in general-domain LLMs

Well documented and not claimed as novel: positional bias across prompt
orderings (Findings of EMNLP 2025), *Fragile preferences* (PNAS Nexus 2026)
on order effects in preference elicitation, position bias in audio-language
models (arXiv:2510.00628), and evidence that query timing produces positional
biases opposite to humans' (arXiv:2608.12387). One line of this work reports
that **apparent stability in final decisions can mask substantial intermediate
belief updating** — again the dissociation our ablation 15 targets.

What is absent from all of it: a clinical posterior with an exact reference,
separation from sampling noise, a severity-weighted reading, and a remedy with
a guarantee.

---

## 2. What this leaves as the contribution

1. **A diachronic coherence axiom with an exact zero, measured against three
   anchors.** Not "models show order effects" — that is known — but *how much
   of the observed divergence is attributable to order*, once the model's own
   resampling noise and the between-patient scale are both measured. No prior
   work in this space reports a test–retest floor.

2. **An assumption-free absolute reference for belief updating.** DDXPlus
   ships no conditional probabilities (audit: 0 of 888 slots). Rather than
   assume conditional independence to reconstruct them — which we show fails,
   at 11x the generator's own noise — we condition directly on 1.29M released
   patients. This makes A2/A3 absolute while making **no** independence
   assumption, which removes the standard objection to Bayesian clinical
   baselines from the measurement entirely.

3. **A ground-truth noise floor for DDXPlus that the field does not know
   about.** 25.8% of exact-replicate cases receive different differentials
   (mean JSD 0.049; top-1 flips 0.48%). Every DDXPlus differential-matching
   metric therefore has a ceiling below 1.0 — a result that applies to
   MedEinst, to DDXPlus leaderboards, and to any future work on the corpus.

4. **A remedy whose guarantee is a theorem, not a result.** ELR-Fusion
   achieves exactly zero order effect at any budget, verified numerically at
   machine epsilon, against permutation ensembling whose residual decays only
   as O(1/sqrt(K)) and which yields no per-finding explanation.

5. **Instrument calibration reported as a first-class artifact.** Ten
   documented calibration steps, several of which would each have produced a
   plausible but wrong headline number. This is the register the venue
   rewards.

---

## 3. Venue fit (MAKE, MDPI)

Founded and edited by Andreas Holzinger; *causability* — explanation as the
computation rather than post-hoc attribution — is his distinction and a stated
concern of the journal. MAKE's own research-frontiers article
(*Mach. Learn. Knowl. Extr.* 8(1):6) names **neuro-symbolic integration** —
"large language models act as controllers that delegate subtasks to symbolic
solvers" — and **knowledge-grounded learning** among its frontiers, and flags
uncertainty quantification as "a foundational requirement for trustworthy AI".

ELR-Fusion is literally that architecture: the LLM supplies per-finding
likelihood ratios, a symbolic layer supplies the coherent combination, and the
intermediate quantities *are* the explanation. The fit is structural, not
rhetorical.

---

## Sources

- Dutch Books for Language Models — https://arxiv.org/abs/2609.02797
- Measuring the Unmeasurable (Wang 2026) — https://doi.org/10.64898/2026.03.27.26349583
- MedEinst — https://arxiv.org/abs/2601.06636 · https://aclanthology.org/2026.acl-long.1847/
- Quantifying and Mitigating Premature Closure in Frontier LLMs — https://arxiv.org/pdf/2605.15000
- Characterizing Positional Bias in LLMs — https://aclanthology.org/2025.findings-emnlp.1124/
- Fragile preferences (PNAS Nexus 2026) — https://academic.oup.com/pnasnexus/article/5/8/pgag246/8756895
- Query Timing Produces Opposite Positional Biases — https://arxiv.org/html/2608.12387
- Hearing the Order (audio-LM position bias) — https://arxiv.org/abs/2510.00628
- Probabilistic coherence, logical consistency, and Bayesian learning (PLOS One) — https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0281372
- Research Frontiers in ML & Knowledge Extraction — https://www.mdpi.com/2504-4990/8/1/6
- Towards Accurate Differential Diagnosis with LLMs — https://arxiv.org/abs/2312.00164
- Uncertainty Quantification for ML in Healthcare: A Survey — https://arxiv.org/pdf/2505.02874
