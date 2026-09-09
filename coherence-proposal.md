# Probabilistic Coherence as a Test of Clinical Reasoning in Language Models

## One paper, designed backwards from MAKE's acceptance criteria

**Prepared:** 8 September 2026
**Constraint given:** free choice of topic; the only hard requirement is acceptance at *Machine Learning and Knowledge Extraction* (MDPI MAKE)
**Relationship to prior documents:** replaces the DDXPlus conformal / VOI / invariance portfolio and the lab-interpretation proposal. Keeps the register discipline and methodological hygiene from both.

---

# Part 0 — The design principle

You asked for one condition: it gets accepted. So the paper should be engineered around the failure modes that actually cause rejection, not around maximum novelty.

Papers get rejected because a reviewer can attack the **ground truth**, the **metric**, or the **scope fit**. Chasing an unscooped frontier idea does not protect against any of those, and it adds execution risk.

The design below has one structural property that almost nothing else in this space has:

> **The correctness criterion is a theorem, not an annotation.**

Bayesian belief updating over a fixed evidence set is order-invariant. `P(d | e₁, e₂)` is a function of the *set* `{e₁, e₂}`, not the sequence. This holds unconditionally — it needs no independence assumption, no clinician panel, no LLM judge, no severity weighting, no threshold.

So if you present a clinician's findings to a model in a different order and its differential diagnosis changes, that is an unambiguous reasoning failure with an exact zero. There is no reviewer objection available. That is worth more, for your stated goal, than a more original idea with a contestable metric.

A second consequence, and it is the one that makes the whole project cheap: **the order-invariance test requires no labels at all.** It runs on any clinical corpus, including real de-identified notes, because you are comparing the model against itself. The synthetic-data objection — the thing that would have sunk the DDXPlus proposals — simply does not apply to the core measurement.

---

# Part 1 — Where the field actually is (September 2026)

I checked the adjacent literature before committing. Summary:

| Area | Status | Implication |
|---|---|---|
| Severity-weighted conformal prediction for interactive diagnosis | **Taken.** arXiv:2608.27847 (28 Aug 2026), on DDXPlus and MediQ, uses clinical risk as a planning signal, reports fewer questions and fewer high-risk errors | Kills PS-1 and most of PS-2 from your v2 document |
| Conformal risk control in clinical LLM tasks generally | Crowded: medical entity extraction (2603.00924), CARE for summarisation (2606.08969), cost-aware deferral for triage (*Sci Rep* 2026) | "Conformal + medical LLM" is no longer a novelty claim |
| Epistemic vs. aleatoric decomposition in clinical LLMs | Crowded and contested. ConfiDx (*npj Digit Med* 2025), Zhou et al. (*npj Digit Med* 9:141, Feb 2026), clinical text-to-SQL ambiguity/instability (2602.12015). Kirchhof et al. argue the classical split loses meaning for open-ended language tasks | Avoid. You would be defending the framing, not the finding |
| Reasoning-trace faithfulness / decorative CoT | Active: FaithMed, "The illusion of reasoning" (2603.22816), Patel's clinical reasoning graphs (2606.29876) | Real, but you would be a third entrant on someone else's instrument |
| Structured reasoning-graph consistency | Patel published the null result and released the pipeline | PS-3 survives but is derivative |
| **Probabilistic coherence of clinical belief updating** | **Open.** MedAction (2605.07305) names "unreliable diagnostic update" and "degraded multi-turn coherence" as recognised failure modes but supplies no principled metric. BayesBench (2606.30850) does belief trajectories for *moral* judgment. The Bayesian Coherence Coefficient (Imran et al., 2025) is general-domain. MedClarify *uses* Bayes as machinery without testing whether models are Bayes-coherent | **This is the gap.** The field has named the failure and has no instrument for it |

That last row is the opening. MedAction's own paper says current models "adhere rigidly to an initial diagnosis despite contradictory evidence, or shift erratically between unrelated diagnoses" — a qualitative observation begging for a formal measure. You supply it.

---

# Part 2 — The paper

## 2.1 Title

> **Probabilistic Coherence as a Test of Clinical Reasoning in Large Language Models: Measurement and Neuro-Symbolic Correction**

Alternates:

- *The Same Patient, Told Differently: Order Effects in Language Model Differential Diagnosis*
- *Bayes-Coherent Clinical Reasoning: An Assumption-Free Coherence Battery and a Provably Invariant Remedy*

Two named artifacts so the work is citable by component:

- **CoDx** — the Coherence in Differential diagnosis battery (the instrument)
- **ELR-Fusion** — Elicited Likelihood-Ratio fusion (the method)

Naming is negotiable; the structure is not.

## 2.2 The general reasoning claim

**Coherent belief revision under accumulating evidence.**

An agent operating under partial observability must maintain a belief state and revise it as evidence arrives. Coherence — that the belief depends on *what* the agent has learned, not on *the sequence in which it was told* — is the minimum condition for that belief state to be a representation of the world rather than a trace of the conversation.

This is a general property, not a medical one. It governs any multi-turn agent, any tool-using system that accumulates observations, any assistant reading a long document. Medicine makes it measurable because differential diagnosis is one of the few tasks where the target is explicitly a *distribution over hypotheses*, where a rule-based generator can supply the exact posterior, and where a century of cognitive science has already named the human versions of the failure: anchoring, premature closure, conservatism in belief revision.

Put the general claim in one closing paragraph of the discussion. Keep it out of the title and abstract. That is the register discipline from your v2 document and it is correct.

## 2.3 The instrument: four coherence axioms

Each axiom has an exact zero. Each is measurable without human annotation.

### A1 — Order invariance (the load-bearing one)

Present the same evidence set `E = {e₁ … eₙ}` in `K` permutations. Elicit a posterior over pathologies after each. Under Bayes, all `K` posteriors are identical.

**Metric.** Mean pairwise Jensen–Shannon divergence across permutations.

**Critical correction, and do not skip it.** Sampling noise alone produces divergence. The honest estimand is

```
OrderEffect = JSD(across permutations) − JSD(same permutation, resampled)
```

The subtrahend is your **test–retest floor**. Also report a **shuffled-evidence ceiling** (posteriors from unrelated cases) so the scale is anchored at both ends. Reporting a bare divergence number without both anchors is the single fastest route to rejection at MAKE, and reporting all three is exactly the methodological care that journal rewards.

**Requires no ground truth.** This is what lets A1 run on real clinical narratives.

### A2 — Update fidelity

For each finding `e` added to context `C`, compare the model's implied change in log-odds against the true change from the generator's knowledge base. Regress implied on true:

- `β < 1` → conservatism, under-updating (the classic human pattern, Edwards 1968)
- `β > 1` → overreaction
- `R²` → how much of the correct update signal the model tracks at all

This is where DDXPlus's exact posteriors earn their place. Report `β` and `R²` per model, per evidence type.

### A3 — Redundancy insensitivity

Insert a finding that is either already entailed by present evidence or conditionally uninformative given the pathology set. The true posterior change is exactly zero.

**Metric.** Magnitude of spurious update. This catches the "more text ⇒ more confidence" pathology, which is invisible to accuracy metrics and clinically dangerous.

### A4 — Positional anchoring

Regress the final posterior on the position of the decisive finding within the presentation order. A coherent reasoner shows a null slope. Deviation quantifies **premature closure** with a number.

A4 is where the clinical cognitive-science literature does work for you rather than being decorative. Diagnostic anchoring has been described qualitatively for decades and never given a computable magnitude.

### Risk weighting (one extra table, high value)

DDXPlus attaches a severity level to each pathology. Report incoherence twice: unweighted, and weighted by whether the reordering flips a high-severity hypothesis in or out of the top-k. Incoherence that shuffles two benign diagnoses is not the same event as incoherence that drops a pulmonary embolism. This costs you one extra column and pre-empts the "so what?" reviewer.

## 2.4 The method: ELR-Fusion

The instrument shows the failure. The method fixes it by construction, which is what converts an audit into a methodology paper.

**Elicit per-finding evidence weights instead of a posterior.** For each finding `eᵢ` and each candidate pathology `d`, ask the model for a likelihood ratio — conditioned on the finding and the pathology alone, not on the accumulated narrative. Then combine symbolically:

```
log-odds(d | E)  =  log-odds(d)  +  Σᵢ  log LR(eᵢ | d)
```

**Three properties, and the first is the selling point:**

1. **Order invariance is provable, not empirical.** Addition commutes. The method achieves `OrderEffect = 0` by construction, for every model, on every case. You can state this in the abstract as a theorem rather than a result. Methodology papers in applied ML journals rarely get to do that.

2. **It is auditable.** Every finding carries a numeric evidence weight toward every hypothesis. That is a quantitative, inspectable explanation of the diagnosis — which is *causability* in Holzinger's sense, not post-hoc saliency. The explanation is the computation.

3. **It decomposes the failure.** If ELR-Fusion is coherent but less accurate than direct prompting, the deficit is in likelihood estimation. If it is both coherent and more accurate, direct prompting was losing information to positional effects. Either result is a clean, reportable finding.

**The honest caveat, stated in the paper, not buried.** Order invariance of the *true* posterior is unconditional. Naive multiplication of likelihood ratios is only exact under conditional independence given the pathology. DDXPlus's generator specifies its dependency structure, so you can *quantify* the error the naive assumption introduces, and add pairwise correction terms to measure how much is recovered. Handling this explicitly, with numbers, is the difference between a careful paper and a rejected one. Put it in its own subsection.

## 2.5 The comparison that decides the paper

There is one baseline that a competent reviewer will demand, and you should build the paper around it rather than wait to be asked:

> **Permutation ensembling.** Run the model on `K` random orderings and average the posteriors. This *also* achieves order invariance, trivially.

You must include it, and you must win on grounds other than coherence:

| | Order-invariant | Cost | Auditable per finding | Scales with |
|---|---|---|---|---|
| Direct prompting | No | 1× | No | — |
| Self-consistency (k samples) | No | k× | No | k |
| **Permutation ensemble (K orders)** | Approximately, at large K | **K×** | No | K |
| **ELR-Fusion** | **Exactly, at any budget** | n queries (n = findings) | **Yes** | n, not K! |

The argument is: permutation ensembling buys approximate invariance at a cost that grows with the number of orderings sampled, and delivers no explanation. ELR-Fusion buys exact invariance at a cost linear in the number of findings, and the intermediate quantities *are* the explanation. Make that table Table 1.

**Full baseline set:** direct posterior elicitation; CoT then posterior; self-consistency ensemble; permutation ensemble; ELR-Fusion; exact Bayesian oracle (upper bound); prior-only (floor).

---

# Part 3 — Datasets

| Dataset | Role | Access | Latency |
|---|---|---|---|
| **DDXPlus** (Fansi Tchango et al., arXiv:2205.09148) | Primary. ~1.3M synthetic patients, 49 pathologies, 223 evidences, ground-truth differential *distributions*, per-pathology severity, rule-based generator | CC-BY-4.0, **ungated**, Figshare | **Zero** |
| **MIMIC-IV-ED + MIMIC-IV-Note** | A1 and A4 on real clinical narratives. **No labels needed** — this is the external-validity arm and it costs you nothing but credentialing | PhysioNet credentialed | ~1 week |
| **MedQA** | A1 generalisation beyond DDX-style cases | Public | Zero |
| **MedR-Bench** (*Nat Commun* 2025, 1453 structured cases across 13 body systems) | A1 on realistic case reports with richer narrative structure | Public | Zero |

**Week-1 verification item.** DDXPlus's utility here depends on the released condition/evidence knowledge base being sufficient to recompute posteriors for *arbitrary evidence subsets*, not just the full-case differentials shipped with each record. Confirm this before anything else. If the KB is insufficient, A2 and A3 degrade to relative rather than absolute measures — the paper survives, but A1 and A4 become the backbone. Know which world you are in by day three.

**API/DUA split, declared in Methods.** Frontier API models run on DDXPlus, MedQA, and MedR-Bench (all public). They never touch MIMIC-derived text. All MIMIC experiments use locally-deployed open weights, with no patient-derived text leaving the machine. State this explicitly; it doubles as the justification for the open-model emphasis and reviewers in this space notice.

---

# Part 4 — Models

Structure the model set as a factorial comparison. Each axis is a research question, not a leaderboard row.

| Axis | Contrast | Question |
|---|---|---|
| Reasoning vs. non-reasoning | Qwen3-32B thinking vs. non-thinking (same weights) | Does test-time reasoning buy coherence, or just accuracy? |
| Medical-tuned vs. general | MedGemma 1.5 27B vs. Qwen3-32B | Does domain tuning improve belief revision or only knowledge? |
| Scale | MedGemma 1.5 4B vs. 27B; GPT-OSS-20B vs. 120B | Does coherence emerge with scale? |
| Open vs. frontier | Above vs. GPT-5.x / Claude Opus / Gemini 3 Pro (public data only) | Is coherence a capability frontier or a training-recipe artifact? |

Recommended set: **MedGemma 1.5 27B and 4B** (Google, Jan 2026, Gemma 3 architecture, ~91% MedQA), **Qwen3-32B and 8B** (thinking/non-thinking contrast inside one family — this is the cleanest ablation available and costs nothing extra), **GPT-OSS-20B and 120B**, **DeepSeek-R1**, plus frontier APIs on public data.

The Qwen3 thinking/non-thinking contrast is the highest-value single comparison in the paper. Same weights, same knowledge, different inference-time reasoning. If coherence does not improve, that is a strong and quotable claim about what test-time reasoning does and does not buy.

**Compute.** Pure inference plus a light symbolic layer. 4-bit serving via vLLM puts 27B–32B comfortably on a 48 GB card. No RL, no full fine-tuning, no multi-seed training sweeps. If H100 hours are available, use them for the 120B arm and for an optional LoRA run that trains schema-compliant likelihood-ratio emission — but the paper stands without either. **This is the lowest-execution-risk design in any of the three proposals you now have.**

---

# Part 5 — Ablations

Every row is a planned experiment. Collectively they make completeness attacks unavailable.

| # | Ablation | Isolates | Priority |
|---|---|---|---|
| 1 | K permutations ∈ {2, 5, 10, 20} | Order-effect magnitude and its convergence | **Core** |
| 2 | Test–retest floor and shuffled-evidence ceiling | Anchoring of every divergence figure | **Core** |
| 3 | Evidence set size ∈ {3, 5, 10, 20} | Does incoherence grow with context load? | **Core** |
| 4 | Decisive-finding position (first / middle / last) | A4 anchoring | **Core** |
| 5 | ELR vs. all baselines incl. permutation ensemble | The central comparison | **Core** |
| 6 | Independence correction on / off | The honest caveat, quantified | **Core** |
| 7 | LR elicitation format: numeric / log-odds / ordinal verbal bands | Is the method robust to how you ask? | **Core** |
| 8 | Thinking vs. non-thinking (Qwen3, same weights) | What test-time reasoning buys | High |
| 9 | Prior supplied vs. withheld | Base-rate neglect | High |
| 10 | Narrative prose vs. bulleted findings | Presentation-format dependence | High |
| 11 | Salient-but-irrelevant distractor insertion | Framing susceptibility | High |
| 12 | Severity-weighted vs. unweighted incoherence | Clinical consequence | High |
| 13 | Temperature ∈ {0, 0.3, 0.7} | Is incoherence just sampling noise? (It is not — #2 proves it) | Medium |
| 14 | MIMIC real-narrative arm, A1 only | External validity without labels | Medium |
| 15 | Does answer-level stability predict posterior-level coherence? | Dissociation between behaviour and belief | Medium |

Ablation 15 deserves a note: if a model's top-1 diagnosis is stable across orderings while its full posterior is not, that is a dissociation between behaviour and belief, and it is the more citable half of the paper. Patel's accuracy-blind null result suggests it will hold. Pre-register the prediction.

---

# Part 6 — Why this specific paper gets accepted at MAKE

Since that is your only condition, here is the mapping, made explicit:

1. **The metric cannot be attacked.** Order invariance is a theorem. No annotator, no LLM judge, no arbitrary threshold, no severity weights in the primary analysis.
2. **The method claim is provable.** "ELR-Fusion achieves exact order invariance by construction" is a mathematical statement, not an empirical one. Applied ML journals accept those readily.
3. **Neuro-symbolic hybrid.** MAKE's own 2026 research-agenda paper (*Mach. Learn. Knowl. Extr.* 8(1):6) names hybrid architectures that embed causality and domain knowledge as a leading frontier. ELR-Fusion is literally that: LLM supplies likelihoods, symbolic layer supplies coherent combination.
4. **Causability, not saliency.** Per-finding evidence weights are a quantitative, inspectable explanation where the explanation *is* the computation. This is Holzinger's distinction, and hitting it deliberately matters at his journal.
5. **Verbatim scope fit.** Uncertainty quantification for LLMs is an open special issue. See Part 7.
6. **Correct methodological register.** Noise floors, test–retest ceilings, honest independence caveat, pre-registered predictions, negative results reported. This is the house style — MAKE recently published an EEG study that explicitly declined to claim its main effect because a non-circular check did not corroborate it.
7. **It cannot fail to produce a result.** Order effects either exist or they do not. Large effects → a strong finding. Small effects in frontier models but large in deployable open models → a *better* finding, because that is the deployment-relevant regime. Null everywhere → a clean negative result that MAKE publishes.

---

# Part 7 — Venue

**Primary: MAKE Special Issue — *LLM-Inspired New Generation Machine Learning: Hyperparameter Optimization and Uncertainty Quantification*. Deadline 31 December 2026.** Verbatim topical match, and the deadline gives you scheduling discipline without being tight.

**Fallback within MAKE: Topical Collection — *Robust and Uncertainty-Aware Learning from Real-World Data*,** edited by Federico Cabitza and Andrea Campagner. Rolling deadline. Their published line is precisely about the inadequacy of accuracy metrics and the treatment of uncertainty and disagreement in medical ML. The MIMIC real-narrative arm is aimed at them.

**Do not target** *Clinically Robust and Transparent AI-Assisted Medical Diagnostics* — it closes 30 September 2026, three weeks away.

**Dual-track it anyway.** arXiv preprint first, then AMIA 2027, CHIL, or ML4H in parallel, with MAKE as the journal version. I would keep this advice from your v2 document: MAKE has a real 8.4 IF and Q1 standing, but a meaningful fraction of CS admissions committees discount MDPI regardless of metrics, and the ~19-day first-decision cycle reads to some as thin scrutiny. arXiv plus a community venue plus a Q1 journal version costs you nothing and dominates a single MDPI entry.

---

# Part 8 — Timeline

Ten weeks. No training on the critical path.

| Week | Work | Gate |
|---|---|---|
| 1 | DDXPlus ingest. **Verify posterior recomputation over arbitrary evidence subsets.** Start MIMIC credentialing in parallel | Know by day 3 whether A2/A3 are absolute or relative |
| 2 | Serving stack (vLLM, 4-bit); posterior elicitation schema; permutation harness; schema validation | ≥98% valid structured outputs across all models |
| 3 | **A1 at scale.** Test–retest floor, shuffled ceiling, K sweep, full model set | Order-effect magnitudes with CIs — this is the headline |
| 4 | A4 anchoring; evidence-set-size sweep; distractor and format ablations | Positional effects quantified |
| 5 | A2 update fidelity (β, R²); A3 redundancy insensitivity | Belief-revision profile per model |
| 6 | ELR-Fusion implementation; elicitation-format ablation | Method runs end to end |
| 7 | Full baseline comparison including permutation ensemble; accuracy, calibration, cost | Table 1 complete |
| 8 | Independence correction; quantify naive-assumption error using the DDXPlus dependency structure | The honest caveat, with numbers |
| 9 | MIMIC real-narrative arm; severity-weighted incoherence; MedQA and MedR-Bench generalisation | External validity |
| 10 | Writing, code release, arXiv, submission | Preprint out |

**Slip risk is week 1, not week 10.** Everything downstream depends on the DDXPlus knowledge base supporting subset posteriors. Resolve that first and the rest is inference plus analysis.

**Freeze the scope.** The obvious temptations are a fine-tuning arm, a multi-turn interactive arm, and an information-acquisition arm. All three are follow-up papers. This one is sized to ship.

---

# Part 9 — Reviewer attacks and defences

| Attack | Defence |
|---|---|
| "Order effects in LLMs are already known." | Known qualitatively and in general domains. Never formalised against an exact clinical posterior, never separated from sampling noise with a test–retest floor, never given a severity-weighted clinical reading, and never fixed by a provably invariant method. MedAction names the failure and supplies no metric — cite it as the motivating gap. |
| "Permutation ensembling already solves this." | Table 1. It buys approximate invariance at cost growing in K and yields no explanation. ELR is exact at cost linear in findings and produces per-finding evidence weights. Pre-empt this; do not wait for it. |
| "Naive Bayes assumes independence, which is false clinically." | Stated in the paper, in its own subsection, with the error quantified against DDXPlus's dependency structure, plus a pairwise-correction ablation. Note also that the *measurement* (A1) makes no independence assumption at all — only the method does. |
| "DDXPlus is synthetic." | The synthetic generator is *why* exact posteriors exist — a feature for A2/A3. And A1/A4 require no ground truth, so they run on real MIMIC narratives. That arm exists precisely to answer this. |
| "This is an evaluation paper." | ELR-Fusion is a method with a provable guarantee and a full baseline comparison. And MAKE demonstrably publishes instruments. |
| "No clinician validation." | Optional and cheap to add: have 2–3 clinicians review ~50 cases where reordering flips a high-severity diagnosis, and rate whether the flip is clinically defensible. Small ask, large credibility gain, authorship attached. Worth doing if a clinician is reachable. |
| "How is this AGI-relevant?" | Do not put it in the title or abstract. One paragraph in the discussion: coherent belief revision under accumulating evidence is a prerequisite for any multi-turn or tool-using agent, and its absence is a general limitation rather than a medical one. |

---

# Part 10 — Honest assessment

**What I am confident about.** The measurement design is the most attack-resistant in this space, because the correctness criterion is a theorem. Data access is near-zero-latency. The compute fits on one card. The core experiment returns a number regardless of which way it comes out. The ablations are the contribution rather than a defensive appendix, which is the structural property you asked for two messages ago and it carries over.

**What I am less confident about.**

- **Effect size in frontier models.** They may be substantially coherent already. If so, the paper's centre of gravity shifts to deployable open models and to the real-narrative MIMIC arm, where longer and messier context should produce larger effects. Plan the framing so either outcome is a result, and pre-register the prediction.
- **Likelihood-ratio elicitation quality.** Models may give noisy or refused numeric LRs. Mitigate with ordinal verbal bands mapped to LR ranges, and make elicitation format a first-class ablation (#7) rather than an implementation detail.
- **DDXPlus knowledge-base sufficiency.** Flagged above as the week-1 gate. This is the one thing that could force a redesign, and you will know within three days.
- **Field velocity.** Two of your v2 statements were scooped inside three months. Re-run the search at week 8 and again before submission. Instruments and coherence results age better than SOTA claims, so a competing paper is more likely to strengthen your related-work section than to kill you.

**What I set aside at your instruction.** This design no longer routes through your PI's laboratory-test work. That was deliberate and you authorised it. Two practical assets still shape execution regardless of topic: clinician reachability for the optional validation in Part 9, and the NAIRR H100 allocation, which expires 14 December 2026 and would cover the 120B arm and an optional LoRA run. Neither is on the critical path here — which is itself an argument for this design over the previous two.
