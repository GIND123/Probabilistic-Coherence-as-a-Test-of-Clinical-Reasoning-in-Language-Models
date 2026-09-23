# PI comments — round 2 (received 2026-09-23)

Status: **recorded, not yet actioned.** No code or analysis changed in response to
these. This file exists so the comments survive to the next working session with
their reasoning intact, and so the triage below does not have to be redone.

Round 1 (the external review) is answered in [`comments.txt`](../../comments.txt).
These are separate, and the two should not be merged: round 1 asked for
measurements, round 2 asks mostly for framing and for a construct-validity
argument.

---

## 1. Verbatim comments

### 1.1 On the abstract

> The abstract is technically sound but hard to enter. It opens with a theorem
> before saying why anyone should care, so a clinical informatics reader has to
> get through three sentences before learning that this is a study of LLM
> diagnostic reasoning. It is also very dense after that point. Terms like
> "three anchors," "assumption-free empirical oracle," "probabilistic
> interface," and "ELR Fusion" appear with little or no definition, and the
> final ELR Fusion versus canonical-order concession, while honest, takes up a
> lot of room and ends the abstract on a hedge.
>
> There is also a small accuracy issue in the opening. "Needs no annotator, no
> judge model and no independence assumption, because the target value is known
> to be zero exactly" is true only for A1 (and A3). A2 depends on the
> corpus-derived oracle. As written, the sentence seems to cover the whole
> instrument.

**Suggested rewrite of the opening** (PI's own wording, to be used as the base
text rather than paraphrased):

> Clinical evidence arrives in sequence (history, examination, then tests), yet
> a coherent diagnostician's belief should depend only on what was found, not on
> the order in which it was found. Because Bayesian belief revision is
> order-invariant as a theorem, this yields an exact, annotation-free test for
> clinical language models: a coherent reasoner's order effect is zero. We
> combine this with three related coherence axioms (update fidelity, redundancy
> insensitivity and positional anchoring) to ask whether LLMs revise diagnostic
> beliefs as probabilistic reasoners should. We evaluate eleven open-weight
> models (4B–32B; general, medically tuned and reasoning-distilled) on 1.12
> million elicited responses across 1,956 DDXPlus patients and 49 pathologies.

### 1.2 On DDXPlus — what the oracle measures

> **What does the oracle measure: clinical reasoning or fidelity to the
> generator?** DDXPlus patients are sampled from a rule-based simulator with its
> own priors and pathology set. The empirical P(d | E) therefore encodes that
> simulator's implicit model, not real-world epidemiology. A model whose
> clinical knowledge differs from the generator's assumptions could look
> "incoherent" on A2 even if its updates are clinically reasonable. The authors
> could discuss this, and report A2 on the subset of findings whose oracle
> log-odds change is large, where the expected direction is unambiguous. They
> could also check whether direction agreement improves for pathologies where
> DDXPlus's evidence profiles match standard clinical references.

### 1.3 On DDXPlus — subset selection and prompt rendering

> **How were oracle-eligible evidence subsets selected, and how was evidence
> presented?** With a median of 19 findings per patient and a minimum support of
> 500 matching patients, the full evidence set of most cases cannot be
> conditioned on directly. Please clarify which subsets were used for A2 and A3,
> and whether the support threshold biases the analysis toward common,
> low-information findings. Relatedly, DDXPlus evidence is structured, with
> categorical and multi-valued codes whose English phrasing was translated from
> the original French release. How were these findings rendered into prompts?
> Permuting a structured list may probe something different from reordering a
> clinical narrative. The authors could also comment on possible training-data
> exposure, since DDXPlus has been public since 2022.

---

## 2. Triage

Six distinct asks. Three are text-only; three need a measurement that does not
yet exist. None needs a GPU — every one of them runs off the corpus and the
already-elicited responses.

| # | ask | kind | cost | where it lands |
|---|---|---|---|---|
| P1 | Reopen the abstract on the clinical motivation; use the PI's draft | text | minutes | `README.md` §Abstract, `reports/writeup.tex` |
| P2 | Fix the over-broad annotation-free claim (A2 is *not* annotation-free) | text, **accuracy fix** | minutes | same, plus C2 in §1 Contributions |
| P3 | Define or cut the dense terms; shorten the ELR-Fusion concession | text | ~1 h | same |
| P4 | Discuss generator-fidelity vs clinical-reasoning as a construct-validity limit | text | ~1 h | new §Limitations; `reports/paper/02_methodology.md` |
| P5 | Report A2 restricted to large-\|Δ log-odds\| findings | **new analysis** | CPU, ~1 h | new table `rev_a2_large_delta.csv`; §4.2 |
| P6 | Document A2/A3 subset selection, the ≥500 support threshold's bias, and prompt rendering of structured codes | text, mostly already measured | ~2 h | §3 Method; `reports/paper/02_methodology.md` |

Two further asks are worth separating out because they are open-ended:

- **P5b — concordance with standard clinical references.** Check whether A2
  direction agreement improves for pathologies whose DDXPlus evidence profiles
  match an external reference. This needs a reference source that does not
  exist in the repo, and a defensible matching rule. Scope it before starting;
  it could easily become its own section.
- **P6b — training-data exposure.** DDXPlus has been public since 2022, so
  every evaluated checkpoint may have seen it. Note that A1 and A4 are
  *unharmed* by contamination (their target is zero regardless of whether the
  model memorised the corpus — memorisation would, if anything, make a model
  look more coherent, so contamination biases against our finding, not for it).
  A2 and the accuracy numbers are the exposed quantities. This asymmetry is a
  genuine strength of the instrument and should be stated rather than buried.

---

## 3. Notes for whoever picks this up

**P2 is the one to do first.** It is a correctness issue in a sentence we
already ship, not a matter of taste. The current README opening says the
criterion "requires no annotator, no judge model, and no independence
assumption". Only A1 and A4 are clean in that sense: their target is zero by
theorem, and nothing outside the model's own responses is consulted to score
them.

The PI writes "true only for A1 (and A3)", where our own Hub dataset card
marks A3 as needing ground truth. **Both are right, about different things**,
and the paper should say which:

- **A3's scoring target is zero**, exactly as A1's is. In that sense the PI is
  correct and the card is misleading.
- **A3's items cannot be built without the oracle.** `coherence/axioms/a3_redundancy.py`
  certifies a finding as uninformative by checking that the empirical posterior
  moves by < 1e-3 JSD with and without it, at ≥ 500 patients of support. The
  corpus is what licenses the claim that the correct update is zero.
- **A2 is the genuinely oracle-dependent axiom.** Its target is a corpus-derived
  log-odds change, not a constant, so the whole measurement inherits the
  simulator's priors — which is precisely comment 1.2's point.

So the honest formulation is: *the oracle enters A2 as the scoring target and
A3 as the item-selection criterion, and does not enter A1 or A4 at all.* Fix
this in three places that currently share the loose wording: the README
abstract, contribution C2, and `DATASET_CARD` in `coherence/hub/hf_sync.py`
(whose "needs ground truth" column should become "needs the corpus oracle", with
A3 marked *for item construction*).

**P5 is cheap and probably strengthens the paper.** The headline A2 result is
direction agreement 0.454–0.549, i.e. chance. The PI's point is that some of
that band is findings where the oracle itself barely moves, so the "correct"
direction is close to arbitrary and a coherent model has nothing to agree with.
Restricting to findings with a large oracle log-odds change is the fair test. If
agreement stays at chance there, the finding is considerably stronger than it is
now. If it rises, that is a real and reportable qualification. Either outcome is
worth the hour — and note that it cuts against our own headline, so it should be
run before the framing is rewritten around the current number.

**P4 and P6b are the same argument seen from two sides**, and reviewers of the
submitted version will likely raise both. The oracle measures fidelity to the
generator; the generator has been public since 2022. Answering them together,
as one "what this instrument can and cannot see" subsection, will read better
than two scattered defences.
