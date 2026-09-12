"""Turn DDXPlus question/answer tokens into clinical statements.

The corpus stores findings as questionnaire items ("Do you have a fever?").
Presenting them verbatim is defensible for the bulleted format but produces
nonsense as running prose, and ablation #10 contrasts exactly those two
formats -- so the contrast would be confounded by fluency rather than by
structure. This module converts a token into a declarative third-person
finding that both formats share, so the two differ only in presentation.

Coverage is asserted, not assumed: `unmatched_questions` must return empty
for all 223 released questions, and a test enforces that.
"""
from __future__ import annotations

import re

from coherence.data.ddxplus import DDXPlusKB, split_token

# --- second-person -> third-person -------------------------------------
_PRONOUNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\byourself\b", re.I), "themselves"),
    (re.compile(r"\byour\b", re.I), "their"),
    (re.compile(r"\byou\b", re.I), "they"),
]

_IRREGULAR_3SG = {
    "have": "has", "be": "is", "do": "does", "go": "goes",
    "lose": "loses", "eat": "eats", "vomit": "vomits", "suffer": "suffers",
}
_IRREGULAR_PAST = {
    "lose": "lost", "vomit": "vomited", "eat": "ate", "have": "had",
    "suffer": "suffered", "turn": "turned", "be": "was",
}


def _third_person(verb: str) -> str:
    v = verb.lower()
    if v in _IRREGULAR_3SG:
        return _IRREGULAR_3SG[v]
    if v.endswith(("s", "sh", "ch", "x", "z")):
        return v + "es"
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    return v + "s"


def _past(verb: str) -> str:
    v = verb.lower()
    if v in _IRREGULAR_PAST:
        return _IRREGULAR_PAST[v]
    if v.endswith("e"):
        return v + "d"
    if v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


def _conjugate_head(rest: str) -> str:
    """Conjugate the leading verb of `rest` to third person singular."""
    m = re.match(r"^(\w+)(.*)$", rest)
    return f"{_third_person(m.group(1))}{m.group(2)}" if m else rest


def _depersonalise(text: str) -> str:
    for pat, repl in _PRONOUNS:
        text = pat.sub(repl, text)
    return text


# --- interrogative -> declarative --------------------------------------
# Ordered; first match wins. `fn` receives the remainder after the opener.
_RULES: list[tuple[re.Pattern, object]] = [
    # "Do you have X" / "Have you had X" ... fixed-phrase openers
    (re.compile(r"^do you have\b", re.I), lambda r: f"has {r}"),
    (re.compile(r"^do you feel\b", re.I), lambda r: f"feels {r}"),
    (re.compile(r"^do you live\b", re.I), lambda r: f"lives {r}"),
    (re.compile(r"^do you take\b", re.I), lambda r: f"takes {r}"),
    (re.compile(r"^do you suffer\b", re.I), lambda r: f"suffers {r}"),
    (re.compile(r"^do you work\b", re.I), lambda r: f"works {r}"),
    (re.compile(r"^do you regularly\b", re.I), lambda r: f"regularly {_conjugate_head(r)}"),
    (re.compile(r"^do you currently\b", re.I), lambda r: f"currently {_conjugate_head(r)}"),
    (re.compile(r"^do your\b", re.I), lambda r: f"their {r}"),
    (re.compile(r"^do any members of\b", re.I), lambda r: f"among {r}"),
    (re.compile(r"^does your\b", re.I), lambda r: f"their {r}"),
    (re.compile(r"^does the\b", re.I), lambda r: f"the {r}"),
    (re.compile(r"^have you ever\b", re.I), lambda r: f"has ever {r}"),
    (re.compile(r"^have you had\b", re.I), lambda r: f"has had {r}"),
    (re.compile(r"^have you been\b", re.I), lambda r: f"has been {r}"),
    (re.compile(r"^have you noticed\b", re.I), lambda r: f"has noticed {r}"),
    (re.compile(r"^have you recently\b", re.I), lambda r: f"has recently {r}"),
    (re.compile(r"^have you lost\b", re.I), lambda r: f"has lost {r}"),
    (re.compile(r"^have you traveled\b", re.I), lambda r: f"has travelled {r}"),
    (re.compile(r"^have any of\b", re.I), lambda r: f"any of {r}"),
    (re.compile(r"^has any\b", re.I), lambda r: f"any {r}"),
    (re.compile(r"^are you currently\b", re.I), lambda r: f"is currently {r}"),
    (re.compile(r"^are you more\b", re.I), lambda r: f"is more {r}"),
    (re.compile(r"^are your symptoms\b", re.I), lambda r: f"symptoms are {r}"),
    (re.compile(r"^are you consulting because you have\b", re.I),
     lambda r: f"is consulting because they have {r}"),
    (re.compile(r"^are you being\b", re.I), lambda r: f"is being {r}"),
    (re.compile(r"^are you feeling\b", re.I), lambda r: f"is feeling {r}"),
    (re.compile(r"^are you experiencing\b", re.I), lambda r: f"is experiencing {r}"),
    (re.compile(r"^are you taking\b", re.I), lambda r: f"is taking {r}"),
    (re.compile(r"^are you exposed\b", re.I), lambda r: f"is exposed {r}"),
    (re.compile(r"^are you unable\b", re.I), lambda r: f"is unable {r}"),
    (re.compile(r"^are you infected\b", re.I), lambda r: f"is infected {r}"),
    (re.compile(r"^are you significantly\b", re.I), lambda r: f"is significantly {r}"),
    (re.compile(r"^are you of\b", re.I), lambda r: f"is of {r}"),
    (re.compile(r"^are you a\b", re.I), lambda r: f"is a {r}"),
    (re.compile(r"^are you an\b", re.I), lambda r: f"is an {r}"),
    (re.compile(r"^have you or any member of your family ever had\b", re.I),
     lambda r: f"or a family member has had {r}"),
    (re.compile(r"^have you started or taken\b", re.I), lambda r: f"has started or taken {r}"),
    (re.compile(r"^are your\b", re.I), lambda r: f"their {r} are up to date"
     if r.strip().lower().startswith("vaccinations") else f"their {r}"),
    (re.compile(r"^are there any members of your family who have been diagnosed\b", re.I),
     lambda r: f"has a family member diagnosed with {r}"),
    (re.compile(r"^are there members of your family who have been diagnosed with\b", re.I),
     lambda r: f"has a family member diagnosed with {r}"),
    (re.compile(r"^do any members of your immediate family have\b", re.I),
     lambda r: f"has an immediate family member with {r}"),
    (re.compile(r"^are there any members of\b", re.I), lambda r: f"has a family member among {r}"),
    (re.compile(r"^are there members of\b", re.I), lambda r: f"has a family member among {r}"),
    (re.compile(r"^are the\b", re.I), lambda r: f"the {r}"),
    (re.compile(r"^is your\b", re.I), lambda r: f"their {r}"),
    (re.compile(r"^is the\b", re.I), lambda r: f"the {r}"),
    (re.compile(r"^were you diagnosed\b", re.I), lambda r: f"was diagnosed {r}"),
    (re.compile(r"^were you born\b", re.I), lambda r: f"was born {r}"),
    (re.compile(r"^in the last month, have you been\b", re.I),
     lambda r: f"in the last month has been {r}"),
    (re.compile(r"^did your\b", re.I), lambda r: f"their {r}"),
    (re.compile(r"^did you previously\b", re.I), lambda r: f"previously {r}"),
    # Generic verb openers, conjugated.
    (re.compile(r"^did you (\w+)\b", re.I),
     lambda r, v=None: None),  # replaced below by _generic_did
    (re.compile(r"^do you (\w+)\b", re.I), lambda r: None),  # _generic_do
]


def _declarative(question: str) -> str | None:
    q = question.strip().rstrip("?").strip()
    m = re.match(r"^did you (\w+)\b(.*)$", q, re.I)
    if m:
        return f"{_past(m.group(1))}{m.group(2)}".strip()
    m = re.match(r"^do you (\w+)\b(.*)$", q, re.I)
    if m and not re.match(r"^do you (have|feel|live|take|suffer|work|regularly|currently)\b",
                          q, re.I):
        return f"{_third_person(m.group(1))}{m.group(2)}".strip()
    for pat, fn in _RULES[:-2]:
        if pat.search(q):
            return fn(pat.sub("", q).strip())
    m = re.match(r"^are you (\w+)\b(.*)$", q, re.I)
    if m:
        return f"is {m.group(1)}{m.group(2)}".strip()
    m = re.match(r"^have you (\w+)\b(.*)$", q, re.I)
    if m:
        return f"has {m.group(1)}{m.group(2)}".strip()
    return None


# Non-binary questions rendered as labelled attributes, plus the verb phrase
# used when they appear in running prose.
_ATTRIBUTES: dict[str, tuple[str, str]] = {
    # question: (label, prose verb phrase)
    "Do you feel pain somewhere?": ("pain location", "has pain located at"),
    "Does the pain radiate to another location?": ("pain radiates to", "has pain radiating to"),
    "Characterize your pain:": ("pain character", "describes the pain as"),
    "How fast did the pain appear?":
        ("pain onset speed, 0 gradual to 10 sudden", "rates the speed of pain onset, on a scale where 0 is gradual and 10 is sudden, at"),
    "How intense is the pain?": ("pain intensity, 0-10", "rates the pain intensity, out of 10, at"),
    "How precisely is the pain located?":
        ("pain localisation precision, 0-10", "rates how precisely the pain is localised, out of 10, at"),
    "Where is the affected region located?": ("affected region", "has an affected region at"),
    "What color is the rash?": ("rash colour", "has a rash coloured"),
    "Characterize your pain:": ("pain character", "describes the pain as"),
    "How intense is the pain caused by the rash?":
        ("rash pain intensity, 0-10", "rates the pain from the rash, out of 10, at"),
    "Is the rash swollen?": ("rash swelling, 0-10", "has rash swelling, out of 10, of"),
    "How severe is the itching?": ("itching severity, 0-10", "rates the itching, out of 10, at"),
    "Is the lesion (or are the lesions) larger than 1cm?":
        ("lesion larger than 1cm", "has lesions larger than 1cm:"),
    "Do your lesions peel off?": ("lesions peel off", "has lesions that peel off:"),
    "Where is the swelling located?": ("swelling location", "has swelling located at"),
    "Have you traveled out of the country in the last 4 weeks?":
        ("recent travel outside the country", "has recently travelled to"),
}

_NO_TRAVEL = {"N", "n"}


def finding_phrase(kb: DDXPlusKB, token: str, prose: bool = False) -> str:
    """One finding as a lower-case clinical phrase, without a leading subject.

    `prose=True` returns a verb phrase that follows "The patient ..."; the
    default returns a compact form suited to a bullet. Both carry identical
    information, which is what ablation #10 requires.
    """
    code, value = split_token(token)
    ev = kb.evidences[code]
    q = ev.question_en.strip()

    if ev.is_binary and value is None:
        d = _declarative(q)
        return _depersonalise(d if d else f"reports {q.rstrip('?')}")

    label, verb = _ATTRIBUTES.get(q, (None, None))
    rendered = ev.render_value(value)
    if code == "E_204" and str(rendered) in _NO_TRAVEL:
        return "has not travelled outside the country in the last 4 weeks"
    if label is None:
        d = _declarative(q) or f"reports {q.rstrip('?')}"
        return _depersonalise(f"{d} ({rendered})")
    return _depersonalise(f"{verb} {rendered}" if prose else f"{label}: {rendered}")


def bulleted(kb: DDXPlusKB, tokens: list[str]) -> str:
    return "\n".join(f"- {finding_phrase(kb, t).capitalize()}" for t in tokens)


def narrative(kb: DDXPlusKB, tokens: list[str], subject: str = "The patient") -> str:
    """Running prose preserving the given order exactly.

    Findings are grouped three to a sentence so long cases stay readable. The
    order is never altered: order is the independent variable.
    """
    phrases = [finding_phrase(kb, t, prose=True) for t in tokens]
    sentences, buf = [], []
    for i, p in enumerate(phrases):
        buf.append(p)
        if len(buf) == 3 or i == len(phrases) - 1:
            sentences.append(f"{subject} {'; '.join(buf)}.")
            buf = []
    return " ".join(sentences)


def unmatched_questions(kb: DDXPlusKB) -> list[str]:
    """Diagnostic: questions that fall through to the generic rendering."""
    out = []
    for ev in kb.evidences.values():
        if ev.is_binary:
            if _declarative(ev.question_en) is None:
                out.append(ev.question_en)
        elif ev.question_en not in _ATTRIBUTES:
            out.append(ev.question_en)
    return out
