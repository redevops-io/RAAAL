"""Fusion — syntax contributes evidence, and never has authority.

The parser corpus did not show that syntax is correct. It showed where syntax
is informative, where it is unstable, and where no amount of structure settles
the question. This module is written from that, and its central rule is a
consequence of one measured fact rather than a preference:

    In "I contribute monthly and rebalance at year end" Stanza attaches
    `year end` to *contribute*. Confidently, with a clean governor chain the
    scorer walks happily.

So a policy of the form "syntax wins when its score is high" would adopt that
error, because the score *is* high — the parser is not uncertain, it is wrong.
There is no confidence threshold that separates the two, which is why there is
no threshold here at all.

    model proposes      syntax says      outcome
    ────────────────────────────────────────────────────────────
    X                   supports X       AGREE
    X                   neutral          AGREE          (model alone is enough)
    X                   contradicts X    DISAGREE       (never "syntax wins")
    X                   —                INSUFFICIENT_RELATION
                                         (the value needs a binding nobody has)
    anything            —                AMBIGUOUS_BY_LANGUAGE
                                         (the term is one people use both ways)
    nothing             proposes X       DISAGREE       (syntax alone never
                                                         carries a field)

**Only `AGREE` proceeds automatically on a material dimension.** The other
three become `Unresolved` on the intent, which is the contract's way of saying
the meaning is still open — and for a material dimension that blocks sealing,
so nothing downstream can execute a guess.

**`AMBIGUOUS_BY_LANGUAGE` is the outcome this project would not have predicted.**
A Bogleheads thread is titled *"Don't Know How To Rebalance/Reallocate"*. People
writing about their own portfolios do not reliably separate "rebalance back to
target" from "change the target". When the ambiguity is in the language, a
better parser and a better model are both irrelevant: the only correct move is
to ask. Treating that as a low-confidence reading to be resolved is how a
runtime ends up confidently answering a question the user did not ask.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from .syntax import SyntaxEvidence

#: Bumped when an outcome's *meaning* changes, so a stored decision can be told
#: apart from one made under a different policy.
POLICY_VERSION = "quantify-fusion@1"


class Fusion(str, Enum):
    """What fusion concluded about one dimension.

    Four members, and the three that are not `AGREE` are deliberately distinct
    rather than a single `UNRESOLVED`. They call for different repairs: a
    disagreement needs adjudication, a missing relation needs a schema or a
    reader that binds it, and a language ambiguity needs a question to the user.
    Collapsing them would make the ledger say "unresolved" and stop there.
    """

    AGREE = "AGREE"
    """The model proposed a value and syntax did not contradict it."""

    DISAGREE = "DISAGREE"
    """Syntax and the model point at different values, or syntax proposed one
    the model never mentioned. Never resolved by score."""

    INSUFFICIENT_RELATION = "INSUFFICIENT_RELATION"
    """The value cannot mean anything without a binding nobody supplied — three
    ratios and three accounts, with nothing saying which belongs to which."""

    AMBIGUOUS_BY_LANGUAGE = "AMBIGUOUS_BY_LANGUAGE"
    """The words themselves carry both readings, in attested usage. Not a
    parser failure and not a model failure."""

    @property
    def proceeds(self) -> bool:
        return self is Fusion.AGREE


#: Terms people demonstrably use for more than one thing.
#:
#: Every entry needs a source, and `test_fusion.py` asserts it. A list anyone
#: may add a hunch to becomes a list of things nobody wants to implement, and
#: the whole point of this outcome is that the ambiguity was *observed* in how
#: people write rather than predicted from how the code is shaped.
AMBIGUOUS_TERMS: Mapping[str, Mapping[str, Any]] = {
    "rebalance": {
        "readings": "rebalance back to target | change the target allocation",
        "between": ("periodic_rebalancing", "stated_weights"),
        "evidence": "Bogleheads thread 'Don't Know How To Rebalance/Reallocate'",
        "source": "https://www.bogleheads.org/forum/viewtopic.php?t=459742"},
    "reallocate": {
        "readings": "rebalance back to target | change the target allocation",
        "between": ("periodic_rebalancing", "stated_weights"),
        "evidence": "same thread; the two verbs are used interchangeably",
        "source": "https://www.bogleheads.org/forum/viewtopic.php?t=459742"},
}
"""Terms people demonstrably use for more than one thing.

`between` names the contract fields the ambiguity is *between*, and it is what
keeps this outcome from firing on its own vocabulary. "rebalanced annually"
carries the word and no ambiguity at all: the reading is
`periodic_rebalancing=annual`, and the competing reading — that the target is
being changed — needs a target, which the sentence does not contain. "rebalance
to 70/30" does contain one, and there the two readings are both available and
neither is chosen by the words.

So the rule is not "the word appeared". It is "both readings are on the table",
which is what ambiguity means."""


@dataclass(frozen=True)
class Proposal:
    """One reader's answer for one dimension, whoever the reader is."""

    dimension: str
    value: Any
    reader_id: str
    source_span: str = ""


@dataclass(frozen=True)
class Decision:
    """What fusion concluded, and everything needed to explain it later."""

    dimension: str
    outcome: Fusion
    value: Any = None
    """Present only when the outcome proceeds. Deliberately `None` otherwise:
    a decision that carried a value alongside `DISAGREE` is a value a caller
    will render, and then a figure exists for a question that was not settled.
    """

    material: bool = True
    model: Optional[Proposal] = None
    syntax: Sequence[SyntaxEvidence] = ()
    detail: str = ""
    policy_version: str = POLICY_VERSION

    @property
    def proceeds(self) -> bool:
        return self.outcome.proceeds

    def to_json(self) -> dict:
        return {"dimension": self.dimension, "outcome": self.outcome.value,
                "value": self.value, "material": self.material,
                "model": (None if self.model is None else
                          {"value": self.model.value,
                           "reader_id": self.model.reader_id,
                           "source_span": self.model.source_span}),
                "syntax": [e.to_json() for e in self.syntax],
                "detail": self.detail, "policy_version": self.policy_version}


@dataclass(frozen=True)
class Requirement:
    """What a dimension needs before a value means anything.

    `binds` names the relation a value is meaningless without. `60/40` alone is
    a fact; `50/50` in a sentence naming three accounts is not, until something
    says which account it belongs to.
    """

    material: bool = True
    binds: Optional[str] = None
    compare_as: str = "TEXT"
    """How two readers' values for this dimension are the same value. Mirrors
    `Dimension.compare_as` in the schema, and is declared rather than guessed
    for the same reason it is there."""


#: Declared per dimension rather than inferred. A dimension absent here is
#: material and unbound — the conservative reading, since treating an unknown
#: dimension as immaterial would let anything new proceed unexamined.
REQUIREMENTS: Mapping[str, Requirement] = {
    "cadence": Requirement(material=True),
    "amount": Requirement(material=True, compare_as="NUMBER"),
    "assets": Requirement(material=True, compare_as="SET"),
    "allocation_method": Requirement(material=True),
    "moving_average_window": Requirement(material=True,
                                        compare_as="NUMBER"),
    "trigger_semantics": Requirement(material=True),
    "execution_timing": Requirement(material=True),
    "day_rule": Requirement(material=False),
    "evaluation_period": Requirement(material=True),
    "periodic_rebalancing": Requirement(material=False),
    "objective": Requirement(material=True),
    "stated_weights": Requirement(material=True, compare_as="WEIGHTS"),
    "account_allocation": Requirement(material=True, binds="account"),
    "dividend_policy": Requirement(material=False),
}


#: Dimensions where a trailing `m` counts periods rather than millions.
#:
#: `m` is the one genuinely ambiguous magnitude letter. A reader writing `12m`
#: for an amount means twelve million; writing `12m` for a moving-average
#: window it means twelve months, and scaling that produced a twelve-million
#: session window that disagreed with syntax's 12 — a case that had been
#: answered correctly for months, broken by the fix for `2.5k`.
#:
#: `k`, `b` and `bn` are not ambiguous and are scaled everywhere.
PERIOD_DIMENSIONS = frozenset({
    "moving_average_window", "evaluation_period", "holding_period",
    "rebalancing_period", "lookback_window",
})


def same_value(one: Any, other: Any, compare_as: str = "TEXT",
               dimension: str = "") -> bool:
    """Whether two readers said the same thing, by the schema's own rule.

    `Dimension.compare_as` has existed since the first shadow run, for exactly
    this: the model reads an amount as `$500` and the deterministic path
    normalises it to `500`, and a string comparison calls that a disagreement.
    It is a formatting difference, and reporting it as conflict buries the real
    ones — the finding the schema already carries a field to prevent.

    Nothing here can make two different amounts equal. NUMBER strips currency
    and separators, SET compares tokens unordered, and TEXT — the default — is
    still exact, because assuming two spellings mean the same thing is the
    failure this project is about.
    """
    left, right = str(one).strip(), str(other).strip()
    if left.lower() == right.lower():
        return True
    if compare_as == "NUMBER":
        # Canonicalised by the *normaliser*, not by stripping punctuation.
        #
        # A regex that kept digits turned `£1k` into 1 and `12-month` into
        # nothing, so the model's rendering and the deterministic path's value
        # disagreed on three cases that mean the same thing. The normaliser
        # already knows `£1k` is 1000 and `12-month` is a 12-period window;
        # using it here means one place decides what a written number means,
        # rather than two places deciding differently.
        def number(raw: str):
            from .syntax import normalize

            for value in normalize(raw):
                if value.kind in ("money", "duration", "percentage",
                                  "moving_average_window"):
                    return Decimal(str(value.canonical))

            # A bare magnitude suffix, with no currency symbol in front of it.
            #
            # The normaliser knows `£2.5k`; it does not know `2.5k`, and a
            # reader that returns the amount without the currency — which
            # gpt-5.4 does — landed on the fallback below, which stripped the
            # `k` and produced 2.5. Not a failure to compare: a *wrong number*,
            # a thousand times too small, that this case only caught because
            # the other witness happened to disagree with it. Two readers both
            # writing `2.5k` would have agreed on it and settled.
            letters = ("k|bn?" if dimension in PERIOD_DIMENSIONS
                       else "k|m|bn?")
            suffix = re.fullmatch(rf"\s*([\d.,]+)\s*({letters})\s*", raw, re.I)
            if suffix:
                scale = {"k": 1_000, "m": 1_000_000,
                         "b": 1_000_000_000, "bn": 1_000_000_000}
                try:
                    return (Decimal(suffix.group(1).replace(",", ""))
                            * scale[suffix.group(2).lower()])
                except InvalidOperation:
                    return None

            # The blanket rule that was here — refuse any string containing a
            # letter — was far too wide. It took out `500 dollars`, `monthly`
            # rendered numerically, and every unit-carrying value the digit
            # strip below had always handled, and 131 tests went red at once.
            #
            # What actually needed fixing is narrower and is fixed above: a
            # magnitude suffix must not be silently discarded. An unrecognised
            # unit still falls through to the digits, which is what it did
            # before and is a separate question from this one.
            cleaned = re.sub(r"[^\d.]", "", raw)
            try:
                return Decimal(cleaned) if cleaned else None
            except InvalidOperation:
                return None
        a, b = number(left), number(right)
        return a is not None and b is not None and a == b
    if compare_as == "WEIGHTS":
        # A split and the same split with its holdings attached are the same
        # reading, not a disagreement.
        #
        # The hosted reader returns `60/40`; `weight_binding` returns
        # `VTI=60,BND=40` for the same sentence. Compared as text those differ,
        # and fusion asked a question about a split both witnesses had read
        # identically — the derived one simply also says which side is which.
        # What must still disagree is a different split: 60/40 against 70/30.
        def shares(raw: str):
            found = re.findall(r"(\d+(?:\.\d+)?)", str(raw))
            return [Decimal(v) for v in found]

        left_shares, right_shares = shares(left), shares(right)
        return bool(left_shares) and left_shares == right_shares

    if compare_as == "SET":
        split = re.compile(r"[,;]|\band\b")

        # A leading article is dropped before comparing. "a core index fund"
        # and "core index fund" are one holding, and two readers disagreeing
        # about a determiner is a rendering difference of exactly the kind
        # `compare_as` exists to absorb — the same reason NUMBER does not
        # distinguish `$500` from `500`.
        #
        # This is not a loosening towards "close enough". It removes one
        # closed, meaningless class of English function word. Nothing here can
        # make two different holdings equal: `SPX ETF` and `SPY` still differ,
        # which is the substitution the whole boundary prevents.
        article = re.compile(r"^(?:a|an|the)\s+", re.I)

        def members(raw: str) -> set:
            return {article.sub("", p.strip()).lower()
                    for p in split.split(raw) if p.strip()}

        return members(left) == members(right)
    return False


def contradicts(evidence: SyntaxEvidence, proposal: Proposal,
                compare_as: str = "TEXT") -> bool:
    """Whether this evidence argues against the model's value.

    Two ways, and both are about the *sign* of the score rather than its size.
    A negative score on the model's own value is syntax saying "not here"; a
    positive score on a different value is syntax saying "there instead".
    Magnitude is deliberately unused — the `year end` case scored confidently
    and wrongly, so size is not a signal about correctness.
    """
    agrees = same_value(evidence.proposed_value, proposal.value, compare_as,
                        dimension=proposal.dimension)
    return (evidence.score < 0) if agrees else (evidence.score > 0)


def fuse(dimension: str, *, model: Optional[Proposal] = None,
         syntax: Sequence[SyntaxEvidence] = (),
         derived: Optional[Proposal] = None,
         requirement: Optional[Requirement] = None,
         bound: bool = False,
         available: Sequence[str] = ()) -> Decision:
    """One dimension, one decision.

    `available` names the other contract fields something proposed a reading
    for in this utterance. It exists only for the ambiguity check: a lexical
    ambiguity is real when both of its readings are on the table, and
    "rebalanced annually" has only one of them.

    `bound` says whether the relation this dimension requires has actually been
    established. Passed in rather than worked out here, because fusion must
    never inspect a parse — that is `binding.py`'s job, and a fusion layer that
    learned to read structure would become the second parser this architecture
    exists to avoid. `fuse_with_bindings` is the wiring; this is the decision.
    """
    requirement = requirement or REQUIREMENTS.get(dimension, Requirement())
    syntax = tuple(syntax)

    # A derived reader is a reader. Its claim is weighed against the model's by
    # the ordinary rules — agreement settles, disagreement asks — and neither
    # wins by being what it is. That is what "no privileged reader" means: a
    # source type does not decide a disagreement, not that a material fact
    # needs two witnesses before it may exist.
    if derived is not None:
        if model is None:
            return Decision(
                dimension=dimension, outcome=Fusion.AGREE,
                value=derived.value, material=requirement.material,
                model=derived, syntax=syntax,
                detail=f"{derived.reader_id} derived it from the sentence's "
                       "structure and the hosted reader did not answer")
        if not same_value(model.value, derived.value,
                          requirement.compare_as, dimension=dimension):
            return Decision(
                dimension=dimension, outcome=Fusion.DISAGREE,
                material=requirement.material, model=model, syntax=syntax,
                detail=f"the hosted reader read {model.value!r} and "
                       f"{derived.reader_id} derived {derived.value!r} from "
                       "the structure. Two readers, two answers, and the "
                       "difference changes how often the strategy fires")
        # The derived value, not the model's, when the two agree.
        #
        # A derived reader is the declared author of its dimension, and where it
        # speaks it says something the hosted reader did not: `60/40` and
        # `VTI=60,BND=40` agree about the split, and only one of them says which
        # holding takes which share. Settling the model's value here discarded
        # the binding and left the compiler refusing a split it had been handed.
        #
        # Safe precisely because they agree — `same_value` has just established
        # the two readings are the same reading, so preferring the richer one
        # cannot change what the plan means, only how much of it survives.
        return Decision(
            dimension=dimension, outcome=Fusion.AGREE, value=derived.value,
            material=requirement.material, model=model, syntax=syntax,
            detail=f"the hosted reader and {derived.reader_id} agree; the "
                   f"derived reading is kept because it carries the binding")

    ambiguous = _ambiguity(dimension, model, syntax, available)
    if ambiguous is not None:
        return Decision(
            dimension=dimension, outcome=Fusion.AMBIGUOUS_BY_LANGUAGE,
            material=requirement.material, model=model, syntax=syntax,
            detail=f"{ambiguous!r} is used for both readings in attested "
                   f"writing ({AMBIGUOUS_TERMS[ambiguous]['readings']}); a "
                   "better parse cannot settle what the words do not")

    if model is None:
        if not syntax:
            return Decision(dimension=dimension, outcome=Fusion.DISAGREE,
                            material=requirement.material,
                            detail="no reader answered")
        # Syntax alone. Never carries a field, however strong — the whole
        # reason this module exists.
        return Decision(
            dimension=dimension, outcome=Fusion.DISAGREE,
            material=requirement.material, syntax=syntax,
            detail="only syntax proposed a value, and syntax is evidence "
                   "rather than authority")

    if requirement.binds and not bound:
        return Decision(
            dimension=dimension, outcome=Fusion.INSUFFICIENT_RELATION,
            material=requirement.material, model=model, syntax=syntax,
            detail=f"the value needs a {requirement.binds} it was not given; "
                   "three ratios and three accounts are not an allocation "
                   "until something says which belongs to which")

    against = [e for e in syntax
               if contradicts(e, model, requirement.compare_as)]
    if against:
        named = "; ".join(f"{e.proposed_value!r} scored {e.score:+d}"
                          for e in against)
        return Decision(
            dimension=dimension, outcome=Fusion.DISAGREE,
            material=requirement.material, model=model, syntax=syntax,
            detail=f"the model read {model.value!r}; syntax argues otherwise "
                   f"({named}). Not resolved by score: a parser can be "
                   "confident and wrong")

    return Decision(dimension=dimension, outcome=Fusion.AGREE,
                    value=model.value, material=requirement.material,
                    model=model, syntax=syntax,
                    detail="syntax supports the reading" if syntax
                           else "the model read it and syntax was silent")


def fuse_with_bindings(dimension: str, value, *, bindings: Sequence[Any],
                       model: Optional[Proposal] = None,
                       syntax: Sequence[SyntaxEvidence] = (),
                       requirement: Optional[Requirement] = None) -> Decision:
    """`fuse`, with `bound` answered by a real binder rather than a caller.

    The whole seam in one function: `binding.is_bound` reads structure and
    returns a boolean, `fuse` reads the boolean and decides. Neither imports
    the other's judgement — fusion still cannot see a parse, and the binder
    still cannot see an outcome.

    `value` is the normalised `Value` a binding would be about. Passing the
    value rather than its id keeps the identity function in one place; two
    modules computing an id separately is how a lookup starts silently missing.
    """
    from .binding import is_bound

    requirement = requirement or REQUIREMENTS.get(dimension, Requirement())
    established = (True if not requirement.binds
                   else bool(value is not None and is_bound(bindings, value)))
    return fuse(dimension, model=model, syntax=syntax,
                requirement=requirement, bound=established)


def _ambiguity(dimension: str, model: Optional[Proposal],
               syntax: Sequence[SyntaxEvidence],
               available: Sequence[str] = ()) -> Optional[str]:
    """Whether an attested-ambiguous term genuinely leaves this open.

    Two conditions, and the second is what stops the outcome firing on its own
    ontology. The term must appear in the *user's words* — never the dimension
    name, never the reader's paraphrase. And the dimension being decided must
    be one of the fields the ambiguity is between, with at least one of the
    others also proposed: an ambiguity nobody could have resolved differently
    in this sentence is not an ambiguity.
    """
    # The *user's words*, never the dimension name and never the reader's
    # paraphrase of them. Ambiguity is a property of the language someone
    # wrote, not of how a reader labelled it — and with the dimension name in
    # here, `periodic_rebalancing` matched "rebalance" on every reading it ever
    # produced, which is a check that fires on its own subject.
    haystack = " ".join(filter(None, [
        "" if model is None else model.source_span,
        " ".join(e.source_span for e in syntax)])).lower()
    proposed = set(available)
    for term, record in AMBIGUOUS_TERMS.items():
        if term not in haystack:
            continue
        between = set(record.get("between", ()))
        if between and dimension not in between:
            continue
        if between and not (between - {dimension}) & proposed:
            # The competing reading is not on the table in this sentence.
            continue
        return term
    return None


@dataclass
class FusionReport:
    """Every decision for one utterance, and what it means for sealing."""

    decisions: Sequence[Decision] = field(default_factory=tuple)

    @property
    def settled(self) -> Sequence[Decision]:
        return tuple(d for d in self.decisions if d.proceeds)

    @property
    def open(self) -> Sequence[Decision]:
        return tuple(d for d in self.decisions if not d.proceeds)

    @property
    def blocks_sealing(self) -> Sequence[Decision]:
        """Open *and* material. These are what `seal()` will refuse on, and the
        reason `Unresolved.result_changing` is not a free-text note."""
        return tuple(d for d in self.open if d.material)

    def unresolved_for_contract(self) -> list:
        """The open decisions, in the shape `VerifiedIntent` takes.

        Mapped rather than invented: `AMBIGUOUS_BY_LANGUAGE` becomes
        `NOT_ASKED` because nobody has yet put the question to the user, and
        everything else becomes `UNRESOLVED_DISAGREEMENT`, which is the one
        open state a consumer must not proceed through.
        """
        from runtime_contracts import OpenReason, Unresolved

        return [Unresolved(
            dimension=d.dimension,
            reason=(OpenReason.NOT_ASKED
                    if d.outcome is Fusion.AMBIGUOUS_BY_LANGUAGE
                    else OpenReason.UNRESOLVED_DISAGREEMENT),
            detail=d.detail,
            result_changing=d.material) for d in self.open]
