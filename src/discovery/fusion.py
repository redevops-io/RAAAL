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

from dataclasses import dataclass, field
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
AMBIGUOUS_TERMS: Mapping[str, Mapping[str, str]] = {
    "rebalance": {
        "readings": "rebalance back to target | change the target allocation",
        "evidence": "Bogleheads thread 'Don't Know How To Rebalance/Reallocate'",
        "source": "https://www.bogleheads.org/forum/viewtopic.php?t=459742"},
    "reallocate": {
        "readings": "rebalance back to target | change the target allocation",
        "evidence": "same thread; the two verbs are used interchangeably",
        "source": "https://www.bogleheads.org/forum/viewtopic.php?t=459742"},
}


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


#: Declared per dimension rather than inferred. A dimension absent here is
#: material and unbound — the conservative reading, since treating an unknown
#: dimension as immaterial would let anything new proceed unexamined.
REQUIREMENTS: Mapping[str, Requirement] = {
    "cadence": Requirement(material=True),
    "amount": Requirement(material=True),
    "assets": Requirement(material=True),
    "allocation_method": Requirement(material=True),
    "moving_average_window": Requirement(material=True),
    "trigger_semantics": Requirement(material=True),
    "execution_timing": Requirement(material=True),
    "day_rule": Requirement(material=False),
    "account_allocation": Requirement(material=True, binds="account"),
    "dividend_policy": Requirement(material=False),
}


def contradicts(evidence: SyntaxEvidence, proposal: Proposal) -> bool:
    """Whether this evidence argues against the model's value.

    Two ways, and both are about the *sign* of the score rather than its size.
    A negative score on the model's own value is syntax saying "not here"; a
    positive score on a different value is syntax saying "there instead".
    Magnitude is deliberately unused — the `year end` case scored confidently
    and wrongly, so size is not a signal about correctness.
    """
    same_value = str(evidence.proposed_value) == str(proposal.value)
    return (evidence.score < 0) if same_value else (evidence.score > 0)


def fuse(dimension: str, *, model: Optional[Proposal] = None,
         syntax: Sequence[SyntaxEvidence] = (),
         requirement: Optional[Requirement] = None,
         bound: bool = False) -> Decision:
    """One dimension, one decision.

    `bound` says whether the relation this dimension requires has actually been
    supplied. Passed in rather than inspected, because whether a ratio is bound
    to an account is something the reader above knows and this module cannot
    see from a value.
    """
    requirement = requirement or REQUIREMENTS.get(dimension, Requirement())
    syntax = tuple(syntax)

    ambiguous = _ambiguity(dimension, model, syntax)
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

    against = [e for e in syntax if contradicts(e, model)]
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


def _ambiguity(dimension: str, model: Optional[Proposal],
               syntax: Sequence[SyntaxEvidence]) -> Optional[str]:
    """Whether an attested-ambiguous term is what this decision rests on."""
    haystack = " ".join(filter(None, [
        dimension,
        "" if model is None else f"{model.value} {model.source_span}",
        " ".join(e.source_span for e in syntax)])).lower()
    for term in AMBIGUOUS_TERMS:
        if term in haystack:
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
