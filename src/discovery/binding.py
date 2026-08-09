"""Who does this value modify? — typed bindings, and nothing semantic.

The gap Phase 5 left is concrete. Fusion can say a reading is untrustworthy; it
cannot say *what a value belongs to*, and it must not learn how, or it becomes
the second parser this architecture exists to avoid.

    401k (50/50), Roth IRA (85/15), taxable brokerage (70/30)

Normalisation gives three ratios. Fusion can only refuse them. What is missing
is a producer of the relationship:

    50/50 → 401k        85/15 → Roth IRA        70/30 → taxable brokerage

or, when the structure does not establish it, the honest alternative — a ratio
detected, a binding not established, and `INSUFFICIENT_RELATION` downstream.

**This module emits structure, never meaning.** A binding says a value is the
appositive of the token `401k`. It does not say that is an account, or that the
ratio is its allocation. Each rule declares which semantic pairing it is
*evidence for*, and the declaration lives beside the rule rather than inside the
output, so a consumer reading a binding cannot mistake a structural fact for a
settled field.

**Three statuses, and the middle one is the point.**

    BOUND       exactly one candidate, by a declared strategy
    AMBIGUOUS   more than one, and structure does not choose between them
    UNBOUND     none

`AMBIGUOUS` is not `UNBOUND` with extra steps. A value with two candidate
targets is a value where picking either is a coin toss dressed as a reading,
and this project has spent its whole history removing those. Both become
`INSUFFICIENT_RELATION` at the fusion boundary today; they are kept apart
because the repairs differ — one needs a rule, the other needs a question.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence

from .syntax import Aligned, Parse, Sentence, Token, Value, align

#: Bumped when a strategy's *meaning* changes, so a stored binding can be told
#: apart from one produced under different rules.
BINDING_VERSION = "quantify-binding@1"


class BindingStatus(str, Enum):
    BOUND = "BOUND"
    AMBIGUOUS = "AMBIGUOUS"
    UNBOUND = "UNBOUND"

    @property
    def established(self) -> bool:
        return self is BindingStatus.BOUND


def value_id(value: Value) -> str:
    """A stable name for one normalised value.

    Derived from kind and span rather than generated, so the same text produces
    the same id on every run — a binding that referred to a fresh id each time
    could not be stored beside an intent and read back.
    """
    return f"{value.kind}@{value.start_char}-{value.end_char}"


@dataclass(frozen=True)
class RelationBinding:
    """One value, and what the structure attaches it to."""

    value_id: str
    relation: str
    """Structural, not semantic: `appositive_of`, `shares_head_with`,
    `governed_by`. What that *means* is the schema's business."""

    status: BindingStatus
    target_span: str = ""
    target_index: Optional[int] = None
    candidates: Sequence[str] = ()
    """Every target the strategy found. One when BOUND, several when AMBIGUOUS,
    none when UNBOUND — carried in all three cases so a reader can see what the
    binder was choosing between rather than only what it chose."""

    evidence: Sequence[str] = ()
    """The dependency edges that justified it, as `token:relation->head`. This
    is what makes a binding checkable a year later; without it a binding is an
    assertion."""

    target_lemma: str = ""
    """The head's lemma, unrebuilt. `target_span` is for a person to read —
    "put in", "401k" — and matching a verb table against it fails on exactly
    the multiword cases, so the lemma is carried separately for code."""

    modifiers: Sequence[str] = ()
    """Lemmas of the words modifying the value's own phrase — `fixed` in "a
    fixed $500", `last` in "the last session".

    Carried by the binder because a derivation needs this evidence and must not
    read a parse to get it. Still structure and still no meaning: the binder
    reports that `fixed` modifies the amount, and says nothing about what that
    implies for `amount_kind`."""

    target_modifiers: Sequence[str] = ()
    """Lemmas of the words modifying the *target*, as distinct from the value.

    `10% of my salary` puts the `of` on `salary`, not on the percentage — so a
    derivation looking only at the value's own modifiers never saw it, and the
    proportional reading never fired. Which side a marker sits on is a fact
    about the parse rather than about the meaning, so the binder reports both
    and the mapper decides which it needs."""

    supports: str = ""
    """The semantic pairing this is evidence *for* — declared by the rule, not
    concluded by the binder."""

    binding_version: str = BINDING_VERSION

    @property
    def established(self) -> bool:
        return self.status.established

    def to_json(self) -> dict:
        return {"value_id": self.value_id, "relation": self.relation,
                "status": self.status.value, "target_span": self.target_span,
                "target_lemma": self.target_lemma,
                "modifiers": list(self.modifiers),
                "target_modifiers": list(self.target_modifiers),
                "target_index": self.target_index,
                "candidates": list(self.candidates),
                "evidence": list(self.evidence), "supports": self.supports,
                "binding_version": self.binding_version}


@dataclass(frozen=True)
class BindingRule:
    """A structural strategy, and what it is evidence for.

    Deliberately narrow. These five are where this project has repeatedly
    attached one dimension to the wrong neighbour, and a sixth invented ahead of
    a failure would be a rule tuned on nothing.
    """

    relation: str
    applies_to: frozenset
    strategy: str
    supports: str
    """`account↔allocation`, `asset↔weight`, `cadence↔action`. Read by people,
    never matched on."""


RULES: Sequence[BindingRule] = (
    BindingRule(relation="appositive_of",
                applies_to=frozenset({"ratio", "percentage"}),
                strategy="appositive_head", supports="account↔allocation"),
    BindingRule(relation="shares_head_with",
                applies_to=frozenset({"percentage", "ratio"}),
                strategy="shared_head", supports="asset↔weight"),
    BindingRule(relation="governed_by",
                applies_to=frozenset({"duration", "moving_average_window",
                                      "money", "cadence", "residual"}),
                strategy="governing_verb",
                supports="cadence↔action | condition↔action | timing↔action"),
)

#: Universal Dependencies labels for a verbal governor. `governing_verb` walks
#: past nominal heads to reach one, because "invest five hundred a month" puts
#: a noun between the value and the verb in most parses.
_VERBAL = frozenset({"VERB", "AUX"})


def _edge(sentence: Sentence, token: Token) -> str:
    head = sentence.head_of(token)
    return f"{token.text}:{token.relation}->{head.text if head else 'ROOT'}"


#: Words that are part of a head's own name rather than separate participants.
_NAME_PARTS = frozenset({"compound", "amod", "nummod", "det:predet", "flat"})

#: Relations that mark a word as *describing* the value's phrase.
_MODIFYING = frozenset({"amod", "advmod", "det", "nmod:poss", "case", "mark",
                        "compound", "cop", "nsubj"})


def modifiers_of(aligned: Aligned) -> Sequence[str]:
    """Lemmas describing the value's own phrase, and the words introducing it.

    "a fixed $500" gives `fixed`; "the last session" gives `last`; "when it
    crosses" gives `when`. A derivation needs these and must not open a parse to
    find them, so the binder reports them — as lemmas, with no claim about what
    any of them means.
    """
    covered = {token.index for token in aligned.tokens}
    found = []
    for token in aligned.sentence.tokens:
        if token.index in covered:
            continue
        if token.head in covered and token.relation.split(":")[0] in _MODIFYING:
            found.append(token.lemma)
    # And the words attached to the value's own head, which is where `when`,
    # `whenever` and the copula sit in a condition clause.
    for token in aligned.tokens:
        head = aligned.sentence.head_of(token)
        if head is None:
            continue
        for sibling in aligned.sentence.tokens:
            if (sibling.head == head.index and sibling.index not in covered
                    and sibling.relation.split(":")[0] in _MODIFYING):
                found.append(sibling.lemma)
    return tuple(sorted(set(found)))


def phrase_of(sentence: Sentence, head: Token) -> str:
    """The head plus the words that name it, in text order.

    Without this a binding reports its target as `k`, because Stanza splits
    `401k` into `401` and `k` and makes `k` the head. `k` is structurally
    correct and useless to anybody reading the binding — and a target nobody
    can identify is a binding nobody can check.
    """
    parts = [head] + [token for token in sentence.tokens
                      if token.head == head.index
                      and token.relation.split(":")[0] in _NAME_PARTS]
    parts.sort(key=lambda token: token.start_char)

    # Rejoined by character offset rather than with spaces. `401` and `k` are
    # adjacent in the source and `Roth` and `IRA` are not, and only the offsets
    # know which — a space-joined version needed a special case for `k`, which
    # is the shape of a rule that will be wrong on the next abbreviation.
    rebuilt = parts[0].text
    for previous, token in zip(parts, parts[1:]):
        rebuilt += ("" if token.start_char == previous.end_char else " ")
        rebuilt += token.text
    return rebuilt


def modifiers_of_target(aligned: Aligned, target: Token) -> Sequence[str]:
    """Lemmas describing the bound target. Same shape, other side."""
    covered = {token.index for token in aligned.tokens}
    return tuple(sorted({
        token.lemma for token in aligned.sentence.tokens
        if token.head == target.index and token.index not in covered
        and token.relation.split(":")[0] in _MODIFYING}))


def _appositive_head(aligned: Aligned) -> tuple:
    """A value carried as an appositive binds to the noun it renames.

    `401k (50/50)` parses with `50/50` as `appos` of `401k`. Every token the
    value covers is checked, not only the anchor: Stanza splits `85/15` into
    three, and the `appos` edge sits on the first of them.
    """
    targets, evidence = [], []
    for token in aligned.tokens:
        if token.relation.split(":")[0] != "appos":
            continue
        head = aligned.sentence.head_of(token)
        if head is not None:
            targets.append(head)
            evidence.append(_edge(aligned.sentence, token))
    return targets, evidence


#: Relations that mark a *participant* hanging off a value's phrase, rather
#: than a word that is part of that phrase.
#:
#: `compound` and `amod` are excluded because `Roth` in `Roth IRA` and `taxable`
#: in `taxable brokerage` belong to the head's own name — binding a ratio to
#: `Roth` would be binding it to half of its own target. `conj` is excluded
#: because a conjunct is a sibling of the head, not a participant of the value:
#: with it in, `50/50` matched both `Roth IRA` and `taxable brokerage`.
_PARTICIPANT = frozenset({"nmod", "obl", "obj", "nsubj"})


def _shared_head(aligned: Aligned) -> tuple:
    """A weight binds to the noun hanging off its own phrase.

    `60% to VTI` parses with `60` as `nummod` of a `%` and `VTI` as `nmod` of
    that same `%`, so the target is a *dependent of a token the value covers*.

    Written that way round rather than "shares a head with", which was the first
    version and was wrong on the sentence that matters: `40%` covers the second
    `%`, whose own head is the first `%` by `conj`, so a head-set built from
    covered tokens reached across the coordination and matched `VTI` as well as
    `BND`. Looking downwards instead of upwards cannot cross that boundary.
    """
    covered = {token.index for token in aligned.tokens}
    targets, evidence = [], []
    for token in aligned.sentence.tokens:
        if token.index in covered or token.head not in covered:
            continue
        if token.relation.split(":")[0] not in _PARTICIPANT:
            continue
        if token.upos not in {"PROPN", "NOUN"}:
            continue
        targets.append(token)
        evidence.append(_edge(aligned.sentence, token))
    return targets, evidence


def _governing_verb(aligned: Aligned) -> tuple:
    """The nearest verbal governor, walking past nominal heads."""
    anchor = aligned.anchor
    if anchor is None:
        return [], []
    for governor in aligned.sentence.governor_chain(anchor):
        if governor.upos in _VERBAL:
            return [governor], [_edge(aligned.sentence, anchor)]
    return [], []


_STRATEGIES = {"appositive_head": _appositive_head,
               "shared_head": _shared_head,
               "governing_verb": _governing_verb}


def bind_value(aligned: Aligned, rule: BindingRule) -> RelationBinding:
    """One value, one rule, one binding — established or honestly not."""
    targets, evidence = _STRATEGIES[rule.strategy](aligned)
    spans = [phrase_of(aligned.sentence, token) for token in targets]
    identity = value_id(aligned.value)

    if not targets:
        return RelationBinding(value_id=identity, relation=rule.relation,
                               status=BindingStatus.UNBOUND,
                               modifiers=modifiers_of(aligned),
                               supports=rule.supports)
    modifiers = modifiers_of(aligned)
    if len(targets) > 1:
        # Not resolved by position. "the first candidate" is a coin toss with a
        # rule's name on it, and the sentences this binder exists for are
        # exactly the ones with several candidates.
        return RelationBinding(value_id=identity, relation=rule.relation,
                               status=BindingStatus.AMBIGUOUS,
                               candidates=tuple(spans), evidence=tuple(evidence),
                               modifiers=modifiers, supports=rule.supports)
    return RelationBinding(value_id=identity, relation=rule.relation,
                           status=BindingStatus.BOUND,
                           target_span=spans[0], target_index=targets[0].index,
                           target_lemma=targets[0].lemma,
                           target_modifiers=modifiers_of_target(aligned,
                                                                targets[0]),
                           candidates=tuple(spans), evidence=tuple(evidence),
                           modifiers=modifiers, supports=rule.supports)


def bind(parse: Parse, values: Sequence[Value],
         rules: Sequence[BindingRule] = RULES) -> Sequence[RelationBinding]:
    """Every binding the declared rules establish over one parsed sentence.

    A value matching no rule produces no binding at all, rather than an
    `UNBOUND` one — "no rule applies here" and "a rule applied and found
    nothing" are different facts, and collapsing them would make an unwritten
    rule look like a searched-for absence.
    """
    bindings = []
    for aligned in align(parse, values):
        for rule in rules:
            if aligned.value.kind in rule.applies_to:
                binding = bind_value(aligned, rule)
                if binding.status is not BindingStatus.UNBOUND or len(
                        [r for r in rules
                         if aligned.value.kind in r.applies_to]) == 1:
                    bindings.append(binding)
    return tuple(bindings)


def is_bound(bindings: Sequence[RelationBinding], value: Value,
             relation: Optional[str] = None) -> bool:
    """Whether anything established a binding for this value.

    The predicate fusion's `bound=` takes. Kept here rather than in `fusion.py`
    so that fusion never inspects a parse — the separation is the point:
    fusion decides whether a reading proceeds, this decides what it attaches
    to, and neither does the other's job.
    """
    identity = value_id(value)
    return any(b.established and b.value_id == identity
               and (relation is None or b.relation == relation)
               for b in bindings)


def summary(bindings: Sequence[RelationBinding]) -> Mapping[str, int]:
    counts: dict = {}
    for binding in bindings:
        counts[binding.status.value] = counts.get(binding.status.value, 0) + 1
    return counts
