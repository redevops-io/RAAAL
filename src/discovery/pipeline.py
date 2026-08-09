"""Both witnesses, neither privileged.

                     ┌─ normalise → bind → derive ─┐
    utterance ───────┤                             ├─ fusion → field | Unresolved
                     └─ hosted reader ─────────────┘

The shape matters more than the code. The hosted reader runs on **every**
utterance, not only when the deterministic path comes up empty, because a
fallback is not independent evidence: a reader consulted only when the other is
silent inherits the other's authority wherever the other speaks. The lesson this
whole project is built on is that one reader being quiet must never grant the
other the right to decide.

So both paths produce proposals for the same dimensions, and `fuse` sees them
together. What follows from that:

**The model proposes; syntax argues.** `fuse` takes one model proposal and any
amount of syntax evidence, and the value that proceeds is always the model's.
The deterministic path's candidates are therefore evidence *about* the model's
reading rather than a competing reading — supporting it when they agree,
contradicting it when they do not, and never replacing it.

**A missing witness is not agreement.** When the model has no reading for a
dimension the deterministic path proposed, that is `DISAGREE` and not `AGREE`:
syntax alone never carries a field. When the model proposes and the
deterministic path is silent, that is `AGREE` — silence is not an argument.
Those two are asymmetric on purpose, and the asymmetry is the whole policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .binding import RelationBinding, bind
from .fusion import Decision, Proposal, fuse
from .reader import ReadingSet, Schema
from .semantics import SemanticCandidate, propose
from .syntax import Parse, SyntaxEvidence, Value, normalize

PIPELINE_VERSION = "quantify-pipeline@1"


@dataclass(frozen=True)
class Read:
    """Everything both witnesses said about one utterance, and what was decided."""

    text: str
    values: Sequence[Value] = ()
    bindings: Sequence[RelationBinding] = ()
    candidates: Sequence[SemanticCandidate] = ()
    model: Optional[ReadingSet] = None
    decisions: Sequence[Decision] = ()
    pipeline_version: str = PIPELINE_VERSION

    @property
    def by_field(self) -> Mapping[str, Decision]:
        return {d.dimension: d for d in self.decisions}

    @property
    def settled(self) -> Mapping[str, Any]:
        return {d.dimension: d.value for d in self.decisions if d.proceeds}

    @property
    def open(self) -> Sequence[Decision]:
        return tuple(d for d in self.decisions if not d.proceeds)

    @property
    def intermediate(self) -> Sequence[SemanticCandidate]:
        """Semantics this pipeline computed that the contract does not carry.

        Kept and reported rather than dropped: `amount_kind=fixed` is a real
        reading of a real sentence, and discarding it because no contract field
        exists would lose the evidence that the contract might need one."""
        return tuple(c for c in self.candidates if not c.is_contract_field)


#: What a deterministic candidate asserts, as a score. Always the same number.
#:
#: A positive score means "syntax asserts this value", and `fuse.contradicts`
#: does the rest: asserting the model's value supports it, asserting a different
#: one argues against it. So the sign carries everything and the magnitude
#: carries nothing — which is the point. The `year end` case showed a parser
#: being confident and wrong, so a bigger number would only have meant "more
#: certainly wrong", and leaving a magnitude here is leaving something for a
#: future rule to read.
ASSERTS = 1


def as_evidence(candidate: SemanticCandidate) -> SyntaxEvidence:
    """A deterministic candidate, restated as evidence about a model reading."""
    return SyntaxEvidence(
        dimension=candidate.field, proposed_value=candidate.value,
        score=ASSERTS, features=tuple(candidate.evidence),
        source_span=candidate.source_span)


def read(text: str, parse: Parse, model_reading: ReadingSet, schema: Schema,
         *, language: str = "en") -> Read:
    """One utterance through both paths and into fusion.

    `parse` and `model_reading` are passed in rather than produced here, so the
    same function serves a live run and a replay of recordings. That is not a
    convenience: a pipeline that fetched its own witnesses could not be tested
    without them, and a test that cannot run without a provider is a test
    nobody runs.
    """
    from .hosted_recording import proposals

    values = normalize(text, language)
    bindings = bind(parse, values)
    candidates = propose(bindings, values)

    # Only contract fields enter fusion. An intermediate — `amount_kind`,
    # `holding_period_days`, `account_allocation` — has no contract field for
    # the other witness to answer with, so fusing it would report DISAGREE
    # against a silence that could never have been anything else. They are
    # carried on the Read as computed semantics, not as decisions.
    from .semantics import INTERMEDIATE_FIELDS

    model_by_field = {p.dimension: p for p in proposals(model_reading)}
    fields = sorted((set(model_by_field)
                     | {c.field for c in candidates if c.is_contract_field}))

    decisions = []
    for name in fields:
        proposal = model_by_field.get(name)
        supporting = [as_evidence(c) for c in candidates if c.field == name]
        decisions.append(fuse(name, model=proposal, syntax=supporting,
                              available=fields))

    return Read(text=text, values=values, bindings=bindings,
                candidates=candidates, model=model_reading,
                decisions=tuple(decisions))
