"""Readers: anything that turns a sentence into readings of dimensions.

**Provider-neutral from the first commit.** The Phase 3 decision names
`claude-sonnet-5` as the reader to evaluate with, and that decision is about
which reader is *known good enough not to be the variable under test* — not
about making one vendor structural. Everything here is written to the interface
so a local model, a challenger provider or a router is a later measurement
rather than a rewrite.

    read(text, schema) -> Sequence[Reading]

That is the whole contract. A reader does not build a `VerifiedIntent`, decide
what is material, resolve a disagreement or know what the engine can execute.
It reads, says where it read it, and says how sure it is.

**Readers are not ranked.** There is deliberately no precedence anywhere in
this package. Encoding "the model wins" or "the rule wins" is how one reader's
blind spot becomes the system's answer, and this project already has the
counter-example in both directions: the deterministic parser was wrong about
"crosses below" where the model was right, and the model has been wrong about
things the parser had exactly correct.

This module is written to move to `agentic_os/discovery/` unchanged. It imports
nothing from Quantify.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Any, Dict, Mapping, Optional, Protocol, Sequence,
                    runtime_checkable)


@dataclass(frozen=True)
class Reading:
    """One reader's view of one dimension of meaning."""

    dimension: str
    value: Any
    confidence: str = "1"
    """Decimal string. Floats hash differently across languages, and these
    readings end up inside a content-hashed artifact."""

    source_span: str = ""
    """The user's own words that carried it. Empty is legitimate — a default
    or a prior has no span — and is why `author` is a separate question."""

    note: str = ""
    """Why the reader thinks so, when it can say. Never parsed."""


@dataclass(frozen=True)
class RelationReading:
    """One reader's view of one relationship.

    Compared structurally, never as a string. Two readers can ground the same
    relation in slightly different text and still mean the same thing, so
    `source_span` sits outside the equality operation — the same rule already
    applied to `evaluation_period`.
    """

    kind: str
    members: Sequence[tuple] = ()
    """(role, subject, qualifiers) in the order read."""

    attributes: Mapping[str, str] = field(default_factory=dict)
    source_span: str = ""

    def comparable(self, ordered: bool = True):
        """What two readers must match on: kind, roles, subjects, qualifiers
        and attributes. Not spans, not confidence, not which reader."""
        members = tuple(
            (role.strip().lower(), str(subject).strip().lower(),
             tuple(sorted((k.strip().lower(), str(v).strip().lower())
                          for k, v in (quals or {}).items())))
            for role, subject, quals in self.members)
        if not ordered:
            members = tuple(sorted(members))
        return (self.kind.strip().lower(), members,
                tuple(sorted((k.strip().lower(), str(v).strip().lower())
                             for k, v in self.attributes.items())))

    def to_json(self):
        return {"kind": self.kind,
                "members": [{"role": r, "subject": s, "qualifiers": q}
                            for r, s, q in self.members],
                "attributes": dict(self.attributes),
                "source_span": self.source_span}


@dataclass(frozen=True)
class ReadingSet:
    """Everything one reader saw in one sentence, plus what it could not."""

    reader_id: str
    readings: Sequence[Reading] = ()
    relations: Sequence[RelationReading] = ()
    unread: Sequence[str] = ()
    """Dimensions the reader was asked about and declined to answer.

    Distinct from a dimension it did not mention: "I looked and the sentence
    does not say" is a reading, and "I was never asked" is not. Collapsing them
    makes a silent reader look like a confident one."""

    failed: Optional[str] = None
    """Set when the reader itself failed — a timeout, a refusal, malformed
    output. **Not** a reading of the sentence, and never counted as agreement
    or disagreement: an evaluator that scores its own transport failure as a
    product result manufactures evidence."""

    @property
    def ok(self) -> bool:
        return self.failed is None

    def relation_of(self, kind: str) -> Optional["RelationReading"]:
        for one in self.relations:
            if one.kind == kind:
                return one
        return None

    def value_of(self, dimension: str) -> Optional[Reading]:
        for one in self.readings:
            if one.dimension == dimension:
                return one
        return None

    @property
    def dimensions(self) -> frozenset:
        return frozenset(r.dimension for r in self.readings)


@runtime_checkable
class DiscoveryReader(Protocol):
    """The one thing every reader is."""

    id: str
    """Stable and versioned — `quantify-compiler@1`, `claude-sonnet-5@1`. This
    becomes `produced_by` on the evidence, so a replay that diverges after an
    upgrade can be traced to the reader that changed."""

    def read(self, text: str, schema: "Schema") -> ReadingSet:
        ...


@dataclass(frozen=True)
class Dimension:
    """A dimension of meaning a reader may be asked about.

    Carries a name, a description and — where the concept genuinely has a
    closed vocabulary — the values that *mean different things*. It does not
    carry what the engine can execute. That distinction is the whole reason
    this type exists separately from the capability manifest:

        schema      what can be meant
        manifest    what can be run

    A reader told only what can be run does not refuse the rest; it renders the
    rest as the nearest runnable thing, and "by inverse volatility" comes back
    as an equal split. The schema therefore lists `inverse_volatility` as a
    perfectly sayable allocation method, and Mission refuses it later, by name.
    """

    name: str
    describes: str
    compare_as: str = "TEXT"
    """How two readers' values for this dimension are the same value.

    A property of the dimension's type, declared here rather than guessed by
    the comparator. The first shadow run over 35 prompts produced 26 contested
    fields of which roughly three were disagreements about meaning; the rest
    were `"VTI, BND"` against `"VTI and BND"`, and `"200"` against `"200-day"`.
    A comparator that reports those as conflict buries the real ones.

        TEXT     compared as written, after stripping. The default, because
                 assuming two spellings mean the same thing is the failure
                 this whole project is about.
        NUMBER   compared as numbers, after removing currency symbols,
                 separators and unit suffixes. A type coercion: it cannot make
                 two different amounts equal.
        SET      compared as an unordered set of tokens. "VTI and BND" and
                 "VTI, BND" name the same two instruments; which conjunction a
                 reader used is not a reading.

    Note what is deliberately absent: no synonym mode. "annual" against
    "yearly" stays contested, because resolving it needs a table nobody can
    audit, and a dimension that needs one is a dimension whose vocabulary is
    wrong."""

    values: Sequence[str] = ()
    """Closed only where the *concept* is closed. `cadence` is: a contribution
    is weekly or monthly or annual and there is no continuum. `amount` is not.
    Where this is non-empty it is a vocabulary, never a permission."""

    examples: Sequence[str] = ()


@dataclass(frozen=True)
class RelationSpec:
    """A relationship a reader may be asked to report.

    The rule for when a dimension should become one of these, so that
    `VerifiedIntent` does not drift into being a graph language:

        Use a relation when meaning depends on which value belongs to which
        participant, or on direction or order between participants.

    By that rule `"60% VTI / 40% BND"` can stay a scalar pair — the mapping is
    unambiguous. `"a core fund plus a 30% satellite"` cannot, because the
    qualifier belongs to one particular member. `"traditional IRA to a Roth"`
    cannot, because the direction *is* the meaning.
    """

    kind: str
    describes: str
    roles: Sequence[str] = ()
    """Every role a member may play."""

    required_roles: Sequence[str] = ()
    """Roles without which the relation is not the thing it claims to be. A
    transition with no `to` is not a transition."""

    repeatable_roles: Sequence[str] = ()
    """Roles that may appear more than once. Three satellites is a portfolio;
    two `from` accounts is not a transfer."""

    qualifiers: Mapping[str, str] = field(default_factory=dict)
    """Per-member facts, name -> what it means. `allocation` on a sleeve."""

    attributes: Mapping[str, str] = field(default_factory=dict)
    """Whole-relation facts. `action` on a transition, which belongs to
    neither end."""

    ordered: bool = True
    """Whether member order carries meaning. True for anything directional:
    `from`/`to` reversed is the opposite transaction."""

    examples: Sequence[str] = ()


@dataclass(frozen=True)
class Schema:
    """What a reader is asked to look for."""

    dimensions: Sequence[Dimension] = ()
    relations: Sequence[RelationSpec] = ()
    version: str = "discovery-schema@1"

    def dimension(self, name: str) -> Optional[Dimension]:
        for one in self.dimensions:
            if one.name == name:
                return one
        return None

    def relation(self, kind: str) -> Optional[RelationSpec]:
        for one in self.relations:
            if one.kind == kind:
                return one
        return None

    @property
    def names(self) -> frozenset:
        return frozenset(d.name for d in self.dimensions)

    @property
    def relation_kinds(self) -> frozenset:
        return frozenset(r.kind for r in self.relations)

    def semantic_surface(self) -> Dict[str, Any]:
        """Everything capable of changing what a reader can be asked or say.

        This is what the run fingerprint hashes. It deliberately covers the
        relation surface as well as the dimensions: without it, schema@2 could
        add two relation kinds — materially changing the instrument — and
        present the same digest as schema@1, so two incomparable runs would
        look like one experiment.

        Prose descriptions are excluded. Rewording what a model is told is a
        real change and the run records it through the model's own readings;
        hashing it would invalidate every baseline over a typo.
        """
        return {
            "version": self.version,
            "dimensions": {
                d.name: {"compare_as": d.compare_as, "values": list(d.values)}
                for d in self.dimensions},
            "relations": {
                r.kind: {"roles": list(r.roles),
                         "required_roles": list(r.required_roles),
                         "repeatable_roles": list(r.repeatable_roles),
                         "qualifiers": sorted(r.qualifiers),
                         "attributes": sorted(r.attributes),
                         "ordered": r.ordered}
                for r in self.relations},
        }
