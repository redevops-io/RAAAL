"""VerifiedIntent — the Discovery → Mission boundary.

    Discovery answers "what did the human mean?".
    Mission answers "can I execute exactly that?".

`VerifiedIntent` is the artifact between them, and it is the centre of the
split:

    Everything above creates it.  Everything below consumes it.
    Nothing below may rewrite it.

That third clause is the whole discipline. When an execution engine cannot
honour a field, the answer is a refusal naming the field — never an intent
quietly adjusted until it happens to be executable. An engine that may edit the
intent to suit itself can always produce a figure, and the figure will describe
a plan nobody asked for.

**Why this is a contract and not an application type.** Two runtimes exchange
it, so neither may own it. The failure it prevents is the one where the
consumer's limits leak into the producer's vocabulary: a reader that cannot
*express* "inverse volatility" does not refuse it, it renders it as the nearest
thing it can say. Discovery is therefore free to understand more than any
Mission can execute, and the contract has to be wide enough to carry that.

**Author and producer are different questions.**

    author       who asserted this value          USER · READER · MODEL · POLICY · DEFAULT
    produced_by  which runtime version produced   discovery-runtime@0.4.2
                 the artifact

A user-authored value elicited by ``discovery-runtime@0.4.2`` and the same value
elicited by ``@0.5.0`` are one assertion by one author and two elicitations.
Without the second field, a replay that diverges after an upgrade is
undiagnosable, and a migration cannot find the intents produced by a version
with a known reading bug.

**What participates in identity.** Values, authors, unresolved dimensions and
amendments. Not confidence, not source spans, not evidence, not ``produced_by``
— by the same rule ``EvidenceCandidate`` already applies to retrieval scores:
re-reading changes how confidently a value was reached, it does not change what
was asserted. Two runs that asserted the same values with the same authors are
the same intent, and pinning them apart on a confidence float would make every
model upgrade look like a different request.

**Absent, unresolved and refused are three states.** A dimension nobody asked
about, one the user deliberately left open, and one the user ruled out are
different facts, and a consumer that cannot tell them apart will invent a
default for at least one of them. `unresolved` carries the second explicitly;
the third is a stated value like any other.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

from .canonical import CONTRACT_VERSION, content_hash, decimal_string


class Author(str, Enum):
    """Who asserted a value. Ordered by authority, weakest first."""

    DEFAULT = "DEFAULT"
    """Nobody asserted it; a declared default applied. The value a consumer is
    most entitled to question, and the one most often mistaken for a choice."""

    MODEL = "MODEL"
    """A model's reading of the user's words."""

    READER = "READER"
    """A deterministic reader — a rule, a grammar, a lookup."""

    POLICY = "POLICY"
    """A policy supplied it; the user had no say and should be told so."""

    USER = "USER"
    """The user said it, or settled it when asked. Dominates every other
    author and is never overwritten by a re-read."""

    @property
    def is_user_authored(self) -> bool:
        return self is Author.USER

    @property
    def dominates(self) -> bool:
        """Whether a later reading may replace this value.

        Only USER is final. The point is narrow and was bought expensively:
        "declared" means the user expressed it, not that a compiler
        instantiated it, and a product that cannot tell the two apart will
        offer its own assumption back as the user's choice.
        """
        return self is Author.USER


class ReaderKind(str, Enum):
    """What kind of reader produced a piece of evidence.

    Recorded so a disagreement can be described, never so it can be resolved.
    No kind outranks another: encoding "the model wins" or "the rule wins" is
    how one reader's blind spot becomes the system's answer.
    """

    RULE = "RULE"
    MODEL = "MODEL"
    POLICY = "POLICY"
    RETRIEVAL = "RETRIEVAL"
    PRIOR = "PRIOR"
    HUMAN = "HUMAN"


@dataclass(frozen=True)
class DecisionEvidence:
    """One reader's view of one field, kept whether or not it won.

    Losing readings are retained deliberately. A field that was contested and
    then settled is a different fact from one that was never in doubt, and only
    the first justifies asking again when the readers change.
    """

    reader_id: str
    kind: ReaderKind
    value: Any
    confidence: str = "1"
    """Decimal string, not a float. Cross-language float formatting is the
    classic source of hashes that agree on one runtime and disagree on
    another."""

    source_ref: str = ""
    """Where the reader got it: a rule id, a character span, a chunk id."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", decimal_string(self.confidence))

    def to_json(self) -> Dict[str, Any]:
        return {
            "reader_id": self.reader_id,
            "kind": self.kind.value,
            "value": self.value,
            "confidence": self.confidence,
            "source_ref": self.source_ref,
        }


@dataclass(frozen=True)
class IntentField:
    """One settled dimension of meaning, with its attribution."""

    value: Any
    author: Author
    produced_by: str = ""
    """The runtime and version that produced this reading — not who asserted
    it. See the module docstring; the distinction is load-bearing for replay."""

    confidence: str = "1"
    source_span: str = ""
    """The user's own words that carried it, for showing back to them. Never
    parsed: a value recovered from display text is a value nobody stored."""

    evidence: Sequence[DecisionEvidence] = ()
    contested: bool = False
    """Whether readers disagreed before this value was settled. Distinct from
    low confidence — a unanimous uncertain reading and a resolved fight are
    different situations."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", decimal_string(self.confidence))
        if self.contested and not self.evidence:
            raise ValueError(
                "a contested field must carry the evidence that contested it; "
                "recording the fight without the readings makes it unreviewable")

    @property
    def is_user_authored(self) -> bool:
        return self.author.is_user_authored

    def canonical_form(self) -> Dict[str, Any]:
        """Identity: what was asserted, and by whom.

        Confidence, span, evidence and producer are excluded — see the module
        docstring. `contested` is excluded for the same reason: it describes how
        the value was reached, not what it is.
        """
        return {"value": self.value, "author": self.author.value}

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "produced_by": self.produced_by,
            "confidence": self.confidence,
            "source_span": self.source_span,
            "contested": self.contested,
            "evidence": [e.to_json() for e in self.evidence],
        }


class OpenReason(str, Enum):
    """Why a dimension has no value."""

    NOT_ASKED = "NOT_ASKED"
    """Nobody raised it. Not evidence of anything about the user's wishes."""

    USER_DECLINED = "USER_DECLINED"
    """Asked, and the user chose not to constrain it. A decision, and one a
    consumer may act on by applying a declared default *and saying so*."""

    UNRESOLVED_DISAGREEMENT = "UNRESOLVED_DISAGREEMENT"
    """Readers disagreed materially and it was not settled. The one state in
    which a consumer must not proceed: proceeding means picking a reading
    nobody chose."""

    @property
    def blocks_execution(self) -> bool:
        return self is OpenReason.UNRESOLVED_DISAGREEMENT


@dataclass(frozen=True)
class Unresolved:
    """A dimension deliberately carried as open."""

    dimension: str
    reason: OpenReason
    detail: str = ""
    evidence: Sequence[DecisionEvidence] = ()

    result_changing: bool = True
    """Whether settling this differently would change what the consumer
    produces.

    **Defaults to True, and the default is the point.** An intent may only be
    sealed when every remaining open dimension is *not* result-changing, so
    silence blocks sealing rather than permitting it. Marking something
    cosmetic is a claim someone has to make on purpose.

    "The user declined to name a dividend policy, and this engine runs on price
    series so it cannot change a figure" is a defensible False. "Nobody asked
    which account it is" is not, whatever the pressure to ship."""

    def canonical_form(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "reason": self.reason.value}

    def to_json(self) -> Dict[str, Any]:
        return {**self.canonical_form(), "detail": self.detail,
                "evidence": [e.to_json() for e in self.evidence]}


@dataclass(frozen=True)
class Amendment:
    """A change the user made after first stating their intent.

    Kept in order and never collapsed into the field it changed. "They asked
    for X" and "they asked for Y, then changed it to X" are different histories,
    and only the second explains why a saved plan does not match a sentence.
    """

    dimension: str
    from_value: Any
    to_value: Any
    author: Author = Author.USER
    at: Optional[str] = None
    """Excluded from identity: when they changed their mind is not part of what
    they asked for."""

    def canonical_form(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "from_value": self.from_value,
                "to_value": self.to_value, "author": self.author.value}

    def to_json(self) -> Dict[str, Any]:
        return {**self.canonical_form(), "at": self.at}


class IntentState(str, Enum):
    """Whether the meaning is closed."""

    DRAFT = "DRAFT"
    """Still being formed. Discovery may amend it, ask about it, re-read it.
    A consumer that executes a DRAFT is executing a guess."""

    VERIFIED = "VERIFIED"
    """Sealed. Every result-changing ambiguity is settled, and nothing
    downstream may change what it means — only decide whether it may run."""


class NotSealable(ValueError):
    """An intent was offered as VERIFIED with meaning still open.

    Raised rather than downgraded. An intent that quietly stays DRAFT while a
    caller believes it sealed is the same failure as an unverified result that
    looks verified, and it is the one this whole boundary exists to prevent.
    """


@dataclass(frozen=True)
class VerifiedIntent:
    """What the user asked for, settled and attributable.

    The consumer's contract with this object is short:

        read it · refuse what you cannot execute, by name · never edit it

    **Discovery may not seal it while meaning is open.** `seal()` refuses when
    any remaining open dimension is result-changing, so "we ran out of time
    asking" cannot become "the user agreed". The state is on the artifact
    rather than in the producer's head, because the consumer is a different
    runtime and cannot see the producer's head.
    """

    SCHEMA_ID: str = field(default="runtime-contracts/verified-intent",
                           init=False, repr=False)
    SCHEMA_VERSION: str = field(default="0.1", init=False, repr=False)

    objective: str
    fields: Mapping[str, IntentField] = field(default_factory=dict)
    unresolved: Sequence[Unresolved] = ()
    amendments: Sequence[Amendment] = ()

    state: IntentState = IntentState.DRAFT
    """Fail-closed: an intent nobody sealed is a draft, whatever it looks
    like."""

    produced_by: str = ""
    """Runtime and version, e.g. ``discovery-runtime@0.4.2``. Recorded, not
    hashed: the same assertion elicited by two versions is one intent."""

    utterance_ref: str = ""
    """A reference to the user's original words — an id, not the text. The text
    is shown to the user; nothing downstream may re-read it, because a value
    recovered from prose is a value nobody stored."""

    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        overlap = {u.dimension for u in self.unresolved} & set(self.fields)
        if overlap:
            raise ValueError(
                f"{sorted(overlap)} is both settled and unresolved; a consumer "
                "reading one and not the other would act on half a decision")

    # --- what a consumer asks -------------------------------------------

    def value(self, dimension: str, default: Any = None) -> Any:
        f = self.fields.get(dimension)
        return default if f is None else f.value

    def author_of(self, dimension: str) -> Optional[Author]:
        f = self.fields.get(dimension)
        return None if f is None else f.author

    def state_of(self, dimension: str) -> str:
        """`SETTLED` · `OPEN` · `ABSENT`.

        Three states, deliberately. A consumer that collapses them will invent
        a default for a dimension the user deliberately left open, or treat a
        question nobody asked as an answer.
        """
        if dimension in self.fields:
            return "SETTLED"
        if any(u.dimension == dimension for u in self.unresolved):
            return "OPEN"
        return "ABSENT"

    @property
    def blocking(self) -> list:
        """Open dimensions that must be settled before anything runs."""
        return [u for u in self.unresolved if u.reason.blocks_execution]

    @property
    def unsealable(self) -> list:
        """Open dimensions that prevent sealing.

        Broader than `blocking`: an unresolved *disagreement* blocks execution
        outright, and any open dimension that would change the result blocks
        *sealing* — because sealing is the claim that the meaning is closed,
        and a question whose answer changes the number is not closed.
        """
        return [u for u in self.unresolved
                if u.result_changing or u.reason.blocks_execution]

    def seal(self) -> "VerifiedIntent":
        """The same intent, marked VERIFIED. Raises if meaning is still open.

        This is the only way to reach VERIFIED, and it is a method rather than
        a constructor argument so that the check cannot be skipped by passing
        the field. Discovery calls it; nothing downstream does.
        """
        open_dimensions = self.unsealable
        if open_dimensions:
            named = ", ".join(sorted(u.dimension for u in open_dimensions))
            raise NotSealable(
                f"cannot seal: {named} would change the result and is still "
                "open. Settle it, or record it as not result-changing and say "
                "why — 'we stopped asking' is not the same as 'the user agreed'")
        return replace(self, state=IntentState.VERIFIED)

    @property
    def is_verified(self) -> bool:
        return self.state is IntentState.VERIFIED

    @property
    def is_executable_in_principle(self) -> bool:
        """Sealed, and no unresolved disagreement. Says nothing about
        capability — that is the consumer's question, answered against its own
        manifest."""
        return self.is_verified and not self.blocking

    @property
    def contested_dimensions(self) -> list:
        return sorted(k for k, f in self.fields.items() if f.contested)

    @property
    def user_authored(self) -> list:
        return sorted(k for k, f in self.fields.items() if f.is_user_authored)

    # --- identity --------------------------------------------------------

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "schema_id": self.SCHEMA_ID,
            "schema_version": self.SCHEMA_VERSION,
            "contract_version": CONTRACT_VERSION,
            "objective": self.objective,
            "fields": {k: v.canonical_form() for k, v in self.fields.items()},
            "unresolved": sorted(
                (u.canonical_form() for u in self.unresolved),
                key=lambda u: u["dimension"]),
            # Ordered, not sorted: the sequence of changes is the history.
            "amendments": [a.canonical_form() for a in self.amendments],
        }

    @property
    def intent_hash(self) -> str:
        """The identity a plan re-runs from.

        A model is non-deterministic; the same sentence twice may not produce
        the same intent. Pinning the intent makes a plan reproducible.
        Re-reading the sentence on reopen would make history rewritable, which
        is the same defect as recovering an authoritative answer from rendered
        prose.
        """
        return content_hash(self.canonical_form())

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "intent_hash": self.intent_hash,
            "produced_by": self.produced_by,
            "utterance_ref": self.utterance_ref,
            "created_at": self.created_at,
            "fields": {k: v.to_json() for k, v in self.fields.items()},
            "unresolved": [u.to_json() for u in self.unresolved],
            "amendments": [a.to_json() for a in self.amendments],
        }


@dataclass(frozen=True)
class MissionProposal:
    """Something Discovery thinks is worth doing, and why.

    Discovery has two entry modes and one output. A world observation —
    "churn moved" — and a human sentence — "evaluate this SPY strategy" — are
    the same kind of thing at this boundary: a candidate for work, whose
    meaning has been made explicit and whose merit has been argued. Neither is
    execution yet.

    **Discovery ranks what is worth proposing. Mission decides what is
    admissible to execute.** That is the whole of the split, and it is why
    there is no `authorized`, `affordable` or `executable` field here. Those
    are Mission's answers, and a proposal that carried them would be a second
    planner wearing a discovery hat — which is how "decides what deserves
    attention" quietly becomes "decides what happens".

    `priority` is an argument, not a permission.
    """

    intent: VerifiedIntent
    rationale: str = ""
    priority: str = "0"
    """Decimal string in [0, 1]. Discovery's own ordering, comparable only
    against other proposals from the same producer and version — a number that
    means "more urgent than the others I raised", not "urgent"."""

    expected_value: str = ""
    """Free text. Deliberately not a number: a monetary estimate here would be
    read as a commitment by everything downstream, and Discovery is not in a
    position to make one."""

    evidence: Sequence[DecisionEvidence] = ()
    """What made this worth raising — a metric that moved, a sentence someone
    typed. The observation, not the conclusion."""

    observed_at: Optional[str] = None
    produced_by: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "priority", decimal_string(self.priority))

    @property
    def is_actionable(self) -> bool:
        """Whether Mission may even consider it. A proposal whose intent is
        still a draft is a question, not a candidate."""
        return self.intent.is_verified

    def to_json(self) -> Dict[str, Any]:
        return {"intent": self.intent.to_json(), "rationale": self.rationale,
                "priority": self.priority, "expected_value": self.expected_value,
                "evidence": [e.to_json() for e in self.evidence],
                "observed_at": self.observed_at, "produced_by": self.produced_by}


class MissionOutcome(str, Enum):
    """What Mission may answer when handed a verified intent.

    Five answers, and the missing sixth is the point. There is no
    `REINTERPRETED`: "I could not do what you meant, so I did something else"
    is not an outcome this boundary permits, and every expensive defect in the
    system that produced this contract had exactly that shape.

    A refusal is cheap. A figure computed for a plan nobody described is not.
    """

    EXECUTABLE = "EXECUTABLE"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    """The engine cannot run this dimension or this value. Carries a
    `CapabilityRefusal` naming it."""

    NEEDS_APPROVAL = "NEEDS_APPROVAL"
    """Runnable, and someone must say yes first. The "may I do this?" gate —
    distinct from Discovery's "what did you mean?", and never merged with it:
    one is authorising, the other is authoring."""

    POLICY_DENIED = "POLICY_DENIED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

    @property
    def may_execute(self) -> bool:
        return self is MissionOutcome.EXECUTABLE


@dataclass(frozen=True)
class Derivation:
    """How an executable artifact came from an intent.

    Deliberately *not* fields on `MissionProgram`: that type is a lifecycle
    declaration — states and transitions — and a compiled plan is a different
    object that happens to share the name in some implementations. A separate
    record can be embedded by whichever artifact actually did the compiling
    without either type growing the other's concerns.

    The chain this closes: a figure names a run, a run names a compiled plan,
    a plan names the intent it compiled and the runtime that compiled it, and
    the intent names its author per field. Break any link and "why does this
    number say that?" stops being answerable.
    """

    compiled_from: str
    """The `intent_hash`. By hash, not by reference: an intent that changed is
    a different intent, and a plan pointing at a mutable id would silently
    re-describe itself."""

    compiled_by: str
    """Runtime and version, e.g. ``mission-runtime@0.6.1``."""

    manifest_hash: str = ""
    """Which capability manifest decided what was executable. Two runtimes at
    the same version with different manifests reach different refusals, and
    without this the disagreement is invisible."""

    def canonical_form(self) -> Dict[str, Any]:
        return {"compiled_from": self.compiled_from,
                "compiled_by": self.compiled_by,
                "manifest_hash": self.manifest_hash}

    def to_json(self) -> Dict[str, Any]:
        return self.canonical_form()


class RefusalKind(str, Enum):
    """Why an intent could not be executed as stated."""

    UNSUPPORTED_DIMENSION = "UNSUPPORTED_DIMENSION"
    """The engine models no such thing at all — rebalancing, tax lots."""

    UNSUPPORTED_VALUE = "UNSUPPORTED_VALUE"
    """The dimension is executable; this value is not — `cadence=payroll`."""

    UNRESOLVED_INPUT = "UNRESOLVED_INPUT"
    """A blocking `Unresolved` reached execution. Proceeding would mean
    choosing a reading nobody chose."""

    NO_DATA = "NO_DATA"
    """Nothing priceable, no history, no snapshot."""


@dataclass(frozen=True)
class CapabilityRefusal:
    """A refusal that names what it refused.

    The shape matters more than it looks. "This result is unavailable" sends a
    reader nowhere; "you asked for allocation by inverse volatility and this
    build allocates equally at purchase" tells them what to change and tells a
    reviewer which capability to add. A refusal that cannot be aggregated by
    dimension also cannot be turned into a roadmap.

    Never a degradation. The engine substituting the nearest thing it *can*
    execute is the defect this whole boundary exists to prevent.
    """

    kind: RefusalKind
    dimension: str
    stated_value: Any = None
    executable_values: Sequence[Any] = ()
    """What this build could have run instead — for the reader's benefit, and
    explicitly not applied on their behalf."""

    detail: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "dimension": self.dimension,
                "stated_value": self.stated_value,
                "executable_values": list(self.executable_values),
                "detail": self.detail}
