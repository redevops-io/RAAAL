"""A Mission, and the provenance of everything the compiler decided for you.

Externally this is a **Plan** — "New Plan", "describe your investing idea". The
internal name stays `Mission` because that is what the runtime executes, and a
user-facing word for an engine concept is how vocabulary drifts until the code
and the interface disagree about what a thing is.

The provenance block is the point of the type. Natural language is not merely
imprecise, it is *underspecified*: "invest every paycheck" does not contain the
information needed to execute it, and no amount of model quality recovers what
was never said. Those are not validation errors — they are missing execution
context, and the difference matters because a validation error can be rejected
while missing context has to be asked about.

So a Mission separates four things, and only the first came from the user:

    stated          verbatim, quoted back
    inferred        the compiler chose; the user must confirm
    contradictions  the description conflicts with itself
    unresolved      nobody has decided, and it cannot run until someone does
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

MISSION_SPEC_VERSION = "0.1"


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class Objective(str, Enum):
    REPLAY = "REPLAY"
    """What would have happened. Historical, and the only honest one to start with."""

    PROJECT = "PROJECT"
    """What might happen. Requires assumptions about the future, each declared."""

    TRACK = "TRACK"
    """What is happening. Forward-only, and structurally unlinkable from REPLAY."""


@dataclass(frozen=True)
class Inference:
    """A choice the compiler made because the user did not.

    `why` is not decoration. A user confirming nine inferences needs to know
    which ones move the number, and an inference presented without its
    consequence gets waved through.
    """

    field: str
    value: str
    why: str
    confirmed: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {"field": self.field, "value": self.value, "why": self.why,
                "confirmed": self.confirmed}


@dataclass(frozen=True)
class Unresolved:
    """Something nobody has decided. Blocks saving, not simulating.

    Simulating with a stated placeholder is useful — it shows the shape of the
    answer. Saving or tracking it would turn a placeholder into a commitment the
    user never made.
    """

    field: str
    question: str
    """Asked in plain language, of the user, answerable without domain knowledge."""

    why_it_matters: str

    def to_json(self) -> Dict[str, Any]:
        return {"field": self.field, "question": self.question,
                "why_it_matters": self.why_it_matters}


@dataclass(frozen=True)
class Contradiction:
    """Where the description conflicts with itself.

    "Never sell" and "hold them equally" cannot both hold once prices move. The
    correct behaviour is to surface it, not to pick a winner — resolving a user's
    contradiction silently means executing a plan they did not describe.
    """

    between: Sequence[str]
    detail: str
    resolution: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.resolution)

    def to_json(self) -> Dict[str, Any]:
        return {"between": list(self.between), "detail": self.detail,
                "resolution": self.resolution, "resolved": self.resolved}


@dataclass(frozen=True)
class Provenance:
    """Who decided what. The whole reason a Mission is not just a config file."""

    stated: Sequence[str] = ()
    inferred: Sequence[Inference] = ()
    contradictions: Sequence[Contradiction] = ()
    unresolved: Sequence[Unresolved] = ()

    @property
    def unconfirmed(self) -> List[Inference]:
        return [i for i in self.inferred if not i.confirmed]

    @property
    def open_contradictions(self) -> List[Contradiction]:
        return [c for c in self.contradictions if not c.resolved]

    @property
    def is_complete(self) -> bool:
        """Whether every choice in this Mission was made by someone on purpose."""
        return not (self.unresolved or self.unconfirmed or self.open_contradictions)

    def checklist(self) -> Dict[str, Any]:
        """The confirmation screen, as data.

        The interface renders this; it does not compute it. Same discipline as
        every other view model here — a screen that decided what counts as
        confirmed would be a second opinion about whether the user agreed.
        """
        return {
            "understood": [
                {"field": i.field, "value": i.value, "why": i.why,
                 "confirmed": i.confirmed}
                for i in self.inferred
            ],
            "missing": [u.to_json() for u in self.unresolved],
            "conflicts": [c.to_json() for c in self.open_contradictions],
            "ready": self.is_complete,
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "stated": list(self.stated),
            "inferred": [i.to_json() for i in self.inferred],
            "contradictions": [c.to_json() for c in self.contradictions],
            "unresolved": [u.to_json() for u in self.unresolved],
            "is_complete": self.is_complete,
        }


@dataclass(frozen=True)
class FlowSchedule:
    """The contribution and withdrawal program.

    Separate from the event program because it is the thing benchmarks must
    share. Two Missions with identical rules and different schedules are not
    comparable, and keeping the schedule addressable is what lets the
    comparability engine say so.
    """

    cadence: str
    """e.g. "monthly", "biweekly", "once". Declared, because "every paycheck"
    does not say which."""

    amount: float
    day_rule: str = "first_session_of_period"
    inflation_adjusted: bool = False
    starting_capital: float = 0.0

    funding_source: str = "contribution"
    """Where a conditional buy takes its money from: `contribution` or
    `additional_cash`.

    Part of the schedule, not the rule. Out of the contribution the plan invests
    the same total; as additional cash it invests more, and more money in a
    rising market always looks like a better rule. The compiler has always
    separated these two readings in its questions — but until the representation
    check was written it never carried the answer, so both compiled identically
    and the distinction the question exists to draw was lost immediately after
    the user drew it."""

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "cadence": self.cadence,
            "amount": self.amount,
            "day_rule": self.day_rule,
            "inflation_adjusted": self.inflation_adjusted,
            "starting_capital": self.starting_capital,
            "funding_source": self.funding_source,
        }

    @property
    def schedule_hash(self) -> str:
        """Identity of the *schedule alone*.

        Benchmark comparability turns on this and nothing else: two runs are
        flow-comparable exactly when their schedule hashes match.
        """
        return _hash(self.canonical_form())

    def to_json(self) -> Dict[str, Any]:
        return {**self.canonical_form(), "schedule_hash": self.schedule_hash}


@dataclass(frozen=True)
class Mission:
    """A cash-flow-and-event program, versioned like everything else here."""

    name: str
    version: int
    title: str
    objective: Objective
    flows: FlowSchedule
    events: Sequence[Dict[str, Any]] = ()
    constraints: Sequence[str] = ()
    benchmarks: Sequence[str] = ()
    provenance: Provenance = field(default_factory=Provenance)
    intent_ref: Optional[str] = None
    """The intent this was compiled from. Absent means hand-authored, which is
    allowed and worth being able to tell apart."""

    tax_treatment: str = "NONE_APPLIED"
    """Declared from the first version. "After taxes" is a claim about someone's
    tax situation, and a default of NONE_APPLIED that says so is honest where a
    silent assumption of long-term capital gains is not."""

    spec_version: str = MISSION_SPEC_VERSION

    @property
    def concept_id(self) -> str:
        return f"mission/{self.name}"

    @property
    def artifact_id(self) -> str:
        return f"mission/{self.name}@{self.version}"

    @property
    def can_simulate(self) -> bool:
        """A Mission with open questions can still be run to show the shape."""
        return True

    @property
    def can_save(self) -> bool:
        """Saving turns a placeholder into a commitment the user never made.

        Same principle as `unrealized_declaration` blocking publication: an
        inference the user never saw is a declaration they did not make.
        """
        return self.provenance.is_complete

    @property
    def can_track(self) -> bool:
        """Forward tracking additionally requires knowing what is being tracked."""
        return self.can_save and self.objective in {Objective.TRACK, Objective.REPLAY}

    def blocking_reasons(self) -> List[str]:
        p = self.provenance
        reasons = [f"{u.field}: {u.question}" for u in p.unresolved]
        reasons += [f"{i.field}: inferred as {i.value}, not confirmed"
                    for i in p.unconfirmed]
        reasons += [f"conflict between {' and '.join(c.between)}: {c.detail}"
                    for c in p.open_contradictions]
        return reasons

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "objective": self.objective.value,
            "flows": self.flows.canonical_form(),
            "events": list(self.events),
            "constraints": sorted(self.constraints),
            "benchmarks": sorted(self.benchmarks),
            "tax_treatment": self.tax_treatment,
            # Inferences are part of identity: a Mission that assumed a simple
            # moving average is a different program from one that assumed an
            # exponential one, whatever the user typed.
            "inferred": sorted(
                ({"field": i.field, "value": i.value} for i in self.provenance.inferred),
                key=lambda d: d["field"],
            ),
        }

    @property
    def content_hash(self) -> str:
        return _hash(self.canonical_form())

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "artifact_id": self.artifact_id,
            "concept_id": self.concept_id,
            "content_hash": self.content_hash,
            "title": self.title,
            "intent_ref": self.intent_ref,
            "provenance": self.provenance.to_json(),
            "can_save": self.can_save,
            "can_track": self.can_track,
            "blocking_reasons": self.blocking_reasons(),
        }
