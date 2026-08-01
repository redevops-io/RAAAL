"""Forward tracking: what actually happened, recorded beside what was planned.

The plan is a declaration and does not change. Observations are historical facts
that accumulate against it, and current status is derived from both — the same
split the rest of the system uses, applied forward in time:

    Plan          = declaration
    Observation   = historical fact
    Current status = derivation

Mutating the saved plan when reality diverges would destroy the only thing
tracking is for: knowing that reality diverged.

**The linked return is not merely discouraged, it is uncomputable here.** A chart
flowing seamlessly from backtest into live is the most damaging artifact in this
product category, and GIPS forbids linking actual to theoretical performance. So
`SegmentedPerformance` holds the two segments and raises on any attempt to join
them, rather than exposing a combined series and asking renderers to behave.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .proposal import Eligibility, Proposal, ProposalStatus, lifecycle_summary


class Segment(str, Enum):
    HISTORICAL = "HISTORICAL"
    """Replay. Hypothetical, no capital, no orders."""

    FORWARD = "FORWARD"
    """Paper-tracked after the plan was saved. Still hypothetical — no orders
    were placed — but recorded prospectively rather than fitted."""


class LinkedSeriesRefused(RuntimeError):
    """Someone tried to join a backtest to forward tracking."""


@dataclass(frozen=True)
class SegmentedPerformance:
    """Two segments, never one series."""

    historical: Optional[Any] = None
    forward: Optional[Any] = None

    def linked_series(self):
        raise LinkedSeriesRefused(
            "A backtested segment and a forward segment cannot be joined into "
            "one series. They are different kinds of number: one was fitted "
            "with the whole period visible, the other was recorded as it "
            "happened. Linking them presents the first as evidence for the "
            "second, which is the claim this product exists not to make."
        )

    # Aliases people reach for. Each raises, so the refusal cannot be routed
    # around by finding a differently-named accessor.
    combined = linked_series
    since_inception = linked_series
    full_history = linked_series

    def to_json(self) -> Dict[str, Any]:
        return {
            "historical": (self.historical.to_json() if self.historical else None),
            "forward": (self.forward.to_json() if self.forward else None),
            "linked": None,
            "note": (
                "Reported as two segments. The historical segment is a replay "
                "over a period that was already known; the forward segment was "
                "recorded as it happened. Neither is a track record, and they "
                "are not comparable to each other."
            ),
        }


class DeviationKind(str, Enum):
    MISSING = "MISSING"
    """Expected and did not happen."""

    UNEXPECTED = "UNEXPECTED"
    """Happened and was not expected."""

    DIFFERENT = "DIFFERENT"
    """Happened differently — a different date, size or price."""


@dataclass(frozen=True)
class ExpectedEvent:
    date: str
    kind: str
    detail: str = ""
    source: str = ""
    """Which part of the plan predicted it, so a deviation points somewhere."""

    def key(self) -> tuple:
        return (self.date, self.kind)

    def to_json(self) -> Dict[str, Any]:
        return {"date": self.date, "kind": self.kind, "detail": self.detail,
                "source": self.source}


@dataclass(frozen=True)
class ObservedEvent:
    date: str
    kind: str
    detail: str = ""

    def key(self) -> tuple:
        return (self.date, self.kind)

    def to_json(self) -> Dict[str, Any]:
        return {"date": self.date, "kind": self.kind, "detail": self.detail}


@dataclass(frozen=True)
class Deviation:
    kind: DeviationKind
    expected: Optional[ExpectedEvent]
    observed: Optional[ObservedEvent]
    why_it_matters: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "expected": self.expected.to_json() if self.expected else None,
            "observed": self.observed.to_json() if self.observed else None,
            "why_it_matters": self.why_it_matters,
        }


@dataclass(frozen=True)
class PlanObservation:
    """One look at a saved plan, at one moment, against one data snapshot."""

    plan_id: str
    observed_at: str
    data_snapshot: str
    expected_events: Sequence[ExpectedEvent] = ()
    observed_events: Sequence[ObservedEvent] = ()
    deviations: Sequence[Deviation] = ()
    paper_action_proposals: Sequence[Proposal] = ()
    unresolved: Sequence[str] = ()

    @property
    def artifact_id(self) -> str:
        return f"observation/{self.plan_id}@{self.observed_at}"

    @property
    def has_drifted(self) -> bool:
        return bool(self.deviations)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True,
                                         separators=(",", ":"),
                                         default=str).encode()).hexdigest()

    def summary(self) -> str:
        """Plain language, stating facts and proposing nothing."""
        if not self.deviations:
            return (f"As of {self.observed_at[:10]}, the plan and what happened "
                    f"agree on every event.")
        counts: Dict[str, int] = {}
        for d in self.deviations:
            counts[d.kind.value] = counts.get(d.kind.value, 0) + 1
        parts = ", ".join(f"{n} {k.lower()}" for k, n in sorted(counts.items()))
        return (f"As of {self.observed_at[:10]}, {len(self.deviations)} "
                f"difference(s) between the plan and what happened: {parts}.")

    def to_json(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "observed_at": self.observed_at,
            "data_snapshot": self.data_snapshot,
            "expected_events": [e.to_json() for e in self.expected_events],
            "observed_events": [o.to_json() for o in self.observed_events],
            "deviations": [d.to_json() for d in self.deviations],
            "action_eligibility": sorted(
                {p.eligibility.value for p in self.paper_action_proposals}),
            "proposal_lifecycle": lifecycle_summary(self.paper_action_proposals),
            "paper_action_proposals": [
                p.to_json() for p in self.paper_action_proposals],
            "unresolved": list(self.unresolved),
            "has_drifted": self.has_drifted,
            "summary": self.summary(),
        }


def reconcile(expected: Sequence[ExpectedEvent],
              observed: Sequence[ObservedEvent]) -> List[Deviation]:
    """Match what was planned against what happened.

    Matching is on (date, kind) rather than on order, because an event arriving
    late is a *different* event from one that never arrived, and a positional
    comparison would report both as the same thing.
    """
    by_expected = {e.key(): e for e in expected}
    by_observed = {o.key(): o for o in observed}

    deviations: List[Deviation] = []
    for key, event in by_expected.items():
        if key not in by_observed:
            deviations.append(Deviation(
                kind=DeviationKind.MISSING, expected=event, observed=None,
                why_it_matters=(
                    "The plan assumed this would happen. Everything downstream "
                    "of it in the simulation assumed it too."
                ),
            ))
        elif by_observed[key].detail and event.detail and \
                by_observed[key].detail != event.detail:
            deviations.append(Deviation(
                kind=DeviationKind.DIFFERENT, expected=event,
                observed=by_observed[key],
                why_it_matters=(
                    "It happened, but not as planned. The saved plan still says "
                    "what it said; this records that reality differed."
                ),
            ))

    for key, event in by_observed.items():
        if key not in by_expected:
            deviations.append(Deviation(
                kind=DeviationKind.UNEXPECTED, expected=None, observed=event,
                why_it_matters=(
                    "This was not in the plan. It may be a life event worth "
                    "adding, or a sign the plan no longer describes what you do."
                ),
            ))

    return deviations
