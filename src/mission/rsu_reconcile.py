"""Planned against observed, with four kinds of "not there" kept apart.

    not yet observed   the window is still open
    observed late      it arrived outside its expected date, inside tolerance
    observed differently  it arrived, and not as predicted
    confirmed absent   someone with evidence says it did not happen

The generic forward tracker matched on `(date, kind)` exactly and had no pending
state, so a vest due on the 15th and examined on the 10th reported MISSING, and
a vest that settled four days late reported MISSING *and* UNEXPECTED — two
deviations describing one event that happened once.

**The plan never moves.** An expectation that turned out wrong is the evidence
that it was wrong, and rewriting it to match reality destroys the only thing
tracking is for. A user may revise the plan for future vests; the earlier
prediction stays as it was.

**Nothing is assigned automatically when it is ambiguous.** Two grants that could
both explain one observation are not resolved by picking the nearer date. The
observation is held for the user to resolve, because attaching it to the wrong
grant silently moves shares between two positions.

The matching window is declared and versioned rather than silently fuzzy:
payroll and settlement dates shift, and a tolerance nobody can see is a
tolerance nobody can check.
"""
from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Bumped when a change would reconcile the same records differently. Stored on
#: every reconciliation, so a historical one says which rules produced it.
MATCHING_POLICY_VERSION = "rsu-matching@1"


class ReconciliationStatus(str, Enum):
    PENDING = "PENDING"
    """Expected, and its observation window is still open. Not missing."""

    UNOBSERVED_OVERDUE = "UNOBSERVED_OVERDUE"
    """The window closed and nothing was reported. Wants attention; it does not
    claim the vest failed, and it does not rewrite the plan."""

    MATCHED = "MATCHED"
    MATCHED_WITH_VARIANCE = "MATCHED_WITH_VARIANCE"
    """The same event, differing in a dimension. Not an unexpected event."""

    LATE = "LATE"
    """Arrived outside its expected date and inside the declared tolerance. One
    event, not a missing one plus an unexpected one."""

    MISSING_CONFIRMED = "MISSING_CONFIRMED"
    """Someone with evidence says it did not happen. Reachable only from an
    explicit confirmation, never from silence."""

    UNEXPECTED = "UNEXPECTED"
    """Observed and matching no plan. Kept visible rather than attached to the
    nearest one."""

    AMBIGUOUS = "AMBIGUOUS"
    """Two or more plans could explain it. Held for the user."""

    CONFLICTING = "CONFLICTING"
    """Two observations claim one planned event."""


#: States that mean "we do not know yet". Distinguished from absence everywhere
#: they are read.
UNKNOWN_STATES = frozenset({ReconciliationStatus.PENDING,
                            ReconciliationStatus.UNOBSERVED_OVERDUE,
                            ReconciliationStatus.AMBIGUOUS,
                            ReconciliationStatus.CONFLICTING})


@dataclass(frozen=True)
class MatchingPolicy:
    """How near counts as the same event. Declared, versioned, checkable."""

    date_tolerance_days: int = 7
    """Payroll and settlement dates shift. Outside this, a late vest becomes a
    missing one plus an unexpected one — which is the right answer once the gap
    is large enough that they might genuinely be different events."""

    quantity_tolerance: float = 0.0
    """Exact by default. A share count that differs is a variance to report,
    not a rounding to absorb."""

    observation_grace_days: int = 3
    """How long after the expected date the window stays open before an absence
    becomes overdue."""

    version: str = MATCHING_POLICY_VERSION

    def to_json(self) -> Dict[str, Any]:
        return {"date_tolerance_days": self.date_tolerance_days,
                "quantity_tolerance": self.quantity_tolerance,
                "observation_grace_days": self.observation_grace_days,
                "version": self.version}


@dataclass(frozen=True)
class PlannedEvent:
    """What the confirmed scenario expected. Never edited."""

    event_id: str
    grant_ref: str
    expected_date: str
    kind: str = "vest"
    employer_asset: str = ""
    expected_gross_shares: Optional[float] = None
    expected_withheld_shares: Optional[float] = None
    expected_delivered_shares: Optional[float] = None
    expected_value: Optional[float] = None
    source_declaration: str = ""
    version_pin: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"event_id": self.event_id, "grant_ref": self.grant_ref,
                "expected_date": self.expected_date, "kind": self.kind,
                "employer_asset": self.employer_asset,
                "expected_gross_shares": self.expected_gross_shares,
                "expected_withheld_shares": self.expected_withheld_shares,
                "expected_delivered_shares": self.expected_delivered_shares,
                "expected_value": self.expected_value,
                "source_declaration": self.source_declaration,
                "version_pin": self.version_pin}


@dataclass(frozen=True)
class ObservedEvent:
    """What was later reported. Also never edited."""

    observation_id: str
    observed_date: str
    """When it was reported."""

    effective_date: str
    """When it actually happened. Distinct from `observed_date`, because a vest
    reported in July may have settled in June, and reconciling against the
    report date would call an on-time vest late."""

    kind: str = "vest"
    grant_ref: str = ""
    employer_asset: str = ""
    gross_shares: Optional[float] = None
    withheld_shares: Optional[float] = None
    delivered_shares: Optional[float] = None
    value: Optional[float] = None
    evidence_ref: str = ""
    source: str = "user"
    confirms_absence: bool = False
    """Set only by an explicit report that the event did not occur. Silence
    never sets it."""

    def to_json(self) -> Dict[str, Any]:
        return {"observation_id": self.observation_id,
                "observed_date": self.observed_date,
                "effective_date": self.effective_date, "kind": self.kind,
                "grant_ref": self.grant_ref,
                "employer_asset": self.employer_asset,
                "gross_shares": self.gross_shares,
                "withheld_shares": self.withheld_shares,
                "delivered_shares": self.delivered_shares, "value": self.value,
                "evidence_ref": self.evidence_ref, "source": self.source,
                "confirms_absence": self.confirms_absence}


@dataclass(frozen=True)
class Variance:
    """One dimension that differed. Kept apart from the others, because a vest
    can match in identity and differ materially in exactly one."""

    dimension: str
    expected: Any
    observed: Any
    delta: Any = None

    def to_json(self) -> Dict[str, Any]:
        return {"dimension": self.dimension, "expected": self.expected,
                "observed": self.observed, "delta": self.delta}


@dataclass(frozen=True)
class EventReconciliation:
    """The derived relationship. Neither record is changed to produce it."""

    reconciliation_id: str
    status: ReconciliationStatus
    planned_ref: Optional[str] = None
    observed_ref: Optional[str] = None
    variances: Sequence[Variance] = ()
    candidates: Sequence[str] = ()
    """Populated for AMBIGUOUS and CONFLICTING: what could not be told apart."""

    matching_policy_version: str = MATCHING_POLICY_VERSION
    derived_at: str = ""
    evidence_refs: Sequence[str] = ()
    detail: str = ""

    @property
    def is_unknown(self) -> bool:
        """Whether this says "we do not know" rather than "it did not happen"."""
        return self.status in UNKNOWN_STATES

    def to_json(self) -> Dict[str, Any]:
        return {"reconciliation_id": self.reconciliation_id,
                "status": self.status.value, "planned_ref": self.planned_ref,
                "observed_ref": self.observed_ref,
                "variances": [one.to_json() for one in self.variances],
                "candidates": list(self.candidates),
                "matching_policy_version": self.matching_policy_version,
                "derived_at": self.derived_at,
                "evidence_refs": list(self.evidence_refs),
                "is_unknown": self.is_unknown, "detail": self.detail}


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(str(value)[:10])


def _days_apart(left: str, right: str) -> int:
    return abs((_date(left) - _date(right)).days)


def _new_id() -> str:
    return f"rec-{uuid.uuid4().hex[:16]}"


def _could_match(planned: PlannedEvent, observed: ObservedEvent,
                 policy: MatchingPolicy) -> bool:
    """Identity and semantics, never list position."""
    if planned.kind != observed.kind:
        return False
    if observed.grant_ref and planned.grant_ref != observed.grant_ref:
        return False
    if observed.employer_asset and planned.employer_asset and \
            planned.employer_asset != observed.employer_asset:
        return False
    return _days_apart(planned.expected_date, observed.effective_date) \
        <= policy.date_tolerance_days


def _variances(planned: PlannedEvent, observed: ObservedEvent,
               policy: MatchingPolicy) -> List[Variance]:
    found: List[Variance] = []

    if planned.expected_date != observed.effective_date:
        found.append(Variance(
            dimension="date", expected=planned.expected_date,
            observed=observed.effective_date,
            delta=(_date(observed.effective_date)
                   - _date(planned.expected_date)).days))

    for dimension, expected, actual in (
            ("gross_shares", planned.expected_gross_shares,
             observed.gross_shares),
            ("withheld_shares", planned.expected_withheld_shares,
             observed.withheld_shares),
            ("delivered_shares", planned.expected_delivered_shares,
             observed.delivered_shares),
            ("value", planned.expected_value, observed.value)):
        if expected is None or actual is None:
            # Unknown on either side is not a variance. Reporting one would
            # invent a difference from an absence of information.
            continue
        if abs(actual - expected) > policy.quantity_tolerance:
            found.append(Variance(dimension=dimension, expected=expected,
                                  observed=actual, delta=actual - expected))
    return found


def reconcile(planned: Sequence[PlannedEvent],
              observed: Sequence[ObservedEvent], *, as_of: str,
              policy: Optional[MatchingPolicy] = None
              ) -> List[EventReconciliation]:
    """Derive the relationship between two immutable records.

    `as_of` decides pending from overdue. Without it there is no way to say that
    an unreported vest is simply in the future, and every plan would look like a
    plan going wrong.
    """
    policy = policy or MatchingPolicy()
    today = _date(as_of)
    results: List[EventReconciliation] = []

    candidates: Dict[str, List[ObservedEvent]] = {
        one.event_id: [other for other in observed
                       if _could_match(one, other, policy)]
        for one in planned}

    claimed: Dict[str, List[PlannedEvent]] = {}
    for one in planned:
        for other in candidates[one.event_id]:
            claimed.setdefault(other.observation_id, []).append(one)

    for one in planned:
        matches = candidates[one.event_id]

        absent = [other for other in matches if other.confirms_absence]
        if absent:
            results.append(EventReconciliation(
                reconciliation_id=_new_id(),
                status=ReconciliationStatus.MISSING_CONFIRMED,
                planned_ref=one.event_id,
                observed_ref=absent[0].observation_id, derived_at=as_of,
                matching_policy_version=policy.version,
                evidence_refs=tuple(x.evidence_ref for x in absent
                                    if x.evidence_ref),
                detail="reported as not having occurred"))
            continue

        real = [other for other in matches if not other.confirms_absence]

        if len(real) > 1:
            results.append(EventReconciliation(
                reconciliation_id=_new_id(),
                status=ReconciliationStatus.CONFLICTING,
                planned_ref=one.event_id, derived_at=as_of,
                matching_policy_version=policy.version,
                candidates=tuple(x.observation_id for x in real),
                detail=("more than one observation could be this event, and "
                        "choosing between them is not this system's to make")))
            continue

        if not real:
            window_closes = _date(one.expected_date) + dt.timedelta(
                days=policy.observation_grace_days)
            overdue = today > window_closes
            results.append(EventReconciliation(
                reconciliation_id=_new_id(),
                status=(ReconciliationStatus.UNOBSERVED_OVERDUE if overdue
                        else ReconciliationStatus.PENDING),
                planned_ref=one.event_id, derived_at=as_of,
                matching_policy_version=policy.version,
                detail=("nothing has been reported and the window has closed; "
                        "this is not a claim that it did not happen"
                        if overdue else
                        "expected, and its observation window is still open")))
            continue

        match = real[0]
        if len(claimed.get(match.observation_id, ())) > 1:
            results.append(EventReconciliation(
                reconciliation_id=_new_id(),
                status=ReconciliationStatus.AMBIGUOUS,
                planned_ref=one.event_id,
                observed_ref=match.observation_id, derived_at=as_of,
                matching_policy_version=policy.version,
                candidates=tuple(x.event_id
                                 for x in claimed[match.observation_id]),
                detail=("more than one planned event could explain this "
                        "observation; attaching it to the nearer one would "
                        "move shares between two grants")))
            continue

        found = _variances(one, match, policy)
        date_moved = any(v.dimension == "date" for v in found)
        other_moved = [v for v in found if v.dimension != "date"]

        if not found:
            status = ReconciliationStatus.MATCHED
            detail = "as planned"
        elif date_moved and not other_moved:
            status = ReconciliationStatus.LATE
            detail = "the same event, outside its expected date"
        else:
            status = ReconciliationStatus.MATCHED_WITH_VARIANCE
            detail = "the same event, differing in " + ", ".join(
                v.dimension for v in other_moved)

        results.append(EventReconciliation(
            reconciliation_id=_new_id(), status=status,
            planned_ref=one.event_id, observed_ref=match.observation_id,
            variances=tuple(found), derived_at=as_of,
            matching_policy_version=policy.version,
            evidence_refs=((match.evidence_ref,) if match.evidence_ref else ()),
            detail=detail))

    matched_ids = {r.observed_ref for r in results if r.observed_ref}
    for other in observed:
        if other.observation_id in matched_ids or other.confirms_absence:
            continue
        results.append(EventReconciliation(
            reconciliation_id=_new_id(),
            status=ReconciliationStatus.UNEXPECTED,
            observed_ref=other.observation_id, derived_at=as_of,
            matching_policy_version=policy.version,
            evidence_refs=((other.evidence_ref,) if other.evidence_ref else ()),
            detail=("this matches no planned event; it is not attached to the "
                    "nearest one")))
    return results


@dataclass(frozen=True)
class CounterfactualRun:
    """A hypothetical, labelled as one.

    Never written into the observation record. A counterfactual that could be
    mistaken for what happened is worse than none, because it is the version
    that flatters whichever decision it explores.
    """

    counterfactual_id: str
    observed_state_ref: str
    changed_dimension: str
    run_ref: str = ""
    comparability_verdict: Optional[Mapping[str, Any]] = None
    isolates: str = ""
    hypothetical: bool = True

    @property
    def is_isolated(self) -> bool:
        """Only when exactly one dimension changed and the verdict agrees."""
        verdict = self.comparability_verdict or {}
        return bool(self.changed_dimension) and \
            bool(verdict.get("attribution_isolated"))

    def to_json(self) -> Dict[str, Any]:
        return {"counterfactual_id": self.counterfactual_id,
                "observed_state_ref": self.observed_state_ref,
                "changed_dimension": self.changed_dimension,
                "run_ref": self.run_ref,
                "comparability_verdict": dict(self.comparability_verdict or {}),
                "isolates": self.isolates, "hypothetical": self.hypothetical,
                "is_isolated": self.is_isolated,
                "label": "HYPOTHETICAL — this did not happen"}
