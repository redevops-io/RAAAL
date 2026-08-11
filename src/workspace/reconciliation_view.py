"""Planned, observed and reconciliation, side by side, from stored records.

    stored records -> verify against a fresh derivation -> three lanes

Two layers, as the result context uses for presentability: the conclusion is
stored *and* re-derived from the pinned inputs and the matching-policy version
it names. Stored alone, a status can be edited to say MATCHED beside a variance
that says otherwise. Derived alone, a historical row silently re-judges itself
whenever the rules change.

The view builds nothing. No matching, no date arithmetic, no tolerance logic —
those decide whether an event is late, and a second implementation in the view
layer would disagree with the first on exactly the rows that are hard to call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class VerificationState(str, Enum):
    VERIFIED = "VERIFIED"
    """Stored and re-derived agree."""

    DERIVATION_MISMATCH = "RECONCILIATION_DERIVATION_MISMATCH"
    """The records no longer produce the stored conclusion. Shown as history,
    never as a verified verdict."""

    NOT_VERIFIABLE = "NOT_VERIFIABLE"
    """Stored under a matching policy this build cannot reproduce. Not a
    failure — an older policy is a fact about the record, not a defect in it."""


#: How each status is worded. Explicit strings rather than a formatted enum,
#: because "pending" must never render as "missing" and the difference is the
#: whole point of having nine states.
LANE_WORDING: Mapping[str, str] = {
    "PENDING": "Not yet due",
    "UNOBSERVED_OVERDUE": "Overdue — nothing reported yet",
    "MATCHED": "As planned",
    "MATCHED_WITH_VARIANCE": "Happened, with differences",
    "LATE": "Late",
    "MISSING_CONFIRMED": "Confirmed not received",
    "UNEXPECTED": "Not in the plan",
    "AMBIGUOUS": "Could match more than one plan",
    "CONFLICTING": "More than one report for this event",
}

#: Statuses with no observation. The observed lane says what kind of nothing.
NO_OBSERVATION_WORDING: Mapping[str, str] = {
    "PENDING": "No observation yet",
    "UNOBSERVED_OVERDUE": "No observation",
    "MISSING_CONFIRMED": "Confirmed not received",
    "CONFLICTING": "Several reports",
}


@dataclass(frozen=True)
class ReconciliationRow:
    """One line across three lanes."""

    status: str
    verification: VerificationState

    planned_summary: str = ""
    planned_date: str = ""
    planned_ref: str = ""

    observed_summary: str = ""
    observed_effective_date: str = ""
    observed_reported_date: str = ""
    observed_ref: str = ""

    verdict: str = ""
    detail: str = ""
    variances: Sequence[Mapping[str, Any]] = ()
    candidates: Sequence[str] = ()
    evidence_refs: Sequence[str] = ()

    @property
    def has_observation(self) -> bool:
        return bool(self.observed_ref)

    def to_json(self) -> Dict[str, Any]:
        return {"status": self.status,
                "verification": self.verification.value,
                "planned_summary": self.planned_summary,
                "planned_date": self.planned_date,
                "planned_ref": self.planned_ref,
                "observed_summary": self.observed_summary,
                "observed_effective_date": self.observed_effective_date,
                "observed_reported_date": self.observed_reported_date,
                "observed_ref": self.observed_ref,
                "has_observation": self.has_observation,
                "verdict": self.verdict, "detail": self.detail,
                "variances": [dict(one) for one in self.variances],
                "candidates": list(self.candidates),
                "evidence_refs": list(self.evidence_refs)}


@dataclass(frozen=True)
class RSUReconciliationView:
    rows: Sequence[ReconciliationRow] = ()
    counterfactuals: Sequence[Mapping[str, Any]] = ()
    """Kept out of the three lanes. A hypothetical sitting in the observed
    column is a thing that did not happen, in the column for things that did."""

    unverified_count: int = 0

    @classmethod
    def from_records(cls, planned_events: Sequence[Mapping[str, Any]],
                     observed_events: Sequence[Mapping[str, Any]],
                     reconciliations: Sequence[Mapping[str, Any]],
                     *, verification: Optional[Mapping[str, VerificationState]] = None,
                     counterfactuals: Sequence[Mapping[str, Any]] = ()
                     ) -> "RSUReconciliationView":
        """Arrange stored records. Decides nothing about whether they match.

        Takes records and a verification map computed elsewhere. Given no
        matching policy and no clock, this cannot re-decide a status even by
        accident.
        """
        by_planned = {one["planned_event_id"]: one for one in planned_events}
        by_observed = {one["observed_event_id"]: one for one in observed_events}
        states = dict(verification or {})

        rows: List[ReconciliationRow] = []
        for record in reconciliations:
            payload = record.get("payload") or {}
            status = record["status"]
            planned = by_planned.get(record.get("planned_event_id") or "")
            observed = by_observed.get(record.get("observed_event_id") or "")

            rows.append(ReconciliationRow(
                status=status,
                verification=states.get(record["reconciliation_id"],
                                        VerificationState.NOT_VERIFIABLE),
                planned_summary=_summarise_planned(planned),
                planned_date=(planned or {}).get("expected_effective_date", ""),
                planned_ref=record.get("planned_event_id") or "",
                observed_summary=(_summarise_observed(observed) if observed
                                  else NO_OBSERVATION_WORDING.get(
                                      status, "No observation")),
                observed_effective_date=(observed or {}).get("effective_date", ""),
                observed_reported_date=(observed or {}).get("observed_at", ""),
                observed_ref=record.get("observed_event_id") or "",
                verdict=LANE_WORDING.get(status, status),
                detail=payload.get("detail", ""),
                variances=tuple(payload.get("variances") or ()),
                candidates=tuple(payload.get("candidates") or ()),
                evidence_refs=tuple(payload.get("evidence_refs") or ())))

        return cls(rows=tuple(rows), counterfactuals=tuple(counterfactuals),
                   unverified_count=sum(
                       1 for row in rows
                       if row.verification is not VerificationState.VERIFIED))

    def to_json(self) -> Dict[str, Any]:
        return {"rows": [row.to_json() for row in self.rows],
                "counterfactuals": [dict(one) for one in self.counterfactuals],
                "unverified_count": self.unverified_count}


def _summarise_planned(record: Optional[Mapping[str, Any]]) -> str:
    if not record:
        return "Not in the plan"
    quantity = record.get("expected_quantity")
    asset = record.get("asset") or ""
    return (f"{quantity:g} {asset} shares".strip()
            if quantity is not None else (asset or record.get("kind", "")))


def _summarise_observed(record: Mapping[str, Any]) -> str:
    quantity = record.get("quantity")
    asset = record.get("asset") or ""
    return (f"{quantity:g} {asset} shares".strip()
            if quantity is not None else (asset or record.get("kind", "")))


def verify(stored: Sequence[Mapping[str, Any]],
           rederived: Sequence[Any]) -> Dict[str, VerificationState]:
    """Compare each stored conclusion with a fresh one from the same inputs.

    Matched on the pair the reconciliation is *about*, not on id: a fresh
    derivation mints new ids, so comparing those would report every row as a
    mismatch and the check would be discarded as noise within a week.
    """
    fresh = {(one.planned_ref, one.observed_ref): one for one in rederived}
    states: Dict[str, VerificationState] = {}

    for record in stored:
        key = (record.get("planned_event_id"), record.get("observed_event_id"))
        current = fresh.get(key)
        if current is None:
            states[record["reconciliation_id"]] = \
                VerificationState.DERIVATION_MISMATCH
            continue
        if current.status.value != record["status"]:
            states[record["reconciliation_id"]] = \
                VerificationState.DERIVATION_MISMATCH
            continue

        stored_variances = (record.get("payload") or {}).get("variances") or []
        if [dict(one) for one in current.to_json()["variances"]] != \
                [dict(one) for one in stored_variances]:
            states[record["reconciliation_id"]] = \
                VerificationState.DERIVATION_MISMATCH
            continue
        states[record["reconciliation_id"]] = VerificationState.VERIFIED
    return states
