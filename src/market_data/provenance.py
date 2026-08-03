"""Which market data produced a stored number, recorded with the number.

A figure that cannot say where its data came from is a figure nobody can check
later. Reading the answer from the deployment's *current* configuration is worse
than having none: a run made under snapshot A, reopened after the default moved
to B, would report B and be wrong in a way that looks authoritative.

    RECORDED         this exact snapshot, and the decision that allowed it
    NOT_RECORDED     the run predates provenance; the fact is that nobody knows
    NOT_APPLICABLE   the result contains no market-derived value

**`NOT_RECORDED` is never inferred.** Not from the current environment, not from
the default snapshot, not from a cache, not from the nearest timestamp, not from
matching asset coverage. Historical absence is a fact about the record, and the
same replay-versus-reinterpretation rule already governs scope and compiler
versions.

**The access decision is part of the provenance.** A snapshot id says which data;
it does not say the read was permitted. A result must not exist for a denied
decision, and if direct tampering produces one, verification rejects it rather
than trusting the row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

PROVENANCE_VERSION = "market-data-provenance@1"


class ProvenanceStatus(str, Enum):
    RECORDED = "RECORDED"
    NOT_RECORDED = "MARKET_DATA_PROVENANCE_NOT_RECORDED"
    """A stored value from before provenance was carried. Stated, never
    reconstructed — a reconstructed one would be indistinguishable from a real
    one and wrong."""

    NOT_APPLICABLE = "NOT_APPLICABLE"
    """No market-derived value. Declared rather than omitted, because an
    omitted field and an inapplicable one look identical and mean different
    things."""


class AccessDecision(str, Enum):
    SYNTHETIC_ALLOWED = "SYNTHETIC_ALLOWED"
    VENDOR_INTERNAL_ONLY = "VENDOR_INTERNAL_ONLY"
    PILOT_VENDOR_APPROVED = "PILOT_VENDOR_APPROVED"
    DENIED = "DENIED"


@dataclass(frozen=True)
class MarketDataProvenance:
    """The realized market data behind one stored value."""

    status: ProvenanceStatus
    snapshot_id: Optional[str] = None
    content_digest: Optional[str] = None
    content_digest_version: Optional[str] = None
    license_class: Optional[str] = None
    license_review_status: Optional[str] = None
    policy_version: Optional[str] = None
    access_decision: Optional[AccessDecision] = None
    access_decision_reason: str = ""
    accessed_at: Optional[str] = None
    version: str = PROVENANCE_VERSION

    @property
    def identifies_data(self) -> bool:
        """Whether this names a specific realization.

        A snapshot id alone does not: two snapshots can share a friendly label
        and hold different objects, so the content digest is what distinguishes
        them.
        """
        return (self.status is ProvenanceStatus.RECORDED
                and bool(self.snapshot_id) and bool(self.content_digest))

    @property
    def permitted(self) -> bool:
        return self.access_decision is not AccessDecision.DENIED

    def to_json(self) -> Dict[str, Any]:
        return {"status": self.status.value, "snapshot_id": self.snapshot_id,
                "content_digest": self.content_digest,
                "content_digest_version": self.content_digest_version,
                "license_class": self.license_class,
                "license_review_status": self.license_review_status,
                "policy_version": self.policy_version,
                "access_decision": (self.access_decision.value
                                    if self.access_decision else None),
                "access_decision_reason": self.access_decision_reason,
                "accessed_at": self.accessed_at, "version": self.version}


def from_json(payload: Optional[Mapping[str, Any]]) -> "MarketDataProvenance":
    """Read a stored provenance, treating absence as absence.

    A missing field becomes `NOT_RECORDED` rather than a default, because a
    default would claim knowledge the record does not contain.
    """
    if not payload:
        return not_recorded("no provenance was stored with this record")
    status = ProvenanceStatus(payload.get("status")) \
        if payload.get("status") else ProvenanceStatus.NOT_RECORDED
    decision = payload.get("access_decision")
    return MarketDataProvenance(
        status=status, snapshot_id=payload.get("snapshot_id"),
        content_digest=payload.get("content_digest"),
        content_digest_version=payload.get("content_digest_version"),
        license_class=payload.get("license_class"),
        license_review_status=payload.get("license_review_status"),
        policy_version=payload.get("policy_version"),
        access_decision=AccessDecision(decision) if decision else None,
        access_decision_reason=payload.get("access_decision_reason", ""),
        accessed_at=payload.get("accessed_at"),
        version=payload.get("version", PROVENANCE_VERSION))


def not_recorded(reason: str) -> MarketDataProvenance:
    """For a value whose provenance nobody captured."""
    return MarketDataProvenance(status=ProvenanceStatus.NOT_RECORDED,
                                access_decision_reason=reason)


def not_applicable(reason: str = "no market-derived value") -> MarketDataProvenance:
    return MarketDataProvenance(status=ProvenanceStatus.NOT_APPLICABLE,
                                access_decision_reason=reason)


def recorded(snapshot, *, policy_version: str, decision: AccessDecision,
             accessed_at: str, reason: str = "") -> MarketDataProvenance:
    """From the snapshot that was actually read and the decision that allowed it."""
    return MarketDataProvenance(
        status=ProvenanceStatus.RECORDED,
        snapshot_id=snapshot.snapshot_id,
        content_digest=getattr(snapshot, "content_digest", None),
        content_digest_version=getattr(snapshot, "content_digest_version", None),
        license_class=getattr(snapshot, "license_class", None),
        license_review_status=getattr(snapshot, "license_review_status", None),
        policy_version=policy_version, access_decision=decision,
        access_decision_reason=reason, accessed_at=accessed_at)


def verify(stored: Mapping[str, Any]) -> Sequence[str]:
    """Problems with a stored provenance, read from the record itself.

    Does not consult the environment. The question is whether the record is
    internally coherent, not whether it matches what this deployment happens to
    be configured with today.
    """
    provenance = from_json(stored)
    problems = []

    if provenance.status is ProvenanceStatus.RECORDED:
        if not provenance.snapshot_id:
            problems.append("RECORDED with no snapshot id")
        if not provenance.content_digest:
            problems.append(
                "RECORDED with no content digest, so two snapshots sharing a "
                "label cannot be told apart")
        if provenance.access_decision is None:
            problems.append(
                "RECORDED with no access decision; a snapshot id says which "
                "data, not that the read was permitted")
        elif provenance.access_decision is AccessDecision.DENIED:
            problems.append(
                "a result exists for a DENIED access decision, which should "
                "not have been producible")
        if not provenance.accessed_at:
            problems.append("RECORDED with no access time")
    elif provenance.status is ProvenanceStatus.NOT_RECORDED:
        if provenance.snapshot_id:
            problems.append(
                "NOT_RECORDED and yet names a snapshot; absence that names "
                "something has been reconstructed")
    return tuple(problems)
