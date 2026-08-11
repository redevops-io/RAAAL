"""Proposal as an artifact, with a lifecycle that never includes execution.

It was a payload field, which was wrong for the same reason a finding was once a
prose paragraph: a thing with a lifecycle, an origin and a fate needs to be
addressable. A proposal is generated from a plan clause, may expire because a
blackout outlasted it, may be superseded by a later one, and may be recorded as
accepted or ignored by the person — none of which a dictionary key can express.

    Intent → Scenario → Mission → Proposal → Observation

**`placed` is a property, not a field.** The object is incapable of reporting
that an order was submitted, rather than defaulting to False and trusting every
caller. Likewise `ExecutionMode` is an enum with one member: adding a second is a
deliberate, reviewable change rather than an edit to a string literal.

`ACCEPTED` means *the person recorded that they acted*, never that the platform
acted. That distinction is the whole basis for this not being execution.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class ExecutionMode(str, Enum):
    """How a proposal may be carried out by this platform.

    One member on purpose. The enum exists so that a future mode is a visible
    change to a type rather than a different string appearing in a payload, and
    so a renderer can switch on it exhaustively today.
    """

    NONE = "NONE"
    """This platform places no orders. It is not an integration that is switched
    off; there is no integration."""


class ProposalStatus(str, Enum):
    OPEN = "OPEN"
    """Generated, not yet resolved."""

    ACCEPTED = "ACCEPTED"
    """The person recorded that they acted on it. The platform did not."""

    IGNORED = "IGNORED"
    """The person recorded that they did not act."""

    EXPIRED = "EXPIRED"
    """Its window closed before it was resolved — usually a blackout outlasting
    it. Retained, because an expired proposal is the evidence that a constraint
    cost something."""

    SUPERSEDED = "SUPERSEDED"
    """A later proposal replaced it."""

    @property
    def is_resolved(self) -> bool:
        return self is not ProposalStatus.OPEN


class Eligibility(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    BLOCKED_BY_WINDOW = "BLOCKED_BY_WINDOW"
    BLOCKED_BY_DATA = "BLOCKED_BY_DATA"
    BLOCKED_BY_UNRESOLVED = "BLOCKED_BY_UNRESOLVED"


class ExecutionAttempted(RuntimeError):
    """Something tried to mark a proposal as placed."""


@dataclass(frozen=True)
class Proposal:
    """What a plan would do next, as an addressable thing with a fate."""

    proposal_id: str
    plan_id: str
    generated_at: str
    generated_from: str
    """The plan clause that produced it. A proposal that cannot name its origin
    is the platform's idea rather than a consequence of the user's."""

    reason: str
    event: str = ""
    """The observed event that triggered it — a vest, a threshold crossing."""

    ticker: str = ""
    notional: float = 0.0
    benchmark_context: Sequence[str] = ()
    """Benchmarks the same event was evaluated against, so the proposal is
    readable beside alternatives rather than alone."""

    eligibility: Eligibility = Eligibility.ELIGIBLE
    detail: str = ""
    expires: Optional[str] = None
    status: ProposalStatus = ProposalStatus.OPEN
    superseded_by: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status is ProposalStatus.SUPERSEDED and not self.superseded_by:
            raise ValueError(
                f"{self.proposal_id}: superseded by nothing. A proposal replaced "
                "by an unnamed successor cannot be traced forward"
            )
        if self.superseded_by and self.status is not ProposalStatus.SUPERSEDED:
            raise ValueError(
                f"{self.proposal_id}: names a successor but its status is "
                f"{self.status.value}"
            )

    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.NONE

    @property
    def placed(self) -> bool:
        """Always False, and not settable.

        A field defaulting to False trusts every future caller not to set it. A
        property cannot be set at all, so a downstream interface cannot render
        "order submitted" because there is no state in which that is true.
        """
        return False

    @property
    def actionable(self) -> bool:
        """Whether the *person* could act on it, not whether anything will."""
        return (self.eligibility is Eligibility.ELIGIBLE
                and self.status is ProposalStatus.OPEN)

    @property
    def artifact_id(self) -> str:
        return f"proposal/{self.proposal_id}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps({
            "plan_id": self.plan_id, "generated_from": self.generated_from,
            "event": self.event, "ticker": self.ticker,
            "notional": self.notional, "generated_at": self.generated_at,
        }, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    def resolve(self, status: ProposalStatus, *,
                superseded_by: Optional[str] = None) -> "Proposal":
        """Return a resolved copy. Proposals are never edited in place.

        The same reason plans are not mutated when reality diverges: the record
        of what was proposed, and when, is the thing being kept.
        """
        if self.status.is_resolved:
            raise ValueError(
                f"{self.proposal_id} is already {self.status.value}; a resolved "
                "proposal is a historical fact and does not change"
            )
        return Proposal(**{**self.__dict__, "status": status,
                           "superseded_by": superseded_by})

    def to_json(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "plan_id": self.plan_id,
            "generated_at": self.generated_at,
            "generated_from": self.generated_from,
            "reason": self.reason,
            "event": self.event,
            "ticker": self.ticker,
            "notional": self.notional,
            "benchmark_context": list(self.benchmark_context),
            "eligibility": self.eligibility.value,
            "detail": self.detail,
            "expires": self.expires,
            "status": self.status.value,
            "superseded_by": self.superseded_by,
            "execution_mode": self.execution_mode.value,
            "placed": self.placed,
            "actionable": self.actionable,
            "note": (
                "This platform places no orders. A proposal records what the "
                "plan you wrote would do next; acting on it, or not, is yours."
            ),
        }


def expire_overdue(proposals: Sequence[Proposal], *, as_of: str) -> List[Proposal]:
    """Expire open proposals whose window has closed.

    Expired proposals are kept rather than deleted: three proposals expiring
    because a blackout outlasted them is precisely the evidence that the
    constraint cost something, and it is only visible if they are still there.
    """
    out: List[Proposal] = []
    for proposal in proposals:
        if (proposal.status is ProposalStatus.OPEN and proposal.expires
                and proposal.expires < as_of):
            out.append(proposal.resolve(ProposalStatus.EXPIRED))
        else:
            out.append(proposal)
    return out


def lifecycle_summary(proposals: Sequence[Proposal]) -> Dict[str, Any]:
    """Counts by status, stated as facts with no next step attached."""
    counts: Dict[str, int] = {}
    for proposal in proposals:
        counts[proposal.status.value] = counts.get(proposal.status.value, 0) + 1

    expired = [p for p in proposals if p.status is ProposalStatus.EXPIRED]
    blocked = [p for p in expired
               if p.eligibility is Eligibility.BLOCKED_BY_WINDOW]
    note = ""
    if blocked:
        note = (
            f"{len(blocked)} proposal(s) expired while a trading window was "
            "closed. History can be replayed assuming they had executed at the "
            "first eligible date, which measures what the constraint cost."
        )
    return {"counts": counts, "expired_in_blackout": len(blocked), "note": note}
