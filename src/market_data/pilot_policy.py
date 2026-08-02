"""Which snapshots a pilot identity may use at all.

Distinct from the egress policy, which asks *where data may go*. This asks
whether a principal may touch a dataset in the first place, and it sits above
egress because a vendor snapshot reached by an ordinary code path is a licence
problem before it is an export problem.

    egress policy  -> may this leave by this route?
    pilot policy   -> may this principal use this dataset at all?

**Fails closed.** An unset policy raises rather than defaulting, because the
default that matters here is the one nobody chose deliberately, and the vendor
snapshot is already sitting in S3 with six licensing questions open.

**No fallback.** There is no path from a denied vendor snapshot to the synthetic
one. Silently substituting would produce a run whose figures come from data the
plan did not name, which is worse than a refusal a user can read.

Closed pilot v1 is `SYNTHETIC_ONLY` until every licensing question is resolved
affirmatively and recorded. Lifting it introduces a *named, versioned* policy —
`market-data-egress/pilot-vendor-approved@1` — rather than flipping this to a
generic production flag, so the change is reviewable and a stored run says which
authorisation it ran under.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

#: The environment variable a deployment must set. Absent, every pilot request
#: refuses: guidance in a runbook is not a gate.
POLICY_VARIABLE = "PILOT_DATA_POLICY"

#: Snapshot kinds a synthetic-only pilot may use. `licensed` is deliberately
#: absent, and an unrecognised kind is not admitted by omission.
SYNTHETIC_KINDS = frozenset({"synthetic", "fixture"})


class PilotDataPolicy(str, Enum):
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    """Closed pilot v1. Only redistributable synthetic fixtures."""

    PILOT_VENDOR_APPROVED = "market-data-egress/pilot-vendor-approved@1"
    """Named and versioned rather than a generic production flag.

    Valid only once every licensing question carries a recorded answer, source
    reference, reviewer, decision date, permitted audience, permitted derived
    outputs, export rule and retention rule. A version makes that change
    reviewable and lets a stored run say which authorisation produced it."""


class PilotPolicyMissing(RuntimeError):
    """No pilot data policy is configured.

    Refused rather than assumed. The assumption would be made once, by
    deployment, and would then be invisible.
    """


class PilotDataDenied(PermissionError):
    """A pilot principal was about to use a dataset it is not authorised for."""


@dataclass(frozen=True)
class Authorisation:
    """The answer, with enough to explain it rather than assert it."""

    permitted: bool
    policy: Optional[PilotDataPolicy]
    snapshot_id: str
    reason: str

    def to_json(self) -> Dict[str, Any]:
        return {"permitted": self.permitted,
                "policy": self.policy.value if self.policy else None,
                "snapshot_id": self.snapshot_id, "reason": self.reason}


def configured_policy(environ: Optional[Mapping[str, str]] = None
                      ) -> PilotDataPolicy:
    """The deployment's policy, or a refusal."""
    raw = (environ if environ is not None else os.environ).get(POLICY_VARIABLE)
    if not raw:
        raise PilotPolicyMissing(
            f"{POLICY_VARIABLE} is not set. A pilot deployment must state which "
            "data its users may reach; leaving it unset would let an ordinary "
            "code path decide, which is how licensed data reaches an "
            "unauthorised audience")
    try:
        return PilotDataPolicy(raw)
    except ValueError as unknown:
        raise PilotPolicyMissing(
            f"{POLICY_VARIABLE}={raw!r} is not a recognised policy. Valid "
            f"values: {', '.join(one.value for one in PilotDataPolicy)}"
        ) from unknown


def evaluate(snapshot, *, policy: PilotDataPolicy) -> Authorisation:
    """Whether a pilot principal may use this snapshot. Explains either way."""
    kind = str(getattr(snapshot, "kind", "") or "").lower()
    snapshot_id = str(getattr(snapshot, "snapshot_id", "") or "")

    if policy is PilotDataPolicy.SYNTHETIC_ONLY:
        if kind in SYNTHETIC_KINDS and getattr(snapshot, "redistributable",
                                               False):
            return Authorisation(True, policy, snapshot_id,
                                 "a redistributable synthetic fixture")
        return Authorisation(
            False, policy, snapshot_id,
            f"closed pilot v1 is synthetic-only and this snapshot is {kind!r} "
            f"(redistributable={getattr(snapshot, 'redistributable', False)})")

    # The approved policy still requires the licence review to be complete. A
    # deployment flag cannot substitute for reading the agreement.
    if not getattr(snapshot, "review_complete", False):
        return Authorisation(
            False, policy, snapshot_id,
            "the licence review for this snapshot is still open, so the "
            "approved policy does not extend to it")
    return Authorisation(True, policy, snapshot_id,
                         "licence review complete and the policy permits it")


def authorise(snapshot, *, environ: Optional[Mapping[str, str]] = None,
              context: str = "") -> Authorisation:
    """Gate a pilot run. Raises unless the snapshot is permitted.

    Deliberately offers no fallback. A denied vendor snapshot does not become
    the synthetic one: a run whose figures came from data the plan did not name
    is worse than a refusal, because nothing in the result would say so.
    """
    policy = configured_policy(environ)
    verdict = evaluate(snapshot, policy=policy)
    if verdict.permitted:
        return verdict

    where = f" ({context})" if context else ""
    raise PilotDataDenied(
        f"{verdict.snapshot_id or 'this snapshot'} is not available to pilot "
        f"users{where}: {verdict.reason}. Policy {policy.value}.")


#: The six questions that gate lifting SYNTHETIC_ONLY. Each needs a recorded
#: answer, a source or contract reference, a reviewer, a decision date, the
#: permitted audience, the permitted derived outputs, the export rule and the
#: retention rule. Listed here so "resolved" is checkable rather than asserted.
LICENSING_QUESTIONS: Sequence[str] = (
    "may derived results be shown to end users",
    "may derived results be exported in a case bundle",
    "may raw prices be redistributed",
    "may data be sent to a model provider",
    "what retention applies to derived artifacts",
    "what attribution or notice is required",
)

REQUIRED_ANSWER_FIELDS: Sequence[str] = (
    "answer", "source_ref", "reviewer", "decided_on", "audience",
    "derived_outputs", "export_rule", "retention_rule",
)


def licensing_resolved(record: Mapping[str, Mapping[str, Any]]) -> bool:
    """Whether every question carries a complete, recorded answer.

    Checked rather than trusted. "We looked into it" is not a record, and the
    absence of a field is what a rushed review leaves behind.
    """
    for question in LICENSING_QUESTIONS:
        answer = record.get(question)
        if not answer:
            return False
        if any(not answer.get(field) for field in REQUIRED_ANSWER_FIELDS):
            return False
    return True
