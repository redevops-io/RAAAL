"""Would today's standard reach yesterday's verdict?

This is the question the three-layer split exists to make answerable. Statistical
assessment records *facts*; policy applies a *standard*; the two are separate
artifacts, so a standard can change without the facts changing. Re-judging a
recorded assessment under the current policy is therefore not a recomputation —
nothing about the run is re-derived — it is one question asked twice.

The result is deliberately kept out of the run's chain. A run keeps the verdict
it received; what today's policy would say is a second, clearly labelled fact.
Merging them would destroy the ledger's only guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..statistics.assessment import StatisticalAssessment


@dataclass(frozen=True)
class RequirementChange:
    """One requirement whose outcome moved between the two policies."""

    code: str
    description: str
    then: str
    now: str
    severity: str


@dataclass(frozen=True)
class PolicyDrift:
    """The recorded verdict beside the verdict today's policy would give."""

    recorded_policy_id: str
    current_policy_id: str
    recorded_status: str
    current_status: str
    recorded_grade: str
    current_grade: str
    changes: Sequence[RequirementChange]
    comparable: bool = True
    detail: str = ""

    @property
    def same_policy(self) -> bool:
        return self.recorded_policy_id == self.current_policy_id

    @property
    def differs(self) -> bool:
        return (self.recorded_status != self.current_status
                or self.recorded_grade != self.current_grade)

    def headline(self) -> str:
        if not self.comparable:
            return "Today's policy cannot be applied to this run"
        if self.same_policy and not self.differs:
            return "The same policy is in force, and it still reaches this verdict"
        if not self.differs:
            return (f"{self.current_policy_id} reaches the same verdict as "
                    f"{self.recorded_policy_id}")
        return (f"{self.current_policy_id} would reach {self.current_status}, "
                f"not {self.recorded_status}")

    def to_json(self) -> Dict[str, Any]:
        return {
            "recorded_policy_id": self.recorded_policy_id,
            "current_policy_id": self.current_policy_id,
            "recorded_status": self.recorded_status,
            "current_status": self.current_status,
            "recorded_grade": self.recorded_grade,
            "current_grade": self.current_grade,
            "same_policy": self.same_policy,
            "differs": self.differs,
            "comparable": self.comparable,
            "headline": self.headline(),
            "detail": self.detail,
            "changes": [
                {"code": c.code, "description": c.description, "then": c.then,
                 "now": c.now, "severity": c.severity}
                for c in self.changes
            ],
        }


def _outcomes(evaluation) -> Dict[str, Any]:
    """Requirement code -> (passed, severity, detail), from either form.

    A recorded evaluation arrives as a dict and a fresh one as an object; both
    are read here rather than at two call sites, so the comparison cannot end up
    depending on which side it came from.
    """
    findings = (evaluation.get("findings", []) if isinstance(evaluation, dict)
                else [f.to_json() for f in evaluation.findings])
    return {f["code"]: f for f in findings}


def evaluate_drift(recorded_run: Dict[str, Any], policy, now: str) -> Optional[PolicyDrift]:
    """Re-judge a run's recorded assessment under `policy`.

    Returns None when the run predates the assessment layer entirely — an honest
    absence, rather than a comparison against a reconstructed zero.
    """
    recorded_policy = recorded_run.get("policy_evaluation")
    recorded_assessment = recorded_run.get("assessment")
    if not recorded_policy or not recorded_assessment:
        return None

    try:
        assessment = StatisticalAssessment.from_json(recorded_assessment)
    except TypeError:
        # A schema change since the run was recorded. Say so; do not guess.
        return PolicyDrift(
            recorded_policy_id=recorded_policy.get("policy_id", "—"),
            current_policy_id=policy.policy_id,
            recorded_status=recorded_policy.get("status", "—"),
            current_status="—", recorded_grade=recorded_policy.get("evidence_grade", "—"),
            current_grade="—", changes=(), comparable=False,
            detail="The recorded assessment predates the current assessment schema, "
                   "so today's policy cannot be applied to it without re-running.",
        )

    current = policy.evaluate(assessment, now=now)
    then, now_out = _outcomes(recorded_policy), _outcomes(current)

    changes: List[RequirementChange] = []
    for code in sorted(set(then) | set(now_out)):
        a, b = then.get(code), now_out.get(code)
        a_state = "not checked" if a is None else ("pass" if a["passed"] else "fail")
        b_state = "not checked" if b is None else ("pass" if b["passed"] else "fail")
        if a_state != b_state:
            source = b or a
            changes.append(RequirementChange(
                code=code,
                description=source.get("description") or source.get("detail", ""),
                then=a_state, now=b_state,
                severity=source.get("severity", "—"),
            ))

    return PolicyDrift(
        recorded_policy_id=recorded_policy.get("policy_id", "—"),
        current_policy_id=current.policy_id,
        recorded_status=recorded_policy.get("status", "—"),
        current_status=current.status.value,
        recorded_grade=recorded_policy.get("evidence_grade", "—"),
        current_grade=current.evidence_grade.value,
        changes=tuple(changes),
        detail=(
            "The facts are the recorded facts. Only the standard applied to them "
            "differs, which is the whole reason assessment and policy are "
            "separate artifacts."
        ),
    )
