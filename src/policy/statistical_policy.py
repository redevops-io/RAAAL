"""Versioned statistical policy — what standard applies.

The estimators say what the evidence is. This layer says what is required. It is
a first-class artifact for the same reason the methodology and the protocol are:
a threshold that lives in code is a hidden choice, and changing it silently
reprices every result ever judged under it.

Severity is deliberately three-valued. A statistically weak result is still a
result: the library thesis is that a modest, fully-documented finding is more
valuable than an impressive one with hidden degeneracy, and blocking everything
under a single numeric bar would contradict that. So `WARN` exists to publish a
weak result *labelled as weak*, and `BLOCK` is reserved for defects that make a
number meaningless rather than merely unimpressive.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

POLICY_SPEC_VERSION = "0.1"


class Severity(str, Enum):
    BLOCK = "BLOCK"
    """The number cannot be interpreted. Not a judgement about quality."""

    WARN = "WARN"
    """The number is interpretable but weak; publish with the weakness stated."""

    INFO = "INFO"
    """Worth recording, no consequence for publication."""


class PolicyStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class EvidenceGrade(str, Enum):
    """How much weight the evidence bears. Orthogonal to whether it is valid."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class Requirement:
    """One checkable condition, with the severity of failing it."""

    code: str
    description: str
    severity: Severity
    threshold: Optional[float] = None
    comparison: str = "gte"          # gte | lte | eq | truthy

    def to_json(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "description": self.description,
            "severity": self.severity.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
        }


@dataclass(frozen=True)
class Finding:
    """The outcome of evaluating one requirement."""

    code: str
    severity: Severity
    passed: bool
    observed: Any
    threshold: Optional[float]
    detail: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "passed": self.passed,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PolicyEvaluation:
    """The verdict of one policy against one assessment."""

    policy_id: str
    policy_hash: str
    status: PolicyStatus
    evidence_grade: EvidenceGrade
    findings: Sequence[Finding]
    evaluated_at: str

    @property
    def blocking_findings(self) -> List[Finding]:
        return [f for f in self.findings if not f.passed and f.severity is Severity.BLOCK]

    @property
    def warning_findings(self) -> List[Finding]:
        return [f for f in self.findings if not f.passed and f.severity is Severity.WARN]

    def to_json(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "status": self.status.value,
            "evidence_grade": self.evidence_grade.value,
            "findings": [f.to_json() for f in self.findings],
            "evaluated_at": self.evaluated_at,
        }


@dataclass(frozen=True)
class StatisticalPolicy:
    """A named, versioned set of statistical requirements."""

    name: str
    version: int
    title: str
    requirements: Sequence[Requirement]
    rationale: str = ""
    spec_version: str = POLICY_SPEC_VERSION

    @property
    def policy_id(self) -> str:
        return f"stat-policy/{self.name}@{self.version}"

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "requirements": sorted(
                (r.to_json() for r in self.requirements), key=lambda r: r["code"]
            ),
        }

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.canonical_form(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "policy_id": self.policy_id,
            "content_hash": self.content_hash,
            "title": self.title,
            "rationale": self.rationale,
        }

    # ---- evaluation -------------------------------------------------------

    def evaluate(self, assessment, *, now: str) -> PolicyEvaluation:
        """Judge an assessment. Reads facts; produces a status, never a number."""
        findings: List[Finding] = []
        observed = _extract(assessment)

        for requirement in self.requirements:
            value = observed.get(requirement.code)
            passed, detail = _check(requirement, value)
            findings.append(
                Finding(
                    code=requirement.code,
                    severity=requirement.severity,
                    passed=passed,
                    observed=value,
                    threshold=requirement.threshold,
                    detail=detail,
                )
            )

        failed_block = [f for f in findings if not f.passed and f.severity is Severity.BLOCK]
        failed_warn = [f for f in findings if not f.passed and f.severity is Severity.WARN]

        if failed_block:
            status = PolicyStatus.FAIL
        elif failed_warn:
            status = PolicyStatus.WARN
        else:
            status = PolicyStatus.PASS

        return PolicyEvaluation(
            policy_id=self.policy_id,
            policy_hash=self.content_hash,
            status=status,
            evidence_grade=_grade(observed, status),
            findings=findings,
            evaluated_at=now,
        )


def _extract(assessment) -> Dict[str, Any]:
    """Flatten an assessment into the codes requirements refer to."""
    dsr = (assessment.dsr or {}).get("value")
    psr = (assessment.psr or {}).get("value")
    pbo = (assessment.pbo or {}).get("value") if assessment.pbo else None
    mtrl = assessment.min_track_record_length or {}

    return {
        "minimum_observations": assessment.observations,
        "minimum_dsr": dsr,
        "minimum_psr": psr,
        "maximum_pbo": pbo,
        "require_trial_count": assessment.trial_count,
        "require_factor_neutralization": assessment.factor_neutralization is not None,
        "require_complete_computation": assessment.computation_status == "VALID",
        "sufficient_track_record": mtrl.get("sufficient"),
    }


def _check(requirement: Requirement, value: Any) -> tuple[bool, str]:
    if value is None:
        return False, f"{requirement.code} was not computed"

    if requirement.comparison == "truthy":
        passed = bool(value)
        return passed, ("satisfied" if passed else f"{requirement.code} not satisfied")

    if requirement.threshold is None:
        return True, "no threshold declared"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False, f"{requirement.code} value {value!r} is not numeric"

    if requirement.comparison == "gte":
        passed = numeric >= requirement.threshold
        detail = f"{numeric:.4f} {'>=' if passed else '<'} {requirement.threshold}"
    elif requirement.comparison == "lte":
        passed = numeric <= requirement.threshold
        detail = f"{numeric:.4f} {'<=' if passed else '>'} {requirement.threshold}"
    else:
        passed = numeric == requirement.threshold
        detail = f"{numeric} vs {requirement.threshold}"
    return passed, detail


def _grade(observed: Mapping[str, Any], status: PolicyStatus) -> EvidenceGrade:
    """Grade the weight of evidence, separately from whether it passed.

    A result can pass a permissive policy on thin evidence, or fail a strict one
    while still being moderately well evidenced. The grade describes the evidence;
    the status describes conformance to a standard.
    """
    if status is PolicyStatus.FAIL and not observed.get("require_complete_computation"):
        return EvidenceGrade.INSUFFICIENT

    dsr = observed.get("minimum_dsr")
    pbo = observed.get("maximum_pbo")
    obs = observed.get("minimum_observations") or 0

    if dsr is None:
        return EvidenceGrade.INSUFFICIENT

    strong = dsr >= 0.95 and obs >= 756 and (pbo is None or pbo <= 0.25)
    moderate = dsr >= 0.75 and obs >= 504 and (pbo is None or pbo <= 0.50)

    if strong:
        return EvidenceGrade.STRONG
    if moderate:
        return EvidenceGrade.MODERATE
    return EvidenceGrade.WEAK
