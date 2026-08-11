"""Publication gate — what a given surface may show, and how it must be labelled.

The third and last decision. The estimators said what the evidence is; the policy
said whether it meets a standard; this decides who may see it and under what
label.

**Failed results are not hidden.** A weak or even failed result remains visible
as a documented research result — that is the library thesis, and suppressing
failures is how a catalogue becomes a highlight reel. What the gate controls is
whether a result may be *represented as validated*, and what disclosure must
travel with it.

Surfaces differ because the claims differ. A private draft asserts nothing. A
public methodology page asserts "this is our documented work". A validated badge
asserts "this survived a strict standard". One universal threshold cannot serve
all three.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .statistical_policy import EvidenceGrade, PolicyEvaluation, PolicyStatus, Severity


class Surface(str, Enum):
    PRIVATE_DRAFT = "PRIVATE_DRAFT"
    RESEARCH_SANDBOX = "RESEARCH_SANDBOX"
    PUBLIC_LIBRARY = "PUBLIC_LIBRARY"
    VALIDATED_BADGE = "VALIDATED_BADGE"
    FORWARD_TRACK_RECORD = "FORWARD_TRACK_RECORD"
    INSTITUTIONAL_EXPORT = "INSTITUTIONAL_EXPORT"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_WITH_DISCLOSURE = "ALLOW_WITH_DISCLOSURE"
    BLOCK = "BLOCK"


#: Defects that make a number *meaningless* rather than merely unimpressive.
#: These block on every surface that makes any claim at all. Note none of them is
#: a performance threshold — a low return is a finding, not a defect.
HARD_BLOCKERS = (
    "missing_run_provenance",
    "contract_violation",
    "incompatible_pairing",
    "unclassified_performance",
    "missing_cost_model",
    "failed_reproducibility",
    "leakage_finding",
    "missing_trial_count",
    "missing_required_statistic",
    "severe_economic_degeneracy",
    "unrealized_declaration",
)


@dataclass(frozen=True)
class PublicationDecision:
    surface: Surface
    decision: Decision
    policy_status: Optional[PolicyStatus]
    evidence_grade: Optional[EvidenceGrade]
    hard_blockers: Sequence[str]
    disclosures: Sequence[str]
    acknowledgements: Sequence[str]
    may_claim_validated: bool
    detail: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "surface": self.surface.value,
            "decision": self.decision.value,
            "policy_status": self.policy_status.value if self.policy_status else None,
            "evidence_grade": self.evidence_grade.value if self.evidence_grade else None,
            "hard_blockers": list(self.hard_blockers),
            "disclosures": list(self.disclosures),
            "acknowledgements": list(self.acknowledgements),
            "may_claim_validated": self.may_claim_validated,
            "detail": self.detail,
        }


def _collect_hard_blockers(
    result_status: Mapping[str, Any],
    assessment,
    compatibility_ok: bool,
    has_performance_class: bool,
    reproducible: bool,
) -> List[str]:
    """Defects that prevent interpretation, independent of any threshold."""
    blockers: List[str] = []

    if not result_status.get("computation_valid", False):
        blockers.append("missing_run_provenance")
    if not result_status.get("contract_valid", False):
        blockers.append("contract_violation")
    if not compatibility_ok:
        blockers.append("incompatible_pairing")
    if not has_performance_class:
        blockers.append("unclassified_performance")
    # `reproducible` arrives from two places — the caller's argument and the
    # result status. Requiring both to be true avoids the case where one source
    # says a run is unreproducible and the other silently overrides it.
    if not reproducible or not result_status.get("reproducible", True):
        blockers.append("failed_reproducibility")
    if assessment is not None and assessment.trial_count < 1:
        blockers.append("missing_trial_count")
    if assessment is not None and assessment.computation_status == "FAILED":
        blockers.append("missing_required_statistic")

    # A methodology that asserts something its own fields do not support is
    # making a claim it cannot keep. This is the second failure mode of a
    # declarative system — declaration without behaviour — and it blocks for the
    # same reason a contract violation does.
    if result_status.get("unrealized_declarations"):
        blockers.append("unrealized_declaration")

    # Economic degeneracy blocks only when severe — a concentrated portfolio is a
    # finding worth publishing with a caveat; a portfolio whose reported ratio is
    # an arithmetic artifact of a near-zero denominator is not interpretable.
    flags = " ".join(result_status.get("flags", []) or [])
    if "degenerate volatility" in flags or "effective breadth" in flags:
        blockers.append("severe_economic_degeneracy")

    return blockers


def decide(
    *,
    surface: Surface,
    result_status: Mapping[str, Any],
    assessment=None,
    policy_evaluation: Optional[PolicyEvaluation] = None,
    compatibility_ok: bool = True,
    has_performance_class: bool = True,
    reproducible: bool = True,
    acknowledgements: Sequence[str] = (),
) -> PublicationDecision:
    """Decide whether `surface` may show this result, and with what label."""
    blockers = _collect_hard_blockers(
        result_status, assessment, compatibility_ok, has_performance_class, reproducible
    )
    # An operator may knowingly accept a documented defect on internal surfaces;
    # the acknowledgement is recorded rather than silently clearing the flag.
    blockers = [b for b in blockers if b not in set(acknowledgements)]

    policy_status = policy_evaluation.status if policy_evaluation else None
    grade = policy_evaluation.evidence_grade if policy_evaluation else None
    disclosures: List[str] = []

    if policy_evaluation:
        for finding in policy_evaluation.warning_findings:
            disclosures.append(f"{finding.code}: {finding.detail}")
        for finding in policy_evaluation.blocking_findings:
            disclosures.append(f"{finding.code}: {finding.detail}")

    # PRIVATE_DRAFT asserts nothing to anyone, so it shows everything.
    if surface is Surface.PRIVATE_DRAFT:
        return PublicationDecision(
            surface=surface, decision=Decision.ALLOW, policy_status=policy_status,
            evidence_grade=grade, hard_blockers=blockers, disclosures=disclosures,
            acknowledgements=acknowledgements, may_claim_validated=False,
            detail="Private draft: all statistics shown, no threshold applied.",
        )

    # INSTITUTIONAL_EXPORT must include failures — an export that omitted them
    # would misrepresent the search, which is the opposite of its purpose.
    if surface is Surface.INSTITUTIONAL_EXPORT:
        return PublicationDecision(
            surface=surface, decision=Decision.ALLOW_WITH_DISCLOSURE,
            policy_status=policy_status, evidence_grade=grade,
            hard_blockers=blockers, disclosures=disclosures + list(blockers),
            acknowledgements=acknowledgements,
            may_claim_validated=policy_status is PolicyStatus.PASS and not blockers,
            detail="Institutional export includes all results and policy outcomes, "
                   "including failures and blockers.",
        )

    if blockers:
        return PublicationDecision(
            surface=surface, decision=Decision.BLOCK, policy_status=policy_status,
            evidence_grade=grade, hard_blockers=blockers, disclosures=disclosures,
            acknowledgements=acknowledgements, may_claim_validated=False,
            detail=(
                "Blocked by defects that make the number uninterpretable: "
                + ", ".join(blockers)
            ),
        )

    if surface is Surface.RESEARCH_SANDBOX:
        return PublicationDecision(
            surface=surface, decision=Decision.ALLOW_WITH_DISCLOSURE,
            policy_status=policy_status, evidence_grade=grade, hard_blockers=[],
            disclosures=disclosures + ["Research sandbox result; not published."],
            acknowledgements=acknowledgements, may_claim_validated=False,
            detail="Sandbox: warnings shown, non-publication status explicit.",
        )

    if surface is Surface.FORWARD_TRACK_RECORD:
        # Backtest statistics do not qualify a forward record; only realized
        # out-of-sample performance does. Ranking on a backtest here would
        # reintroduce exactly the failure this surface exists to avoid.
        return PublicationDecision(
            surface=surface, decision=Decision.BLOCK, policy_status=policy_status,
            evidence_grade=grade, hard_blockers=["backtest_not_eligible_for_forward_surface"],
            disclosures=disclosures, acknowledgements=acknowledgements,
            may_claim_validated=False,
            detail="Forward track-record ranking uses realized out-of-sample "
                   "performance only; backtested results are not eligible.",
        )

    if surface is Surface.VALIDATED_BADGE:
        eligible = policy_status is PolicyStatus.PASS and grade is EvidenceGrade.STRONG
        return PublicationDecision(
            surface=surface,
            decision=Decision.ALLOW if eligible else Decision.BLOCK,
            policy_status=policy_status, evidence_grade=grade, hard_blockers=[],
            disclosures=disclosures, acknowledgements=acknowledgements,
            may_claim_validated=eligible,
            detail=(
                "Validated badge requires a passing policy and STRONG evidence."
                if not eligible
                else "Meets the validated standard."
            ),
        )

    # PUBLIC_LIBRARY — a weak result is publishable, but must say it is weak.
    if policy_status is PolicyStatus.PASS and not disclosures:
        decision = Decision.ALLOW
    else:
        decision = Decision.ALLOW_WITH_DISCLOSURE
        if policy_status is PolicyStatus.FAIL:
            disclosures.append(
                "This result does not meet the library's statistical policy and is "
                "published as a documented research result, not a validated one."
            )

    return PublicationDecision(
        surface=surface, decision=decision, policy_status=policy_status,
        evidence_grade=grade, hard_blockers=[], disclosures=disclosures,
        acknowledgements=acknowledgements,
        may_claim_validated=(
            policy_status is PolicyStatus.PASS and grade is EvidenceGrade.STRONG
        ),
        detail="Public library: statistically weak results are publishable when "
               "labelled as such; only a passing policy with strong evidence may "
               "be represented as validated.",
    )
