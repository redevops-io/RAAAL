"""Reasoning as versioned artifacts.

    claim/hrp-outperforms-mvo-out-of-sample@1
    assumption/sample-covariance@1
    evidence/lopez-de-prado-2016@1

Methodologies reference claims and assumptions; they do not contain them.
"""
from .artifacts import (
    KNOWLEDGE_SPEC_VERSION,
    Assumption,
    AssumptionKind,
    Claim,
    ClaimStatus,
    Evidence,
    EvidenceKind,
    Finding,
    FindingStatus,
    Impact,
    ImpactRelation,
    Investigation,
    InvestigationOutcome,
    Realization,
    Stance,
)
from .graph import ClaimAssessment, KnowledgeGraph, assess_claim
from .registry import (
    AssumptionRegistry,
    ClaimRegistry,
    EvidenceRegistry,
    FindingRegistry,
    InvestigationRegistry,
)

__all__ = [
    "KNOWLEDGE_SPEC_VERSION",
    "Assumption",
    "AssumptionKind",
    "AssumptionRegistry",
    "Claim",
    "ClaimAssessment",
    "ClaimRegistry",
    "ClaimStatus",
    "Evidence",
    "EvidenceKind",
    "EvidenceRegistry",
    "Finding",
    "FindingRegistry",
    "FindingStatus",
    "Impact",
    "ImpactRelation",
    "Investigation",
    "InvestigationOutcome",
    "InvestigationRegistry",
    "KnowledgeGraph",
    "Realization",
    "Stance",
    "assess_claim",
]
