"""Three separate decisions, deliberately not collapsed.

    statistics  — what does the evidence say?      (src/statistics)
    policy      — does it meet a declared standard? (statistical_policy)
    publication — who may see it, and labelled how? (publication)

Each changes for different reasons and at a different rate. A single
`statistical_valid` boolean would freeze all three together and make "valid" mean
too many things at once.
"""
from .publication import (
    HARD_BLOCKERS,
    Decision,
    PublicationDecision,
    Surface,
    decide,
)
from .registry import PolicyRegistry
from .statistical_policy import (
    EvidenceGrade,
    Finding,
    PolicyEvaluation,
    PolicyStatus,
    Requirement,
    Severity,
    StatisticalPolicy,
)

__all__ = [
    "HARD_BLOCKERS",
    "Decision",
    "EvidenceGrade",
    "Finding",
    "PolicyEvaluation",
    "PolicyRegistry",
    "PolicyStatus",
    "PublicationDecision",
    "Requirement",
    "Severity",
    "StatisticalPolicy",
    "Surface",
    "decide",
]
