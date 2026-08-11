"""Quantify Methodology Specification 0.1 and its governed merge.

See `spec.py` for the AST and identity rules, `merge.py` for the layered merge,
and `registry.py` for on-disk loading.
"""
from .merge import (
    ComparabilityStatus,
    ContractStatus,
    EconomicStatus,
    MergeResult,
    StructuralStatus,
    merge,
)
from .registry import MethodologyRegistry
from .spec import (
    FIELD_SEMANTICS,
    REQUIRED_DISCLOSURE,
    SPEC_VERSION,
    Citation,
    Methodology,
    OutputContract,
    Param,
    PerformanceClass,
    Rule,
    Semantics,
    from_dict,
)

__all__ = [
    "SPEC_VERSION",
    "FIELD_SEMANTICS",
    "REQUIRED_DISCLOSURE",
    "Citation",
    "Methodology",
    "MethodologyRegistry",
    "OutputContract",
    "Param",
    "PerformanceClass",
    "Rule",
    "Semantics",
    "from_dict",
    "merge",
    "MergeResult",
    "StructuralStatus",
    "ContractStatus",
    "EconomicStatus",
    "ComparabilityStatus",
]
