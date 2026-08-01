"""Comparability as a prepared view model.

The comparison page answers one question — *can these two figures sit in the
same table?* — and the answer has a fixed shape: a verdict, where the boundary
falls, what blocks it, why each blocker matters, what would restore it, and only
then whether a performance visual may be drawn.

That order is deliberate. Leading with a chart and footnoting the verdict invites
the reader to compare numbers first and discover afterwards that they were never
comparable. Everything here is computed once, so the template arranges prepared
values and calculates nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..methodology.spec import ContractBreak
from .eligibility import PerformanceVisualEligibility, evaluate_eligibility


@dataclass(frozen=True)
class FieldDiff:
    """A field that differs without blocking comparison."""

    field: str
    a: str
    b: str


@dataclass(frozen=True)
class ComparabilityView:
    """Everything the comparison page renders, in the order it renders it."""

    left_id: str
    right_id: str
    comparable: bool
    verdict: str
    verdict_detail: str
    blocking: Sequence[ContractBreak]
    non_blocking: Sequence[FieldDiff]
    restoration: Sequence[str]
    eligibility: PerformanceVisualEligibility

    def to_json(self) -> Dict[str, Any]:
        return {
            "left": self.left_id,
            "right": self.right_id,
            "comparable": self.comparable,
            "verdict": self.verdict,
            "verdict_detail": self.verdict_detail,
            "blocking": [b.to_json() for b in self.blocking],
            "non_blocking": [{"field": d.field, "a": d.a, "b": d.b}
                             for d in self.non_blocking],
            "restoration": list(self.restoration),
            "eligibility": self.eligibility.to_json(),
        }


#: Fields whose difference is the *reason a new version exists*, so surfacing
#: them as "differences" would be noise. `version` is definitionally different;
#: the hashes and rationale are consequences of the change, not the change.
_EXPECTED_TO_DIFFER = {"version", "content_hash", "change_rationale", "created_at"}


def _restoration_steps(blocking: Sequence[ContractBreak],
                       left_id: str, right_id: str) -> List[str]:
    """What would have to be true for these two to become comparable.

    Stated as work someone could actually do. "Not comparable" with no route
    forward reads as a dead end, when in every case here there is a specific
    action — usually re-running the older version under the newer contract.
    """
    if not blocking:
        return []

    steps = [
        f"Re-run {left_id} under {right_id}'s output contract and compare the "
        f"re-run against {right_id}. This is the only route that removes the "
        f"difference rather than hiding it.",
    ]
    fields = {b.field for b in blocking}
    if any(f.startswith("universe") for f in fields):
        steps.append(
            "Alternatively, restrict both to the instruments they share. The "
            "result is comparable but answers a narrower question than either "
            "version was designed for."
        )
    if "rebalance frequency" in fields:
        steps.append(
            "Comparing turnover-adjusted figures is not a substitute: the cost "
            "model is applied inside the run, so it cannot be reversed out afterwards."
        )
    steps.append(
        "Reporting both figures side by side without a comparability note is the "
        "one option that is not available — it is the failure this page exists to "
        "prevent."
    )
    return steps


def build_comparability_view(
    left,
    right,
    *,
    publication_decision: Optional[str] = None,
    performance_class: Optional[str] = None,
    series_encoding_separated: bool = True,
) -> ComparabilityView:
    """Prepare the whole comparison page from two methodology versions."""
    blocking = right.contract.compatibility_breaks(left.contract)
    comparable = not blocking

    la, lb = left.canonical_form(), right.canonical_form()
    blocked_fields = {b.field for b in blocking}
    non_blocking = [
        FieldDiff(field=key, a=str(la.get(key)), b=str(lb.get(key)))
        for key in sorted(set(la) | set(lb))
        if la.get(key) != lb.get(key)
        and key not in _EXPECTED_TO_DIFFER
        and key not in blocked_fields
    ]

    if comparable:
        verdict = "Comparable"
        detail = (
            "These versions share an output contract, so their figures describe "
            "the same promise measured the same way."
        )
    else:
        verdict = "Not comparable"
        noun = "difference" if len(blocking) == 1 else "differences"
        detail = (
            f"{len(blocking)} contract {noun} prevent{'s' if len(blocking) == 1 else ''} "
            f"these figures from being read against each other."
        )

    return ComparabilityView(
        left_id=left.version_id,
        right_id=right.version_id,
        comparable=comparable,
        verdict=verdict,
        verdict_detail=detail,
        blocking=tuple(blocking),
        non_blocking=tuple(non_blocking),
        restoration=tuple(_restoration_steps(blocking, left.version_id, right.version_id)),
        eligibility=evaluate_eligibility(
            comparable=comparable,
            comparability_detail=(
                "" if comparable else
                f"{len(blocking)} blocking contract difference(s)"
            ),
            performance_class=performance_class,
            publication_decision=publication_decision,
            series_encoding_separated=series_encoding_separated,
        ),
    )
