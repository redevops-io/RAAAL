"""Performance-visual eligibility — why a chart is, or is not, drawn.

A performance visualisation is itself a claim, so it requires the artifacts that
make the claim interpretable. Four gates, all of which must pass.

The object exists so the *absence* of a chart is informative rather than looking
unfinished. "Performance comparison unavailable" with a checklist tells a reader
something true about the data; a blank panel tells them the page is broken.

Graph, dependency, lineage, evidence-balance and timeline visuals are exempt —
they express the provenance model and imply no investment comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Gate:
    code: str
    question: str
    passed: bool
    detail: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "question": self.question,
            "passed": self.passed,
            "detail": self.detail,
            "symbol": "✓" if self.passed else "✕",
        }


@dataclass(frozen=True)
class PerformanceVisualEligibility:
    """The four gates, evaluated together."""

    gates: Sequence[Gate]

    @property
    def eligible(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def failures(self) -> List[Gate]:
        return [g for g in self.gates if not g.passed]

    def headline(self) -> str:
        if self.eligible:
            return "Performance comparison available"
        blocked = ", ".join(g.question.lower() for g in self.failures)
        return f"Performance comparison unavailable — {blocked}"

    def accessibility_summary(self) -> str:
        passed = sum(1 for g in self.gates if g.passed)
        return (
            f"{passed} of {len(self.gates)} conditions met for a performance "
            f"visualisation; {'eligible' if self.eligible else 'not eligible'}."
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "eligible": self.eligible,
            "headline": self.headline(),
            "gates": [g.to_json() for g in self.gates],
            "accessibility_summary": self.accessibility_summary(),
        }


def evaluate_eligibility(
    *,
    comparable: bool,
    comparability_detail: str = "",
    performance_class: Optional[str] = None,
    publication_decision: Optional[str] = None,
    series_encoding_separated: bool = True,
) -> PerformanceVisualEligibility:
    """Evaluate all four gates.

    `series_encoding_separated` guards the single most damaging chart in this
    category: one continuous line running from backtest into live. Carrying a
    performance class is not sufficient if the encoding still blends them.
    """
    return PerformanceVisualEligibility(gates=(
        Gate(
            code="COMPARABLE",
            question="Versions are directly comparable",
            passed=comparable,
            detail=comparability_detail or (
                "" if comparable else "a comparability verdict blocks this pairing"
            ),
        ),
        Gate(
            code="PERFORMANCE_CLASS",
            question="Performance classes are declared",
            passed=performance_class is not None,
            detail=performance_class or "no performance_class attached",
        ),
        Gate(
            code="PUBLICATION_SURFACE",
            question="Publication permits this surface",
            passed=publication_decision in {"ALLOW", "ALLOW_WITH_DISCLOSURE"},
            detail=publication_decision or "no publication decision recorded",
        ),
        Gate(
            code="SERIES_SEPARATION",
            question="Historical and forward series would be separated",
            passed=series_encoding_separated,
            detail=(
                "" if series_encoding_separated
                else "backtest and live would share one unbroken encoding"
            ),
        ),
    ))
