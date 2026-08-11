"""Whether a scenario response is a recommendation — derived, not declared.

The first version of `comparison_payload` emitted `"is_recommendation": False`
as a literal. That is worthless: it is the platform asserting its own compliance,
and it is the same defect as a methodology declaring a rule the executor never
enforces. A flag that cannot be wrong is not evidence of anything.

So the property is decomposed into checks, each of which is either **derived**
from the payload or **declared** by the caller, and the difference is recorded.
A response is non-recommendational only when every check passes, and a response
resting on declared checks is visibly weaker evidence than one resting on derived
ones.

The 206(4)-1 interactive-analysis carve-out is narrow and conditional — it turns
on required disclosures being present — and it governs adviser advertising rather
than settling adviser status. Nothing here is a legal conclusion; it is the
engineering posture made checkable so that counsel has something concrete to
review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class Basis(str, Enum):
    DERIVED = "DERIVED"
    """Computed from the payload. Cannot be wrong without the payload being wrong."""

    DECLARED = "DECLARED"
    """Asserted by the caller. Recorded as weaker, never treated as equivalent."""


@dataclass(frozen=True)
class Check:
    code: str
    question: str
    must_be: bool
    observed: bool
    basis: Basis
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.observed is self.must_be

    def to_json(self) -> Dict[str, Any]:
        return {"code": self.code, "question": self.question,
                "must_be": self.must_be, "observed": self.observed,
                "basis": self.basis.value, "passed": self.passed,
                "detail": self.detail}


@dataclass(frozen=True)
class RecommendationAssessment:
    """The nine conditions, evaluated together."""

    checks: Sequence[Check]

    @property
    def is_recommendation(self) -> bool:
        """Fails closed: any failing check makes this a recommendation."""
        return any(not c.passed for c in self.checks)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def derivation_complete(self) -> bool:
        """Whether the verdict rests on the payload alone."""
        return all(c.basis is Basis.DERIVED for c in self.checks)

    @property
    def declared_only(self) -> List[Check]:
        return [c for c in self.checks if c.basis is Basis.DECLARED]

    def headline(self) -> str:
        if self.is_recommendation:
            return ("This response would read as a recommendation: "
                    + ", ".join(c.code.lower().replace("_", " ") for c in self.failures))
        if not self.derivation_complete:
            return ("No condition failed, but "
                    f"{len(self.declared_only)} condition(s) rest on assertion "
                    "rather than on the payload")
        return "Every condition is satisfied and derived from the payload"

    def to_json(self) -> Dict[str, Any]:
        return {
            "is_recommendation": self.is_recommendation,
            "derivation_complete": self.derivation_complete,
            "headline": self.headline(),
            "checks": [c.to_json() for c in self.checks],
            "failures": [c.code for c in self.failures],
        }


#: Language that turns an analytical statement into a prescriptive one. Crude by
#: design: a lexical rule is deterministic and testable, where an instruction to
#: "stay neutral" in a system prompt is neither. False positives cost a rephrase;
#: false negatives cost the posture.
_PRESCRIPTIVE = re.compile(
    r"\b("
    r"you should|we recommend|recommended for you|best for (?:you|your)|"
    r"consider (?:switching|moving|buying|selling)|"
    r"would you like to see why|optimal for your|right for you|"
    r"we suggest|our advice|you ought to|better choice for you"
    r")\b",
    re.IGNORECASE,
)

#: Superlatives are only a problem when attached to the reader. "Lowest drawdown"
#: is a fact about a series; "best for your situation" is a judgement about a
#: person, and the second is what the posture forbids.
_PERSONALIZED_SUPERLATIVE = re.compile(
    r"\b(best|worst|optimal|ideal|safest|strongest)\b[^.]{0,40}\b"
    r"(for you|for your|your situation|your goals|your profile)\b",
    re.IGNORECASE,
)

_PEER_BEHAVIOUR = re.compile(
    r"\b(people like you|investors like you|others (?:with|in) your|"
    r"users (?:with|like)|similar (?:investors|users|people)|typically switched)\b",
    re.IGNORECASE,
)

_TRADE_INSTRUCTION = re.compile(
    r"\b(place (?:an? )?order|execute (?:this |the )?trade|buy now|sell now|"
    r"rebalance now|submit (?:this )?order)\b",
    re.IGNORECASE,
)


def scan_language(text: str) -> Dict[str, bool]:
    """Deterministic lexical checks over everything the user will read."""
    return {
        "next_action_suggestion": bool(_PRESCRIPTIVE.search(text)),
        "personalized_superlative_language": bool(_PERSONALIZED_SUPERLATIVE.search(text)),
        "peer_behavior_used": bool(_PEER_BEHAVIOUR.search(text)),
        "execution_or_trade_instruction": bool(_TRADE_INSTRUCTION.search(text)),
    }


def assess(
    *,
    benchmarks: Sequence[Any],
    rendered_text: str = "",
    user_originated_rule: Optional[bool] = None,
    platform_generated_action: Optional[bool] = None,
    portfolio_selection_performed: Optional[bool] = None,
    declared_order: Optional[Sequence[str]] = None,
    payload_order: Optional[Sequence[str]] = None,
    ordering_metric: Optional[Sequence[float]] = None,
) -> RecommendationAssessment:
    """Evaluate all nine conditions over a prepared comparison.

    The ordering check compares the payload's order against the order the
    benchmarks were *declared* in, which is exact. Detecting "is this sorted by
    outcome?" was the first attempt and was wrong: three benchmarks fall into
    sorted order by chance a third of the time, so the check fired on payloads
    that had done nothing. A heuristic that cries wolf gets switched off, and a
    switched-off check protects nothing.

    Coincidental sorting is still reported, as detail rather than as a failure —
    a set declared before results cannot have been ordered by them.
    """
    checks: List[Check] = []

    comparable = [b for b in benchmarks if getattr(b, "comparable", False)]
    symmetric, symmetry_detail = _symmetry(comparable)
    checks.append(Check(
        code="BENCHMARK_SET_SYMMETRIC",
        question="Did every benchmark receive identical flows, costs and period?",
        must_be=True, observed=symmetric, basis=Basis.DERIVED,
        detail=symmetry_detail,
    ))

    if declared_order is not None and payload_order is not None:
        reordered = list(declared_order) != list(payload_order)
        detail = ("the payload was reordered after the set was declared, which is "
                  "a ranking however it is labelled" if reordered else "")
        if not reordered and ordering_metric and _is_sorted(ordering_metric):
            detail = ("results happen to fall in payload order; the order was "
                      "fixed before they were known")
        checks.append(Check(
            code="BENCHMARK_ORDER_SEMANTICALLY_UNRANKED",
            question="Does the payload preserve the declared order?",
            must_be=False, observed=reordered, basis=Basis.DERIVED, detail=detail,
        ))
    else:
        checks.append(Check(
            code="BENCHMARK_ORDER_SEMANTICALLY_UNRANKED",
            question="Does the payload preserve the declared order?",
            must_be=False, observed=False, basis=Basis.DECLARED,
            detail="declared order not supplied, so preservation cannot be verified",
        ))

    language = scan_language(rendered_text)
    for code, must_be_false in (
        ("NEXT_ACTION_SUGGESTION", "next_action_suggestion"),
        ("PERSONALIZED_SUPERLATIVE_LANGUAGE", "personalized_superlative_language"),
        ("PEER_BEHAVIOR_USED", "peer_behavior_used"),
        ("EXECUTION_OR_TRADE_INSTRUCTION", "execution_or_trade_instruction"),
    ):
        found = language[must_be_false]
        checks.append(Check(
            code=code,
            question=f"Is the response free of {must_be_false.replace('_', ' ')}?",
            must_be=False, observed=found,
            basis=Basis.DERIVED if rendered_text else Basis.DECLARED,
            detail="matched in the rendered response" if found else "",
        ))

    for code, question, value, must_be in (
        ("USER_ORIGINATED_RULE",
         "Did the rule come from the user rather than the platform?",
         user_originated_rule, True),
        ("PLATFORM_GENERATED_ACTION",
         "Did the platform propose an action?", platform_generated_action, False),
        ("PORTFOLIO_SELECTION_PERFORMED",
         "Did the platform choose the holdings?", portfolio_selection_performed, False),
    ):
        checks.append(Check(
            code=code, question=question, must_be=must_be,
            observed=must_be if value is None else bool(value),
            basis=Basis.DECLARED,
            detail=("not supplied; assumed compliant and recorded as declared"
                    if value is None else ""),
        ))

    return RecommendationAssessment(checks=tuple(checks))


def _symmetry(benchmarks: Sequence[Any]) -> tuple:
    """Every benchmark must have been funded and charged identically."""
    if len(benchmarks) < 2:
        return True, "fewer than two comparable benchmarks"

    signature = set()
    for b in benchmarks:
        path = b.result.path
        signature.add((
            round(path.contributed, 6),
            round(path.withdrawn, 6),
            len(path.value),
            (path.cash_policy.annual_rate if path.cash_policy else None),
        ))
    if len(signature) == 1:
        return True, "identical flows, period and cash policy across the set"
    return False, (
        "benchmarks were funded or evaluated differently, so any difference "
        "between them cannot be attributed to the strategy"
    )


def _is_sorted(values: Sequence[float]) -> bool:
    numeric = [v for v in values if v is not None]
    if len(numeric) < 3:
        # Two elements are sorted half the time by chance; calling that a ranking
        # would fail closed on nothing.
        return False
    ascending = all(a <= b for a, b in zip(numeric, numeric[1:]))
    descending = all(a >= b for a, b in zip(numeric, numeric[1:]))
    return ascending or descending
