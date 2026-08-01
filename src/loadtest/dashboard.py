"""Compiler quality as metrics, not a pass count.

"980 tests passed" says nothing about whether recognition got better or worse.
These metrics do, and each one is derived from the corpus run rather than
declared — a target a caller could assert is a literal, and a literal cannot
regress in a way a test notices.

Every metric here exists because the corpus found the corresponding defect:

    cadence recognition        four of nine cadences had no recognizer
    weighting recognition      a stated rule was dropped before validation
    contradiction detection    94% of contradictions went unreported
    funding-source recognition ordinary wordings were missed
    save-blocking correctness  an offer sat in the blocking list without blocking
    false-inference rate       the direction nobody tests: answering a question
                               the description never settled
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .harness import Outcome, Report
from .paraphrase import Expect, Klass


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    target: float
    unit: str = "%"
    detail: str = ""
    higher_is_better: bool = True

    @property
    def meets_target(self) -> bool:
        return (self.value >= self.target if self.higher_is_better
                else self.value <= self.target)

    def render(self) -> str:
        mark = "ok " if self.meets_target else "MISS"
        if self.unit == "%":
            return (f"  {mark}  {self.name:32} {self.value:7.1f}%  "
                    f"target {self.target:.0f}%   {self.detail}")
        return (f"  {mark}  {self.name:32} {self.value:7.1f}{self.unit}  "
                f"target {self.target:.0f}{self.unit}   {self.detail}")


def _rate(hit: int, total: int) -> float:
    return 100.0 * hit / total if total else 100.0


def _probe_rate(outcomes: Sequence[Outcome], field: str) -> tuple:
    probed = [o for o in outcomes if o.probes == field]
    return _rate(sum(1 for o in probed if o.probe_recognized), len(probed)), len(probed)


def build(report: Report, *, strategies: int = 144) -> List[Metric]:
    outcomes = report.outcomes
    by_class = report.by_class()

    covered = len({o.strategy_id for o in outcomes})
    cadence_rate, cadence_n = _probe_rate(outcomes, "contribution_day_rule")
    weighting_rate, weighting_n = _probe_rate(outcomes, "weighting")
    funding_rate, funding_n = _probe_rate(outcomes, "funding_source")
    trigger_rate, trigger_n = _probe_rate(outcomes, "trigger_semantics")

    contradictory = by_class.get(Klass.CONTRADICTORY.value, {})
    bait = by_class.get(Klass.RECOMMENDATION_BAIT.value, {})
    complete = [o for o in outcomes if o.klass == Klass.COMPLETE.value]

    # The direction nobody tests: a description that settles nothing must not be
    # answered. `can_save` on a prompt that owes a question is a guess presented
    # as a plan.
    should_ask = [o for o in outcomes if o.expect == Expect.ASKS_A_QUESTION.value]
    false_inferences = sum(1 for o in should_ask if o.can_save)

    latency = report.latency()

    return [
        Metric("strategy corpus coverage", _rate(covered, strategies), 100.0,
               detail=f"{covered}/{strategies} rows exercised"),
        Metric("cadence/day-rule recognition", cadence_rate, 100.0,
               detail=f"{cadence_n:,} prompts state it"),
        Metric("weighting rule recognition", weighting_rate, 100.0,
               detail=f"{weighting_n:,} prompts state it"),
        Metric("funding source recognition", funding_rate, 100.0,
               detail=f"{funding_n:,} prompts state it"),
        Metric("trigger semantics recognition", trigger_rate, 100.0,
               detail=f"{trigger_n:,} prompts state it"),
        Metric("contradiction detection",
               _rate(contradictory.get("contradicted", 0),
                     contradictory.get("n", 0)), 100.0,
               detail=f"{contradictory.get('n', 0):,} contradictory prompts"),
        Metric("refuses to recommend",
               _rate(bait.get("n", 0) - bait.get("saveable", 0),
                     bait.get("n", 0)), 100.0,
               detail=f"{bait.get('n', 0):,} recommendation requests"),
        Metric("save-blocking correctness",
               _rate(len(outcomes) - len(report.disagreements), len(outcomes)),
               100.0, detail=f"{len(report.disagreements):,} disagreements"),
        Metric("false inference rate", _rate(false_inferences, len(should_ask)),
               0.0, detail=f"{false_inferences}/{len(should_ask):,} answered "
                           "a question the text never settled",
               higher_is_better=False),
        Metric("crash rate", _rate(len(report.crashes), len(outcomes)), 0.0,
               detail=f"{len(report.crashes)} of {len(outcomes):,}",
               higher_is_better=False),
        Metric("compiler p95 latency", latency["total_us"]["p95"], 100.0,
               unit="us", detail="parse + compile, per description",
               higher_is_better=False),
    ]


def render(metrics: Sequence[Metric]) -> str:
    lines = ["Compiler quality", ""]
    lines += [m.render() for m in metrics]
    missed = [m for m in metrics if not m.meets_target]
    lines += ["", (f"  {len(missed)} metric(s) below target" if missed
                   else "  every metric at target")]
    return "\n".join(lines)


def as_json(metrics: Sequence[Metric]) -> Dict[str, Any]:
    return {m.name: {"value": round(m.value, 2), "target": m.target,
                     "unit": m.unit, "meets_target": m.meets_target,
                     "detail": m.detail} for m in metrics}
