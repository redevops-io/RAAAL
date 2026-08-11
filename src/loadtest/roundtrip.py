"""Round-trip fidelity: Mission -> words -> the same Mission.

    stability      many texts   -> one Mission
    round-trip     one Mission  -> text -> the same Mission

Different directions, different failures. Stability catches a compiler that
reads wording as meaning. Round-trip catches one that *cannot say* what it
understood — a field that survives compilation and has no way back into
language is a field no user can ever correct.

Two modes, because a benchmark that only measures the machine-oriented form can
pass on prose nobody would write:

    SPECIFICATION   claims losslessness, and is held to it
    SUMMARY         ordinary prose, may omit; reports what it dropped

A summary is not required to be perfect. It is required to *say* it is a summary,
so a plan card cannot quietly become an authoritative export that someone
re-imports expecting identical behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..mission.compiler import compile_scenario
from ..mission.evolution import SemanticDiff, diff_stored_against
from ..mission.render import Purpose, render

BENCHMARK_RULE = "benchmark-policy/public-default@1"


@dataclass
class Trip:
    """One Mission, said back and read again."""

    source: str
    purpose: Purpose
    regenerated: str
    omitted: Sequence[str]
    diff: Optional[SemanticDiff] = None
    rule_hash_kept: bool = False
    schedule_hash_kept: bool = False
    content_hash_kept: bool = False
    new_questions: Sequence[str] = ()
    failed: str = ""

    @property
    def exact(self) -> bool:
        return (self.content_hash_kept and self.rule_hash_kept
                and self.schedule_hash_kept and not self.failed)

    def as_row(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose.value, "exact": self.exact,
            "rule_hash_kept": self.rule_hash_kept,
            "schedule_hash_kept": self.schedule_hash_kept,
            "content_hash_kept": self.content_hash_kept,
            "new_questions": list(self.new_questions),
            "omitted": list(self.omitted),
            "changes": [str(c) for c in self.diff.changes] if self.diff else [],
            "failed": self.failed,
            "regenerated": self.regenerated,
        }


def trip(text: str, purpose: Purpose = Purpose.SPECIFICATION) -> Optional[Trip]:
    """Compile, say it back, compile again, compare."""
    first = compile_scenario(text, name="p", version=1,
                             benchmark_rule=BENCHMARK_RULE)
    if first.scenario is None:
        return None

    rendered = render(first.scenario, purpose)
    second = compile_scenario(rendered.text, name="p", version=1,
                              benchmark_rule=BENCHMARK_RULE)
    result = Trip(source=text, purpose=purpose, regenerated=rendered.text,
                  omitted=rendered.omitted)
    if second.scenario is None:
        result.failed = "the regenerated text did not compile to a scenario"
        return result

    result.diff = diff_stored_against(
        first.scenario.to_json(), second.scenario,
        stored_unresolved=[u.field for u in first.unresolved],
        current_unresolved=[u.field for u in second.unresolved])
    result.rule_hash_kept = first.scenario.rule_hash == second.scenario.rule_hash
    result.schedule_hash_kept = (first.scenario.flow_schedule.schedule_hash
                                 == second.scenario.flow_schedule.schedule_hash)
    result.content_hash_kept = (first.scenario.content_hash
                                == second.scenario.content_hash)
    result.new_questions = tuple(result.diff.added_questions)
    return result


def cycles(text: str, rounds: int = 3) -> List[str]:
    """Mission -> text -> Mission -> text -> Mission.

    Identity must not drift over repeated cycles. A round trip that is lossless
    once and shifts on the third pass is worse than one that fails immediately:
    it looks correct in every test anyone writes.
    """
    hashes: List[str] = []
    current = text
    for _ in range(rounds):
        result = compile_scenario(current, name="p", version=1,
                                  benchmark_rule=BENCHMARK_RULE)
        if result.scenario is None:
            break
        hashes.append(result.scenario.content_hash)
        current = render(result.scenario, Purpose.SPECIFICATION).text
    return hashes


@dataclass
class Report:
    trips: List[Trip] = field(default_factory=list)

    def by_purpose(self, purpose: Purpose) -> List[Trip]:
        return [t for t in self.trips if t.purpose is purpose]

    def summarize(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for purpose in Purpose:
            trips = self.by_purpose(purpose)
            if not trips:
                continue
            out[purpose.value] = {
                "n": len(trips),
                "exact": sum(1 for t in trips if t.exact),
                "rule_hash_kept": sum(1 for t in trips if t.rule_hash_kept),
                "schedule_hash_kept": sum(1 for t in trips
                                          if t.schedule_hash_kept),
                "new_questions": sum(len(t.new_questions) for t in trips),
                "failed": sum(1 for t in trips if t.failed),
                "exact_rate": round(
                    100.0 * sum(1 for t in trips if t.exact) / len(trips), 1),
            }
        return out

    def losses(self, purpose: Purpose) -> Dict[str, int]:
        """Which fields a mode drops, counted. The useful output for SUMMARY."""
        counts: Dict[str, int] = {}
        for t in self.by_purpose(purpose):
            for change in (t.diff.changes if t.diff else []):
                counts[change.path] = counts.get(change.path, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def run(texts: Sequence[str],
        purposes: Sequence[Purpose] = (Purpose.SPECIFICATION, Purpose.SUMMARY)
        ) -> Report:
    report = Report()
    for text in texts:
        for purpose in purposes:
            result = trip(text, purpose)
            if result is not None:
                report.trips.append(result)
    return report
