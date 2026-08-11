"""Per-benchmark comparability, persisted rather than recomputed.

The worksheet previously rendered every dimension as "not matched", because a
missing value and a negative verdict looked identical. Those are different
claims:

    false  checked, and different
    null   not checked

A page that shows the second as the first is misleading in the opposite
direction from the usual failure — it looks cautious while being wrong, and a
reader takes it for an actual verdict.

So every dimension carries three states, and an unevaluated one can never render
as negative. The verdicts are computed once, when the run happens, and stored
beside it; the worksheet reads them and never recomputes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..mission.comparability import (
    REGISTERED_DIMENSIONS,
    ComparabilityVerdict,
    RunConditions,
    classify,
)


class Match(str, Enum):
    MATCHED = "MATCHED"
    NOT_MATCHED = "NOT_MATCHED"
    NOT_EVALUATED = "NOT_EVALUATED"
    """No verdict was reached. Distinct from a negative one, and rendered as
    such — a dimension nobody checked is not a dimension that differs."""

    @property
    def is_negative(self) -> bool:
        return self is Match.NOT_MATCHED


#: The dimensions a reader is shown, in the order they matter for reading a
#: comparison: whether the money was the same, over the same time, under the
#: same tax and account treatment, on the same data.
DISPLAYED = ("flow_schedule", "evaluation_period", "tax_treatment", "account",
             "calendar", "market_data", "data_snapshot")


def _state(dimension: str, verdict: ComparabilityVerdict,
           left: RunConditions, right: RunConditions) -> Match:
    """Three states, decided from the verdict *and* the inputs.

    The engine treats two empty hashes as equal, and documents that as an honest
    "not checked here". It is honest in the verdict — `unchecked_dimensions`
    exists — but rendering it as MATCHED would tell a reader the account
    treatment was compared when neither run recorded one. Equality by absence is
    not a match, so it is displayed as NOT_EVALUATED.

    The engine's semantics are left alone deliberately: changing what
    `comparable` means is a change to every stored verdict, and this is a
    display concern.
    """
    if dimension in (verdict.unchecked_dimensions or ()):
        return Match.NOT_EVALUATED
    if dimension not in REGISTERED_DIMENSIONS:
        return Match.NOT_EVALUATED

    a, b = left.dimension_map().get(dimension), right.dimension_map().get(dimension)
    if a in (None, "") and b in (None, ""):
        return Match.NOT_EVALUATED

    return (Match.NOT_MATCHED if dimension in (verdict.differing_dimensions or ())
            else Match.MATCHED)


@dataclass(frozen=True)
class BenchmarkComparability:
    """One benchmark, and whether it can be read beside the strategy."""

    name: str
    comparison_class: str
    comparable: bool
    attribution_isolated: bool
    isolates: str
    dimensions: Mapping[str, str]
    differing: Sequence[str]
    unchecked: Sequence[str]
    disclosure: str
    why_included: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "comparison_class": self.comparison_class,
            "comparable": self.comparable,
            "attribution_isolated": self.attribution_isolated,
            "isolates": self.isolates,
            "dimensions": dict(self.dimensions),
            "differing": list(self.differing),
            "unchecked": list(self.unchecked),
            "disclosure": self.disclosure,
            "why_included": self.why_included,
        }


def record(strategy: RunConditions,
           benchmarks: Mapping[str, RunConditions],
           *, reasons: Optional[Mapping[str, str]] = None
           ) -> List[BenchmarkComparability]:
    """Classify each benchmark against the strategy, once.

    Order follows the mapping the caller passed, which is declaration order.
    Sorting here would be ranking with extra steps, and this is the one place
    that could do it invisibly.
    """
    reasons = reasons or {}
    out: List[BenchmarkComparability] = []
    for name, conditions in benchmarks.items():
        verdict = classify(strategy, conditions)
        out.append(BenchmarkComparability(
            name=name,
            comparison_class=verdict.comparison_class.value,
            comparable=verdict.comparable,
            attribution_isolated=verdict.attribution_isolated,
            isolates=verdict.isolates,
            dimensions={d: _state(d, verdict, strategy, conditions).value
                        for d in DISPLAYED},
            differing=tuple(verdict.differing_dimensions or ()),
            unchecked=tuple(verdict.unchecked_dimensions or ()),
            disclosure=verdict.disclosure_key,
            why_included=reasons.get(name, ""),
        ))
    return out


def as_payload(records: Sequence[BenchmarkComparability]) -> Dict[str, Any]:
    """The shape the worksheet reads. Stored with the run."""
    return {"benchmarks": [r.to_json() for r in records],
            "displayed_dimensions": list(DISPLAYED)}


def from_payload(payload: Optional[Mapping[str, Any]]
                 ) -> List[Dict[str, Any]]:
    """Read back, defaulting every unknown dimension to NOT_EVALUATED.

    The default is the whole point. A benchmark stored before this existed has
    no dimension verdicts at all, and rendering those as differences would
    invent a finding.
    """
    entries = (payload or {}).get("benchmarks") or []
    out = []
    for entry in entries:
        dimensions = dict(entry.get("dimensions") or {})
        out.append({**entry,
                    "dimensions": {d: dimensions.get(d, Match.NOT_EVALUATED.value)
                                   for d in DISPLAYED}})
    return out
