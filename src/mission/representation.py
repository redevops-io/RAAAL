"""Recognition is not representation.

The compiler can be correct at two layers and wrong in the middle:

    Recognized    the parser read the phrase             correct
    Represented   it reached the compiled scenario       MISSING
    Validated     checks ran over what was represented   vacuous
    Compiled      a scenario was produced                correct
    Confirmed     the screen quoted the phrase back      correct

Both defects found this way had exactly that shape. A user said "hold the
dividends as cash", the confirmation screen quoted it under *you stated*, and
the compiled scenario contained no trace of it — reinvesting and holding as cash
produced an identical content hash. The same was true of "simple" versus
"exponential" moving average, which cross at different times and are therefore
different rules.

Neither was a parser bug. The parser was right both times. The gap was between
reading a thing and *carrying* it, and nothing checked that edge because both
ends looked fine.

This module makes the edge checkable. Every field stage 1 can recognise must
either name where it lands in the compiled form, or say why it does not travel.
A field in neither list is a defect, and adding a recogniser without a
destination now fails a test instead of silently doing nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: recognised field -> where it must appear in `ScenarioSpecification`.
#: A dotted path into `canonical_form()`, or a `(path, key)` pair for a value
#: that must appear inside a list of steps.
CARRIES: Mapping[str, str] = {
    "trigger_semantics": "methodology.event_program",
    "moving_average_kind": "methodology.event_program",
    "weighting": "methodology.allocation_rule.weighting",
    "sells_allowed": "methodology.holdings_policy.sells_allowed",
    "dividends": "methodology.holdings_policy.dividend_policy",
    "cadence": "flows.cadence",
    "amount": "flows.amount_per_period",
    "contribution_day_rule": "flows.day_rule",
    "funding_source": "flows.funding_source",
}

@dataclass(frozen=True)
class Exemption:
    """A recognised field deliberately not carried, and who owes it.

    `owner` and `blocked_by` are what make an exemption expire. Without them
    "deliberately uncarried" is a permanent hiding place: nothing can find the
    entries that became obsolete when the responsible artifact finally shipped.
    """

    status: str                      # "delegated" | "unsupported"
    reason: str
    owner: Optional[str] = None      # the artifact that realizes it instead
    blocked_by: Optional[str] = None  # the capability that would unblock it

    def __post_init__(self) -> None:
        if self.status == "delegated" and not self.owner:
            raise ValueError(
                "a delegated field must name the artifact that realizes it, or "
                "nothing can tell whether the delegation is still true")
        if self.status == "unsupported" and not self.blocked_by:
            raise ValueError(
                "an unsupported field must name what would unblock it, or the "
                "exemption cannot expire when that arrives")


#: Recognised, deliberately not carried into the scenario, and why. An entry
#: here is a decision someone wrote down; the absence of one is a defect.
UNCARRIED: Mapping[str, Exemption] = {
    "earnings_timing": Exemption(
        status="unsupported",
        blocked_by="EarningsCalendarRuntime",
        reason=(
            "The engine has no earnings calendar, so an earnings-relative rule "
            "cannot be executed. Recognised so the confirmation screen can say "
            "the description mentions one and it is not being simulated."),
    ),
    "vesting_action": Exemption(
        status="delegated",
        owner="template/rsu-vesting@1",
        reason=(
            "Vesting is owned by the RSU life-event template, which states "
            "cited rules. The generic scenario deliberately does not paraphrase "
            "them."),
    ),
}


def obsolete_exemptions(available: Sequence[str]) -> List[str]:
    """Exemptions whose blocker or owner now exists.

    Queried by a migration check rather than remembered. The reason an
    exemption was written is the reason it should stop being one.
    """
    present = set(available)
    return sorted(
        field for field, exemption in UNCARRIED.items()
        if (exemption.blocked_by in present) or (exemption.owner in present))


@dataclass(frozen=True)
class Gap:
    """A recognised field that did not reach the compiled scenario."""

    field: str
    value: str
    span: str
    expected_at: Optional[str]

    def __str__(self) -> str:
        where = f" (expected at {self.expected_at})" if self.expected_at else ""
        return (f"{self.field}={self.value!r} was read from {self.span!r} and "
                f"does not appear in the compiled scenario{where}")


def _walk(node: Any) -> str:
    return json.dumps(node, sort_keys=True, default=str)


def _at(form: Mapping[str, Any], path: str) -> Any:
    node: Any = form
    for part in path.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return None
    return node


def undeclared_fields(known: Sequence[str]) -> List[str]:
    """Recognisable fields that name neither a destination nor a reason.

    The check that stops this file going stale. A new recogniser with no entry
    fails here rather than quietly reading something nothing carries.
    """
    return sorted(f for f in known
                  if f not in CARRIES and f not in UNCARRIED)


def representation_gaps(parsed, scenario) -> List[Gap]:
    """Which recognised values did not survive into the compiled scenario.

    Presence is checked against the *canonical form*, because that is what the
    content hash is taken over and therefore what identity, comparison and
    replay all see. A value carried on the object but excluded from the
    canonical form is still invisible to everything that matters.
    """
    if scenario is None:
        return []

    form = scenario.canonical_form()
    gaps: List[Gap] = []
    for recognition in parsed.recognitions:
        field = recognition.field
        if field in UNCARRIED:
            continue
        destination = CARRIES.get(field)
        node = _at(form, destination) if destination else form
        haystack = _walk(node if node is not None else form)
        value = str(recognition.value)
        # `sells_allowed` is recognised as the string "false" and represented as
        # a boolean, so compare on the represented form rather than the token.
        candidates = {value, value.lower(),
                      {"false": "false", "true": "true"}.get(value.lower(), value)}
        if not any(c in haystack for c in candidates):
            gaps.append(Gap(field=field, value=value, span=recognition.span,
                            expected_at=destination))
    return gaps
