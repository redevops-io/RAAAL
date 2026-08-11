"""How money arrives and leaves — semantics, not amounts.

Two layers, kept apart for the same reason methodology semantics are kept apart
from methodology parameters:

    CashFlowRuntime   what a "monthly contribution" or an "RSU vest" *means*
    FlowSchedule      this person's amounts and dates

The runtime is reusable and public: it knows that a vest delivers shares in kind,
that a paycheque lands on a session rather than a calendar date, that a
withdrawal may be refused by an account. The schedule is one person's salary and
is never any of those things. Putting personal values inside a reusable artifact
is the boundary violation the whole workspace split exists to prevent.

This is also the runtime that lets `ISOLATION_DIMENSIONS` stop being maintained
by hand: `flow_schedule` and `starting_capital` were the last two dimensions
living outside runtime registration, so with this present the comparison surface
can be generated from registered kinds instead of curated.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Sequence

from .base import Exclusion, RuntimeArtifact, RuntimeAssumption, RuntimeLimitation


class FlowKind(str, Enum):
    """Sources of money, each of which behaves differently."""

    SALARY = "SALARY"
    BONUS = "BONUS"
    RSU_VEST = "RSU_VEST"
    """Arrives as shares, not cash. Not a purchase."""

    ESPP_PURCHASE = "ESPP_PURCHASE"
    DIVIDEND = "DIVIDEND"
    PENSION = "PENSION"
    SOCIAL_SECURITY = "SOCIAL_SECURITY"
    RENTAL_INCOME = "RENTAL_INCOME"
    INHERITANCE = "INHERITANCE"
    WITHDRAWAL = "WITHDRAWAL"


class DayRule(str, Enum):
    FIRST_SESSION_OF_PERIOD = "first_session_of_period"
    LAST_SESSION_OF_PERIOD = "last_session_of_period"
    CALENDAR_DATE_ROLLED_FORWARD = "calendar_date_rolled_forward"


@dataclass(frozen=True)
class CashFlowRuntime:
    """Placeholder base to keep the dataclass ordering readable."""


@dataclass(frozen=True)
class CashFlowRuntime(RuntimeArtifact):  # noqa: F811 - single definition wins
    kind: ClassVar[str] = "flow"
    undefined_without: ClassVar[Sequence[str]] = ("calendar",)
    """A cadence is meaningless without sessions to land on: "monthly" names no
    day, and which day it lands on changes the money-weighted return."""

    interpreted_with: ClassVar[Sequence[str]] = ("account",)
    """An account may cap a contribution or refuse a withdrawal. The flow still
    means what it means without one; it just may not be permitted."""

    affects_causal_isolation: ClassVar[Sequence[str]] = ()
    """Empty deliberately. A differing account changes whether a flow is
    *allowed*, not what it *is*, so "only the schedule differs" remains a true
    statement about causation. Marking it causal would defeat isolation on a
    relation that does not bear on it."""

    name: str
    version: int
    supported_kinds: Sequence[FlowKind] = ()
    day_rule: DayRule = DayRule.FIRST_SESSION_OF_PERIOD
    inflation_adjusted: bool = False
    in_kind_delivery: bool = True
    """Whether share-settled flows are delivered as shares. False would model a
    vest as cash plus a purchase, inventing an execution decision nobody made."""

    title: str = ""

    def declared_form(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "name": self.name, "version": self.version,
            "supported_kinds": sorted(k.value for k in self.supported_kinds),
            "day_rule": self.day_rule.value,
            "inflation_adjusted": self.inflation_adjusted,
            "in_kind_delivery": self.in_kind_delivery,
            "title": self.title,
        }

    def comparable_form(self) -> Dict[str, Any]:
        declared = self.declared_form()
        for prose in ("title", "name", "version"):
            declared.pop(prose, None)
        return declared

    def supports(self, flow_kind: FlowKind) -> bool:
        return flow_kind in self.supported_kinds

    @property
    def delivers_in_kind(self) -> bool:
        return self.in_kind_delivery

    @property
    def assumptions(self) -> Sequence[RuntimeAssumption]:
        out = [RuntimeAssumption(
            name="day-rule",
            statement=(f"Recurring flows land on the "
                       f"{self.day_rule.value.replace('_', ' ')}."),
            realized_by="flows_from",
            risk=("'Every month' names no day, and the day moves the "
                  "money-weighted return even when the strategy is identical."),
        )]
        if self.in_kind_delivery and FlowKind.RSU_VEST in self.supported_kinds:
            out.append(RuntimeAssumption(
                name="in-kind-vesting",
                statement="Vested shares arrive as shares at the vest price. No "
                          "cash is spent and no order is placed.",
                realized_by="grants_for",
            ))
        return tuple(out)

    @property
    def limitations(self) -> Sequence[RuntimeLimitation]:
        out = []
        if not self.inflation_adjusted:
            out.append(RuntimeLimitation(
                name="nominal-amounts",
                statement=("Contribution amounts are nominal. Over a long "
                           "horizon a fixed contribution buys steadily less, "
                           "which this does not show."),
                reason=Exclusion.OUT_OF_SCOPE,
            ))
        unsupported = [k for k in FlowKind if k not in self.supported_kinds]
        if unsupported:
            out.append(RuntimeLimitation(
                name="unsupported-flow-kinds",
                statement=("This runtime does not interpret "
                           + ", ".join(sorted(k.value for k in unsupported))
                           + ". A scenario using one needs a runtime that does."),
                reason=Exclusion.NOT_APPLICABLE,
            ))
        return tuple(out)


SALARY_AND_VESTS = CashFlowRuntime(
    name="salary-and-vests", version=1,
    supported_kinds=(FlowKind.SALARY, FlowKind.BONUS, FlowKind.RSU_VEST),
    title="Recurring salary contributions and share-settled vesting",
)

IMPLEMENTED = ("flows_from", "grants_for")
