"""How money arrives. One concept, two policies, nowhere to state both.

    FundingPolicy → ContributionEvents → execution → portfolio

`cadence` and a conditional purchase rule were both answers to *when does money
appear*, held in different places, and the engine consumed only one of them. A
plan reading "buy $1,000 every time SPY crosses below its 200-day average"
compiled to `cadence=once` with the rule recorded beside it — one contribution
for a five-year period, and a figure that was really buy-and-hold.

Making them a sum type is what removes the contradiction. There is no
`FundingPolicy` that carries a cadence *and* a trigger, so the compiler cannot
build one, the builder never asks a question whose answer it would discard, and
the engine has one thing to consume.

**Contributions, not trades.** An `EventTriggered` policy emits dated cash, and
the allocation rule decides what that cash buys. This is not a simplification —
it is the correct decomposition: "invest $1,000 on each crossing" and "invest
$1,000 monthly" differ in *when the money arrives* and in nothing else, and a
benchmark is honest precisely because it receives the same events.

It also means every future funding source — RSU vesting, salary deduction, a
bonus, an inheritance — lands here as a policy rather than as a special case in
the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd

from .signals import (
    Estimator,
    Signal,
    SignalKind,
    UnsupportedSignal,
    moving_average_signals,
)


class FundingKind(str, Enum):
    SCHEDULED = "SCHEDULED"
    EVENT_TRIGGERED = "EVENT_TRIGGERED"


class ExecutionTiming(str, Enum):
    NEXT_SESSION_OPEN = "next_session_open"
    """The signal is observed at a close and acted on at the next session. A
    plan that traded the close which produced its own signal would be reading
    the future by one bar — the most common way a backtest flatters itself."""

    SAME_SESSION_CLOSE = "same_session_close"


#: What this build executes. `same_session_close` is refused rather than
#: silently treated as next-session: it is a materially different and
#: optimistic assumption, and a user who asked for it must be told no.
SUPPORTED_TIMING = frozenset({ExecutionTiming.NEXT_SESSION_OPEN})


class UnsupportedFunding(ValueError):
    """A funding policy this build cannot execute."""


@dataclass(frozen=True)
class Trigger:
    """The condition an event-triggered policy watches for."""

    subject: str
    window: int
    estimator: Estimator = Estimator.SIMPLE
    kind: SignalKind = SignalKind.CROSSED_BELOW_MOVING_AVERAGE

    def to_json(self) -> Dict[str, Any]:
        return {"subject": self.subject, "window": self.window,
                "estimator": self.estimator.value, "kind": self.kind.value}

    def signals(self, frame: pd.DataFrame) -> Sequence[Signal]:
        return moving_average_signals(
            frame, subject=self.subject, window=self.window,
            estimator=self.estimator, kind=self.kind)


@dataclass(frozen=True)
class Scheduled:
    """Money arrives on a calendar."""

    cadence: str
    amount: Decimal
    day_rule: str = "first_session_of_period"
    starting_capital: Decimal = Decimal("0")
    kind: FundingKind = field(default=FundingKind.SCHEDULED, init=False)

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "cadence": self.cadence,
                "amount": str(self.amount), "day_rule": self.day_rule,
                "starting_capital": str(self.starting_capital)}


@dataclass(frozen=True)
class EventTriggered:
    """Money arrives when something happens."""

    trigger: Trigger
    amount: Decimal
    execution_timing: ExecutionTiming = ExecutionTiming.NEXT_SESSION_OPEN
    starting_capital: Decimal = Decimal("0")
    kind: FundingKind = field(default=FundingKind.EVENT_TRIGGERED, init=False)

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "trigger": self.trigger.to_json(),
                "amount": str(self.amount),
                "execution_timing": self.execution_timing.value,
                "starting_capital": str(self.starting_capital)}


FundingPolicy = Union[Scheduled, EventTriggered]


@dataclass(frozen=True)
class ContributionEvent:
    """One dated arrival of money, and why it arrived.

    The common currency. Whatever produced it — a monthly cadence, a crossing,
    a vest — the engine downstream sees the same object, and a benchmark
    receives exactly these events with a different allocation rule. That is
    what makes the strategy-effect claim true rather than aspirational.
    """

    session: pd.Timestamp
    """When the money is available to invest."""

    amount: Decimal
    reason: str
    signal: Optional[Signal] = None
    """The observation that caused it, for an event-triggered policy. Carried
    so the ledger can show signal date beside execution date; a row that could
    not name its cause would be a purchase nobody can check."""

    def to_json(self) -> Dict[str, Any]:
        return {"session": str(self.session.date()), "amount": str(self.amount),
                "reason": self.reason,
                "signal": self.signal.to_json() if self.signal else None}


def _next_session(sessions: pd.DatetimeIndex, after: pd.Timestamp
                  ) -> Optional[pd.Timestamp]:
    """The first session strictly after `after`, or None past the end.

    None matters. A crossing on the final session of the period has no next
    session to execute on, and inventing one — or silently executing at the
    same close — would manufacture a purchase the data cannot support.
    """
    later = sessions[sessions > after]
    return later[0] if len(later) else None


def contribution_events(policy: FundingPolicy, *, frame: pd.DataFrame,
                        sessions: Optional[pd.DatetimeIndex] = None,
                        scheduled_flows: Sequence[Any] = ()
                        ) -> Sequence[ContributionEvent]:
    """The dated money a policy produces over a price frame.

    `scheduled_flows` is how a `Scheduled` policy stays exactly what it was:
    the existing schedule expansion is authoritative and is adapted here rather
    than reimplemented. Two implementations of a cadence would be two answers
    to when money arrived, which is the defect this module exists to remove.
    """
    sessions = frame.index if sessions is None else sessions

    if isinstance(policy, Scheduled):
        return tuple(
            ContributionEvent(session=pd.Timestamp(flow.date),
                              amount=Decimal(str(flow.amount)),
                              reason=f"{policy.cadence} contribution")
            for flow in scheduled_flows)

    if not isinstance(policy, EventTriggered):
        raise UnsupportedFunding(f"unknown funding policy {type(policy).__name__}")

    if policy.execution_timing not in SUPPORTED_TIMING:
        raise UnsupportedFunding(
            f"{policy.execution_timing.value} execution is not supported. "
            f"Acting on the same close that produced the signal reads one bar "
            f"into the future.")

    events = []
    for signal in policy.trigger.signals(frame):
        session = _next_session(sessions, signal.session)
        if session is None:
            # Observed and unexecutable. Not dropped silently: the ledger
            # reports it, because "the condition occurred and we could not act
            # on it" is a different fact from "it did not occur".
            continue
        events.append(ContributionEvent(
            session=session, amount=policy.amount,
            reason=f"triggered: {signal.reason}", signal=signal))
    return tuple(events)


def unexecutable_signals(policy: FundingPolicy, *, frame: pd.DataFrame,
                         sessions: Optional[pd.DatetimeIndex] = None
                         ) -> Sequence[Signal]:
    """Signals that fired with no later session to execute on.

    Separated from the events rather than folded into them. A crossing on the
    last day of the period is a real observation and a real reason the totals
    are what they are; counting it as a purchase would overstate the plan, and
    dropping it without a word would make `count(signals)` and
    `count(purchases)` disagree with no explanation.
    """
    if not isinstance(policy, EventTriggered):
        return ()
    sessions = frame.index if sessions is None else sessions
    return tuple(signal for signal in policy.trigger.signals(frame)
                 if _next_session(sessions, signal.session) is None)
