"""What a price series says happened, before anything decides what to do.

    prices → SignalGenerator → Signals → FundingPolicy → ContributionEvents

A `Signal` is an observation with a date and a reason: *"SPY closed below its
200-session simple moving average on 2022-01-21."* It is not an instruction. It
does not know what money will be contributed, what will be bought, or whether
anything happens at all — those are decisions, and they belong downstream.

The separation exists so that the first rule this platform supports does not
become its permanent API. RSI, MACD, realised volatility, an earnings date, a
Fed meeting and someone's own research all answer the same question — *did the
thing I am watching for occur, and when* — and all of them can emit this
object. Event-triggered funding subscribes to signals; it never learns what a
moving average is.

**No look-ahead, by construction.** A generator receives the full frame and
must produce each signal from data at or before its own session. The moving
average is backward-looking, so this holds naturally here; the property is
asserted rather than assumed, because the next generator may not be so honest.

**Warm-up is not a signal.** A 200-session average is undefined until 200
sessions exist. Emitting one from a partial window would report a crossing that
the data cannot support, and it would land at the start of every plan — exactly
where a user is least able to judge it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import pandas as pd


class SignalKind(str, Enum):
    """What was observed. The runtime's vocabulary, not prose."""

    CROSSED_BELOW_MOVING_AVERAGE = "CROSSED_BELOW_MOVING_AVERAGE"
    """The subject closed below its moving average having been at or above it
    on the previous session. A transition, not a state."""

    BELOW_MOVING_AVERAGE = "BELOW_MOVING_AVERAGE"
    """The subject closed below its moving average. True on every day of a
    drawdown, not only the first."""


class Estimator(str, Enum):
    SIMPLE = "simple"
    EXPONENTIAL = "exponential"


#: Estimators this build computes. `exponential` is recognised by the compiler
#: and named in the confirmation text, and a plan asking for one must be refused
#: rather than served a simple average under an exponential label — the two
#: cross on different days, which is the whole reason the compiler distinguishes
#: them.
SUPPORTED_ESTIMATORS = frozenset({Estimator.SIMPLE})


@dataclass(frozen=True)
class Signal:
    """One observation, dated, attributed and explainable."""

    kind: SignalKind
    session: pd.Timestamp
    """The session whose data produced it. Not the session anything executes
    on — a signal observed at a close cannot be traded at that same close, and
    keeping the two dates apart is what makes that checkable."""

    subject: str
    reason: str
    """A sentence a person can read, naming the numbers it rests on."""

    detail: Mapping[str, Any] = field(default_factory=dict)
    """Structured, for a machine: the level, the average, the window."""

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "session": str(self.session.date()),
                "subject": self.subject, "reason": self.reason,
                "detail": dict(self.detail)}


#: Given a price frame, the signals it contains. Every generator has this shape,
#: so a new indicator is a new function rather than a new concept downstream.
SignalGenerator = Callable[[pd.DataFrame], Sequence[Signal]]


class UnsupportedSignal(ValueError):
    """Asked for an indicator this build does not compute."""


def moving_average_signals(
    frame: pd.DataFrame, *, subject: str, window: int,
    estimator: Estimator = Estimator.SIMPLE,
    kind: SignalKind = SignalKind.CROSSED_BELOW_MOVING_AVERAGE,
) -> Sequence[Signal]:
    """Sessions where `subject` was below — or crossed below — its average.

    The two kinds are different rules and produce different plans. "Every time
    it crosses below" fires once per drawdown; "every day it is below" fires on
    each of them, and over five years the difference is not marginal. The
    compiler asks the user which they meant; this honours the answer rather
    than picking the more flattering one.
    """
    if estimator not in SUPPORTED_ESTIMATORS:
        raise UnsupportedSignal(
            f"{estimator.value} moving averages are not computed by this "
            f"build. A simple average crosses on different days, so serving "
            f"one under this label would answer a different question.")
    if subject not in frame.columns:
        raise UnsupportedSignal(
            f"no price history for {subject}, so the condition cannot be "
            f"evaluated. This is a data gap, not an absence of crossings.")
    if window < 2:
        raise UnsupportedSignal("a moving average needs at least two sessions")

    closes = frame[subject].astype(float)
    # `min_periods=window` leaves the warm-up as NaN rather than averaging
    # whatever is available. A 40-session average labelled 200 is a different
    # indicator, and it would produce its first crossing in the first weeks of
    # every plan.
    average = closes.rolling(window=window, min_periods=window).mean()

    below = closes < average
    if kind is SignalKind.BELOW_MOVING_AVERAGE:
        firing = below
    else:
        # A crossing needs a defined previous session. `shift(1)` makes the
        # first comparable session the one *after* warm-up ends, which is
        # correct: on the first session with an average there is no previous
        # state to have crossed from.
        was_at_or_above = ~below.shift(1).fillna(False).astype(bool)
        firing = below & was_at_or_above

    # Warm-up sessions are excluded explicitly rather than relying on NaN
    # comparisons being false. `NaN < x` is False, which produces the right
    # answer for the wrong reason and would silently change if the comparison
    # ever moved.
    firing = firing & average.notna()

    signals = []
    for session in frame.index[firing.to_numpy()]:
        level, mean = float(closes.loc[session]), float(average.loc[session])
        signals.append(Signal(
            kind=kind, session=session, subject=subject,
            reason=(f"{subject} closed at {level:,.2f}, "
                    f"{'crossing below' if kind is SignalKind.CROSSED_BELOW_MOVING_AVERAGE else 'below'} "
                    f"its {window}-session {estimator.value} moving average of "
                    f"{mean:,.2f}"),
            detail={"close": level, "average": mean, "window": window,
                    "estimator": estimator.value}))
    return tuple(signals)


def warmup_sessions(window: int) -> int:
    """How many sessions are consumed before any signal can exist.

    Named rather than left implicit at the call site: a plan asked to cover
    five years must evaluate its condition over five years, which means the
    frame has to start earlier than the period being reported. Confusing the
    two silently shortens the evaluated period and drops the earliest
    crossings.
    """
    return window
