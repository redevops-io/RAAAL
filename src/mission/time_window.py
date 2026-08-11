"""What period did the user mean, and which sessions is that?

"for the past 5 years" is an instruction, not a date range. A trailing
five-year window and an explicit 2021-08-05 to 2026-08-05 range resolve to the
same sessions today and are not the same thing: re-run next month, one moves
and one does not. So the instruction is stored and the dates are derived.

Four decisions that are easy to make by accident and hard to notice
afterwards:

**The anchor is the snapshot, not the clock.** Resolving "now" to wall-clock
time makes the same plan produce different figures on different days, from the
same data. It resolves to the latest session in the pinned price snapshot.

**Five years is calendar, not 5 x 252.** A user saying "five years" means five
years. Trading-day arithmetic drifts against the calendar by roughly a fortnight
per year, and nobody asked for that.

**Boundaries are inclusive**, both ends, and said out loud — a moving-average
trigger firing on the first or last session of the window is exactly the kind
of edge two implementations silently disagree about.

**Warm-up is not analysis.** A 200-day moving average needs 200 sessions
before the first session it can judge. Shortening the analysis window to make
room answers a different question than the user asked; including the warm-up
in the reported totals reports returns from a period they did not ask about.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple


class WindowKind(str, Enum):
    TRAILING = "trailing"
    """"the past five years" — a duration back from the anchor."""

    EXPLICIT_RANGE = "explicit_range"
    """"from January 2020 to December 2024"."""

    SINCE = "since"
    """"since 2021"."""

    UNTIL = "until"
    """"through 2024"."""

    EVENT_RELATIVE = "event_relative"
    """"since the 2022 drawdown" — needs an event the engine can date."""

    ROLLING = "rolling"
    """"every month over the past five years" — many windows, not one."""


#: What this build resolves. Everything else is recognised, typed and refused
#: rather than coerced: reading "since 2021" as a trailing window would answer
#: a different question with a plausible number.
SUPPORTED = frozenset({WindowKind.TRAILING})

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
}

_TRAILING = re.compile(
    r"\b(?:for\s+|over\s+)?(?:the\s+)?(?:past|last|previous|trailing)\s+"
    r"(\d{1,2}|" + "|".join(_WORD_NUMBERS) + r")[\s-]*(year|yr|month|mo)s?\b",
    re.IGNORECASE)
_LOOKBACK = re.compile(
    r"\b(\d{1,2})[\s-]*(year|yr|month|mo)s?[\s-]*(?:lookback|look-back|history|window)\b",
    re.IGNORECASE)

_SINCE = re.compile(r"\bsince\s+(?:the\s+)?(\d{4}|\w+\s+\d{4})\b", re.IGNORECASE)
_UNTIL = re.compile(r"\b(?:through|until|up\s+to)\s+(\d{4})\b", re.IGNORECASE)
_RANGE = re.compile(r"\bfrom\s+.{3,20}\s+(?:to|through|until)\s+.{3,20}", re.IGNORECASE)
_ROLLING = re.compile(r"\b(?:each|every)\s+(?:month|quarter|year)\b.*\b(?:past|last|over)\b",
                      re.IGNORECASE)
_EVENT = re.compile(r"\bsince\s+the\s+\w+\s+(?:crash|drawdown|selloff|peak|bottom)\b",
                    re.IGNORECASE)


@dataclass(frozen=True)
class TimeWindow:
    """The instruction, preserved. Dates are derived from it, never instead."""

    kind: WindowKind
    observed: str
    years: Optional[int] = None
    months: Optional[int] = None

    @property
    def supported(self) -> bool:
        return self.kind in SUPPORTED

    @property
    def label(self) -> str:
        if self.years:
            return f"the past {self.years} year{'' if self.years == 1 else 's'}"
        if self.months:
            return f"the past {self.months} month{'' if self.months == 1 else 's'}"
        return self.observed

    def to_json(self) -> dict:
        return {"kind": self.kind.value, "observed": self.observed,
                "years": self.years, "months": self.months,
                "supported": self.supported}


@dataclass(frozen=True)
class ResolvedWindow:
    """Sessions, and where they came from."""

    window: TimeWindow
    start: dt.date
    end: dt.date
    warmup_start: Optional[dt.date] = None
    warmup_sessions: int = 0
    anchor_source: str = "snapshot"
    #: Inclusive at both ends. Stated because a trigger on the first or last
    #: session is exactly where two implementations quietly disagree.
    inclusive: bool = True
    short: bool = False
    """The snapshot does not reach back far enough for the window asked for."""

    def to_json(self) -> dict:
        return {"window": self.window.to_json(),
                "start": self.start.isoformat(), "end": self.end.isoformat(),
                "warmup_start": self.warmup_start.isoformat()
                if self.warmup_start else None,
                "warmup_sessions": self.warmup_sessions,
                "anchor_source": self.anchor_source,
                "inclusive": self.inclusive, "short": self.short}


def detect(text: str) -> Optional[TimeWindow]:
    """The temporal instruction in a description, typed.

    Order matters: the more specific forms are tested first, because "every
    month over the past five years" contains "the past five years" and is not
    one window.
    """
    if not text:
        return None

    found = _ROLLING.search(text)
    if found:
        return TimeWindow(WindowKind.ROLLING, found.group(0).strip())
    found = _EVENT.search(text)
    if found:
        return TimeWindow(WindowKind.EVENT_RELATIVE, found.group(0).strip())
    found = _RANGE.search(text)
    if found:
        return TimeWindow(WindowKind.EXPLICIT_RANGE, found.group(0).strip())

    for pattern in (_TRAILING, _LOOKBACK):
        found = pattern.search(text)
        if found:
            raw, unit = found.group(1), found.group(2).lower()
            count = _WORD_NUMBERS.get(raw.lower(), None)
            if count is None:
                count = int(raw)
            if unit.startswith("y"):
                return TimeWindow(WindowKind.TRAILING, found.group(0).strip(),
                                  years=count)
            return TimeWindow(WindowKind.TRAILING, found.group(0).strip(),
                              months=count)

    found = _SINCE.search(text)
    if found:
        return TimeWindow(WindowKind.SINCE, found.group(0).strip())
    found = _UNTIL.search(text)
    if found:
        return TimeWindow(WindowKind.UNTIL, found.group(0).strip())
    return None


def _back(anchor: dt.date, *, years: int = 0, months: int = 0) -> dt.date:
    """Calendar arithmetic, not trading-day arithmetic.

    29 February has no counterpart in a non-leap year; clamping to the 28th is
    the ordinary reading of "five years before".
    """
    month_index = anchor.month - 1 - months
    year = anchor.year - years + month_index // 12
    month = month_index % 12 + 1
    day = anchor.day
    while day > 28:
        try:
            return dt.date(year, month, day)
        except ValueError:
            day -= 1
    return dt.date(year, month, day)


def resolve(window: TimeWindow, sessions: Sequence[dt.date], *,
            warmup_sessions: int = 0) -> Optional[ResolvedWindow]:
    """Turn the instruction into sessions, against a pinned snapshot.

    `sessions` is the snapshot's own trading calendar, ascending. The last one
    is the anchor — "now" for a historical replay is the latest session the
    data actually has, not the moment the request arrived.
    """
    if not window.supported or not sessions:
        return None

    ordered = sorted(sessions)
    end = ordered[-1]
    boundary = _back(end, years=window.years or 0, months=window.months or 0)

    # The first session on or after the calendar boundary. Aligning to
    # sessions after computing the calendar date keeps "five years" five
    # years, rather than five years of trading days.
    later = [one for one in ordered if one >= boundary]
    if not later:
        return None
    start = later[0]
    short = boundary < ordered[0]

    warmup_start = None
    if warmup_sessions:
        index = ordered.index(start)
        # Warm-up extends *before* the window rather than eating into it.
        # Taking it out of the analysis period would silently answer a
        # question about four years and three months.
        warmup_start = ordered[max(0, index - warmup_sessions)]

    return ResolvedWindow(
        window=window, start=start, end=end, warmup_start=warmup_start,
        warmup_sessions=warmup_sessions, anchor_source="snapshot latest session",
        short=short)
