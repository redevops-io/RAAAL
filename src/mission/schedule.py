"""Turning a declared cadence into dated money.

Lifted out of `workspace.routes` for one reason: **the capability manifest has
to be derived from the executor, and it cannot import a web module.**

The tables below are the single source of truth for which cadences this build
executes. `capability.py` reads `EXECUTABLE_CADENCES` from here rather than
restating it, so a cadence added to the engine and not to the manifest — or
offered in a menu and never executed — is not expressible.

That is not hypothetical. `quarterly`, `annual` and `daily` were each offered
in the product's own confirmation menu, rendered back as "every quarter" and
"every year", and executed as a single one-off contribution, because the
expansion matched three cadences and let everything else fall through a default
branch. "$1,000 every year for five years" reported $1,000 contributed, with no
refusal and no caveat.

The day rule is applied here rather than assumed, because "monthly" names no
day and the day moves the money-weighted return.
"""
from __future__ import annotations

from typing import Callable, List, Mapping, Optional, Sequence

import pandas as pd


class UnsupportedCadence(Exception):
    """A declared cadence this engine cannot turn into dated money.

    Raised, not defaulted. The default it replaces produced a wrong number for
    a right answer, silently.
    """


#: Cadences that group sessions into calendar periods. The value is the
#: grouping key; the day rule then picks a session within each period.
#:
#: Keyed by the same strings `compiler._CADENCE` produces and `render`
#: verbalises, so the three vocabularies are comparable by name.
PERIOD_KEYS: Mapping[str, Callable[[pd.DatetimeIndex], list]] = {
    "annual": lambda s: [s.year],
    "quarterly": lambda s: [s.year, s.quarter],
    "monthly": lambda s: [s.year, s.month],
    "weekly": lambda s: [s.isocalendar().year.values,
                         s.isocalendar().week.values],
    # `isoweek // 2` restarts the pairing at each year boundary, so a year
    # whose last ISO week is odd contributes a singleton group and one extra
    # purchase — about 136 contributions over five years rather than 130. A
    # real imprecision, recorded rather than blessed, and separate from the
    # defect this module was written for.
    "biweekly": lambda s: [s.isocalendar().year.values,
                           s.isocalendar().week.values // 2],
}

#: Every session is its own period, so the day rule has nothing to choose.
EVERY_SESSION = "daily"

#: One contribution, on the first session. Genuinely a lump sum — not the
#: fallback that used to absorb everything unrecognised.
SINGLE = "once"

#: A contribution on a named day of the month, written `calendar_day:15`.
#:
#: Parameterised because the day is the whole point: "the 15th" and "the 1st"
#: are different plans, and a vocabulary that can only say first-or-last has no
#: word for either. Somebody who wrote "on the same day each month — the 15th"
#: was read as `calendar_first_rolled_forward` and then refused for asking for
#: the first of the month, which is not what they said and not what the record
#: should show.
CALENDAR_DAY = "calendar_day"

#: Which session within a period the money lands on.
EXECUTABLE_DAY_RULES: Sequence[str] = (
    "first_session_of_period", "last_session_of_period", CALENDAR_DAY)

#: **The manifest's source for `cadence`.** Derived, not restated.
EXECUTABLE_CADENCES: Sequence[str] = tuple(
    sorted(set(PERIOD_KEYS) | {EVERY_SESSION, SINGLE}))

#: Cadences the vocabulary offers that this engine will not execute, and why.
#: A pay cycle is not a calendar period: it may be weekly, biweekly,
#: semi-monthly or monthly, and picking one invents the user's employer. The
#: lump sum it used to become was further from what they said than any of the
#: four.
REFUSED_CADENCES: Mapping[str, str] = {
    "payroll": ("a pay cycle is not a calendar period — it may be weekly, "
                "biweekly, semi-monthly or monthly, and choosing one would "
                "invent your pay schedule"),
}


def expand(schedule, sessions: pd.DatetimeIndex, *, cash_flow) -> List:
    """The dated contributions a declared schedule produces.

    `cash_flow` is injected so this module stays free of the simulation
    package's import graph; callers pass `mission.CashFlow`.
    """
    if schedule.amount <= 0:
        return ([cash_flow(sessions[0], schedule.starting_capital,
                           "starting capital")]
                if schedule.starting_capital > 0 else [])

    cadence = schedule.cadence

    if cadence == SINGLE:
        return [cash_flow(sessions[0], schedule.amount, "one-off")]

    if cadence == EVERY_SESSION:
        return [cash_flow(d, schedule.amount, "contribution") for d in sessions]

    if cadence not in PERIOD_KEYS:
        raise UnsupportedCadence(
            f"This build does not execute a {cadence!r} contribution cadence, "
            "so no figure can be produced for it. Naming a calendar cadence — "
            "weekly, biweekly, monthly, quarterly or yearly — would let this "
            "plan run.")

    groups = sessions.to_series().groupby(PERIOD_KEYS[cadence](sessions))
    nominated = day_of_month(schedule.day_rule)
    if nominated is not None:
        dates = groups.apply(lambda period: _on_or_after(period, nominated))
    elif schedule.day_rule == "last_session_of_period":
        dates = groups.max()
    else:
        dates = groups.min()
    return [cash_flow(d, schedule.amount, "contribution") for d in dates]


def day_of_month(day_rule: str) -> Optional[int]:
    """The day a `calendar_day:15` rule names, or None for the session rules.

    Refuses a day outside 1–31 by returning None rather than guessing, so an
    unreadable rule falls through to the first session and is caught by the
    manifest rather than silently landing money on a date nobody named.
    """
    if not isinstance(day_rule, str) or not day_rule.startswith(CALENDAR_DAY + ":"):
        return None
    _, _, stated = day_rule.partition(":")
    try:
        day = int(stated)
    except ValueError:
        return None
    return day if 1 <= day <= 31 else None


def _on_or_after(period: pd.Series, day: int):
    """The first session in this period on or after the nominated day.

    This is `ModifiedFollowing` — roll forward, unless that leaves the month,
    in which case take the last session of the month. It was reasoned out here
    before anybody checked, and it matches
    `ql.UnitedStates(ql.UnitedStates.NYSE).adjust(date, ql.ModifiedFollowing)`
    for every month of 2024 on both the 15th and the 31st. See
    `mission.conventions`, which now names it.

    Computed from the sessions rather than from the calendar, deliberately.
    QuantLib knows which days the exchange was open; only the snapshot knows
    which days it has a price for, and a purchase on a date with no bar is a
    worse error than a contribution landing a day late.
    """
    on_or_after = period[period.dt.day >= day]
    return on_or_after.min() if len(on_or_after) else period.max()
