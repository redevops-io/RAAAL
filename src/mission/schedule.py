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

from typing import Callable, List, Mapping, Sequence

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

#: Which session within a period the money lands on.
EXECUTABLE_DAY_RULES: Sequence[str] = (
    "first_session_of_period", "last_session_of_period")

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
    dates = (groups.max() if schedule.day_rule == "last_session_of_period"
             else groups.min())
    return [cash_flow(d, schedule.amount, "contribution") for d in dates]
