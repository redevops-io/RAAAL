"""The execution environment a workspace run actually used.

Every run should pin the runtimes it ran under. Until now `account_hash`,
`calendar_hash` and `market_data_hash` were empty on every workspace run, which
meant the comparability engine compared two absences and reported them equal —
a stored verdict claiming those dimensions were *checked and equivalent* when
nothing had been checked at all.

The pins are derived from what the run used, never from configuration read back
later. A default filled in at read time describes the current setup rather than
the historical one, which is precisely the substitution replay exists to
prevent.

A runtime that cannot be pinned is a **declared limitation on the run**, not a
silent blank. An omission with a name can be closed; one without is a fact of
life.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ..calendars import CalendarRegistry
from ..runtime import (
    AccountKind,
    AccountRuntime,
    AdjustmentPolicy,
    MarketDataRuntime,
    PointInTimePolicy,
)

#: The compiler's account vocabulary, mapped onto the account runtime's kinds.
#: Values the runtime cannot represent are left unmapped so the run declares the
#: gap rather than pinning something adjacent.
ACCOUNT_KINDS: Mapping[str, AccountKind] = {
    "TAXABLE": AccountKind.TAXABLE,
    "ROTH": AccountKind.ROTH_IRA,
    "TRADITIONAL_IRA": AccountKind.TRADITIONAL_IRA,
    "TRADITIONAL_401K": AccountKind.TRADITIONAL_401K,
    "ROTH_401K": AccountKind.ROTH_401K,
}


@dataclass(frozen=True)
class EnvironmentPins:
    """What a run pinned, and what it could not."""

    account_hash: str = ""
    calendar_hash: str = ""
    market_data_hash: str = ""
    unpinned: tuple = ()
    """Dimensions with no runtime behind them on this run. Carried into the
    result's modelling scope so a reader sees the gap beside the figure."""

    def as_conditions(self) -> Dict[str, str]:
        return {"account_hash": self.account_hash,
                "calendar_hash": self.calendar_hash,
                "market_data_hash": self.market_data_hash}

    unrealized: tuple = ()
    """Behaviours the pinned runtimes declare and nothing enforces.

    A pinned account makes the dimension comparable. It does not make the
    contribution limit apply. Keeping the two apart is the difference between
    "we compared the account" and "we enforced its rules", and only the first
    is currently true.
    """

    def limitations(self) -> List[Dict[str, str]]:
        out = [{"dimension": name,
                "why": ("no runtime was pinned for this run, so comparisons "
                        "cannot establish whether it matched")}
               for name in self.unpinned]
        out += [{"dimension": f"account:{name}",
                 "why": ("the account runtime declares this and nothing "
                         "enforces it yet, so it is recorded and not applied")}
                for name in self.unrealized]
        return out


def pins_for(scenario, *, calendar_ref: str = "nyse@1",
             snapshot: str = "") -> EnvironmentPins:
    """Pin the runtimes this run used.

    An account kind outside the runtime's vocabulary is deliberately left
    unpinned: pinning the nearest available kind would record a tax treatment
    the user did not describe.

    Pinning an account is not the same as simulating one. The account runtime
    declares contribution limits, employer matching, the shared employee
    deferral limit and the tax treatment of employer money — and
    `ACCOUNT_IMPLEMENTED` is empty, so none of it is enforced yet. The pin makes
    comparisons able to *evaluate* the dimension; `unrealized_declarations()`
    below is what stops that being mistaken for enforcement.
    """
    unpinned: List[str] = []

    account_hash, unrealized = "", ()
    kind = ACCOUNT_KINDS.get(getattr(scenario, "tax_treatment", ""))
    if kind is not None:
        from ..runtime import ACCOUNT_IMPLEMENTED

        # Built with the governing limit, exactly as `account_support` builds
        # it. Constructed without one, a run would report no account rules while
        # the confirmation card for the same plan reported a contribution limit
        # — two answers to one question, from two constructions of one runtime.
        import datetime as _dt

        from ..runtime.account_limits import load as _load_limits

        _limit = _load_limits().limit_for(kind.value, _dt.date.today().year)
        runtime = AccountRuntime(name=f"account/{kind.value.lower()}", version=1,
                                 account_kind=kind,
                                 annual_contribution_limit=_limit.amount)
        account_hash = runtime.compatibility_hash
        unrealized = tuple(runtime.unrealized(ACCOUNT_IMPLEMENTED))
    else:
        unpinned.append("account")

    calendar_hash = ""
    try:
        calendar_hash = CalendarRegistry().resolve(calendar_ref).compatibility_hash
    except Exception:                                           # noqa: BLE001
        unpinned.append("calendar")

    market_data_hash = ""
    if snapshot:
        market_data_hash = MarketDataRuntime(
            name="market-data/workspace", version=1,
            provider="synthetic-or-pinned-snapshot", dataset=snapshot,
            adjustment_policy=AdjustmentPolicy.ADJUSTED_ONLY,
            point_in_time_policy=PointInTimePolicy.LATEST_RESTATED,
        ).compatibility_hash
    else:
        unpinned.append("market_data")

    return EnvironmentPins(account_hash=account_hash, calendar_hash=calendar_hash,
                           market_data_hash=market_data_hash,
                           unpinned=tuple(unpinned), unrealized=unrealized)
