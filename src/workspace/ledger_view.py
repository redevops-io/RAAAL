"""The ledger a run produced, in one shape, for every kind of plan.

Everything the page shows is derived from what the engine actually did:
positions, valuation, the time series, the return, the chart. Until now none
of it was shown. A person read a figure whose provenance stopped at "the
engine says so", while the lines behind it existed, were reconciled against
that figure, and were rendered nowhere.

**Two sources, because the engine has two.** `mission.ledger` joins
contributions to the fills they caused and is built only when a plan has
*contribution events* — a rule that fires on an observation. A plain monthly
schedule has no events, so it had no ledger at all, even though the portfolio
path recorded every purchase it made. The fills are the more primitive fact
and are always there; the event ledger adds which observation caused each one.

So: the event ledger where it exists, the fills otherwise, and one shape either
way. A page that showed a ledger for triggered strategies and nothing for
scheduled ones would be teaching people that scheduled plans are less
accountable, which is the opposite of true.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Line:
    """One thing that happened, at the price that was available."""

    #: The observation that caused it, where a rule caused it at all. Absent
    #: for a scheduled contribution, and the absence is meaningful: nothing was
    #: watched, the date came round.
    signal: str = ""
    #: When the money arrived, and when the purchase executed. Distinct on
    #: purpose — with only one date, a policy that acted on the very session
    #: that produced its signal is indistinguishable from one that waited, and
    #: the look-ahead check passes either way.
    contributed: str = ""
    executed: str = ""
    subject: str = ""
    amount: str = ""
    shares: str = ""
    price: str = ""
    reason: str = ""

    def to_json(self) -> Mapping[str, Any]:
        return {"signal": self.signal, "contributed": self.contributed,
                "executed": self.executed, "subject": self.subject,
                "amount": self.amount, "shares": self.shares,
                "price": self.price, "reason": self.reason}


def _date(value) -> str:
    if value is None:
        return ""
    date = getattr(value, "date", None)
    return str(date() if callable(date) else value)


def _money(value) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value or "")


def _shares(value) -> str:
    try:
        return f"{float(value):,.6f}"
    except (TypeError, ValueError):
        return str(value or "")


def lines(run: Optional[Mapping[str, Any]]) -> Sequence[Line]:
    """Every purchase this run made, whichever way the plan caused it."""
    if not run:
        return ()

    ledger = run.get("ledger")
    if ledger is not None and getattr(ledger, "rows", ()):
        return tuple(
            Line(signal=_date(row.signal_session),
                 contributed=_date(row.contribution_session),
                 executed=_date(row.execution_session),
                 subject=row.subject, amount=_money(row.contribution),
                 shares=_shares(row.shares), price=_money(row.price),
                 reason=row.reason)
            for row in ledger.rows)

    result = run.get("result")
    path = getattr(result, "path", None)
    fills = getattr(path, "fills", ()) if path is not None else ()
    return tuple(
        Line(contributed=_date(fill.date), executed=_date(fill.date),
             subject=fill.ticker, amount=_money(fill.notional),
             shares=_shares(fill.shares), price=_money(fill.price),
             reason=fill.reason or "purchase")
        for fill in fills)


def unfilled(run: Optional[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
    """Orders that could not execute, kept rather than dropped.

    An order that silently vanished is the difference between what the plan
    declared and what it did, and it is invisible in a total.
    """
    if not run:
        return ()
    result = run.get("result")
    path = getattr(result, "path", None)
    orders = getattr(path, "unfilled", ()) if path is not None else ()
    return tuple({"date": _date(getattr(order, "date", None)),
                  "subject": getattr(order, "ticker", ""),
                  "amount": _money(getattr(order, "notional", 0)),
                  "reason": getattr(order, "reason", "") or "did not execute"}
                 for order in orders)
