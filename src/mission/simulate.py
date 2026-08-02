"""Replay a cash-flow-and-event program against history.

A methodology answers *"given capital, what weights?"*. A Mission answers *"given
a life, what transactions?"* — and "never sell", "buy $2,000 worth" and "whenever
SPY drops below its 200DMA" are all statements about transactions, none of which
survives translation into a weight vector.

The engine keeps the disciplines the weight-based engine established, because
they were established by finding real defects:

* **Causality.** An order formed from data through session *d* executes at *d+1*.
  Filling on *d* credits a price that was not available when the decision was made.
* **Costs.** Every fill is charged. A gross-only replay is not a number anyone
  should act on.
* **Nothing vanishes silently.** An order that could not fill is retained and
  reported, because an order that quietly disappeared is the gap between what the
  Mission declared and what it did.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .accounting import (
    CashFlow,
    InKindFlow,
    CashPolicy,
    CashPolicyError,
    Fill,
    Grant,
    Order,
    PortfolioPath,
    money_weighted_return,
    time_weighted_returns,
)

#: Signature of an event program: given the session, the prices seen *so far*,
#: current holdings and cash, return the orders to submit. Prices are truncated
#: at the current session by the caller, so a program physically cannot read
#: forward — the leak is prevented by construction rather than by review.
EventProgram = Callable[
    [pd.Timestamp, pd.DataFrame, Dict[str, float], float], Sequence[Order]
]


@dataclass
class MissionResult:
    """Both return bases, always, plus the path that produced them."""

    path: PortfolioPath
    time_weighted: pd.Series
    money_weighted: Optional[float]
    periods_per_year: int
    rsu_context: Optional[Any] = None
    """The structured RSU result context, when this run used RSU mechanics.

    An explicit field rather than a key inside `modelling_scope`, because it is
    required for some runs and meaningless for others, and a generic bag cannot
    express that difference."""

    requires_rsu_context: bool = False
    """Declared by the run, never inferred from the presence of diagnostics.

    Inferring it would mean a clean RSU run — no unpriced arrivals, no failures
    — looked exactly like a run that never touched RSU mechanics, so the one
    case where the context is most reassuring is the one where its absence
    would go unnoticed."""

    result_schema_version: int = 2
    """1 predates structured result contexts. Absence in a version-1 record is
    not evidence that no diagnostic existed; it is evidence that none was
    recorded, which reads as NOT_DECLARED rather than as clean."""

    modelling_scope: Optional[Dict[str, Any]] = None
    """What was and was not modelled, carried by the result itself.

    A scope shown only on the configuration screen is absent from the number
    someone quotes, and the number is what travels. Attaching it here means an
    export or a saved plan cannot present a figure while leaving behind the
    statement of what it excludes.
    """

    @property
    def time_weighted_annualized(self) -> Optional[float]:
        r = self.time_weighted.dropna()
        if r.empty:
            return None
        total = float((1.0 + r).prod())
        if total <= 0:
            return None
        return float(total ** (self.periods_per_year / len(r)) - 1.0)

    @property
    def final_value(self) -> float:
        return self.path.terminal_value

    @property
    def gain(self) -> float:
        """Terminal value less net money put in. The number a user recognises."""
        return self.final_value - (self.path.contributed - self.path.withdrawn)

    def to_json(self) -> Dict[str, Any]:
        return {
            "final_value": self.final_value,
            "contributed": self.path.contributed,
            "withdrawn": self.path.withdrawn,
            "gain": self.gain,
            "time_weighted_annualized": self.time_weighted_annualized,
            "money_weighted_annualized": self.money_weighted,
            "return_basis_note": (
                "Time-weighted return answers 'is this a good strategy?' by "
                "removing the effect of when money arrived. Money-weighted return "
                "answers 'how did I do?' by keeping it. Neither substitutes for "
                "the other, and a figure quoted without saying which it is cannot "
                "be interpreted."
            ),
            "unfilled_orders": [o.to_json() for o in self.path.unfilled],
            "cash_policy": (self.path.cash_policy.to_json()
                            if self.path.cash_policy else None),
            "result_schema_version": self.result_schema_version,
            "requires_rsu_context": self.requires_rsu_context,
            "rsu_context": (self.rsu_context.to_json()
                            if self.rsu_context is not None else None),
            "modelling_scope": self.modelling_scope,
            "scope_note": (
                "This figure excludes everything listed under not_modelled. "
                "Quoting it without them overstates what it accounts for."
                if self.modelling_scope else
                "No modelling scope was attached to this result."
            ),
        }


def simulate(
    prices: pd.DataFrame,
    *,
    flows: Sequence[CashFlow],
    program: EventProgram,
    grants: Sequence[Grant] = (),
    in_kind: Sequence[InKindFlow] = (),
    cash_policy: Optional[CashPolicy] = None,
    execution_lag: int = 1,
    cost_bps: float = 10.0,
    periods_per_year: int = 252,
    allow_fractional: bool = True,
    modelling_scope: Optional[Dict[str, Any]] = None,
) -> MissionResult:
    """Replay a flow schedule and an event program over `prices`.

    `cash_policy` is required rather than defaulted. Uninvested cash earning
    nothing is a legitimate answer and a different one from not having decided,
    and every erratum this platform has published came from a choice that was
    made by omission.
    """
    if cash_policy is None:
        raise CashPolicyError(
            "cash_policy is required. Money that has arrived but not yet been "
            "invested must earn something or nothing by declaration — "
            "CashPolicy.idle() states that it earns nothing."
        )
    if prices.empty:
        raise ValueError("no price history to replay against")

    prices = prices.sort_index()
    sessions = prices.index
    daily_cash_rate = cash_policy.daily_rate(periods_per_year)

    # Two series, deliberately. `cash_series` is money that becomes spendable
    # cash; `flow_series` is every dated external contribution, cash or in kind,
    # and is what the money-weighted return sees.
    #
    # Held as one series this worked only because the in-kind addition happened
    # to sit after the line that read cash for the same session. Reordering two
    # steps in the loop would have turned every vest into cash and funded a
    # purchase with it — the exact defect the in-kind model exists to prevent.
    cash_series = pd.Series(0.0, index=sessions)
    flow_series = pd.Series(0.0, index=sessions)
    for flow in flows:
        session = _next_session(sessions, flow.date)
        if session is not None:
            cash_series.loc[session] += flow.amount
            flow_series.loc[session] += flow.amount

    # A `Grant` is an in-kind flow whose value the engine still resolves at the
    # landing session. Converted here so the engine has one primitive; callers
    # that pin the valuation supply `InKindFlow` directly and keep their value.
    arrivals: Dict[pd.Timestamp, List[InKindFlow]] = {}
    deferred: List[InKindFlow] = list(in_kind)
    for grant in grants:
        deferred.append(InKindFlow(
            date=grant.date, asset=grant.ticker, quantity=grant.shares,
            valuation_price=float("nan"), external_value=float("nan"),
            source_ref=grant.reason))

    unpriced: List[Dict[str, Any]] = []
    for arriving in deferred:
        session = _next_session(sessions, arriving.date)
        if session is None:
            unpriced.append({"asset": arriving.asset,
                             "quantity": arriving.quantity,
                             "why": "no trading session on or after this date"})
            continue
        arrivals.setdefault(session, []).append(arriving)

    holdings: Dict[str, float] = {}
    cash = 0.0
    pending: List[Order] = []
    fills: List[Fill] = []
    unfilled: List[Order] = []

    value_rows: List[float] = []
    cash_rows: List[float] = []
    holding_rows: List[Dict[str, float]] = []

    for position, session in enumerate(sessions):
        # 1. Cash earns its declared rate on the balance carried in.
        cash *= 1.0 + daily_cash_rate

        # 2. External money lands, uninvested for this session. Only cash
        #    flows — in-kind arrivals are not cash and never fund a purchase.
        cash += float(cash_series.iloc[position])

        # 3. Assets arrive in kind. No order is placed and no cash is spent:
        #    the delivery is not a trade, so it carries no fill, no cost and no
        #    execution lag. The shares are owned on arrival.
        for arriving in arrivals.get(session, ()):
            value = arriving.external_value
            if not np.isfinite(value):
                # Unpinned: resolve at the landing session, as before.
                price = float(prices.at[session, arriving.asset]) \
                    if arriving.asset in prices.columns else float("nan")
                value = arriving.quantity * price if np.isfinite(price) else \
                    float("nan")
            if not np.isfinite(value) or arriving.quantity == 0:
                # A named gap, not a silent skip. An arrival that vanishes
                # quietly leaves a portfolio missing shares the user believes
                # it holds, and nothing on the result says why.
                unpriced.append({"asset": arriving.asset,
                                 "quantity": arriving.quantity,
                                 "source_ref": arriving.source_ref,
                                 "why": "no usable price at the arrival session"})
                continue
            holdings[arriving.asset] = holdings.get(arriving.asset, 0.0) \
                + arriving.quantity
            flow_series.iloc[position] += value

        # 4. Orders submitted `execution_lag` sessions ago fill now, at today's
        #    price — the first price available after the decision was made.
        due = [o for o in pending
               if _sessions_between(sessions, o.date, session) >= execution_lag]
        pending = [o for o in pending if o not in due]
        for order in due:
            fill = _execute(order, session, prices, holdings, cash,
                            cost_bps=cost_bps, allow_fractional=allow_fractional)
            if fill is None:
                unfilled.append(order)
                continue
            holdings[order.ticker] = holdings.get(order.ticker, 0.0) + fill.shares
            cash -= fill.notional + fill.cost
            fills.append(fill)

        # 5. The program sees history through today and nothing after it.
        visible = prices.iloc[: position + 1]
        for order in program(session, visible, dict(holdings), cash) or ():
            pending.append(order)

        row = prices.iloc[position]
        invested = sum(
            shares * float(row.get(ticker, np.nan))
            for ticker, shares in holdings.items()
            if not pd.isna(row.get(ticker, np.nan))
        )
        value_rows.append(invested + cash)
        cash_rows.append(cash)
        holding_rows.append(dict(holdings))

    # Orders still queued when history ran out never executed. Saying so is the
    # difference between a replay and a story about a replay.
    unfilled.extend(pending)

    path = PortfolioPath(
        value=pd.Series(value_rows, index=sessions),
        cash=pd.Series(cash_rows, index=sessions),
        holdings=pd.DataFrame(holding_rows, index=sessions).fillna(0.0),
        flows=flow_series,
        fills=tuple(fills),
        unfilled=tuple(unfilled),
        cash_policy=cash_policy,
    )
    # An arrival that could not be priced is a data gap on the result, not a
    # silent omission. Left unsaid, a portfolio is simply missing shares the
    # user believes it holds and every figure below is quietly smaller.
    if unpriced:
        modelling_scope = {
            **(modelling_scope or {}),
            "unpriced_in_kind_arrivals": unpriced,
        }

    return MissionResult(
        path=path,
        time_weighted=time_weighted_returns(path.value, path.flows),
        money_weighted=money_weighted_return(
            path.flows, path.terminal_value, periods_per_year=periods_per_year
        ),
        periods_per_year=periods_per_year,
        modelling_scope=modelling_scope,
    )


def _next_session(sessions: pd.DatetimeIndex, date: pd.Timestamp) -> Optional[pd.Timestamp]:
    """The first session on or after `date`.

    A contribution dated the 1st of a month lands on the next trading session,
    not on the 1st. Treating the calendar date as tradeable is the same defect as
    the weekend padding that inflated annualized figures by 31%.
    """
    later = sessions[sessions >= pd.Timestamp(date)]
    return later[0] if len(later) else None


def _sessions_between(sessions: pd.DatetimeIndex, start, end) -> int:
    return int(sessions.get_loc(end)) - int(sessions.get_loc(start))


def _execute(
    order: Order,
    session: pd.Timestamp,
    prices: pd.DataFrame,
    holdings: Dict[str, float],
    cash: float,
    *,
    cost_bps: float,
    allow_fractional: bool,
) -> Optional[Fill]:
    """Fill an order, or decline to and say why by returning None."""
    if order.ticker not in prices.columns:
        return None
    price = float(prices.at[session, order.ticker])
    if not np.isfinite(price) or price <= 0:
        return None

    notional = order.notional
    cost_rate = cost_bps / 10_000.0

    if notional > 0:
        affordable = cash / (1.0 + cost_rate)
        notional = min(notional, affordable)
        if notional <= 0:
            return None
    else:
        held = holdings.get(order.ticker, 0.0)
        notional = max(notional, -held * price)
        if notional >= 0:
            return None

    shares = notional / price
    if not allow_fractional:
        shares = float(np.trunc(shares))
        notional = shares * price
        if shares == 0:
            return None

    return Fill(date=session, ticker=order.ticker, shares=shares, price=price,
                notional=notional, cost=abs(notional) * cost_rate,
                reason=order.reason)
