"""Share-level portfolio accounting with external cash flows.

Everything published so far is a **time-weighted return**. That is not a
labelling detail: `strategy_daily_returns()` takes a weight matrix, and a weight
matrix presupposes a fully invested portfolio with no money arriving or leaving.
Chaining daily portfolio returns removes the effect of contribution timing by
construction — which is correct for evaluating a manager, and wrong for
answering "what would have happened if I had invested this way?", because
contribution timing is the thing the question is about.

A Mission has external cash flows by definition. So this module accounts in
**shares and cash**, not weights:

* "Buy $2,000 of VTI" is a notional order, not a target weight.
* "Never sell" is expressible on holdings and meaningless on weights.
* Uninvested cash is a position with a declared return, not a residual.

Both return bases are produced, always, and neither is offered alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


class CashPolicyError(ValueError):
    """Raised when uninvested cash has no declared treatment.

    $2,000 arrives on the 1st and the trigger fires on the 9th. Earning nothing,
    earning a money-market rate, or sitting in the default holding are three
    different answers producing three different numbers, and nobody will think to
    ask. Defaulting silently is how the platform's previous errata happened.
    """


@dataclass(frozen=True)
class CashFlow:
    """Money entering or leaving the portfolio from outside it.

    Positive is a contribution. This is the object time-weighted return exists to
    neutralise and money-weighted return exists to capture.
    """

    date: pd.Timestamp
    amount: float
    label: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"date": str(self.date.date()), "amount": self.amount,
                "label": self.label}


@dataclass(frozen=True)
class Grant:
    """Shares delivered from outside the portfolio, not bought with its cash.

    Vested equity is not a purchase, and modelling it as "cash arrives, then buy"
    is wrong in two ways: it puts a session of slippage between the vest and the
    holding, and it credits the plan with a trading decision nobody made.

    The delivered shares' value at the vest price *is* an external contribution —
    money entering the portfolio from outside — so it lands in the flow series
    and the money-weighted return counts it. Shares withheld for tax never
    arrive, so they are simply not granted rather than granted and sold.
    """

    date: pd.Timestamp
    ticker: str
    shares: float
    reason: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"date": str(self.date.date()), "ticker": self.ticker,
                "shares": self.shares, "reason": self.reason}


@dataclass(frozen=True)
class Order:
    """An instruction to trade a notional amount, submitted on `date`.

    Notional rather than shares because that is how the instruction is actually
    given — "buy $2,000 of VTI" — and converting to shares requires the execution
    price, which is not known when the instruction is formed.
    """

    date: pd.Timestamp
    ticker: str
    notional: float
    reason: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"date": str(self.date.date()), "ticker": self.ticker,
                "notional": self.notional, "reason": self.reason}


@dataclass(frozen=True)
class Fill:
    """What actually happened, at the price that was actually available."""

    date: pd.Timestamp
    ticker: str
    shares: float
    price: float
    notional: float
    cost: float
    reason: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"date": str(self.date.date()), "ticker": self.ticker,
                "shares": self.shares, "price": self.price,
                "notional": self.notional, "cost": self.cost, "reason": self.reason}


@dataclass(frozen=True)
class CashPolicy:
    """What uninvested cash does while it waits.

    Declared rather than assumed. `annual_rate` of 0.0 is a real choice — it says
    cash earns nothing — and it is a different choice from not having decided.
    """

    annual_rate: float
    basis: str = "declared"
    detail: str = ""

    @classmethod
    def idle(cls, detail: str = "cash earns nothing while uninvested") -> "CashPolicy":
        return cls(annual_rate=0.0, detail=detail)

    def daily_rate(self, periods_per_year: int) -> float:
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        return (1.0 + self.annual_rate) ** (1.0 / periods_per_year) - 1.0

    def to_json(self) -> Dict[str, Any]:
        return {"annual_rate": self.annual_rate, "basis": self.basis,
                "detail": self.detail}


@dataclass
class PortfolioPath:
    """The full valuation path, with flows kept separable from performance."""

    value: pd.Series
    """End-of-day total portfolio value, including uninvested cash."""

    cash: pd.Series
    holdings: pd.DataFrame
    """Shares held per ticker, end of day."""

    flows: pd.Series
    """External cash flows, indexed by the session they landed on."""

    fills: Sequence[Fill] = field(default_factory=tuple)
    unfilled: Sequence[Order] = field(default_factory=tuple)
    """Orders that could not execute — insufficient cash, no price, halted.
    Retained rather than dropped: an order that silently vanished is the
    difference between what the Mission declared and what it did."""

    cash_policy: Optional[CashPolicy] = None

    @property
    def contributed(self) -> float:
        return float(self.flows[self.flows > 0].sum())

    @property
    def withdrawn(self) -> float:
        return float(-self.flows[self.flows < 0].sum())

    @property
    def terminal_value(self) -> float:
        return float(self.value.iloc[-1]) if len(self.value) else 0.0


def time_weighted_returns(value: pd.Series, flows: pd.Series) -> pd.Series:
    """Daily returns with the effect of external flows removed.

    Convention, stated because it changes the number: a flow lands at the *start*
    of its session and is uninvested for that session, so it contributes to
    end-of-day value without having had the chance to earn. The return earned by
    capital that was actually at work is therefore::

        r_t = (V_t - F_t) / V_{t-1} - 1

    This matches the execution-lag discipline the rest of the engine uses — money
    that arrives today cannot be traded today.
    """
    value = value.astype(float)
    flows = flows.reindex(value.index).fillna(0.0).astype(float)

    prior = value.shift(1)
    invested_base = prior.where(prior > 0)
    returns = (value - flows) / invested_base - 1.0
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def money_weighted_return(
    flows: pd.Series,
    terminal_value: float,
    *,
    periods_per_year: int = 252,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> Optional[float]:
    """Annualized internal rate of return on the actual cash flow stream.

    What the investor experienced, as opposed to what the strategy did. The two
    diverge exactly when contribution timing mattered, which for anyone
    contributing regularly is most of the time.

    `flows` must be indexed on the **full session calendar**, not only on the
    days money moved: elapsed time is measured in sessions, and taking positions
    within a sparse flow index would price a three-month gap as three days.

    Solved by bisection rather than by a Newton method because the sign structure
    guarantees bisection converges. Contributions are outflows and the terminal
    value is a single inflow, so the coefficient sequence has exactly one sign
    change and Descartes' rule gives a unique positive root. Newton would be
    faster and could leave the bracket on a pathological path, and a silently
    wrong rate of return is worse than a slow one.

    Returns None when the stream has no sign change — no money was contributed,
    or nothing remains — because an IRR is undefined there rather than zero.
    """
    if flows.empty or terminal_value <= 0:
        return None

    values = flows.to_numpy(dtype=float)
    moved = np.flatnonzero(values != 0.0)
    if moved.size == 0:
        return None

    sessions = moved.astype(float)
    amounts = values[moved]
    horizon = float(len(flows) - 1)

    def npv(rate: float) -> float:
        growth = (1.0 + rate) ** (horizon - sessions)
        return float(np.sum(amounts * growth) - terminal_value)

    low, high = -0.9999, 1.0
    f_low, f_high = npv(low), npv(high)
    widened = 0
    while f_low * f_high > 0 and widened < 60:
        high *= 2.0
        f_high = npv(high)
        widened += 1
    if f_low * f_high > 0:
        return None

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < tolerance or (high - low) / 2.0 < tolerance:
            low = high = mid
            break
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid

    per_session = (low + high) / 2.0
    return float((1.0 + per_session) ** periods_per_year - 1.0)
