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
from enum import Enum
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
class InKindFlow:
    """Assets arriving from outside the account, already owned on arrival.

    The generic form of what a vest does. Deliberately not RSU-specific: an
    inherited security, a stock gift, a transfer in kind and an employer stock
    contribution are the same event to an account, and only their *semantics*
    differ. RSU rules live in `RSUVestingRuntime`; the engine understands this.

    Both the quantity and the value it entered at are carried, rather than the
    engine recomputing value from whichever session the event lands on. A vest
    whose date falls on a holiday lands on the next session, and valuing it at
    that session's price would make the external flow disagree with the
    withholding computed at the vest price — the conservation identity would
    then fail by an amount nobody could see.
    """

    date: pd.Timestamp
    asset: str
    quantity: float
    valuation_price: float
    external_value: float
    """What entered the account. Pinned, not derived.

    For a vest this is the *delivered* value, not the gross: withheld shares
    never arrive, so counting them would credit the account with something it
    does not hold."""

    source_ref: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"date": str(pd.Timestamp(self.date).date()), "asset": self.asset,
                "quantity": self.quantity,
                "valuation_price": self.valuation_price,
                "external_value": self.external_value,
                "source_ref": self.source_ref}


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


class MWRStatus(str, Enum):
    """What Quantify can say about a money-weighted return.

    Four outcomes because `docs/MWR.md` names four, plus one the
    *implementation* needs and the financial contract does not have. The old
    signature returned `Optional[float]`, which could say two things, so "no
    admissible rate" and "the data does not determine one" arrived identically
    and a non-unique series was reported as a number.
    """

    RATE = "rate"
    NO_SOLUTION = "no_solution"
    NON_UNIQUE = "non_unique"
    INSUFFICIENT_CASH_FLOWS = "insufficient_cash_flows"

    INDETERMINATE = "indeterminate"
    """The search could not establish which of the above applies.

    An implementation state, not a financial one: `docs/MWR.md` has four
    outcomes and this is not a fifth. A bounded numerical scan can miss a root
    that touches zero without crossing, or one beyond the range it searched,
    and reporting `NO_SOLUTION` there would turn "could not establish" into
    "established" — the substitution this whole boundary exists to prevent.
    """


@dataclass(frozen=True)
class MWRResult:
    """A rate, or the reason there is not one.

    The invariant is enforced rather than documented: a result carrying both
    `NO_SOLUTION` and a number is exactly what the old return type allowed.
    """

    status: MWRStatus
    rate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status is MWRStatus.RATE and self.rate is None:
            raise ValueError("a RATE result must carry a rate")
        if self.status is not MWRStatus.RATE and self.rate is not None:
            raise ValueError(
                f"{self.status.value} carries a rate; every outcome other "
                "than RATE means no number may be published")

    @property
    def reportable(self) -> bool:
        return self.status is MWRStatus.RATE

    def to_json(self) -> Dict[str, Any]:
        return {"status": self.status.value, "rate": self.rate}


def _sign_changes(coefficients: Sequence[float]) -> int:
    """Sign changes in the coefficient sequence, ignoring zeros.

    Descartes' rule of signs: the number of positive real roots is at most
    this and differs from it by an even number. One change therefore means
    exactly one positive root, which is the only case where this build can
    establish uniqueness without searching for it.
    """
    signs = [1 if c > 0 else -1 for c in coefficients if c != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def _admissible_roots(npv, tolerance: float, *,
                      lowest: float = -0.9999, highest: float = 1e4,
                      steps: int = 4000) -> List[float]:
    """Per-session rates above -1 where the present value crosses zero.

    Crossings only. A root that touches zero without crossing is invisible
    here, which is why the caller treats "fewer roots than Descartes permits"
    as unresolved rather than as an answer.
    """
    found: List[float] = []
    width = (highest - lowest) / steps
    previous_rate = lowest
    previous = npv(previous_rate)
    for step in range(1, steps + 1):
        rate = lowest + step * width
        value = npv(rate)
        if previous * value < 0:
            low, high = previous_rate, rate
            for _ in range(200):
                mid = (low + high) / 2.0
                if npv(low) * npv(mid) <= 0:
                    high = mid
                else:
                    low = mid
                if high - low < tolerance:
                    break
            found.append((low + high) / 2.0)
        previous_rate, previous = rate, value
    return [r for i, r in enumerate(found)
            if i == 0 or abs(r - found[i - 1]) > 1e-6]


def money_weighted_return(
    flows: pd.Series,
    terminal_value: float,
    *,
    periods_per_year: int = 252,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> MWRResult:
    """Annualized internal rate of return on the actual cash flow stream.

    What the investor experienced, as opposed to what the strategy did. The two
    diverge exactly when contribution timing mattered, which for anyone
    contributing regularly is most of the time.

    `flows` must be indexed on the **full session calendar**, not only on the
    days money moved: elapsed time is measured in sessions, and taking positions
    within a sparse flow index would price a three-month gap as three days.

    **Structured as the contract is** — validate, search, classify — because
    the previous version fused search and classification. It justified
    uniqueness by Descartes' rule in its docstring while nothing checked that
    the series had the shape the argument needs, so on a series containing a
    withdrawal, where the coefficients read `- + -` and two positive roots
    exist, bisection returned whichever one its opening bracket straddled and
    reported it unqualified.

    Uniqueness is now established rather than assumed. One sign change means
    exactly one positive root and a rate may be published. More than one means
    the rule permits several, so this searches, says `NON_UNIQUE` when it finds
    them, and refuses to publish a rate when it cannot tell.
    """
    if flows.empty or terminal_value <= 0:
        return MWRResult(MWRStatus.INSUFFICIENT_CASH_FLOWS)

    values = flows.to_numpy(dtype=float)
    moved = np.flatnonzero(values != 0.0)
    if moved.size == 0:
        return MWRResult(MWRStatus.INSUFFICIENT_CASH_FLOWS)

    sessions = moved.astype(float)
    amounts = values[moved]
    horizon = float(len(flows) - 1)

    def npv(rate: float) -> float:
        growth = (1.0 + rate) ** (horizon - sessions)
        return float(np.sum(amounts * growth) - terminal_value)

    # The polynomial in the growth factor, highest power first, with the
    # terminal value as the constant term. Descartes reads this, not the flows.
    exponents = horizon - sessions
    powers = sorted({int(p) for p in exponents}, reverse=True)
    coefficients = [float(np.sum(amounts[exponents == p])) for p in powers]
    if 0 not in powers:
        coefficients.append(0.0)
    coefficients[-1] -= terminal_value

    changes = _sign_changes(coefficients)
    if changes == 0:
        return MWRResult(MWRStatus.NO_SOLUTION)

    roots = _admissible_roots(npv, tolerance)

    if changes == 1:
        if not roots:
            return MWRResult(MWRStatus.INDETERMINATE)
        per_session = roots[0]
        return MWRResult(
            MWRStatus.RATE,
            float((1.0 + per_session) ** periods_per_year - 1.0))

    if len(roots) > 1:
        return MWRResult(MWRStatus.NON_UNIQUE)

    return MWRResult(MWRStatus.INDETERMINATE)
