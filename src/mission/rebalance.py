"""Buying to a stated split, and restoring it on a calendar.

The engine could already do both and never did. `simulate` fills a negative
order by clamping it to the position held, and `buy_and_hold` has taken a
`weights` argument since it was written — nothing passed one, no program ever
emitted a sale, and the capability manifest recorded `stated_weights` and
`periodic_rebalancing` as refused. Those refusals described the compiler, not
the simulator, and a user reading them was told this build cannot do something
it very nearly could.

Two rules a rebalancing program has to get right, and both are about the fact
that orders fill on the *next* session:

**Sells are emitted before buys.** `simulate` processes due orders in the order
they were queued, subtracting each fill from a running cash balance, and clamps
a purchase to the cash available at that moment. A rebalance that queued its
buys first would find them unfunded, fill them partially, and then sell into
cash nobody spent — the portfolio would drift *away* from the target it was
restoring, on the day it was supposed to be restored.

**A rebalance session does not also invest its cash separately.** The target is
computed over the whole portfolio including uninvested cash, so the orders that
restore it already spend the money that arrived. Emitting a contribution buy as
well would spend it twice, and the second order would be clamped rather than
refused — a quiet under-purchase rather than an error.

What this does not do is claim precision it does not have. Targets are computed
from today's prices and fill at tomorrow's, so a rebalance lands near the stated
split rather than on it. That is what actually happens to anyone rebalancing a
real portfolio, and pretending otherwise would require executing at a price the
decision could not have seen.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .accounting import Order
from .schedule import PERIOD_KEYS


class UnsupportedRebalancing(ValueError):
    """Raised for a rebalancing cadence this build cannot place on a calendar."""


def normalised(tickers: Sequence[str],
               weights: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    """Target shares that sum to one.

    An unstated split is equal by declaration rather than by accident: the
    caller gets the same answer whether it passed nothing or passed equal
    weights, and `weighting` on the allocation rule is what records which of
    those the user actually said.
    """
    if not tickers:
        return {}
    if not weights:
        return {t: 1.0 / len(tickers) for t in tickers}

    stated = {t: float(weights.get(t, 0.0)) for t in tickers}
    total = sum(stated.values())
    if total <= 0:
        raise ValueError(
            "the stated weights sum to zero, so nothing would ever be bought")
    return {t: w / total for t, w in stated.items()}


def _boundaries(sessions: pd.DatetimeIndex, cadence: str) -> set:
    """The first session of each period, after the first period.

    The opening period is excluded deliberately. Rebalancing on day one
    restores a split that the first purchase is about to establish anyway, and
    it would fire before any drift exists — an order with nothing to correct,
    charged a spread.
    """
    if cadence not in PERIOD_KEYS:
        raise UnsupportedRebalancing(
            f"This build cannot rebalance on a {cadence!r} calendar. Naming "
            "annual, quarterly, monthly, weekly or biweekly would let this "
            "plan run.")
    groups = sessions.to_series().groupby(PERIOD_KEYS[cadence](sessions))
    firsts = list(groups.min())
    return set(firsts[1:])


def weighted(tickers: Sequence[str], *,
             weights: Optional[Mapping[str, float]] = None,
             rebalance: str = "",
             sessions: Optional[pd.DatetimeIndex] = None,
             reason: str = "stated split"):
    """Invest arriving cash at the target split, and optionally restore it.

    With no `rebalance` cadence this is `buy_and_hold` with the weights
    honoured: money is divided as stated on the way in and nothing is ever
    sold, which is a different strategy from rebalancing and must stay
    distinguishable from it.
    """
    share = normalised(tickers, weights)
    dates = (_boundaries(sessions, rebalance)
             if rebalance and sessions is not None else set())

    def program(session, visible, holdings, cash):
        if session in dates:
            return _restore(session, visible, holdings, cash, share, reason)
        if cash <= 0:
            return ()
        return [Order(date=session, ticker=t, notional=cash * s, reason=reason)
                for t, s in share.items() if s > 0]

    return program


def strategy_driven(tickers: Sequence[str], *,
                    capability_id: str,
                    cadence: str,
                    sessions: pd.DatetimeIndex,
                    min_history: int = 1,
                    reason: str = "strategy allocation"):
    """Restore, on a calendar, to weights a research strategy computes.

    This is `weighted` with the split recomputed each period instead of stated
    once. On every rebalancing boundary the strategy is handed the price history
    *visible through that session* — never further — and the weights it returns
    become the new target the portfolio is restored to. Between boundaries,
    arriving cash is invested at the last computed split, exactly as a stated
    one would be, so a contribution is never left idle.

    The look-ahead safety is `_restore`'s: a boundary decides on that session's
    close and the orders fill on the next open, so the strategy sees only what a
    person deciding that morning could have seen. `run_capability` is imported
    lazily, here and nowhere else, so the heavy research stack loads only when a
    strategy plan actually runs — not on every compile that merely asks whether
    a weighting *is* one.

    Until `min_history` sessions have accrued the strategy has too little to
    compute on; those early boundaries invest equally rather than refusing, and
    the first boundary with enough history rebalances to the real split. That
    warm-up is equal weight by declaration, not the strategy wearing its name.
    """
    dates = (_boundaries(sessions, cadence)
             if cadence and sessions is not None else set())
    state: Dict[str, Dict[str, float]] = {"share": {}}

    def _compute(visible: pd.DataFrame) -> Dict[str, float]:
        from src.strategies import run_capability     # lazy: heavy research stack
        window = visible.dropna(how="all")
        if len(window) < max(2, int(min_history)):
            return {}
        returns = (np.log(window / window.shift(1))
                   .replace([np.inf, -np.inf], np.nan).dropna(how="all"))
        raw = run_capability(capability_id, window, returns, None, {}) or {}
        picked = {t: float(raw.get(t, 0.0)) for t in tickers}
        total = sum(w for w in picked.values() if w > 0)
        if total <= 0:
            return {}
        return {t: (w / total if w > 0 else 0.0) for t, w in picked.items()}

    def program(session, visible, holdings, cash):
        if session in dates:
            computed = _compute(visible)
            if computed:
                state["share"] = computed
                return _restore(session, visible, holdings, cash,
                                computed, reason)
            # Too little history yet: fall through and invest the contribution
            # at the warm-up split rather than leaving it in cash.
        share = state["share"] or normalised(tickers)
        if cash <= 0:
            return ()
        return [Order(date=session, ticker=t, notional=cash * s, reason=reason)
                for t, s in share.items() if s > 0]

    return program


def _restore(session, visible, holdings, cash, share, reason):
    """Orders that move the whole portfolio back to `share`.

    Includes cash in the total on purpose. A rebalance that ignored it would
    restore the split among invested assets and leave the contribution sitting
    idle until the next one, which is neither what anybody means by rebalancing
    nor what the flow-matched benchmark does.
    """
    row = visible.iloc[-1]

    held: Dict[str, float] = {}
    for ticker in share:
        price = float(row.get(ticker, np.nan))
        if not np.isfinite(price) or price <= 0:
            # An asset with no usable price today cannot be valued, so the
            # portfolio total is unknown and every target computed from it
            # would be wrong. Skipping the whole rebalance is the honest
            # response: a partial one silently retargets the assets that did
            # price, which is a different strategy nobody asked for.
            return ()
        held[ticker] = float(holdings.get(ticker, 0.0)) * price

    total = sum(held.values()) + float(cash)
    if total <= 0:
        return ()

    orders = []
    for ticker, target_share in share.items():
        delta = total * target_share - held[ticker]
        # A threshold, not a strict inequality. Floating point leaves a few
        # cents of drift on every session, and an order for $0.004 costs a
        # spread and moves nothing.
        if abs(delta) >= 1.0:
            orders.append(Order(date=session, ticker=ticker, notional=delta,
                                reason=reason))

    # Sells first. See the module docstring: `simulate` funds each purchase
    # from a running cash balance, so a buy queued ahead of the sale that pays
    # for it is clamped, and the portfolio ends further from target than it
    # started.
    orders.sort(key=lambda o: o.notional)
    return orders
