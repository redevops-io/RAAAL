"""The evaluation engine, checked against arithmetic it did not perform.

Every other test asks the engine whether it agrees with itself. This one
reconstructs the answer from the two things the engine does not get to define —
the raw price snapshot and the trade log it emitted — and checks the number it
reported against a valuation computed here, in a few lines that share no code
with `mission.simulate`.

The reconstruction is deliberately naive: walk the sessions in order, add each
contribution to cash, apply each fill (shares in, notional and cost out of
cash), and value the holdings at that session's snapshot price. If the engine's
`value` series disagrees with that anywhere, one of them is wrong, and the naive
one has nowhere to hide a mistake.

Two properties are checked that no reconstruction could establish on its own:

  * **fills are priced against the snapshot** — a fill at a price the data does
    not carry is a number invented, and it would reconcile perfectly against
    itself while being fiction. Checked against the parquet, not the fill.
  * **no look-ahead** — corrupt every price after a cutoff and the path up to
    the cutoff may not move by a cent. A decision that reads tomorrow's price
    changes here and nowhere a self-consistent check would look.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from runtime_contracts import Author, IntentField, VerifiedIntent

from src.evaluation.core import evaluate_plan
from src.mission.from_intent import compile_intent

FIXTURE = "tests/fixtures/prices_synthetic.parquet"


def _prices() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


def _run(fields, prices, stated_text=""):
    built = {k: IntentField(value=v, author=Author.READER)
             for k, v in fields.items()}
    draft = VerifiedIntent(objective="evaluate_investment_strategy",
                           produced_by="independent", utterance_ref="independent",
                           fields=built, unresolved=())
    scenario = compile_intent(draft.seal()).scenario
    assert scenario is not None, f"{fields} did not compile"
    result = evaluate_plan(scenario, prices, stated_text=stated_text)
    assert result.refusal is None, f"{fields} refused: {result.refusal}"
    return result.result.path


def _reconstruct(path, prices):
    """Cash, holdings and NAV per session, from flows and fills alone.

    Shares no code with the simulator: a running cash balance, a running share
    count per ticker, and holdings valued at the snapshot's own price. Returns
    the three series so each can be checked against the engine's separately.
    """
    sessions = path.value.index
    price_at = prices.reindex(sessions)

    fills_by_day: dict = {}
    for fill in path.fills:
        fills_by_day.setdefault(pd.Timestamp(fill.date), []).append(fill)

    cash = 0.0
    holdings: dict = {}
    nav = pd.Series(index=sessions, dtype=float)
    cash_series = pd.Series(index=sessions, dtype=float)
    held_series = {}

    flows = path.flows
    for session in sessions:
        if session in flows.index:
            cash += float(flows.loc[session])
        for fill in fills_by_day.get(session, ()):
            holdings[fill.ticker] = holdings.get(fill.ticker, 0.0) + fill.shares
            cash -= fill.notional
            cash -= fill.cost
        invested = 0.0
        for ticker, shares in holdings.items():
            if shares == 0.0:
                continue
            price = price_at.at[session, ticker] if ticker in price_at.columns \
                else np.nan
            if np.isfinite(price):
                invested += shares * price
        nav.loc[session] = cash + invested
        cash_series.loc[session] = cash
        held_series[session] = dict(holdings)

    return nav, cash_series, held_series


#: A representative spread: one holding, a stated split that rebalances, and two
#: computed strategies whose weights change every quarter. The accounting is the
#: same machine under all of them; the strategies exercise the most of it.
PLANS = {
    "single_fund": {"amount": "500", "cadence": "monthly", "assets": "SPY"},
    "stated_split": {"amount": "500", "cadence": "monthly", "assets": "SPY,TLT",
                     "allocation_method": "stated_weights",
                     "stated_weights": "SPY=60,TLT=40",
                     "periodic_rebalancing": "annual"},
    "risk_parity": {"amount": "500", "cadence": "monthly",
                    "allocation_method": "risk_parity",
                    "periodic_rebalancing": "quarterly"},
    "momentum": {"amount": "500", "cadence": "monthly",
                 "allocation_method": "time_series_momentum",
                 "periodic_rebalancing": "quarterly"},
}


@pytest.fixture(scope="module")
def prices():
    return _prices()


@pytest.mark.parametrize("name", list(PLANS))
def test_reported_value_matches_an_independent_valuation(name, prices):
    """The engine's NAV, recomputed from raw prices and its own fills."""
    path = _run(PLANS[name], prices)
    nav, _cash, _held = _reconstruct(path, prices)

    diff = (nav - path.value).abs()
    scale = path.value.abs().clip(lower=1.0)
    worst_rel = float((diff / scale).max())
    assert worst_rel < 1e-9, (
        f"{name}: reconstructed NAV diverges from the reported value by "
        f"{worst_rel:.2e} (max ${float(diff.max()):.4f}) — the engine's "
        "valuation does not match the trades it recorded")


@pytest.mark.parametrize("name", list(PLANS))
def test_cash_and_holdings_reconstruct(name, prices):
    """The cash and share series, rebuilt from flows and fills."""
    path = _run(PLANS[name], prices)
    _nav, cash, held = _reconstruct(path, prices)

    cash_diff = float((cash - path.cash).abs().max())
    assert cash_diff < 1e-6, (
        f"{name}: reconstructed cash diverges by ${cash_diff:.6f}")

    last = path.value.index[-1]
    for ticker, shares in held[last].items():
        if ticker in path.holdings.columns:
            engine = float(path.holdings.at[last, ticker])
            assert abs(engine - shares) < 1e-6, (
                f"{name}: {ticker} shares {engine} != reconstructed {shares}")


@pytest.mark.parametrize("name", list(PLANS))
def test_fills_are_priced_against_the_snapshot(name, prices):
    """A fill at a price the data does not carry is a number invented."""
    path = _run(PLANS[name], prices)
    for fill in path.fills:
        day = pd.Timestamp(fill.date)
        assert day in prices.index, f"{name}: fill on a non-session {day}"
        snapshot = float(prices.at[day, fill.ticker])
        assert abs(fill.price - snapshot) < 1e-9, (
            f"{name}: {fill.ticker} filled at {fill.price}, snapshot says "
            f"{snapshot} — the engine priced against something else")
        # notional is shares at that price, and cost is never negative.
        assert abs(fill.notional - fill.shares * fill.price) < 1e-6
        assert fill.cost >= -1e-12


@pytest.mark.parametrize("name", list(PLANS))
def test_every_contributed_dollar_is_invested_or_held(name, prices):
    """Nothing leaks. Ending cash is exactly what went in minus what was spent
    on fills and their costs — no dollar created, none destroyed."""
    path = _run(PLANS[name], prices)
    contributed = float(path.flows.sum())
    spent = sum(f.notional + f.cost for f in path.fills)
    ending_cash = float(path.cash.iloc[-1])
    assert abs(contributed - spent - ending_cash) < 1e-4, (
        f"{name}: ${contributed:.2f} in, ${spent:.2f} spent, ${ending_cash:.2f} "
        "left — the three do not balance")


def test_hold_cash_earns_exactly_the_contributions(prices):
    """A plan that never invests ends at the sum of what was paid in — the
    control that proves the contribution schedule itself is counted right."""
    from src.mission.benchmark import hold_cash
    from src.mission.accounting import CashFlow, CashPolicy
    from src.mission.simulate import simulate

    path = _run(PLANS["single_fund"], prices)
    flows = tuple(CashFlow(date=d, amount=float(v))
                  for d, v in path.flows.items() if v)
    cash_only = simulate(prices, flows=flows, program=hold_cash(),
                         cash_policy=CashPolicy.idle())
    contributed = sum(f.amount for f in flows)
    assert abs(float(cash_only.path.value.iloc[-1]) - contributed) < 1e-6


def test_single_fund_matches_a_from_scratch_backtest(prices):
    """The whole loop, reimplemented in fifteen lines that call nothing.

    NAV reconstruction trusts the engine's fills and checks only that it valued
    them right. This trusts nothing: it derives the contribution calendar from
    the snapshot, buys the fund itself under the documented conventions — money
    lands on the first session of each month, the order fills the next session
    at that session's price, a 10bps cost makes `notional + cost` exactly the
    cash spent — and values the position at every close. If the engine's path
    matches this, its decisions and its accounting are both right, checked
    against an implementation that shares no line with it.
    """
    cost_rate = 10.0 / 10_000.0
    col = "SPY"
    sessions = prices.index
    first_of_month = set(
        prices.groupby([sessions.year, sessions.month]).apply(
            lambda g: g.index.min()).values)

    cash = 0.0
    shares = 0.0
    pending = False
    reference = pd.Series(index=sessions, dtype=float)
    for day in sessions:
        if pending:                       # fill last session's order, here/now
            price = float(prices.at[day, col])
            notional = cash / (1.0 + cost_rate)
            shares += notional / price
            cash -= notional * (1.0 + cost_rate)   # == cash, to float precision
            pending = False
        if day in first_of_month:
            cash += 500.0
        if cash > 1e-9:
            pending = True                # buy the whole balance next session
        reference.loc[day] = shares * float(prices.at[day, col]) + cash

    path = _run(PLANS["single_fund"], prices)
    diff = (reference - path.value).abs()
    scale = path.value.abs().clip(lower=1.0)
    worst_rel = float((diff / scale).max())
    assert worst_rel < 1e-9, (
        f"a from-scratch single-fund backtest diverges from the engine by "
        f"{worst_rel:.2e} (max ${float(diff.max()):.4f})")


@pytest.mark.parametrize("name", ["risk_parity", "momentum"])
def test_no_look_ahead(name, prices):
    """Corrupt every price after a cutoff; the path up to it may not move.

    The sharpest check of the lot, and the one a self-consistent reconstruction
    cannot make: it compares two runs of the engine against each other, and a
    decision that read a future price is the only thing that moves the earlier
    path when a later price changes."""
    cutoff = prices.index[len(prices) // 2]
    baseline = _run(PLANS[name], prices)

    corrupted = prices.copy()
    after = corrupted.index > cutoff
    corrupted.loc[after] = corrupted.loc[after] * 1.37 + 3.0
    perturbed = _run(PLANS[name], corrupted)

    a = baseline.value.loc[:cutoff]
    b = perturbed.value.loc[:cutoff]
    worst = float((a - b).abs().max())
    assert worst < 1e-6, (
        f"{name}: corrupting prices after {cutoff.date()} moved the earlier "
        f"path by ${worst:.6f} — a decision read a price it could not have seen")


def test_an_uncovered_instrument_refuses_as_a_named_data_gap(prices):
    """A symbol the snapshot cannot price refuses explicitly, by name, and blames
    the data rather than the description. This is the NVDA report: 'invest $50
    into NVDA every week' against a snapshot that prices a fixed ETF/macro
    universe. The old message — 'no price history for X over this period' — read
    as a bad ticker or a wrong date and sent the person editing an input that was
    never the problem."""
    fields = {"amount": "50", "cadence": "weekly", "assets": "NVDA"}
    built = {k: IntentField(value=v, author=Author.READER) for k, v in fields.items()}
    draft = VerifiedIntent(objective="evaluate_investment_strategy",
                           produced_by="independent", utterance_ref="independent",
                           fields=built, unresolved=())
    scenario = compile_intent(draft.seal()).scenario
    assert scenario is not None, "the plan did not compile"
    assert "NVDA" not in prices.columns, "fixture unexpectedly prices NVDA"

    result = evaluate_plan(scenario, prices)
    assert not result.publishable
    assert result.refusal_kind == "data_gap"
    # Named, and attributed to the data, not the input.
    assert "no pricing data for NVDA" in result.refusal
    assert "not a problem with your description" in result.refusal
    # The misleading period framing is gone for an instrument simply not covered.
    assert "over this period" not in result.refusal
