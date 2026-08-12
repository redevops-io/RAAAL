"""Stated splits and periodic rebalancing, against a price history that moves.

The prices here are constructed so the two strategies *must* diverge: one asset
triples over the run and the other is flat. Buy-and-hold at 60/40 ends far from
60/40; rebalancing sells the winner all the way up. If a test in this file can
pass with rebalancing disabled, it is testing nothing — several assertions below
exist only to make that impossible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mission.accounting import CashPolicy, CashFlow
from src.mission.rebalance import (UnsupportedRebalancing, normalised,
                                   weighted)
from src.mission.simulate import simulate

SESSIONS = pd.bdate_range("2020-01-01", "2023-12-29")


def prices() -> pd.DataFrame:
    """GROWTH triples on a smooth ramp; FLAT never moves.

    Deliberately monotone rather than random: a rebalancing test on noisy data
    passes or fails by luck, and the property under test is about drift, which
    a ramp produces in one direction and is easy to reason about.
    """
    ramp = np.linspace(100.0, 300.0, len(SESSIONS))
    return pd.DataFrame({"GROWTH": ramp,
                         "FLAT": np.full(len(SESSIONS), 100.0)},
                        index=SESSIONS)


def run(program, *, monthly: float = 1_000.0):
    frame = prices()
    flows = [CashFlow(date=d, amount=monthly, label="contribution")
             for d in pd.bdate_range(SESSIONS[0], SESSIONS[-1], freq="BMS")]
    return simulate(frame, flows=flows, program=program,
                    cash_policy=CashPolicy.idle())


def final_split(result) -> dict:
    row = prices().iloc[-1]
    holdings = result.path.holdings.iloc[-1]
    value = {t: float(holdings.get(t, 0.0)) * float(row[t])
             for t in ("GROWTH", "FLAT")}
    total = sum(value.values())
    return {t: v / total for t, v in value.items()} if total else {}


class TestTheSplitIsHonoured:
    def test_an_unstated_split_is_equal(self):
        assert normalised(["A", "B"]) == {"A": 0.5, "B": 0.5}

    def test_a_stated_split_is_normalised(self):
        assert normalised(["A", "B"], {"A": 60, "B": 40}) == {"A": 0.6,
                                                              "B": 0.4}

    def test_percentages_and_fractions_agree(self):
        assert (normalised(["A", "B"], {"A": 0.6, "B": 0.4})
                == normalised(["A", "B"], {"A": 60, "B": 40}))

    def test_weights_that_buy_nothing_are_refused(self):
        """Not silently equalised. A split summing to zero is a statement
        nobody meant, and inventing equal weights for it would run a strategy
        the user did not describe."""
        with pytest.raises(ValueError):
            normalised(["A", "B"], {"A": 0, "B": 0})

    def test_money_goes_in_at_the_stated_split(self):
        """60/40 on the way in, with no rebalancing. Measured on the *first*
        month, before drift has had time to move it."""
        result = run(weighted(["GROWTH", "FLAT"],
                              weights={"GROWTH": 60, "FLAT": 40}))
        early = result.path.holdings.iloc[25]
        row = prices().iloc[25]
        value = {t: float(early[t]) * float(row[t]) for t in ("GROWTH", "FLAT")}
        total = sum(value.values())
        assert value["GROWTH"] / total == pytest.approx(0.6, abs=0.02)


class TestRebalancingChangesTheOutcome:
    """The falsification. Every assertion here fails if the rebalance branch
    never fires, which is what makes the rest of the file evidence."""

    WEIGHTS = {"GROWTH": 60, "FLAT": 40}

    def held(self):
        return run(weighted(["GROWTH", "FLAT"], weights=self.WEIGHTS))

    def rebalanced(self):
        return run(weighted(["GROWTH", "FLAT"], weights=self.WEIGHTS,
                            rebalance="annual", sessions=SESSIONS))

    def test_buy_and_hold_drifts_away_from_the_target(self):
        """If this fails, the price history is not doing its job and every
        comparison below is against a portfolio that never drifted."""
        assert final_split(self.held())["GROWTH"] > 0.65

    def test_rebalancing_holds_near_the_target(self):
        assert final_split(self.rebalanced())["GROWTH"] == pytest.approx(
            0.6, abs=0.05)

    def test_the_two_are_not_the_same_strategy(self):
        """The figure differs, not just the composition. A rebalancing
        implementation that emitted orders which netted to nothing would pass
        every split assertion above and change no outcome at all."""
        assert self.rebalanced().final_value != pytest.approx(
            self.held().final_value, rel=0.01)

    def test_rebalancing_actually_sold_something(self):
        """The engine's first sale. Asserted directly rather than inferred
        from a figure, because a negative order that never filled would leave
        the split assertions passing on a portfolio that only ever bought."""
        sells = [f for f in self.rebalanced().path.fills if f.shares < 0]
        assert sells, "no sale was filled, so nothing was rebalanced"
        assert all(f.ticker == "GROWTH" for f in sells), (
            "the winner is what a rebalance sells on a monotone ramp")

    def test_buy_and_hold_never_sells(self):
        """The reciprocal, and the property `sells_allowed=False` depends on.
        A program that sold without being asked would be a different strategy
        wearing the same name."""
        assert not [f for f in self.held().path.fills if f.shares < 0]


class TestTheOrderOfOrdersMatters:
    def test_sells_are_queued_before_buys(self):
        """`simulate` funds each purchase from a running cash balance, so a buy
        ahead of the sale that pays for it is clamped to whatever cash happens
        to be there — and the rebalance ends further from target than it
        started, on the day it was meant to correct it."""
        program = weighted(["GROWTH", "FLAT"], weights={"GROWTH": 60,
                                                        "FLAT": 40},
                           rebalance="annual", sessions=SESSIONS)
        frame = prices()
        # A drifted portfolio: far too much GROWTH, and no cash to buy with.
        boundary = sorted(_boundaries_of("annual"))[0]
        visible = frame.loc[:boundary]
        orders = program(boundary, visible, {"GROWTH": 100.0, "FLAT": 1.0},
                         0.0)
        assert orders, "a drifted portfolio produced no rebalancing orders"
        signs = [o.notional < 0 for o in orders]
        assert signs == sorted(signs, reverse=True), (
            f"buys are queued before sells: {[o.notional for o in orders]}")


class TestItRefusesWhatItCannotPlace:
    def test_an_unknown_cadence_is_refused_by_name(self):
        with pytest.raises(UnsupportedRebalancing) as refused:
            weighted(["A"], rebalance="fortnightly", sessions=SESSIONS)
        assert "fortnightly" in str(refused.value)

    def test_the_first_period_is_not_a_boundary(self):
        """Rebalancing on day one corrects a drift that does not exist yet and
        pays a spread to do it."""
        assert SESSIONS[0] not in _boundaries_of("annual")

    def test_an_unpriced_asset_skips_the_whole_rebalance(self):
        """Not a partial one. Retargeting only the assets that priced is a
        different strategy, and it would run without saying so."""
        program = weighted(["GROWTH", "FLAT"], rebalance="annual",
                           sessions=SESSIONS)
        boundary = sorted(_boundaries_of("annual"))[0]
        visible = prices().loc[:boundary].copy()
        visible.loc[visible.index[-1], "FLAT"] = float("nan")
        assert program(boundary, visible, {"GROWTH": 100.0, "FLAT": 1.0},
                       5_000.0) == ()


def _boundaries_of(cadence: str) -> set:
    from src.mission.rebalance import _boundaries

    return _boundaries(SESSIONS, cadence)
