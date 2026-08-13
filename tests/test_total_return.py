"""Dividends, credited to the figure rather than recorded beside it.

`dividend_policy` was one of the project's own examples of a declared behaviour
that did nothing: the compiler recognised it, the confirmation screen quoted it
back, the scenario hashed it into its identity, and the engine ran on price
series only — so "reinvest the dividends" and "hold them as cash" produced the
same number. The disclosure said so honestly, which made it a known gap rather
than a hidden one, and it has been a known gap for a long time.

The snapshots carry a total-return series beside the price series. Reinvestment
is now run on it. The half that remains unmodelled is the other reading —
distributions paid out and left uninvested — and that is disclosed on its own
terms, because a blanket "dividends are not modelled" became false for the
common case, and a false disclosure is worse than no disclosure.

The assertions that matter here are the ones about a *figure*. A test that only
checked which frame was loaded would pass on an engine that loaded the right
series and then ignored it, which is precisely the shape of the defect this
closes.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.market_data.loader import (load_prices, synthetic_snapshot,
                                    total_return_path)

#: A high-yielding bond proxy, a growth fund, and a non-distributing asset.
#: The third is the control: it must be identical in both series, or the twin
#: is applying a blanket scale factor rather than crediting declared yields.
YIELDING, GROWTH, NON_PAYING = "BND", "QQQ", "GLD"


@pytest.fixture(scope="module")
def series():
    return (load_prices(synthetic_snapshot()),
            load_prices(synthetic_snapshot(), reinvested=True))


class TestTheTwinIsRealAndDeclared:
    def test_the_snapshot_has_one(self):
        assert total_return_path(synthetic_snapshot()) is not None

    def test_a_distributing_asset_compounds_ahead_of_its_price(self, series):
        price, total = series
        assert (total[YIELDING].dropna().iloc[-1]
                > price[YIELDING].dropna().iloc[-1] * 1.2), (
            "the bond proxy's total return is barely ahead of its price; a "
            "declared 3.8% yield over ten years is not a rounding difference")

    def test_a_non_distributing_asset_is_identical(self, series):
        """The control. Without it a twin built by scaling every column would
        pass every assertion above while modelling nothing."""
        price, total = series
        assert price[NON_PAYING].equals(total[NON_PAYING])

    def test_the_yields_differ_between_assets(self, series):
        """A uniform yield would let a total-return bug hide as a scale factor
        that cancels out of every comparison."""
        price, total = series
        def lift(column):
            return (total[column].dropna().iloc[-1]
                    / price[column].dropna().iloc[-1])
        assert lift(YIELDING) > lift(GROWTH) > 1.0

    def test_both_series_cover_the_same_sessions(self, series):
        """Two frames of different length would silently change what a
        backtest measured, not only what it earned."""
        price, total = series
        assert price.index.equals(total.index)
        assert list(price.columns) == list(total.columns)


class TestThePolicyDecidesWhichSeriesRuns:
    def test_reinvested_asks_for_the_twin(self):
        from src.mission.scenario import HoldingsPolicy
        from src.workspace.run_boundary import _reinvests

        class Scenario:
            holdings_policy = HoldingsPolicy(dividend_policy="reinvested")

        assert _reinvests(Scenario())

    def test_held_as_cash_does_not(self):
        from src.mission.scenario import HoldingsPolicy
        from src.workspace.run_boundary import _reinvests

        class Scenario:
            holdings_policy = HoldingsPolicy(dividend_policy="held_as_cash")

        assert not _reinvests(Scenario())


class TestItChangesTheFigure:
    """The assertions that would fail on an engine that loaded the right series
    and then ignored it."""

    def _final_value(self, frame, ticker):
        from src.mission.accounting import CashFlow, CashPolicy
        from src.mission.rebalance import weighted
        from src.mission.simulate import simulate

        flows = [CashFlow(date=d, amount=500.0, label="contribution")
                 for d in pd.bdate_range(frame.index[0], frame.index[-1],
                                         freq="BMS")]
        return simulate(frame[[ticker]].dropna(), flows=flows,
                        program=weighted([ticker]),
                        cash_policy=CashPolicy.idle()).final_value

    def test_reinvesting_ends_ahead_on_a_distributing_asset(self, series):
        price, total = series
        assert (self._final_value(total, YIELDING)
                > self._final_value(price, YIELDING) * 1.1), (
            "reinvested dividends did not change the figure; the policy is "
            "being recorded again rather than honoured")

    def test_the_two_policies_no_longer_produce_one_number(self, series):
        """The defect in one line. Two materially different strategies used to
        return the same figure because both ran on the price series."""
        price, total = series
        assert (self._final_value(price, YIELDING)
                != self._final_value(total, YIELDING))

    def test_a_non_distributing_asset_is_unaffected(self, series):
        """The control again, at the level of the figure rather than the frame."""
        price, total = series
        assert (self._final_value(price, NON_PAYING)
                == self._final_value(total, NON_PAYING))
