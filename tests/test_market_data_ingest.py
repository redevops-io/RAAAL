"""A vendor that rewrites history, and what has to survive it.

The concrete case: NVDA split 10:1 on 2024-06-10. Before the split Yahoo
reported 2024-06-06 as ~1209; today it reports ~121. Both `auto_adjust=True`
and `auto_adjust=False` do this — the unadjusted `Close` is split-adjusted
too, so there is no raw price to fall back on.

That produces two failure modes, and they are opposites:

    append two fetches   -> a 10x cliff that is an artifact of the assembly
    overwrite the cache  -> the cliff disappears, and so does the fact that
                            every number a stored run cited has changed

The second is the dangerous one, because it is silent. These tests fix the
behaviour that makes it audible.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.market_data.ingest import (
    MOVE_THRESHOLD,
    PanelFetch,
    reconcile,
    unexplained,
)


def sessions(count: int, start: str = "2024-05-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=count)


def panel(prices: pd.DataFrame, splits=None, dividends=None) -> PanelFetch:
    return PanelFetch(
        prices=prices,
        splits=splits or {},
        dividends=dividends or {},
        fetched_at=pd.Timestamp("2026-08-04", tz="UTC").to_pydatetime(),
    )


class TestAStitchedSplitIsCaught:
    """Two fetches either side of a split, concatenated: the exact defect."""

    def test_the_cliff_is_reported_when_no_split_explains_it(self):
        index = sessions(20)
        # Pre-split prices for the first half, post-split for the second: what
        # appending a fresh fetch onto a stale cache produces.
        values = [1200.0] * 10 + [120.0] * 10
        frame = pd.DataFrame({"NVDA": values}, index=index)

        found = unexplained(panel(frame))

        assert "NVDA" in found, "a 10x single-session drop went unreported"
        assert (found["NVDA"].abs() >= MOVE_THRESHOLD).all()

    def test_the_same_cliff_is_excused_when_a_split_explains_it(self):
        """The discriminator is the corporate action, not the size."""
        index = sessions(20)
        frame = pd.DataFrame({"NVDA": [1200.0] * 10 + [120.0] * 10}, index=index)
        splits = {"NVDA": pd.Series([10.0], index=[index[10]])}

        assert "NVDA" not in unexplained(panel(frame, splits=splits))


class TestTheThresholdAloneWouldBeNoise:
    """Why the check attributes rather than thresholds.

    Run against the real cache, a bare threshold flagged BTC-USD at -37% and
    ^VIX at -36%. Both are ordinary sessions for those instruments. A check
    that fires on healthy data every week is one nobody reads by the second.
    """

    @pytest.mark.parametrize("ticker,move", [("BTC-USD", -0.372), ("^VIX", -0.358)])
    def test_a_volatile_instrument_still_reports_without_an_action(self, ticker, move):
        index = sessions(6)
        base = [100.0, 101.0, 102.0]
        after = 102.0 * (1 + move)
        frame = pd.DataFrame({ticker: base + [after, after, after]}, index=index)

        # Honest about the limit: with no action table, this *does* fire. The
        # fix is not a cleverer threshold, it is that a real fetch carries the
        # splits table and a genuine split is then excused.
        assert ticker in unexplained(panel(frame))

    def test_and_is_silenced_by_the_action_table(self):
        index = sessions(6)
        frame = pd.DataFrame({"T": [100.0, 101.0, 102.0, 10.2, 10.2, 10.2]},
                             index=index)
        splits = {"T": pd.Series([10.0], index=[index[3]])}
        assert unexplained(panel(frame, splits=splits)) == {}


class TestSilentRewritesBecomeVisible:
    """The failure the old path had: overwrite, and say nothing."""

    def test_a_retro_adjustment_is_attributed_to_its_split(self):
        index = sessions(20)
        before = pd.DataFrame({"NVDA": [1200.0] * 20}, index=index)
        after = pd.DataFrame({"NVDA": [120.0] * 20}, index=index)
        splits = {"NVDA": pd.Series([10.0], index=[index[-1]])}

        result = reconcile(before, panel(after, splits=splits))

        assert result.changed["NVDA"] == 20
        assert result.explained["NVDA"] == 20
        assert result.clean, "a split-explained rewrite should not raise an alarm"

    def test_a_rewrite_with_no_action_behind_it_is_not_clean(self):
        """A vendor revising history is the operator's decision, not a default."""
        index = sessions(20)
        before = pd.DataFrame({"SPY": [400.0] * 20}, index=index)
        after = pd.DataFrame({"SPY": [402.0] * 20}, index=index)

        result = reconcile(before, panel(after))

        assert not result.clean
        assert result.changed["SPY"] == 20
        assert len(result.unexplained_changes["SPY"]) == 20

    def test_an_unchanged_refetch_is_clean_and_quiet(self):
        """The premise. If identical data reported changes, every later
        assertion here would be about a broken comparator."""
        index = sessions(20)
        frame = pd.DataFrame({"SPY": [400.0 + i for i in range(20)]}, index=index)

        result = reconcile(frame, panel(frame.copy()))

        assert result.changed == {}
        assert result.clean

    def test_a_new_ticker_is_reported_rather_than_absorbed(self):
        index = sessions(10)
        before = pd.DataFrame({"SPY": [400.0] * 10}, index=index)
        after = pd.DataFrame({"SPY": [400.0] * 10, "QQQ": [300.0] * 10}, index=index)

        result = reconcile(before, panel(after))

        assert result.added_tickers == ["QQQ"]
        assert result.clean, "adding a ticker is not a rewrite of an existing one"

    def test_a_dropped_ticker_is_reported(self):
        index = sessions(10)
        before = pd.DataFrame({"SPY": [400.0] * 10, "QQQ": [300.0] * 10}, index=index)
        after = pd.DataFrame({"SPY": [400.0] * 10}, index=index)

        assert reconcile(before, panel(after)).removed_tickers == ["QQQ"]


class TestOnlyPricesBeforeTheActionAreExcused:
    """A split rewrites the past, not the future.

    Without this, `reconcile` would excuse any change at all for a ticker that
    has ever had a split — which is most of them — and the check would pass on
    exactly the corruption it exists to catch.
    """

    def test_a_change_after_the_last_action_stays_unexplained(self):
        index = sessions(20)
        before = pd.DataFrame({"NVDA": [120.0] * 20}, index=index)
        values = [120.0] * 10 + [125.0] * 10
        after = pd.DataFrame({"NVDA": values}, index=index)
        # The split is early; the changes are all after it.
        splits = {"NVDA": pd.Series([10.0], index=[index[2]])}

        result = reconcile(before, panel(after, splits=splits))

        assert not result.clean
        assert result.explained["NVDA"] == 0
        assert len(result.unexplained_changes["NVDA"]) == 10


class TestTheComparatorIsNeitherNoisyNorVacuous:
    """Both thresholds were wrong, in opposite directions, and both looked fine.

    A tolerance of 1e-6 sat below the vendor's own arithmetic noise: two
    fetches minutes apart moved 110 values by up to 1.2e-6. And the excuse
    rule pointed at the latest corporate action of any kind, so a
    dividend-paying ticker had all of its history excused — those same 110
    noise changes were reported as "explained by an action".

    Noisy and vacuous at once: it fired on identical data, then forgave it.
    """

    def test_vendor_arithmetic_noise_is_not_a_change(self):
        index = sessions(20)
        before = pd.DataFrame({"BND": [80.533485] * 20}, index=index)
        after = pd.DataFrame({"BND": [80.533585] * 20}, index=index)  # 1.2e-6

        result = reconcile(before, panel(after))

        assert result.changed == {}, "the comparator fires on its own round-trip"
        assert result.clean

    def test_a_real_move_is_still_a_change(self):
        """The other side of the same knob. Raising a tolerance until nothing
        fires is not the same as fixing it."""
        index = sessions(20)
        before = pd.DataFrame({"BND": [80.50] * 20}, index=index)
        after = pd.DataFrame({"BND": [80.60] * 20}, index=index)  # 1.2e-3

        assert not reconcile(before, panel(after)).clean

    def test_an_old_split_excuses_nothing(self):
        """A split already applied to the previous snapshot has already been
        accounted for there. Only one that arrived since can explain a
        rewrite."""
        index = sessions(20)
        before = pd.DataFrame({"NVDA": [120.0] * 20}, index=index)
        after = pd.DataFrame({"NVDA": [121.0] * 20}, index=index)
        splits = {"NVDA": pd.Series([10.0], index=[index[-1]])}

        result = reconcile(before, panel(after, splits=splits),
                           previous_splits={"NVDA": [index[-1].date()]})

        assert not result.clean, "a split known to both snapshots excused a rewrite"
        assert result.explained["NVDA"] == 0

    def test_a_dividend_never_excuses_a_rewrite_on_its_own(self):
        """Dividends are quarterly, so excusing on them forgives all history."""
        index = sessions(20)
        before = pd.DataFrame({"VTI": [200.0] * 20}, index=index)
        after = pd.DataFrame({"VTI": [201.0] * 20}, index=index)
        dividends = {"VTI": pd.Series([0.9], index=[index[-1]])}

        assert not reconcile(before, panel(after, dividends=dividends)).clean


class TestAShareCountCarriesAnEpoch:
    """A holding recorded before a split is recorded in shares that no longer
    exist, at a price no series reports.

    Somebody who bought 10 NVDA at $1,209 in May 2024 holds 100 today and
    their statement still says 10 at $1,209. Valuing that position against a
    split-adjusted series without converting gives 10 x 120.998 = $1,210 for a
    holding worth $12,100 — off by the split ratio, and entirely plausible.
    """

    NVDA = pd.Series([4.0, 10.0],
                     index=pd.to_datetime(["2021-07-20", "2024-06-10"]))

    def test_the_factor_compounds_across_every_later_split(self):
        from datetime import date as _date
        from src.market_data.ingest import split_factor

        # Before both splits: 4 x 10.
        assert split_factor(self.NVDA, _date(2020, 1, 2)) == 40.0
        # Between them: only the 10:1 is still ahead.
        assert split_factor(self.NVDA, _date(2024, 6, 6)) == 10.0
        # After both.
        assert split_factor(self.NVDA, _date(2025, 1, 2)) == 1.0

    def test_a_split_on_the_day_itself_does_not_apply(self):
        """The split-adjusted price on the split date is already post-split."""
        from datetime import date as _date
        from src.market_data.ingest import split_factor

        assert split_factor(self.NVDA, _date(2024, 6, 10)) == 1.0

    def test_the_as_traded_price_is_the_one_on_the_statement(self):
        from datetime import date as _date
        from src.market_data.ingest import as_traded_price

        # Yahoo's split-adjusted close for 2024-06-06 is 120.998.
        assert round(as_traded_price(120.998, self.NVDA, _date(2024, 6, 6)), 2) \
            == 1209.98

    def test_value_survives_the_split_in_both_directions(self):
        """The invariant. Quantity and price each change by the ratio; what a
        holding is worth does not."""
        from datetime import date as _date
        from src.market_data.ingest import current_shares

        bought = _date(2024, 6, 6)
        as_traded_shares, as_traded = 10.0, 1209.98
        today_shares = current_shares(as_traded_shares, self.NVDA, bought)

        assert today_shares == 100.0
        assert round(as_traded_shares * as_traded, 2) \
            == round(today_shares * 120.998, 2)

    def test_no_splits_means_no_conversion(self):
        from datetime import date as _date
        from src.market_data.ingest import current_shares, split_factor

        assert split_factor(None, _date(2020, 1, 1)) == 1.0
        assert current_shares(37.0, None, _date(2020, 1, 1)) == 37.0


class TestTheTwoSeriesAnswerDifferentQuestions:
    """`Close` values a holding; `Adj Close` measures a strategy.

    Using the total-return series to value a position credits reinvested
    dividends into the share price and then credits them again as cash. The
    error is small per dividend and compounds silently over a decade.
    """

    def test_the_fetch_carries_both(self):
        index = sessions(5)
        market = pd.DataFrame({"VTI": [100.0] * 5}, index=index)
        total = pd.DataFrame({"VTI": [101.0] * 5}, index=index)
        fetched = PanelFetch(prices=market, splits={}, dividends={},
                             fetched_at=pd.Timestamp("2026-08-04", tz="UTC")
                             .to_pydatetime(), total_return=total)

        assert fetched.prices["VTI"].iloc[0] == 100.0
        assert fetched.total_return["VTI"].iloc[0] == 101.0
        assert fetched.adjustment == "split_adjusted_close"
