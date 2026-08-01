"""The flow-aware engine, and the wrong answer it exists to prevent.

Every figure this platform publishes today is a time-weighted return, because
the engine is weight-based and a weight matrix presupposes no external cash
flows. A Mission has cash flows by construction, and TWR removes the effect of
contribution timing — the exact thing "what would have happened if I had
invested this way?" is asking about.

The first test class is the important one: it produces the scenario where the two
bases disagree so sharply that reporting one alone answers a different question
than the user asked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mission import (
    CashFlow,
    CashPolicy,
    CashPolicyError,
    Order,
    buy_and_hold,
    compare,
    comparison_payload,
    hold_cash,
    money_weighted_return,
    simulate,
    time_weighted_returns,
)


def v_shaped(sessions: int = 505) -> pd.DataFrame:
    """A market that halves, then recovers exactly to where it started.

    The canonical case: a buy-and-hold investor ends flat, and anyone who kept
    contributing through the trough ends well ahead — because most of their money
    bought cheap shares. TWR reports the first; MWR reports the second. Both are
    correct and they answer different questions.
    """
    half = sessions // 2
    down = np.linspace(100.0, 50.0, half)
    up = np.linspace(50.0, 100.0, sessions - half)
    index = pd.bdate_range("2020-01-01", periods=sessions)
    return pd.DataFrame({"VTI": np.concatenate([down, up])}, index=index)


def monthly(prices: pd.DataFrame, amount: float = 2000.0) -> list:
    """A paycheque on the first session of each month."""
    firsts = prices.index.to_series().groupby(
        [prices.index.year, prices.index.month]
    ).min()
    return [CashFlow(date=d, amount=amount, label="paycheque") for d in firsts]


class TestTheTwoReturnsAnswerDifferentQuestions:
    """The landmine. If these ever converge, the test data stopped being a V."""

    def test_dca_through_a_crash_shows_flat_twr_and_positive_mwr(self):
        prices = v_shaped()
        result = simulate(
            prices, flows=monthly(prices), program=buy_and_hold(["VTI"]),
            cash_policy=CashPolicy.idle(), cost_bps=0.0,
        )

        twr = result.time_weighted_annualized
        mwr = result.money_weighted

        assert abs(twr) < 0.02, (
            "the market ended where it began; a time-weighted return should say so"
        )
        assert mwr > 0.10, (
            "money that bought at the bottom earned a real return, and that is "
            "what the user is asking about"
        )
        assert result.gain > 0

    def test_the_result_says_which_question_each_number_answers(self):
        prices = v_shaped()
        result = simulate(prices, flows=monthly(prices),
                          program=buy_and_hold(["VTI"]),
                          cash_policy=CashPolicy.idle())
        note = result.to_json()["return_basis_note"]

        assert "is this a good strategy" in note
        assert "how did I do" in note

    def test_lump_sum_and_dca_share_a_twr_and_differ_in_mwr(self):
        """Same strategy, same market, different schedule.

        TWR is a property of the strategy and must not move; MWR is a property of
        the investor's schedule and must. A platform reporting only TWR would
        tell both users the same thing.
        """
        prices = v_shaped()
        program, policy = buy_and_hold(["VTI"]), CashPolicy.idle()

        lump = simulate(prices, flows=[CashFlow(prices.index[0], 24_000.0)],
                        program=program, cash_policy=policy, cost_bps=0.0)
        dca = simulate(prices, flows=monthly(prices, 1000.0),
                       program=program, cash_policy=policy, cost_bps=0.0)

        assert abs(lump.time_weighted_annualized
                   - dca.time_weighted_annualized) < 0.03
        assert dca.money_weighted > lump.money_weighted + 0.05

    def test_an_irr_is_undefined_rather_than_zero_without_contributions(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        assert money_weighted_return(pd.Series(0.0, index=idx), 500.0) is None

    def test_a_single_doubling_contribution_recovers_its_rate(self):
        idx = pd.bdate_range("2020-01-01", periods=253)
        flows = pd.Series(0.0, index=idx)
        flows.iloc[0] = 100.0

        assert money_weighted_return(flows, 200.0) == pytest.approx(1.0, abs=1e-4)

    def test_elapsed_time_is_measured_in_sessions_not_flow_count(self):
        """Two contributions months apart are not two days apart."""
        idx = pd.bdate_range("2020-01-01", periods=253)
        far = pd.Series(0.0, index=idx)
        far.iloc[0], far.iloc[251] = 100.0, 100.0

        near = pd.Series(0.0, index=idx)
        near.iloc[0], near.iloc[1] = 100.0, 100.0

        assert money_weighted_return(far, 250.0) != pytest.approx(
            money_weighted_return(near, 250.0), abs=1e-3
        )


class TestTimeWeightedReturnRemovesFlowsCorrectly:
    def test_a_contribution_alone_is_not_a_return(self):
        """The single most likely way to overstate performance."""
        idx = pd.bdate_range("2020-01-01", periods=3)
        value = pd.Series([1000.0, 3000.0, 3000.0], index=idx)
        flows = pd.Series([1000.0, 2000.0, 0.0], index=idx)

        assert time_weighted_returns(value, flows).abs().max() < 1e-12

    def test_growth_on_top_of_a_contribution_is_a_return(self):
        idx = pd.bdate_range("2020-01-01", periods=2)
        value = pd.Series([1000.0, 3100.0], index=idx)
        flows = pd.Series([1000.0, 2000.0], index=idx)

        assert time_weighted_returns(value, flows).iloc[0] == pytest.approx(0.10)


class TestTheEngineKeepsTheDisciplinesItInherited:
    def test_cash_policy_must_be_declared(self):
        """Cash earning nothing is an answer; not deciding is not."""
        prices = v_shaped(20)
        with pytest.raises(CashPolicyError, match="by declaration"):
            simulate(prices, flows=[], program=hold_cash(), cash_policy=None)

    def test_idle_cash_is_a_declared_zero_not_a_missing_value(self):
        policy = CashPolicy.idle()
        assert policy.annual_rate == 0.0
        assert policy.detail

    def test_orders_cannot_fill_on_the_session_that_formed_them(self):
        prices = v_shaped(10)
        seen = []

        def program(session, visible, holdings, cash):
            seen.append(session)
            return [Order(session, "VTI", cash)] if cash > 0 else []

        result = simulate(prices, flows=[CashFlow(prices.index[0], 1000.0)],
                          program=program, cash_policy=CashPolicy.idle())
        first_fill = result.path.fills[0]

        assert first_fill.date > seen[0], "an order filled at a price it could not see"

    def test_costs_are_charged_on_every_fill(self):
        prices = v_shaped(30)
        flows = [CashFlow(prices.index[0], 10_000.0)]

        free = simulate(prices, flows=flows, program=buy_and_hold(["VTI"]),
                        cash_policy=CashPolicy.idle(), cost_bps=0.0)
        charged = simulate(prices, flows=flows, program=buy_and_hold(["VTI"]),
                           cash_policy=CashPolicy.idle(), cost_bps=50.0)

        assert charged.final_value < free.final_value

    def test_a_program_cannot_see_beyond_the_current_session(self):
        """Lookahead is prevented by construction, not by review."""
        prices = v_shaped(40)
        widest = []

        def program(session, visible, holdings, cash):
            widest.append(visible.index.max())
            return ()

        simulate(prices, flows=[], program=program, cash_policy=CashPolicy.idle())
        assert all(seen <= session
                   for seen, session in zip(widest, prices.index))

    def test_unfillable_orders_are_reported_not_dropped(self):
        prices = v_shaped(10)

        def program(session, visible, holdings, cash):
            return [Order(session, "VTI", 1_000_000.0, reason="more than we have")]

        result = simulate(prices, flows=[], program=program,
                          cash_policy=CashPolicy.idle())

        assert result.path.unfilled, (
            "an order that vanished silently is the gap between what the Mission "
            "declared and what it did"
        )

    def test_a_contribution_dated_to_a_weekend_lands_on_the_next_session(self):
        prices = v_shaped(20)
        saturday = pd.Timestamp("2020-01-04")
        result = simulate(prices, flows=[CashFlow(saturday, 1000.0)],
                          program=hold_cash(), cash_policy=CashPolicy.idle())

        landed = result.path.flows[result.path.flows > 0]
        assert len(landed) == 1
        assert landed.index[0] in prices.index
        assert landed.index[0] > saturday


class TestBenchmarksReceiveTheSameMoney:
    def test_every_benchmark_gets_identical_flows(self):
        prices = v_shaped()
        prices["SPY"] = prices["VTI"] * 4.0
        flows = monthly(prices)

        results = compare(
            prices, flows=flows, cash_policy=CashPolicy.idle(),
            benchmarks=[
                {"name": "DCA VTI", "tickers": ["VTI"], "program": buy_and_hold(["VTI"])},
                {"name": "DCA SPY", "tickers": ["SPY"], "program": buy_and_hold(["SPY"])},
                {"name": "Cash", "tickers": [], "program": hold_cash()},
            ],
        )

        contributed = {r.result.path.contributed for r in results}
        assert len(contributed) == 1, "benchmarks were funded differently"

    def test_holding_cash_is_offered_as_a_benchmark(self):
        """The comparison nobody runs and everybody needs."""
        prices = v_shaped(60)
        flows = [CashFlow(prices.index[0], 5000.0)]
        [cash_result] = compare(
            prices, flows=flows, cash_policy=CashPolicy.idle(),
            benchmarks=[{"name": "Cash", "tickers": [], "program": hold_cash()}],
        )

        assert cash_result.result.final_value == pytest.approx(5000.0)
        assert cash_result.result.money_weighted == pytest.approx(0.0, abs=1e-6)

    def test_an_unfundable_benchmark_is_reported_not_dropped(self):
        prices = v_shaped(30)
        [result] = compare(
            prices, flows=[CashFlow(prices.index[0], 1000.0)],
            cash_policy=CashPolicy.idle(),
            benchmarks=[{"name": "Gold", "tickers": ["GLD"],
                         "program": buy_and_hold(["GLD"])}],
        )

        assert not result.comparable
        assert result.mismatch.field == "price_coverage"
        assert "same contributions on the same days" in result.mismatch.why

    def test_the_set_is_returned_in_declaration_order(self):
        prices = v_shaped(60)
        prices["SPY"] = prices["VTI"] * 2.0
        declared = ["Cash", "DCA SPY", "DCA VTI"]

        results = compare(
            prices, flows=[CashFlow(prices.index[0], 1000.0)],
            cash_policy=CashPolicy.idle(),
            benchmarks=[
                {"name": "Cash", "tickers": [], "program": hold_cash()},
                {"name": "DCA SPY", "tickers": ["SPY"], "program": buy_and_hold(["SPY"])},
                {"name": "DCA VTI", "tickers": ["VTI"], "program": buy_and_hold(["VTI"])},
            ],
        )
        assert [r.name for r in results] == declared

    def test_the_payload_derives_whether_it_is_a_recommendation(self):
        """It used to declare it. A flag that cannot be wrong is not evidence."""
        prices = v_shaped(60)
        flows = [CashFlow(prices.index[0], 1000.0)]
        mission = simulate(prices, flows=flows, program=buy_and_hold(["VTI"]),
                           cash_policy=CashPolicy.idle())
        payload = comparison_payload(
            mission,
            compare(prices, flows=flows, cash_policy=CashPolicy.idle(),
                    benchmarks=[{"name": "Cash", "tickers": [],
                                 "program": hold_cash()}]),
            rendered_text="Here is what the plan and one benchmark did.",
        )

        assert payload["is_recommendation"] is False
        assert payload["recommendation_assessment"]["checks"]
        assert "ranking it is the reader's to do" in payload["note"]

    def test_incomparable_benchmarks_are_counted_in_the_payload(self):
        """A set that quietly excluded what did not fit is a curated argument."""
        prices = v_shaped(30)
        flows = [CashFlow(prices.index[0], 1000.0)]
        mission = simulate(prices, flows=flows, program=buy_and_hold(["VTI"]),
                           cash_policy=CashPolicy.idle())
        payload = comparison_payload(
            mission,
            compare(prices, flows=flows, cash_policy=CashPolicy.idle(),
                    benchmarks=[{"name": "Gold", "tickers": ["GLD"],
                                 "program": buy_and_hold(["GLD"])}]),
        )

        assert payload["incomparable_count"] == 1
