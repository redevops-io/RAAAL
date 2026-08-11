"""In-kind external flow: shares and outside value arriving in one event.

The invariant this file defends:

    A vest adds outside economic value and shares to the account at the same
    event. It is not a purchase funded from existing cash.

Treated as "cash arrives, then buy", a vest acquires a session of execution lag,
a transaction cost and a fill price different from the vest price — and it
credits the plan with a trading decision nobody made. Every figure downstream is
then slightly wrong in a way that looks like performance.

The engine understands `InKindFlow` and nothing about RSUs. Inherited securities,
stock gifts, transfers in kind and employer stock contributions are the same
event to an account; only their semantics differ, and those live in
`RSUVestingRuntime`.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from tests.vest_fixtures import resolved_for

from src.mission.accounting import CashFlow, CashPolicy, InKindFlow
from src.mission.benchmark import buy_and_hold
from src.mission.simulate import simulate
from src.runtime.rsu import (
    BenchmarkFlowMode,
    VestEvent,
    WithholdingMethod,
    benchmark_flows_for,
    conserved,
    in_kind_flow_for,
    vest_accounting,
)

SCOPE = {"excludes": []}


@pytest.fixture
def sessions():
    return pd.bdate_range("2026-03-02", "2026-04-30")


@pytest.fixture
def prices(sessions):
    return pd.DataFrame({"ACME": 50.0, "VTI": 100.0}, index=sessions)


def vest(**overrides) -> VestEvent:
    fields = dict(
        grant_id="g1", employer_ticker="ACME", vest_date="2026-03-02",
        gross_shares=100.0, vest_price_source="prices@2026-03-02",
        withholding_method=WithholdingMethod.SHARE_WITHHOLDING,
        withholding_rate=0.22, market_data_ref="md@1",
        corporate_action_ref="ca/none@1")
    fields.update(overrides)
    return VestEvent(**fields)


def run(prices, *, in_kind=(), flows=(), tickers=()):
    return simulate(prices, flows=list(flows), program=buy_and_hold(list(tickers)),
                    in_kind=list(in_kind), cash_policy=CashPolicy.idle(),
                    modelling_scope=dict(SCOPE))


class TestTheInKindBalanceIdentity:
    """Immediately after a vest, before any disposition."""

    def test_shares_are_credited(self, prices):
        arrival, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        result = run(prices, in_kind=[arrival])
        assert float(result.path.holdings["ACME"].iloc[-1]) == pytest.approx(78.0)

    def test_cash_is_unchanged(self, prices):
        """No cash is debited, and none is credited either. The delivery is not
        a cash event."""
        arrival, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        assert float(run(prices, in_kind=[arrival]).path.cash.iloc[-1]) == \
            pytest.approx(0.0)

    def test_the_external_flow_is_the_delivered_value(self, prices):
        """Delivered, not gross. Withheld shares never enter the account, so
        crediting their value would give the portfolio money it does not hold."""
        arrival, accounting = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        result = run(prices, in_kind=[arrival])
        assert float(result.path.flows.sum()) == pytest.approx(3_900.0)
        assert accounting["gross_vest_value"] == pytest.approx(5_000.0)

    def test_an_in_kind_flow_never_becomes_spendable_cash(self, prices):
        """Held in one series this worked only because of statement order in
        the loop. Two series make funding a purchase from a vest impossible."""
        arrival, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        result = run(prices, in_kind=[arrival], tickers=["VTI"])
        assert float(result.path.holdings.get("VTI", pd.Series([0.0])).iloc[-1]) \
            == pytest.approx(0.0)


class TestNoArtificialTrade:

    def test_no_purchase_fill_is_generated(self, prices):
        arrival, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        assert len(run(prices, in_kind=[arrival]).path.fills) == 0

    def test_no_transaction_cost_is_charged(self, prices):
        """78 shares at $50 is exactly $3,900 of value. A cost would make the
        holding worth less than the flow that delivered it."""
        arrival, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        result = run(prices, in_kind=[arrival])
        assert result.path.terminal_value == pytest.approx(3_900.0)

    def test_no_execution_lag_is_applied_to_delivery(self, prices, sessions):
        """The shares are owned on the vest session, not a session later."""
        arrival, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        result = run(prices, in_kind=[arrival])
        assert float(result.path.holdings["ACME"].iloc[0]) == pytest.approx(78.0)


class TestTheValuationIsPinnedNotRediscovered:

    def test_a_vest_dated_on_a_non_session_keeps_its_vest_price(self, sessions):
        """The vest date is a Sunday; the arrival lands on Monday at a
        different price. Re-pricing at the landing session would make the
        external flow disagree with the withholding computed at the vest
        price, and conservation would fail by an invisible amount.
        """
        moving = pd.DataFrame(
            {"ACME": [50.0 + i for i in range(len(sessions))]}, index=sessions)
        arrival, accounting = in_kind_flow_for(vest(vest_date="2026-03-08"), vest_price=50.0, resolved=resolved_for(vest(vest_date="2026-03-08")))          # a Sunday

        result = run(moving, in_kind=[arrival])
        assert float(result.path.flows.sum()) == pytest.approx(
            accounting["external_flow_value"])
        assert float(result.path.flows.sum()) == pytest.approx(3_900.0)


class TestTWRAndMWRSeparate:

    def rising(self, sessions):
        return pd.DataFrame(
            {"ACME": [50.0 * (1.001 ** i) for i in range(len(sessions))]},
            index=sessions)

    def test_twr_is_unchanged_by_when_the_vest_arrived(self, sessions):
        """Time-weighted return measures the strategy, so contribution timing
        must not move it."""
        prices = self.rising(sessions)
        early, _ = in_kind_flow_for(vest(vest_date="2026-03-02"), vest_price=50.0, resolved=resolved_for(vest(vest_date="2026-03-02")))
        late, _ = in_kind_flow_for(vest(vest_date="2026-04-01"), vest_price=50.0, resolved=resolved_for(vest(vest_date="2026-04-01")))

        first = run(prices, in_kind=[early]).time_weighted
        second = run(prices, in_kind=[late]).time_weighted
        overlap = first.index.intersection(second.index)
        assert float((1 + first[overlap]).prod()) == pytest.approx(
            float((1 + second[overlap]).prod()), rel=1e-9)

    def test_the_money_weighted_return_does_depend_on_when_it_arrived(
            self, sessions):
        """Money-weighted return answers what happened to the person's wealth,
        so timing must move it.

        The terminal value is deliberately *not* the assertion: 78 shares held
        to the end are worth the same whenever they arrived. What differs is
        how long that value was exposed, which is exactly what MWR measures and
        TWR removes."""
        prices = self.rising(sessions)
        early, _ = in_kind_flow_for(vest(vest_date="2026-03-02"), vest_price=50.0, resolved=resolved_for(vest(vest_date="2026-03-02")))
        late, _ = in_kind_flow_for(vest(vest_date="2026-04-01"), vest_price=50.0, resolved=resolved_for(vest(vest_date="2026-04-01")))

        first = run(prices, in_kind=[early])
        second = run(prices, in_kind=[late])
        assert first.path.terminal_value == pytest.approx(
            second.path.terminal_value)
        assert first.money_weighted.rate != pytest.approx(second.money_weighted.rate)


class TestBenchmarksReceiveTheSameFlow:

    def test_a_value_matched_benchmark_gets_the_same_dated_value(self, prices):
        arrival, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        matched = benchmark_flows_for([arrival],
                                      mode=BenchmarkFlowMode.VALUE_MATCHED)

        strategy = run(prices, in_kind=[arrival])
        benchmark = run(prices, flows=matched, tickers=["VTI"])
        assert list(strategy.path.flows) == pytest.approx(
            list(benchmark.path.flows))

    def test_it_matches_by_date_not_merely_by_total(self, prices):
        """An annual total that agrees while the dates differ is a different
        investment, and money-weighted return will say so."""
        arrival, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        matched = benchmark_flows_for([arrival],
                                      mode=BenchmarkFlowMode.VALUE_MATCHED)
        assert [f.date for f in matched] == [arrival.date]

    def test_an_in_kind_hold_benchmark_receives_the_shares_themselves(self,
                                                                     prices):
        arrival, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        held = benchmark_flows_for([arrival], mode=BenchmarkFlowMode.IN_KIND_HOLD)
        benchmark = run(prices, in_kind=held)
        assert float(benchmark.path.holdings["ACME"].iloc[-1]) == \
            pytest.approx(78.0)

    def test_the_two_modes_are_not_interchangeable(self, prices):
        """One compares allocation strategies; the other compares dispositions
        of the same asset. Substituting one answers a question nobody asked."""
        arrival, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        matched = benchmark_flows_for([arrival],
                                      mode=BenchmarkFlowMode.VALUE_MATCHED)
        held = benchmark_flows_for([arrival], mode=BenchmarkFlowMode.IN_KIND_HOLD)

        assert isinstance(matched[0], CashFlow)
        assert isinstance(held[0], InKindFlow)


class TestAMissingPriceIsANamedGap:

    def test_no_holding_mutation_when_the_asset_has_no_price(self, prices):
        unpriced = InKindFlow(date=pd.Timestamp("2026-03-02"), asset="NOPRICE",
                              quantity=10.0, valuation_price=float("nan"),
                              external_value=float("nan"), source_ref="vest:x")
        result = run(prices, in_kind=[unpriced])
        assert "NOPRICE" not in result.path.holdings.columns \
            or float(result.path.holdings["NOPRICE"].iloc[-1]) == 0.0

    def test_no_flow_is_recorded(self, prices):
        unpriced = InKindFlow(date=pd.Timestamp("2026-03-02"), asset="NOPRICE",
                              quantity=10.0, valuation_price=float("nan"),
                              external_value=float("nan"), source_ref="vest:x")
        assert float(run(prices, in_kind=[unpriced]).path.flows.sum()) == 0.0

    def test_the_gap_is_named_on_the_result(self, prices):
        """Skipped silently, a portfolio is simply missing shares the user
        believes it holds and nothing says why."""
        unpriced = InKindFlow(date=pd.Timestamp("2026-03-02"), asset="NOPRICE",
                              quantity=10.0, valuation_price=float("nan"),
                              external_value=float("nan"), source_ref="vest:x")
        scope = run(prices, in_kind=[unpriced]).modelling_scope
        assert scope["unpriced_in_kind_arrivals"]
        assert scope["unpriced_in_kind_arrivals"][0]["asset"] == "NOPRICE"


class TestConservation:

    def test_the_three_values_account_for_the_gross(self):
        accounting = vest_accounting(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        assert conserved(accounting)
        assert accounting["gross_vest_value"] == pytest.approx(
            accounting["withheld_value"] + accounting["external_flow_value"]
            + accounting["cash_remainder"])

    def test_it_holds_across_the_threshold_split(self):
        big = vest(gross_shares=30_000.0)
        accounting = vest_accounting(big, vest_price=50.0,
                                     resolved=resolved_for(big),
                                     cumulative_supplemental=800_000.0)
        assert conserved(accounting)

    def test_any_remainder_is_explicit_rather_than_absorbed(self):
        """A remainder folded into delivered shares is rounding in the
        account's favour that nobody chose."""
        accounting = vest_accounting(vest(gross_shares=101.0), vest_price=37.13, resolved=resolved_for(vest(gross_shares=101.0)))
        assert "cash_remainder" in accounting
        assert conserved(accounting)

    def test_the_basis_says_what_the_figure_is(self):
        """Account value after share withholding — not total compensation
        economics, and not final tax liability."""
        basis = vest_accounting(vest(), vest_price=50.0, resolved=resolved_for(vest()))["basis"]
        assert "after share withholding" in basis
        assert "not final tax liability" in basis


class TestTheEngineStaysGeneric:

    def test_the_engine_names_no_rsu_concept(self):
        """RSU semantics belong to the runtime. An engine that knew about
        vesting could not receive an inheritance or a gift."""
        import inspect

        from src.mission import simulate as engine

        # Identifiers and string literals, not comments. The claim is that the
        # engine has no RSU *logic*; a comment may name a vest as the example
        # that motivates the design without the engine knowing what one is.
        # (Checked as whole words, too — "uninvested" contains "vest".)
        import ast

        tree = ast.parse(inspect.getsource(engine))
        vocabulary = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                vocabulary.add(node.id.lower())
            elif isinstance(node, ast.arg):
                vocabulary.add(node.arg.lower())
            elif isinstance(node, ast.Attribute):
                vocabulary.add(node.attr.lower())
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                vocabulary.add(node.name.lower())
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                vocabulary.update(node.value.lower().split())

        for term in ("vest", "vesting", "rsu", "withholding", "employer",
                     "grant_id"):
            offenders = [word for word in vocabulary
                         if re.search(rf"\b{term}\b", word)]
            assert not offenders, (term, offenders)

    def test_an_inherited_security_uses_the_same_event(self, prices):
        inherited = InKindFlow(date=pd.Timestamp("2026-03-02"), asset="ACME",
                               quantity=40.0, valuation_price=50.0,
                               external_value=2_000.0,
                               source_ref="inheritance:estate-1")
        result = run(prices, in_kind=[inherited])
        assert float(result.path.holdings["ACME"].iloc[-1]) == pytest.approx(40.0)
        assert float(result.path.flows.sum()) == pytest.approx(2_000.0)
        assert len(result.path.fills) == 0
