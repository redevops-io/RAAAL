"""Employer concentration, and the sale solved to reach a declared cap.

The strongest claim here, and the one that proves the engine is *sizing* rather
than choosing a plausible number:

    the chosen quantity reaches the declared cap, while the next smaller
    permitted quantity does not.

A cap is declared, never recommended. Twenty percent is not safe, prudent or
optimal in this system; it is a constraint a user or methodology stated.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.mission.accounting import CashPolicy, InKindFlow
from src.mission.simulate import simulate
from src.runtime.concentration import (
    IMPLEMENTED,
    ConcentrationPolicy,
    EmployerStockInTargets,
    Feasibility,
    RoundingPolicy,
    assess,
    projected_concentration,
    reaches_cap,
    realized_concentration,
    refuse_employer_in_targets,
    solve,
)
from src.runtime.disposition import DispositionSchedule, UnsupportedPolicy
from src.runtime.disposition import instruction_for as sell_for

CAP = ConcentrationPolicy(target=0.20, cost_rate=0.001)
FRACTIONAL = ConcentrationPolicy(target=0.20, cost_rate=0.001,
                                 rounding=RoundingPolicy.FRACTIONAL_ALLOWED)


def portfolio(acme_shares=500.0, acme_price=50.0, vti_shares=200.0, cash=5_000.0,
              policy=CAP, prices=None):
    return assess(holdings={"ACME": acme_shares, "VTI": vti_shares},
                  prices=prices or {"ACME": acme_price, "VTI": 100.0},
                  cash=cash, employer_asset="ACME", policy=policy,
                  measured_at="2026-03-11")


class TestTheDenominatorIsExplicit:

    def test_it_includes_settled_holdings_and_cash(self):
        assessment = portfolio()
        assert assessment.portfolio_value == pytest.approx(50_000.0)
        assert assessment.employer_value == pytest.approx(25_000.0)
        assert assessment.concentration == pytest.approx(0.5)

    def test_the_exclusions_are_named_not_omitted(self):
        excluded = portfolio().excluded_components
        for component in ("unvested grants", "pending dispositions",
                          "pending allocation orders", "unreconciled fills"):
            assert component in excluded

    def test_the_scope_says_it_is_this_account_not_household_wealth(self):
        assert "household" in portfolio().scope_note

    def test_the_excess_is_measured_against_the_cap(self):
        # 25,000 held against a 20% cap on 50,000 = 10,000 permitted.
        assert portfolio().excess_value == pytest.approx(15_000.0)


class TestAlreadySatisfied:

    def test_below_the_cap_produces_no_sale(self):
        assessment = portfolio(acme_shares=100.0)      # 5,000 of 30,000
        plan = solve(assessment, price=50.0, held_shares=100.0, policy=CAP)
        assert plan.feasibility is Feasibility.ALREADY_SATISFIED
        assert plan.shares_to_sell == 0.0

    def test_zero_here_is_computed_not_unknown(self):
        """Distinct from an uncomputable sizing, which yields no quantity."""
        plan = solve(portfolio(acme_shares=100.0), price=50.0,
                     held_shares=100.0, policy=CAP)
        assert plan.minimum_continuous_quantity == 0.0
        assert plan.unresolved_inputs == ()

    def test_exactly_at_the_cap_produces_no_sale(self):
        # 10,000 ACME within a 50,000 portfolio is exactly 20%.
        assessment = assess(holdings={"ACME": 200.0, "VTI": 350.0},
                            prices={"ACME": 50.0, "VTI": 100.0}, cash=5_000.0,
                            employer_asset="ACME", policy=CAP,
                            measured_at="t")
        assert assessment.concentration == pytest.approx(0.20)
        assert solve(assessment, price=50.0, held_shares=200.0,
                     policy=CAP).feasibility is Feasibility.ALREADY_SATISFIED


class TestTheSolvedQuantityIsMinimal:
    """The mechanical proof that the engine sizes the sale."""

    def test_the_chosen_quantity_reaches_the_cap(self):
        plan = solve(portfolio(), price=50.0, held_shares=500.0, policy=CAP)
        assert plan.feasibility is Feasibility.SOLVED
        assert reaches_cap(plan)

    def test_the_next_smaller_permitted_quantity_does_not(self):
        """Minimality, not mere sufficiency. Without this the solver could
        return "sell everything" and still pass every other test."""
        assessment = portfolio()
        plan = solve(assessment, price=50.0, held_shares=500.0, policy=CAP)

        smaller = projected_concentration(
            plan.shares_to_sell - 1.0, employer_value=assessment.employer_value,
            portfolio_value=assessment.portfolio_value, price=50.0,
            cost_rate=CAP.cost_rate)
        assert smaller > CAP.target

    def test_the_fractional_solution_lands_exactly_on_the_cap(self):
        assessment = portfolio()
        plan = solve(assessment, price=50.0, held_shares=500.0,
                     policy=FRACTIONAL)
        assert plan.projected_post_sale_concentration == pytest.approx(
            0.20, abs=1e-9)

    def test_selling_everything_is_not_the_answer(self):
        plan = solve(portfolio(), price=50.0, held_shares=500.0, policy=CAP)
        assert plan.shares_to_sell < 500.0


class TestWholeShareRounding:

    def test_it_rounds_up_not_down(self):
        assessment = portfolio()
        exact = solve(assessment, price=50.0, held_shares=500.0,
                      policy=FRACTIONAL).shares_to_sell
        whole = solve(assessment, price=50.0, held_shares=500.0,
                      policy=CAP).shares_to_sell
        assert whole == pytest.approx(float(int(exact)) + 1)

    def test_rounding_down_would_stay_above_the_cap(self):
        """The one rounding direction that produces a false pass."""
        assessment = portfolio()
        exact = solve(assessment, price=50.0, held_shares=500.0,
                      policy=FRACTIONAL).shares_to_sell
        rounded_down = projected_concentration(
            float(int(exact)), employer_value=assessment.employer_value,
            portfolio_value=assessment.portfolio_value, price=50.0,
            cost_rate=CAP.cost_rate)
        assert rounded_down > CAP.target

    def test_both_quantities_are_reported(self):
        """Rounding is visible rather than absorbed."""
        plan = solve(portfolio(), price=50.0, held_shares=500.0, policy=CAP)
        assert plan.minimum_continuous_quantity < plan.shares_to_sell


class TestTransactionCosts:

    def test_costs_reduce_the_post_sale_portfolio(self):
        assessment = portfolio()
        free = projected_concentration(
            300.0, employer_value=assessment.employer_value,
            portfolio_value=assessment.portfolio_value, price=50.0,
            cost_rate=0.0)
        charged = projected_concentration(
            300.0, employer_value=assessment.employer_value,
            portfolio_value=assessment.portfolio_value, price=50.0,
            cost_rate=0.05)
        assert charged > free

    def test_the_naive_no_cost_quantity_fails_when_costs_are_material(self):
        """(E - cP)/price ignores that the cost shrinks the denominator."""
        expensive = ConcentrationPolicy(
            target=0.20, cost_rate=0.05,
            rounding=RoundingPolicy.FRACTIONAL_ALLOWED)
        assessment = portfolio(policy=expensive)

        naive = (assessment.employer_value
                 - expensive.target * assessment.portfolio_value) / 50.0
        missed = projected_concentration(
            naive, employer_value=assessment.employer_value,
            portfolio_value=assessment.portfolio_value, price=50.0,
            cost_rate=expensive.cost_rate)
        assert missed > expensive.target

        solved = solve(assessment, price=50.0, held_shares=500.0,
                       policy=expensive)
        assert solved.shares_to_sell > naive
        assert reaches_cap(solved)

    def test_the_estimated_cost_is_reported(self):
        plan = solve(portfolio(), price=50.0, held_shares=500.0, policy=CAP)
        assert plan.estimated_cost == pytest.approx(
            plan.estimated_gross_proceeds * CAP.cost_rate)


class TestEveryCapIsReachable:
    """There is no infeasibility branch, and this is why.

    Proceeds stay inside the portfolio — the value moves from the holding to
    cash and only the transaction cost leaves — so selling the whole position
    drives concentration to zero. If proceeds ever leave the account, these
    fail and the branch has to come back.
    """

    def test_selling_into_cash_leaves_the_denominator_almost_intact(self):
        after = projected_concentration(
            500.0, employer_value=25_000.0, portfolio_value=25_000.0,
            price=50.0, cost_rate=0.001)
        assert after == pytest.approx(0.0)

    def test_an_all_employer_portfolio_is_still_solvable(self):
        """The hardest case: nothing but employer stock, no other holdings,
        no cash."""
        assessment = portfolio(acme_shares=500.0, vti_shares=0.0, cash=0.0)
        assert assessment.concentration == pytest.approx(1.0)

        plan = solve(assessment, price=50.0, held_shares=500.0, policy=CAP)
        assert plan.feasibility is Feasibility.SOLVED
        assert plan.shares_to_sell <= 500.0
        assert reaches_cap(plan)

    @pytest.mark.parametrize("cap", [0.05, 0.10, 0.20, 0.50, 0.90])
    def test_any_cap_is_reachable_by_selling_into_cash(self, cap):
        policy = ConcentrationPolicy(target=cap, cost_rate=0.001)
        assessment = portfolio(policy=policy)
        plan = solve(assessment, price=50.0, held_shares=500.0, policy=policy)
        assert plan.feasibility in {Feasibility.SOLVED,
                                    Feasibility.ALREADY_SATISFIED}
        assert plan.shares_to_sell <= 500.0


class TestMissingPricesRefuseSizing:

    def test_an_unpriced_non_employer_holding_blocks_the_assessment(self):
        """Dropping it shrinks the denominator, inflates the measured
        concentration, and sizes the corrective sale too small."""
        assessment = portfolio(prices={"ACME": 50.0})
        assert not assessment.data_complete
        assert "VTI" in assessment.missing_prices

    def test_concentration_is_unknown_rather_than_estimated(self):
        assert portfolio(prices={"ACME": 50.0}).concentration is None

    def test_sizing_refuses(self):
        plan = solve(portfolio(prices={"ACME": 50.0}), price=50.0,
                     held_shares=500.0, policy=CAP)
        assert plan.feasibility is Feasibility.UNCOMPUTABLE
        assert "VTI" in plan.unresolved_inputs
        assert plan.shares_to_sell == 0.0

    def test_an_unpriced_employer_holding_also_refuses(self):
        plan = solve(portfolio(), price=float("nan"), held_shares=500.0,
                     policy=CAP)
        assert plan.feasibility is Feasibility.UNCOMPUTABLE


class TestSizingHappensAtExecutionNotVest:

    def test_a_moving_price_changes_the_quantity(self):
        """Both the employer price and the rest of the portfolio move between
        the vest and the first eligible session."""
        sessions = pd.bdate_range("2026-03-02", "2026-04-30")
        doubling = [50.0] * 7 + [100.0] * (len(sessions) - 7)
        prices = pd.DataFrame({"ACME": doubling, "VTI": 100.0}, index=sessions)

        schedule = DispositionSchedule([sell_for(
            vest_ref="g1", grant_ref="g1", asset="ACME", delivered_shares=0.0,
            policy="REDUCE_CONCENTRATION_BELOW_20",
            delivery_session=pd.Timestamp("2026-03-02"),
            blackouts=[("2026-03-02", "2026-03-10")], sizing_policy=CAP)])

        seed = [InKindFlow(date=pd.Timestamp("2026-03-02"), asset="ACME",
                           quantity=500.0, valuation_price=50.0,
                           external_value=25_000.0, source_ref="vest:g1"),
                InKindFlow(date=pd.Timestamp("2026-03-02"), asset="VTI",
                           quantity=200.0, valuation_price=100.0,
                           external_value=20_000.0, source_ref="seed")]
        simulate(prices, flows=[], program=schedule.program(), in_kind=seed,
                 cash_policy=CashPolicy.idle(),
                 modelling_scope={"excludes": []})

        sized = schedule.instructions[0]
        assert sized.sized_at == pd.Timestamp("2026-03-11")
        # At the vest price of $50 the answer would have been 301 shares.
        assert sized.sizing_plan.shares_to_sell != 301

    def test_the_quantity_is_unknown_at_creation(self):
        instruction = sell_for(
            vest_ref="g1", grant_ref="g1", asset="ACME", delivered_shares=78.0,
            policy="REDUCE_CONCENTRATION_BELOW_20",
            delivery_session=pd.Timestamp("2026-03-02"), sizing_policy=CAP)
        assert instruction.quantity == 0.0
        assert instruction.sizes_from_portfolio

    def test_without_a_concentration_policy_it_is_still_refused(self):
        """The old behaviour must survive: no policy, no approximation."""
        with pytest.raises(UnsupportedPolicy, match="concentration"):
            sell_for(vest_ref="g1", grant_ref="g1", asset="ACME",
                     delivered_shares=78.0,
                     policy="REDUCE_CONCENTRATION_BELOW_20",
                     delivery_session=pd.Timestamp("2026-03-02"))


class TestProjectedAndRealizedStayDistinct:

    def test_realized_is_computed_from_actual_holdings(self):
        realized = realized_concentration(
            holdings={"ACME": 199.0, "VTI": 200.0},
            prices={"ACME": 50.0, "VTI": 100.0}, cash=15_000.0,
            employer_asset="ACME")
        assert realized == pytest.approx(9_950.0 / 44_950.0)

    def test_a_partial_fill_cannot_report_the_cap_met(self):
        """A plan reporting its target met on the strength of an order it
        placed is describing an intention as an outcome."""
        assessment = portfolio()
        plan = solve(assessment, price=50.0, held_shares=500.0, policy=CAP)
        assert reaches_cap(plan)

        # Only half the solved quantity actually filled.
        half = plan.shares_to_sell / 2
        realized = realized_concentration(
            holdings={"ACME": 500.0 - half, "VTI": 200.0},
            prices={"ACME": 50.0, "VTI": 100.0},
            cash=5_000.0 + half * 50.0, employer_asset="ACME")
        assert realized > CAP.target

    def test_realized_is_unknown_when_a_holding_is_unpriced(self):
        assert realized_concentration(
            holdings={"ACME": 100.0, "VTI": 200.0}, prices={"ACME": 50.0},
            cash=0.0, employer_asset="ACME") is None


class TestEmployerStockInTheTargetAllocation:

    def test_buying_the_employer_back_is_refused(self):
        """The solver sizes the sale assuming the proceeds leave the position."""
        with pytest.raises(EmployerStockInTargets, match="ACME"):
            refuse_employer_in_targets({"ACME": 0.2, "VTI": 0.8}, "ACME")

    def test_an_allocation_without_the_employer_passes(self):
        refuse_employer_in_targets({"VTI": 0.6, "BND": 0.4}, "ACME")


class TestTheComparabilityFieldsAreExposed:
    """A state-dependent quantity means identical vest flows are no longer
    enough for a strategy-effect claim."""

    @pytest.mark.parametrize("field", [
        "target", "included", "excluded", "rounding", "cost_rate",
        "execution_lag", "blackout_ref"])
    def test_the_policy_states_it(self, field):
        assert field in CAP.to_json()


class TestEveryDeclaredMechanismExists:

    def test_the_registry_names_only_real_callables(self):
        import src.runtime.concentration as module

        for name in IMPLEMENTED:
            assert callable(getattr(module, name, None)), name
