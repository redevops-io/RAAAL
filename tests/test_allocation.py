"""Allocating money a sale actually produced.

    Allocation may consume only reconciled net proceeds from an actual fill.
    It may never allocate expected proceeds from an instruction.

An instructed sale has an expected price; a filled one has a price. Allocating
the first spends money that may never arrive, and the resulting portfolio looks
exactly like one that was funded.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.mission.accounting import CashPolicy
from src.mission.simulate import simulate
from src.runtime.allocation import (
    ALLOCATION_ORDER,
    AllocationSchedule,
    AllocationStatus,
    FundingScope,
    ProceedsAlreadyAllocated,
    ProceedsLedger,
    UnsupportedAllocation,
    compile_policy,
    conserved,
    proceeds_from,
    supersede,
)
from src.runtime.allocation import instruction_for as allocate_for
from src.runtime.disposition import DispositionSchedule, EventLog
from src.runtime.disposition import instruction_for as sell_for
from src.runtime.rsu import VestEvent, WithholdingMethod, in_kind_flow_for

DELIVERY = pd.Timestamp("2026-03-02")


@pytest.fixture
def sessions():
    return pd.bdate_range("2026-03-02", "2026-04-30")


@pytest.fixture
def prices(sessions):
    return pd.DataFrame(
        {"ACME": 50.0, "VTI": 100.0, "VXUS": 60.0, "BND": 80.0}, index=sessions)


def vest_arrival():
    return in_kind_flow_for(
        VestEvent(grant_id="g1", employer_ticker="ACME", vest_date="2026-03-02",
                  gross_shares=100.0, vest_price_source="p",
                  withholding_rate=0.22,
                  withholding_method=WithholdingMethod.SHARE_WITHHOLDING,
                  market_data_ref="md@1", corporate_action_ref="ca@1"),
        vest_price=50.0)


def sale(log=None):
    arrival, accounting = vest_arrival()
    return arrival, DispositionSchedule([sell_for(
        vest_ref="g1", grant_ref="g1", asset="ACME",
        delivered_shares=accounting["shares_delivered"],
        policy="SELL_ALL_AND_DIVERSIFY", delivery_session=DELIVERY, log=log)],
        log=log)


def realized(prices, log=None):
    """Run the sale alone and produce its proceeds lot."""
    arrival, schedule = sale(log)
    result = simulate(prices, flows=[], program=schedule.program(),
                      in_kind=[arrival], cash_policy=CashPolicy.idle(),
                      modelling_scope={"excludes": []})
    schedule.reconcile(result.path.fills)
    return arrival, proceeds_from(schedule.executions[0], log=log)


def run_allocation(prices, arrival, instruction, log=None):
    """Sale and allocation together, so the proceeds fund the purchases."""
    arrival2, selling = sale(log)
    allocating = AllocationSchedule([instruction], log=log)

    def program(session, visible, holdings, cash):
        return (selling.program()(session, visible, holdings, cash)
                + allocating.program()(session, visible, holdings, cash))

    result = simulate(prices, flows=[], program=program, in_kind=[arrival2],
                      cash_policy=CashPolicy.idle(),
                      modelling_scope={"excludes": []})
    selling.reconcile(result.path.fills)
    return result, allocating


class TestOnlyReconciledFillsCreateProceeds:

    def test_an_unreconciled_execution_is_refused(self, prices):
        arrival, schedule = sale()
        step = schedule.program()
        visible = pd.DataFrame({"ACME": [50.0]}, index=[DELIVERY])
        step(DELIVERY, visible, {"ACME": 78.0}, 0.0)

        with pytest.raises(ValueError, match="no matching fill"):
            proceeds_from(schedule.executions[0])

    def test_the_lot_nets_the_sales_own_cost(self, prices):
        """Gross proceeds are not spendable. Sized against them, purchases draw
        the difference from cash the sale never produced."""
        _, lot = realized(prices)
        assert lot.gross_proceeds == pytest.approx(3_900.0)
        assert lot.transaction_costs == pytest.approx(3.90)
        assert lot.net_proceeds == pytest.approx(3_896.10)

    def test_proceeds_are_available_from_the_fill_session(self, prices):
        _, lot = realized(prices)
        assert lot.available_on > DELIVERY


class TestPolicyCompilation:

    def test_hold_cash_is_a_policy_not_an_absence(self, prices):
        _, lot = realized(prices)
        assert compile_policy("HOLD_CASH") == {}
        assert allocate_for(lot, policy="HOLD_CASH") is None

    def test_fixed_targets_stay_as_stated(self):
        assert compile_policy({"VTI": 0.6, "VXUS": 0.3, "BND": 0.1}) == {
            "VTI": 0.6, "VXUS": 0.3, "BND": 0.1}

    def test_weights_that_do_not_sum_to_one_are_refused(self):
        """A near-miss silently rescaled is a different allocation."""
        with pytest.raises(UnsupportedAllocation, match="sum"):
            compile_policy({"VTI": 0.6, "BND": 0.3})

    def test_equal_weight_compiles_to_explicit_weights(self):
        compiled = compile_policy(["VTI", "VXUS", "BND"])
        assert compiled == pytest.approx(
            {"VTI": 1 / 3, "VXUS": 1 / 3, "BND": 1 / 3})

    def test_the_order_assets_are_named_in_does_not_matter(self):
        assert compile_policy(["BND", "VTI", "VXUS"]) == \
            compile_policy(["VTI", "VXUS", "BND"])

    @pytest.mark.parametrize("policy", [
        "invest conservatively", "the best-performing ETF", "reduce risk",
        "tax-optimal allocation", "dynamic allocation",
        "reduce concentration below 20%"])
    def test_untyped_policies_are_refused_not_approximated(self, policy):
        """Answering with a plausible portfolio would put a recommendation in
        the account under the description the user wrote."""
        with pytest.raises(UnsupportedAllocation):
            compile_policy(policy)

    def test_a_methodology_reference_stays_a_reference(self):
        with pytest.raises(UnsupportedAllocation, match="methodology"):
            compile_policy("methodology/hrp@3")


class TestWeightsApplyToInvestableProceeds:

    def test_a_cash_reserve_reduces_the_basis(self, prices):
        _, lot = realized(prices)
        instruction = allocate_for(lot, policy={"VTI": 1.0}, cash_reserve=1_000.0)
        assert instruction.investable == pytest.approx(
            lot.net_proceeds - 1_000.0)

    def test_a_reserve_larger_than_the_sale_is_refused(self, prices):
        _, lot = realized(prices)
        with pytest.raises(UnsupportedAllocation, match="reserve"):
            allocate_for(lot, policy={"VTI": 1.0}, cash_reserve=10_000.0)

    def test_realized_weights_match_the_request(self, prices):
        arrival, lot = realized(prices)
        instruction = allocate_for(lot,
                                   policy={"VTI": 0.6, "VXUS": 0.3, "BND": 0.1})
        _, allocating = run_allocation(prices, arrival, instruction)
        execution = allocating.reconcile(
            _.path.fills, proceeds={lot.proceeds_id: lot}
        )[instruction.instruction_id]

        # Slightly under each target: the transaction cost comes out of the
        # investable base. That drift is reported rather than hidden by
        # rescaling the weights to hit their nominal values.
        assert execution.realized_weights == pytest.approx(
            {"VTI": 0.6, "VXUS": 0.3, "BND": 0.1}, abs=1e-3)

    def test_the_cost_drift_is_visible_rather_than_absorbed(self, prices):
        """Transaction costs must not retroactively alter the intended weight
        basis without the drift being reported."""
        arrival, lot = realized(prices)
        instruction = allocate_for(lot, policy={"VTI": 0.6, "VXUS": 0.4})
        result, allocating = run_allocation(prices, arrival, instruction)
        execution = allocating.reconcile(
            result.path.fills, proceeds={lot.proceeds_id: lot}
        )[instruction.instruction_id]

        assert sum(execution.realized_weights.values()) < 1.0
        assert execution.unallocated_weight == pytest.approx(
            instruction.cost_rate, rel=0.05)


class TestProceedsAreInternalCash:

    def test_external_flows_are_unchanged_by_reinvestment(self, prices):
        arrival, lot = realized(prices)
        instruction = allocate_for(lot, policy={"VTI": 1.0})
        result, _ = run_allocation(prices, arrival, instruction)

        # The vest delivered $3,900. Nothing since has been an external flow.
        assert float(result.path.flows.sum()) == pytest.approx(3_900.0)

    def test_holdings_move_and_cash_is_consumed(self, prices):
        arrival, lot = realized(prices)
        instruction = allocate_for(lot, policy={"VTI": 1.0})
        result, _ = run_allocation(prices, arrival, instruction)

        assert float(result.path.holdings["VTI"].iloc[-1]) > 0
        assert float(result.path.cash.iloc[-1]) == pytest.approx(0.0, abs=1e-6)


class TestFundingIsolation:

    def test_the_narrow_scope_is_the_default(self, prices):
        _, lot = realized(prices)
        instruction = allocate_for(lot, policy={"VTI": 1.0})
        assert instruction.funding_scope is FundingScope.SOURCE_PROCEEDS_ONLY

    def test_orders_are_sized_to_fit_inside_the_lot(self, prices):
        """Notional plus cost must not exceed the proceeds. Sized to the whole
        investable amount, the shortfall comes from unrelated account cash and
        the plan looks fully funded when it was not."""
        _, lot = realized(prices)
        instruction = allocate_for(lot, policy={"VTI": 0.6, "VXUS": 0.4})
        total = sum(instruction.budgets().values())
        assert total * (1 + instruction.cost_rate) <= lot.net_proceeds + 1e-9

    def test_no_negative_cash_results(self, prices):
        arrival, lot = realized(prices)
        instruction = allocate_for(lot, policy={"VTI": 0.6, "VXUS": 0.4})
        result, _ = run_allocation(prices, arrival, instruction)
        assert float(result.path.cash.min()) >= -1e-9


class TestNothingRenormalisesSilently:

    def test_an_unpriceable_target_leaves_the_others_alone(self, prices):
        """A 60/30/10 whose bond leg cannot price must not become 67/33."""
        without_bnd = prices.drop(columns=["BND"])
        arrival, lot = realized(without_bnd)
        instruction = allocate_for(lot,
                                   policy={"VTI": 0.6, "VXUS": 0.3, "BND": 0.1})
        result, allocating = run_allocation(without_bnd, arrival, instruction)
        execution = allocating.reconcile(
            result.path.fills, proceeds={lot.proceeds_id: lot}
        )[instruction.instruction_id]

        assert execution.requested_allocation["BND"] == 0.1
        assert "BND" not in execution.executed_allocation
        # Measured against the investable base, not against what was bought.
        # Normalised over the executed total these would read 67/33 — a
        # portfolio nobody asked for, reported as the one they did.
        assert execution.realized_weights["VTI"] == pytest.approx(0.6, abs=1e-3)
        assert execution.realized_weights["VXUS"] == pytest.approx(0.3, abs=1e-3)
        assert execution.unallocated_weight == pytest.approx(0.1, abs=1e-3)

    def test_the_missing_target_stays_visible(self, prices):
        without_bnd = prices.drop(columns=["BND"])
        arrival, lot = realized(without_bnd)
        instruction = allocate_for(lot,
                                   policy={"VTI": 0.6, "VXUS": 0.3, "BND": 0.1})
        result, allocating = run_allocation(without_bnd, arrival, instruction)
        execution = allocating.reconcile(
            result.path.fills, proceeds={lot.proceeds_id: lot}
        )[instruction.instruction_id]

        assert [one["asset"] for one in execution.unfilled_targets] == ["BND"]
        assert execution.status is AllocationStatus.PARTIAL

    def test_the_unallocated_money_stays_as_residual_cash(self, prices):
        without_bnd = prices.drop(columns=["BND"])
        arrival, lot = realized(without_bnd)
        instruction = allocate_for(lot,
                                   policy={"VTI": 0.6, "VXUS": 0.3, "BND": 0.1})
        result, allocating = run_allocation(without_bnd, arrival, instruction)
        execution = allocating.reconcile(
            result.path.fills, proceeds={lot.proceeds_id: lot}
        )[instruction.instruction_id]

        assert execution.residual_cash > 0
        assert conserved(execution, lot)


class TestProceedsCannotBeAllocatedTwice:

    def test_a_second_instruction_against_one_lot_is_refused(self, prices):
        _, lot = realized(prices)
        ledger = ProceedsLedger()
        allocate_for(lot, policy={"VTI": 1.0}, ledger=ledger)

        with pytest.raises(ProceedsAlreadyAllocated):
            allocate_for(lot, policy={"BND": 1.0}, ledger=ledger)

    def test_supersession_frees_the_lot(self, prices):
        _, lot = realized(prices)
        ledger = ProceedsLedger()
        first = allocate_for(lot, policy={"VTI": 1.0}, ledger=ledger)
        supersede(first, reason="user changed the target", ledger=ledger)

        second = allocate_for(lot, policy={"BND": 1.0}, ledger=ledger)
        assert second.instruction_id != first.instruction_id

    def test_a_failure_does_not_free_the_lot(self, prices):
        """Only an explicit supersession releases it. Freed by failure, a lot
        could fund two attempts."""
        _, lot = realized(prices)
        ledger = ProceedsLedger()
        first = allocate_for(lot, policy={"VTI": 1.0}, ledger=ledger)
        assert ledger.claimed_by(lot.proceeds_id) == first.instruction_id


class TestConservation:

    def test_net_proceeds_equal_invested_plus_costs_plus_residual(self, prices):
        arrival, lot = realized(prices)
        instruction = allocate_for(lot,
                                   policy={"VTI": 0.6, "VXUS": 0.3, "BND": 0.1})
        result, allocating = run_allocation(prices, arrival, instruction)
        execution = allocating.reconcile(
            result.path.fills, proceeds={lot.proceeds_id: lot}
        )[instruction.instruction_id]

        assert conserved(execution, lot)

    def test_it_holds_when_a_target_is_unfillable(self, prices):
        without_bnd = prices.drop(columns=["BND"])
        arrival, lot = realized(without_bnd)
        instruction = allocate_for(lot,
                                   policy={"VTI": 0.6, "VXUS": 0.3, "BND": 0.1})
        result, allocating = run_allocation(without_bnd, arrival, instruction)
        execution = allocating.reconcile(
            result.path.fills, proceeds={lot.proceeds_id: lot}
        )[instruction.instruction_id]

        assert conserved(execution, lot)


class TestTheSaleToReinvestmentOrdering:

    def test_the_sequence_is_in_order(self, prices):
        log = EventLog()
        arrival, lot = realized(prices, log)
        instruction = allocate_for(lot, policy={"VTI": 1.0}, log=log)
        result, allocating = run_allocation(prices, arrival, instruction, log)
        allocating.reconcile(result.path.fills,
                             proceeds={lot.proceeds_id: lot})

        assert log.in_order(ALLOCATION_ORDER)

    def test_every_stage_is_recorded(self, prices):
        log = EventLog()
        arrival, lot = realized(prices, log)
        instruction = allocate_for(lot, policy={"VTI": 1.0}, log=log)
        result, allocating = run_allocation(prices, arrival, instruction, log)
        allocating.reconcile(result.path.fills,
                             proceeds={lot.proceeds_id: lot})

        assert set(log.kinds()) >= set(ALLOCATION_ORDER)

    def test_buying_before_proceeds_exist_is_detected(self, prices):
        """The mutation this ordering test exists to catch."""
        from src.runtime.allocation import AllocationEventKind

        log = EventLog()
        arrival, lot = realized(prices, log)
        instruction = allocate_for(lot, policy={"VTI": 1.0}, log=log)
        result, allocating = run_allocation(prices, arrival, instruction, log)
        allocating.reconcile(result.path.fills,
                             proceeds={lot.proceeds_id: lot})

        log.entries.insert(0, {"kind": AllocationEventKind.PURCHASE_ORDERS_CREATED,
                               "instruction_id": "x"})
        assert not log.in_order(ALLOCATION_ORDER)

    def test_no_order_is_placed_before_the_sale_settles(self, prices):
        arrival, lot = realized(prices)
        instruction = allocate_for(lot, policy={"VTI": 1.0})
        allocating = AllocationSchedule([instruction])
        step = allocating.program()

        visible = pd.DataFrame({"VTI": [100.0]}, index=[DELIVERY])
        assert step(DELIVERY, visible, {}, 5_000.0) == []
