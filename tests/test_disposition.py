"""The disposition lifecycle: a sale that was decided must not quietly not happen.

    A disposition instruction survives until it executes, expires by declared
    policy, or is explicitly superseded. It is never dropped because the vest
    date was ineligible.

A sale discarded because the vest landed inside a blackout converts a
diversification plan into a hold. The portfolio still looks reasonable
afterwards, which is exactly why the failure is invisible.

A vest is a fact about compensation; a disposition is a decision about it. They
are separate objects because one can be certain while the other is still
pending, and merging them makes the pending one look settled.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tests.vest_fixtures import resolved_for

from src.mission.accounting import CashPolicy
from src.mission.simulate import simulate
from src.runtime.disposition import (
    CANONICAL_ORDER,
    DispositionSchedule,
    DispositionStatus,
    EventLog,
    UnsupportedPolicy,
    VestEventKind,
    eligibility,
    instruction_for,
    supersede,
)
from src.runtime.rsu import VestEvent, WithholdingMethod, in_kind_flow_for

DELIVERY = pd.Timestamp("2026-03-02")


@pytest.fixture
def sessions():
    return pd.bdate_range("2026-03-02", "2026-04-30")


@pytest.fixture
def prices(sessions):
    return pd.DataFrame({"ACME": 50.0, "VTI": 100.0}, index=sessions)


def vest(**overrides) -> VestEvent:
    fields = dict(grant_id="g1", employer_ticker="ACME", vest_date="2026-03-02",
                  gross_shares=100.0, vest_price_source="p",
                  withholding_rate=0.22,
                  withholding_method=WithholdingMethod.SHARE_WITHHOLDING,
                  market_data_ref="md@1", corporate_action_ref="ca@1")
    fields.update(overrides)
    return VestEvent(**fields)


def instruction(policy="SELL_ALL_AND_DIVERSIFY", *, blackouts=(), shares=78.0,
                expires_at=None, log=None):
    return instruction_for(
        vest_ref="g1", grant_ref="g1", asset="ACME", delivered_shares=shares,
        policy=policy, delivery_session=DELIVERY, blackouts=blackouts,
        expires_at=expires_at, log=log)


def drive(prices, schedule, *, in_kind=()):
    result = simulate(prices, flows=[], program=schedule.program(),
                      in_kind=list(in_kind), cash_policy=CashPolicy.idle(),
                      modelling_scope={"excludes": []})
    schedule.reconcile(result.path.fills)
    return result


def vested(price=50.0):
    arrival, accounting = in_kind_flow_for(vest(), vest_price=price, resolved=resolved_for(vest()))
    return arrival, accounting


class TestPolicyProducesATypedInstructionOrNone:

    def test_hold_creates_no_instruction(self):
        """Not a zero-quantity one. A zero-share sale in the history reads as
        an attempt that failed."""
        assert instruction("HOLD") is None

    def test_sell_all_takes_the_whole_delivered_quantity(self):
        assert instruction("SELL_ALL_AND_DIVERSIFY").quantity == 78.0

    def test_sell_half_takes_half(self):
        assert instruction("SELL_HALF_AND_DIVERSIFY").quantity == 39.0

    def test_a_concentration_target_is_unsupported_not_approximated(self):
        """"Sell enough to get employer stock under 20%" is not "sell half",
        and substituting one would sell a different number of shares with
        nothing in the result showing it."""
        with pytest.raises(UnsupportedPolicy, match="concentration"):
            instruction("REDUCE_CONCENTRATION_BELOW_20")

    def test_an_unknown_policy_is_refused(self):
        with pytest.raises(UnsupportedPolicy, match="unknown"):
            instruction("SELL_SOME_MAYBE")

    def test_an_instruction_cannot_predate_its_delivery(self):
        assert instruction().earliest_eligible_date == DELIVERY


class TestBlackoutDefersAndPreserves:

    def test_shares_arrive_even_though_the_sale_cannot(self, prices):
        arrival, _ = vested()
        schedule = DispositionSchedule(
            [instruction(blackouts=[("2026-03-02", "2026-03-10")])])
        result = drive(prices, schedule, in_kind=[arrival])
        assert float(result.path.holdings["ACME"].max()) == pytest.approx(78.0)

    def test_the_instruction_is_not_cancelled(self, prices):
        arrival, _ = vested()
        schedule = DispositionSchedule(
            [instruction(blackouts=[("2026-03-02", "2026-03-10")])])
        drive(prices, schedule, in_kind=[arrival])
        assert schedule.instructions[0].status is DispositionStatus.EXECUTED

    def test_it_executes_on_the_first_eligible_session(self, prices):
        arrival, _ = vested()
        schedule = DispositionSchedule(
            [instruction(blackouts=[("2026-03-02", "2026-03-10")])])
        drive(prices, schedule, in_kind=[arrival])
        assert schedule.executions[0].instructed_on == pd.Timestamp("2026-03-11")

    def test_a_blackout_ending_on_a_weekend_resumes_on_the_next_session(
            self, prices):
        """The window ends Saturday the 7th; the next session is Monday."""
        arrival, _ = vested()
        schedule = DispositionSchedule(
            [instruction(blackouts=[("2026-03-02", "2026-03-07")])])
        drive(prices, schedule, in_kind=[arrival])
        assert schedule.executions[0].instructed_on == pd.Timestamp("2026-03-09")

    def test_it_survives_multiple_windows_and_sells_once(self, prices):
        arrival, _ = vested()
        schedule = DispositionSchedule([instruction(blackouts=[
            ("2026-03-02", "2026-03-10"), ("2026-03-11", "2026-03-20")])])
        drive(prices, schedule, in_kind=[arrival])
        assert len(schedule.executions) == 1
        assert schedule.executions[0].instructed_on == pd.Timestamp("2026-03-23")

    def test_every_deferral_names_its_reason(self):
        blocked = eligibility(
            instruction(blackouts=[("2026-03-02", "2026-03-10")]),
            pd.Timestamp("2026-03-04"), held_shares=78.0, price=50.0)
        assert not blocked.eligible
        assert any("blackout" in reason for reason in blocked.blocked_by)


class TestExecutionHappensOnlyAfterDelivery:

    def test_a_session_before_delivery_is_ineligible(self):
        verdict = eligibility(instruction(), pd.Timestamp("2026-02-27"),
                              held_shares=0.0, price=50.0)
        assert not verdict.eligible
        assert any("delivered" in reason for reason in verdict.blocked_by)

    def test_the_fill_follows_the_instruction_by_the_execution_lag(self, prices):
        """Instruction and fill are different facts. An order placed on the
        first eligible session fills later, at a price nobody knew when
        deciding."""
        arrival, _ = vested()
        schedule = DispositionSchedule([instruction()])
        drive(prices, schedule, in_kind=[arrival])
        execution = schedule.executions[0]
        assert execution.filled_on > execution.instructed_on

    def test_proceeds_are_unknown_until_reconciled(self):
        """Not zero. An unreconciled sale has unknown proceeds, and zero is a
        number."""
        schedule = DispositionSchedule([instruction()])
        step = schedule.program()
        visible = pd.DataFrame({"ACME": [50.0]}, index=[DELIVERY])
        step(DELIVERY, visible, {"ACME": 78.0}, 0.0)
        assert schedule.executions[0].proceeds is None
        assert not schedule.executions[0].reconciled


class TestTheSaleIsNotAnExternalFlow:
    """The only external flow is the delivered in-kind value at vest. Counting
    the proceeds again credits the same compensation twice in MWR, and the
    number looks entirely plausible."""

    def test_external_flows_are_unchanged_by_the_disposition(self, prices):
        arrival, accounting = vested()

        held = simulate(prices, flows=[], program=DispositionSchedule().program(),
                        in_kind=[arrival], cash_policy=CashPolicy.idle(),
                        modelling_scope={"excludes": []})
        schedule = DispositionSchedule([instruction()])
        sold = drive(prices, schedule, in_kind=[arrival])

        assert float(sold.path.flows.sum()) == pytest.approx(
            float(held.path.flows.sum()))
        assert float(sold.path.flows.sum()) == pytest.approx(
            accounting["external_flow_value"])

    def test_cash_rises_by_the_net_proceeds(self, prices):
        arrival, _ = vested()
        schedule = DispositionSchedule([instruction()])
        result = drive(prices, schedule, in_kind=[arrival])
        # 78 shares at $50 less 10bps of transaction cost.
        assert float(result.path.cash.iloc[-1]) == pytest.approx(
            3_900.0 * (1 - 0.001), rel=1e-6)

    def test_shares_fall_by_the_sold_quantity(self, prices):
        arrival, _ = vested()
        schedule = DispositionSchedule([instruction("SELL_HALF_AND_DIVERSIFY")])
        result = drive(prices, schedule, in_kind=[arrival])
        assert float(result.path.holdings["ACME"].iloc[-1]) == pytest.approx(39.0)


class TestCostsApplyOnlyToTheSale:

    def test_delivery_is_free_and_the_sale_is_not(self, prices):
        arrival, _ = vested()

        delivered = simulate(prices, flows=[],
                             program=DispositionSchedule().program(),
                             in_kind=[arrival], cash_policy=CashPolicy.idle(),
                             modelling_scope={"excludes": []})
        assert delivered.path.terminal_value == pytest.approx(3_900.0)

        schedule = DispositionSchedule([instruction()])
        sold = drive(prices, schedule, in_kind=[arrival])
        assert sold.path.terminal_value < 3_900.0


class TestFailureAndSupersessionAreExplicit:

    def test_insufficient_shares_blocks_rather_than_going_negative(self, prices):
        oversized = instruction(shares=500.0)
        arrival, _ = vested()
        schedule = DispositionSchedule([oversized])
        result = drive(prices, schedule, in_kind=[arrival])

        assert float(result.path.holdings["ACME"].min()) >= 0.0
        assert schedule.instructions[0].status is not DispositionStatus.EXECUTED

    def test_an_unexecutable_instruction_stays_inspectable(self, prices):
        arrival, _ = vested()
        schedule = DispositionSchedule([instruction(shares=500.0)])
        drive(prices, schedule, in_kind=[arrival])

        [report] = schedule.unsettled_report()
        assert report["quantity"] == 500.0
        assert "shares held" in report["why"]

    def test_no_pending_instruction_disappears(self, prices):
        arrival, _ = vested()
        schedule = DispositionSchedule([instruction(shares=500.0)])
        drive(prices, schedule, in_kind=[arrival])
        assert len(schedule.instructions) == 1

    def test_an_expired_instruction_says_so(self, prices):
        arrival, _ = vested()
        schedule = DispositionSchedule([instruction(
            blackouts=[("2026-03-02", "2026-04-30")],
            expires_at=pd.Timestamp("2026-03-20"))])
        drive(prices, schedule, in_kind=[arrival])
        assert schedule.instructions[0].status is DispositionStatus.EXPIRED

    def test_supersession_keeps_both(self):
        original = instruction()
        replaced = supersede(original, reason="user chose to hold instead")
        assert replaced.status is DispositionStatus.SUPERSEDED
        assert original.status is DispositionStatus.PENDING

    def test_a_settled_instruction_cannot_be_superseded(self):
        from dataclasses import replace as _replace

        executed = _replace(instruction(), status=DispositionStatus.EXECUTED)
        with pytest.raises(ValueError, match="EXECUTED"):
            supersede(executed, reason="too late")

    def test_a_price_gap_defers_rather_than_filling_at_zero(self):
        verdict = eligibility(instruction(), pd.Timestamp("2026-03-05"),
                              held_shares=78.0, price=float("nan"))
        assert not verdict.eligible
        assert any("price" in reason for reason in verdict.blocked_by)


class TestStatusIsNotRederivedFromDates:

    def test_an_executed_sale_stays_executed(self, prices):
        """Recomputed from dates, an executed sale would become pending again
        the moment a later blackout window covered its date."""
        from src.runtime.disposition import advance

        from dataclasses import replace as _replace

        executed = _replace(instruction(), status=DispositionStatus.EXECUTED)
        moved = advance(executed, pd.Timestamp("2026-04-01"), held_shares=0.0,
                        price=50.0)
        assert moved.status is DispositionStatus.EXECUTED


class TestTheVestToSaleOrdering:

    def full_sequence(self, prices, blackouts=()):
        log = EventLog()
        log.record(VestEventKind.VEST_VALUED)
        log.record(VestEventKind.WITHHOLDING_APPLIED)
        arrival, accounting = vested()
        log.record(VestEventKind.SHARES_DELIVERED,
                   shares=accounting["shares_delivered"])
        log.record(VestEventKind.EXTERNAL_FLOW_RECORDED,
                   value=accounting["external_flow_value"])
        schedule = DispositionSchedule(
            [instruction(blackouts=blackouts, log=log)], log=log)
        drive(prices, schedule, in_kind=[arrival])
        return log, schedule

    def test_the_sequence_is_in_canonical_order(self, prices):
        log, _ = self.full_sequence(prices)
        assert log.in_canonical_order()

    def test_every_stage_occurs(self, prices):
        log, _ = self.full_sequence(prices)
        recorded = set(log.kinds())
        assert recorded >= set(CANONICAL_ORDER)

    def test_deferral_does_not_break_the_order(self, prices):
        """An instruction may be deferred any number of times, between any two
        stages."""
        log, _ = self.full_sequence(
            prices, blackouts=[("2026-03-02", "2026-03-10")])
        assert log.in_canonical_order()
        assert VestEventKind.DISPOSITION_DEFERRED in log.kinds()

    def test_selling_before_delivery_is_detected_as_out_of_order(self, prices):
        """The mutation the ordering test exists to catch."""
        log, _ = self.full_sequence(prices)
        log.entries.insert(
            0, {"kind": VestEventKind.SALE_EXECUTED, "instruction_id": "x"})
        assert not log.in_canonical_order()

    def test_flow_recorded_before_delivery_is_out_of_order(self, prices):
        log, _ = self.full_sequence(prices)
        log.entries.insert(
            0, {"kind": VestEventKind.EXTERNAL_FLOW_RECORDED, "value": 1.0})
        assert not log.in_canonical_order()
