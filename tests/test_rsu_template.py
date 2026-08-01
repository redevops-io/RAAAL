"""RSU vesting: can the generic model carry a materially personal workflow?

The question the first template answers is not "does RSU work" but whether the
generic compiler and artifact model represent a real, messy, personal financial
workflow **without adding hidden defaults and without weakening the
public/private boundary**. Every test here is aimed at that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mission import CashPolicy, RunConditions, classify, compare, simulate
from src.mission.templates import (
    RSU_IMPLEMENTED,
    RSU_TEMPLATE,
    SUPPLEMENTAL_RATE,
    SUPPLEMENTAL_RATE_HIGH,
    SUPPLEMENTAL_THRESHOLD,
    disposition_program,
    grants_for,
    net_shares,
    next_open_session,
    withholding_for,
)
from src.mission.templates.base import InputKind


@pytest.fixture
def prices():
    """Company stock triples; the market rises modestly. The interesting case,
    because concentration looks like genius right up until it does not."""
    idx = pd.bdate_range("2021-01-04", periods=520)
    return pd.DataFrame(
        {"ACME": np.linspace(50.0, 150.0, 520), "SPY": np.linspace(100.0, 130.0, 520)},
        index=idx,
    )


@pytest.fixture
def inputs():
    return {
        "ticker": "ACME",
        "vest_dates": ["2021-02-15", "2021-05-17", "2021-08-16", "2021-11-15",
                       "2022-02-15", "2022-05-16"],
        "shares_per_vest": 100.0,
        "withholding_rate": SUPPLEMENTAL_RATE,
        "blackout_windows": [],
        "disposition": "hold",
        "diversify_into": "SPY",
    }


class TestTheTemplateDeclaresWhatItDoes:
    def test_every_assumption_names_something_that_exists(self):
        """The methodology verifier's rule, pointed at a template."""
        assert RSU_TEMPLATE.unrealized_assumptions(RSU_IMPLEMENTED) == []

    def test_a_missing_realization_is_caught(self):
        assert RSU_TEMPLATE.unrealized_assumptions(("net_shares",))

    def test_the_withholding_rule_is_cited(self):
        rates = [a for a in RSU_TEMPLATE.assumptions
                 if a.name == "statutory-supplemental-rate"]
        assert rates and rates[0].citation == "irs-pub-15-supplemental"
        assert any(c.identifier == "irs-pub-15-supplemental"
                   for c in RSU_TEMPLATE.citations)

    def test_the_under_withholding_trap_is_recorded_as_a_risk(self):
        """A withholding rate is not a tax rate, and the gap arrives in April."""
        [rate] = [a for a in RSU_TEMPLATE.assumptions
                  if a.name == "statutory-supplemental-rate"]

        assert "not a tax rate" in rate.risk
        assert "under-withhold" in rate.risk

    def test_capital_gains_are_declared_unmodelled_rather_than_assumed(self):
        [limit] = [l for l in RSU_TEMPLATE.limitations if l.name == "no-capital-gains"]
        assert "not taxed in this simulation" in limit.statement

    def test_payroll_and_state_tax_are_declared_unmodelled(self):
        names = {l.name for l in RSU_TEMPLATE.limitations}
        assert "no-payroll-or-state-tax" in names
        assert "no-10b5-1-plan" in names


class TestInputsAreTypedAndRefuseTheHundredfoldError:
    def test_a_rate_typed_as_a_percentage_is_refused(self):
        problems = RSU_TEMPLATE.validate({
            "ticker": "ACME", "vest_dates": ["2021-01-04"],
            "shares_per_vest": 100.0, "withholding_rate": 22.0,
        })
        assert any("hundredfold" in p for p in problems)

    def test_missing_required_inputs_are_named(self):
        problems = RSU_TEMPLATE.validate({})
        assert any("ticker is required" in p for p in problems)

    def test_unknown_inputs_are_refused_rather_than_ignored(self):
        """A template that silently accepts a field will appear to honour it."""
        problems = RSU_TEMPLATE.validate({
            "ticker": "ACME", "vest_dates": ["2021-01-04"],
            "shares_per_vest": 100.0, "salary": 200_000,
        })
        assert any("unrecognised input" in p for p in problems)

    def test_a_bad_disposition_is_refused(self):
        problems = RSU_TEMPLATE.validate({
            "ticker": "ACME", "vest_dates": ["2021-01-04"],
            "shares_per_vest": 100.0, "disposition": "yolo",
        })
        assert any("must be one of" in p for p in problems)

    def test_every_input_states_its_unit_or_type(self):
        for spec in RSU_TEMPLATE.inputs:
            assert spec.kind in InputKind
            if spec.kind in (InputKind.RATE, InputKind.SHARES, InputKind.MONEY):
                assert spec.unit, f"{spec.name} has no unit"


class TestWithholding:
    def test_the_flat_rate_applies_below_the_threshold(self):
        assert withholding_for(500_000) == pytest.approx(500_000 * SUPPLEMENTAL_RATE)

    def test_a_vest_straddling_the_threshold_is_split(self):
        """One rate applied to the whole vest is wrong in the flattering direction."""
        withheld = withholding_for(400_000, cumulative_supplemental=800_000)
        expected = 200_000 * SUPPLEMENTAL_RATE + 200_000 * SUPPLEMENTAL_RATE_HIGH

        assert withheld == pytest.approx(expected)
        assert withheld > 400_000 * SUPPLEMENTAL_RATE

    def test_withheld_shares_never_reach_the_account(self):
        assert net_shares(100, 100.0) == pytest.approx(78.0)

    def test_the_threshold_is_the_statutory_one(self):
        assert SUPPLEMENTAL_THRESHOLD == 1_000_000.0


class TestVestingIsNotAPurchase:
    def test_grants_are_net_of_withholding(self, prices, inputs):
        grants = grants_for(inputs, prices)

        assert len(grants) == len(inputs["vest_dates"])
        assert all(g.shares == pytest.approx(78.0) for g in grants)

    def test_a_vest_lands_on_the_next_session(self, prices, inputs):
        """A vest dated to a weekend is not tradeable on that weekend."""
        grants = grants_for({**inputs, "vest_dates": ["2021-02-13"]}, prices)
        assert grants[0].date in prices.index

    def test_delivered_value_counts_as_contributed_capital(self, prices, inputs):
        """Shares arriving from outside are money entering the portfolio."""
        result = simulate(
            prices, flows=[], grants=grants_for(inputs, prices),
            program=disposition_program(inputs, prices.index),
            cash_policy=CashPolicy.idle())

        assert result.path.contributed > 0
        assert result.money_weighted is not None

    def test_no_cash_is_spent_receiving_a_vest(self, prices, inputs):
        result = simulate(
            prices, flows=[], grants=grants_for(inputs, prices),
            program=disposition_program(inputs, prices.index),
            cash_policy=CashPolicy.idle())

        assert result.path.fills == (), "receiving shares placed a trade"


class TestBlackoutsDeferRatherThanCancel:
    def test_a_sale_inside_a_window_moves_to_the_next_open_session(self, prices):
        blackouts = [("2021-02-10", "2021-02-25")]
        when = next_open_session(pd.Timestamp("2021-02-16"), prices.index, blackouts)

        assert when > pd.Timestamp("2021-02-25")
        assert when in prices.index

    def test_a_session_outside_every_window_is_returned_unchanged(self, prices):
        session = pd.Timestamp("2021-06-15")
        assert next_open_session(session, prices.index, [("2021-02-10", "2021-02-25")]) \
            == session

    def test_a_deferred_sale_still_happens(self, prices, inputs):
        """Dropping it would silently turn a diversification plan into a hold."""
        values = {**inputs, "disposition": "sell_all_and_diversify",
                  "blackout_windows": [("2021-02-10", "2021-03-25")]}
        result = simulate(
            prices, flows=[], grants=grants_for(values, prices),
            program=disposition_program(values, prices.index),
            cash_policy=CashPolicy.idle())

        sales = [f for f in result.path.fills if f.shares < 0]
        assert sales, "the deferred sale never happened"


class TestTheComparisonTheTemplateExistsToMake:
    def _run(self, prices, values):
        return simulate(prices, flows=[], grants=grants_for(values, prices),
                        program=disposition_program(values, prices.index),
                        cash_policy=CashPolicy.idle())

    def test_holding_and_diversifying_receive_identical_vests(self, prices, inputs):
        """Otherwise the difference is the schedule, not the decision."""
        hold = self._run(prices, {**inputs, "disposition": "hold"})
        diversify = self._run(
            prices, {**inputs, "disposition": "sell_all_and_diversify"})

        assert hold.path.contributed == pytest.approx(diversify.path.contributed)

    def test_the_comparison_isolates_the_disposition(self, prices, inputs):
        # Every dimension pinned. "Same vests, same costs, same period" is a
        # claim about what was held identical, and under classifier @2 a
        # dimension nobody pinned was not held identical — it was not looked at.
        conditions = RunConditions(
            flow_schedule_hash="rsu-vest-schedule", starting_capital=0.0,
            cash_policy_rate=0.0, tax_treatment="NONE_APPLIED", cost_bps=10.0,
            execution_lag=1, period_start="2021-01-04", period_end="2023-01-01",
            allocation_rule_hash="rsu-disposition", data_snapshot="prices@2023-01-01",
            account_hash="account/taxable@1", calendar_hash="calendar/nyse@1",
            market_data_hash="market-data/test@1")
        verdict = classify(conditions, conditions)

        assert verdict.attribution_isolated, (
            "same vests, same costs, same period — the only difference is what "
            "was done with the shares"
        )

    def test_acceptance_diversifying_out_of_a_tripling_stock_costs_money(
            self, prices, inputs):
        """ACCEPTANCE: the platform must state an inconvenient truth in the same
        tone as a convenient one.

        Concentration in a stock that tripled beat diversification, and the
        product has to be able to say so. This is stronger evidence of neutrality
        than any wording policy, because a wording policy is only tested against
        outputs someone thought to check.
        """
        hold = self._run(prices, {**inputs, "disposition": "hold"})
        diversify = self._run(
            prices, {**inputs, "disposition": "sell_all_and_diversify"})

        assert hold.final_value > diversify.final_value

    def test_selling_half_sits_between_the_two(self, prices, inputs):
        results = {
            d: self._run(prices, {**inputs, "disposition": d}).final_value
            for d in ("hold", "sell_half_and_diversify", "sell_all_and_diversify")
        }
        assert results["sell_all_and_diversify"] <= \
            results["sell_half_and_diversify"] <= results["hold"]


class TestTheBoundaryIsUnaffected:
    def test_the_template_carries_no_personal_data(self):
        """Inputs describe *shapes* of personal data; the template holds none."""
        from src.mission.boundary import scan_for_personal_data

        assert scan_for_personal_data(RSU_TEMPLATE.to_json()) == []

    def test_a_filled_in_template_is_private(self, inputs):
        """The values a user supplies are exactly what may not go public."""
        from src.mission.boundary import scan_for_personal_data

        leaks = scan_for_personal_data({"employer": "ACME Corp", **inputs})
        assert leaks

    def test_the_template_is_content_hashed(self):
        assert len(RSU_TEMPLATE.content_hash) == 64
        assert RSU_TEMPLATE.artifact_id == "template/rsu-vesting@1"


class TestTheScopeTravelsWithTheNumber:
    """A scope shown only on a configuration screen is absent from the figure
    someone quotes, and the figure is what travels."""

    def test_a_result_can_carry_its_scope(self, prices, inputs):
        from src.mission import CashPolicy, simulate
        from src.mission.templates import RSU_TEMPLATE, grants_for

        result = simulate(
            prices, flows=[], grants=grants_for(inputs, prices),
            program=disposition_program(inputs, prices.index),
            cash_policy=CashPolicy.idle(),
            modelling_scope=RSU_TEMPLATE.modelling_scope())
        payload = result.to_json()

        assert payload["modelling_scope"]["not_modelled"]
        assert "overstates what it accounts for" in payload["scope_note"]

    def test_a_result_without_a_scope_says_so_rather_than_omitting_it(self, prices):
        from src.mission import CashFlow, CashPolicy, buy_and_hold, simulate

        result = simulate(prices, flows=[CashFlow(prices.index[0], 1000.0)],
                          program=buy_and_hold(["SPY"]),
                          cash_policy=CashPolicy.idle())

        assert result.to_json()["modelling_scope"] is None
        assert "No modelling scope" in result.to_json()["scope_note"]

    def test_the_scope_separates_mechanical_from_jurisdictional(self):
        scope = RSU_TEMPLATE.modelling_scope()
        modelled = {m["name"] for m in scope["modelled"]}
        not_modelled = {n["name"] for n in scope["not_modelled"]}

        assert "withholding-is-share-reduction" in modelled
        assert "no-capital-gains" in not_modelled
        assert "depends on facts about you that have not been stated" in scope["note"]

    def test_every_unmodelled_item_gives_a_reason(self):
        for item in RSU_TEMPLATE.modelling_scope()["not_modelled"]:
            assert len(item["reason"]) > 40, f"{item['name']} states no reason"

    def test_a_stored_run_without_a_scope_is_refused(self, tmp_path):
        """A stored number that lost its scope will be read as excluding nothing."""
        from src.workspace.store import NotSaveable, WorkspaceStore

        store = WorkspaceStore(tmp_path / "w.db")
        with pytest.raises(NotSaveable, match="excluding nothing"):
            store.record_run(run_id="r", plan_id="p", ran_at="2026-07-31",
                             result={"final_value": 1.0}, comparison={})
