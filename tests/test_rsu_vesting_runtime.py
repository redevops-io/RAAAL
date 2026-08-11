"""RSU vesting semantics, before any result surface exists.

A vest is the first event that touches ownership, cash flow, taxation,
concentration, liquidity and benchmark attribution at once. A partially modelled
one produces a figure that looks complete while being wrong in several
directions simultaneously, so every case here is deliberately adversarial and
the declarations are checked before a worksheet block is allowed to display
anything.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tests.vest_fixtures import resolved_for

from src.runtime import ACCOUNT_IMPLEMENTED
from src.runtime.rsu import (
    IMPLEMENTED,
    NEVER_INFERRED,
    US_SHARE_WITHHOLDING,
    DispositionPolicy,
    RSUVestingRuntime,
    UnpinnedVest,
    VestEvent,
    WithholdingMethod,
    allocate_disposition_proceeds,
    apply_share_withholding,
    apply_supplemental_wage_threshold,
    apply_vest_delivery,
    compute_employer_concentration,
    next_eligible_disposition_session,
)


def vest(**overrides) -> VestEvent:
    """A fully pinned vest. Tests remove fields to prove the refusals."""
    fields = dict(
        grant_id="g1", employer_ticker="ACME", vest_date="2026-03-02",
        gross_shares=100.0, vest_price_source="prices@2026-03-02",
        withholding_method=WithholdingMethod.SHARE_WITHHOLDING,
        withholding_rate=0.22, market_data_ref="md@1",
        corporate_action_ref="ca/none@1")
    fields.update(overrides)
    return VestEvent(**fields)


class TestAVestIsNotAPurchase:
    """The founding distinction. Modelled as "cash arrives, then buy", a vest
    acquires a session of slippage and credits the plan with a trade nobody
    made."""

    def test_it_delivers_shares_in_kind(self):
        grant, _ = apply_vest_delivery(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        assert grant.ticker == "ACME"
        assert grant.shares > 0

    def test_no_order_is_generated_for_the_delivered_shares(self):
        """A `Grant` is not an `Order`. The delivery has no fill price, no
        slippage and no transaction cost, because no transaction happened."""
        from src.mission.accounting import Order

        grant, _ = apply_vest_delivery(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        assert not isinstance(grant, Order)
        assert not hasattr(grant, "notional")


class TestWithheldSharesNeverEnterHoldings:

    def test_a_hundred_share_vest_at_22_percent_delivers_78(self):
        split = apply_share_withholding(
            100.0, 50.0, rate=0.22,
            method=WithholdingMethod.SHARE_WITHHOLDING)
        assert split["gross_shares"] == 100.0
        assert split["withheld_shares"] == pytest.approx(22.0)
        assert split["delivered_shares"] == pytest.approx(78.0)

    def test_the_grant_carries_only_the_delivered_count(self):
        """There must be no intermediate state holding 100 shares followed by a
        sale of 22 — the withheld shares are never granted."""
        grant, _ = apply_vest_delivery(vest(), vest_price=50.0, resolved=resolved_for(vest()))
        assert grant.shares == pytest.approx(78.0)

    def test_both_counts_are_reported_not_only_the_net(self):
        """"78 arrived" and "100 vested, 22 withheld" answer different
        questions, and the second is what a user checks against a statement."""
        split = apply_share_withholding(
            100.0, 50.0, rate=0.22,
            method=WithholdingMethod.SHARE_WITHHOLDING)
        assert set(split) >= {"gross_shares", "withheld_shares",
                              "delivered_shares", "vest_value",
                              "withheld_value"}


class TestTheSupplementalThresholdSplits:

    def test_a_vest_below_the_threshold_uses_the_standard_rate(self):
        assert apply_supplemental_wage_threshold(
            100_000.0, rate=0.22) == pytest.approx(22_000.0)

    def test_a_straddling_vest_applies_both_rates(self):
        """One rate applied to the whole vest is wrong in the direction that
        flatters the result."""
        withheld = apply_supplemental_wage_threshold(
            400_000.0, rate=0.22, cumulative_supplemental=800_000.0)
        # $200k below the $1M threshold at 22%, $200k above at 37%.
        assert withheld == pytest.approx(200_000 * 0.22 + 200_000 * 0.37)

    def test_it_is_not_one_rate_on_the_whole_amount(self):
        withheld = apply_supplemental_wage_threshold(
            400_000.0, rate=0.22, cumulative_supplemental=800_000.0)
        assert withheld != pytest.approx(400_000 * 0.22)
        assert withheld != pytest.approx(400_000 * 0.37)

    def test_a_vest_entirely_above_the_threshold_uses_the_high_rate(self):
        withheld = apply_supplemental_wage_threshold(
            50_000.0, rate=0.22, cumulative_supplemental=2_000_000.0)
        assert withheld == pytest.approx(50_000 * 0.37)


class TestBlackoutDefersRatherThanCancels:

    @pytest.fixture
    def sessions(self):
        return pd.bdate_range("2026-03-02", "2026-03-20")

    def test_a_sale_inside_a_blackout_moves_to_the_next_eligible_session(
            self, sessions):
        when = next_eligible_disposition_session(
            pd.Timestamp("2026-03-03"), sessions,
            [("2026-03-03", "2026-03-06")])
        assert when == pd.Timestamp("2026-03-09")

    def test_a_sale_outside_a_blackout_is_unaffected(self, sessions):
        when = next_eligible_disposition_session(
            pd.Timestamp("2026-03-10"), sessions,
            [("2026-03-03", "2026-03-06")])
        assert when == pd.Timestamp("2026-03-10")

    def test_it_is_not_cancelled(self, sessions):
        """Dropping a blocked sale silently converts a diversification plan
        into a hold — a different strategy nobody chose."""
        assert next_eligible_disposition_session(
            pd.Timestamp("2026-03-03"), sessions,
            [("2026-03-03", "2026-03-06")]) is not None


class TestUnknownsStayUnknown:

    @pytest.mark.parametrize("missing", [
        "vest_price_source", "market_data_ref", "corporate_action_ref"])
    def test_a_missing_requirement_makes_the_vest_unmodellable(self, missing):
        incomplete = vest(**{missing: None})
        assert not incomplete.modellable
        assert missing in incomplete.unresolved()

    def test_an_unspecified_withholding_method_is_unresolved(self):
        """Share withholding and sell-to-cover deliver different share counts
        and different cost bases; neither stands in for the other."""
        assert "withholding_method" in vest(
            withholding_method=WithholdingMethod.UNSPECIFIED).unresolved()

    def test_an_unmodellable_vest_is_refused_not_approximated(self):
        with pytest.raises(UnpinnedVest, match="corporate_action_ref"):
            apply_vest_delivery(vest(corporate_action_ref=None), vest_price=50.0, resolved=resolved_for(vest(corporate_action_ref=None)))

    def test_a_missing_withholding_rate_is_refused(self):
        with pytest.raises(UnpinnedVest, match="withholding rate"):
            apply_vest_delivery(vest(withholding_rate=None), vest_price=50.0, resolved=resolved_for(vest(withholding_rate=None)))

    def test_a_missing_vest_price_is_a_data_gap_not_a_free_vest(self):
        """No inferred price, no zero-price grant."""
        with pytest.raises(UnpinnedVest, match="vest price"):
            apply_vest_delivery(vest(), vest_price=0.0, resolved=resolved_for(vest()))

    def test_nothing_infers_a_tax_or_account_fact(self):
        for name in ("marginal_tax_rate", "cost_basis_method", "state_tax",
                     "disposition_treatment", "tax_jurisdiction"):
            assert name in NEVER_INFERRED

    def test_an_unstated_allocation_rule_refuses_rather_than_defaulting(self):
        """Selling into an unnamed default picks an investment for the user."""
        with pytest.raises(UnpinnedVest, match="allocation rule"):
            allocate_disposition_proceeds(10_000.0, None)

    def test_a_stated_allocation_rule_is_honoured(self):
        assert allocate_disposition_proceeds(10_000.0, "VTI") == {"VTI": 10_000.0}


class TestTheCorporateActionGate:
    """The vest engine must not silently trust raw share counts across a split,
    a symbol change or a merger."""

    def test_an_unpinned_corporate_action_policy_blocks_the_vest(self):
        assert "corporate_action_ref" in vest(
            corporate_action_ref=None).unresolved()

    def test_blocking_happens_before_any_arithmetic(self):
        with pytest.raises(UnpinnedVest):
            apply_vest_delivery(vest(corporate_action_ref=None), vest_price=50.0, resolved=resolved_for(vest(corporate_action_ref=None)))


class TestConcentrationIsMeasured:

    def test_it_reports_the_employer_share_of_the_portfolio(self):
        result = compute_employer_concentration(
            {"ACME": 100.0, "VTI": 100.0}, {"ACME": 60.0, "VTI": 40.0}, "ACME")
        assert result["employer_fraction"] == pytest.approx(0.6)

    def test_an_empty_portfolio_reports_zero_rather_than_dividing(self):
        result = compute_employer_concentration({}, {}, "ACME")
        assert result["employer_fraction"] == 0.0

    def test_a_rising_employer_stock_raises_concentration(self):
        """The number a vesting plan exists to move, and the one a rising
        employer stock quietly raises while every other figure looks healthy."""
        before = compute_employer_concentration(
            {"ACME": 100.0, "VTI": 100.0}, {"ACME": 50.0, "VTI": 50.0}, "ACME")
        after = compute_employer_concentration(
            {"ACME": 100.0, "VTI": 100.0}, {"ACME": 150.0, "VTI": 50.0}, "ACME")
        assert after["employer_fraction"] > before["employer_fraction"]


class TestTheRuntimeDeclaresAndIsChecked:

    def test_every_implemented_name_is_a_real_callable(self):
        """The registry cannot claim a mechanism into existence."""
        import src.runtime.rsu as module

        for name in IMPLEMENTED:
            assert callable(getattr(module, name, None)), name

    def test_every_declared_assumption_names_an_implemented_mechanism(self):
        assert US_SHARE_WITHHOLDING.unrealized(IMPLEMENTED) == []

    def test_an_undeclared_mechanism_leaves_the_runtime_partial(self):
        """A declaration with no callable must keep the scenario PARTIAL,
        exactly as the account work does."""
        without_concentration = [n for n in IMPLEMENTED
                                 if n != "compute_employer_concentration"]
        assert "employer-concentration" in US_SHARE_WITHHOLDING.unrealized(
            without_concentration)

    def test_a_runtime_declaring_nothing_it_performs_reports_everything_unrealized(
            self):
        pretender = RSUVestingRuntime(
            name="pretend", version=1, models_blackouts=True,
            models_disposition=True, measures_concentration=True,
            withholding_method=WithholdingMethod.SHARE_WITHHOLDING,
            supplemental_rate=0.22, supplemental_threshold=1_000_000.0,
            high_rate=0.37)
        assert len(pretender.unrealized(())) == len(pretender.assumptions)


class TestTheScopeIsStatedBeforeAnyFigure:

    def test_withholding_is_not_claimed_to_be_tax(self):
        [entry] = [l for l in US_SHARE_WITHHOLDING.limitations
                   if l.name == "withholding-is-not-tax"]
        assert "marginal" in entry.statement

    @pytest.mark.parametrize("name", [
        "no-capital-gains-lots", "no-state-or-local-tax", "no-wash-sale",
        "no-83b-or-espp", "no-plan-documents"])
    def test_the_unmodelled_things_are_named(self, name):
        assert name in {l.name for l in US_SHARE_WITHHOLDING.limitations}

    def test_no_legal_advice_is_offered(self):
        [entry] = [l for l in US_SHARE_WITHHOLDING.limitations
                   if l.name == "no-plan-documents"]
        assert "no legal advice" in entry.statement.lower()

    def test_rounding_is_filed_as_rounding_not_as_missing_withholding(self):
        """Filing this as "withholding not modelled" would make a correct
        treatment look like a gap."""
        names = {l.name for l in US_SHARE_WITHHOLDING.limitations}
        assert "no-whole-share-rounding" in names
        assert "share-withholding" in {a.name for a
                                       in US_SHARE_WITHHOLDING.assumptions}

    def test_the_scope_travels_with_the_runtime(self):
        scope = US_SHARE_WITHHOLDING.scope()
        assert scope["modelled"]
        assert scope["not_modelled"]
