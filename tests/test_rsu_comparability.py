"""Comparability of the whole vest → sale → reinvest mechanism.

    A difference may be attributed to strategy only after every non-strategy
    dimension that can change the resulting sale quantity, proceeds, or
    reinvestment path has been checked and matched.

Matching vest flows used to be nearly enough. Once a sale quantity became a
function of live portfolio state, two runs could agree on every flow, price and
date, disagree only about whether fractional shares may be sold, and produce
different sales — a difference that is not strategy and would be reported as one.

The essential test in this file is `TestEveryFieldIsLoadBearing`, which mutates
each declared field in turn and requires the verdict to change. Without it the
equality list can look complete while omitting the dimension that actually moves
the sale.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from src.comparison.rsu_profile import (
    ALL_FIELDS,
    ALLOCATION,
    CONCENTRATION,
    DISPOSITION,
    ENVIRONMENT,
    PROFILE_VERSION,
    VEST_FLOW,
    BenchmarkStatus,
    RSUComparisonProfile,
    classify,
    evaluate,
)

BASE = RSUComparisonProfile(
    grant_identity="grant/g1", vest_dates=("2026-03-02",),
    delivered_values=(3_900.0,), withholding_runtime="tax/us-federal@1",
    corporate_action_runtime="ca/none@1",
    corporate_action_snapshot="actions@2026-06",
    policy_kind="SELL_ALL_AND_DIVERSIFY", blackout_policy=(),
    execution_lag=1, transaction_cost_model="bps@10",
    fractional_share_policy="FRACTIONAL_ALLOWED",
    employer_asset="ACME", target_cap=0.20,
    denominator_scope=("settled holdings", "settled cash"),
    valuation_session="2026-03-11", included_assets_policy="ALL_SETTLED",
    missing_price_policy="REFUSE", rounding_policy="WHOLE_SHARES_UP",
    allocation_policy="FIXED_TARGET", target_assets=("VTI", "BND"),
    target_weights={"VTI": 0.6, "BND": 0.4},
    funding_scope="SOURCE_PROCEEDS_ONLY", cash_reserve=0.0,
    purchase_cost_model="bps@10",
    account_runtime="account/taxable@1", tax_runtime="tax/us-federal@1",
    calendar_runtime="nyse@1", market_data_runtime="prices@2026-03-31",
    evaluation_period=("2026-03-02", "2026-04-30"))


class TestTheProfileIsOneIdentity:

    def test_identical_profiles_share_a_hash(self):
        assert BASE.compatibility_hash == replace(BASE).compatibility_hash

    def test_any_change_moves_the_hash(self):
        assert replace(BASE, target_cap=0.25).compatibility_hash \
            != BASE.compatibility_hash

    def test_the_version_travels_with_it(self):
        assert BASE.to_json()["profile_version"] == PROFILE_VERSION

    def test_every_group_is_represented(self):
        for group in (VEST_FLOW, DISPOSITION, CONCENTRATION, ALLOCATION,
                      ENVIRONMENT):
            assert set(group) <= set(ALL_FIELDS)

    def test_an_unset_field_is_reported_as_unevaluated(self):
        assert "corporate_action_runtime" in replace(
            BASE, corporate_action_runtime=None).unevaluated()


class TestEveryFieldIsLoadBearing:
    """The mutation test. Omitting a field from the equality check must be
    detectable, or the list can look complete while missing what matters."""

    def test_the_equality_list_covers_every_declared_field(self):
        """Guards the guard.

        The mutation tests below are parametrized over `ALL_FIELDS`, so
        deleting a name from one of the group tuples silently deletes its test
        cases rather than failing anything — 85 passing becomes 82 passing and
        nothing objects. This compares the list against the dataclass itself,
        so a field that exists on the profile and not in the equality check is
        a failure rather than an absence.
        """
        from dataclasses import fields

        declared = {f.name for f in fields(RSUComparisonProfile)} - {
            "profile_version"}
        assert declared == set(ALL_FIELDS), (
            "these profile fields are not in the equality check: "
            f"{sorted(declared - set(ALL_FIELDS))}")

    def test_every_field_belongs_to_exactly_one_group(self):
        from src.comparison.rsu_profile import GROUP_OF

        assert set(GROUP_OF) == set(ALL_FIELDS)
        assert len(ALL_FIELDS) == len(set(ALL_FIELDS))

    @pytest.mark.parametrize("field", ALL_FIELDS)
    def test_changing_it_defeats_an_otherwise_identical_comparison(self, field):
        original = getattr(BASE, field)
        mutated = _perturb(original)
        other = replace(BASE, **{field: mutated})

        verdict = classify(BASE, other, benchmark_id="b")
        assert verdict.status is not BenchmarkStatus.COMPARABLE, field
        assert not verdict.attribution_isolated, field

    @pytest.mark.parametrize("field", CONCENTRATION)
    def test_each_concentration_field_participates(self, field):
        """The seven fields that make a state-dependent sale reproducible."""
        other = replace(BASE, **{field: _perturb(getattr(BASE, field))})
        verdict = classify(BASE, other, benchmark_id="b")
        assert field in verdict.differing_dimensions

    @pytest.mark.parametrize("field", ALL_FIELDS)
    def test_leaving_it_unchecked_defeats_isolation(self, field):
        """Absent on either side is NOT_EVALUATED, never equal."""
        other = replace(BASE, **{field: None})
        verdict = classify(BASE, other, benchmark_id="b")
        assert not verdict.attribution_isolated, field
        assert field in verdict.unchecked_dimensions

    @pytest.mark.parametrize("field", ALL_FIELDS)
    def test_absent_on_both_sides_is_still_unchecked(self, field):
        """Two absences are not an agreement.

        This is the defect that forced classifier @2: two empty hashes compared
        equal, and a stored verdict claimed a dimension had been checked and
        matched when nothing had been checked at all. Setting the field to None
        on one side only never exercises it.
        """
        left = replace(BASE, **{field: None})
        right = replace(BASE, **{field: None})
        verdict = classify(left, right, benchmark_id="b")

        assert field in verdict.unchecked_dimensions, field
        assert field not in verdict.matched_dimensions, field
        assert not verdict.attribution_isolated, field


def _perturb(value):
    if isinstance(value, str):
        return value + "-changed"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, dict):
        return {**value, "__extra": 1.0}
    if isinstance(value, (list, tuple)):
        return tuple(value) + ("changed",)
    return "changed"


class TestStrategyEffectRequiresEverythingElseMatch:

    def test_hold_versus_diversify_isolates_the_named_dimensions(self):
        holding = replace(BASE, policy_kind="HOLD",
                          allocation_policy="HOLD_CASH",
                          target_assets=(), target_weights={})
        verdict = classify(BASE, holding, benchmark_id="hold",
                           isolating=("policy_kind", "allocation_policy",
                                      "target_assets", "target_weights"))
        assert verdict.status is BenchmarkStatus.COMPARABLE
        assert verdict.attribution_isolated
        assert "policy_kind" in verdict.isolates

    def test_a_different_fractional_share_policy_is_not_strategy(self):
        """Identical flows, and a different sale quantity."""
        other = replace(BASE, policy_kind="HOLD",
                        fractional_share_policy="WHOLE_SHARES_UP")
        verdict = classify(BASE, other, benchmark_id="b",
                           isolating=("policy_kind",))
        assert verdict.status is BenchmarkStatus.INCOMPARABLE
        assert "fractional_share_policy" in verdict.differing_dimensions

    def test_different_transaction_costs_do_not_isolate_strategy(self):
        other = replace(BASE, policy_kind="HOLD",
                        transaction_cost_model="bps@25")
        assert not classify(BASE, other, benchmark_id="b",
                            isolating=("policy_kind",)).attribution_isolated

    def test_a_different_denominator_scope_defeats_concentration_attribution(
            self):
        other = replace(BASE, target_cap=0.25,
                        denominator_scope=("settled holdings",))
        verdict = classify(BASE, other, benchmark_id="b",
                           isolating=("target_cap",))
        assert verdict.status is BenchmarkStatus.INCOMPARABLE
        assert "denominator_scope" in verdict.differing_dimensions

    def test_a_different_market_data_snapshot_is_not_isolated(self):
        other = replace(BASE, policy_kind="HOLD",
                        market_data_runtime="prices@2026-04-30")
        assert not classify(BASE, other, benchmark_id="b",
                            isolating=("policy_kind",)).attribution_isolated

    def test_a_missing_corporate_action_pin_is_never_silently_equal(self):
        other = replace(BASE, policy_kind="HOLD",
                        corporate_action_runtime=None)
        verdict = classify(BASE, other, benchmark_id="b",
                           isolating=("policy_kind",))
        assert verdict.status is \
            BenchmarkStatus.COMPARABLE_WITH_UNCHECKED_DIMENSIONS
        assert "corporate_action_runtime" in verdict.unchecked_dimensions
        assert not verdict.attribution_isolated


class TestPersonalOutcome:

    def test_different_vest_dates_combine_flows_and_strategy(self):
        other = replace(BASE, vest_dates=("2026-06-01",), policy_kind="HOLD")
        verdict = classify(BASE, other, benchmark_id="b",
                           isolating=("policy_kind",))
        assert verdict.status is BenchmarkStatus.PERSONAL_OUTCOME
        assert not verdict.attribution_isolated

    def test_it_says_it_isolates_neither(self):
        other = replace(BASE, delivered_values=(7_800.0,))
        assert "isolates neither" in classify(
            BASE, other, benchmark_id="b").reason


class TestUncheckedDimensionsDefeatIsolation:

    def test_the_wording_names_what_was_not_established(self):
        other = replace(BASE, policy_kind="HOLD", tax_runtime=None)
        verdict = classify(BASE, other, benchmark_id="b",
                           isolating=("policy_kind",))
        assert "not isolated" in verdict.reason
        assert "tax_runtime" in verdict.reason

    def test_the_comparison_is_still_shown(self):
        """Weaker than COMPARABLE, and a very different thing from
        INCOMPARABLE."""
        other = replace(BASE, policy_kind="HOLD", tax_runtime=None)
        verdict = classify(BASE, other, benchmark_id="b",
                           isolating=("policy_kind",))
        assert verdict.status is not BenchmarkStatus.INCOMPARABLE


class TestNoHiddenOmission:

    def test_every_requested_benchmark_produces_a_row(self):
        rows = evaluate(BASE, {
            "hold": {"profile": replace(BASE, policy_kind="HOLD"),
                     "isolating": ("policy_kind",)},
            "different_costs": {"profile": replace(
                BASE, transaction_cost_model="bps@50")},
            "unsupported": {"reason": "value-matched mode not built"},
        })
        assert [row.benchmark_id for row in rows] == [
            "hold", "different_costs", "unsupported"]

    def test_an_unsupported_benchmark_persists_with_its_reason(self):
        [row] = evaluate(BASE, {"x": {"reason": "no price series"}})
        assert row.status is BenchmarkStatus.NOT_EVALUATED
        assert row.requested is True
        assert "no price series" in row.reason

    def test_not_evaluated_is_distinct_from_incomparable(self):
        """One was judged; the other never happened."""
        rows = {r.benchmark_id: r for r in evaluate(BASE, {
            "never_ran": {"reason": "not built"},
            "judged": {"profile": replace(BASE, calendar_runtime="crypto@1")},
        })}
        assert rows["never_ran"].status is BenchmarkStatus.NOT_EVALUATED
        assert rows["judged"].status is BenchmarkStatus.INCOMPARABLE

    def test_the_profile_hash_travels_on_every_row(self):
        rows = evaluate(BASE, {"a": {"reason": "x"},
                               "b": {"profile": replace(BASE)}})
        assert all(row.comparison_profile_hash == BASE.compatibility_hash
                   for row in rows)

    def test_the_classifier_version_travels_on_every_row(self):
        [row] = evaluate(BASE, {"a": {"profile": replace(BASE)}})
        assert row.classifier_version == PROFILE_VERSION

    def test_the_flow_mode_is_recorded(self):
        [row] = evaluate(BASE, {"a": {"profile": replace(BASE, policy_kind="HOLD"),
                                      "isolating": ("policy_kind",),
                                      "flow_mode": "IN_KIND_HOLD"}})
        assert row.benchmark_flow_mode == "IN_KIND_HOLD"


class TestTheVerdictNeverSeesAResult:

    def test_classify_takes_no_outcome(self):
        """A verdict computed from outcomes is a verdict about which answer is
        convenient."""
        import inspect

        signature = inspect.signature(classify)
        for name in ("result", "outcome", "returns", "performance", "value"):
            assert name not in signature.parameters
