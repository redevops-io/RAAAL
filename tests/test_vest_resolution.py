"""A vest cannot reach withholding without passing through its pinned runtime.

Before this, `corporate_action_ref` was a required string and nothing checked
that the quantity had been through the runtime it named. The system could say
"corporate-action runtime pinned" while the declared grant-date share count went
straight to withholding — declared semantics without executed behaviour.

**The discriminating fixture must survive the real engine.** Whole-share
withholding distinguishes the orders arithmetically, but this engine withholds
fractionally, and a plain percentage commutes with a split: 101 shares through a
two-for-one gives 157.56 delivered either way. The supplemental threshold does
not commute, because it is non-linear in *value* — and the wrong order
over-delivers, which is the direction that flatters.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from src.runtime.corporate_action import (
    US_CORPORATE_ACTIONS,
    CorporateActionEvent,
    CorporateActionKind,
    RealizedCorporateActions,
    UnresolvedCorporateAction,
    UnsupportedCorporateAction,
)
from src.runtime.rsu import (
    UnpinnedVest,
    VestEvent,
    WithholdingMethod,
    apply_supplemental_wage_threshold,
    apply_vest_delivery,
    in_kind_flow_for,
    resolve_for_vest,
    vest_accounting,
)

ISSUER = "issuer/acme"
SNAPSHOT = "actions@2026-06"

#: Chosen so the adjusted vest value crosses the statutory threshold and the
#: unadjusted one does not. At $6,000 a share, 101 is $606,000 and 202 is
#: $1,212,000.
THRESHOLD_PRICE = 6_000.0


def vest(**overrides) -> VestEvent:
    base = dict(grant_id="g1", employer_ticker="ACME", vest_date="2026-06-15",
                gross_shares=101.0, vest_price_source="p",
                withholding_rate=0.22,
                withholding_method=WithholdingMethod.SHARE_WITHHOLDING,
                market_data_ref="md@1", corporate_action_ref=SNAPSHOT)
    base.update(overrides)
    return VestEvent(**base)


def action(kind, **overrides) -> CorporateActionEvent:
    base = dict(issuer_ref=ISSUER, effective_date="2026-04-01", kind=kind,
                source_ref="vendor/2026-04")
    base.update(overrides)
    return CorporateActionEvent(**base)


def split(numerator=2, denominator=1):
    return action(CorporateActionKind.SPLIT,
                  ratio_numerator=Decimal(numerator),
                  ratio_denominator=Decimal(denominator))


def history(*events, snapshot=SNAPSHOT):
    return RealizedCorporateActions(snapshot_ref=snapshot, events=events)


def resolve(event=None, *, granted=101.0, realized=None):
    return resolve_for_vest(event or vest(), granted_shares=granted,
                            issuer_ref=ISSUER,
                            realized=realized if realized is not None
                            else history(split()),
                            runtime=US_CORPORATE_ACTIONS)


class TestTheDiscriminatingFixtureSurvivesTheRealEngine:

    def test_fractional_withholding_commutes_with_a_split(self):
        """Recorded so nobody adopts this as the ordering test.

        At a price below the threshold both orders deliver the same count, and
        a test built on it passes against a bypassed resolution.
        """
        adjusted = 202.0 * (1 - 0.22)
        raw_then_split = 101.0 * (1 - 0.22) * 2
        assert adjusted == pytest.approx(raw_then_split) == pytest.approx(157.56)

    def test_the_threshold_does_not_commute(self):
        def delivered(shares):
            withheld = apply_supplemental_wage_threshold(
                shares * THRESHOLD_PRICE, rate=0.22)
            return shares - withheld / THRESHOLD_PRICE

        right = delivered(202.0)
        wrong = delivered(101.0) * 2
        assert right == pytest.approx(152.26, abs=0.01)
        assert wrong == pytest.approx(157.56, abs=0.01)
        assert wrong > right, "the wrong order must over-deliver"

    def test_the_engine_uses_the_adjusted_quantity(self):
        accounting = vest_accounting(vest(), vest_price=THRESHOLD_PRICE,
                                     resolved=resolve())
        assert accounting["adjusted_gross_shares"] == 202.0
        assert accounting["granted_shares"] == 101.0
        assert accounting["shares_delivered"] == pytest.approx(152.26, abs=0.01)

    def test_bypassing_resolution_would_change_the_answer(self):
        """The mutation this whole slice exists to prevent, stated as a fact
        about the numbers rather than about the code."""
        unadjusted = resolve(realized=history())
        assert unadjusted.adjusted_quantity == 101

        bypassed = vest_accounting(vest(), vest_price=THRESHOLD_PRICE,
                                   resolved=unadjusted)
        assert bypassed["shares_delivered"] != pytest.approx(152.26, abs=0.01)
        assert bypassed["shares_delivered"] == pytest.approx(78.78, abs=0.01)


class TestRawQuantityCannotBypassResolution:

    def test_vest_accounting_requires_a_resolved_grant(self):
        with pytest.raises(UnpinnedVest, match="corporate-action runtime"):
            vest_accounting(vest(), vest_price=50.0)

    def test_delivery_requires_a_resolved_grant(self):
        with pytest.raises(UnpinnedVest, match="corporate-action runtime"):
            apply_vest_delivery(vest(), vest_price=50.0)

    def test_the_in_kind_flow_requires_one_too(self):
        with pytest.raises(UnpinnedVest):
            in_kind_flow_for(vest(), vest_price=50.0)

    def test_the_declared_count_is_not_used_even_when_present(self):
        """`VestEvent.gross_shares` says 101; the resolved grant says 202."""
        accounting = vest_accounting(vest(), vest_price=50.0,
                                     resolved=resolve())
        assert vest().gross_shares == 101.0
        assert accounting["gross_shares"] == 202.0


class TestRefusalsBeforeArithmetic:

    def test_a_missing_snapshot_refuses(self):
        with pytest.raises(UnresolvedCorporateAction, match="trusted blindly"):
            resolve_for_vest(vest(), granted_shares=101.0, issuer_ref=ISSUER,
                             realized=None, runtime=US_CORPORATE_ACTIONS)

    def test_a_snapshot_other_than_the_pinned_one_refuses(self):
        """A run declared one history and was handed another."""
        with pytest.raises(UnresolvedCorporateAction, match="pins corporate"):
            resolve(realized=history(split(), snapshot="actions@2026-07"))

    def test_an_unsupported_action_refuses(self):
        with pytest.raises(UnsupportedCorporateAction):
            resolve(realized=history(split(),
                                     action(CorporateActionKind.SPINOFF)))

    def test_a_cancelled_grant_produces_no_delivery(self):
        cancelled = resolve(realized=history(
            action(CorporateActionKind.GRANT_CANCELLED)))
        assert not cancelled.vests
        with pytest.raises(UnpinnedVest, match="does not vest"):
            vest_accounting(vest(), vest_price=50.0, resolved=cancelled)

    def test_a_replaced_grant_produces_no_delivery_under_its_old_identity(self):
        replaced = resolve(realized=history(
            action(CorporateActionKind.GRANT_REPLACED,
                   replacement_grant_ref="grant/g2")))
        assert replaced.replaced_by == "grant/g2"
        with pytest.raises(UnpinnedVest, match="does not vest"):
            vest_accounting(vest(), vest_price=50.0, resolved=replaced)

    def test_no_partial_result_survives_a_refusal(self):
        with pytest.raises(UnpinnedVest):
            vest_accounting(vest(), vest_price=50.0,
                            resolved=resolve(realized=history(
                                action(CorporateActionKind.GRANT_CANCELLED))))


class TestResolvedIdentityReachesTheHolding:

    def test_a_symbol_change_moves_the_delivered_asset(self):
        renamed = resolve(realized=history(
            action(CorporateActionKind.SYMBOL_CHANGE, new_symbol="XYZ")))
        grant, _ = apply_vest_delivery(vest(), vest_price=50.0,
                                       resolved=renamed)
        assert grant.ticker == "XYZ"

    def test_the_in_kind_flow_uses_the_resolved_symbol(self):
        renamed = resolve(realized=history(
            action(CorporateActionKind.SYMBOL_CHANGE, new_symbol="XYZ")))
        flow, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=renamed)
        assert flow.asset == "XYZ"

    def test_a_symbol_change_creates_no_trade(self):
        renamed = resolve(realized=history(
            action(CorporateActionKind.SYMBOL_CHANGE, new_symbol="XYZ")))
        flow, _ = in_kind_flow_for(vest(), vest_price=50.0, resolved=renamed)
        assert flow.quantity > 0
        assert flow.source_ref.startswith("vest:")


class TestProvenanceReachesTheResult:

    def test_the_snapshot_and_runtime_travel_with_the_accounting(self):
        accounting = vest_accounting(vest(), vest_price=50.0,
                                     resolved=resolve())
        assert accounting["corporate_action_snapshot_ref"] == SNAPSHOT
        assert accounting["corporate_action_runtime_ref"].endswith("@1")

    def test_the_applied_action_references_travel(self):
        accounting = vest_accounting(vest(), vest_price=50.0,
                                     resolved=resolve())
        assert accounting["corporate_actions_applied"] == ["SPLIT"]
        assert accounting["corporate_action_refs"] == ["vendor/2026-04"]

    def test_they_reach_the_result_context(self):
        from src.mission.rsu_result import build

        context = build(
            vest_accounting=vest_accounting(vest(), vest_price=50.0,
                                            resolved=resolve()),
            modelling_scope={"modelled": ("share delivery",)})
        rendered = context.to_json()["vest_accounting"]

        assert rendered["granted_shares"] == 101.0
        assert rendered["adjusted_gross_shares"] == 202.0
        assert rendered["corporate_action_snapshot_ref"] == SNAPSHOT
        assert rendered["corporate_action_refs"] == ["vendor/2026-04"]

    def test_a_run_with_no_actions_says_so_rather_than_staying_silent(self):
        accounting = vest_accounting(vest(), vest_price=50.0,
                                     resolved=resolve(realized=history()))
        assert accounting["corporate_actions_applied"] == []
        assert accounting["corporate_action_snapshot_ref"] == SNAPSHOT


class TestComparabilityChecksBothRuntimeAndSnapshot:

    def test_a_differing_snapshot_defeats_isolation(self):
        """Two runs can share an interpretation policy and receive different
        histories — one knowing about a split the other does not."""
        from dataclasses import replace as _replace

        from src.comparison.rsu_profile import BenchmarkStatus, classify
        from tests.test_rsu_comparability import BASE

        other = _replace(BASE, policy_kind="HOLD",
                         corporate_action_snapshot="actions@2026-07")
        verdict = classify(BASE, other, benchmark_id="b",
                           isolating=("policy_kind",))

        assert verdict.status is BenchmarkStatus.INCOMPARABLE
        assert "corporate_action_snapshot" in verdict.differing_dimensions

    def test_both_fields_are_in_the_profile(self):
        from src.comparison.rsu_profile import VEST_FLOW

        assert "corporate_action_runtime" in VEST_FLOW
        assert "corporate_action_snapshot" in VEST_FLOW


class TestRestatementsDoNotMoveOldRuns:

    def test_a_run_pinned_to_the_old_snapshot_keeps_its_quantity(self):
        original = resolve(realized=history(split()))
        assert original.adjusted_quantity == 202

        corrected = RealizedCorporateActions(
            snapshot_ref="actions@2026-07", events=(split(3, 1),),
            restates=SNAPSHOT)
        restated = resolve_for_vest(
            vest(corporate_action_ref="actions@2026-07"), granted_shares=101.0,
            issuer_ref=ISSUER, realized=corrected,
            runtime=US_CORPORATE_ACTIONS)

        assert restated.adjusted_quantity == pytest.approx(Decimal(303))
        assert original.adjusted_quantity == 202
        assert original.snapshot_ref != restated.snapshot_ref
