"""What happened to a grant between grant date and vest date.

The load-bearing claim is ordering: adjustment precedes withholding. The obvious
fixture does not prove it — a plain percentage commutes with a split, so 100
shares at 22% gives 156 delivered whichever order you use. The permanent test
uses whole-share withholding, where the same grant of 101 gives 157 in the right
order and 156 in the wrong one.
"""
from __future__ import annotations

import math
from decimal import Decimal

import pytest

from src.runtime.corporate_action import (
    IMPLEMENTED,
    SUPPORTED,
    UNSUPPORTED,
    US_CORPORATE_ACTIONS,
    CorporateActionEvent,
    CorporateActionKind,
    CorporateActionRuntime,
    FractionalPolicy,
    GrantCancelled,
    RealizedCorporateActions,
    UnresolvedCorporateAction,
    UnsupportedCorporateAction,
    resolve_grant,
)

ISSUER = "issuer/acme"


def event(kind, **overrides) -> CorporateActionEvent:
    base = dict(issuer_ref=ISSUER, effective_date="2026-04-01", kind=kind,
                source_ref="vendor/actions@2026-06", observed_at="2026-04-02")
    base.update(overrides)
    return CorporateActionEvent(**base)


def split(numerator=2, denominator=1, **overrides):
    return event(CorporateActionKind.SPLIT,
                 ratio_numerator=Decimal(numerator),
                 ratio_denominator=Decimal(denominator), **overrides)


def realized(*events, snapshot="actions@2026-06") -> RealizedCorporateActions:
    return RealizedCorporateActions(snapshot_ref=snapshot, events=events)


def resolve(granted, *events, runtime=US_CORPORATE_ACTIONS,
            vest_date="2026-06-15", symbol="ACME"):
    return resolve_grant(grant_ref="grant/g1", granted_shares=Decimal(granted),
                         symbol=symbol, issuer_ref=ISSUER, vest_date=vest_date,
                         realized=realized(*events), runtime=runtime)


def whole_share_withholding(gross: Decimal, rate=0.22) -> tuple:
    """Withheld and delivered, rounding withheld up to whole shares."""
    withheld = Decimal(math.ceil(float(gross) * rate))
    return withheld, gross - withheld


class TestAdjustmentPrecedesWithholding:
    """The permanent ordering test."""

    def test_the_obvious_fixture_does_not_discriminate(self):
        """Recorded so nobody replaces the real test with this one.

        A plain percentage commutes with a scalar split, so both orders give
        156 and the test would pass against the wrong implementation.
        """
        right = Decimal(100) * 2 * Decimal("0.78")
        wrong = Decimal(100) * Decimal("0.78") * 2
        assert right == wrong == 156

    def test_whole_share_withholding_breaks_the_commutativity(self):
        adjusted = resolve(101, split()).gross_shares
        assert adjusted == 202

        _, delivered_right = whole_share_withholding(adjusted)
        _, delivered_pre = whole_share_withholding(Decimal(101))
        delivered_wrong = delivered_pre * 2

        assert delivered_right == 157
        assert delivered_wrong == 156
        assert delivered_right != delivered_wrong

    def test_the_runtime_adjusts_before_returning_a_quantity(self):
        """`resolve_grant` hands back the adjusted gross, so withholding
        applied to its output is applied to the adjusted count."""
        assert resolve(101, split()).gross_shares == 202

    def test_a_three_for_two_split_is_exact(self):
        """Held as 1.5 in binary floating point, a share count becomes one
        nobody can reproduce."""
        assert resolve(100, split(3, 2)).gross_shares == Decimal(150)


class TestSplits:

    def test_a_two_for_one_doubles(self):
        assert resolve(100, split()).gross_shares == 200

    def test_a_reverse_split_divides(self):
        reverse = event(CorporateActionKind.REVERSE_SPLIT,
                        ratio_numerator=Decimal(1),
                        ratio_denominator=Decimal(10))
        assert resolve(100, reverse).gross_shares == 10

    def test_a_split_with_no_ratio_is_refused(self):
        with pytest.raises(UnresolvedCorporateAction, match="no ratio"):
            resolve(100, event(CorporateActionKind.SPLIT))

    def test_actions_apply_in_effective_date_order(self):
        first = split(2, 1, effective_date="2026-02-01")
        second = split(3, 1, effective_date="2026-03-01")
        assert resolve(10, second, first).gross_shares == 60


class TestFractionalTreatment:

    def test_an_unresolved_policy_refuses_rather_than_rounding(self):
        """Rounding in either direction is a decision nobody made."""
        runtime = CorporateActionRuntime(
            name="x", version=1, fractional_policy=FractionalPolicy.UNRESOLVED)
        reverse = event(CorporateActionKind.REVERSE_SPLIT,
                        ratio_numerator=Decimal(1),
                        ratio_denominator=Decimal(3))
        with pytest.raises(UnresolvedCorporateAction, match="no fractional"):
            resolve(100, reverse, runtime=runtime)

    def test_cash_in_lieu_records_the_fraction_explicitly(self):
        reverse = event(CorporateActionKind.REVERSE_SPLIT,
                        ratio_numerator=Decimal(1),
                        ratio_denominator=Decimal(3))
        result = resolve(100, reverse)
        assert result.gross_shares == 33
        assert result.cash_in_lieu > 0
        assert result.fractional_shares > 0

    def test_retaining_fractions_keeps_them(self):
        runtime = CorporateActionRuntime(
            name="x", version=1, fractional_policy=FractionalPolicy.RETAIN)
        reverse = event(CorporateActionKind.REVERSE_SPLIT,
                        ratio_numerator=Decimal(1),
                        ratio_denominator=Decimal(3))
        result = resolve(100, reverse, runtime=runtime)
        assert result.gross_shares != result.gross_shares.to_integral_value()

    def test_nothing_is_truncated_silently(self):
        reverse = event(CorporateActionKind.REVERSE_SPLIT,
                        ratio_numerator=Decimal(1),
                        ratio_denominator=Decimal(3))
        result = resolve(100, reverse)
        assert result.fractional_shares + result.gross_shares == \
            Decimal(100) / Decimal(3)


class TestSymbolChangeIsNotATrade:

    def test_the_grant_carries_forward(self):
        result = resolve(100, event(CorporateActionKind.SYMBOL_CHANGE,
                                    old_symbol="ACME", new_symbol="XYZ"))
        assert result.symbol == "XYZ"
        assert result.gross_shares == 100

    def test_the_grant_reference_is_unchanged(self):
        result = resolve(100, event(CorporateActionKind.SYMBOL_CHANGE,
                                    new_symbol="XYZ"))
        assert result.grant_ref == "grant/g1"

    def test_it_creates_no_new_grant_and_no_flow(self):
        """Not a sale, not a purchase, not a new grant."""
        result = resolve(100, event(CorporateActionKind.SYMBOL_CHANGE,
                                    new_symbol="XYZ"))
        assert not result.cancelled
        assert result.replaced_by is None
        assert result.cash_in_lieu == 0

    def test_a_change_with_no_new_symbol_is_refused(self):
        with pytest.raises(UnresolvedCorporateAction, match="no new symbol"):
            resolve(100, event(CorporateActionKind.SYMBOL_CHANGE))


class TestMergers:

    def test_stock_for_stock_converts_at_the_stated_ratio(self):
        merger = event(CorporateActionKind.MERGER_STOCK_FOR_STOCK,
                       ratio_numerator=Decimal(1), ratio_denominator=Decimal(2),
                       replacement_security="BIGCO")
        result = resolve(100, merger)
        assert result.gross_shares == 50
        assert result.symbol == "BIGCO"

    def test_a_conversion_without_a_ratio_is_refused(self):
        merger = event(CorporateActionKind.MERGER_STOCK_FOR_STOCK,
                       replacement_security="BIGCO")
        with pytest.raises(UnresolvedCorporateAction, match="conversion ratio"):
            resolve(100, merger)

    def test_cash_only_is_blocked(self):
        with pytest.raises(UnsupportedCorporateAction, match="cash-only"):
            resolve(100, event(CorporateActionKind.MERGER_CASH_ONLY))

    def test_mixed_consideration_is_blocked(self):
        with pytest.raises(UnsupportedCorporateAction, match="mixed"):
            resolve(100, event(CorporateActionKind.MERGER_MIXED))

    def test_a_stock_merger_carrying_cash_is_blocked(self):
        merger = event(CorporateActionKind.MERGER_STOCK_FOR_STOCK,
                       ratio_numerator=Decimal(1), ratio_denominator=Decimal(2),
                       replacement_security="BIGCO",
                       cash_component=Decimal("5.00"))
        with pytest.raises(UnsupportedCorporateAction):
            resolve(100, merger)


class TestCancellationAndReplacement:

    def test_a_cancelled_grant_does_not_vest(self):
        result = resolve(100, event(CorporateActionKind.GRANT_CANCELLED))
        assert result.cancelled
        assert not result.vests
        assert result.gross_shares == 0

    def test_it_stays_visible_rather_than_vanishing(self):
        result = resolve(100, event(CorporateActionKind.GRANT_CANCELLED))
        assert result.grant_ref == "grant/g1"
        assert "GRANT_CANCELLED" in result.applied

    def test_a_replacement_links_rather_than_mutating(self):
        result = resolve(100, event(CorporateActionKind.GRANT_REPLACED,
                                    replacement_grant_ref="grant/g2"))
        assert result.replaced_by == "grant/g2"
        assert result.grant_ref == "grant/g1"
        assert result.cancelled

    def test_the_replacement_does_not_inherit_the_original_identity(self):
        result = resolve(100, event(CorporateActionKind.GRANT_REPLACED,
                                    replacement_grant_ref="grant/g2"))
        assert result.replaced_by != result.grant_ref

    def test_a_replacement_with_no_target_is_refused(self):
        with pytest.raises(UnresolvedCorporateAction, match="replacement grant"):
            resolve(100, event(CorporateActionKind.GRANT_REPLACED))


class TestMissingHistoryBlocksBeforeArithmetic:

    def test_no_pinned_history_refuses(self):
        with pytest.raises(UnresolvedCorporateAction, match="trusted blindly"):
            resolve_grant(grant_ref="grant/g1", granted_shares=Decimal(100),
                          symbol="ACME", issuer_ref=ISSUER,
                          vest_date="2026-06-15", realized=None,
                          runtime=US_CORPORATE_ACTIONS)

    def test_an_empty_history_is_not_a_missing_one(self):
        """A snapshot that reports no actions is an answer. No snapshot is not."""
        assert resolve(100).gross_shares == 100

    def test_an_unsupported_action_blocks_before_any_quantity(self):
        with pytest.raises(UnsupportedCorporateAction):
            resolve(100, split(), event(CorporateActionKind.SPINOFF))


class TestPointInTime:

    def test_only_actions_effective_by_the_vest_date_apply(self):
        later = split(2, 1, effective_date="2026-09-01")
        assert resolve(100, later, vest_date="2026-06-15").gross_shares == 100

    def test_a_restatement_is_a_new_snapshot_naming_what_it_corrects(self):
        original = realized(split(), snapshot="actions@2026-06")
        corrected = RealizedCorporateActions(
            snapshot_ref="actions@2026-07", events=(split(3, 1),),
            restates="actions@2026-06")

        assert corrected.restates == original.snapshot_ref
        assert original.events != corrected.events

    def test_a_run_pinned_to_the_old_snapshot_is_unchanged(self):
        """A later correction cannot silently move an old result."""
        original = realized(split(), snapshot="actions@2026-06")
        pinned = resolve_grant(
            grant_ref="grant/g1", granted_shares=Decimal(100), symbol="ACME",
            issuer_ref=ISSUER, vest_date="2026-06-15", realized=original,
            runtime=US_CORPORATE_ACTIONS)
        assert pinned.gross_shares == 200

    def test_policy_and_history_are_separate_objects(self):
        """The runtime says how a split is read; the snapshot says which splits
        were known."""
        import dataclasses

        runtime_fields = {f.name
                          for f in dataclasses.fields(CorporateActionRuntime)}
        assert "events" not in runtime_fields
        assert "snapshot_ref" not in runtime_fields


class TestEveryActionKindIsClassified:
    """Derived from the enum, not from the classification lists.

    A new member fails until somebody classifies it, rather than silently
    joining whichever list the loop happens to read.
    """

    def test_every_kind_is_supported_or_explicitly_unsupported(self):
        classified = set(SUPPORTED) | set(UNSUPPORTED)
        missing = sorted(one.value for one in CorporateActionKind
                         if one not in classified)
        assert not missing, (
            f"these action kinds are neither implemented nor declared "
            f"unsupported: {missing}")

    def test_nothing_is_in_both_lists(self):
        assert not set(SUPPORTED) & set(UNSUPPORTED)

    @pytest.mark.parametrize("kind", list(UNSUPPORTED))
    def test_each_unsupported_kind_gives_a_reason(self, kind):
        assert len(UNSUPPORTED[kind]) > 30

    @pytest.mark.parametrize("kind", list(SUPPORTED))
    def test_each_supported_kind_names_a_real_callable(self, kind):
        import src.runtime.corporate_action as module

        assert callable(getattr(module, SUPPORTED[kind], None))

    def test_the_implemented_registry_resolves(self):
        import src.runtime.corporate_action as module

        for name in IMPLEMENTED:
            assert getattr(module, name, None) is not None or hasattr(
                RealizedCorporateActions, name), name


class TestTheRuntimeDeclaresWhatItDoes:

    def test_the_ordering_assumption_is_declared(self):
        [one] = [a for a in US_CORPORATE_ACTIONS.assumptions
                 if a.name == "adjustment-before-withholding"]
        assert one.realized_by == "resolve_grant"
        assert "156" in one.risk

    def test_every_assumption_is_realized(self):
        assert US_CORPORATE_ACTIONS.unrealized(IMPLEMENTED) == []

    def test_each_unsupported_action_becomes_a_limitation(self):
        names = {one.name for one in US_CORPORATE_ACTIONS.limitations}
        for kind in UNSUPPORTED:
            assert f"unsupported-{kind.value.lower()}" in names
