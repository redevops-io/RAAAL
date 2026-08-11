"""A real description, routed into the declaration and card path.

    user text -> parse -> template hint -> RSUDeclaration -> confirmation card

No declaration is constructed by hand here. The point of 11b is that the live
path builds one, so a test that builds its own would prove only that the types
fit together.
"""
from __future__ import annotations

import pytest

from src.mission.compiler import parse
from src.mission.rsu_declaration import (
    TEMPLATE_HINT,
    DeclarationVersionMismatch,
    DeclarationVersions,
    Destination,
    TemplateHandlerMissing,
    UnrepresentedRecognition,
    build_rsu_declaration,
    check_versions,
    handler_for,
)
from src.mission.rsu_recognize import RSURecognition, recognize
from src.runtime.rsu import US_SHARE_WITHHOLDING
from src.workspace.rsu_confirmation import build as build_card

DESCRIPTION = (
    "100 ACME shares vest quarterly. Withhold 22% in shares. "
    "Sell as soon as I can after the blackout window. "
    "Keep company stock below 20%. "
    "Allocate proceeds 60% VTI, 30% VXUS, 10% BND.")

REFS = {"corporate_action_ref": "ca/none@1",
        "tax_runtime_ref": "tax/us-federal@1",
        "account_runtime_ref": "account/taxable@1",
        "market_data_ref": "prices@2026-03-31",
        "account_destination": "TAXABLE"}

VERSIONS = DeclarationVersions(
    template_version="template/rsu-vesting@1",
    rsu_runtime_version="rsu-vesting/us-share-withholding@1",
    account_runtime_version="account/taxable@1",
    tax_runtime_version="tax/us-federal@1",
    corporate_action_runtime_version="ca/none@1",
    scope_schema_version="rsu-result-context@1")


def declared(text=DESCRIPTION, **kwargs):
    return build_rsu_declaration(parse(text), versions=VERSIONS,
                                 runtime_refs={**REFS, **kwargs})


class TestTheRouteBranchesOnTheHint:

    def test_vest_language_reaches_the_rsu_builder(self):
        assert parse(DESCRIPTION).template_hint == TEMPLATE_HINT
        assert handler_for(TEMPLATE_HINT) is build_rsu_declaration

    def test_an_unknown_hint_fails_closed(self):
        """Generic compilation is not a fallback: it would read a vest as cash
        arriving and then a purchase."""
        with pytest.raises(TemplateHandlerMissing,
                           match="TEMPLATE_HANDLER_MISSING"):
            handler_for("mortgage-refinance")

    def test_no_hint_fails_closed(self):
        with pytest.raises(TemplateHandlerMissing):
            handler_for(None)

    def test_the_builder_refuses_a_non_rsu_parse(self):
        with pytest.raises(TemplateHandlerMissing):
            build_rsu_declaration(parse("I put $500 into SPY every month"))


class TestEveryRecognisedFieldReachesTheDeclaration:

    def test_the_employer_is_the_vested_stock_not_an_allocation_target(self):
        """Naming the wrong employer would measure concentration against the
        wrong holding."""
        assert declared().employer_ticker == "ACME"

    def test_the_share_count_is_read(self):
        assert declared().gross_shares == 100.0

    def test_the_withholding_rate_is_read_as_a_fraction(self):
        assert declared().withholding_rate == pytest.approx(0.22)

    def test_the_withholding_method_is_read_from_in_shares(self):
        assert declared().withholding_method == "SHARE_WITHHOLDING"

    def test_selling_at_the_first_opportunity_is_read(self):
        assert declared().disposition_policy == "SELL_ALL_AND_DIVERSIFY"

    def test_the_cap_is_read_as_a_fraction(self):
        assert declared().concentration_cap == pytest.approx(0.20)

    def test_the_allocation_weights_are_read(self):
        assert declared().allocation_policy == pytest.approx(
            {"VTI": 0.6, "VXUS": 0.3, "BND": 0.1})

    def test_the_withholding_rate_does_not_become_an_allocation_weight(self):
        """Scanned across the sentence rather than the allocation clause, the
        withholding rate becomes a weight on whichever ticker follows it.

        The employer must be named beside the rate for this to discriminate:
        with "22% in shares" the pattern finds no ticker either way, and the
        test would pass against a scan that reads the whole sentence.
        """
        one = declared(
            "100 ACME shares vest quarterly. Withhold 22% in ACME shares. "
            "Sell as soon as I can. "
            "Allocate proceeds 60% VTI, 30% VXUS, 10% BND.")
        assert "ACME" not in one.allocation_policy
        assert one.allocation_policy == pytest.approx(
            {"VTI": 0.6, "VXUS": 0.3, "BND": 0.1})
        assert sum(one.allocation_policy.values()) == pytest.approx(1.0)

    def test_the_cadence_reaches_the_vest_schedule(self):
        assert declared().vest_schedule == ("quarterly",)


class TestNoRecognitionIsDropped:
    """The live form of "no line is only copy"."""

    def test_an_unrepresentable_recognition_raises(self):
        with pytest.raises(UnrepresentedRecognition, match="nowhere to put it"):
            build_rsu_declaration(
                parse(DESCRIPTION),
                recognitions=[RSURecognition("dividend_reinvestment", True,
                                             "reinvest dividends")])

    def test_every_real_recognition_is_representable(self):
        parsed = parse(DESCRIPTION)
        build_rsu_declaration(parsed, recognitions=recognize(
            parsed.text, assets=parsed.assets), runtime_refs=REFS)


class TestIncompleteDescriptionsStayUnresolved:

    def test_a_vest_with_no_quantity_leaves_shares_unresolved(self):
        one = declared("My ACME RSUs vest quarterly and I hold them")
        assert one.gross_shares is None
        assert "gross_shares" in one.unresolved()

    def test_two_possible_employers_leaves_it_unresolved(self):
        """Guessing would measure concentration against the wrong holding."""
        one = declared("My ACME and BETA shares vest quarterly")
        assert one.employer_ticker is None

    def test_concentration_language_with_no_threshold_stays_unresolved(self):
        one = declared("My ACME RSUs vest quarterly. Reduce my concentration.")
        assert one.concentration_cap is None

    def test_ambiguous_withholding_stays_unresolved(self):
        one = declared("My ACME RSUs vest quarterly and tax is withheld")
        assert one.withholding_method is None

    def test_a_blackout_is_recognised_without_inventing_dates(self):
        """"After earnings" names a window whose dates the text does not give,
        and inventing them would schedule a sale on a day nobody described."""
        assert declared().blackout_schedule == ()

    def test_no_blackout_language_leaves_it_unresolved(self):
        one = declared("100 ACME shares vest quarterly and I hold them")
        assert one.blackout_schedule is None


class TestTheCardRendersFromTheBuiltDeclaration:

    def card(self, **kwargs):
        return build_card(declared(**kwargs), runtime=US_SHARE_WITHHOLDING)

    def test_it_renders(self):
        assert self.card().described

    def test_the_described_fields_carry_what_the_user_wrote(self):
        values = {one.field_name: one.value for one in self.card().described}
        assert values["employer_ticker"] == "ACME"
        assert values["withholding_rate"] == "22%"
        assert values["concentration_cap"] == "20%"

    def test_every_card_field_is_typed(self):
        for line in self.card().all_fields:
            assert isinstance(line.destination, Destination)

    def test_unresolved_fields_appear_as_questions(self):
        names = {one.field_name for one in self.card().unresolved}
        assert "grant_identity" in names

    def test_an_unpinned_corporate_action_blocks(self):
        blocked = build_card(
            build_rsu_declaration(
                parse(DESCRIPTION), versions=VERSIONS,
                runtime_refs={k: v for k, v in REFS.items()
                              if k != "corporate_action_ref"}),
            runtime=US_SHARE_WITHHOLDING)
        assert "corporate_action_ref" in blocked.blocking
        assert not blocked.can_run

    def test_a_complete_description_can_run(self):
        assert self.card().can_run

    def test_no_computed_value_appears(self):
        rendered = self.card().to_json()
        names = {line["field"] for group in ("described", "inferred",
                                             "unresolved")
                 for line in rendered[group]}
        for computed in ("delivered_shares", "proceeds",
                         "projected_concentration"):
            assert computed not in names


class TestAllocationWeightsAreCheckedBeforeExecution:

    def test_weights_that_do_not_sum_are_refused_by_the_compiler(self):
        from src.runtime.allocation import UnsupportedAllocation, compile_policy

        one = declared("100 ACME shares vest quarterly. "
                       "Allocate proceeds 60% VTI, 30% VXUS.")
        with pytest.raises(UnsupportedAllocation, match="sum"):
            compile_policy(one.allocation_policy)

    def test_summing_weights_compile(self):
        from src.runtime.allocation import compile_policy

        assert compile_policy(declared().allocation_policy) == pytest.approx(
            {"VTI": 0.6, "VXUS": 0.3, "BND": 0.1})


class TestTheFullJourney:
    """parse -> declaration -> card -> confirm -> run -> stored -> worksheet."""

    def test_it_completes_without_a_hand_built_declaration(self, tmp_path):
        import pandas as pd

        from src.workspace.rsu_view import ContextState, RSUWorksheetView
        from tests.test_rsu_result_live import messy_run, store_with_plan

        parsed = parse(DESCRIPTION)
        assert parsed.template_hint == TEMPLATE_HINT

        declaration = handler_for(parsed.template_hint)(
            parsed, versions=VERSIONS, runtime_refs=REFS)
        card = build_card(declaration, runtime=US_SHARE_WITHHOLDING)
        assert card.can_run
        confirmed = declaration.versions

        # Execution accepts only the pins the user confirmed under.
        check_versions(confirmed, VERSIONS)

        store = store_with_plan(tmp_path)
        result = messy_run(pd.bdate_range("2026-03-02", "2026-04-30"))
        store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                         result=result.to_json(), comparison={})

        view = RSUWorksheetView.from_result(
            store.get_run("r1", "pilot")["result"])
        assert view.context_state is ContextState.PRESENT
        assert view.vest_accounting.delivered_value is not None

    def test_a_moved_version_between_confirming_and_running_refuses(self):
        from dataclasses import replace

        declaration = declared()
        moved = replace(VERSIONS, rsu_runtime_version="rsu-vesting/us@2")
        with pytest.raises(DeclarationVersionMismatch):
            check_versions(declaration.versions, moved)
