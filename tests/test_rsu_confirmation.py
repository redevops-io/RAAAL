"""What a run will and will not establish, said before it runs.

    declarations -> confirmation card        (before)
    stored result -> worksheet               (after)

Both are projections; the engine is the only layer that decides a number. The
one remaining way they can describe a run differently is version drift, which is
refused rather than refreshed.
"""
from __future__ import annotations

from dataclasses import fields, replace

import pytest

from src.mission.compiler import parse
from src.mission.rsu_declaration import (
    FORBIDDEN_ON_DECLARATION,
    TEMPLATE_HINT,
    DeclarationVersionMismatch,
    DeclarationVersions,
    Destination,
    RSUDeclaration,
    check_versions,
)
from src.runtime.rsu import US_SHARE_WITHHOLDING
from src.workspace.rsu_confirmation import build

VERSIONS = DeclarationVersions(
    template_version="template/rsu-vesting@1",
    rsu_runtime_version="rsu-vesting/us-share-withholding@1",
    account_runtime_version="account/taxable@1",
    tax_runtime_version="tax/us-federal@1",
    corporate_action_runtime_version="ca/none@1",
    scope_schema_version="rsu-result-context@1")


def declaration(**overrides) -> RSUDeclaration:
    base = dict(
        grant_identity="grant/g1", employer_ticker="ACME",
        vest_schedule=("2026-03-02", "2026-06-01"), gross_shares=100.0,
        withholding_method="SHARE_WITHHOLDING", withholding_rate=0.22,
        corporate_action_ref="ca/none@1",
        disposition_policy="SELL_ALL_AND_DIVERSIFY",
        blackout_schedule=(), allocation_policy={"VTI": 0.6, "BND": 0.4},
        concentration_cap=0.20, account_destination="TAXABLE",
        tax_runtime_ref="tax/us-federal@1",
        account_runtime_ref="account/taxable@1",
        market_data_ref="prices@2026-03-31", versions=VERSIONS)
    base.update(overrides)
    return RSUDeclaration(**base)


def card(**overrides):
    return build(declaration(**overrides), runtime=US_SHARE_WITHHOLDING)


class TestVestLanguageRoutesToTheTemplate:

    @pytest.mark.parametrize("text", [
        "My RSUs vest quarterly and I sell them",
        "I have restricted stock vesting next year",
        "100 shares of my equity grant vest each quarter",
    ])
    def test_it_hands_off_rather_than_improvising(self, text):
        assert parse(text).template_hint == TEMPLATE_HINT

    def test_ordinary_contribution_language_does_not(self):
        """A vest must never be read by the generic cash-flow path: it is not
        cash arriving and then a purchase."""
        assert parse("I put $500 into SPY every month").template_hint \
            != TEMPLATE_HINT


class TestTheDeclarationCarriesNoComputedOutput:

    @pytest.mark.parametrize("forbidden", FORBIDDEN_ON_DECLARATION)
    def test_no_execution_output_is_a_declaration_field(self, forbidden):
        """A confirmation screen showing a computed figure answers a question
        the user has not yet agreed to ask."""
        assert forbidden not in {f.name for f in fields(RSUDeclaration)}

    @pytest.mark.parametrize("forbidden", FORBIDDEN_ON_DECLARATION)
    def test_no_execution_output_reaches_the_card(self, forbidden):
        """Checked against field names, not rendered prose. "Where proceeds go"
        is a legitimate question about allocation policy; a *field* called
        `proceeds` would be a computed figure on a pre-run screen."""
        rendered = card().to_json()
        names = {line["field"] for group in
                 ("described", "inferred", "unresolved")
                 for line in rendered[group]}
        assert forbidden not in names

    def test_the_card_shows_no_figure_the_engine_has_not_produced(self):
        """Positively: every value on the card traces to a declaration."""
        declared = declaration()
        for line in card().all_fields:
            if line.destination is Destination.UNRESOLVED_QUESTION:
                continue
            assert getattr(declared, line.field_name) is not None


class TestEveryFieldHasADestination:

    def test_no_line_is_only_copy(self):
        """A line that reaches nothing is recognition without representation,
        at the confirmation layer."""
        for line in card().all_fields:
            assert isinstance(line.destination, Destination)
            assert line.field_name

    def test_every_declaration_field_appears_somewhere(self):
        declared = {f.name for f in fields(RSUDeclaration)} - {"versions"}
        shown = {line.field_name for line in card().all_fields}
        assert declared == shown

    def test_runtime_references_are_typed_as_declarations(self):
        [tax] = [one for one in card().described
                 if one.field_name == "tax_runtime_ref"]
        assert tax.destination is Destination.RUNTIME_DECLARATION

    def test_stated_values_are_typed_as_engine_inputs(self):
        [shares] = [one for one in card().described
                    if one.field_name == "gross_shares"]
        assert shares.destination is Destination.ENGINE_INPUT

    def test_an_inference_carries_its_reason_and_default_set(self):
        rendered = build(declaration(), runtime=US_SHARE_WITHHOLDING,
                         inferred={"withholding_rate": ("statutory default",)},
                         defaults_ref="compiler-defaults/rsu@1")
        [rate] = [one for one in rendered.inferred
                  if one.field_name == "withholding_rate"]
        assert rate.why == "statutory default"
        assert rate.defaults_ref == "compiler-defaults/rsu@1"


class TestUnresolvedFieldsAskRatherThanDefault:

    def test_an_absent_withholding_method_becomes_a_question(self):
        rendered = card(withholding_method=None)
        [question] = [one for one in rendered.unresolved
                      if one.field_name == "withholding_method"]
        assert question.destination is Destination.UNRESOLVED_QUESTION
        assert "deliver different share counts" in question.why

    def test_an_absent_withholding_method_blocks(self):
        assert not card(withholding_method=None).can_run

    def test_an_absent_corporate_action_reference_blocks(self):
        """Share counts cannot be trusted across a split without knowing."""
        rendered = card(corporate_action_ref=None)
        assert "corporate_action_ref" in rendered.blocking
        assert not rendered.can_run

    def test_concentration_language_with_no_threshold_stays_unresolved(self):
        rendered = card(concentration_cap=None)
        [question] = [one for one in rendered.unresolved
                      if one.field_name == "concentration_cap"]
        assert "What maximum share" in question.why

    def test_an_absent_blackout_schedule_does_not_block(self):
        """It changes when a sale executes, not whether the numbers mean
        anything."""
        assert card(blackout_schedule=None).can_run

    def test_nothing_is_silently_defaulted(self):
        rendered = card(withholding_method=None, disposition_policy=None)
        names = {one.field_name for one in rendered.unresolved}
        assert {"withholding_method", "disposition_policy"} <= names
        assert not any(one.field_name in names for one in rendered.described)


class TestTheCapIsDeclaredNotRecommended:

    def test_the_cap_is_shown_as_a_stated_input(self):
        [cap] = [one for one in card().described
                 if one.field_name == "concentration_cap"]
        assert cap.destination is Destination.ENGINE_INPUT
        assert cap.value == "20%"

    def test_the_card_never_calls_a_threshold_safe_or_optimal(self):
        rendered = str(card().to_json()).lower()
        for word in ("safe", "optimal", "recommended", "we suggest"):
            assert word not in rendered


class TestScopeComesFromVersionedRuntimeDeclarations:

    def test_will_model_reads_the_runtime_assumptions(self):
        assert set(card().will_model) == {
            one.statement for one in US_SHARE_WITHHOLDING.assumptions}

    def test_will_not_model_reads_the_runtime_limitations(self):
        assert set(card().will_not_model) == {
            one.statement for one in US_SHARE_WITHHOLDING.limitations}

    def test_it_says_withholding_is_not_final_tax(self):
        assert any("marginal" in one for one in card().will_not_model)

    def test_it_says_household_assets_are_excluded(self):
        from src.runtime.concentration import ConcentrationPolicy

        assert "household" in ConcentrationPolicy(target=0.2).scope_note

    def test_the_scope_is_not_restated_in_the_card_module(self):
        """Restated, the card and the result's modelling scope become two lists
        that drift."""
        import inspect

        from src.workspace import rsu_confirmation

        source = inspect.getsource(rsu_confirmation)
        for statement in list(card().will_not_model)[:3]:
            assert statement not in source


class TestVersionPins:

    def test_the_pin_travels_on_the_card(self):
        assert card().version_pin == VERSIONS.pin

    def test_every_declared_version_is_shown(self):
        shown = card().versions
        for entry in fields(DeclarationVersions):
            assert entry.name in shown

    def test_matching_versions_permit_the_run(self):
        check_versions(VERSIONS, replace(VERSIONS))

    def test_a_moved_runtime_refuses_the_run(self):
        moved = replace(VERSIONS,
                        rsu_runtime_version="rsu-vesting/us-share-withholding@2")
        with pytest.raises(DeclarationVersionMismatch,
                           match="DECLARATION_VERSION_MISMATCH"):
            check_versions(VERSIONS, moved)

    def test_the_refusal_names_what_moved(self):
        moved = replace(VERSIONS, tax_runtime_version="tax/us-federal@2")
        with pytest.raises(DeclarationVersionMismatch, match="tax_runtime"):
            check_versions(VERSIONS, moved)

    def test_it_refuses_rather_than_refreshing(self):
        """Re-reading the plan against newer rules would execute something the
        user never agreed to."""
        moved = replace(VERSIONS, scope_schema_version="rsu-result-context@2")
        with pytest.raises(DeclarationVersionMismatch, match="re-confirm|Re-confirm"):
            check_versions(VERSIONS, moved)


class TestTheJourneyEndToEnd:
    """description -> card -> confirm -> run -> stored context -> worksheet."""

    def test_the_card_declares_and_the_worksheet_reports(self, tmp_path):
        import pandas as pd

        from src.workspace.rsu_view import ContextState, RSUWorksheetView
        from tests.test_rsu_result_live import messy_run, store_with_plan

        before = card()
        assert before.can_run
        assert before.will_model and before.will_not_model

        store = store_with_plan(tmp_path)
        result = messy_run(pd.bdate_range("2026-03-02", "2026-04-30"))
        store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                         result=result.to_json(), comparison={})

        after = RSUWorksheetView.from_result(
            store.get_run("r1", "pilot")["result"])
        assert after.context_state is ContextState.PRESENT

        # The card promised a delivered figure; the worksheet reports one.
        assert "vest" in " ".join(before.will_model).lower()
        assert after.vest_accounting.delivered_value is not None

    def test_the_card_contains_no_stored_output_and_the_view_no_declaration(
            self, tmp_path):
        import pandas as pd

        from src.workspace.rsu_view import RSUWorksheetView
        from tests.test_rsu_result_live import messy_run, store_with_plan

        store = store_with_plan(tmp_path)
        result = messy_run(pd.bdate_range("2026-03-02", "2026-04-30"))
        store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                         result=result.to_json(), comparison={})
        view = RSUWorksheetView.from_result(
            store.get_run("r1", "pilot")["result"])

        assert "delivered_value" not in str(card().to_json())
        assert view.vest_accounting.delivered_value == 3_900.0
