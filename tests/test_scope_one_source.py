"""One disclosure, three surfaces.

    runtime declarations
      -> ScopeDisclosure
         |- confirmation card       (before the run)
         |- persisted result        (at the run)
         `- worksheet               (after the run, stored only)

The card and the worksheet may word a rule differently. They must not derive
different facts, and the worksheet must not rebuild scope from today's runtimes
— that would rewrite what an old run disclosed the moment realization coverage
changed, leaving the figures in place while their stated scope moved.
"""
from __future__ import annotations

import pytest

from src.mission.rsu_declaration import DeclarationVersions, RSUDeclaration
from src.mission.rsu_result import build as build_context
from src.mission.rsu_result import from_json as context_from_json
from src.runtime.rsu import IMPLEMENTED, US_SHARE_WITHHOLDING
from src.workspace.rsu_confirmation import build as build_card
from src.workspace.rsu_view import RSUWorksheetView
from src.workspace.scope_disclosure import (
    SCOPE_SCHEMA_VERSION,
    Enforcement,
    for_account,
    for_rsu,
)
from src.workspace.scope_disclosure import from_json as scope_from_json


def declaration() -> RSUDeclaration:
    return RSUDeclaration(employer_ticker="ACME", gross_shares=100.0,
                          withholding_rate=0.22,
                          versions=DeclarationVersions())


def card(**kwargs):
    return build_card(declaration(), runtime=US_SHARE_WITHHOLDING, **kwargs)


def stored_context(scope=None):
    disclosure = scope or for_rsu(US_SHARE_WITHHOLDING,
                                  implemented=IMPLEMENTED).to_json()
    return build_context(scope_disclosure=disclosure,
                         modelling_scope={"modelled": ("share delivery",)})


class TestOneSource:

    def test_the_card_renders_the_disclosure(self):
        assert card().to_json()["scope"]["schema_version"] == \
            SCOPE_SCHEMA_VERSION

    def test_the_result_stores_it_verbatim(self):
        disclosure = for_rsu(US_SHARE_WITHHOLDING,
                             implemented=IMPLEMENTED).to_json()
        assert stored_context(disclosure).to_json()["scope_disclosure"] == \
            disclosure

    def test_the_worksheet_reads_the_stored_one(self):
        context = stored_context()
        view = RSUWorksheetView.from_result(
            {"rsu_context": context.to_json()})
        assert view.scope_disclosure["schema_version"] == SCOPE_SCHEMA_VERSION

    def test_all_three_carry_the_same_rules(self):
        disclosure = for_rsu(US_SHARE_WITHHOLDING,
                             implemented=IMPLEMENTED).to_json()
        view = RSUWorksheetView.from_result(
            {"rsu_context": stored_context(disclosure).to_json()})

        from_card = {one["rule"]: one["enforcement"]
                     for one in card(scope=disclosure).to_json()["scope"]["rules"]}
        from_sheet = {one["rule"]: one["enforcement"]
                      for one in view.scope_disclosure["rules"]}
        assert from_card == from_sheet

    def test_the_card_and_worksheet_may_word_it_differently(self):
        """Different copy is fine; different facts are not."""
        disclosure = for_account("ROTH", tax_year=2026)
        [shared] = [one for one in disclosure.rules
                    if one.rule == "shared limit across related accounts"]
        assert shared.enforcement is Enforcement.PARTIAL
        assert shared.why


class TestTheWorksheetNeverRebuilds:

    def test_it_uses_the_stored_disclosure_when_the_runtime_moves(self):
        """The drift case: realization coverage changes after a historical run.

        The old worksheet must keep showing what that run disclosed."""
        historical = for_rsu(US_SHARE_WITHHOLDING, implemented=()).to_json()
        assert historical["coverage"]["realized"] == 0

        view = RSUWorksheetView.from_result(
            {"rsu_context": stored_context(historical).to_json()})

        # Today's runtime realizes everything; the stored disclosure does not.
        current = for_rsu(US_SHARE_WITHHOLDING, implemented=IMPLEMENTED)
        assert current.coverage["realized"] == 6
        assert view.scope_disclosure["coverage"]["realized"] == 0

    def test_it_builds_no_disclosure_of_its_own(self):
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(RSUWorksheetView.from_result)))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)}
        assert "for_rsu" not in called
        assert "for_account" not in called

    def test_the_view_module_imports_no_disclosure_builder(self):
        import ast
        import inspect

        from src.workspace import rsu_view

        tree = ast.parse(inspect.getsource(rsu_view))
        imported = " ".join(getattr(node, "module", "") or ""
                            for node in tree.body
                            if isinstance(node, (ast.Import, ast.ImportFrom)))
        assert "scope_disclosure" not in imported


class TestMissingHistoricalScope:

    def test_a_run_without_a_disclosure_says_so(self):
        """Not reconstructed from today's runtime."""
        context = build_context(modelling_scope={"modelled": ("x",)})
        view = RSUWorksheetView.from_result({"rsu_context": context.to_json()})

        assert view.scope_disclosure is None
        assert view.to_json()["scope_recorded"] is False

    def test_it_is_not_silently_filled_in(self):
        context = build_context(modelling_scope={"modelled": ("x",)})
        rendered = RSUWorksheetView.from_result(
            {"rsu_context": context.to_json()}).to_json()
        assert rendered["scope_disclosure"] is None


class TestCoverageCategorisation:

    def test_exclusions_stay_outside_the_denominator(self):
        scope = for_rsu(US_SHARE_WITHHOLDING, implemented=IMPLEMENTED)
        assert scope.coverage["declared"] < len(scope.rules)
        assert scope.coverage["out_of_scope"] > 0

    def test_exclusions_remain_visible(self):
        scope = card().to_json()["scope"]
        excluded = [one for one in scope["rules"]
                    if one["enforcement"] == "NOT_MODELLED"]
        assert excluded

    def test_coverage_is_not_presented_as_confidence(self):
        assert "not confidence" in card().to_json()["scope"]["coverage"]["note"]

    def test_it_survives_a_round_trip(self):
        original = for_rsu(US_SHARE_WITHHOLDING, implemented=IMPLEMENTED)
        assert scope_from_json(original.to_json()).coverage == original.coverage


class TestNoDuplicateRule:

    def test_the_shared_limit_appears_once(self):
        rules = [one.rule for one in for_account("ROTH", tax_year=2026).rules]
        assert len(rules) == len(set(rules))

    def test_it_survives_persistence(self):
        disclosure = for_account("ROTH", tax_year=2026).to_json()
        stored = context_from_json(
            stored_context(disclosure).to_json()).scope_disclosure
        rules = [one["rule"] for one in stored["rules"]]
        assert len(rules) == len(set(rules))


class TestRuntimeRefsArePinned:

    def test_the_disclosure_names_the_runtimes_it_came_from(self):
        assert for_rsu(US_SHARE_WITHHOLDING,
                       implemented=IMPLEMENTED).runtime_refs
        assert for_account("ROTH", tax_year=2026).runtime_refs

    def test_the_account_ref_names_the_tax_year(self):
        [ref] = for_account("ROTH", tax_year=2026).runtime_refs
        assert "2026" in ref

    def test_a_different_year_pins_a_different_ruleset(self):
        assert for_account("ROTH", tax_year=2025).runtime_refs != \
            for_account("ROTH", tax_year=2026).runtime_refs

    def test_the_stored_refs_survive(self):
        disclosure = for_rsu(US_SHARE_WITHHOLDING,
                             implemented=IMPLEMENTED).to_json()
        view = RSUWorksheetView.from_result(
            {"rsu_context": stored_context(disclosure).to_json()})
        assert view.scope_disclosure["runtime_refs"] == \
            disclosure["runtime_refs"]
