"""The worksheet reads. It does not recompute.

    engine computes -> MissionResult stores -> WorkspaceStore preserves
                    -> this reads

The load-bearing test patches every engine calculator to raise and requires the
page to render unchanged. A figure recomputed at render time is a second
implementation of the engine living in the view layer, and the two disagree on
exactly the runs where something went wrong — the runs a reader most needs to
trust.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from src.workspace.rsu_view import (
    BLOCK_ORDER,
    ContextState,
    RSUWorksheetView,
)

OWNER = "pilot"


@pytest.fixture
def sessions():
    return pd.bdate_range("2026-03-02", "2026-04-30")


@pytest.fixture
def stored(tmp_path, sessions):
    """A real run, persisted and read back."""
    from tests.test_rsu_result_live import messy_run, store_with_plan

    store = store_with_plan(tmp_path)
    result = messy_run(sessions)
    store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                     result=result.to_json(), comparison={})
    return store, store.get_run("r1", OWNER)["result"]


class TestItRendersWithEveryCalculatorBroken:
    """Patched at the imported symbol, not only at its definition — an earlier
    no-recompile fix showed a module can hold its own reference."""

    def test_the_page_renders(self, stored, monkeypatch):
        _, payload = stored

        def explode(*args, **kwargs):
            raise AssertionError("the worksheet recomputed a financial value")

        import importlib

        import src.comparison.rsu_profile as profile
        import src.runtime.allocation as allocation
        import src.runtime.concentration as concentration
        import src.runtime.disposition as disposition

        # `import src.mission.simulate as engine` binds the re-exported
        # *function*, because src.mission.__init__ rebinds that name on the
        # package. Patching it would have raised rather than proving anything.
        engine = importlib.import_module("src.mission.simulate")

        for module, name in (
                (concentration, "assess"), (concentration, "solve"),
                (concentration, "projected_concentration"),
                (concentration, "realized_concentration"),
                (allocation, "proceeds_from"), (allocation, "compile_policy"),
                (allocation, "instruction_for"),
                (disposition, "instruction_for"), (disposition, "advance"),
                (disposition, "eligibility"),
                (profile, "classify"), (profile, "evaluate"),
                (engine, "simulate")):
            monkeypatch.setattr(module, name, explode)

        view = RSUWorksheetView.from_result(payload)
        assert view.context_state is ContextState.PRESENT

    def test_the_values_equal_the_stored_context(self, stored, monkeypatch):
        _, payload = stored

        def explode(*args, **kwargs):
            raise AssertionError("recomputed")

        import src.runtime.concentration as concentration

        monkeypatch.setattr(concentration, "assess", explode)
        monkeypatch.setattr(concentration, "solve", explode)

        view = RSUWorksheetView.from_result(payload)
        context = payload["rsu_context"]
        assert view.concentration.projected == \
            context["concentration"]["projected"]
        assert view.vest_accounting.delivered_value == \
            context["vest_accounting"]["delivered_value"]

    def test_the_constructor_accepts_nothing_to_recompute_with(self):
        """The API is the guard. Adding prices or a store back would be visible
        in review in a way a quiet call to a calculator is not."""
        import inspect

        parameters = inspect.signature(RSUWorksheetView.from_result).parameters
        assert list(parameters) == ["result"]

    def test_the_module_imports_no_engine_package(self):
        """Not sufficient alone, and it stops an accidental dependency
        appearing later."""
        import ast
        import inspect

        from src.workspace import rsu_view

        tree = ast.parse(inspect.getsource(rsu_view))
        top_level = [node for node in tree.body
                     if isinstance(node, (ast.Import, ast.ImportFrom))]
        imported = " ".join(
            getattr(node, "module", "") or "" for node in top_level)
        for package in ("concentration", "comparability", "allocation",
                        "simulate", "market_data", "disposition"):
            assert package not in imported, package


class TestNothingCollapses:

    @pytest.fixture
    def with_projection(self, stored):
        """The messy run's sizing refuses on a missing price, so it carries no
        projection. Projected-versus-realized needs a run that has one."""
        from src.mission.rsu_result import (ConcentrationContext,
                                            RSUResultContext)

        _, payload = stored
        context = RSUResultContext(
            concentration=ConcentrationContext(
                current=0.5, target=0.2, projected=0.199, realized=None),
            modelling_scope={"modelled": ("share delivery",)})
        return {**payload, "rsu_context": context.to_json()}

    def test_projected_and_realized_stay_separate(self, with_projection):
        view = RSUWorksheetView.from_result(with_projection)
        assert view.concentration.projected == 0.199
        assert view.concentration.realized is None
        assert view.concentration.projected_only is True

    def test_a_projection_is_not_reported_as_the_cap_being_met(
            self, with_projection):
        assert RSUWorksheetView.from_result(
            with_projection).concentration.cap_achieved is None

    def test_pending_and_failed_are_separate_lists(self, stored):
        _, payload = stored
        view = RSUWorksheetView.from_result(payload)
        assert view.disposition.pending is not view.disposition.failed
        assert view.disposition.has_outstanding

    def test_requested_filled_and_unfilled_are_distinct(self, stored):
        _, payload = stored
        view = RSUWorksheetView.from_result(payload)
        rendered = view.allocation.to_json()
        assert {"requested", "executed", "unfilled", "residual_cash",
                "unallocated_weight"} <= set(rendered)

    def test_every_benchmark_row_survives_to_the_view(self, stored):
        _, payload = stored
        view = RSUWorksheetView.from_result(payload)
        assert [row.benchmark_id for row in view.comparisons] == [
            "hold", "value_matched", "never_ran"]

    def test_incomparable_and_not_evaluated_are_not_merged(self, stored):
        _, payload = stored
        statuses = {row.status
                    for row in RSUWorksheetView.from_result(payload).comparisons}
        assert {"COMPARABLE", "INCOMPARABLE", "NOT_EVALUATED"} <= statuses

    def test_there_is_no_single_summary_badge(self, stored, with_projection):
        """Collapsed into one status, pending and failed land on the same side
        of it, and so do projected and realized."""
        _, payload = stored
        disposition = RSUWorksheetView.from_result(payload).to_json()
        assert "pending" in disposition["disposition"]
        assert "failed" in disposition["disposition"]

        concentration = RSUWorksheetView.from_result(
            with_projection).to_json()["concentration"]
        assert concentration["projected"] is not None
        assert concentration["realized"] is None


class TestMissingAndCorruptContexts:

    def test_an_absent_context_is_not_declared(self):
        view = RSUWorksheetView.from_result({"modelling_scope": {}})
        assert view.context_state is ContextState.NOT_DECLARED
        assert "absence of record" in view.note

    def test_an_absent_context_shows_no_financial_blocks(self):
        assert not RSUWorksheetView.from_result({}).financial_blocks_available

    def test_a_corrupt_context_is_typed_as_corrupt(self, stored):
        _, payload = stored
        broken = {**payload,
                  "rsu_context": {**payload["rsu_context"],
                                  "presentability": "COMPLETE"}}
        view = RSUWorksheetView.from_result(broken)
        assert view.context_state is ContextState.CORRUPT

    def test_a_corrupt_context_hides_the_figures(self, stored):
        _, payload = stored
        broken = {**payload,
                  "rsu_context": {**payload["rsu_context"],
                                  "presentability": "COMPLETE"}}
        view = RSUWorksheetView.from_result(broken)
        assert not view.financial_blocks_available
        assert view.vest_accounting.delivered_value is None
        assert "could not be verified" in view.note

    def test_a_missing_section_is_corrupt_not_partial(self, stored):
        _, payload = stored
        context = {k: v for k, v in payload["rsu_context"].items()
                   if k != "disposition"}
        view = RSUWorksheetView.from_result({**payload, "rsu_context": context})
        assert view.context_state is ContextState.CORRUPT

    def test_no_recovery_is_attempted(self, stored):
        """A partially rendered corrupt result is an edited record presented as
        an original."""
        _, payload = stored
        context = {k: v for k, v in payload["rsu_context"].items()
                   if k != "allocation"}
        view = RSUWorksheetView.from_result({**payload, "rsu_context": context})
        assert view.comparisons == ()
        assert view.allocation.residual_cash is None


class TestHistoricalRevisionsRenderFromTheirOwnResult:

    def test_an_older_run_keeps_its_stored_context_after_a_newer_one(
            self, tmp_path, sessions):
        from tests.test_rsu_result_live import messy_run, store_with_plan
        from src.mission.rsu_result import build

        store = store_with_plan(tmp_path)
        old = messy_run(sessions)
        store.record_run(run_id="old", plan_id="plan-1", ran_at="t1",
                         result=old.to_json(), comparison={})

        # A later, cleaner run for the same plan.
        newer = build(vest_accounting={"gross_vest_value": 1.0,
                                       "withheld_value": 0.0,
                                       "external_flow_value": 1.0,
                                       "cash_remainder": 0.0},
                      modelling_scope={"modelled": ["x"]})
        payload = {**old.to_json(), "rsu_context": newer.to_json()}
        store.record_run(run_id="new", plan_id="plan-1", ran_at="t2",
                         result=payload, comparison={})

        reopened = RSUWorksheetView.from_result(
            store.get_run("old", OWNER)["result"])
        assert reopened.concentration.missing_prices == ("VTI",)
        assert reopened.vest_accounting.delivered_value == 3_900.0

    def test_the_view_never_resolves_to_the_newest_run(self, stored):
        """It is handed one payload and has no way to reach another.

        Checked against the code, not the prose: the docstring explains that it
        takes no store, and a naive substring search matches that explanation.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(RSUWorksheetView.from_result)))
        names = {node.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute)}
        names |= {node.id for node in ast.walk(tree)
                  if isinstance(node, ast.Name)}
        for reach in ("store", "runs_for", "get_run", "latest",
                      "worksheet_revisions"):
            assert reach not in names, reach


class TestTheBlockOrder:

    def test_scope_sits_with_the_figures_it_qualifies(self):
        """At the foot of the page it is read after the number is believed."""
        assert BLOCK_ORDER.index("modelling_scope") < \
            BLOCK_ORDER.index("provenance")

    def test_status_comes_first(self):
        assert BLOCK_ORDER[0] == "result_status"

    def test_every_section_has_a_place(self):
        for block in ("vest_accounting", "disposition", "allocation",
                      "concentration", "benchmark_comparability"):
            assert block in BLOCK_ORDER
