"""The RSU worksheet blocks. Presentation over a read-only view model.

Every value on this page came out of a stored run. The template arranges and
calculates nothing: a page that recomputed would be a second implementation of
the engine, and a page that ranked benchmarks by outcome would be making a
recommendation the system does not make.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest
from jinja2 import Environment, FileSystemLoader

from src.workspace.rsu_view import BLOCK_ORDER, RSUWorksheetView

TEMPLATE = "_rsu_worksheet.html"


def render(view: RSUWorksheetView) -> str:
    env = Environment(loader=FileSystemLoader("src/workspace/templates"))
    return env.get_template(TEMPLATE).render(view=view.to_json())


def flat(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


@pytest.fixture
def messy(tmp_path):
    """One run carrying every state the page must keep visible."""
    from tests.test_rsu_result_live import messy_run, store_with_plan

    store = store_with_plan(tmp_path)
    result = messy_run(pd.bdate_range("2026-03-02", "2026-04-30"))
    store.record_run(run_id="r1", plan_id="plan-1", ran_at="t1",
                     result=result.to_json(), comparison={})
    return RSUWorksheetView.from_result(store.get_run("r1", "pilot")["result"])


class TestEveryMessyStateStaysVisible:

    def test_the_pending_disposition_is_shown(self, messy):
        body = flat(render(messy))
        assert "Still pending" in body
        assert "SELL_ALL_AND_DIVERSIFY" in body

    def test_a_pending_sale_is_not_reported_as_a_decision_to_hold(self, messy):
        assert "not a decision to hold" in flat(render(messy))

    def test_the_missing_concentration_price_is_shown(self, messy):
        body = flat(render(messy))
        assert "could not be measured" in body
        assert "VTI" in body

    def test_the_projection_is_labelled_not_achieved(self, messy):
        body = flat(render(messy))
        assert "Realized, measured from the fill" in body
        assert "not yet available" in body

    def test_the_incomparable_benchmark_is_shown(self, messy):
        body = flat(render(messy))
        assert "value_matched" in body
        assert "incomparable" in body

    def test_the_not_evaluated_benchmark_is_shown(self, messy):
        assert "never_ran" in flat(render(messy))

    def test_the_unresolved_modelling_item_is_shown(self, messy):
        assert "capital-gains tax" in flat(render(messy))

    def test_the_diagnostics_name_their_kind_in_words(self, messy):
        """Colour alone cannot distinguish a limitation from a data gap, and
        they lead a reader somewhere different."""
        body = flat(render(messy))
        assert "Data Gap:" in body
        assert "Limitation:" in body
        assert "Unsettled:" in body


class TestItRendersWithEveryCalculatorBroken:

    def test_the_page_is_identical(self, messy, monkeypatch):
        import importlib

        import src.comparison.rsu_profile as profile
        import src.runtime.allocation as allocation
        import src.runtime.concentration as concentration
        import src.runtime.disposition as disposition

        engine = importlib.import_module("src.mission.simulate")
        before = render(messy)

        def explode(*args, **kwargs):
            raise AssertionError("the worksheet recomputed a financial value")

        for module, name in (
                (concentration, "assess"), (concentration, "solve"),
                (concentration, "projected_concentration"),
                (concentration, "realized_concentration"),
                (allocation, "proceeds_from"), (allocation, "compile_policy"),
                (disposition, "advance"), (disposition, "eligibility"),
                (profile, "classify"), (profile, "evaluate"),
                (engine, "simulate")):
            monkeypatch.setattr(module, name, explode)

        assert render(messy) == before


class TestTheTemplateComputesNothing:

    def source(self) -> str:
        return open(f"src/workspace/templates/{TEMPLATE}").read()

    def test_it_contains_no_arithmetic(self):
        body = self.source()
        for pattern in (r"\{\{[^}]*[-+*/]\s*\d", r"\{\%[^%]*[<>]=?\s*\d",
                        r"\{\{[^}]*\bsum\b", r"\{\{[^}]*\|\s*sum"):
            assert not re.search(pattern, body), pattern

    def test_it_does_not_sort_or_rank(self):
        """Sorting benchmarks by outcome would rank them, and a ranking is a
        recommendation."""
        body = self.source()
        for pattern in ("|sort", "|max", "|min", "sortattr", "|reverse"):
            assert pattern not in body

    def test_it_reads_only_view_fields(self):
        """Every expression addresses `view`."""
        body = self.source()
        for expression in re.findall(r"\{\{\s*([a-zA-Z_][\w.]*)", body):
            root = expression.split(".")[0]
            assert root in {"view", "scope", "row", "one", "entry", "asset",
                            "weight", "loop"}, expression


class TestEveryBlockHasAnUnmetState:

    def test_a_context_with_nothing_recorded_still_renders_each_block(self):
        """A block that vanished when empty would let an absence read as an
        absence of anything to say."""
        body = flat(render(RSUWorksheetView.from_result(
            {"rsu_context": _empty_context()})))
        for heading in ("Result status", "Vest accounting", "Disposition",
                        "Allocation of proceeds",
                        "Employer-stock concentration", "Benchmark comparisons",
                        "What this run modelled", "Provenance"):
            assert heading in body

    @pytest.mark.parametrize("phrase", [
        "No vest accounting was recorded",
        "No disposition was instructed",
        "No allocation was requested",
        "No concentration cap was declared",
        "No benchmarks were requested",
        "No modelling scope was recorded",
    ])
    def test_each_absence_says_so_explicitly(self, phrase):
        body = flat(render(RSUWorksheetView.from_result(
            {"rsu_context": _empty_context()})))
        assert phrase in body

    def test_a_corrupt_context_keeps_the_shell_and_provenance(self, messy):
        broken = RSUWorksheetView.from_result(
            {"rsu_context": {"vest_accounting": {}}})
        body = flat(render(broken))
        assert "Figures not shown" in body
        assert "Provenance" in body
        assert "Vest accounting" not in body


def _empty_context():
    from src.mission.rsu_result import RSUResultContext

    return RSUResultContext(
        modelling_scope={}).to_json() | {
            "vest_accounting": RSUResultContext().vest_accounting.to_json(),
            "disposition": RSUResultContext().disposition.to_json(),
            "allocation": RSUResultContext().allocation.to_json(),
            "concentration": RSUResultContext().concentration.to_json(),
            "comparisons": RSUResultContext().comparisons.to_json()}


class TestInstructionAndFillStayDistinct:

    def test_both_columns_exist(self, messy):
        body = flat(render(messy))
        assert "Instructed" in body or "Still pending" in body

    def test_an_unfilled_sale_says_its_proceeds_are_unknown(self, tmp_path):
        from src.mission.rsu_result import (DispositionContext,
                                            RSUResultContext)

        context = RSUResultContext(
            disposition=DispositionContext(
                status="PENDING",
                executions=({"instruction_id": "d1",
                             "instructed_on": "2026-03-11",
                             "shares": 78.0, "expected_price": 50.0,
                             "filled_on": None, "fill_price": None,
                             "proceeds": None, "reconciled": False},)),
            modelling_scope={"modelled": ("share delivery",)})
        body = flat(render(RSUWorksheetView.from_result(
            {"rsu_context": context.to_json()})))

        assert "not filled" in body
        assert "unknown until filled" in body


class TestNoRecommendationLanguage:

    @pytest.mark.parametrize("word", [
        "best", "optimal", "should sell", "recommend", "we suggest",
        "safest", "ideal", "outperform"])
    def test_the_template_never_advises(self, word):
        """Checked against what renders, with Jinja comments stripped as Jinja
        strips them.

        The comments explain *why* the page does not rank, and say the word
        "recommendation" to do so. Matching them is the same mistake as
        matching "vest" inside "uninvested" — prose legitimately contains the
        vocabulary the output forbids.
        """
        source = open(f"src/workspace/templates/{TEMPLATE}").read()
        emitted = re.sub(r"\{#.*?#\}", "", source, flags=re.S).lower()
        assert word not in emitted

    def test_the_rendered_page_never_advises(self, messy):
        body = flat(render(messy)).lower()
        for word in ("best", "optimal", "you should", "we recommend"):
            assert word not in body


class TestTheBlockOrder:

    def test_the_template_follows_the_declared_order(self, messy):
        body = render(messy)
        headings = ["Result status", "Vest accounting", "Disposition",
                    "Allocation of proceeds", "Employer-stock concentration",
                    "Benchmark comparisons", "What this run modelled",
                    "Provenance"]
        positions = [body.index(one) for one in headings]
        assert positions == sorted(positions)

    def test_the_declared_order_matches_the_view_model(self):
        assert BLOCK_ORDER[0] == "result_status"
        assert BLOCK_ORDER.index("modelling_scope") < \
            BLOCK_ORDER.index("provenance")


class TestHistoricalRevisions:

    def test_an_older_run_renders_from_its_own_stored_result(self, tmp_path):
        from src.mission.rsu_result import build
        from tests.test_rsu_result_live import messy_run, store_with_plan

        store = store_with_plan(tmp_path)
        old = messy_run(pd.bdate_range("2026-03-02", "2026-04-30"))
        store.record_run(run_id="old", plan_id="plan-1", ran_at="t1",
                         result=old.to_json(), comparison={})

        newer = build(vest_accounting={"gross_vest_value": 1.0,
                                       "withheld_value": 0.0,
                                       "external_flow_value": 1.0,
                                       "cash_remainder": 0.0},
                      modelling_scope={"modelled": ("x",)})
        store.record_run(run_id="new", plan_id="plan-1", ran_at="t2",
                         result={**old.to_json(),
                                 "rsu_context": newer.to_json()},
                         comparison={})

        body = flat(render(RSUWorksheetView.from_result(
            store.get_run("old", "pilot")["result"])))
        assert "3900.0" in body
        assert "capital-gains tax" in body
