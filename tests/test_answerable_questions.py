"""A question the user cannot answer sends them back to rewriting the prose.

From a real pilot journey: "i buy 1000 usd of SPX etf every time SP500 trades
below its 200 DMA". The page asked six questions. Two of them — how much are
you contributing, how much are you starting with — rendered no control at all,
because the template only emitted radios and those fields have no finite
answers. And every one of them was labelled "This request depends on a
capability Quantify does not currently model", because a missing price for
SPX had relabelled every open item as unsupported.

Amount is modelled. The plan simply had nothing to price. The two failures
compound into what reads as a flat rejection.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.mission.compiler import compile_scenario
from src.workspace.feasibility import Resolution, classify

UNRUNNABLE = ("i buy 1000 usd of SPX etf every time SP500 trades below its "
              "200 DMA for the past 5 years")
RUNNABLE = "I put $500 a month into VOO and never sell."
#: Runnable, but says nothing about how much — so `amount` is genuinely asked.
AMOUNT_MISSING = "I buy VOO every month and never sell."


def items_for(text, executable):
    scenario = compile_scenario(text).scenario
    return classify(scenario.provenance.unresolved, executable=executable)


class TestAnUnrunnablePlanDoesNotRelabelItsQuestions:
    def test_amount_is_not_reported_as_an_unmodelled_capability(self):
        found = {one.field: one for one in items_for(UNRUNNABLE, executable=False)}
        assert found["amount"].resolution is Resolution.REQUIRED_CLARIFICATION
        assert found["amount"].answerable

    def test_account_type_is_not_either(self):
        found = {one.field: one for one in items_for(UNRUNNABLE, executable=False)}
        assert found["account_type"].resolution is Resolution.REQUIRED_CLARIFICATION

    def test_nothing_is_dismissible_while_the_plan_cannot_run(self):
        """The original intent, preserved. Setting every item aside would
        still leave no result — but that is a different sentence from "we do
        not model this"."""
        assert not any(one.dismissible for one in items_for(UNRUNNABLE, executable=False))

    def test_separable_items_are_dismissible_again_once_it_can_run(self):
        text = "I put $500 a month into VTI and also want to feel calmer about money."
        items = items_for(text, executable=True)
        separable = [one for one in items
                     if one.resolution is Resolution.UNSUPPORTED_SEPARABLE]
        if separable:
            assert all(one.dismissible for one in separable)


class TestEveryRequiredQuestionCanBeAnswered:
    @pytest.mark.parametrize("field", ["amount", "starting_capital", "account_type"])
    def test_the_field_is_answerable(self, field):
        found = {one.field: one for one in items_for(UNRUNNABLE, executable=False)}
        assert found[field].answerable, f"{field} is asked but not answerable"


class TestThePageRendersAControlForEveryQuestion:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        import src.api as api
        import src.web.routes as web_routes
        import src.workspace.routes as workspace_routes
        from src.ledger import Ledger
        from src.workspace.store import WorkspaceStore

        ledger = Ledger(tmp_path / "public.db")
        monkeypatch.setattr(api, "_ledger", ledger)
        monkeypatch.setattr(web_routes, "Ledger", lambda *a, **k: ledger)
        store = WorkspaceStore(tmp_path / "workspace.db")
        monkeypatch.setattr(workspace_routes, "_store", lambda: store)
        api._bootstrap()
        return TestClient(api.app)

    def test_no_question_is_rendered_without_a_way_to_answer_it(self, client):
        """The check that would have caught this. Parses the rendered page and
        requires every question block to contain an input naming its field."""
        import re

        page = client.get("/workspace/new", params={"describe": RUNNABLE})
        assert page.status_code == 200
        body = page.text

        # Both control kinds count. Questions post `answer:`, unconfirmed
        # inferences post `confirm:` — the first version of this regex looked
        # only for `answer:` and reported the inference blocks as
        # unanswerable, which would have been a defect in the test.
        asked = set(re.findall(r'controls ([a-z_]+)</div>', body))
        answered = set(re.findall(r'name="answer:([a-z_]+)"', body))
        confirmed = set(re.findall(r'name="confirm:([a-z_]+)"', body))
        unanswerable = asked - answered - confirmed
        assert not unanswerable, (
            f"asked with no control: {sorted(unanswerable)}")

    def test_an_amount_question_gets_a_text_input(self, client):
        page = client.get("/workspace/new", params={"describe": AMOUNT_MISSING})
        assert 'name="answer:amount"' in page.text, (
            "the amount question rendered with no way to answer it")

    def test_every_question_on_an_unrunnable_plan_is_answerable_too(self, client):
        """The journey that produced this. Nothing can be priced, and the
        questions must still be answerable rather than relabelled."""
        import re

        body = client.get("/workspace/new", params={"describe": UNRUNNABLE}).text
        asked = set(re.findall(r'controls ([a-z_]+)</div>', body))
        answered = set(re.findall(r'name="answer:([a-z_]+)"', body))
        confirmed = set(re.findall(r'name="confirm:([a-z_]+)"', body))
        assert not (asked - answered - confirmed)
        assert "capability Quantify does not currently model" not in body

    def test_the_answer_is_accepted_and_the_plan_saves(self, client):
        """End to end: the whole point is not having to start over."""
        from tests.conftest import submit_rendered_confirmation

        response, plan_id = submit_rendered_confirmation(
            client, RUNNABLE, title="Answered inline",
            answers={"amount": "500", "starting_capital": "0",
                     "account_type": "TAXABLE", "cadence": "monthly"})
        assert response.status_code == 303, response.text
        assert plan_id
