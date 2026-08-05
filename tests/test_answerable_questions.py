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
        asked = set(re.findall(r'data-field="([a-z_]+)"', body))
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
        asked = set(re.findall(r'data-field="([a-z_]+)"', body))
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


class TestEveryQuestionOffersSomeControl:
    """Answer box, choice, or acknowledgement — but never nothing.

    A blocked plan rendered four "What did you mean by X?" questions with no
    control beneath any of them. Two causes met: an `unclear:` item has no
    answer box, because the compiler has nowhere to put a free-text reply, and
    `dismissible` had been gated on the plan being runnable — so the one
    control it did have disappeared exactly when the user most needed
    something to do.

    The earlier check only asserted that *asked* fields had an `answer:` or
    `confirm:` input. `unclear:` items are asked and take neither, so they
    passed through it.
    """

    UNCLEAR = ["unclear:SP500 ETF (specific ticker not given)",
               "unclear:5-year lookback period for backtest"]

    def _items(self, executable):
        from src.workspace.feasibility import classify

        class Raised:
            def __init__(self, field):
                self.field = field
                self.question = "What did you mean by X?"
                self.why_it_matters = "w"

        return classify([Raised(f) for f in self.UNCLEAR], executable=executable)

    @pytest.mark.parametrize("executable", [True, False])
    def test_an_unclear_item_always_has_at_least_one_control(self, executable):
        for item in self._items(executable):
            assert item.answerable or item.dismissible, (
                f"{item.field} is asked with nothing the user can do")

    def test_a_blocked_plan_does_not_remove_the_acknowledgement(self):
        """The regression itself. Runnability is a property of the plan and
        belongs in its own banner, not in whether a control exists."""
        assert all(item.dismissible for item in self._items(executable=False))

    def test_dismissal_still_does_not_extend_to_required_fields(self):
        """The other direction. Making everything dismissible would let a user
        acknowledge away the account type, which changes what the result
        answers rather than narrowing it."""
        from src.workspace.feasibility import OpenItem, Resolution

        required = OpenItem("account_type", "q", "w",
                            Resolution.REQUIRED_CLARIFICATION)
        material = OpenItem("x", "q", "w", Resolution.MATERIAL)
        assert not required.dismissible
        assert not material.dismissible


class TestNoControlPromisesMoreThanTheCompilerDelivers:
    """An input whose value the compiler never reads is worse than none.

    `asset_identity:SPX` and `template:x` are generated from the parse and
    have no settle site. Classified REQUIRED_CLARIFICATION they rendered a
    free-text box, the reply was recorded as an amendment, and the same
    question came back unchanged — the user having every reason to think they
    had answered it.
    """

    def consumed_by_the_compiler(self):
        """The fields `settle()`/`answered()` actually read, from the source.

        Derived rather than restated. A hand-written list here would be the
        second copy that stops matching the compiler, and this check exists
        precisely because the two disagreed.
        """
        import ast
        import pathlib

        tree = ast.parse(
            pathlib.Path("src/mission/compiler.py").read_text())
        found = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id in ("answered", "settle")
                    and node.args
                    and isinstance(node.args[0], ast.Constant)):
                found.add(node.args[0].value)
        return found

    def test_every_answerable_field_is_one_the_compiler_reads(self):
        from src.workspace.feasibility import OpenItem, Resolution

        consumed = self.consumed_by_the_compiler()
        assert consumed, "the derivation found nothing; the check is vacuous"

        for field in ("asset_identity:SPX", "template:rsu-vesting",
                      "unclear:5-year lookback"):
            item = OpenItem(field, "q", "w", Resolution.REQUIRED_CLARIFICATION)
            assert not item.answerable, (
                f"{field} offers an input the compiler never reads")

    def test_the_plain_fields_are_still_answerable(self):
        """The other direction: excluding too much would take away the
        controls that do work."""
        from src.workspace.feasibility import OpenItem, Resolution

        consumed = self.consumed_by_the_compiler()
        for field in ("amount", "account_type", "cadence"):
            assert field in consumed, f"{field} is no longer read by the compiler"
            item = OpenItem(field, "q", "w", Resolution.REQUIRED_CLARIFICATION)
            assert item.answerable


class TestThePageIsForAUserNotADeveloper:
    """Field identifiers are implementation detail, not page content.

    Every question and inference carried a monospace `controls <field>` line.
    It is genuinely useful — in an HTML name, an accessibility attribute, a
    log, a test — and none of those is the middle of a sentence a person is
    reading.
    """

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

    def test_no_controls_line_reaches_the_page(self, client):
        page = client.get("/workspace/new", params={"describe": UNRUNNABLE})
        assert "controls account_type" not in page.text
        assert ">controls " not in page.text

    def test_the_identifier_is_still_recoverable(self, client):
        """Removing it from the prose must not remove it from the machine:
        a bug report, a screen reader and these tests all need it."""
        page = client.get("/workspace/new", params={"describe": UNRUNNABLE})
        assert 'data-field="account_type"' in page.text
        assert 'name="answer:account_type"' in page.text

    def test_the_rationale_is_present_but_collapsed(self, client):
        """Behind a disclosure, not deleted. A user who wants to know why the
        account type matters must still be able to find out."""
        page = client.get("/workspace/new", params={"describe": UNRUNNABLE})
        assert "<details" in page.text
        assert "Why this matters" in page.text
        assert "Tax treatment changes the result" in page.text

    def test_the_sections_say_what_they_are(self, client):
        page = client.get("/workspace/new", params={"describe": UNRUNNABLE})
        assert "Understood so far" in page.text
        assert "Needs your input" in page.text
