"""Answering makes progress instead of starting over.

The interaction was describe -> compile -> reject -> rewrite. Every answer the
user had already given went with the 422, so a description the compiler almost
understood cost as many rewrites as it had open questions.

The description stays immutable. What accumulates is the amendment set, echoed
back into the form so the next submission carries every earlier answer and the
question list shrinks.
"""
from __future__ import annotations

import html
import re

import pytest
from fastapi.testclient import TestClient

RUNNABLE = "I buy VOO every month and never sell."
BLOCKED = "I buy SPX every month and never sell."


def asked(body):
    return set(re.findall(r'controls ([a-z_]+)</div>', body))


def carried(body):
    return set(re.findall(r'type="hidden" name="answer:([a-z_]+)"', body))


def parse_token(body):
    found = re.search(r'name="parse" value="([^"]*)"', body)
    return html.unescape(found.group(1)) if found else ""


@pytest.fixture
def client(tmp_path, monkeypatch):
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


def submit(client, describe, body, **answers):
    payload = {"describe": describe, "title": "Draft",
               "parse": parse_token(body)}
    payload.update({f"answer:{k}": v for k, v in answers.items()})
    payload.update({f"answer:{k}": v for k, v in
                    {f: "carried" for f in ()}.items()})
    # Replay whatever the page is already carrying, as a browser would.
    for field in carried(body):
        found = re.search(
            rf'name="answer:{field}" value="([^"]*)"', body)
        if found and f"answer:{field}" not in payload:
            payload[f"answer:{field}"] = html.unescape(found.group(1))
    return client.post("/workspace/save", data=payload)


class TestQuestionsShrinkAcrossPasses:
    def test_an_incomplete_plan_re_renders_rather_than_rejecting(self, client):
        body = client.get("/workspace/new", params={"describe": RUNNABLE}).text
        response = submit(client, RUNNABLE, body, amount="500")

        assert response.status_code == 200, (
            "an incomplete plan returned a status the page cannot continue from")

    def test_the_question_list_gets_shorter(self, client):
        body = client.get("/workspace/new", params={"describe": RUNNABLE}).text
        before = asked(body)

        response = submit(client, RUNNABLE, body,
                          amount="500", account_type="TAXABLE")
        after = asked(response.text)

        assert len(after) < len(before), f"{before} -> {after}"
        assert "amount" not in after
        assert "account_type" not in after

    def test_earlier_answers_are_carried_into_the_next_submission(self, client):
        """Without this the page resets: answer three, be asked all eight."""
        body = client.get("/workspace/new", params={"describe": RUNNABLE}).text
        response = submit(client, RUNNABLE, body,
                          amount="500", account_type="TAXABLE")

        assert {"amount", "account_type"} <= carried(response.text)

    def test_two_passes_converge_rather_than_oscillating(self, client):
        """The premise for the whole loop. If pass three re-asked something
        pass two settled, none of the above would mean progress."""
        body = client.get("/workspace/new", params={"describe": RUNNABLE}).text
        first = asked(body)
        second = submit(client, RUNNABLE, body, amount="500").text
        third = submit(client, RUNNABLE, second, account_type="TAXABLE").text

        assert asked(second) <= first
        assert asked(third) <= asked(second)

    def test_a_carried_answer_is_not_also_asked(self, client):
        """A hidden replay beside a live control would submit the field twice
        and the stale value could win.

        A settled answer stops being asked, so this is vacuous on the happy
        path — it passed with the guard removed. The case that discriminates
        is an answer that does *not* settle its field: `_as_amount` rejects a
        malformed value, so the question survives and a hidden copy would sit
        beside the live input.
        """
        body = client.get("/workspace/new", params={"describe": RUNNABLE}).text
        response = submit(client, RUNNABLE, body, amount="not a number")

        assert "amount" in asked(response.text), (
            "the premise: a malformed amount must leave the question open")
        assert not (carried(response.text) & asked(response.text))


class TestABlockedPlanStillContinues:
    def test_an_unpriceable_instrument_re_renders(self, client):
        """"There is no price history for SPX" was the end of the
        interaction. The plan still cannot run; the questions beside it are
        still answerable."""
        body = client.get("/workspace/new", params={"describe": BLOCKED}).text
        response = submit(client, BLOCKED, body, amount="1000")

        assert response.status_code == 200
        assert "no price history" in response.text.lower()

    def test_answers_survive_the_blocker(self, client):
        body = client.get("/workspace/new", params={"describe": BLOCKED}).text
        response = submit(client, BLOCKED, body,
                          amount="1000", account_type="TAXABLE")

        assert {"amount", "account_type"} <= carried(response.text)


class TestTheDescriptionIsNeverRewritten:
    def test_the_original_text_survives_every_pass(self, client):
        body = client.get("/workspace/new", params={"describe": RUNNABLE}).text
        second = submit(client, RUNNABLE, body, amount="500").text
        third = submit(client, RUNNABLE, second, account_type="TAXABLE").text

        assert RUNNABLE in html.unescape(third)


class TestProgressIsReportedByState:
    def test_states_are_distinct_rather_than_one_list(self):
        from src.workspace.feasibility import ItemState, OpenItem, Resolution

        answerable = OpenItem("account_type", "q", "w",
                              Resolution.REQUIRED_CLARIFICATION)
        capability = OpenItem("unclear:x", "q", "w",
                              Resolution.UNSUPPORTED_SEPARABLE)
        blocked = OpenItem("y", "q", "w", Resolution.MATERIAL)

        assert answerable.state is ItemState.NEEDS_ANSWER
        assert capability.state is ItemState.NEEDS_CAPABILITY
        assert blocked.state is ItemState.BLOCKED

    def test_an_unrunnable_plan_does_not_relabel_answerable_items(self):
        """Whether the plan can run belongs in one banner, not in every item."""
        from src.workspace.feasibility import ItemState, OpenItem, Resolution

        item = OpenItem("account_type", "q", "w",
                        Resolution.REQUIRED_CLARIFICATION, executable=False)
        assert item.state is ItemState.NEEDS_ANSWER
