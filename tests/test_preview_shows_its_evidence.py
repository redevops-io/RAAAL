"""The screen where a figure first appears shows what produced it.

The draft page rendered Contributed, Final value and both returns, and nothing
about what the rule did. The full ledger existed — on the saved-plan page, one
confirmation later. So the screen where a user first meets a number carried
none of what makes the number checkable.

That is not a cosmetic gap. Every defect this branch found was invisible in
the returns figure and visible here:

* a stated period never applied  -> the period evaluated is the whole snapshot
* a second funding mode dropped  -> contributions are a fraction of the plan
* a sell leg discarded           -> coverage is short of the declared count
* a crossing read as persistent  -> signal count is several times too high

None of them changes the *shape* of a returns figure, which is why three
materially different plans once returned the same number and read as fine.

Deliberately compact: five measures beside the figure, not a second copy of
the timeline. The claim is that a person can check the number in front of
them, not that the draft page becomes the plan page.
"""
from __future__ import annotations

import re

import pytest

CONTROL = ("I buy $1,000 of SPY whenever it crosses below its 200-day moving "
           "average, over the past five years.")

NO_PERIOD = ("I buy $1,000 of SPY whenever it crosses below its 200-day "
             "moving average.")

MEASURES = ("Period evaluated", "Signals detected", "Purchases executed",
            "Total contributed", "Coverage")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.api as api
    import src.workspace.routes as routes
    from src.deploy.context import bind, resolve, unbind
    from src.workspace.store import WorkspaceStore

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))

    class Refusing:
        def complete(self, *, system, user):
            raise TimeoutError("not answered")

    monkeypatch.setattr(routes, "_parser_client", lambda: Refusing())
    monkeypatch.setattr(routes, "_store", lambda: WorkspaceStore(tmp_path / "w.db"))
    api._bootstrap()
    try:
        yield TestClient(api.app)
    finally:
        unbind()


def draft(http, describe=CONTROL):
    response = http.get("/workspace/new", params={"describe": describe})
    assert response.status_code == 200
    return response.text


class TestThePremise:
    def test_the_draft_shows_a_figure_at_all(self, client):
        """Everything below is about evidence beside a figure. If the draft
        refuses, there is no figure and the file proves nothing."""
        body = draft(client)
        assert "Final value" in body, (
            "the control does not produce a figure on the draft page")


class TestTheEvidenceIsOnThePage:
    @pytest.mark.parametrize("measure", MEASURES)
    def test_each_measure_is_present(self, client, measure):
        assert measure in draft(client), measure

    def test_it_precedes_the_figures(self, client):
        """A reader who has formed a view from the returns will not revise it
        from a table underneath."""
        body = draft(client)
        assert body.index("What produced this figure") < body.index("Final value")

    def test_the_period_is_the_stated_one_not_the_snapshot(self, client):
        """The F1 shape. A page that reported a five-year plan over ten years
        of data said nothing about the period at all."""
        body = draft(client)
        block = body[body.index("Period evaluated"):][:400]
        assert re.search(r"\d{4}-\d{2}-\d{2}\s*to\s*\d{4}-\d{2}-\d{2}", block), \
            block[:200]

    def test_a_plan_with_no_stated_period_says_so(self, client):
        """Rather than printing a range that looks like the user's choice."""
        body = draft(client, NO_PERIOD)
        block = body[body.index("Period evaluated"):][:400]
        assert "whole snapshot" in block or "stated no period" in block, \
            block[:200]

    def test_the_counts_come_from_the_ledger(self, client):
        """Not recomputed by the template. A number computed in two places is
        a number that can disagree with itself."""
        import src.workspace.routes as routes
        from src.workspace.draft import compile_draft

        body = draft(client)
        access = routes._market_data("evidence test")
        run = routes._run(compile_draft(CONTROL, name="p",
                                        context="evidence test").scenario,
                          access, stated_text=CONTROL)
        summary = run["ledger"].summary()
        block = body[body.index("What produced this figure"):][:2000]
        assert str(summary["signals_detected"]) in block
        assert str(summary["purchases_executed"]) in block
        assert f"{float(run['ledger'].total_contributed):,.0f}" in block


class TestItDiscriminates:
    """A summary that printed the same thing for every plan would satisfy the
    tests above and detect nothing."""

    #: Both produce figures. "Two years" and "three years" refuse outright —
    #: the condition never occurs in them — so a comparison against those
    #: would test the refusal path, not the summary.
    SHORTER = ("I buy $1,000 of SPY whenever it crosses below its 200-day "
               "moving average, over the past four years.")

    def test_a_shorter_period_reports_fewer_signals(self, client):
        short = self.SHORTER

        def signals(body):
            block = body[body.index("Signals detected"):][:300]
            found = re.search(r'mono">\s*(\d+)', block)
            assert found, block[:200]
            return int(found.group(1))

        assert signals(draft(client, short)) < signals(draft(client, CONTROL))

    def test_the_period_moves_with_the_sentence(self, client):
        def period(body):
            block = body[body.index("Period evaluated"):][:400]
            found = re.search(r"(\d{4}-\d{2}-\d{2})\s*to", block)
            return found.group(1) if found else None

        five = period(draft(client, CONTROL))
        four = period(draft(client, self.SHORTER))
        assert five and four and five < four, (five, four)

    def test_the_two_readings_of_one_rule_are_distinguishable(self, client):
        """The defect the browser agent found, seen from the page: a crossing
        and a persistent condition over the same window differ by several
        times in signal count, and nothing in the returns figure says so."""
        persistent = ("I buy $1,000 of SPY every day it is below its 200-day "
                      "moving average, over the past five years.")

        def signals(body):
            block = body[body.index("Signals detected"):][:300]
            return int(re.search(r'mono">\s*(\d+)', block).group(1))

        assert signals(draft(client, persistent)) > signals(draft(client, CONTROL))
