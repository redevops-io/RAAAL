"""Failing to compile a declared element raises uncertainty; it never erases it.

Coverage took `declared` for the conditional rule from `is_event_funded` — a
*compiled* output. So when the compiler could not settle a trigger, the plan
declared nothing, coverage reported **1/1, every element executed and
evidenced**, and a buy-and-hold figure was published for a conditional
strategy while the page asked which reading was meant.

    open questions : answer:trigger_semantics, answer:account_type
    Contributed    : $1,000
    Final value    : $3,640
    Coverage       : 1/1

That is the loophole in the honesty gate itself: the harder an element was to
understand, the easier it became for the record to forget the user had said
it. A page that asks about the trigger and prints a return has answered its
own question.

    declared means the user expressed it,
    not that the compiler successfully instantiated it

`declared` now comes from the description; `compiled` from what the compiler
made of it. The two are separate columns and always were — only the first was
being filled from the wrong place.

**What is deliberately not counted.** A compiler-raised question is not a
declaration. The first attempt counted every unanswered material field, so a
first submission with `funding_source` open — nearly all of them — published
no figure at all, and the ask-and-refine loop the product is built on stopped
working. Blocking is driven by what the user's own sentence states: a period,
a condition, a second funding source, a sell leg.
"""
from __future__ import annotations

import pytest

CONTESTED = ("I buy $1,000 of SPY whenever it crosses below and stays below "
             "its 200-day moving average, over the past five years.")

SETTLED = ("I buy $1,000 of SPY whenever it crosses below its 200-day moving "
           "average, over the past five years.")

FIRST_ROUND = ("I buy $1,000 of SPY whenever it crosses below its 200-day "
               "moving average, over the past five years.")


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
    monkeypatch.setattr(routes, "_store",
                        lambda: WorkspaceStore(tmp_path / "w.db"))
    api._bootstrap()
    try:
        yield TestClient(api.app)
    finally:
        unbind()


def draft(http, describe):
    response = http.get("/workspace/new", params={"describe": describe})
    assert response.status_code == 200
    return response.text


def coverage_for(describe, *, excluded=()):
    """The record the route builds, via the route's own inputs."""
    import src.mission.coverage as coverage_module
    import src.workspace.routes as routes
    from src.workspace.draft import compile_draft

    scenario = compile_draft(describe, name="p", context="coverage").scenario
    access = routes._market_data("coverage")
    run = routes._run(scenario, access, stated_text=describe)
    resolved = routes._resolve_window(scenario, access.frame)
    return coverage_module.assess(
        scenario, stated_text=describe, resolved_window=resolved,
        frame_sessions=len(access.frame), ledger=run.get("ledger"),
        excluded_items=excluded)


@pytest.fixture
def deployment(monkeypatch):
    from src.deploy.context import bind, resolve, unbind

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    try:
        yield
    finally:
        unbind()


class TestThePremise:
    def test_the_contested_sentence_compiles_no_trigger(self, deployment):
        """Otherwise there is nothing for coverage to forget."""
        from src.workspace.draft import compile_draft

        scenario = compile_draft(CONTESTED, name="p",
                                 context="coverage").scenario
        assert not scenario.is_event_funded
        assert "trigger_semantics" in [one.field for one
                                       in scenario.provenance.unresolved]

    def test_the_sentence_plainly_describes_a_condition(self, deployment):
        """`declared` must be readable from the words alone, because the
        compiled artifact is exactly what is missing."""
        from src.mission.coverage import _CONDITIONAL_PURCHASE

        assert _CONDITIONAL_PURCHASE.search(CONTESTED)


class TestTheContestedTriggerIsCounted:
    def test_it_appears_in_coverage(self, deployment):
        record = coverage_for(CONTESTED)
        found = {one.element_id: one for one in record.elements}
        element = found["event_triggered_funding"]
        assert element.declared, (
            "the user described a conditional purchase and coverage does not "
            "know it was declared")
        assert not element.compiled

    def test_coverage_is_incomplete(self, deployment):
        record = coverage_for(CONTESTED)
        assert not record.publishable
        assert "event_triggered_funding" in [one.element_id
                                             for one in record.blocking]

    def test_the_refusal_names_the_element(self, deployment):
        """"This cannot be shown" sends a user nowhere. Naming the unsettled
        reading tells them what to answer."""
        assert "condition" in coverage_for(CONTESTED).refusal().lower()

    def test_no_figure_is_published(self, client):
        body = draft(client, CONTESTED)
        assert "Final value" not in body, (
            "a figure was published for a plan whose conditional rule was "
            "never settled")

    def test_the_question_is_still_asked(self, client):
        """Blocking the figure must not remove the way forward."""
        assert "answer:trigger_semantics" in draft(client, CONTESTED)


class TestTheUserMayProceedWithoutIt:
    """The escape hatch, and the reason `EXCLUDED_BY_USER` exists. Coverage
    must be able to complete without pretending the element executed."""

    def test_an_authorized_exclusion_unblocks(self, deployment):
        record = coverage_for(CONTESTED,
                              excluded=("event_triggered_funding",))
        found = {one.element_id: one for one in record.elements}
        assert found["event_triggered_funding"].state.value == \
            "EXCLUDED_BY_USER"

    def test_it_is_not_recorded_as_executed(self, deployment):
        """The distinction the state model exists for. A plan proceeding
        without its rule is a smaller plan, and the record says so."""
        record = coverage_for(CONTESTED,
                              excluded=("event_triggered_funding",))
        element = {one.element_id: one for one in record.elements}[
            "event_triggered_funding"]
        assert not element.executed
        assert not element.evidenced


class TestTheOrdinaryPathIsUntouched:
    """A gate that blocked the working case would be removed within a week."""

    def test_the_settled_control_still_publishes(self, client):
        assert "Final value" in draft(client, SETTLED)

    def test_the_settled_control_covers_everything(self, deployment):
        record = coverage_for(SETTLED)
        assert record.publishable, [one.element_id
                                    for one in record.blocking]

    def test_an_open_compiler_question_does_not_block(self, client):
        """`funding_source` and `account_type` are asked on a first
        submission. They are questions this compiler raised, not things the
        user declared, and counting them stopped every provisional figure."""
        body = draft(client, FIRST_ROUND)
        assert "answer:" in body, "premise: this draft still asks something"
        assert "Final value" in body
