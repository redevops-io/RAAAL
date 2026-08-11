"""Confirming an inference must not discard the rest of the provenance.

`_with_decisions` marks the inferences a user agreed to. It did so by
constructing a new `Provenance` and naming five of its eight fields, so every
plan whose owner confirmed anything was stored without `excluded`,
`asset_resolutions` or `time_window` — the three added after that function was
written.

Found in production, not here. The control journey saved, ran, and rendered
its ledger; nothing failed. The stored plan simply had `"time_window": null`
against a description that says *over the past five years*, and no test asked.

Two tests, deliberately different in kind:

* one names the three fields that were actually lost, so the defect has a
  regression test that reads as what it is;
* one enumerates `Provenance`'s fields from the dataclass itself, so a ninth
  field added tomorrow is covered without anyone remembering to come back.

The second is the one that matters. This is the second instance of the same
shape — `Provenance.to_json` dropped four of eight before `3eaa5eb` — and
enumerating by hand is what both had in common.
"""
from __future__ import annotations

import dataclasses

import pytest

from src.mission.spec import (
    AssetResolution,
    Contradiction,
    Inference,
    Provenance,
    ScenarioAmendment,
    ScenarioExclusion,
    Unresolved,
)
from src.workspace.routes import _with_decisions

#: Every field carries a distinguishable value, so a dropped one shows up as a
#: default rather than coinciding with what it should have been.
FULL = Provenance(
    stated=("whenever it crosses below", "$1,000"),
    inferred=(Inference("moving_average_kind", "simple", "because", False),
              Inference("signal_series", "close", "because", False)),
    contradictions=(Contradiction(between=("monthly", "event-triggered"),
                                  detail="one funding policy per scenario"),),
    unresolved=(Unresolved(field="funding_source",
                           question="Does this buy come from new money?",
                           why_it_matters="it decides the return basis"),),
    amended=(ScenarioAmendment(question_id="account_type", answer="TAXABLE",
                               recorded_at="t"),),
    excluded=(ScenarioExclusion(item="employer matching",
                                reason="no representation"),),
    asset_resolutions=(AssetResolution(observed_phrase="SP500 ETF",
                                       registry_digest="d1",
                                       chosen_instrument_id="SPY"),),
    time_window=object(),
)


@pytest.fixture
def scenario():
    """A real compiled scenario, with `FULL` put in place of its provenance.

    Built rather than hand-constructed so the function under test receives the
    type it receives in production.
    """
    from src.deploy.context import bind, resolve, unbind
    from src.workspace.draft import compile_draft

    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    try:
        compiled = compile_draft(
            "I buy $1,000 of SPY whenever it crosses below its 200-day "
            "moving average, over the past five years.",
            name="p", context="confirmation test")
        yield dataclasses.replace(compiled.scenario, provenance=FULL)
    finally:
        unbind()


class TestThePremise:
    def test_the_fixture_populates_every_provenance_field(self):
        """Otherwise a field could be 'preserved' only because it was empty on
        both sides, and the test would pass over the defect."""
        for field in dataclasses.fields(Provenance):
            value = getattr(FULL, field.name)
            assert value, f"{field.name} is empty; it cannot show a loss"

    def test_something_is_actually_confirmed(self, scenario):
        """`_with_decisions` returns its input unchanged when nothing was
        agreed, which is the path that cannot exhibit the defect."""
        assert any(not one.confirmed for one in scenario.provenance.inferred)


class TestConfirmationChangesOnlyTheConfirmations:
    AGREED = {"moving_average_kind"}

    def test_the_agreed_inference_is_marked(self, scenario):
        after = _with_decisions(scenario, self.AGREED)
        marked = {one.field: one.confirmed for one in after.provenance.inferred}
        assert marked["moving_average_kind"] is True

    def test_an_inference_nobody_acted_on_stays_unconfirmed(self, scenario):
        """Silence is not agreement. This is the rule the whole confirmation
        screen exists to enforce, so it must survive the fix."""
        after = _with_decisions(scenario, self.AGREED)
        marked = {one.field: one.confirmed for one in after.provenance.inferred}
        assert marked["signal_series"] is False

    @pytest.mark.parametrize("field", ["time_window", "asset_resolutions",
                                       "excluded"])
    def test_the_three_that_were_lost_survive(self, scenario, field):
        """Named individually because these are the ones production lost, and
        a regression test should say which."""
        after = _with_decisions(scenario, self.AGREED)
        assert getattr(after.provenance, field) == getattr(FULL, field), (
            f"confirming an inference discarded {field}")

    def test_every_field_but_inferred_is_untouched(self, scenario):
        """Enumerated from the dataclass, not from a list in this file. A
        ninth field is covered the day it is added — which is exactly what did
        not happen the first three times."""
        after = _with_decisions(scenario, self.AGREED)
        for one in dataclasses.fields(Provenance):
            if one.name == "inferred":
                continue
            assert getattr(after.provenance, one.name) == \
                getattr(FULL, one.name), f"{one.name} did not survive"

    def test_nothing_else_about_the_scenario_moves(self, scenario):
        after = _with_decisions(scenario, self.AGREED)
        assert after.allocation_rule == scenario.allocation_rule
        assert after.funding == scenario.funding
        assert after.flow_schedule == scenario.flow_schedule

    def test_no_agreement_returns_the_scenario_itself(self, scenario):
        assert _with_decisions(scenario, set()) is scenario


class TestTheSavedPlanKeepsItsWindow:
    """The end-to-end shape of the production defect: a description with a
    stated period, confirmed through the routes, read back from the store."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
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

        store = WorkspaceStore(tmp_path / "w.db")
        monkeypatch.setattr(routes, "_parser_client", lambda: Refusing())
        monkeypatch.setattr(routes, "_store", lambda: store)
        api._bootstrap()
        try:
            yield TestClient(api.app), store
        finally:
            unbind()

    def test_a_stated_period_survives_confirmation(self, client):
        import html as html_module
        import re

        http, store = client
        describe = ("I buy $1,000 of SPY whenever it crosses below its "
                    "200-day moving average, over the past five years.")
        body = http.get("/workspace/new", params={"describe": describe}).text
        payload = {"describe": describe, "title": "window"}
        payload.update({name: html_module.unescape(value) for name, value in
                        re.findall(r'<input type="hidden" name="([^"]+)" '
                                   r'value="([^"]*)"', body)})
        response = None
        for _ in range(4):
            selects = dict(re.findall(
                r'<select[^>]*name="(answer:[^"]+|confirm:[^"]+)"[^>]*>'
                r'(.*?)</select>', body, re.S))
            boxes = dict(re.findall(
                r'<input type="checkbox" name="(confirm:[^"]+)"\s+'
                r'value="([^"]*)"', body))
            if not selects and not boxes:
                break
            for name, block in selects.items():
                options = [html_module.unescape(one) for one in
                           re.findall(r'<option value="([^"]*)"', block) if one]
                if options:
                    payload[name] = options[0]
            for name, value in boxes.items():
                payload[name] = html_module.unescape(value)
            response = http.post("/workspace/save", data=payload,
                                 follow_redirects=False)
            if response.status_code in (302, 303):
                break
            body = response.text
            payload.update({n: html_module.unescape(v) for n, v in
                            re.findall(r'<input type="hidden" name="([^"]+)" '
                                       r'value="([^"]*)"', body)})

        assert response is not None and response.status_code in (302, 303), (
            "the journey did not save, so it says nothing about what a saved "
            "plan retains")
        plan_id = response.headers["location"].rsplit("/", 1)[-1]
        stored = store.get_plan(plan_id, "pilot")["scenario"]

        assert any(key.startswith("confirm:") for key in payload), (
            "nothing was confirmed on this journey, so `_with_decisions` "
            "returned its input and the defect could not appear")
        assert (stored.get("provenance") or {}).get("time_window"), (
            "the saved plan has no time window for a description that states "
            "one; this is what production had")
