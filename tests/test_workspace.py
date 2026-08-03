"""The private workspace, end to end.

The exit criterion is a person, not a component: someone must be able to
describe a scenario in prose, resolve every material ambiguity, run it, see why
each benchmark is or is not comparable, see how many alternatives count as
trials, save it, revisit the exact verdict later, and reach none of it through a
recommendation.

Each test below is one clause of that sentence.
"""
from __future__ import annotations

from tests.market_fixture import NO_MARKET_DATA
import re

import pandas as pd
import pytest

from src.mission import Objective
from src.mission.compiler import compile_scenario
from src.workspace.chain import SCENARIO_CHAIN_ORDER, build_scenario_chain
from src.workspace.store import NotSaveable, WorkspaceStore

VAGUE = "I buy VTI every month."

#: Fully specified: every material choice is stated, so nothing is inferred and
#: nothing is left open. This is what a saveable plan looks like.
COMPLETE = ("I buy $2000 of VTI on the first trading day of every month in my "
            "taxable brokerage account, reinvest the dividends, and never sell.")
#: Names the account. Without one the compiler cannot know the tax treatment,
#: and it now asks rather than defaulting to NONE_APPLIED — which is what made a
#: Roth and a taxable account compare as identical for every plan compiled from
#: prose. This description was never complete; the compiler was not asking.

#: Specified enough to run, but leaning on defaults the user must confirm.
INFERRING = ("I buy $2000 of AMZN and NVDA every month. Whenever SPY is below "
             "its 200-day average, buy more.")

CONTRADICTORY = ("I buy $2000 of AMZN and NVDA every month, rebalance them to "
                 "equal weights, and never sell.")


@pytest.fixture
def pinned_prices(tmp_path, monkeypatch):
    """A deterministic panel, so the guarantees below never skip.

    Both tests here protect product guarantees rather than edge cases: that a
    benchmark set is never reordered by outcome, and that a comparison carries
    its disclosure. Letting them depend on whether a parquet file happens to
    exist means the non-recommendation posture is verified only on machines that
    have run a backtest.

    The panel is monotonic and boring on purpose. These tests assert ordering and
    disclosure, not returns, and a fixture with interesting dynamics would invite
    someone to assert on its numbers.
    """
    import numpy as np
    import pandas as pd
    import src.workspace.routes as routes

    sessions = pd.bdate_range("2021-01-04", periods=260)
    panel = pd.DataFrame(
        {
            "VTI": np.linspace(100.0, 130.0, len(sessions)),
            "SPY": np.linspace(100.0, 125.0, len(sessions)),
            "QQQ": np.linspace(100.0, 140.0, len(sessions)),
            "AGG": np.linspace(100.0, 102.0, len(sessions)),
            "AMZN": np.linspace(100.0, 118.0, len(sessions)),
            "NVDA": np.linspace(100.0, 160.0, len(sessions)),
        },
        index=sessions,
    )
    path = tmp_path / "prices.parquet"
    panel.to_parquet(path)
    monkeypatch.setattr(routes, "PRICES", path)
    return panel


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.api as api
    import src.web.routes as web_routes
    import src.workspace.routes as workspace_routes
    from src.ledger import Ledger

    ledger = Ledger(tmp_path / "public.db")
    monkeypatch.setattr(api, "_ledger", ledger)
    monkeypatch.setattr(web_routes, "Ledger", lambda *a, **k: ledger)

    store = WorkspaceStore(tmp_path / "workspace.db")
    monkeypatch.setattr(workspace_routes, "_store", lambda: store)
    api._bootstrap()
    return TestClient(api.app)


@pytest.fixture
def store(tmp_path):
    return WorkspaceStore(tmp_path / "w.db")


def text(html: str) -> str:
    body = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


class TestProseToConfirmation:
    def test_the_workspace_is_reachable_and_marked_private(self, client):
        page = text(client.get("/workspace/").text)

        assert "Private" in page
        assert "nothing you write becomes part of the public research library" in page

    def test_a_description_produces_the_confirmation_groups(self, client):
        page = text(client.get("/workspace/new", params={"describe": INFERRING}).text)

        assert "understood directly from what you wrote" in page
        assert "Please confirm" in page

    def test_a_fully_specified_plan_infers_nothing(self, client):
        """Asking about choices the user already made is noise, and asking about
        a condition they never described blocks a complete plan on an ambiguity
        that does not exist in it."""
        page = text(client.get("/workspace/new", params={"describe": COMPLETE}).text)

        assert "Please confirm" not in page
        assert "1 question" not in page and "2 questions" not in page
        assert "Ready to save" in page

    def test_a_contradiction_is_shown_and_not_resolved_for_the_user(self, client):
        page = text(client.get("/workspace/new",
                               params={"describe": CONTRADICTORY}).text)

        assert "Choose which you meant" in page
        assert "picking one for you would run a plan you did not describe" in page

    def test_inferences_name_the_versioned_default_set(self, client):
        page = text(client.get("/workspace/new", params={"describe": INFERRING}).text)
        assert "compiler-defaults/us-equity-scenario@1" in page

    def test_an_underspecified_plan_cannot_be_saved_from_the_page(self, client):
        html = client.get("/workspace/new", params={"describe": VAGUE}).text

        assert "Not ready to save" in text(html)
        assert "/workspace/save" not in html, (
            "the save control must be absent, not merely disabled"
        )

    def test_no_yaml_reaches_the_user(self, client):
        """The whole point: prose in, plain language back."""
        page = client.get("/workspace/new", params={"describe": COMPLETE}).text

        for marker in ("spec_version:", "canonical_form", "provenance:", "---\n"):
            assert marker not in page


class TestTheChainReadsLikeTheLibrarys:
    def test_the_scenario_chain_uses_the_same_symbols(self, client):
        page = client.get("/workspace/new", params={"describe": COMPLETE}).text
        assert any(symbol in page for symbol in ("●", "◐", "✕", "○", "·"))

    def test_the_chain_is_in_canonical_order(self):
        compiled = compile_scenario(COMPLETE)
        chain = build_scenario_chain(subject="x", scenario=compiled.scenario)

        assert [l.step for l in chain.links] == [s for s, _ in SCENARIO_CHAIN_ORDER]

    def test_an_unsaved_plan_says_so_in_the_chain(self):
        compiled = compile_scenario(COMPLETE)
        chain = build_scenario_chain(subject="x", scenario=compiled.scenario)
        [scenario_link] = [l for l in chain.links if l.step == "Scenario"]

        assert "provisional" in scenario_link.summary

    def test_a_contradiction_blocks_the_chain(self):
        compiled = compile_scenario(CONTRADICTORY)
        chain = build_scenario_chain(subject="x", scenario=compiled.scenario)

        assert chain.worst.value == "block"
        assert "Blocked at" in chain.headline

    def test_the_trials_link_reports_no_penalty_without_a_search(self):
        compiled = compile_scenario(COMPLETE)
        chain = build_scenario_chain(subject="x", scenario=compiled.scenario)
        [trials] = [l for l in chain.links if l.step == "Trials"]

        assert "no selection penalty" in trials.summary


class TestSavingIsACommitment:
    def _scenario(self, text_in=COMPLETE, confirmed=False):
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance

        compiled = compile_scenario(text_in, name="my-plan",
                                    benchmark_rule="benchmark-policy/public-default@1")
        if not confirmed:
            return compiled.scenario
        p = compiled.scenario.provenance
        return ScenarioSpecification(**{
            **compiled.scenario.__dict__,
            "provenance": Provenance(
                stated=p.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in p.inferred),
                contradictions=p.contradictions, unresolved=p.unresolved),
        })

    def test_an_unconfirmed_plan_is_refused_by_the_store(self, store):
        with pytest.raises(NotSaveable, match="have not made"):
            store.save_plan(plan_id="p", owner="me",
                            scenario=self._scenario(INFERRING),
                            stated_text=INFERRING, saved_at="2026-07-31")

    def test_a_contradictory_plan_is_refused_by_the_store(self, store):
        with pytest.raises(NotSaveable, match="contradicts itself"):
            store.save_plan(
                plan_id="p", owner="me",
                scenario=self._scenario(CONTRADICTORY, confirmed=True),
                stated_text=CONTRADICTORY, saved_at="2026-07-31")

    def test_a_confirmed_plan_saves_and_reloads(self, store):
        store.save_plan(plan_id="p", owner="me",
                        scenario=self._scenario(confirmed=True),
                        stated_text=COMPLETE, saved_at="2026-07-31")
        record = store.get_plan("p", "me")

        assert record["stated_text"] == COMPLETE
        assert record["rule_hash"] and record["content_hash"]

    def test_plans_are_scoped_to_their_owner_at_the_query(self, store):
        """A get that filters ownership afterwards is one early return away from
        serving someone else's plan."""
        store.save_plan(plan_id="p", owner="me",
                        scenario=self._scenario(confirmed=True),
                        stated_text=COMPLETE, saved_at="2026-07-31")

        assert store.get_plan("p", "someone-else") is None
        assert store.list_plans("someone-else") == []

    def test_runs_are_not_reachable_across_owners(self, store):
        store.save_plan(plan_id="p", owner="me",
                        scenario=self._scenario(confirmed=True),
                        stated_text=COMPLETE, saved_at="2026-07-31")
        store.record_run(run_id="r1", plan_id="p", ran_at="2026-07-31",
                         result={"final_value": 1.0,
                                 "market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"modelled": [],
                                                     "not_modelled": []}},
                         comparison={})

        assert len(store.runs_for("p", "me")) == 1
        assert store.runs_for("p", "someone-else") == []

    def test_a_run_without_a_modelling_scope_is_refused(self, store):
        """A stored figure that lost its scope will be read as excluding nothing."""
        store.save_plan(plan_id="p", owner="me",
                        scenario=self._scenario(confirmed=True),
                        stated_text=COMPLETE, saved_at="2026-07-31")

        with pytest.raises(NotSaveable, match="excluding nothing"):
            store.record_run(run_id="r0", plan_id="p", ran_at="2026-07-31",
                             result={"final_value": 1.0}, comparison={})

    def test_a_recorded_run_keeps_the_verdict_it_received(self, store):
        """Same reason the public ledger stores verdicts rather than recomputing."""
        store.save_plan(plan_id="p", owner="me",
                        scenario=self._scenario(confirmed=True),
                        stated_text=COMPLETE, saved_at="2026-07-31")
        store.record_run(run_id="r1", plan_id="p", ran_at="2026-07-31",
                         result={"money_weighted": 0.31,
                                 "market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"modelled": [],
                                                     "not_modelled": []}},
                         comparison={"class": "X"})

        [run] = store.runs_for("p", "me")
        assert run["result"]["money_weighted"] == 0.31
        assert run["comparison"]["class"] == "X"


class TestTheBoundaryHoldsInBothDirections:
    def test_the_public_library_never_links_to_a_plan(self, client):
        for path in ("/ui/", "/ui/findings", "/ui/claims", "/ui/errata"):
            assert "/workspace/" not in client.get(path).text, (
                f"{path} links into the private workspace"
            )

    def test_the_workspace_may_link_to_public_research(self, client):
        assert "/ui/" in client.get("/workspace/").text

    def test_the_workspace_has_its_own_templates(self):
        """Not a flag on the public ones. The separation is in the file tree."""
        from pathlib import Path

        assert (Path("src/workspace/templates/base.html")).exists()
        assert (Path("src/workspace/templates") != Path("src/web/templates"))

    def test_the_workspace_has_its_own_store(self):
        """A shared table with a visibility column is one forgotten predicate
        away from being no boundary at all."""
        from src.workspace.store import DEFAULT_PATH

        assert "workspace" in str(DEFAULT_PATH)
        assert "quantify.db" not in str(DEFAULT_PATH)


class TestNothingRecommends:
    def test_no_prescriptive_language_reaches_any_page(self, client):
        from src.mission import scan_language

        for path, params in (("/workspace/", None),
                             ("/workspace/new", {"describe": COMPLETE}),
                             ("/workspace/new", {"describe": CONTRADICTORY})):
            page = text(client.get(path, params=params).text)
            found = {k: v for k, v in scan_language(page).items() if v}
            assert not found, f"{path} contains {found}"

    def test_benchmarks_are_never_sorted_by_outcome(self, pinned_prices, client):
        """The declared order is fixed before anything runs.

        QQQ outperforms SPY in the fixture, so an outcome-sorted payload would
        put it first. Declaration order puts SPY first.
        """
        page = client.get("/workspace/new", params={"describe": COMPLETE}).text

        assert "Hold cash" in page, "the benchmark set did not render"
        assert page.index("Hold cash") > page.index("Contribute to S&amp;P 500")
        assert page.index("Contribute to S&amp;P 500") < page.index(
            "Contribute to Nasdaq 100"), (
            "the set was ordered by outcome — Nasdaq outperforms in this fixture"
        )

    def test_the_comparison_carries_its_disclosure(self, pinned_prices, client):
        from src.mission import DISCLOSURES

        page = text(client.get("/workspace/new", params={"describe": COMPLETE}).text)

        assert "Compared with" in page, "the comparison did not render"
        assert any(d[:60] in page for d in DISCLOSURES.values())

    def test_both_return_bases_are_reported(self, pinned_prices, client):
        """Neither substitutes for the other, so neither may appear alone."""
        page = text(client.get("/workspace/new", params={"describe": COMPLETE}).text)

        assert "Money-weighted return" in page
        assert "Time-weighted return" in page


class TestStageOneIsPinnedToTheSavedPlan:
    """The whole point of pinning, exercised through the HTTP surface.

    The workspace recompiles a plan from its stated text on every view. With a
    model in stage 1 and nothing pinned, a plan would be reinterpreted each time
    it was opened, against a model that may have changed since — so a user could
    confirm one thing and find another later, with no record that anything moved.
    """

    @staticmethod
    def _saved(tmp_path):
        """The store the client actually writes to."""
        return WorkspaceStore(tmp_path / "workspace.db")

    def test_saving_stores_the_parse_that_was_confirmed(self, client, tmp_path):
        import json as _json

        from src.mission.compiler import parse

        page = client.get("/workspace/new", params={"describe": COMPLETE})
        assert page.status_code == 200

        response = client.post(
            "/workspace/save",
            params={"describe": COMPLETE, "plan_id": "pinned",
                    "confirm_all": "yes"},
            data={"parse": _json.dumps(parse(COMPLETE).to_json())},
            follow_redirects=False)
        assert response.status_code == 303, response.text

        record = self._saved(tmp_path).get_plan("pinned", "pilot")
        assert record is not None
        assert record["parse"]["text"] == COMPLETE
        assert record["parse"]["recognitions"]

    def test_reopening_shows_what_was_saved(self, client, tmp_path):
        """The user-visible half: revisit a plan, see the same interpretation."""
        import json as _json

        from src.mission.compiler import parse

        client.post("/workspace/save",
                    params={"describe": COMPLETE, "plan_id": "revisit",
                            "confirm_all": "yes"},
                    data={"parse": _json.dumps(parse(COMPLETE).to_json())},
                    follow_redirects=False)

        first = client.get("/workspace/plans/revisit")
        second = client.get("/workspace/plans/revisit")
        assert first.status_code == 200
        assert text(first.text) == text(second.text)

    def test_a_tampered_parse_cannot_inject_a_setting(self, client, tmp_path):
        """The hidden field travels through a browser and is not trusted."""
        import json as _json

        client.post("/workspace/save",
                    params={"describe": COMPLETE, "plan_id": "tampered",
                            "confirm_all": "yes"},
                    data={"parse": _json.dumps({
                        "text": COMPLETE,
                        "recognitions": [{"field": "trigger_semantics",
                                          "value": "crossing_event",
                                          "span": "whenever it dips below"}],
                        "assets": [], "unrecognized": [],
                    })},
                    follow_redirects=False)

        record = self._saved(tmp_path).get_plan("tampered", "pilot")
        assert record is not None
        fields = {r["field"] for r in record["parse"]["recognitions"]}
        assert "trigger_semantics" not in fields, (
            "a span the description does not contain must not become a setting, "
            "whatever route it arrived by")

    def test_a_parse_of_different_text_is_rejected_with_422(self, client):
        import json as _json

        response = client.post(
            "/workspace/save",
            params={"describe": COMPLETE, "plan_id": "mismatch",
                    "confirm_all": "yes"},
            data={"parse": _json.dumps({"text": "a different description",
                                        "recognitions": [], "assets": [],
                                        "unrecognized": []})})
        assert response.status_code == 422
        assert "does not match the description" in response.json()["detail"]

    def test_the_front_door_survives_an_unavailable_model(self, client,
                                                          monkeypatch):
        """No key, no network, a timeout — the page still compiles and renders.

        A conversational front door that stops working when an API does is not a
        front door.
        """
        import src.workspace.routes as routes

        class Broken:
            def complete(self, *, system, user):
                raise TimeoutError("upstream timed out")

        monkeypatch.setattr(routes, "_parser_client", lambda: Broken())
        page = client.get("/workspace/new", params={"describe": COMPLETE})
        assert page.status_code == 200
        assert "Here is what we understood" in text(page.text)
