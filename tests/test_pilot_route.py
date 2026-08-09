"""The acceptance invariant, proved through HTTP rather than by reading source.

> A plan created through the pilot workspace must prove that `compile_intent`
> was reached and `compile_scenario` was not.

Source inspection would not prove it. An import can exist and never be called;
a call can be added on a branch the test never takes. So this drives real
requests and establishes both halves by evidence:

    reached      the stored artifact carries `compiled_by=quantify-mission@1`
                 and `compiled_from=<the intent's own hash>`, which nothing but
                 `compile_intent` produces

    not reached  `compile_scenario` is replaced with a function that raises.
                 If the journey completes, the legacy compiler was not called —
                 and the substitution is verified to be effective by a control
                 that calls it and fails.

That control matters more than it looks. Without it, a typo in the patch target
would make every "the legacy path was not reached" assertion pass by patching
nothing.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"


@pytest.fixture
def pilot_client(monkeypatch, tmp_path):
    """An app whose deployment declares the runtime and reads from recordings."""
    from src.deploy import context as deploy_context

    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used-by-recordings")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/pilot.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    return TestClient(app)


def _legacy_must_not_run(monkeypatch):
    """Replace the legacy compiler with something that cannot be mistaken for
    success, and return a probe that proves the replacement took effect."""
    import src.mission.compiler as compiler
    import src.workspace.routes as routes

    def refuse(*args, **kwargs):
        raise AssertionError(
            "compile_scenario was reached on a pilot journey; the deployment "
            "declares RUNTIME and a plan compiled from prose is a different "
            "artifact from one compiled from a pinned intent")

    monkeypatch.setattr(compiler, "compile_scenario", refuse)
    monkeypatch.setattr(routes, "compile_scenario", refuse, raising=False)
    return refuse


class TestTheRuntimeIsReached:
    def test_a_submission_produces_a_plan_compiled_from_an_intent(
            self, pilot_client, monkeypatch):
        _legacy_must_not_run(monkeypatch)

        page = pilot_client.get("/pilot", params={"describe": SENTENCE})
        assert page.status_code == 200
        body = page.text

        # Evidence in the response, not in the import graph.
        assert "quantify-mission@1" in body or "compiled by" in body
        assert "claude-sonnet-5@1" in body

    def test_the_saved_artifact_names_the_intent_it_was_compiled_from(
            self, pilot_client, monkeypatch, tmp_path):
        _legacy_must_not_run(monkeypatch)

        saved = pilot_client.post(
            "/pilot/save",
            data={"describe": SENTENCE, "answer_assets": "VTI"},
            follow_redirects=True)
        assert saved.status_code == 200

        from src.workspace.pilot_store import every_plan

        (plan,) = every_plan()
        derivation = plan["derivation"]
        assert derivation["compiled_by"] == "quantify-mission@1"
        assert derivation["compiled_from"] == plan["intent"]["intent_hash"]

    def test_the_control_proves_the_substitution_works(self, monkeypatch):
        """Without this, a typo in the patch target would make every
        "the legacy compiler was not reached" assertion pass by patching
        nothing at all."""
        _legacy_must_not_run(monkeypatch)

        import src.mission.compiler as compiler

        with pytest.raises(AssertionError, match="compile_scenario was reached"):
            compiler.compile_scenario("anything")


class TestTheLoopThroughHttp:
    def test_a_question_is_asked_before_the_plan_can_run(self, pilot_client):
        page = pilot_client.get("/pilot", params={"describe": SENTENCE})
        assert "assets" in page.text
        assert "needs an answer" in page.text

    def test_answering_it_and_saving_reopens_to_the_same_plan(
            self, pilot_client, monkeypatch):
        _legacy_must_not_run(monkeypatch)

        saved = pilot_client.post(
            "/pilot/save",
            data={"describe": SENTENCE, "answer_assets": "VTI"},
            follow_redirects=False)
        assert saved.status_code == 303
        location = saved.headers["location"]

        first = pilot_client.get(location)
        second = pilot_client.get(location)
        assert first.status_code == 200
        assert "Reopened from the saved plan" in first.text
        assert first.text == second.text, (
            "two reopens of one plan rendered differently; replay from a "
            "pinned intent is the property this page exists to demonstrate")

    def test_reopening_does_not_read_the_sentence_again(
            self, pilot_client, monkeypatch):
        """The reader is replaced with one that raises. If the page still
        renders, nothing on the reopen path consulted it."""
        _legacy_must_not_run(monkeypatch)

        saved = pilot_client.post(
            "/pilot/save",
            data={"describe": SENTENCE, "answer_assets": "VTI"},
            follow_redirects=False)
        location = saved.headers["location"]

        import src.workspace.pilot_routes as pilot_routes

        def no_reader():
            raise AssertionError(
                "a reader was constructed on the reopen path; a plan reopened "
                "by re-reading its sentence is a fresh request wearing an old "
                "name")

        monkeypatch.setattr(pilot_routes, "configured_reader", no_reader)
        page = pilot_client.get(location)
        assert page.status_code == 200
        assert "Reopened from the saved plan" in page.text


class TestWhatTheUserIsShown:
    def test_one_witness_is_reported_as_one_witness(self, pilot_client):
        """`MODEL_ONLY_ACCEPTED`, never `AGREE`. A pilot page displaying
        agreement while running a single reader would be showing the user
        corroboration that does not exist."""
        page = pilot_client.get("/pilot", params={"describe": SENTENCE}).text
        assert "MODEL_ONLY_ACCEPTED" in page
        assert "one reader only" in page

    def test_a_refusal_is_named(self, pilot_client):
        page = pilot_client.get("/pilot", params={"describe": SENTENCE}).text
        assert "will not do" in page or "needs an answer" in page
        assert "assets" in page


class TestTheGate:
    def test_a_deployment_that_has_not_declared_the_runtime_cannot_reach_it(
            self, monkeypatch, tmp_path):
        """Knowing the URL is not a declaration. The route checks rather than
        the mount, so the endpoint stays in the table the boundary sweep
        audits."""
        from src.deploy import context as deploy_context

        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "DETERMINISTIC")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/x.db")
        resolved = deploy_context.resolve(dict(os.environ))
        monkeypatch.setattr(deploy_context, "current", lambda: resolved)

        from src.api import app

        page = TestClient(app).get("/pilot", params={"describe": SENTENCE})
        assert page.status_code == 404
        assert "does not declare" in page.text

    def test_the_runtime_mode_requires_a_model_like_model_assisted_does(self):
        """Adding `RUNTIME` to the enum without adding it to the coherence
        check would have let a deployment declare the pilot interpreter with no
        key, pass the preflight, and refuse every description at request time
        with the startup proof reporting a valid configuration."""
        from src.deploy.context import ModelTarget, ParserMode

        target = ModelTarget(_api_key=None, model=None,
                             mode=ParserMode.RUNTIME, declared=True)
        problems = target.problems()
        assert len(problems) == 2
        assert all("RUNTIME" in p for p in problems)
