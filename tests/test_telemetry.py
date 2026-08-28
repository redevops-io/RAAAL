"""§10 — website / API funnel telemetry.

Two properties matter and are asserted here: the funnel events fire at their real
routes, and no event ever carries raw strategy text. Plus the operational facts
§10 names where they are cheaply true — that a telemetry failure never fails the
request, and that the save-without-recompute rate is a structural 100%.

The client mirrors `test_exact_save`: the runtime and an identity provider are
declared, sign-in is a header the patched `signed_in` reads, and the recorded
reader answers from fixtures so nothing reaches a provider.
"""
from __future__ import annotations

import json
import os
import urllib.parse

import pytest

pytest.importorskip("jwt", reason="pyjwt is required to verify OIDC tokens")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"
ANSWER = {"describe": SENTENCE, "answer_assets": "VTI"}
SUBJECT_HEADER = "x-test-subject"


@pytest.fixture(autouse=True)
def _clean_state():
    from src.workspace import evaluation_session, telemetry
    from src.workspace.abuse import reset_rate_limits

    evaluation_session.clear()
    reset_rate_limits()
    telemetry.reset()
    yield
    evaluation_session.clear()
    reset_rate_limits()
    telemetry.reset()


@pytest.fixture
def client(monkeypatch, tmp_path):
    from src.deploy import context as deploy_context

    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used-by-recordings")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/pilot.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.deploy.context import IdentityTarget

    target = IdentityTarget(issuer="https://auth.example.test",
                            audience="client-1", client_id="client-1")
    monkeypatch.setattr("src.workspace.auth_routes._target", lambda: target)

    from src.deploy.identity import Identity
    import src.workspace.auth_routes as auth_routes

    def signed_in(request):
        subject = request.headers.get(SUBJECT_HEADER)
        return Identity(subject=subject, email=f"{subject}@x.test") \
            if subject else None

    monkeypatch.setattr(auth_routes, "signed_in", signed_in)

    from src.api import app

    return TestClient(app, base_url="https://testserver", follow_redirects=False)


def _as(subject):
    return {SUBJECT_HEADER: subject}


def _no_prompt_text(event: dict) -> bool:
    """No field of an event carries the strategy sentence."""
    return SENTENCE not in json.dumps(event)


# --- the funnel events fire at their routes, text-free ---------------------

class TestFunnelEventsFireAtTheirRoutes:
    def test_evaluator_opened_on_evaluate_get(self, client):
        from src.workspace import telemetry

        client.get("/evaluate")
        opened = telemetry.events(telemetry.EVALUATOR_OPENED)
        assert opened, "no evaluator_opened event fired on /evaluate GET"
        assert opened[-1]["route"] == "/evaluate"

    def test_prompt_submitted_on_pilot_answer_carries_no_prompt_text(self, client):
        from src.workspace import telemetry

        client.post("/pilot/answer", data=ANSWER)
        submitted = telemetry.events(telemetry.PROMPT_SUBMITTED)
        assert submitted, "no prompt_submitted event fired on /pilot/answer"
        event = submitted[-1]
        # ids/counts/hashes only — a digest and a length, never the words.
        assert event["prompt_len"] == len(SENTENCE)
        assert event["prompt_sha"] and len(event["prompt_sha"]) == 16
        assert _no_prompt_text(event), "the raw prompt reached the funnel"

    def test_save_clicked_on_evaluate_save(self, client):
        from src.workspace import telemetry

        review_id = client.post("/evaluate", data=ANSWER
                                ).headers["location"].rsplit("/", 1)[-1]
        client.post("/evaluate/save", data={"review_id": review_id})
        clicked = telemetry.events(telemetry.SAVE_CLICKED)
        assert clicked, "no save_clicked event fired on /evaluate/save"
        assert clicked[-1]["route"] == "/evaluate/save"
        assert all(_no_prompt_text(e) for e in clicked)

    def test_plan_saved_on_the_resume_bind(self, client):
        from src.workspace import telemetry

        review_id = client.post("/evaluate", data=ANSWER
                                ).headers["location"].rsplit("/", 1)[-1]
        started = client.post("/evaluate/save", data={"review_id": review_id})
        nxt = urllib.parse.parse_qs(
            urllib.parse.urlparse(started.headers["location"]).query)["next"][0]
        client.get(nxt, headers=_as("user-a"))

        saved = telemetry.events(telemetry.PLAN_SAVED)
        assert saved, "no plan_saved event fired on the resume bind"
        assert saved[-1]["recomputed"] is False
        assert all(_no_prompt_text(e) for e in saved)

    def test_advisor_viewed_is_recorded_by_path_not_by_a_route_here(self, client):
        """The advisor-viewed event is emitted by a path-based middleware, not by
        a `/for-advisors` route this branch owns — the real page lands on the
        Gate-4 branch. So the route 404s here, yet a GET of the path still records
        the view, which is what keeps §10's funnel working once the real page
        ships without this branch defining (and colliding on) the route."""
        from src.workspace import telemetry

        assert client.get("/for-advisors").status_code == 404, (
            "this branch must not define /for-advisors — the Gate-4 branch owns "
            "the route, its template and its manifest entry")
        viewed = telemetry.events(telemetry.ADVISOR_VIEWED)
        assert viewed, "the middleware did not record advisor_viewed for the path"
        assert viewed[-1]["route"] == "/for-advisors"

    def test_no_funnel_event_anywhere_carries_the_prompt(self, client):
        """A sweep across the whole journey: after evaluating, submitting,
        saving and viewing, not one recorded event contains the sentence."""
        from src.workspace import telemetry

        client.get("/evaluate")
        review_id = client.post("/pilot/answer", data=ANSWER
                                ).headers["location"].rsplit("/", 1)[-1]
        client.post("/evaluate/save", data={"review_id": review_id})
        client.get("/for-advisors")

        recorded = telemetry.events()
        assert recorded, "the journey recorded nothing at all"
        offenders = [e for e in recorded if not _no_prompt_text(e)]
        assert offenders == [], f"events carried raw prompt text: {offenders}"


# --- best-effort: a raising emitter never fails the request ----------------

class TestTelemetryIsBestEffort:
    def test_emit_swallows_an_internal_failure(self, monkeypatch):
        from src.workspace import telemetry

        def boom(_):
            raise RuntimeError("sink is on fire")

        monkeypatch.setattr(telemetry, "_scrub", boom)
        # Must not raise, despite the internal failure.
        telemetry.emit(telemetry.RESEARCH_VIEW, route="/research")

    def test_a_raising_emitter_does_not_fail_the_request(self, client,
                                                         monkeypatch):
        from src.workspace import telemetry

        def boom(_):
            raise RuntimeError("sink is on fire")

        monkeypatch.setattr(telemetry, "_scrub", boom)
        # The route calls emit; emit swallows the failure and the page is served.
        assert client.get("/evaluate",
                          params={"describe": SENTENCE}).status_code == 200


# --- operational metric: save-without-recompute is a structural 100% -------

class TestSaveWithoutRecomputeRate:
    def test_it_is_one_hundred_percent(self, client):
        from src.workspace import telemetry

        review_id = client.post("/evaluate", data=ANSWER
                                ).headers["location"].rsplit("/", 1)[-1]
        started = client.post("/evaluate/save", data={"review_id": review_id})
        nxt = urllib.parse.parse_qs(
            urllib.parse.urlparse(started.headers["location"]).query)["next"][0]
        client.get(nxt, headers=_as("user-a"))

        assert telemetry.save_without_recompute_rate() == 1.0, (
            "a save recomputed the strategy — the exact-save invariant carries "
            "recomputed=False on every plan_saved, so the rate must be 100%")

    def test_it_is_undefined_before_any_save(self):
        from src.workspace import telemetry

        assert telemetry.save_without_recompute_rate() is None
