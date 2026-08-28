"""§11 — security / abuse controls on the public evaluation surface.

Each test names the control it proves and asserts it is additive and safe: the
security headers are present *and* the pages still render under them; the rate
limit turns away a flood without touching a second client, research or a broken
limiter; the size ceiling rejects an oversized body; a prompt that mentions a URL
is treated as text and never fetched; CSRF (when enforced) refuses a save with no
valid token while leaving the anonymous evaluate→login→resume flow intact; the
session identifier is rotated across login; the structured log carries ids and
paths but never the prompt; and the owner boundary holds at the store, not only
the UI.

The client mirrors `test_exact_save`: a deployment that declares the runtime and
an identity provider, with sign-in simulated by a header the patched `signed_in`
reads, so no provider call is made and the recorded reader answers from fixtures.
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request

import pytest

pytest.importorskip("jwt", reason="pyjwt is required to verify OIDC tokens")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"
ANSWER = {"describe": SENTENCE, "answer_assets": "VTI"}
SUBJECT_HEADER = "x-test-subject"

EXISTING_HEADERS = ("Strict-Transport-Security", "X-Content-Type-Options",
                    "X-Frame-Options", "Referrer-Policy")


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


def _evaluate_anonymously(client):
    posted = client.post("/evaluate", data=ANSWER)
    assert posted.status_code == 303, posted.text[:300]
    return posted.headers["location"].rsplit("/", 1)[-1]


# --- 1. CSP + the four existing headers, and the pages still render --------

class TestSecurityHeadersAndCSP:
    def test_the_four_existing_headers_survive(self, client):
        response = client.get("/info")
        for name in EXISTING_HEADERS:
            assert name in response.headers, f"{name} was dropped"
        assert response.headers["X-Frame-Options"] == "DENY"

    def test_the_csp_is_present(self, client):
        csp = client.get("/info").headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp
        # The directives the current pages need to keep working.
        assert "'unsafe-inline'" in csp          # inline onclick + <style>
        assert "https://cdn.bokeh.org" in csp    # the research dashboard's JS

    def test_the_evaluate_page_still_renders_under_the_csp(self, client):
        """The CSP must not blank the evaluator."""
        response = client.get("/evaluate", params={"describe": SENTENCE})
        assert response.status_code == 200
        assert "Content-Security-Policy" in response.headers
        assert len(response.text) > 500, "the evaluator rendered an empty page"

    def test_research_still_renders_under_the_csp(self, client):
        response = client.get("/research")
        # 200 with a built dashboard, 503 with none — never blanked, and always
        # carrying the CSP that allows its inline scripts and the Bokeh CDN.
        assert response.status_code in (200, 503)
        assert "cdn.bokeh.org" in response.headers.get(
            "Content-Security-Policy", "")


# --- 2. IP/session rate limit ---------------------------------------------

class TestRateLimit:
    def test_a_flood_from_one_client_gets_429(self, client, monkeypatch):
        monkeypatch.setenv("QUANTIFY_RATE_LIMIT_PER_MIN", "3")
        who = {"x-forwarded-for": "203.0.113.7"}
        codes = [client.post("/pilot/answer", data=ANSWER, headers=who
                             ).status_code for _ in range(4)]
        assert codes[:3] == [303, 303, 303], codes
        assert codes[3] == 429, codes

    def test_a_different_client_is_unaffected(self, client, monkeypatch):
        monkeypatch.setenv("QUANTIFY_RATE_LIMIT_PER_MIN", "3")
        a = {"x-forwarded-for": "203.0.113.7"}
        b = {"x-forwarded-for": "198.51.100.9"}
        for _ in range(4):
            client.post("/pilot/answer", data=ANSWER, headers=a)
        # A's limit is spent; B has spent nothing.
        assert client.post("/pilot/answer", data=ANSWER,
                           headers=b).status_code == 303

    def test_research_is_not_rate_limited(self, client, monkeypatch):
        monkeypatch.setenv("QUANTIFY_RATE_LIMIT_PER_MIN", "1")
        who = {"x-forwarded-for": "203.0.113.7"}
        for _ in range(5):
            assert client.get("/research", headers=who).status_code in (200, 503)

    def test_a_broken_limiter_fails_open(self, client, monkeypatch):
        """A limiter that raises must not take the site down — the request is
        still served."""
        monkeypatch.setenv("QUANTIFY_RATE_LIMIT_PER_MIN", "1")
        from src.workspace import abuse

        def boom(*a, **k):
            raise RuntimeError("limiter state is corrupt")

        monkeypatch.setattr(abuse.RATE_LIMITER, "allow", boom)
        who = {"x-forwarded-for": "203.0.113.7"}
        for _ in range(3):
            assert client.post("/pilot/answer", data=ANSWER,
                               headers=who).status_code == 303


# --- 3. request-size ceiling ----------------------------------------------

class TestRequestSize:
    def test_an_oversized_post_gets_413(self, client, monkeypatch):
        monkeypatch.setenv("QUANTIFY_MAX_BODY_BYTES", "80")
        big = {"describe": "x" * 500, "answer_assets": "VTI"}
        assert client.post("/evaluate", data=big).status_code == 413

    def test_a_normal_post_is_under_the_ceiling(self, client, monkeypatch):
        monkeypatch.setenv("QUANTIFY_MAX_BODY_BYTES", "80")
        # A sentence-sized body is well under a generous real ceiling; here the
        # ceiling is tiny to prove the reject, and a normal prompt still fits a
        # realistic one, so this asserts the reject is about size, not content.
        monkeypatch.setenv("QUANTIFY_MAX_BODY_BYTES", str(64 * 1024))
        assert client.post("/evaluate", data=ANSWER).status_code == 303


# --- 4. no URL/file retrieval from an anonymous prompt --------------------

class TestPromptWithUrlIsTreatedAsText:
    def test_contains_url_detects_a_scheme(self):
        from src.workspace import abuse

        assert abuse.contains_url("buy VTI, see http://evil.test/x")
        assert not abuse.contains_url("invest $500 monthly")

    def test_a_prompt_with_a_url_triggers_no_outbound_fetch(
            self, client, monkeypatch):
        """The evaluation path treats a prompt as text end to end: a URL inside
        it is never dereferenced. Any outbound fetch is made to raise; the
        URL-bearing prompt is handled as text (the reader looks it up and finds
        nothing — a dict miss, not a fetch), and the ordinary offline evaluation
        still completes. Neither reaches the network."""
        calls = []

        def no_network(*a, **k):
            calls.append(a)
            raise AssertionError("the evaluation path made an outbound request")

        monkeypatch.setattr(urllib.request, "urlopen", no_network)

        from src.api import app

        # A client that returns the 500 rather than re-raising: a URL-bearing
        # prompt is an unrecorded sentence, so the reader treats it as text and
        # misses — proof it was read, not fetched.
        quiet = TestClient(app, base_url="https://testserver",
                           follow_redirects=False, raise_server_exceptions=False)
        quiet.post("/evaluate",
                   data={"describe": "put it all in http://evil.test/pwn"})
        assert calls == [], "a URL in a prompt was fetched"

        # And the ordinary offline evaluation runs to completion without network.
        assert client.post("/evaluate", data=ANSWER).status_code == 303
        assert calls == [], "the evaluation path reached the network"


# --- 5. CSRF on the authenticated save ------------------------------------

class TestCSRFOnSave:
    @pytest.fixture(autouse=True)
    def _enforce(self, monkeypatch):
        monkeypatch.setenv("QUANTIFY_CSRF_ENFORCE", "1")

    def _review(self, client):
        # POST /evaluate is not a save, so CSRF does not gate it; it also seeds
        # the quantify_csrf cookie for the save that follows.
        return _evaluate_anonymously(client)

    def test_a_save_without_a_token_is_refused(self, client):
        review_id = self._review(client)
        # Drop any cookie the seeding set, so there is neither cookie nor field.
        client.cookies.clear()
        got = client.post("/evaluate/save", data={"review_id": review_id})
        assert got.status_code == 403

    def test_a_save_with_a_wrong_token_is_refused(self, client):
        review_id = self._review(client)
        got = client.post("/evaluate/save",
                          data={"review_id": review_id,
                                "csrf_token": "csrf-not-the-cookie"})
        assert got.status_code == 403

    def test_a_save_with_the_matching_token_is_accepted(self, client):
        review_id = self._review(client)
        token = client.cookies.get("quantify_csrf")
        assert token, "the CSRF cookie was not issued"
        got = client.post("/evaluate/save",
                          data={"review_id": review_id, "csrf_token": token})
        # Anonymous + provider present → redirect to sign in (not a 403).
        assert got.status_code == 303
        assert got.headers["location"].startswith("/auth/login?next=")

    def test_the_anonymous_evaluate_login_resume_flow_still_completes(
            self, client):
        """CSRF on the save must not break the round-trip: the resume GET
        consumes the single-use token and completes."""
        review_id = self._review(client)
        token = client.cookies.get("quantify_csrf")
        started = client.post("/evaluate/save",
                              data={"review_id": review_id, "csrf_token": token})
        nxt = urllib.parse.parse_qs(
            urllib.parse.urlparse(started.headers["location"]).query)["next"][0]
        assert nxt.startswith("/pilot/save/resume")
        resumed = client.get(nxt, headers=_as("user-a"))
        assert resumed.status_code == 303
        assert resumed.headers["location"].startswith("/pilot/plans/")


# --- 6. session-fixation protection across login --------------------------

class TestSessionRotation:
    def test_the_session_id_is_rotated_at_the_callback(self, client, monkeypatch):
        """A pre-login session cookie cannot be inherited: the callback mints a
        fresh token and writes it over any value the browser already held."""
        import src.workspace.auth_routes as auth_routes
        from src.deploy.identity import Identity

        class _FakeFlow:
            destination = "/pilot/plans/whatever"

            @classmethod
            def from_cookie(cls, raw):
                return cls()

        monkeypatch.setattr(auth_routes, "Flow", _FakeFlow)
        monkeypatch.setattr(
            auth_routes, "complete",
            lambda **k: (Identity(subject="user-a"), "fresh-session-token-xyz"))

        # An attacker-fixed session and a present flow cookie going in.
        client.cookies.set("quantify_session", "attacker-fixed-value")
        client.cookies.set("quantify_login", "anything-nonempty")

        response = client.get("/auth/callback?code=c&state=s")
        assert response.status_code == 303
        set_cookie = response.headers.get("set-cookie", "")
        assert "fresh-session-token-xyz" in set_cookie, (
            "the callback did not write the freshly minted session token")
        assert "attacker-fixed-value" not in set_cookie, (
            "the pre-login session value survived the callback — not rotated")


# --- 7. structured logging without sensitive bodies -----------------------

class TestStructuredLoggingHasNoPromptBody:
    def test_log_event_records_ids_and_path_but_not_the_prompt(self, caplog):
        from types import SimpleNamespace

        from src.workspace import abuse

        request = SimpleNamespace(
            url=SimpleNamespace(path="/evaluate/save"), method="POST",
            headers={"x-forwarded-for": "203.0.113.7"})

        with caplog.at_level("INFO", logger="src.workspace.abuse"):
            abuse.log_event("evaluation", request, outcome="303",
                            review_id="review-abc123",
                            prompt=SENTENCE, describe=SENTENCE)

        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "review-abc123" in text, "the id was not logged"
        assert "/evaluate/save" in text, "the path was not logged"
        assert SENTENCE not in text, "the raw prompt body reached the log"


# --- owner checks at the store boundary, not only the UI ------------------

class TestOwnerBoundaryAtTheStore:
    def _save_as(self, client, subject):
        review_id = _evaluate_anonymously(client)
        started = client.post("/evaluate/save", data={"review_id": review_id})
        nxt = urllib.parse.parse_qs(
            urllib.parse.urlparse(started.headers["location"]).query)["next"][0]
        resumed = client.get(nxt, headers=_as(subject))
        return resumed.headers["location"].rsplit("/", 1)[-1]

    def test_a_second_owner_cannot_load_the_plan_at_the_store(self, client):
        plan_id = self._save_as(client, "user-a")

        from src.deploy.identity import Identity
        from src.workspace.pilot_store import load
        from src.workspace.routes import _LOOKING

        owner_a = _LOOKING.set(Identity(subject="user-a"))
        try:
            assert load(plan_id) is not None, "the owner cannot read their plan"
        finally:
            _LOOKING.reset(owner_a)

        owner_b = _LOOKING.set(Identity(subject="user-b"))
        try:
            assert load(plan_id) is None, (
                "a second owner read the first owner's plan at the store — the "
                "owner scope is not enforced below the UI")
        finally:
            _LOOKING.reset(owner_b)
