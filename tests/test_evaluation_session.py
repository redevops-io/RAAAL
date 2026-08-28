"""The anonymous evaluation session (§4, §12.C).

Exercises the session API directly, at the level the routes use it:

* a tampered session id or save token is rejected;
* an expired session is rejected;
* a replayed, already-consumed save token is rejected — no second plan;
* two independent browser sessions over the same evaluation do not collide and
  cannot read across each other;
* a session carries no user identity — an account switch inherits nothing;
* the session envelope holds only strategy meaning and provenance refs, never a
  broker credential or account secret.

The session wraps an *already-evaluated* review, so the fixture drives one real
anonymous evaluation through the app to persist a content-addressed review, then
the tests wrap it. Nothing here reconstructs a strategy — that is the point.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("jwt", reason="pyjwt is required to verify OIDC tokens")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"
ANSWER = {"describe": SENTENCE, "answer_assets": "VTI"}


@pytest.fixture(autouse=True)
def _clean_sessions():
    from src.workspace import evaluation_session

    evaluation_session.clear()
    yield
    evaluation_session.clear()


@pytest.fixture
def review_id(monkeypatch, tmp_path):
    """One anonymous evaluation, persisted, returned by its content address."""
    from src.deploy import context as deploy_context

    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used-by-recordings")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/pilot.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    client = TestClient(app, follow_redirects=False)
    posted = client.post("/evaluate", data=ANSWER)
    assert posted.status_code == 303, posted.text[:300]
    # Keep the deployment bound for the duration of the test — the session store
    # reads the review through the same resolved context.
    yield posted.headers["location"].rsplit("/", 1)[-1]


def _session(review_id, **kwargs):
    from src.workspace import evaluation_session as es

    return es.create_for_review(review_id, **kwargs)


class TestCreation:
    def test_a_session_wraps_the_evaluated_review(self, review_id):
        session = _session(review_id)
        assert session.compiled_plan_hash == review_id
        assert session.evaluated_plan_id.startswith("plan-")
        assert session.session_id.startswith("es-")
        assert session.save_token

    def test_a_missing_review_cannot_be_wrapped(self, review_id):
        from src.workspace import evaluation_session as es

        with pytest.raises(es.SessionError):
            es.create_for_review("review-doesnotexist")


class TestTamperIsRejected:
    def test_an_unknown_session_id_resolves_to_nothing(self, review_id):
        from src.workspace import evaluation_session as es

        _session(review_id)
        assert es.resolve("es-forged") is None

    def test_a_mismatched_save_token_does_not_verify(self, review_id):
        from src.workspace import evaluation_session as es

        session = _session(review_id)
        assert es.verify(session.session_id, session.save_token)
        assert not es.verify(session.session_id, session.save_token + "x")
        assert not es.verify("es-forged", session.save_token)

    def test_a_tampered_token_cannot_be_consumed(self, review_id):
        from src.workspace import evaluation_session as es

        session = _session(review_id)
        assert es.consume_save_token(session.save_token + "tamper") is None
        # The real token still works — the tamper attempt did not spend it.
        assert es.consume_save_token(session.save_token) is not None


class TestExpiryIsRejected:
    def test_an_expired_session_does_not_resolve(self, review_id):
        from src.workspace import evaluation_session as es

        session = _session(review_id, ttl_seconds=-1)
        assert session.is_expired()
        assert es.resolve(session.session_id) is None

    def test_an_expired_token_cannot_be_consumed(self, review_id):
        from src.workspace import evaluation_session as es

        session = _session(review_id, ttl_seconds=-1)
        assert es.consume_save_token(session.save_token) is None


class TestSingleUse:
    def test_a_consumed_token_is_rejected_on_replay(self, review_id):
        from src.workspace import evaluation_session as es

        session = _session(review_id)
        first = es.consume_save_token(session.save_token)
        assert first is not None and first.session_id == session.session_id
        assert es.is_consumed(session.session_id)
        # Every later attempt — a double-click, a refreshed resume URL — is a
        # no-op. Nothing new is minted because nothing new is handed back.
        assert es.consume_save_token(session.save_token) is None
        assert es.consume_save_token(session.save_token) is None


class TestSessionsDoNotCollide:
    def test_two_sessions_over_one_evaluation_are_independent(self, review_id):
        from src.workspace import evaluation_session as es

        first = _session(review_id)
        second = _session(review_id)
        assert first.session_id != second.session_id
        assert first.save_token != second.save_token
        # They share the evaluated artifact (same content address) but nothing
        # else: consuming one leaves the other spendable.
        assert first.compiled_plan_hash == second.compiled_plan_hash
        assert es.consume_save_token(first.save_token) is not None
        assert not es.is_consumed(second.session_id)
        assert es.consume_save_token(second.save_token) is not None

    def test_one_session_token_cannot_resolve_another_session(self, review_id):
        from src.workspace import evaluation_session as es

        first = _session(review_id)
        second = _session(review_id)
        # A token is bound to its own session; it never verifies against another.
        assert not es.verify(second.session_id, first.save_token)
        assert not es.verify(first.session_id, second.save_token)


class TestNoIdentityIsCarried:
    def test_the_session_names_no_user(self, review_id):
        session = _session(review_id)
        # No account field of any kind — the owner is decided at save time.
        for forbidden in ("owner", "subject", "email", "user", "tenant",
                          "account", "credential", "token_broker"):
            assert not hasattr(session, forbidden), (
                f"the session exposes {forbidden!r}; an anonymous session that "
                "named a user would be the durable private workspace §4 forbids")

    def test_an_account_switch_inherits_no_session_state(self, review_id):
        """The session is not keyed to whoever created it — it has no creator.
        Whichever authenticated owner consumes it saves *their own* plan; there
        is no prior owner to inherit from."""
        from src.workspace import evaluation_session as es

        session = _session(review_id)
        resolved = es.resolve(session.session_id)
        assert "review_owner" in resolved.as_dict()
        # review_owner is the *shared* anonymous workspace, never a real subject.
        from src.workspace.owner import SHARED

        assert resolved.review_owner == SHARED


class TestNoSecretsInTheEnvelope:
    def test_the_envelope_is_strategy_meaning_and_provenance_only(self, review_id):
        session = _session(review_id)
        envelope = session.as_dict()

        # The save token — the one value that authorises a save — is never in the
        # serialisable envelope.
        assert "save_token" not in envelope

        blob = repr(envelope).lower()
        for secret in ("password", "secret", "api_key", "apikey", "access_token",
                       "refresh_token", "broker", "account_number", "ssn",
                       "credential", "private_key"):
            assert secret not in blob, (
                f"the anonymous session envelope contains {secret!r}; it must "
                "carry only strategy meaning and provenance refs")

        # What it does carry: references to the evaluated artifact and its
        # provenance, plus the words typed.
        assert envelope["compiled_plan_hash"] == review_id
        assert set(envelope) >= {
            "original_prompt", "parsed_intent", "clarification_answers",
            "compiled_plan_hash", "evaluation_artifact_ids", "methodology_id",
            "protocol_version", "market_data_snapshot_id", "created_at",
            "expires_at"}
