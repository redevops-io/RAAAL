"""Gate 2 — the exact-save handoff (§2, §4, §12.B, §13).

The invariant these cover, stated once:

    After login, the save persists *the exact evaluated artifact* — not a
    re-parsed prompt or a newly recomputed strategy. No provider/model call is
    required merely to save an already-evaluated plan.

So the assertions are not "the save worked". They are:

* an anonymous visitor evaluates with no account and no tenant row;
* clicking Save while anonymous redirects to sign in, and persists nothing;
* after login the visitor returns to the same evaluation and the save completes;
* the saved plan's content hash equals the pre-login evaluated review's — the
  exactness invariant, checked against the hash the evaluation already fixed;
* no parser/model/evaluator runs on the save path — the reader is replaced with
  one that raises, and the save is required to succeed anyway;
* the saved plan belongs to the authenticated owner, and a second owner can
  neither read, nor export it.

The sixth is the one that cannot be taken on trust, so it is witnessed rather
than inferred: the reader factory and `read` are patched to raise, and the whole
anonymous-save-then-resume journey is required to run through them untouched.
"""
from __future__ import annotations

import os
import urllib.parse

import pytest

pytest.importorskip("jwt", reason="pyjwt is required to verify OIDC tokens")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"
ANSWER = {"describe": SENTENCE, "answer_assets": "VTI"}

#: The header the patched `signed_in` reads to name the viewer. Standing in for a
#: verified session cookie: the test bypasses the real OIDC round-trip (which
#: needs a live provider) and asserts the same thing the middleware would — that
#: after login the request carries a verified subject.
SUBJECT_HEADER = "x-test-subject"


@pytest.fixture(autouse=True)
def _clean_sessions():
    from src.workspace import evaluation_session

    evaluation_session.clear()
    yield
    evaluation_session.clear()


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A deployment that declares the runtime *and* an identity provider.

    The provider must be in force or Save has no boundary to cross. Sign-in is
    simulated by a header the patched `signed_in` reads, so no provider call is
    made and the recorded reader answers the evaluation from fixtures.
    """
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

    # Sign-in, simulated. The header names the subject; absent, the request is
    # anonymous. This is exactly the seam the middleware and `_begin_save` read.
    from src.deploy.identity import Identity
    import src.workspace.auth_routes as auth_routes

    def signed_in(request):
        subject = request.headers.get(SUBJECT_HEADER)
        return Identity(subject=subject, email=f"{subject}@x.test") \
            if subject else None

    monkeypatch.setattr(auth_routes, "signed_in", signed_in)

    from src.api import app

    return TestClient(app, follow_redirects=False)


def _as(subject):
    return {SUBJECT_HEADER: subject}


def _evaluate_anonymously(client):
    """Evaluate a strategy with no account. Returns the review id."""
    posted = client.post("/evaluate", data=ANSWER)
    assert posted.status_code == 303, posted.text[:300]
    location = posted.headers["location"]
    assert location.startswith("/pilot/reviews/"), location
    return location.rsplit("/", 1)[-1]


def _expected_plan_id(review_id):
    """The plan identity the evaluated review already determines.

    Computed the way the save must: reopen the stored review (a dict-only
    operation that constructs no reader) and take its content-addressed plan id.
    The anonymous review lives under the shared workspace, so it is read there.
    """
    from src.workspace.owner import SHARED
    from src.workspace.pilot import reopen
    from src.workspace.pilot_store import load_review_under, plan_id_for

    stored = load_review_under(SHARED, review_id)
    assert stored is not None, "the anonymous evaluation persisted no review"
    return plan_id_for(reopen(stored))


# --- §12.B ------------------------------------------------------------------

class TestAnonymousEvaluationNeedsNoAccount:
    def test_an_anonymous_visitor_evaluates_successfully(self, client):
        review_id = _evaluate_anonymously(client)
        assert review_id.startswith("review-")

    def test_no_user_or_tenant_row_is_required_for_evaluation(self, client):
        """Evaluation is public. It must not have written a plan to anyone."""
        _evaluate_anonymously(client)
        from src.workspace.pilot_store import every_plan

        assert every_plan() == [], (
            "evaluating a strategy anonymously wrote a plan; evaluation is "
            "public and must persist nothing that needs an owner")


class TestSaveIsTheAuthenticationBoundary:
    def test_saving_while_anonymous_redirects_to_sign_in(self, client):
        review_id = _evaluate_anonymously(client)
        saved = client.post("/evaluate/save",
                            data={"review_id": review_id, "picked": ""})
        assert saved.status_code == 303
        location = saved.headers["location"]
        assert location.startswith("/auth/login?next="), location

    def test_saving_while_anonymous_persists_nothing(self, client):
        review_id = _evaluate_anonymously(client)
        client.post("/evaluate/save", data={"review_id": review_id})
        from src.workspace.pilot_store import every_plan

        assert every_plan() == [], (
            "an anonymous Save wrote a plan before authentication; the click "
            "must lead to sign-in, not to persistence")

    def test_the_login_next_returns_to_the_resume_endpoint(self, client):
        review_id = _evaluate_anonymously(client)
        saved = client.post("/evaluate/save", data={"review_id": review_id})
        nxt = urllib.parse.parse_qs(
            urllib.parse.urlparse(saved.headers["location"]).query)["next"][0]
        assert nxt.startswith("/pilot/save/resume"), nxt
        query = urllib.parse.parse_qs(urllib.parse.urlparse(nxt).query)
        assert query.get("session") and query.get("save_token"), nxt


class TestAfterLoginTheExactArtifactIsSaved:
    def _begin_then_login(self, client, subject="user-a"):
        """The whole round-trip: evaluate anonymously, click Save, sign in, and
        follow the `next` the login would return to. Returns the resume
        response and the review id."""
        review_id = _evaluate_anonymously(client)
        started = client.post("/evaluate/save", data={"review_id": review_id})
        nxt = urllib.parse.parse_qs(
            urllib.parse.urlparse(started.headers["location"]).query)["next"][0]
        # Post-login, the browser GETs `next`, now carrying a verified session.
        resumed = client.get(nxt, headers=_as(subject))
        return resumed, review_id

    def test_the_save_completes_after_login(self, client):
        resumed, _ = self._begin_then_login(client)
        assert resumed.status_code == 303
        assert resumed.headers["location"].startswith("/pilot/plans/"), (
            resumed.headers.get("location"))

    def test_the_saved_hash_equals_the_pre_login_evaluated_hash(self, client):
        """The exactness invariant. The plan id the save lands on is the one the
        evaluation already fixed — proof the stored artifact was bound, not a
        fresh strategy recomputed from the words."""
        resumed, review_id = self._begin_then_login(client)
        saved_plan_id = resumed.headers["location"].rsplit("/", 1)[-1]
        assert saved_plan_id == _expected_plan_id(review_id), (
            "the saved plan id is not the hash the pre-login evaluation "
            "determined; the save recomputed the strategy instead of binding it")

    def test_the_saved_plan_reopens_to_the_evaluated_intent(self, client):
        """Independent of the id: the stored artifact's intent is the one the
        anonymous evaluation sealed, byte for byte."""
        resumed, review_id = self._begin_then_login(client, subject="user-a")
        plan_id = resumed.headers["location"].rsplit("/", 1)[-1]

        from src.workspace.owner import SHARED
        from src.workspace.pilot_store import load, load_review_under
        from src.workspace.routes import _LOOKING
        from src.deploy.identity import Identity

        review = load_review_under(SHARED, review_id)
        token = _LOOKING.set(Identity(subject="user-a"))
        try:
            plan = load(plan_id)
        finally:
            _LOOKING.reset(token)
        assert plan is not None
        assert plan["intent"]["intent_hash"] == review["intent"]["intent_hash"], (
            "the saved plan's pinned intent differs from the evaluated review's "
            "— the artifact was not carried across the login unchanged")

    def test_no_parser_or_model_runs_on_the_save_path(self, client, monkeypatch):
        """The property that cannot be taken on trust, witnessed. The reader
        factory and `read` are replaced with ones that raise; the anonymous
        Save and the post-login resume must both run without touching them."""
        review_id = _evaluate_anonymously(client)

        import src.workspace.pilot_routes as pilot_routes

        def no_reader(*args, **kwargs):
            raise AssertionError(
                "the save path constructed a reader / re-read the sentence; "
                "saving an already-evaluated plan must cost no model call")

        monkeypatch.setattr(pilot_routes, "configured_reader", no_reader)
        monkeypatch.setattr(pilot_routes, "configured_syntax_reader", no_reader)
        monkeypatch.setattr(pilot_routes, "read", no_reader)

        started = client.post("/evaluate/save", data={"review_id": review_id})
        assert started.status_code == 303
        nxt = urllib.parse.parse_qs(
            urllib.parse.urlparse(started.headers["location"]).query)["next"][0]
        resumed = client.get(nxt, headers=_as("user-a"))
        assert resumed.status_code == 303
        assert resumed.headers["location"].startswith("/pilot/plans/")


class TestOwnerBinding:
    def _save_as(self, client, subject):
        review_id = _evaluate_anonymously(client)
        started = client.post("/evaluate/save", data={"review_id": review_id})
        nxt = urllib.parse.parse_qs(
            urllib.parse.urlparse(started.headers["location"]).query)["next"][0]
        resumed = client.get(nxt, headers=_as(subject))
        return resumed.headers["location"].rsplit("/", 1)[-1]

    def test_the_saved_plan_belongs_to_the_authenticated_owner(self, client):
        plan_id = self._save_as(client, "user-a")
        got = client.get(f"/pilot/plans/{plan_id}", headers=_as("user-a"))
        assert got.status_code == 200

    def test_a_second_owner_cannot_read_it(self, client):
        plan_id = self._save_as(client, "user-a")
        got = client.get(f"/pilot/plans/{plan_id}", headers=_as("user-b"))
        assert got.status_code == 404, (
            "a second account read the first account's plan by its id — the "
            "owner is the save envelope and must scope every read")

    def test_a_second_owner_cannot_export_it(self, client):
        plan_id = self._save_as(client, "user-a")
        got = client.get(f"/pilot/plans/{plan_id}/runtime-artifact",
                         headers=_as("user-b"))
        assert got.status_code == 404

    def test_binding_did_not_change_the_content_hash(self, client):
        """Owner binding changes only the envelope. The plan id under owner A is
        the same content address the anonymous evaluation determined — the owner
        is nowhere in it."""
        review_id = _evaluate_anonymously(client)
        expected = _expected_plan_id(review_id)
        started = client.post("/evaluate/save", data={"review_id": review_id})
        nxt = urllib.parse.parse_qs(
            urllib.parse.urlparse(started.headers["location"]).query)["next"][0]
        plan_id = client.get(nxt, headers=_as("user-a")
                             ).headers["location"].rsplit("/", 1)[-1]
        assert plan_id == expected


class TestSignedInVisitorSavesDirectly:
    """A deployment with a provider, but the visitor is already signed in: Save
    binds the exact artifact immediately, with no login hop and no re-read."""

    def test_a_signed_in_save_binds_without_a_login_redirect(self, client):
        # Evaluate while signed in so the review is the viewer's own.
        posted = client.post("/evaluate", data=ANSWER, headers=_as("user-a"))
        review_id = posted.headers["location"].rsplit("/", 1)[-1]
        saved = client.post("/evaluate/save", data={"review_id": review_id},
                            headers=_as("user-a"))
        assert saved.status_code == 303
        assert saved.headers["location"].startswith("/pilot/plans/")
