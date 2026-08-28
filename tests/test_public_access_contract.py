"""The public access contract (§12.A of the public strategy-lab plan, Gate 1).

A parameterized route-inventory test driven against the live application:

* every PUBLIC_RESEARCH and PUBLIC_EVALUATION route is reachable anonymously —
  it is never sent to the login wall;
* every AUTHENTICATED_PERSISTENCE and PRIVATE_FINANCIAL_STATE route rejects an
  anonymous request where an issuer is configured — a 303 to `/auth/login`, or a
  401/403;
* the canonical `/evaluate` reaches the *same* evaluation as the legacy
  `/pilot`/`/pilot/answer` path — same rendered draft, and a content-addressed
  review id that is byte-identical to the legacy one, which is only possible if
  the same compiler and evaluator ran. That is the proof there is no second
  strategy parser in the website layer;
* a route added to the application but absent from the manifest fails the suite,
  so a new endpoint cannot become public or private by omission.

The classification is read from `boundary_manifest`, the same single source the
auth middleware derives its decision from — so this test and the running service
cannot disagree about which side a route is on.
"""
from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("jwt", reason="pyjwt is required to verify OIDC tokens")
pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.workspace.boundary_manifest import (BoundaryClass, boundary_for,
                                             undeclared)

SENTENCE = "invest $500 monthly"


# --- the live route inventory, split by class ------------------------------

def _all_paths(app) -> list:
    """Every reachable path, including those behind an included router."""
    found = set()

    def walk(routes):
        for route in routes:
            path = getattr(route, "path", "")
            if path:
                found.add(path)
            nested = getattr(route, "routes", None) or getattr(
                route, "router", None)
            if nested is not None:
                walk(getattr(nested, "routes", nested))

    walk(app.routes)
    found.update(app.openapi().get("paths", {}))
    return sorted(found)


def _live_paths() -> list:
    import src.api as api

    return _all_paths(api.app)


def _paths_in(*classes) -> list:
    wanted = set(classes)
    return [p for p in _live_paths() if boundary_for(p).boundary_class in wanted]


PUBLIC_PATHS = _paths_in(BoundaryClass.PUBLIC_RESEARCH,
                         BoundaryClass.PUBLIC_EVALUATION)
PRIVATE_PATHS = _paths_in(BoundaryClass.AUTHENTICATED_PERSISTENCE,
                          BoundaryClass.PRIVATE_FINANCIAL_STATE)


def _concrete(template: str) -> str:
    """A probeable URL: template parameters filled with a throwaway segment."""
    return re.sub(r"\{[^}]+\}", "contract-probe", template)


def _is_login_redirect(response) -> bool:
    return (response.status_code == 303
            and response.headers.get("location", "").startswith("/auth/login"))


# --- the anonymous client, on a deployment that HAS accounts ---------------

@pytest.fixture
def anon_client(monkeypatch, tmp_path):
    """A signed-out client against a deployment that declares both the runtime
    and an identity provider.

    An issuer must be configured, or the private half of the contract is vacuous:
    with no provider the middleware stands aside and nothing is gated. The
    runtime is declared so the evaluator (`/evaluate`, `/pilot/answer`) actually
    serves rather than refusing, and it reads from recordings so no provider call
    is made.
    """
    from src.deploy import context as deploy_context

    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used-by-recordings")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/pilot.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    # An identity provider the routes believe in, without touching the process's
    # real configuration — the same trick `test_auth_routes` uses.
    from src.deploy.context import IdentityTarget

    target = IdentityTarget(issuer="https://auth.example.test",
                            audience="client-1", client_id="client-1")
    monkeypatch.setattr("src.workspace.auth_routes._target", lambda: target)

    from src.api import app

    return TestClient(app, follow_redirects=False)


class TestTheIssuerIsActuallyConfigured:
    """Without this the private half below could pass by gating nothing."""

    def test_a_known_private_route_redirects(self, anon_client):
        response = anon_client.get("/workspace/plans/demo")
        assert _is_login_redirect(response), (
            "the fixture's issuer is not in force; the private contract would "
            "be vacuous")


class TestPublicRoutesAreReachableAnonymously:
    @pytest.mark.parametrize("path", PUBLIC_PATHS)
    def test_no_public_route_is_behind_the_login_wall(self, anon_client, path):
        response = anon_client.get(_concrete(path))
        assert not _is_login_redirect(response), (
            f"{path} is PUBLIC in the manifest but redirected an anonymous "
            "visitor to sign in")

    def test_the_named_public_surfaces_succeed(self, anon_client):
        """The specific routes §12.A names, each with its real method."""
        assert anon_client.get("/research").status_code in (200, 503)
        assert not _is_login_redirect(anon_client.get("/research"))

        evaluate_get = anon_client.get("/evaluate", params={"describe": SENTENCE})
        assert evaluate_get.status_code == 200

        evaluate_post = anon_client.post(
            "/evaluate", data={"describe": SENTENCE, "answer_assets": "VTI"})
        assert evaluate_post.status_code == 303
        assert not _is_login_redirect(evaluate_post)
        assert evaluate_post.headers["location"].startswith("/pilot/reviews/")

        answer = anon_client.post(
            "/pilot/answer", data={"describe": SENTENCE, "answer_assets": "VTI"})
        assert answer.status_code == 303
        assert not _is_login_redirect(answer)


class TestPrivateRoutesRejectTheAnonymous:
    @pytest.mark.parametrize("path", PRIVATE_PATHS)
    def test_every_private_route_sends_the_anonymous_to_sign_in(
            self, anon_client, path):
        response = anon_client.get(_concrete(path))
        assert _is_login_redirect(response) or response.status_code in (401, 403), (
            f"{path} is private in the manifest but served an anonymous request "
            f"(status {response.status_code})")

    def test_the_named_private_surfaces_reject(self, anon_client):
        """The specific routes §5/§12.A name: save, saved-plan read, export."""
        # POST-only, but the gate runs before routing, so a GET is redirected
        # before it can become a 405.
        assert _is_login_redirect(anon_client.get("/pilot/save"))
        assert _is_login_redirect(anon_client.get("/pilot/plans/demo"))
        assert _is_login_redirect(
            anon_client.get("/pilot/plans/demo/runtime-artifact"))


class TestEvaluateIsTheSameEvaluatorNotASecondParser:
    """`/evaluate` must reach the existing compiler/evaluator, proven by result
    identity rather than by reading the import graph."""

    def test_the_draft_page_is_identical_to_the_legacy_public_entry(
            self, anon_client):
        """Compared against `/workspace/new`, the *public* legacy evaluator
        entry — not `/pilot`, which stays login-gated (see the module note): a
        signed-out visitor is redirected from `/pilot` but served by `/evaluate`,
        which is the point of adding the public canonical name. Both `/evaluate`
        and `/workspace/new` reach the same runtime draft, so the rendered page
        is identical."""
        through_evaluate = anon_client.get("/evaluate",
                                           params={"describe": SENTENCE})
        through_legacy = anon_client.get("/workspace/new",
                                         params={"describe": SENTENCE})
        assert through_evaluate.status_code == 200
        assert through_legacy.status_code == 200
        assert through_evaluate.text == through_legacy.text, (
            "the canonical evaluator rendered a different page from the legacy "
            "public entry; they are meant to reach the same evaluator")

    def test_the_evaluated_review_id_is_byte_identical(self, anon_client):
        """The review id is content-addressed — it is a hash of the compiled,
        settled evaluation. Two submissions that produce the same id ran the
        same compiler and evaluator over the same intent; a second parser in the
        `/evaluate` layer could not land on the legacy path's hash by
        coincidence."""
        data = {"describe": SENTENCE, "answer_assets": "VTI"}
        through_evaluate = anon_client.post("/evaluate", data=data)
        through_answer = anon_client.post("/pilot/answer", data=data)

        assert through_evaluate.status_code == 303
        assert through_answer.status_code == 303
        assert (through_evaluate.headers["location"]
                == through_answer.headers["location"]), (
            "the canonical evaluator produced a different evaluation identity "
            "from the legacy path — evidence of a second parser")


class TestAnUndeclaredRouteFailsTheSuite:
    """Route additions fail CI unless classified — the guarantee, restated where
    the public contract lives so it travels with it."""

    def test_a_new_live_route_absent_from_the_manifest_is_flagged(self):
        app = FastAPI()

        @app.get("/a-brand-new-unclassified-surface")
        def _new():  # pragma: no cover - never called
            return {}

        missing = undeclared(_all_paths(app))
        assert "/a-brand-new-unclassified-surface" in missing, (
            "a route with no boundary declaration was not caught; a new "
            "endpoint could ship without stating its side")

    def test_the_real_application_has_no_undeclared_routes(self):
        assert undeclared(_live_paths()) == []
