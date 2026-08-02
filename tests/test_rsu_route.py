"""The HTTP boundary: a request carrying vest language never reaches the
generic compiler.

    POST /new?describe=...
      -> parse
      -> template_hint
      -> handler_for(hint)
      -> RSU confirmation surface

The route dispatches; it does not build. Duplicating the builder at the route
would create a second reading of the same words, and the two would diverge on
exactly the descriptions that are hard to read.
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.api import app

RSU = ("100 ACME shares vest quarterly. Withhold 22% in shares. "
       "Sell as soon as I can after the blackout window. "
       "Keep company stock below 20%. "
       "Allocate proceeds 60% VTI, 30% VXUS, 10% BND.")

GENERIC = "I put $500 into SPY every month in my taxable brokerage"

RECURRING_SHARES = "I receive 100 ACME shares quarterly"


@pytest.fixture
def client():
    return TestClient(app)


def text_of(response) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", response.text))


def ask(client, describe):
    return client.get("/workspace/new", params={"describe": describe})


class TestTheRouteDispatchesOnTheHint:

    def test_vest_language_reaches_the_rsu_surface(self, client):
        body = text_of(ask(client, RSU))
        assert "Employer stock" in body
        assert "ACME" in body

    def test_generic_language_reaches_the_generic_surface(self, client):
        body = text_of(ask(client, GENERIC))
        assert "Employer stock" not in body

    def test_the_rsu_surface_is_a_separate_template(self, client):
        """Folded into the generic card with conditional fields, two sets of
        semantics would share one screen."""
        assert "Employer-stock cap" in text_of(ask(client, RSU))
        assert "Employer-stock cap" not in text_of(ask(client, GENERIC))


class TestTheGenericCompilerIsNeverReached:
    """The HTTP form of the no-fallback rule."""

    def test_an_rsu_request_succeeds_with_the_compiler_broken(
            self, client, monkeypatch):
        import src.workspace.routes as routes

        def explode(*args, **kwargs):
            raise AssertionError("a vest reached the generic compiler")

        monkeypatch.setattr(routes, "compile_scenario", explode)
        response = ask(client, RSU)
        assert response.status_code == 200
        assert "ACME" in text_of(response)

    def test_a_generic_request_does_use_the_compiler(self, client, monkeypatch):
        """Guards the guard: if nothing called it, the test above would pass
        against a route that compiled nothing at all."""
        import src.workspace.routes as routes

        called = []
        original = routes.compile_scenario
        monkeypatch.setattr(
            routes, "compile_scenario",
            lambda *a, **k: (called.append(1), original(*a, **k))[1])

        ask(client, GENERIC)
        assert called

    def test_removing_the_handler_fails_rather_than_falling_back(
            self, client, monkeypatch):
        import src.mission.rsu_declaration as declaration

        monkeypatch.setattr(declaration, "TEMPLATE_HANDLERS", {})

        response = ask(client, RSU)
        assert response.status_code == 501
        assert "TEMPLATE_HANDLER_MISSING" in response.text

    def test_it_does_not_silently_compile_when_the_handler_is_gone(
            self, client, monkeypatch):
        import src.mission.rsu_declaration as declaration
        import src.workspace.routes as routes

        monkeypatch.setattr(declaration, "TEMPLATE_HANDLERS", {})
        monkeypatch.setattr(
            routes, "compile_scenario",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("fell back to generic compilation")))

        assert ask(client, RSU).status_code == 501


class TestFailClosedCases:

    def test_an_unpinned_corporate_action_blocks_the_run(self, client):
        """Share counts cannot be trusted across a split without knowing."""
        body = text_of(ask(client, RSU))
        assert "more before this can run" in body
        assert "Corporate-action history" in body

    def test_an_ambiguous_employer_stays_unresolved(self, client):
        body = text_of(ask(
            client, "My ACME and BETA shares vest quarterly and I hold them"))
        assert "Employer stock" in body
        assert "not stated" in body

    def test_recurring_shares_without_vest_language_stay_generic(self, client):
        """A DRIP, a gift and a transfer are not vests, and routing them to the
        vesting runtime would invent the semantics the template prevents."""
        assert "Employer stock" not in text_of(ask(client, RECURRING_SHARES))

    def test_the_route_does_not_add_the_hint_itself(self, client,
                                                    monkeypatch):
        """Behavioural, not textual. An earlier version of this test compared a
        set against itself and could never fail.

        With the recogniser forced to emit no hint, a route that added one
        would still reach the RSU surface.
        """
        import src.mission.compiler as compiler
        import src.workspace.routes as routes

        original = routes.parse_with_model

        def hintless(text, **kwargs):
            from dataclasses import replace as _replace

            stage1 = original(text, **kwargs)
            return _replace(stage1,
                            parsed=_replace(stage1.parsed, template_hint=None))

        monkeypatch.setattr(routes, "parse_with_model", hintless)
        response = ask(client, RSU)

        # A 200 on the *generic* surface. Asserting only the absence of RSU
        # text cannot tell "did not route" from "routed and then failed": a
        # route that injected the hint would dispatch to a missing handler and
        # return 501, which also contains no RSU text.
        assert response.status_code == 200
        assert "Employer stock" not in text_of(response)
        assert "Confirmed under" not in text_of(response)


class TestTheRouteBuildsNothing:

    def test_it_calls_the_registered_handler(self):
        import inspect

        import src.workspace.routes as routes

        source = inspect.getsource(routes._template_confirmation)
        assert "handler_for(" in source
        assert "handler(" in source

    def test_it_does_not_reimplement_recognition(self):
        import inspect

        import src.workspace.routes as routes

        source = inspect.getsource(routes._template_confirmation)
        for duplicated in ("re.search", "RSUDeclaration(", "recognize("):
            assert duplicated not in source


class TestTheSurfaceCarriesNoComputedOutput:

    def test_no_execution_figure_appears(self, client, monkeypatch):
        """Structural, not textual.

        The scope legitimately says "delivered shares" — that is what the run
        will model. The invariant is that no *computed value* is produced, so
        it is asserted by breaking every engine entry point and requiring the
        page to render, and by checking the template context carries no run.
        """
        import src.workspace.routes as routes

        def explode(*args, **kwargs):
            raise AssertionError("the confirmation surface computed a figure")

        monkeypatch.setattr(routes, "_run", explode)
        monkeypatch.setattr(routes, "_prices", explode)
        monkeypatch.setattr(routes, "compile_scenario", explode)

        response = ask(client, RSU)
        assert response.status_code == 200
        assert "ACME" in text_of(response)

    def test_the_surface_is_given_no_run_to_display(self):
        """The context the template receives holds declarations only."""
        import ast
        import inspect
        import textwrap

        import src.workspace.routes as routes

        tree = ast.parse(textwrap.dedent(
            inspect.getsource(routes._template_confirmation)))
        keys = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        for forbidden in ("run", "result", "benchmarks", "comparability",
                          "chain"):
            assert forbidden not in keys

    def test_the_scope_is_shown_before_any_figure(self, client):
        body = text_of(ask(client, RSU))
        assert "This run will model" in body
        assert "This run will not model" in body

    def test_the_version_pin_is_shown(self, client):
        assert "Confirmed under" in text_of(ask(client, RSU))
