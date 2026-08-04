"""How a plan was interpreted is a fact about the plan, not about the service.

    deployment  declares what parser it intends to use
    plan        records what parser actually interpreted it
    every later surface reads the stored identity

An entire pilot was measured model-assisted because `ANTHROPIC_API_KEY` was set
in a shell. Nobody decided it; the mode was inferred from whether a key
happened to exist, and the startup proof reported a valid configuration
throughout.

So mode is declared. Outside production an unset variable means deterministic —
a developer checkout must not become model-assisted because of a stray key. In
production an unset variable is a refusal, because defaulting to deterministic
there would serve a narrower product than the one reviewed, with fewer
recognitions and different blockers, while everything still looked correct.

**And the deployment's answer must not re-describe old plans.** A worksheet
reopened after the configuration moves has to show the interpretation the user
confirmed, not the one the service would produce today. Same rule as
market-data provenance, and the same failure if it is missed: a stored record
silently re-read against a source that has changed.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

DESCRIPTION = ("I contribute $7,000 a year to a Roth IRA in VOO on the first "
               "trading day of January, reinvesting dividends, and I never "
               "sell.")

BUILD = {"QUANTIFY_COMMIT": "c", "QUANTIFY_RELEASE_REF": "r",
         "QUANTIFY_IMAGE_DIGEST": "d", "QUANTIFY_MIGRATION_HEAD": "h",
         "QUANTIFY_SCOPE_SCHEMA_VERSION": "1",
         "QUANTIFY_CANONICALIZATION_VERSION": "1", "QUANTIFY_SNAPSHOT_ID": "s"}


class TestTheDeploymentMustSayWhatParserItRuns:
    def test_unset_is_deterministic_outside_production(self):
        from src.deploy.context import ParserMode, resolve

        target = resolve({}).model
        assert target.mode is ParserMode.DETERMINISTIC
        assert target.declared is False

    def test_a_stray_key_does_not_make_it_model_assisted(self):
        """The exact way this went unnoticed for a whole session."""
        from src.deploy.context import ParserMode, resolve

        target = resolve({"ANTHROPIC_API_KEY": "sk-present-in-a-shell"}).model
        assert target.mode is ParserMode.DETERMINISTIC

    def test_production_refuses_an_undeclared_parser(self):
        from src.deploy.preflight import Result, run

        outcome = run({"QUANTIFY_DEPLOYMENT_PROFILE": "production",
                       "QUANTIFY_DATABASE_URL": "postgresql://h/d", **BUILD})
        assert outcome.result is Result.REFUSED_CONFIGURATION
        assert "QUANTIFY_PARSER_MODE" in outcome.detail

    def test_production_refuses_model_assisted_without_a_key(self):
        from src.deploy.preflight import Result, run

        outcome = run({"QUANTIFY_DEPLOYMENT_PROFILE": "production",
                       "QUANTIFY_DATABASE_URL": "postgresql://h/d",
                       "QUANTIFY_PARSER_MODE": "MODEL_ASSISTED", **BUILD})
        assert outcome.result is Result.REFUSED_CONFIGURATION
        assert "no API key" in outcome.detail

    def test_production_refuses_an_unpinned_model(self):
        from src.deploy.preflight import Result, run

        outcome = run({"QUANTIFY_DEPLOYMENT_PROFILE": "production",
                       "QUANTIFY_DATABASE_URL": "postgresql://h/d",
                       "QUANTIFY_PARSER_MODE": "MODEL_ASSISTED",
                       "ANTHROPIC_API_KEY": "k", **BUILD})
        assert outcome.result is Result.REFUSED_CONFIGURATION
        assert "no model is pinned" in outcome.detail

    def test_a_local_deployment_is_not_burdened(self):
        """A hard refusal everywhere would make development, tests, CLI
        utilities and migration tooling all need parser configuration for
        behaviour none of them intends."""
        from src.deploy.preflight import Result, run

        assert run({"QUANTIFY_DATABASE_URL": "sqlite:///x.db"}).result \
            is Result.READY

    def test_the_startup_proof_states_the_parser(self):
        from src.deploy.preflight import run

        facts = run({"QUANTIFY_DATABASE_URL": "sqlite:///x.db"}).facts
        assert facts["parser"]["mode"] == "DETERMINISTIC"
        assert facts["parser"]["declared"] is False

    def test_the_proof_carries_no_key(self):
        from src.deploy.preflight import run

        outcome = run({"QUANTIFY_DATABASE_URL": "sqlite:///x.db",
                       "ANTHROPIC_API_KEY": "sk-secret-value"})
        assert "sk-secret-value" not in str(outcome.facts)


class TestTheClientFollowsTheDeclarationNotTheKey:
    @pytest.mark.real_parser_client
    def test_deterministic_uses_no_model_even_with_a_key(self, monkeypatch):
        from src.deploy.context import bind, resolve, unbind

        import src.workspace.routes as routes

        bind(resolve({"ANTHROPIC_API_KEY": "k",
                      "QUANTIFY_PARSER_MODEL": "claude-sonnet-5"}))
        try:
            assert routes._parser_client() is None
        finally:
            unbind()

    @pytest.mark.real_parser_client
    def test_model_assisted_without_a_key_refuses_rather_than_narrowing(self):
        """Silent fallback would hand two users different products under one
        deployment — one model-widened, one grammar-only, neither told.

        Marked `real_parser_client` to reach the real function: the suite
        stubs it to `None` for every unmarked test, so without this the
        assertion would be about the stub.

        Marked `model_stage1` first, which was wrong in a way worth recording —
        that tier is deselected by default because it calls a live API, so the
        three tests here stopped running entirely while reporting as passing
        by absence. A marker that means "needs the real function" is not the
        same as one that means "needs the network".
        """
        from fastapi import HTTPException

        from src.deploy.context import bind, resolve, unbind

        import src.workspace.routes as routes

        bind(resolve({"QUANTIFY_PARSER_MODE": "MODEL_ASSISTED",
                      "QUANTIFY_PARSER_FALLBACK": "REFUSE"}))
        try:
            with pytest.raises(HTTPException) as refusal:
                routes._parser_client()
            assert refusal.value.status_code == 503
        finally:
            unbind()

    @pytest.mark.real_parser_client
    def test_explicit_fallback_is_allowed_when_declared(self):
        from src.deploy.context import bind, resolve, unbind

        import src.workspace.routes as routes

        bind(resolve({"QUANTIFY_PARSER_MODE": "MODEL_ASSISTED",
                      "QUANTIFY_PARSER_FALLBACK": "EXPLICIT_DETERMINISTIC"}))
        try:
            assert routes._parser_client() is None
        finally:
            unbind()


@pytest.mark.skipif(not POSTGRES_URL,
                    reason="the pinning claim is about a stored plan surviving "
                           "a configuration change on the deployed engine")
class TestAStoredPlanKeepsItsOwnParserIdentity:
    """The regression: create under one configuration, change it, reopen."""

    @pytest.fixture
    def created(self, monkeypatch):
        from sqlalchemy import text

        from src.db import migrate
        from src.db.engine import Database

        for name, value in {
                "PILOT_DATA_POLICY": "SYNTHETIC_ONLY",
                "QUANTIFY_DEPLOYMENT_PROFILE": "local",
                "QUANTIFY_DATABASE_URL": POSTGRES_URL,
                "QUANTIFY_PARSER_MODE": "MODEL_ASSISTED",
                "QUANTIFY_PARSER_MODEL": "claude-sonnet-5"}.items():
            monkeypatch.setenv(name, value)

        database = Database(POSTGRES_URL)
        engine = database.sqlalchemy_engine()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()
        migrate.upgrade(database)

        import src.api as api

        from tests.conftest import submit_rendered_confirmation

        with TestClient(api.app) as client:
            response, plan_id = submit_rendered_confirmation(
                client, DESCRIPTION, title="Roth")
        assert response.status_code == 303, response.text
        return plan_id

    def stored(self, plan_id):
        from src.workspace.store import WorkspaceStore

        record = WorkspaceStore(POSTGRES_URL).get_plan(plan_id, "pilot")
        return (record.get("parse") or {}).get("parser") or {}

    def test_the_plan_records_the_parser_that_read_it(self, created):
        identity = self.stored(created)
        assert identity["mode"] == "MODEL_ASSISTED"
        assert identity["model"] == "claude-sonnet-5"
        assert identity["provider"] == "anthropic"
        assert identity["prompt_version"]

    def test_moving_the_deployment_does_not_rewrite_it(self, created,
                                                        monkeypatch):
        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "DETERMINISTIC")
        monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "a-different-model")
        from src.deploy.context import unbind

        unbind()

        identity = self.stored(created)
        assert identity["mode"] == "MODEL_ASSISTED", (
            "the stored plan was re-described by the current configuration; a "
            "reopened plan must show how it was actually interpreted")
        assert identity["model"] == "claude-sonnet-5"

    def test_reopening_calls_no_model(self, created, monkeypatch):
        """The pinned parse is the input to the deterministic stages. Calling
        a model again would silently recompile a plan the user confirmed."""
        import src.mission.parse_model as parse_model

        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "DETERMINISTIC")
        from src.deploy.context import unbind

        unbind()

        def refuse(*args, **kwargs):
            raise AssertionError("a model was called while reopening a plan")

        monkeypatch.setattr(parse_model.AnthropicClient, "complete", refuse)

        import src.api as api

        with TestClient(api.app) as client:
            page = client.get(f"/workspace/plans/{created}")
        assert page.status_code == 200

    def test_the_plan_page_shows_the_stored_identity(self, created,
                                                      monkeypatch):
        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "DETERMINISTIC")
        from src.deploy.context import unbind

        unbind()

        import src.api as api

        with TestClient(api.app) as client:
            body = client.get(f"/workspace/plans/{created}").text
        assert "model assistance" in body.lower()
        assert "claude-sonnet-5" in body, (
            "the page reported the current parser rather than the one that "
            "produced this plan")

    def test_the_export_carries_the_stored_identity(self, created,
                                                     monkeypatch):
        from src.db.transfer import export_bundle
        from src.workspace.store import WorkspaceStore

        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "DETERMINISTIC")
        from src.deploy.context import unbind

        unbind()

        bundle = export_bundle(WorkspaceStore(POSTGRES_URL),
                               exported_at="2026-08-04T00:00:00Z")
        plans = bundle["records"]["plan"]
        assert plans
        identities = [(row.get("parse") or {}).get("parser") for row in plans]
        assert any(one and one["mode"] == "MODEL_ASSISTED"
                   for one in identities), (
            "the exported plan lost the parser that produced it; an export is "
            "the copy nobody can add a caveat to afterwards")

    def test_the_export_states_the_data_boundary(self, created):
        from src.db.transfer import export_bundle
        from src.workspace.store import WorkspaceStore

        bundle = export_bundle(WorkspaceStore(POSTGRES_URL),
                               exported_at="2026-08-04T00:00:00Z")
        boundary = bundle["manifest"]["market_data"]
        assert boundary["synthetic"] is True
        assert "synthetic market data" in boundary["notice"].lower()
        assert "demo" not in boundary["notice"].lower()


class TestThisLaneActuallyRuns:
    """The marker defect, guarded at the level it happened.

    Three of these tests were marked `model_stage1`, which `pytest.ini`
    deselects by default because it calls a live provider. They needed the real
    `_parser_client` and no network — so the pinning guarantee vanished from
    the default suite while looking deliberately gated. Nothing failed; the
    tests simply stopped existing, which is the quietest form of the
    coverage-evidence defect: absence reported as intent.

    "Uses production code" is not "requires live infrastructure", and a marker
    that conflates them removes exactly the tests most worth keeping.
    """

    def test_the_marker_is_not_deselected_by_default(self):
        from pathlib import Path

        options = ""
        for line in Path("pytest.ini").read_text().splitlines():
            if line.strip().startswith("addopts"):
                options = line
        assert options, "pytest.ini declares no addopts to inspect"
        assert "not real_parser_client" not in options, (
            "`real_parser_client` has been added to the default deselection. "
            "That marker means the test needs the real function, not the "
            "network — deselecting it removes the parser-pinning guarantees "
            "from every ordinary run, which is exactly how they were lost the "
            "first time")

    def test_at_least_one_test_carries_it(self):
        """A marker nothing uses is a guarantee nothing checks."""
        import ast
        from pathlib import Path

        marked = 0
        for path in sorted(Path("tests").glob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:                                  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    if "real_parser_client" in ast.unparse(decorator):
                        marked += 1
        assert marked >= 3, (
            f"only {marked} test(s) exercise the real parser client; the "
            "declaration-versus-key distinction is unguarded")

    def test_the_marker_is_declared(self):
        from pathlib import Path

        body = Path("pytest.ini").read_text()
        assert "real_parser_client:" in body, (
            "an undeclared marker is a typo away from silently matching "
            "nothing")
