"""Parser configuration and historical pinning, proven by the deployed image.

    container A   MODEL_ASSISTED, pinned model   -> create and save a plan
    stop A
    container B   DETERMINISTIC, different model -> reopen and export

Two lifecycles against one database, because a real restart re-resolves
everything from scratch. In-process, `unbind()` simulates a configuration
change inside one interpreter — which cannot see a stale module-level default,
a value captured at import, or a setting the process only reads once.

**Egress is proven by removing the route, not by patching.** Container B runs on
an `--internal` network: it reaches PostgreSQL and nothing else. If reopening or
exporting needed the provider, they would fail or silently degrade. Succeeding
identically with no possible egress is what makes "no parser call" a fact about
the deployment rather than about a monkeypatch that could be wrong.

That check is only meaningful with its premise witnessed: container A must have
made a *real* provider request, or "no egress in B" is true of a system that
never had any. `model_available` on A's stored parse is the witness — it is
`true` only when a model answered.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")
DOCKER = shutil.which("docker")
API_KEY = os.environ.get("ANTHROPIC_API_KEY")

pytestmark = [
    pytest.mark.skipif(not DOCKER, reason="needs docker to build the image"),
    pytest.mark.skipif(not POSTGRES_URL, reason="needs a reachable PostgreSQL"),
    pytest.mark.skipif(not API_KEY,
                       reason="container A must make a real provider request; "
                              "without one the egress check in B is vacuous"),
    pytest.mark.container,
    # Correctly classified: this lane *does* reach a live provider, which is
    # what `model_stage1` means. The parser-pinning tests that need only the
    # real function carry `real_parser_client` and run by default — see
    # `tests/test_parser_pinning.py`, where conflating the two removed the
    # guarantee from every ordinary run.
    pytest.mark.model_stage1,
]

IMAGE = "quantify-api:pytest"
ROUTED = "quantify-pinning-net"
SEALED = "quantify-pinning-sealed"
CONTAINER_DB = "postgresql://quantify:quantify_dev@quantify-pg:5432/quantify"

DESCRIPTION = ("I contribute $7,000 a year to a Roth IRA in VOO on the first "
               "trading day of January, reinvesting dividends, and I never "
               "sell.")

BUILD = {"QUANTIFY_COMMIT": "abc123", "QUANTIFY_RELEASE_REF": "v1.0.0",
         "QUANTIFY_IMAGE_DIGEST": "sha256:dead",
         "QUANTIFY_SNAPSHOT_ID": "syn-2026-01"}


def docker(*args, timeout=300):
    return subprocess.run([DOCKER, *args], capture_output=True, text=True,
                          timeout=timeout)


def env_args(**values):
    args = []
    for name, value in {**BUILD, **values}.items():
        if value is not None:
            args += ["-e", f"{name}={value}"]
    return args


@pytest.fixture(scope="module")
def networks():
    """One routed, one sealed. The sealed network is the whole egress check."""
    docker("network", "create", ROUTED)
    docker("network", "create", "--internal", SEALED)
    for network in (ROUTED, SEALED):
        docker("network", "connect", network, "quantify-pg")
    yield ROUTED, SEALED
    for network in (ROUTED, SEALED):
        docker("network", "disconnect", network, "quantify-pg")
        docker("network", "rm", network)


@pytest.fixture(scope="module")
def image(networks):
    built = docker("build", "-q", "-t", IMAGE, "-f", "Dockerfile", ".",
                   timeout=1800)
    assert built.returncode == 0, built.stderr[-2000:]
    return IMAGE


@pytest.fixture(scope="module")
def migrated():
    from sqlalchemy import text

    from src.db import migrate
    from src.db.engine import Database

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    migrate.upgrade(database)
    return database


def script_in_container(image, network, script, **env):
    """Run a Python snippet inside the image, on the given network."""
    return docker("run", "--rm", "--network", network,
                  *env_args(QUANTIFY_DATABASE_URL=CONTAINER_DB, **env),
                  "--entrypoint", "python", image, "-c", script, timeout=300)


CREATE = """
import json, sys
from fastapi.testclient import TestClient
import src.api as api
sys.path.insert(0, "/app/tests")
from conftest import submit_rendered_confirmation
with TestClient(api.app) as client:
    response, plan_id = submit_rendered_confirmation(
        client, %r, title="Roth")
    assert response.status_code == 303, response.text
print("PLAN_ID=" + plan_id)
""" % DESCRIPTION

REOPEN = """
import json
from fastapi.testclient import TestClient
import src.api as api
from src.workspace.store import WorkspaceStore
from src.db.transfer import export_bundle
import os
plan_id = os.environ["PLAN_UNDER_TEST"]
store = WorkspaceStore(os.environ["QUANTIFY_DATABASE_URL"])
record = store.get_plan(plan_id, "pilot")
identity = (record.get("parse") or {}).get("parser") or {}
with TestClient(api.app) as client:
    page = client.get("/workspace/plans/" + plan_id)
bundle = export_bundle(store, exported_at="2026-08-04T00:00:00Z")
exported = [(r.get("parse") or {}).get("parser") for r in bundle["records"]["plan"]]
print("RESULT=" + json.dumps({
    "identity": identity,
    "page_status": page.status_code,
    "page_shows_model": "claude-sonnet-5" in page.text,
    "page_shows_synthetic": "synthetic market data" in page.text.lower(),
    "exported": exported,
    "boundary": bundle["manifest"]["market_data"],
}))
"""


def value_from(output, prefix):
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):]
    return ""


@pytest.fixture(scope="module")
def journey(image, networks, migrated):
    """Container A creates; container B, on a sealed network, reads."""
    routed, sealed = networks

    created = script_in_container(
        image, routed, CREATE,
        PILOT_DATA_POLICY="SYNTHETIC_ONLY",
        QUANTIFY_DEPLOYMENT_PROFILE="local",
        QUANTIFY_PARSER_MODE="MODEL_ASSISTED",
        QUANTIFY_PARSER_MODEL="claude-sonnet-5",
        ANTHROPIC_API_KEY=API_KEY)
    assert created.returncode == 0, created.stderr[-3000:]
    plan_id = value_from(created.stdout, "PLAN_ID=")
    assert plan_id.startswith("plan-"), created.stdout[-2000:]

    # Container A is gone. B resolves everything afresh, under a different
    # declared parser, with no route off the network.
    reopened = script_in_container(
        image, sealed, REOPEN,
        PILOT_DATA_POLICY="SYNTHETIC_ONLY",
        QUANTIFY_DEPLOYMENT_PROFILE="local",
        QUANTIFY_PARSER_MODE="DETERMINISTIC",
        QUANTIFY_PARSER_MODEL="a-different-model",
        PLAN_UNDER_TEST=plan_id)
    assert reopened.returncode == 0, reopened.stderr[-3000:]
    payload = json.loads(value_from(reopened.stdout, "RESULT="))
    return plan_id, payload


class TestStartupConfigurationIsExplicit:
    def test_production_refuses_an_undeclared_parser(self, image, networks):
        routed, _sealed = networks
        result = docker("run", "--rm", "--network", routed,
                        *env_args(QUANTIFY_DATABASE_URL=CONTAINER_DB,
                                  QUANTIFY_DEPLOYMENT_PROFILE="production",
                                  PILOT_DATA_POLICY="SYNTHETIC_ONLY"),
                        image, timeout=180)
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "REFUSED_CONFIGURATION" in combined

    def test_model_assisted_without_a_key_refuses(self, image, networks):
        routed, _sealed = networks
        result = docker("run", "--rm", "--network", routed,
                        *env_args(QUANTIFY_DATABASE_URL=CONTAINER_DB,
                                  QUANTIFY_DEPLOYMENT_PROFILE="production",
                                  QUANTIFY_PARSER_MODE="MODEL_ASSISTED",
                                  PILOT_DATA_POLICY="SYNTHETIC_ONLY"),
                        image, timeout=180)
        assert result.returncode != 0
        assert "REFUSED_CONFIGURATION" in (result.stdout + result.stderr)


class TestTheProviderWasReallyReached:
    """The premise. Without it, "no egress in B" is true of a system that
    never made a request at all."""

    def test_container_a_used_a_model(self, journey):
        _plan_id, payload = journey
        assert payload["identity"]["mode"] == "MODEL_ASSISTED"
        assert payload["identity"]["model"] == "claude-sonnet-5"


class TestRestartDoesNotRewriteHistory:
    def test_the_stored_identity_survives(self, journey):
        _plan_id, payload = journey
        assert payload["identity"]["mode"] == "MODEL_ASSISTED", (
            "a restart under DETERMINISTIC re-described the stored plan")
        assert payload["identity"]["model"] == "claude-sonnet-5"

    def test_the_plan_page_shows_the_original(self, journey):
        _plan_id, payload = journey
        assert payload["page_status"] == 200
        assert payload["page_shows_model"], (
            "the page reported the parser the container is configured with "
            "rather than the one that produced the plan")

    def test_the_export_carries_the_original(self, journey):
        _plan_id, payload = journey
        modes = [one["mode"] for one in payload["exported"] if one]
        assert "MODEL_ASSISTED" in modes


class TestNoProviderCallOnReopenOrExport:
    """Container B has no route off its network. Both operations succeeding
    means neither needed the provider."""

    def test_reopen_succeeded_with_no_route_out(self, journey):
        _plan_id, payload = journey
        assert payload["page_status"] == 200

    def test_export_succeeded_with_no_route_out(self, journey):
        _plan_id, payload = journey
        assert payload["exported"]

    def test_the_sealed_network_really_is_sealed(self):
        """The witness for the witness. An `--internal` network that turned out
        to be routable would make both assertions above meaningless."""
        probe = docker(
            "run", "--rm", "--network", SEALED, "--entrypoint", "python", IMAGE,
            "-c", "import socket; socket.setdefaulttimeout(5);"
                  " socket.create_connection(('api.anthropic.com', 443))",
            timeout=120)
        assert probe.returncode != 0, (
            "the sealed network reached the provider, so the absence of a "
            "parser call proves nothing")


class TestTheDataBoundarySurvivesDeployment:
    def test_the_served_page_states_it(self, journey):
        _plan_id, payload = journey
        assert payload["page_shows_synthetic"]

    def test_the_export_manifest_states_it(self, journey):
        _plan_id, payload = journey
        boundary = payload["boundary"]
        assert boundary["synthetic"] is True
        assert "synthetic market data" in boundary["notice"].lower()
