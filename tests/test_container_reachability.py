"""The built image, started as a process, serving the pilot.

Every other test in this repository runs the application in-process. That
proved a great deal and could not have caught what Gate 3 found: the container
ran a Bokeh dashboard, nothing served `src/api.py`, and the entire Gate 2
preflight sat on a path no deployment took. An in-process test cannot see which
`CMD` an image has.

So this builds the real image and runs it. It is slow and it is gated on Docker
being present — and it is the only test here that answers "does the thing we
ship do this".

    container process -> API entrypoint -> preflight -> PostgreSQL
                      -> market-data gate -> pilot route

Each refusal is asserted from the container's own exit, and each is required to
name its own distinct outcome rather than a generic failure.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import pytest

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")
DOCKER = shutil.which("docker")

pytestmark = [
    pytest.mark.skipif(not DOCKER, reason="needs docker to build the image"),
    pytest.mark.skipif(
        not POSTGRES_URL,
        reason="needs a reachable PostgreSQL for the ready case"),
    pytest.mark.container,
]

IMAGE = "quantify-api:pytest"
NETWORK = "quantify-pytest-net"
CONTAINER_DB = "postgresql://quantify:quantify_dev@quantify-pg:5432/quantify"

#: A production deployment must declare which parser it runs, so every
#: container started here declares one. The refusal is exercised deliberately
#: in `test_container_parser_pinning.py`; letting it fire implicitly meant this
#: whole lane errored with `DeploymentRefused` and stopped reaching the
#: reachability questions it exists to ask.
BUILD_STAMPS = {
    "QUANTIFY_PARSER_MODE": "DETERMINISTIC",
    "QUANTIFY_COMMIT": "abc123",
    "QUANTIFY_RELEASE_REF": "v1.0.0",
    "QUANTIFY_IMAGE_DIGEST": "sha256:dead",
    "QUANTIFY_SNAPSHOT_ID": "syn-2026-01",
}


def docker(*args, **kwargs):
    return subprocess.run([DOCKER, *args], capture_output=True, text=True,
                          timeout=kwargs.pop("timeout", 300), **kwargs)


def env_args(**overrides):
    values = {**BUILD_STAMPS, **overrides}
    args = []
    for name, value in values.items():
        if value is not None:
            args += ["-e", f"{name}={value}"]
    return args


@pytest.fixture(scope="module")
def image():
    built = docker("build", "-q", "-t", IMAGE, "-f", "Dockerfile", ".",
                   timeout=1800)
    assert built.returncode == 0, built.stderr[-2000:]
    docker("network", "create", NETWORK)
    docker("network", "connect", NETWORK, "quantify-pg")
    return IMAGE


@pytest.fixture(scope="module")
def migrated(image):
    """The database the ready case needs, migrated from the host."""
    from src.db import migrate
    from src.db.engine import Database

    database = Database(POSTGRES_URL)
    migrate.upgrade(database)
    return database


def run_and_capture(image, **env):
    """Start the container in the foreground and return what it printed."""
    return docker("run", "--rm", "--network", NETWORK, *env_args(**env), image,
                  timeout=180)


class TestTheImageRunsThePilotApi:
    def test_its_command_is_the_factory(self, image):
        inspected = docker("inspect", "-f", "{{json .Config.Cmd}}", image)
        command = json.loads(inspected.stdout)
        assert "src.api:create_app" in command
        assert "--factory" in command

    def test_it_does_not_run_the_dashboard(self, image):
        inspected = docker("inspect", "-f", "{{json .Config.Cmd}}", image)
        assert "service.py" not in inspected.stdout


class TestTheContainerRefuses:
    """Each refusal from the container's own exit, with its own outcome."""

    def test_without_a_database_url(self, image):
        result = run_and_capture(image)
        assert result.returncode != 0
        assert "REFUSED_CONFIGURATION" in result.stderr + result.stdout

    def test_with_sqlite_under_production(self, image):
        result = run_and_capture(image,
                                 QUANTIFY_DATABASE_URL="sqlite:///data/w.db")
        assert result.returncode != 0
        assert "REFUSED_CONFIGURATION" in result.stderr + result.stdout

    def test_without_a_build_stamp(self, image):
        result = run_and_capture(image, QUANTIFY_COMMIT=None,
                                 QUANTIFY_DATABASE_URL=CONTAINER_DB)
        assert result.returncode != 0
        assert "BUILD_UNOBSERVABLE" in result.stderr + result.stdout

    def test_against_an_unmigrated_database(self, image):
        from sqlalchemy import text

        from src.db.engine import Database

        engine = Database(POSTGRES_URL).sqlalchemy_engine()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()

        result = run_and_capture(image, QUANTIFY_DATABASE_URL=CONTAINER_DB)
        assert result.returncode != 0
        assert "MIGRATION_MISMATCH" in result.stderr + result.stdout

    def test_no_refusal_leaks_the_connection_detail(self, image):
        result = run_and_capture(
            image,
            QUANTIFY_DATABASE_URL="postgresql://user:secret@10.9.9.9:5999/db")
        output = result.stderr + result.stdout
        for leak in ("secret", "10.9.9.9", "5999"):
            assert leak not in output, f"{leak!r} leaked from the container"


class TestTheContainerServes:
    @pytest.fixture
    def running(self, image, migrated):
        docker("rm", "-f", "quantify-api-pytest")
        started = docker(
            "run", "-d", "--name", "quantify-api-pytest",
            "--network", NETWORK,
            *env_args(QUANTIFY_DATABASE_URL=CONTAINER_DB,
                      PILOT_DATA_POLICY="SYNTHETIC_ONLY"),
            "-p", "8123:8000", image)
        assert started.returncode == 0, started.stderr
        for _ in range(40):
            probe = docker("exec", "quantify-api-pytest", "python", "-c",
                           "import urllib.request;"
                           "print(urllib.request.urlopen("
                           "'http://localhost:8000/health/ready').status)")
            if probe.returncode == 0 and "200" in probe.stdout:
                break
            time.sleep(1)
        else:                                                # pragma: no cover
            logs = docker("logs", "quantify-api-pytest")
            pytest.fail(f"container never became ready:\n{logs.stderr[-3000:]}")
        yield "quantify-api-pytest"
        docker("rm", "-f", "quantify-api-pytest")

    def probe(self, container, path):
        return docker("exec", container, "python", "-c",
                      "import urllib.request;"
                      f"r=urllib.request.urlopen('http://localhost:8000{path}');"
                      "print(r.status, r.read().decode())")

    def test_it_is_live(self, running):
        result = self.probe(running, "/health/live")
        assert "200" in result.stdout and '"live": true' in \
            result.stdout.replace('"live":true', '"live": true')

    def test_it_is_ready(self, running):
        result = self.probe(running, "/health/ready")
        assert "200" in result.stdout

    def test_a_pilot_route_serves(self, running):
        result = self.probe(running, "/health")
        assert "200" in result.stdout

    def test_the_dashboard_is_not_loaded_in_the_serving_process(self, running):
        """Reachability from the running process, not from a file listing."""
        result = docker(
            "exec", running, "python", "-c",
            "import os;"
            "pid=[e for e in os.listdir('/proc') if e.isdigit() and "
            "'uvicorn' in open('/proc/'+e+'/cmdline',errors='ignore').read()][0];"
            "m=open('/proc/'+pid+'/maps').read().lower();"
            "print('bokeh' in m, 'visualization' in m)")
        assert "False False" in result.stdout, result.stdout

    def test_the_startup_proof_reaches_the_log(self, running):
        """It did not, at first: uvicorn installs its own logging config and
        does not propagate this application's logger, so a proof written with
        `LOG.info` was emitted nowhere an operator would read."""
        logs = docker("logs", running)
        assert "deployment proof" in logs.stderr + logs.stdout

    def test_the_proof_names_what_was_established(self, running):
        logs = (docker("logs", running).stderr +
                docker("logs", running).stdout)
        line = [one for one in logs.splitlines() if "deployment proof" in one][0]
        proof = json.loads(line.split("deployment proof", 1)[1].strip())
        assert proof["result"] == "READY"
        assert proof["deployment_profile"] == "production"
        assert proof["database"]["engine"] == "postgresql"
        assert proof["database"]["schema_parity"] == "PASS"
        assert proof["build"]["observable"] is True

    def test_the_proof_carries_no_credentials(self, running):
        logs = docker("logs", running).stderr + docker("logs", running).stdout
        line = [one for one in logs.splitlines() if "deployment proof" in one][0]
        for leak in ("quantify_dev", "quantify-pg", "5432", "postgresql://"):
            assert leak not in line, f"{leak!r} appeared in the startup proof"
