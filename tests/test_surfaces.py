"""Which surface a deployment serves, checked against the deployment.

`bokeh_app.py` was declared "the standalone Bokeh demo, served separately and
not part of the pilot surface". That was written from the module's name, and
`scripts/service.py` — which imports it — was the Dockerfile's `CMD`. It was
the production container. Meanwhile nothing anywhere served `src/api.py`, so
the pilot application and its entire startup preflight sat on a path no
deployment took.

Both are now fixed, and these tests read the Dockerfiles rather than the
declarations' own descriptions. A classification that describes itself proves
nothing; the wrong one survived precisely because nothing compared it to an
artifact.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.deploy.surfaces import (
    DASHBOARD_DOCKERFILE,
    DASHBOARD_ENTRYPOINT,
    DataPolicy,
    PRODUCTION_DOCKERFILE,
    PRODUCTION_ENTRYPOINT,
    SURFACES,
    by_module,
    production_surfaces,
    unserved_surfaces,
)


def container_command(dockerfile: str):
    """The `CMD` an image runs.

    A `HEALTHCHECK` carries its own `CMD` continuation, which is not the
    entrypoint; only the exec form at the start of a line is. The command may
    be split across continuation lines, so they are joined first.
    """
    body = Path(dockerfile).read_text().replace("\\\n", " ")
    for line in body.splitlines():
        if line.startswith("CMD") and "[" in line:
            return re.findall(r'"([^"]+)"', line)
    return []


def imports_of(path: str):
    tree = ast.parse(Path(path).read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


class TestTheProductionImageServesThePilot:
    def test_its_command_is_the_application_factory(self):
        command = container_command(PRODUCTION_DOCKERFILE)
        assert PRODUCTION_ENTRYPOINT in command, (
            f"the production image runs {command}, not the pilot API")

    def test_it_uses_the_factory_form(self):
        """`--factory` is what makes the preflight run during startup rather
        than in a lifespan hook after the socket is already open."""
        assert "--factory" in container_command(PRODUCTION_DOCKERFILE)

    def test_it_does_not_start_the_dashboard(self):
        command = " ".join(container_command(PRODUCTION_DOCKERFILE))
        assert "service.py" not in command
        assert "bokeh" not in command.lower()

    def test_the_dashboard_is_not_in_its_startup_graph(self):
        """Reachability, not naming: nothing the production command starts may
        import the dashboard."""
        assert "bokeh_app" not in " ".join(imports_of("src/api.py"))

    def test_its_healthcheck_uses_readiness_not_liveness(self):
        """A process answering on a port is not evidence of a migrated
        database, a current schema or an observable build."""
        body = Path(PRODUCTION_DOCKERFILE).read_text()
        assert "/health/ready" in body
        assert "HEALTHCHECK" in body

    def test_it_sets_no_default_database_url(self):
        """A production default would be the fallback the preflight exists to
        refuse."""
        body = Path(PRODUCTION_DOCKERFILE).read_text()
        for line in body.splitlines():
            if line.strip().startswith("ENV") or "QUANTIFY_DATABASE_URL" in line:
                assert not re.search(r"QUANTIFY_DATABASE_URL\s*=\s*\S", line), line


class TestTheDashboardIsSeparate:
    def test_it_has_its_own_image(self):
        assert Path(DASHBOARD_DOCKERFILE).exists()

    def test_that_image_runs_it(self):
        assert DASHBOARD_ENTRYPOINT in container_command(DASHBOARD_DOCKERFILE)

    def test_it_is_not_production_reachable(self):
        assert not by_module()[
            "src/visualization/bokeh_app.py"].production_reachable

    def test_it_is_pinned_to_synthetic_data(self):
        dashboard = by_module()["src/visualization/bokeh_app.py"]
        assert dashboard.data_policy is DataPolicy.SYNTHETIC_ONLY
        body = Path(DASHBOARD_DOCKERFILE).read_text()
        assert "PILOT_DATA_POLICY=SYNTHETIC_ONLY" in body.replace(" \\", "")

    def test_compose_keeps_it_out_of_the_production_profile(self):
        compose = Path("docker-compose.yml").read_text()
        dashboard = compose.split("dashboard:", 1)[1]
        profiles = dashboard.split("profiles:", 1)[1].splitlines()[0]
        assert "production" not in profiles, (
            f"the dashboard is in the production profile: {profiles}")

    def test_compose_puts_the_api_in_the_production_profile(self):
        compose = Path("docker-compose.yml").read_text()
        api = compose.split("api:", 1)[1]
        profiles = api.split("profiles:", 1)[1].splitlines()[0]
        assert "production" in profiles


class TestEveryProductionSurfaceRunsThePreflight:
    def test_each_one_declares_that_it_must(self):
        for surface in production_surfaces():
            assert surface.requires_preflight, (
                f"{surface.name} is production-reachable and does not require "
                "the preflight")

    def test_the_entrypoint_actually_runs_it(self):
        """Watched, not read. The factory is called and the preflight must
        have been reached before it returns an application."""
        import src.api as api

        called = []
        import src.deploy.preflight as preflight

        original = preflight.run

        def watched(*args, **kwargs):
            called.append(True)
            return original(*args, **kwargs)

        preflight.run = watched
        try:
            api.create_app()
        finally:
            preflight.run = original
        assert called, "create_app returned an application without a preflight"

    def test_a_refusal_prevents_an_application_existing(self, tmp_path,
                                                       monkeypatch):
        import src.api as api

        monkeypatch.setenv("QUANTIFY_DEPLOYMENT_PROFILE", "production")
        for name, value in (("QUANTIFY_COMMIT", "abc"),
                            ("QUANTIFY_RELEASE_REF", "v1"),
                            ("QUANTIFY_IMAGE_DIGEST", "sha256:d"),
                            ("QUANTIFY_SNAPSHOT_ID", "syn-1")):
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/w.db")

        with pytest.raises(api.DeploymentRefused):
            api.create_app()

    def test_no_production_surface_is_unserved(self):
        for surface in unserved_surfaces():
            assert not surface.production_reachable, (
                f"{surface.name} is production-reachable and nothing starts it")


class TestTheDeclarationsMatchTheArtifacts:
    def test_every_declared_dockerfile_exists(self):
        for surface in SURFACES:
            if surface.dockerfile:
                assert Path(surface.dockerfile).exists(), surface.name

    def test_every_declared_module_exists(self):
        for surface in SURFACES:
            assert Path(surface.module).exists(), surface.name

    def test_each_declaration_records_why(self):
        for surface in SURFACES:
            assert surface.reason.strip(), surface.name

    def test_a_gated_surface_reads_no_file_directly(self):
        from tests.test_data_access import modules_reading_files

        readers = set(modules_reading_files())
        for surface in SURFACES:
            if surface.data_policy is DataPolicy.GATED:
                assert surface.module not in readers, (
                    f"{surface.module} is declared GATED and reads a file "
                    "directly")

    def test_the_production_set_is_not_empty(self):
        """A scan finding nothing production would pass everything above."""
        assert production_surfaces()
