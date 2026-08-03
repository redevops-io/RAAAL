"""Which surfaces a deployment serves, checked against the deployment.

`bokeh_app.py` was declared "the standalone Bokeh demo, served separately and
not part of the pilot surface". The declaration was written from the module's
name. `scripts/service.py` imports it and is the Dockerfile's `CMD`, so it is
what the container runs — the label was not merely imprecise, it was the
opposite of true.

So these tests read the Dockerfile and the import graph, and compare them with
the declarations. A classification that describes itself proves nothing; the
whole reason the wrong one survived is that nothing checked it against an
artifact.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.deploy.surfaces import (
    CONTAINER_ENTRYPOINT,
    DataPolicy,
    SURFACES,
    by_module,
    production_surfaces,
    unserved_surfaces,
)

DOCKERFILE = Path("Dockerfile")


def container_command():
    """The `CMD` the image runs, read from the Dockerfile.

    A `HEALTHCHECK` carries its own `CMD` continuation line, which is not the
    entrypoint. Only the exec form — `CMD ["a", "b"]` at the start of a line —
    is the container's command, so that is what this matches.
    """
    for line in DOCKERFILE.read_text().splitlines():
        if line.startswith("CMD") and "[" in line:
            return re.findall(r'"([^"]+)"', line)
    return []


def imports_of(path: Path):
    """Modules a file imports, from its AST."""
    tree = ast.parse(Path(path).read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


class TestTheDeclarationsMatchTheDeployment:
    def test_the_container_entrypoint_is_what_the_dockerfile_runs(self):
        command = container_command()
        assert CONTAINER_ENTRYPOINT in command, (
            f"the declared container entrypoint is {CONTAINER_ENTRYPOINT!r} "
            f"and the Dockerfile runs {command}")

    def test_the_dashboard_is_reachable_from_that_entrypoint(self):
        """The claim that was false. Proven by the import graph, not the name."""
        imported = imports_of(Path(CONTAINER_ENTRYPOINT))
        assert any("bokeh_app" in one for one in imported), (
            f"{CONTAINER_ENTRYPOINT} no longer imports the dashboard; the "
            "surface declaration needs revisiting")
        assert by_module()["src/visualization/bokeh_app.py"].production_reachable

    def test_every_declared_entrypoint_exists(self):
        for surface in SURFACES:
            if surface.entrypoint is not None:
                assert Path(surface.entrypoint).exists(), surface.name

    def test_every_declared_module_exists(self):
        for surface in SURFACES:
            assert Path(surface.module).exists(), surface.name

    def test_each_declaration_records_why(self):
        for surface in SURFACES:
            assert surface.reason.strip(), surface.name


class TestThePilotApiHasNoEntrypoint:
    """A finding, recorded as a test so it cannot be forgotten.

    This test *passes* while the pilot application is unserved. It is written
    to fail the moment someone gives it an entrypoint without updating the
    declaration — and to be deleted, deliberately, when the deployment exists.
    """

    def test_nothing_starts_the_pilot_application(self):
        served = []
        for path in list(Path("scripts").glob("*.py")) + [DOCKERFILE]:
            body = path.read_text()
            if "uvicorn" in body or "src.api:app" in body:
                served.append(str(path))
        assert served == [], (
            f"{served} now serves the pilot API. Update "
            "`src/deploy/surfaces.py`: `pilot-api` is no longer unserved, and "
            "its preflight now runs somewhere")

    def test_the_declaration_says_so(self):
        unserved = {one.name for one in unserved_surfaces()}
        assert "pilot-api" in unserved

    def test_the_reason_is_explicit_about_the_consequence(self):
        """A gap recorded as a shrug is a gap nobody acts on."""
        pilot = by_module()["src/api.py"]
        assert "nothing starts it" in pilot.reason.lower()


class TestTheServedSurfaceIsConstrained:
    def test_the_dashboard_is_restricted_to_synthetic_data(self):
        dashboard = by_module()["src/visualization/bokeh_app.py"]
        assert dashboard.data_policy is DataPolicy.SYNTHETIC_ONLY

    def test_a_gated_surface_reads_no_file_directly(self):
        """Anything declared GATED must not appear in the direct-reader scan."""
        from tests.test_data_access import modules_reading_files

        readers = set(modules_reading_files())
        for surface in SURFACES:
            if surface.data_policy is DataPolicy.GATED:
                assert surface.module not in readers, (
                    f"{surface.module} is declared GATED and reads a file "
                    "directly")

    def test_every_production_surface_declares_a_data_policy(self):
        for surface in production_surfaces():
            assert surface.data_policy in DataPolicy

    def test_the_production_set_is_not_empty(self):
        """A scan finding nothing production would pass everything above."""
        assert production_surfaces()
