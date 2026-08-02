"""What this deployment is, and whether it can say.

A manifest that substituted "unknown", "dev" or the working tree's git state for
a missing deployment variable would report a build that does not exist — and the
report would look exactly like a correct one.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.deploy import (
    REQUIRED_DEPLOYMENT_FACTS,
    BuildManifest,
    code_versions,
    read_manifest,
)
from src.deploy.manifest import OPTIONAL_DEPLOYMENT_FACTS, PUBLIC_FIELDS

COMPLETE = {name: f"value-of-{name.lower()}"
            for name in REQUIRED_DEPLOYMENT_FACTS}


class TestAnIncompleteBuildSaysSo:

    def test_an_empty_environment_is_not_observable(self):
        manifest = read_manifest({})
        assert manifest.observable is False
        assert set(manifest.missing) == set(REQUIRED_DEPLOYMENT_FACTS)

    @pytest.mark.parametrize("absent", list(REQUIRED_DEPLOYMENT_FACTS))
    def test_any_single_missing_fact_defeats_observability(self, absent):
        partial = {k: v for k, v in COMPLETE.items() if k != absent}
        manifest = read_manifest(partial)
        assert not manifest.observable
        assert absent in manifest.missing

    def test_nothing_is_guessed(self):
        """No "unknown", no "dev", no placeholder standing in for a fact."""
        rendered = str(read_manifest({}).private()).lower()
        for guess in ("unknown", "dev", "latest", "none", "n/a"):
            assert f"'{guess}'" not in rendered

    def test_it_does_not_fall_back_to_the_working_tree(self):
        """On a server there is no working tree; in development it would report
        the checkout rather than the running image.

        Checked against imports and calls, not text: the module docstring
        explains that git is never consulted, and says "git" to do so.
        """
        import ast
        import inspect

        from src.deploy import manifest

        tree = ast.parse(inspect.getsource(manifest))
        imported = {alias.name.split(".")[0]
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import)
                    for alias in node.names}
        imported |= {(node.module or "").split(".")[0]
                     for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom)}
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)}

        for reach in ("subprocess", "sh", "pygit2", "git"):
            assert reach not in imported, reach
        assert "check_output" not in called
        assert "run" not in called

    def test_a_complete_environment_is_observable(self):
        assert read_manifest(COMPLETE).observable


class TestTheTwoViews:

    def test_the_private_view_carries_the_deployment_facts(self):
        private = read_manifest(COMPLETE).private()
        for name in ("commit", "image_digest", "migration_head", "snapshot_id"):
            assert name in private

    def test_the_public_view_carries_none_of_them(self):
        public = read_manifest(COMPLETE).public()
        for name in ("image_digest", "migration_head", "snapshot_id",
                     "commit", "release_ref"):
            assert name not in public

    def test_the_public_view_carries_compatibility_versions(self):
        public = read_manifest(COMPLETE).public()
        assert public["compiler_version"]
        assert public["scope_schema_version"]
        assert public["canonicalization_version"]

    def test_the_public_view_reports_observability(self):
        assert read_manifest({}).public()["observable"] is False

    def test_every_public_field_is_declared(self):
        public = read_manifest(COMPLETE).public()
        assert set(public) <= set(PUBLIC_FIELDS)


class TestVersionsAreImportedNotRestated:

    def test_they_match_the_modules_that_declare_them(self):
        """Restated here they would be a second list that drifts, invisibly,
        because both lists look authoritative."""
        from src.mission.comparability import CLASSIFIER_VERSION
        from src.mission.evolution import COMPILER_VERSION
        from src.workspace.scope_disclosure import SCOPE_SCHEMA_VERSION

        versions = code_versions()
        assert versions["compiler_version"] == COMPILER_VERSION
        assert versions["classifier_version"] == CLASSIFIER_VERSION
        assert versions["scope_schema_version"] == SCOPE_SCHEMA_VERSION

    def test_no_version_is_a_literal_in_the_manifest(self):
        import inspect

        from src.deploy import manifest

        source = inspect.getsource(manifest.code_versions)
        for literal in ("@1", "@2", '"3"', '"2"'):
            assert literal not in source

    def test_every_declared_version_is_reported(self):
        assert len(code_versions()) >= 10


class TestTheLivePathConsultsIt:
    """Configuration and reachability, not configuration alone."""

    def test_health_reports_the_build(self):
        response = TestClient(app).get("/health")
        assert response.status_code == 200
        assert "build" in response.json()

    def test_health_exposes_only_the_public_view(self, monkeypatch):
        """The facts must be *present* for their absence to mean anything.

        With no deployment variables set there is nothing for the private view
        to leak, and this passes against an endpoint returning `private()`.
        """
        for name, value in COMPLETE.items():
            monkeypatch.setenv(name, value)

        payload = TestClient(app).get("/health").json()["build"]
        assert payload["observable"] is True
        for private in ("image_digest", "migration_head", "commit",
                        "snapshot_id", "release_ref"):
            assert private not in payload, private
        assert "value-of-quantify_image_digest" not in str(payload)

    def test_health_reports_unobservable_when_facts_are_absent(self,
                                                               monkeypatch):
        for name in REQUIRED_DEPLOYMENT_FACTS:
            monkeypatch.delenv(name, raising=False)
        payload = TestClient(app).get("/health").json()["build"]
        assert payload["observable"] is False

    def test_health_reports_observable_when_they_are_present(self,
                                                             monkeypatch):
        for name, value in COMPLETE.items():
            monkeypatch.setenv(name, value)
        payload = TestClient(app).get("/health").json()["build"]
        assert payload["observable"] is True

    def test_the_endpoint_does_not_restate_the_manifest(self):
        """A second assembly in the route would drift from the first."""
        import inspect

        import src.api as api

        source = inspect.getsource(api.health)
        assert "read_manifest" in source
        assert "QUANTIFY_" not in source
