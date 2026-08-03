"""The application store opens the database the deployment configured.

The preflight validated PostgreSQL — reachable, migrated, schema-parity
checked — and `WorkspaceStore()` then opened `sqlite:///data/workspace.db`,
because it substituted a default before the resolver could consult
`QUANTIFY_DATABASE_URL`. Both halves looked correct in isolation and were
talking about different databases.

Nothing caught it for the whole of Gate 2. Every store test passes an explicit
path, and every preflight test constructs its own `Database` — so the one
combination that matters, *the application picking its own target*, was the one
nothing exercised.
"""
from __future__ import annotations

import pytest

from src.db.engine import DATABASE_URL_VAR
from src.workspace.store import WorkspaceStore

import os

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")


class TestTheStoreHonoursTheConfiguredTarget:
    """Checked with a reachable target, because the store connects on
    construction. The defect is the same either way: the configured URL is
    ignored and a default is opened instead."""

    def test_it_uses_the_configured_url(self, tmp_path, monkeypatch):
        configured = f"sqlite:///{tmp_path}/configured.db"
        monkeypatch.setenv(DATABASE_URL_VAR, configured)
        assert WorkspaceStore().db.url == configured

    def test_it_does_not_open_the_default_instead(self, tmp_path, monkeypatch):
        """The exact failure: a configured database ignored for a default."""
        monkeypatch.setenv(DATABASE_URL_VAR,
                           f"sqlite:///{tmp_path}/configured.db")
        assert "data/workspace.db" not in WorkspaceStore().db.url

    @pytest.mark.skipif(not POSTGRES_URL, reason="needs a real PostgreSQL")
    def test_it_uses_the_configured_engine(self, monkeypatch):
        """The failure was not a wrong path — it was a wrong *engine*. The
        preflight validated PostgreSQL and the store opened SQLite."""
        from src.db.engine import Dialect

        monkeypatch.setenv(DATABASE_URL_VAR, POSTGRES_URL)
        assert WorkspaceStore().db.dialect is Dialect.POSTGRESQL

    def test_it_falls_back_to_a_local_file_when_nothing_is_set(self,
                                                               monkeypatch):
        """Correct for a checkout, and refused in production by the preflight."""
        monkeypatch.delenv(DATABASE_URL_VAR, raising=False)
        assert WorkspaceStore().db.url == "sqlite:///data/workspace.db"

    def test_an_explicit_target_still_wins(self, tmp_path, monkeypatch):
        """Twenty test files pass a path; that must keep meaning what it did."""
        monkeypatch.setenv(DATABASE_URL_VAR,
                           f"sqlite:///{tmp_path}/configured.db")
        store = WorkspaceStore(tmp_path / "explicit.db")
        assert "explicit.db" in store.db.url


class TestThePreflightAndTheStoreAgree:
    """The two halves must name the same database.

    Checking each against the environment separately is what allowed them to
    diverge — each was right about its own half.
    """

    def test_they_resolve_the_same_target(self, tmp_path, monkeypatch):
        from src.db.engine import Database

        monkeypatch.setenv(DATABASE_URL_VAR,
                           f"sqlite:///{tmp_path}/configured.db")
        assert WorkspaceStore().db.url == Database().url

    @pytest.mark.skipif(not POSTGRES_URL, reason="needs a real PostgreSQL")
    def test_they_agree_on_postgresql(self, monkeypatch):
        from src.db.engine import Database

        monkeypatch.setenv(DATABASE_URL_VAR, POSTGRES_URL)
        assert WorkspaceStore().db.url == Database().url

    def test_they_agree_on_a_local_fallback_too(self, monkeypatch):
        from src.db.engine import Database

        monkeypatch.delenv(DATABASE_URL_VAR, raising=False)
        assert WorkspaceStore().db.url == Database().url

    def test_the_preflight_checks_what_the_store_will_open(self, tmp_path,
                                                           monkeypatch):
        """End to end: a preflight reporting SQLite must mean the store opens
        SQLite, and the same for PostgreSQL."""
        from src.deploy.preflight import run

        monkeypatch.setenv("QUANTIFY_DEPLOYMENT_PROFILE", "local")
        monkeypatch.setenv(DATABASE_URL_VAR, f"sqlite:///{tmp_path}/w.db")
        outcome = run()
        assert outcome.facts["database"]["engine"] == \
            WorkspaceStore().db.dialect.value
