"""What a production instance must establish before it serves anything.

    build identity -> URL -> connect -> version -> migration head -> parity

Each step has its own outcome. A single "startup failed" would discard the only
thing an operator needs — which of these is wrong — while the public surface
still reports nothing but `ready: false`.

The refusals are what matter, so each is constructed deliberately and each
fails a distinct test. Several of them protect against a state the ordinary
configuration cannot reach, which is exactly why the fixture has to build it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.deploy.preflight import (
    PROFILE_VAR,
    PROVEN_POSTGRES_MAJOR,
    Profile,
    Result,
    configured_profile,
    run,
)

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

#: A complete set of build stamps, so a test about the database is not
#: accidentally a test about the manifest.
BUILD = {
    "QUANTIFY_COMMIT": "abc123",
    "QUANTIFY_RELEASE_REF": "v1.0.0",
    "QUANTIFY_IMAGE_DIGEST": "sha256:dead",
    "QUANTIFY_SNAPSHOT_ID": "syn-2026-01",
}


def production(**overrides):
    return {PROFILE_VAR: "production", **BUILD, **overrides}


class TestTheProfile:
    def test_it_defaults_to_local(self):
        """Defaulting to production would break every checkout; defaulting away
        from it in production would let a deployment run unchecked. So it is
        explicit, and the default is the harmless one."""
        assert configured_profile({}) is Profile.LOCAL

    def test_an_unknown_profile_is_refused(self):
        with pytest.raises(ValueError, match="not a deployment profile"):
            configured_profile({PROFILE_VAR: "staging-ish"})

    def test_local_permits_sqlite(self, tmp_path):
        outcome = run({PROFILE_VAR: "local",
                       "QUANTIFY_DATABASE_URL": f"sqlite:///{tmp_path}/w.db"})
        assert outcome.result is Result.READY

    def test_test_profile_permits_sqlite(self, tmp_path):
        outcome = run({PROFILE_VAR: "test",
                       "QUANTIFY_DATABASE_URL": f"sqlite:///{tmp_path}/w.db"})
        assert outcome.result is Result.READY


class TestProductionRefusesSqlite:
    def test_a_sqlite_url_is_refused(self, tmp_path):
        outcome = run(production(
            QUANTIFY_DATABASE_URL=f"sqlite:///{tmp_path}/w.db"))
        assert outcome.result is Result.REFUSED_CONFIGURATION
        assert "PostgreSQL" in outcome.detail

    def test_the_sqlite_file_is_never_touched(self, tmp_path):
        """`Database` creates the parent directory on construction, so a check
        made after building one would already have written to disk."""
        target = tmp_path / "untouched" / "w.db"
        run(production(QUANTIFY_DATABASE_URL=f"sqlite:///{target}"))
        assert not target.exists()
        assert not target.parent.exists(), (
            "the preflight created a directory for a database it refuses to use")

    def test_a_missing_url_is_refused_with_no_fallback(self):
        """Falling back to `data/workspace.db` would be the deployment
        equivalent of the `_prices()` bypass: a live path quietly reading
        something nobody authorised."""
        outcome = run(production())
        assert outcome.result is Result.REFUSED_CONFIGURATION
        assert "no production fallback" in outcome.detail

    def test_an_unknown_engine_is_refused(self):
        outcome = run(production(QUANTIFY_DATABASE_URL="mysql://host/db"))
        assert outcome.result is Result.REFUSED_CONFIGURATION


class TestBuildObservability:
    @pytest.mark.parametrize("missing", sorted(BUILD))
    def test_each_missing_stamp_refuses(self, missing, tmp_path):
        environ = production(QUANTIFY_DATABASE_URL="postgresql://h/db")
        environ.pop(missing)
        outcome = run(environ)
        assert outcome.result is Result.BUILD_UNOBSERVABLE
        assert missing in outcome.detail

    def test_it_is_checked_before_the_database(self):
        """A deployment that cannot say what it is cannot be diagnosed when a
        later step fails, so this comes first."""
        environ = production(QUANTIFY_DATABASE_URL="postgresql://nowhere:1/db")
        environ.pop("QUANTIFY_COMMIT")
        assert run(environ).result is Result.BUILD_UNOBSERVABLE

    def test_a_complete_build_passes_this_step(self, tmp_path):
        outcome = run({PROFILE_VAR: "local", **BUILD,
                       "QUANTIFY_DATABASE_URL": f"sqlite:///{tmp_path}/w.db"})
        assert outcome.facts["build"]["observable"] is True


class TestUnreachableDatabase:
    def test_it_is_reported_as_unavailable(self):
        outcome = run(production(
            QUANTIFY_DATABASE_URL="postgresql://quantify:pw@127.0.0.1:1/db"))
        assert outcome.result is Result.DATABASE_UNAVAILABLE

    def test_the_detail_carries_no_credentials_or_host(self):
        outcome = run(production(
            QUANTIFY_DATABASE_URL="postgresql://user:secret@10.9.9.9:5999/db"))
        for leak in ("secret", "10.9.9.9", "5999", "user"):
            assert leak not in outcome.detail, f"{leak!r} leaked"

    def test_the_public_view_says_only_that_it_is_not_ready(self):
        outcome = run(production(
            QUANTIFY_DATABASE_URL="postgresql://user:secret@10.9.9.9:5999/db"))
        assert outcome.public() == {"ready": False}


@pytest.mark.skipif(not POSTGRES_URL, reason="needs a real PostgreSQL")
class TestAgainstARealDatabase:
    @pytest.fixture
    def migrated(self):
        from src.db import migrate
        from src.db.engine import Database
        from sqlalchemy import text

        database = Database(POSTGRES_URL)
        engine = database.sqlalchemy_engine()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()
        migrate.upgrade(database)
        return database

    def test_a_migrated_database_is_ready(self, migrated):
        outcome = run(production(QUANTIFY_DATABASE_URL=POSTGRES_URL),
                      checked_at="2026-08-01T00:00:00Z")
        assert outcome.result is Result.READY, outcome.detail

    def test_an_unmigrated_database_is_a_migration_mismatch(self):
        from sqlalchemy import text

        from src.db.engine import Database

        engine = Database(POSTGRES_URL).sqlalchemy_engine()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()

        outcome = run(production(QUANTIFY_DATABASE_URL=POSTGRES_URL))
        assert outcome.result is Result.MIGRATION_MISMATCH
        assert "never been migrated" in outcome.detail

    def test_a_database_behind_the_application_is_refused(self, migrated):
        from src.db import migrate

        migrate.downgrade(migrated, "-1")
        outcome = run(production(QUANTIFY_DATABASE_URL=POSTGRES_URL))
        assert outcome.result is Result.MIGRATION_MISMATCH

    def test_a_database_ahead_of_the_application_is_refused(self, migrated,
                                                            monkeypatch):
        """Not safe merely because it has everything the app expects — it may
        encode semantics this code does not know about."""
        from src.db import migrate

        monkeypatch.setattr(migrate, "code_head", lambda: "an-older-revision")
        outcome = run(production(QUANTIFY_DATABASE_URL=POSTGRES_URL))
        assert outcome.result is Result.MIGRATION_MISMATCH

    def test_a_hand_edited_schema_is_a_parity_failure(self, migrated):
        """Run against the *connected* database. A freshly migrated scratch
        database proves the migrations agree with the model and says nothing
        about the instance being started."""
        conn = migrated.connect()
        try:
            conn.execute("ALTER TABLE worksheet DROP COLUMN canonical_hash")
            conn.commit()
        finally:
            conn.close()
        outcome = run(production(QUANTIFY_DATABASE_URL=POSTGRES_URL))
        assert outcome.result is Result.SCHEMA_MISMATCH

    def test_a_dropped_index_is_a_parity_failure(self, migrated):
        conn = migrated.connect()
        try:
            conn.execute("DROP INDEX worksheet_intent_sequence")
            conn.commit()
        finally:
            conn.close()
        outcome = run(production(QUANTIFY_DATABASE_URL=POSTGRES_URL))
        assert outcome.result is Result.SCHEMA_MISMATCH

    def test_an_unsupported_version_names_what_is_proven(self, migrated,
                                                        monkeypatch):
        import src.deploy.preflight as preflight

        monkeypatch.setattr(preflight, "PROVEN_POSTGRES_MAJOR", 99)
        outcome = run(production(QUANTIFY_DATABASE_URL=POSTGRES_URL))
        assert outcome.result is Result.UNSUPPORTED_DATABASE
        assert "proven against 99" in outcome.detail
        assert "not unsupported forever" in outcome.detail


@pytest.mark.skipif(not POSTGRES_URL, reason="needs a real PostgreSQL")
class TestTheStartupProof:
    @pytest.fixture
    def outcome(self):
        from src.db import migrate
        from src.db.engine import Database
        from sqlalchemy import text

        database = Database(POSTGRES_URL)
        engine = database.sqlalchemy_engine()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()
        migrate.upgrade(database)
        return run(production(QUANTIFY_DATABASE_URL=POSTGRES_URL),
                   checked_at="2026-08-01T00:00:00Z")

    def test_it_records_what_was_checked(self, outcome):
        proof = outcome.proof()
        assert proof["result"] == "READY"
        assert proof["deployment_profile"] == "production"
        assert proof["database"]["engine"] == "postgresql"
        assert proof["database"]["schema_parity"] == "PASS"
        assert proof["database"]["migration_head"]
        assert proof["build"]["observable"] is True
        assert proof["checked_at"] == "2026-08-01T00:00:00Z"

    def test_it_names_the_version_that_is_running(self, outcome):
        assert outcome.proof()["database"]["version"].startswith(
            str(PROVEN_POSTGRES_MAJOR))

    def test_it_carries_no_credentials_or_network_detail(self, outcome):
        """`@` is deliberately not in this list: version strings like
        `scope-disclosure@1` legitimately contain one, and a check that flags
        them is a check that gets deleted."""
        body = str(outcome.proof())
        for leak in ("quantify_dev", "localhost", "5433", "password",
                     "postgresql://", "://"):
            assert leak not in body, f"{leak!r} appeared in the startup proof"

    def test_it_does_not_carry_the_connection_url(self, outcome):
        assert "url" not in {key.lower() for key in
                             outcome.proof()["database"]}


class TestTheServiceRefusesToServe:
    def test_production_startup_raises_on_a_refusal(self, tmp_path,
                                                    monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api

        monkeypatch.setenv(PROFILE_VAR, "production")
        for name, value in BUILD.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/w.db")

        with pytest.raises(RuntimeError, match="refusing to serve"):
            with TestClient(api.app):
                pass

    def test_the_refusal_carries_no_detail(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api

        monkeypatch.setenv(PROFILE_VAR, "production")
        for name, value in BUILD.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/w.db")

        with pytest.raises(RuntimeError) as caught:
            with TestClient(api.app):
                pass
        assert str(tmp_path) not in str(caught.value)

    def test_a_local_profile_still_starts(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api

        monkeypatch.setenv(PROFILE_VAR, "local")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/w.db")
        with TestClient(api.app) as client:
            assert client.get("/ready").status_code == 200

    def test_readiness_reports_only_the_outcome(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api

        monkeypatch.setenv(PROFILE_VAR, "local")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/w.db")
        with TestClient(api.app) as client:
            body = client.get("/ready").json()
        assert body == {"ready": True}

    def test_readiness_and_liveness_are_separate_endpoints(self, tmp_path,
                                                          monkeypatch):
        """A migration mismatch should make an instance unready, not make it
        indistinguishable from a dead process."""
        from fastapi.testclient import TestClient

        import src.api as api

        monkeypatch.setenv(PROFILE_VAR, "local")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/w.db")
        with TestClient(api.app) as client:
            assert client.get("/health").json()["status"] == "ok"
            assert "ready" in client.get("/ready").json()
