"""A restored deployment still understands its artifacts.

    pg_dump  ->  fresh PostgreSQL  ->  fresh container
             ->  reopen plans and worksheets
             ->  verify parser identity and market-data provenance
             ->  create the next revision

The acceptance criterion is deliberately user-facing. "The rows copied" is a
statement about `pg_dump`, and it is true of a restore that produces a database
this build cannot serve: a schema at the wrong migration head, an access event
whose foreign key did not survive, a plan whose pinned parse is unreadable, a
worksheet citing a run that did not come across.

Every one of those passes a row count and fails a user opening their plan.

So the check is the thing a person would do after a restore — read what they
saved, and then add to it. The last step matters most: reopening proves the
data survived, and creating the next revision proves the restored system can
still *write*, which is where a missing constraint or a stale sequence shows.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")
PG_DUMP = shutil.which("pg_dump") or shutil.which("docker")
DOCKER = shutil.which("docker")

pytestmark = [
    pytest.mark.skipif(not POSTGRES_URL, reason="needs a reachable PostgreSQL"),
    pytest.mark.skipif(not DOCKER, reason="the restore target is a container"),
]

OWNER = "pilot"
DESCRIPTION = ("I contribute $7,000 a year to a Roth IRA in VOO on the first "
               "trading day of January, reinvesting dividends, and I never "
               "sell.")

SOURCE_CONTAINER = "quantify-pg"
RESTORE_CONTAINER = "quantify-pg-restore"
RESTORE_PORT = "5434"
RESTORE_URL = (f"postgresql://quantify:quantify_dev@localhost:{RESTORE_PORT}"
               "/quantify")


def docker(*args, timeout=180):
    return subprocess.run([DOCKER, *args], capture_output=True, text=True,
                          timeout=timeout)


@pytest.fixture(scope="module")
def populated():
    """A workspace with a saved plan, a run, a worksheet and a delivery."""
    from sqlalchemy import text

    from src.db import migrate
    from src.db.engine import Database

    os.environ.setdefault("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    os.environ["QUANTIFY_DATABASE_URL"] = POSTGRES_URL
    os.environ["QUANTIFY_PARSER_MODE"] = "MODEL_ASSISTED"
    os.environ["QUANTIFY_PARSER_MODEL"] = "claude-sonnet-5"

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    migrate.upgrade(database)

    from fastapi.testclient import TestClient

    import src.api as api

    from src.deploy.context import unbind
    from tests.conftest import submit_rendered_confirmation

    unbind()
    with TestClient(api.app) as client:
        response, plan_id = submit_rendered_confirmation(
            client, DESCRIPTION, title="Roth")
    assert response.status_code == 303, response.text
    return plan_id


@pytest.fixture(scope="module")
def restored(populated):
    """Dump the source, start a fresh instance, restore into it."""
    docker("rm", "-f", RESTORE_CONTAINER)
    started = docker(
        "run", "-d", "--name", RESTORE_CONTAINER,
        "-e", "POSTGRES_USER=quantify", "-e", "POSTGRES_PASSWORD=quantify_dev",
        "-e", "POSTGRES_DB=quantify", "-p", f"{RESTORE_PORT}:5432",
        "postgres:16")
    assert started.returncode == 0, started.stderr

    try:
        _wait_until_ready()
        dumped = docker("exec", SOURCE_CONTAINER, "pg_dump", "-U", "quantify",
                        "-d", "quantify", timeout=300)
        assert dumped.returncode == 0, dumped.stderr[-2000:]
        assert dumped.stdout.strip(), "the dump is empty"

        restore = subprocess.run(
            [DOCKER, "exec", "-i", RESTORE_CONTAINER, "psql", "-U", "quantify",
             "-d", "quantify", "-v", "ON_ERROR_STOP=1"],
            input=dumped.stdout, capture_output=True, text=True, timeout=300)
        assert restore.returncode == 0, restore.stderr[-3000:]
        yield populated
    finally:
        docker("rm", "-f", RESTORE_CONTAINER)


def _wait_until_ready(attempts=40):
    import time

    for _ in range(attempts):
        probe = docker("exec", RESTORE_CONTAINER, "pg_isready", "-U", "quantify")
        if probe.returncode == 0:
            return
        time.sleep(1)
    raise AssertionError("the restore target never became ready")


def restored_store():
    from src.workspace.store import WorkspaceStore

    return WorkspaceStore(RESTORE_URL)


class TestTheRestoredDatabaseIsServable:
    """Not "the rows arrived" — whether this build would agree to serve it."""

    def test_the_preflight_accepts_it(self, restored):
        from src.deploy.preflight import Result, run

        outcome = run({"QUANTIFY_DEPLOYMENT_PROFILE": "production",
                       "QUANTIFY_DATABASE_URL": RESTORE_URL,
                       "QUANTIFY_PARSER_MODE": "DETERMINISTIC",
                       "QUANTIFY_COMMIT": "c", "QUANTIFY_RELEASE_REF": "r",
                       "QUANTIFY_IMAGE_DIGEST": "d",
                       "QUANTIFY_SNAPSHOT_ID": "s"})
        assert outcome.result is Result.READY, outcome.detail

    def test_the_migration_head_came_across(self, restored):
        from src.db.engine import Database
        from src.db.migrate import applied_revision, code_head

        assert applied_revision(Database(RESTORE_URL)) == code_head()

    def test_the_constraints_came_across(self, restored):
        """A restore that copied rows and dropped a foreign key produces a
        database where evidence can be deleted out from under a figure."""
        from src.db.errors import DatabaseFailure

        store = restored_store()
        with pytest.raises(DatabaseFailure):
            with store._conn() as conn:
                conn.execute("DELETE FROM market_data_access_event")


class TestAUserCanReadWhatTheySaved:
    def test_the_plan_reopens(self, restored):
        record = restored_store().get_plan(restored, OWNER)
        assert record is not None
        assert record["stated_text"] == DESCRIPTION

    def test_the_worksheet_is_there(self, restored):
        assert restored_store().worksheet_for_scenario(restored, OWNER) \
            is not None

    def test_the_run_is_there(self, restored):
        runs = restored_store().runs_for(restored, OWNER)
        assert runs, "the plan reopened with no run; the figure is gone"

    def test_the_parser_identity_survived(self, restored):
        """How it was interpreted is part of what was saved."""
        record = restored_store().get_plan(restored, OWNER)
        identity = (record.get("parse") or {}).get("parser") or {}
        assert identity.get("mode") == "MODEL_ASSISTED"
        assert identity.get("model") == "claude-sonnet-5"

    def test_the_market_data_provenance_survived(self, restored):
        from src.market_data.provenance import ProvenanceStatus, from_json

        runs = restored_store().runs_for(restored, OWNER)
        assert runs, "no run came across; this case would pass vacuously"
        for run in runs:
            carried = from_json(run["result"].get("market_data"))
            assert carried.status is ProvenanceStatus.RECORDED
            assert carried.identifies_data

    def test_the_delivery_chain_still_verifies(self, restored):
        """The strongest read-side check: every stored figure still traces to
        the frame that produced it."""
        store = restored_store()
        runs = store.runs_for(restored, OWNER)
        assert runs, (
            "no run came across, so this proves nothing about whether the "
            "chain survived — a loop over an empty set is not a check")
        for run in runs:
            assert store.verify_access_chain(run["run_id"], OWNER) == [], (
                f"{run['run_id']} no longer verifies after the restore")


class TestTheRestoredSystemCanStillWrite:
    """Reopening proves the data survived. This proves the system did.

    A missing constraint, a stale sequence or a half-restored index shows up on
    the next write, not on the next read — and a restore nobody has written to
    is a restore nobody has finished testing.
    """

    def test_a_second_plan_can_be_created(self, restored, monkeypatch):
        from fastapi.testclient import TestClient

        from src.deploy.context import unbind

        monkeypatch.setenv("QUANTIFY_DATABASE_URL", RESTORE_URL)
        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "DETERMINISTIC")
        unbind()

        import src.api as api

        from tests.conftest import submit_rendered_confirmation

        with TestClient(api.app) as client:
            response, plan_id = submit_rendered_confirmation(
                client, DESCRIPTION, title="After the restore")
        assert response.status_code == 303, response.text
        assert plan_id and plan_id != restored

    def test_the_new_plan_has_its_own_run_and_worksheet(self, restored,
                                                         monkeypatch):
        from fastapi.testclient import TestClient

        from src.deploy.context import unbind

        monkeypatch.setenv("QUANTIFY_DATABASE_URL", RESTORE_URL)
        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "DETERMINISTIC")
        unbind()

        import src.api as api

        from tests.conftest import submit_rendered_confirmation

        with TestClient(api.app) as client:
            _response, plan_id = submit_rendered_confirmation(
                client, DESCRIPTION, title="Second after restore")
        store = restored_store()
        assert store.runs_for(plan_id, OWNER)
        assert store.worksheet_for_scenario(plan_id, OWNER) is not None

    def test_the_original_plan_is_untouched(self, restored):
        """Writing after a restore must not disturb what was restored."""
        record = restored_store().get_plan(restored, OWNER)
        assert record["stated_text"] == DESCRIPTION
        identity = (record.get("parse") or {}).get("parser") or {}
        assert identity.get("mode") == "MODEL_ASSISTED"
