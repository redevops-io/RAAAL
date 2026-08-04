"""A database failure is operator evidence, never an API payload.

The anchor is a real one. `psycopg` reports a foreign-key violation as:

    foreign key constraint "fk_plan_run_access_event" on table "plan_run"
    DETAIL: Key (owner, access_event_id)=(alice, mdae-3f1c) is still referenced

One string: a constraint name, two tables, the column composition of a key, a
tenant identifier and one of that tenant's object ids. It reached a caller
through `DELETE FROM market_data_access_event` — a path nobody had wrapped,
which is how every instance of this class has arrived.

**The 23503 case runs against real PostgreSQL.** A mocked exception proves the
handler formats a fake correctly; it cannot prove the route does not leak the
driver's actual text, because the mock never contains it. The forbidden-token
assertions are only meaningful when the tokens genuinely exist somewhere in the
call.

**Two channels, asserted separately.** The public one must contain none of it;
the private one must contain all of it. A test that only checked the first
would pass just as well against `from None`, which blinds the operator to solve
the caller's problem.
"""
from __future__ import annotations

import ast
import logging
import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.db.errors import (
    Classification,
    DatabaseFailure,
    InternalReason,
    PublicCode,
    Retry,
    classify,
    translate,
)

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")


@pytest.fixture
def operator_log():
    """What the operator channel actually received.

    A handler attached to `uvicorn.error` directly rather than pytest's
    `caplog`, which captures by propagation to the root logger. Alembic
    reconfigures logging when a migration runs, so the propagation-based
    version passed alone and failed in a full run — an assertion about the
    private channel that only holds in some orders is not evidence about the
    private channel.
    """
    logger = logging.getLogger("uvicorn.error")
    lines = []

    class _Sink(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    sink = _Sink(level=logging.ERROR)
    logger.addHandler(sink)
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield lines
    finally:
        logger.removeHandler(sink)
        logger.setLevel(previous)

#: Everything that must never appear in a response body or header.
#:
#: Derived from the actual PostgreSQL message for the anchor case rather than
#: imagined: constraint name, table names, key columns, key values, the
#: SQLSTATE, the driver class and the server's own vocabulary.
FORBIDDEN = (
    "fk_plan_run_access_event", "plan_run", "market_data_access_event",
    "access_event_id", "23503", "ForeignKeyViolation", "psycopg",
    "sqlite3", "IntegrityError", "DETAIL:", "Key (",
    "foreign key", "SELECT", "INSERT", "DELETE FROM", "UPDATE ",
)
# `DETAIL:` carries the colon deliberately. PostgreSQL's disclosure is its own
# `DETAIL:` field marker; the bare word matches FastAPI's `{"detail": ...}`
# envelope key, so the looser token failed an ordinary, correct 404 — a
# forbidden-token list that flags safe responses is a list that gets widened
# until it flags nothing.

#: Words the public vocabulary legitimately contains. `CONSTRAINT_CONFLICT`
#: says "constraint", and that is the semantic category rather than a
#: disclosure — a constraint *name* is the disclosure. Listed so the
#: distinction is a decision rather than an omission from `FORBIDDEN`.
DELIBERATELY_PUBLIC = ("constraint", "conflict", "database")

#: Modules that *translate* a driver exception rather than handling it — they
#: catch, wrap and re-raise. Distinct from a module that catches one and
#: decides what a caller sees, which is what the inventory forbids.
TRANSLATION_POINTS = {
    "src/db/engine.py",
    "src/db/errors.py",
    # These two open SQLite directly and are not yet ported, so each is its own
    # translation point until it is.
    "src/ledger.py",
    "src/telemetry/trace_store.py",
}


class TestTheTaxonomyMapsEveryStateItClaims:
    @pytest.mark.parametrize("code,public,reason,retry", [
        ("40001", PublicCode.DATABASE_CONTENTION,
         InternalReason.SERIALIZATION_FAILURE, Retry.UNCHANGED),
        ("40P01", PublicCode.DATABASE_CONTENTION, InternalReason.DEADLOCK,
         Retry.UNCHANGED),
        ("55P03", PublicCode.DATABASE_CONTENTION,
         InternalReason.LOCK_UNAVAILABLE, Retry.UNCHANGED),
        ("23505", PublicCode.CONSTRAINT_CONFLICT,
         InternalReason.DUPLICATE_IDENTITY, Retry.AFTER_REREAD),
        ("23503", PublicCode.CONSTRAINT_CONFLICT, InternalReason.MISSING_PARENT,
         Retry.AFTER_REREAD),
        ("23514", PublicCode.CONSTRAINT_CONFLICT,
         InternalReason.CHECK_VIOLATION, Retry.NEVER),
        ("23P01", PublicCode.CONSTRAINT_CONFLICT,
         InternalReason.EXCLUSION_VIOLATION, Retry.AFTER_REREAD),
        ("08006", PublicCode.DATABASE_UNAVAILABLE, InternalReason.UNREACHABLE,
         Retry.UNCHANGED),
        ("57P01", PublicCode.DATABASE_UNAVAILABLE,
         InternalReason.ADMIN_SHUTDOWN, Retry.UNCHANGED),
    ])
    def test_it_classifies(self, code, public, reason, retry):
        found = classify(_with_sqlstate(code))
        assert found.code is public
        assert found.reason is reason
        assert found.retry is retry

    def test_an_unknown_sqlstate_stops_rather_than_retries(self):
        """Failing towards "stop" is the safe direction: a retry loop on an
        unknown fault turns one failure into sustained load against a database
        that is already unwell."""
        found = classify(_with_sqlstate("XX000"))
        assert found.code is PublicCode.DATABASE_INTERNAL_FAILURE
        assert found.retry is Retry.NEVER
        assert not found.retry.retryable

    def test_an_unenumerated_connection_code_is_still_unavailable(self):
        found = classify(_with_sqlstate("08P01"))
        assert found.code is PublicCode.DATABASE_UNAVAILABLE

    def test_a_missing_parent_and_a_duplicate_differ_internally(self):
        """Both `CONSTRAINT_CONFLICT` publicly. Distinguishing them for a
        caller is what would tell them another tenant holds that id."""
        parent = classify(_with_sqlstate("23503"))
        duplicate = classify(_with_sqlstate("23505"))
        assert parent.code is duplicate.code
        assert parent.reason is not duplicate.reason

    def test_a_stale_transition_is_not_a_driver_failure(self):
        """The statement succeeded; the assumption it was issued under did
        not. Folding it into CONSTRAINT_CONFLICT would tell the caller to
        change their request when they must re-read.

        Asserted through the exception production actually raises, not through
        a constructor helper — the helper version passed while nothing in
        `src/` called it.
        """
        from src.workspace.apply import ProposalConflict

        assert ProposalConflict.public_code is PublicCode.STALE_TRANSITION
        assert ProposalConflict.retry_disposition is Retry.AFTER_REREAD

    def test_moving_two_rows_is_an_integrity_failure_not_a_race(self):
        from src.workspace.apply import TransitionIntegrityError

        assert TransitionIntegrityError.public_code is \
            PublicCode.TRANSITION_INTEGRITY_FAILURE
        assert TransitionIntegrityError.retry_disposition is Retry.NEVER

    @pytest.mark.parametrize("code,status", [
        (PublicCode.CONSTRAINT_CONFLICT, 409),
        (PublicCode.STALE_TRANSITION, 409),
        (PublicCode.TRANSITION_INTEGRITY_FAILURE, 500),
        (PublicCode.DATABASE_UNAVAILABLE, 503),
        (PublicCode.DATABASE_INTERNAL_FAILURE, 500),
    ])
    def test_the_status_mapping(self, code, status):
        failure = DatabaseFailure(
            Classification(code, InternalReason.UNCLASSIFIED, Retry.NEVER))
        assert failure.status == status

    def test_contention_splits_on_what_the_caller_must_do(self):
        """`503` when the same request may be reissued, `409` when the state it
        was planned against has moved. The failure knows; the route does not
        decide."""
        unchanged = DatabaseFailure(Classification(
            PublicCode.DATABASE_CONTENTION, InternalReason.DEADLOCK,
            Retry.UNCHANGED))
        reread = DatabaseFailure(Classification(
            PublicCode.DATABASE_CONTENTION, InternalReason.LOCK_UNAVAILABLE,
            Retry.AFTER_REREAD))
        assert unchanged.status == 503
        assert reread.status == 409

    def test_every_public_code_has_a_message_and_a_status(self):
        for code in PublicCode:
            failure = DatabaseFailure(
                Classification(code, InternalReason.UNCLASSIFIED, Retry.NEVER))
            assert failure.public_message.strip()
            assert 400 <= failure.status < 600


def _with_sqlstate(code):
    error = RuntimeError("a driver said something with detail in it")
    error.sqlstate = code
    return error


class TestTheTwoChannelsAreSeparate:
    def test_the_public_payload_carries_no_driver_text(self):
        leaky = RuntimeError(
            'foreign key constraint "fk_plan_run_access_event" on table '
            '"plan_run" DETAIL: Key (owner, access_event_id)=(alice, mdae-3f1c)')
        leaky.sqlstate = "23503"
        failure = translate(leaky, operation="DELETE market_data_access_event")

        rendered = str(failure.public())
        for token in ("fk_plan_run_access_event", "alice", "mdae-3f1c",
                      "23503", "DETAIL"):
            assert token not in rendered, token

    def test_str_of_the_failure_is_the_fixed_message(self):
        """Even a careless `str(exc)` at some future boundary is safe, because
        there is no path from the driver exception into the message."""
        leaky = RuntimeError("Key (owner, plan_id)=(alice, p-1) is not present")
        leaky.sqlstate = "23503"
        assert "alice" not in str(translate(leaky))

    def test_the_private_record_keeps_everything(self):
        leaky = RuntimeError("Key (owner, plan_id)=(alice, p-1) is not present")
        leaky.sqlstate = "23503"
        failure = translate(leaky, operation="INSERT plan_run")

        private = failure.private()
        assert private["sqlstate"] == "23503"
        assert private["reason"] == InternalReason.MISSING_PARENT.value
        assert "alice" in private["driver_detail"]
        assert private["operation"] == "INSERT plan_run"

    def test_the_cause_is_chained_not_discarded(self):
        """`from None` would sanitise the public channel by blinding the
        operator one. Both must work."""
        leaky = RuntimeError("something diagnostic")
        leaky.sqlstate = "40P01"
        assert translate(leaky).__cause__ is leaky

    def test_a_narrowed_reason_does_not_change_the_public_code(self):
        leaky = RuntimeError("...")
        leaky.sqlstate = "23503"
        failure = translate(leaky, reason=InternalReason.CROSS_SCOPE_REFERENCE)
        assert failure.reason is InternalReason.CROSS_SCOPE_REFERENCE
        assert failure.code is PublicCode.CONSTRAINT_CONFLICT


class TestNoDriverExceptionCrossesTheEngine:
    """Translation lives at the lowest layer that can classify mechanically,
    so nothing above it ever holds a `psycopg` or `sqlite3` exception."""

    def test_a_constraint_violation_arrives_translated(self, tmp_path):
        from src.workspace.store import WorkspaceStore

        store = WorkspaceStore(tmp_path / "w.db")
        with pytest.raises(DatabaseFailure) as refusal:
            with store._conn() as conn:
                conn.execute(
                    "INSERT INTO plan_run (owner, run_id, plan_id, ran_at, "
                    "result, comparison) VALUES (?,?,?,?,?,?)",
                    ("alice", "r-1", "no-such-plan", "t", "{}", "{}"))
        assert refusal.value.code is PublicCode.CONSTRAINT_CONFLICT
        assert refusal.value.reason is InternalReason.MISSING_PARENT

    def test_the_operation_names_the_table_not_the_statement(self, tmp_path):
        """A full statement in the operator log would duplicate what
        `__cause__` holds and put user values in a second place to redact."""
        from src.workspace.store import WorkspaceStore

        store = WorkspaceStore(tmp_path / "w.db")
        with pytest.raises(DatabaseFailure) as refusal:
            with store._conn() as conn:
                conn.execute(
                    "INSERT INTO plan_run (owner, run_id, plan_id, ran_at, "
                    "result, comparison) VALUES (?,?,?,?,?,?)",
                    ("alice", "r-1", "secret-plan-name", "t", "{}", "{}"))
        assert refusal.value.operation == "INSERT plan_run"
        assert "secret-plan-name" not in refusal.value.operation

    def test_the_original_is_still_reachable(self, tmp_path):
        import sqlite3

        from src.workspace.store import WorkspaceStore

        store = WorkspaceStore(tmp_path / "w.db")
        with pytest.raises(DatabaseFailure) as refusal:
            with store._conn() as conn:
                conn.execute(
                    "INSERT INTO plan_run (owner, run_id, plan_id, ran_at, "
                    "result, comparison) VALUES (?,?,?,?,?,?)",
                    ("alice", "r-1", "no-such-plan", "t", "{}", "{}"))
        assert isinstance(refusal.value.__cause__, sqlite3.Error)


class TestARouteNeverLeaks:
    """A minimal application over the real handler, so the assertions are
    about the boundary rather than about one route's own care."""

    @pytest.fixture
    def client(self, tmp_path):
        from src.web.failure import install

        app = FastAPI()
        install(app)

        @app.get("/boom")
        def boom():
            from src.workspace.store import WorkspaceStore

            store = WorkspaceStore(tmp_path / "w.db")
            with store._conn() as conn:
                conn.execute(
                    "INSERT INTO plan_run (owner, run_id, plan_id, ran_at, "
                    "result, comparison) VALUES (?,?,?,?,?,?)",
                    ("alice", "r-1", "no-such-plan", "t", "{}", "{}"))

        @app.get("/unhandled")
        def unhandled():
            raise RuntimeError(
                "an unanticipated failure naming /srv/quantify/secrets.yaml")

        return TestClient(app, raise_server_exceptions=False)

    def test_the_status_is_a_conflict(self, client):
        assert client.get("/boom").status_code == 409

    def test_the_body_is_the_envelope_and_nothing_else(self, client):
        body = client.get("/boom").json()
        assert set(body) == {"code", "message", "retryable", "request_id"}
        assert body["code"] == "CONSTRAINT_CONFLICT"
        assert body["retryable"] is True

    @pytest.mark.parametrize("token", FORBIDDEN)
    def test_no_forbidden_token_appears(self, client, token):
        response = client.get("/boom")
        rendered = (response.text + str(dict(response.headers))).lower()
        assert token.lower() not in rendered, token

    def test_an_unhandled_failure_leaks_nothing_either(self, client):
        response = client.get("/unhandled")
        assert response.status_code == 500
        assert "secrets.yaml" not in response.text
        assert "RuntimeError" not in response.text
        assert response.json()["request_id"]

    def test_the_request_id_is_echoed_in_a_header(self, client):
        response = client.get("/boom")
        assert response.headers["X-Request-ID"] == response.json()["request_id"]

    def test_a_supplied_request_id_is_used(self, client):
        response = client.get("/boom", headers={"X-Request-ID": "req-abc123"})
        assert response.json()["request_id"] == "req-abc123"

    def test_a_hostile_request_id_is_not_echoed_raw(self, client):
        """The client controls this value and it reaches the operator log."""
        response = client.get(
            "/boom", headers={"X-Request-ID": "<script>alert(1)</script>"})
        assert "<script>" not in response.text

    def test_the_private_record_reaches_the_log(self, client, operator_log):
        client.get("/boom")
        logged = " ".join(operator_log)
        assert "MISSING_PARENT" in logged
        assert "23503" in logged
        assert "INSERT plan_run" in logged
        assert "driver_detail" in logged, (
            "the operator log lost the driver channel; sanitising the public "
            "one must not blind it")

    def test_both_channels_share_the_correlation_id(self, client, operator_log):
        response = client.get("/boom", headers={"X-Request-ID": "req-xyz"})
        logged = " ".join(operator_log)
        assert "req-xyz" in logged
        assert response.json()["request_id"] == "req-xyz"


@pytest.mark.skipif(not POSTGRES_URL,
                    reason="the anchor case needs the driver text that only a "
                           "real PostgreSQL violation produces")
class TestTheAnchorCaseOnRealPostgreSQL:
    """The exact `23503` that opened this gate, through a live request.

    A mocked exception proves the handler formats a fake correctly. It cannot
    prove the route does not leak the driver's real text, because the mock
    never contains it — the forbidden tokens are only meaningful when they
    genuinely exist in the call.
    """

    @pytest.fixture
    def store(self):
        from sqlalchemy import text

        from src.db import migrate
        from src.db.engine import Database
        from src.workspace.store import WorkspaceStore

        database = Database(POSTGRES_URL)
        engine = database.sqlalchemy_engine()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()
        migrate.upgrade(database)
        return WorkspaceStore(POSTGRES_URL)

    @pytest.fixture
    def cited(self, store, monkeypatch):
        """A run citing a delivery, so deleting the delivery is a real 23503."""
        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        from src.market_data.access import resolve

        from tests.test_producer_inventory import TestInstanceCompleteness

        scenario = TestInstanceCompleteness().scenario()
        store.save_plan(plan_id="p-1", owner="alice", scenario=scenario,
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        access = resolve(context="a run", accessed_at="2026-01-01T00:00:00Z",
                         run_id="run-1", request_id="req-1")
        store.record_access_event(access.access_event, owner="alice")
        store.record_run(
            run_id="run-1", plan_id="p-1", ran_at="2026-01-01T00:00:00Z",
            owner="alice",
            result={"modelling_scope": {"excludes": []},
                    "market_data": access.provenance.to_json()},
            comparison={}, access_event_id=access.access_event_id)
        return store, access

    @pytest.fixture
    def client(self, cited):
        from src.web.failure import install

        store, access = cited
        app = FastAPI()
        install(app)

        @app.post("/delete-evidence")
        def delete_evidence():
            with store._conn() as conn:
                conn.execute(
                    "DELETE FROM market_data_access_event "
                    "WHERE access_event_id = ? AND owner = ?",
                    (access.access_event_id, "alice"))

        return TestClient(app, raise_server_exceptions=False), access

    def test_the_driver_text_really_does_contain_the_secrets(self, cited):
        """The premise. Without this the forbidden-token assertions below
        would pass against an exception that never held anything."""
        store, access = cited
        with pytest.raises(DatabaseFailure) as refusal:
            with store._conn() as conn:
                conn.execute(
                    "DELETE FROM market_data_access_event "
                    "WHERE access_event_id = ? AND owner = ?",
                    (access.access_event_id, "alice"))
        detail = refusal.value.private()["driver_detail"]
        assert "fk_plan_run_access_event" in detail
        assert "plan_run" in detail
        assert "alice" in detail
        assert access.access_event_id in detail

    def test_the_response_is_a_safe_conflict(self, client):
        http, _ = client
        response = http.post("/delete-evidence")
        assert response.status_code == 409
        assert response.json()["code"] == "CONSTRAINT_CONFLICT"

    @pytest.mark.parametrize("token", FORBIDDEN)
    def test_no_forbidden_token_survives(self, client, token):
        http, _ = client
        response = http.post("/delete-evidence")
        rendered = (response.text + str(dict(response.headers))).lower()
        assert token.lower() not in rendered, token

    def test_the_tenant_identifier_does_not_appear(self, client):
        http, access = client
        response = http.post("/delete-evidence")
        assert "alice" not in response.text
        assert access.access_event_id not in response.text

    def test_the_operator_keeps_the_whole_message(self, client, operator_log):
        http, access = client
        http.post("/delete-evidence")
        logged = " ".join(operator_log)
        assert "23503" in logged
        assert "fk_plan_run_access_event" in logged
        assert access.access_event_id in logged


class TestEveryBoundaryIsAccountedFor:
    """Enumerated from the syntax tree, not from a list.

    The recurring defect is fixing the instance rather than the class — five
    tenant-key tables found one at a time, two routers with the same ungated
    read. A hand-written inventory of boundaries would have exactly the failure
    mode `test_single_resolution` already demonstrated: it would agree with
    itself and miss the one nobody thought of.
    """

    def handlers(self):
        found = []
        for path in sorted(Path("src").rglob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:                                  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    rendered = ast.unparse(decorator)
                    if any(f".{verb}(" in rendered for verb in
                           ("get", "post", "put", "delete", "patch")) and \
                       ("router" in rendered or "app" in rendered):
                        found.append((str(path), node.name))
        return found

    def test_the_scan_finds_the_handlers(self):
        """A scan finding nothing would pass every assertion below."""
        assert len(self.handlers()) > 40

    #: Modules permitted to name a driver, and why. None of them *catches* a
    #: driver exception — the distinction the first version of this test
    #: missed by grepping for the module name and flagging a connect call, a
    #: URL scheme and a JSON type adapter.
    DRIVER_USERS = {
        "src/deploy/preflight.py":
            "opens a connection to decide whether the database is reachable, "
            "and already reports DATABASE_UNAVAILABLE rather than the error",
        "src/db/migrate.py": "builds the `postgresql+psycopg://` URL scheme",
        "src/db/types.py": "adapts JSON values to `psycopg.types.json.Jsonb`",
        "src/db/engine.py": "the translation boundary itself",
        "src/db/errors.py": "the taxonomy, which names no driver at all",
        # Found by this inventory rather than by anyone remembering. Both are
        # reachable from public routes and neither goes through
        # `db.engine`, so a failure in them is *not* classified — it reaches
        # the catch-all handler, which is safe but says only INTERNAL_ERROR.
        # Both now translate at their own `_conn`, so they produce the same
        # public categories as the engine rather than relying on the catch-all
        # — which is a final barrier, not a semantic boundary.
        "src/ledger.py":
            "opens SQLite directly; predates the engine and is not yet ported, "
            "so its failures are caught by the application handler rather than "
            "classified",
        "src/telemetry/trace_store.py":
            "opens SQLite directly for trace retention; same position as the "
            "ledger, and deliberately independent of the workspace database",
    }

    def test_no_module_catches_a_driver_exception(self):
        """One boundary translates. A module that catches its own database
        exception decides, alone, what a caller learns.

        Looks for `except <driver>.<Something>` in the syntax tree rather than
        for the driver's name in the text — the first version flagged three
        modules that merely connect, build a URL or adapt a type, which is a
        check that would have been silenced by widening rather than by fixing.
        """
        offenders = []
        for path in sorted(Path("src").rglob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:                                  # pragma: no cover
                continue
            if str(path) in TRANSLATION_POINTS:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler) or node.type is None:
                    continue
                caught = ast.unparse(node.type)
                if "psycopg" in caught or "sqlite3" in caught:
                    offenders.append((str(path), node.lineno, caught))
        assert offenders == [], (
            f"these catch a driver exception outside the translation layer: "
            f"{offenders}")

    def test_every_driver_reference_is_declared(self):
        """Naming a driver is not catching one, and is still worth knowing
        about: each is a place the abstraction is deliberately pierced."""
        referencing = set()
        for path in sorted(Path("src").rglob("*.py")):
            body = path.read_text()
            if "psycopg" in body or "import sqlite3" in body:
                referencing.add(str(path))
        undeclared = referencing - set(self.DRIVER_USERS)
        assert undeclared == set(), (
            f"these name a database driver without a declared reason: "
            f"{sorted(undeclared)}")

    def test_each_driver_declaration_records_why(self):
        for module, reason in self.DRIVER_USERS.items():
            assert Path(module).exists(), module
            assert len(reason.strip()) > 30, module

    def test_the_application_installs_the_handler(self):
        import src.api as api

        assert any("DatabaseFailure" in str(key)
                   for key in api.app.exception_handlers), (
            "no application-level translation; every handler would be deciding "
            "for itself what a caller learns")

    def test_the_handler_covers_unanticipated_failures_too(self):
        import src.api as api

        assert Exception in api.app.exception_handlers

    def test_no_route_builds_a_detail_from_a_database_failure(self):
        """`HTTPException(detail=str(exc))` is the shape that leaks. The
        failure's own `str` is the fixed message, so this is belt and braces —
        and it is the belt that has failed before."""
        offenders = []
        for path in sorted(Path("src").rglob("routes.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if getattr(node.func, "id", "") != "HTTPException":
                    continue
                for keyword in node.keywords:
                    rendered = ast.unparse(keyword.value)
                    if "DatabaseFailure" in rendered:
                        offenders.append((str(path), node.lineno))
        assert offenders == []


class TestTheStoreTranslatesItsOwnRefusals:
    """`NotSaveable` is an application refusal, not a database failure, and its
    messages are written here rather than by a driver — so they are checked for
    the same disclosures."""

    def test_a_refusal_names_no_other_tenant(self, tmp_path):
        from src.market_data.access import resolve
        from src.workspace.store import NotSaveable, WorkspaceStore

        os.environ["PILOT_DATA_POLICY"] = "SYNTHETIC_ONLY"
        store = WorkspaceStore(tmp_path / "w.db")
        from tests.test_producer_inventory import TestInstanceCompleteness

        scenario = TestInstanceCompleteness().scenario()
        store.save_plan(plan_id="p-1", owner="mallory", scenario=scenario,
                        stated_text="x", saved_at="t")
        access = resolve(context="a run", accessed_at="t")
        store.record_access_event(access.access_event, owner="mallory")

        with pytest.raises(NotSaveable) as refusal:
            store.record_run(
                run_id="r-1", plan_id="p-1", ran_at="t", owner="alice",
                result={"modelling_scope": {"excludes": []},
                        "market_data": access.provenance.to_json()},
                comparison={}, access_event_id=access.access_event_id)
        assert "mallory" not in str(refusal.value), (
            "the refusal named the tenant who owns the referenced record, "
            "which tells the requester that tenant exists")


class TestAnUnportedStoreStillCannotLeak:
    """`ledger.py` and `trace_store.py` open SQLite directly, so their failures
    are not classified. They must still be unable to reach a caller.

    This is the difference between "we translate database errors" and "no
    database error can escape". The first is a policy every unported module is
    outside of; the second is a property of the boundary, and it is what the
    catch-all handler exists for.
    """

    @pytest.fixture
    def client(self):
        import sqlite3

        from src.web.failure import install

        app = FastAPI()
        install(app)

        @app.get("/unported")
        def unported():
            raise sqlite3.OperationalError(
                "no such table: methodology in /srv/quantify/data/ledger.db")

        return TestClient(app, raise_server_exceptions=False)

    def test_it_becomes_a_safe_internal_error(self, client):
        response = client.get("/unported")
        assert response.status_code == 500
        assert response.json()["code"] == "INTERNAL_ERROR"

    def test_the_file_path_does_not_escape(self, client):
        body = client.get("/unported").text
        assert "/srv/quantify" not in body
        assert "ledger.db" not in body
        assert "no such table" not in body
        assert "OperationalError" not in body

    def test_it_still_correlates(self, client, operator_log):
        response = client.get("/unported")
        logged = " ".join(operator_log)
        assert response.json()["request_id"] in logged
        assert "OperationalError" in logged, (
            "the operator lost the exception type for a failure the taxonomy "
            "does not cover, which is the one they most need it for")


class TestTheOperatorChannelSurvivesAMigration:
    """A migration used to switch the operator log off.

    `migrations/env.py` calls `logging.config.fileConfig`, whose
    `disable_existing_loggers` defaults to True. That disables every logger
    already configured in the process — including `uvicorn.error`, where the
    private half of every database error is written. Migrations run in-process
    at startup, so a deployment that had migrated served perfectly sanitised
    public errors and recorded nothing about any of them.

    The worst possible shape for this failure: the public channel looked
    correct, so every response-body assertion passed, and the only evidence
    that anything was wrong was an absence. It was found by asserting on the
    private channel — and only in a full-file run, because a single test never
    ran a migration first.
    """

    def test_configuring_alembic_logging_leaves_the_logger_enabled(self):
        from logging.config import fileConfig

        logger = logging.getLogger("uvicorn.error")
        logger.disabled = False
        fileConfig("alembic.ini", disable_existing_loggers=False)
        assert not logger.disabled

    def test_the_migration_environment_says_so_explicitly(self):
        """`fileConfig(path)` and `fileConfig(path, disable_existing_loggers=
        False)` differ by a default nobody sees at the call site, which is why
        this is asserted rather than left to review."""
        body = Path("migrations/env.py").read_text()
        assert "disable_existing_loggers=False" in body

    def test_a_failure_after_a_migration_still_reaches_the_operator(
            self, operator_log, tmp_path):
        """End to end: migrate, then fail, then look at the private channel."""
        from logging.config import fileConfig

        from src.web.failure import install

        fileConfig("alembic.ini", disable_existing_loggers=False)

        app = FastAPI()
        install(app)

        @app.get("/after-migration")
        def after_migration():
            from src.workspace.store import WorkspaceStore

            store = WorkspaceStore(tmp_path / "w.db")
            with store._conn() as conn:
                conn.execute(
                    "INSERT INTO plan_run (owner, run_id, plan_id, ran_at, "
                    "result, comparison) VALUES (?,?,?,?,?,?)",
                    ("alice", "r-1", "no-such-plan", "t", "{}", "{}"))

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/after-migration")

        assert response.status_code == 409
        assert operator_log, (
            "nothing reached the operator channel after a migration "
            "configured logging; the private half of the taxonomy is dead")
        assert "MISSING_PARENT" in " ".join(operator_log)


@pytest.mark.skipif(not POSTGRES_URL,
                    reason="the defect needs a real migration and a real "
                           "constraint violation, in that order")
class TestTheExactDefect:
    """Migration, then a real `23503`, then both channels — in one sequence.

    Not an approximation of the defect: the order is the defect. Alembic
    configures logging while migrating, `disable_existing_loggers` defaulted to
    True, and `uvicorn.error` — owned by a different component entirely — was
    switched off. Every subsequent database failure produced a correct public
    response and no diagnostic at all.

    The six assertions run against one request because each was individually
    satisfiable while the whole was broken. A suite that checked the payload in
    one test and the log in another, with no migration between them, certified
    this system as correct.
    """

    @pytest.fixture
    def migrated_then_failing(self, monkeypatch):
        """Migrate first — as a deployment does — then build the request."""
        from sqlalchemy import text

        from src.db import migrate
        from src.db.engine import Database
        from src.market_data.access import resolve
        from src.web.failure import install
        from src.workspace.store import WorkspaceStore

        from tests.test_producer_inventory import TestInstanceCompleteness

        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")

        database = Database(POSTGRES_URL)
        engine = database.sqlalchemy_engine()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()
        # The step that used to disable the logger. Everything after this line
        # is what a deployed process does on its first request.
        migrate.upgrade(database)

        store = WorkspaceStore(POSTGRES_URL)
        scenario = TestInstanceCompleteness().scenario()
        store.save_plan(plan_id="p-1", owner="alice", scenario=scenario,
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        access = resolve(context="a run", accessed_at="2026-01-01T00:00:00Z",
                         run_id="run-1", request_id="req-1")
        store.record_access_event(access.access_event, owner="alice")
        store.record_run(
            run_id="run-1", plan_id="p-1", ran_at="2026-01-01T00:00:00Z",
            owner="alice",
            result={"modelling_scope": {"excludes": []},
                    "market_data": access.provenance.to_json()},
            comparison={}, access_event_id=access.access_event_id)

        app = FastAPI()
        install(app)

        @app.post("/delete-evidence")
        def delete_evidence():
            with store._conn() as conn:
                conn.execute(
                    "DELETE FROM market_data_access_event "
                    "WHERE access_event_id = ? AND owner = ?",
                    (access.access_event_id, "alice"))

        return TestClient(app, raise_server_exceptions=False), access

    def test_the_whole_sequence(self, migrated_then_failing, operator_log):
        client, access = migrated_then_failing

        # 1. The migration has run and the operator logger is still alive.
        assert logging.getLogger("uvicorn.error").disabled is False, (
            "a migration disabled the operator logger; every database failure "
            "from here on is diagnosed by nothing")

        # 2. A real foreign-key violation, through a real request.
        response = client.post("/delete-evidence",
                               headers={"X-Request-ID": "req-anchor"})

        # 3. The public channel: bounded, semantic, correlated.
        assert response.status_code == 409
        body = response.json()
        assert body["code"] == "CONSTRAINT_CONFLICT"
        assert body["retryable"] is True
        assert body["request_id"] == "req-anchor"
        assert set(body) == {"code", "message", "retryable", "request_id"}

        # 4. Nothing from the deployment survives into it.
        rendered = (response.text + str(dict(response.headers))).lower()
        for token in FORBIDDEN:
            assert token.lower() not in rendered, token
        assert "alice" not in response.text
        assert access.access_event_id not in response.text

        # 5. The private channel: everything, and the same correlation id.
        logged = " ".join(operator_log)
        assert logged, (
            "the operator channel produced nothing after a migration — the "
            "public response above would have certified this as correct")
        assert "req-anchor" in logged
        assert "MISSING_PARENT" in logged
        assert "23503" in logged
        assert "fk_plan_run_access_event" in logged
        assert access.access_event_id in logged

        # 6. The original is still chained, not discarded.
        from src.db.errors import DatabaseFailure
        from src.workspace.store import WorkspaceStore

        store = WorkspaceStore(POSTGRES_URL)
        with pytest.raises(DatabaseFailure) as refusal:
            with store._conn() as conn:
                conn.execute(
                    "DELETE FROM market_data_access_event "
                    "WHERE access_event_id = ? AND owner = ?",
                    (access.access_event_id, "alice"))
        assert refusal.value.__cause__ is not None
        assert "fk_plan_run_access_event" in str(refusal.value.__cause__)


class TestTheUnportedStoresTranslateToo:
    """The catch-all is a final barrier, not the semantic boundary.

    Before this, a ledger failure reached a caller as an unclassified 500 with
    no retry disposition — safe, and useless to both the caller and the
    operator. `ledger.py` and `trace_store.py` open SQLite directly, so they
    translate at their own connection until they are ported.
    """

    def a_ledger(self, tmp_path):
        from src.ledger import Ledger

        return Ledger(tmp_path / "ledger.db")

    def test_a_ledger_failure_is_classified(self, tmp_path):
        from src.db.errors import DatabaseFailure, PublicCode

        ledger = self.a_ledger(tmp_path)
        with pytest.raises(DatabaseFailure) as refusal:
            with ledger._conn() as conn:
                conn.execute("INSERT INTO methodology (version_id) VALUES (?)",
                             ("only-one-column-of-many",))
        assert refusal.value.code in set(PublicCode)
        assert refusal.value.__cause__ is not None

    def test_the_file_path_stays_in_the_private_channel(self, tmp_path):
        from src.db.errors import DatabaseFailure

        ledger = self.a_ledger(tmp_path)
        with pytest.raises(DatabaseFailure) as refusal:
            with ledger._conn() as conn:
                conn.execute("SELECT * FROM no_such_table")
        assert str(tmp_path) not in str(refusal.value)
        assert refusal.value.private()["driver_detail"]

    def test_the_trace_store_translates_as_well(self, tmp_path):
        from src.db.errors import DatabaseFailure
        from src.telemetry.trace_store import TraceStore

        store = TraceStore(tmp_path / "traces.db")
        with pytest.raises(DatabaseFailure):
            with store._conn() as conn:
                conn.execute("SELECT * FROM no_such_table")

    def test_a_telemetry_failure_names_its_own_operation(self, tmp_path):
        """So an operator can tell a telemetry failure from a workspace one
        without reading the path out of the driver text."""
        from src.db.errors import DatabaseFailure
        from src.telemetry.trace_store import TraceStore

        store = TraceStore(tmp_path / "traces.db")
        with pytest.raises(DatabaseFailure) as refusal:
            with store._conn() as conn:
                conn.execute("SELECT * FROM no_such_table")
        assert refusal.value.operation == "trace_store"


class TestGateNineClosureCriteria:
    """The eleven conditions, each checked rather than claimed.

    A closure list that is only prose is a list that stays true by nobody
    rereading it. Each of these has a lane above; this asserts the lane exists
    and covers the condition, so removing one fails here rather than quietly
    reducing what "closed" meant.
    """

    def test_every_public_route_uses_one_failure_handler(self):
        import src.api as api
        from src.db.errors import DatabaseFailure

        assert DatabaseFailure in api.app.exception_handlers
        assert Exception in api.app.exception_handlers

    def test_raw_driver_exceptions_cannot_escape_the_engine(self, tmp_path):
        from src.db.errors import DatabaseFailure
        from src.workspace.store import WorkspaceStore

        store = WorkspaceStore(tmp_path / "w.db")
        with pytest.raises(DatabaseFailure):
            with store._conn() as conn:
                conn.execute("SELECT * FROM no_such_table")

    def test_six_public_categories_have_fixed_safe_messages(self):
        assert len(list(PublicCode)) == 6
        messages = set()
        for code in PublicCode:
            failure = DatabaseFailure(
                Classification(code, InternalReason.UNCLASSIFIED, Retry.NEVER))
            message = failure.public_message
            assert message.strip()
            # Fixed, not composed: no placeholder survived into it.
            assert "{" not in message and "%" not in message
            messages.add(message)
        assert len(messages) == 6, "two categories share a message"

    def test_internal_reasons_never_reach_a_public_payload(self):
        for reason in InternalReason:
            failure = DatabaseFailure(Classification(
                PublicCode.CONSTRAINT_CONFLICT, reason, Retry.NEVER))
            assert reason.value not in str(failure.public()), reason

    def test_retryability_is_explicit_on_every_failure(self):
        for code in PublicCode:
            for retry in Retry:
                failure = DatabaseFailure(Classification(
                    code, InternalReason.UNCLASSIFIED, retry))
                assert isinstance(failure.public()["retryable"], bool)

    def test_causes_remain_chained(self):
        original = RuntimeError("diagnostic")
        original.sqlstate = "23503"
        assert translate(original).__cause__ is original

    def test_migrations_cannot_disable_operator_logging(self):
        assert "disable_existing_loggers=False" in \
            Path("migrations/env.py").read_text()

    def test_both_channels_share_a_correlation_id(self):
        failure = DatabaseFailure(
            Classification(PublicCode.CONSTRAINT_CONFLICT,
                           InternalReason.MISSING_PARENT, Retry.AFTER_REREAD),
            request_id="req-1")
        assert failure.public()["request_id"] == failure.private()["request_id"]

    def test_a_real_23503_is_covered_end_to_end(self):
        """The anchor exists and runs against PostgreSQL rather than a mock."""
        body = Path("tests/test_error_surface.py").read_text()
        assert "class TestTheExactDefect" in body
        assert "migrate.upgrade" in body
        assert "QUANTIFY_TEST_POSTGRES_URL" in body

    def test_every_external_boundary_is_inventoried(self):
        assert len(TestEveryBoundaryIsAccountedFor().handlers()) > 40

    def test_the_remaining_direct_sqlite_stores_are_declared_and_contained(self):
        declared = TestEveryBoundaryIsAccountedFor.DRIVER_USERS
        for module in ("src/ledger.py", "src/telemetry/trace_store.py"):
            assert module in declared, module
            # Contained: each translates at its own connection rather than
            # relying on the catch-all handler for meaning.
            assert "from .db.errors import translate" in Path(module).read_text() \
                or "from ..db.errors import translate" in Path(module).read_text(), \
                module


class TestTranslationDoesNotSwallowDomainRefusals:
    """A translating context manager sees whatever the `with` body raised.

    `Ledger._conn` is a `@contextmanager`, so an exception from inside the
    block arrives at the `yield`. Catching `Exception` there converted the
    ledger's own `ValueError("... already published ... immutable")` — a
    refusal the caller must see and act on — into an opaque
    `DATABASE_INTERNAL_FAILURE`. A meaningful 422 became a 500, and the reason
    it happened was invisible: the wrapper looked like the engine's, which
    wraps a single `execute` call rather than a yielded body.

    Caught by the ledger's existing tests rather than by the new ones, which is
    the argument for running the whole suite rather than the slice being
    worked on.
    """

    def test_a_domain_refusal_passes_through_the_ledger(self, tmp_path):
        from src.db.errors import DatabaseFailure
        from src.ledger import Ledger

        ledger = Ledger(tmp_path / "ledger.db")
        with pytest.raises(ValueError) as refusal:
            with ledger._conn():
                raise ValueError("a domain refusal the caller must see")
        assert not isinstance(refusal.value, DatabaseFailure)

    def test_a_domain_refusal_passes_through_the_trace_store(self, tmp_path):
        from src.db.errors import DatabaseFailure
        from src.telemetry.trace_store import TraceStore

        store = TraceStore(tmp_path / "traces.db")
        with pytest.raises(ValueError) as refusal:
            with store._conn():
                raise ValueError("a domain refusal the caller must see")
        assert not isinstance(refusal.value, DatabaseFailure)

    def test_a_real_database_error_is_still_translated(self, tmp_path):
        """The other half: narrowing to `sqlite3.Error` must not have turned
        the wrapper off."""
        from src.db.errors import DatabaseFailure
        from src.ledger import Ledger

        ledger = Ledger(tmp_path / "ledger.db")
        with pytest.raises(DatabaseFailure):
            with ledger._conn() as conn:
                conn.execute("SELECT * FROM no_such_table")

    def test_the_engine_wraps_a_call_not_a_body(self):
        """Why the engine never had this defect: it wraps one `execute`, so
        there is no body whose exceptions it could capture."""
        import inspect

        from src.db.engine import Connection

        source = inspect.getsource(Connection.execute)
        assert "yield" not in source


class TestConsumersOfRawErrorsStillWork:
    """Adding a translation layer beneath code that read raw errors is a
    general hazard, not a one-off.

    `apply.py` asks `is_conflict(exc)` and converts contention into a domain
    `ProposalConflict` — a refusal that names the proposal and tells the caller
    to re-read. Once the engine classified first, `is_conflict` was handed a
    `DatabaseFailure` with no SQLSTATE attribute, answered no, and the apply
    path silently stopped producing its domain refusal. The layer was correct
    and every consumer above it was looking for the wrong shape.

    Nothing in the new taxonomy tests could see this: the failure it produced
    was a perfectly safe `DATABASE_CONTENTION`. It was caught by the existing
    concurrency lane, which asserts on the *domain* outcome.
    """

    def test_is_conflict_recognises_a_translated_contention(self):
        from src.db.errors import is_conflict

        deadlock = RuntimeError("deadlock detected")
        deadlock.sqlstate = "40P01"
        assert is_conflict(translate(deadlock))

    def test_is_conflict_still_recognises_a_raw_one(self):
        from src.db.errors import is_conflict

        deadlock = RuntimeError("deadlock detected")
        deadlock.sqlstate = "40P01"
        assert is_conflict(deadlock)

    def test_a_missing_parent_is_not_contention(self):
        """The narrowness is the point: `accept` must not report "another
        request was changing the same records" for a plan that is not there."""
        from src.db.errors import is_conflict

        missing = RuntimeError("foreign key violation")
        missing.sqlstate = "23503"
        assert not is_conflict(missing)
        assert not is_conflict(translate(missing))

    def test_sqlite_busy_is_still_recognised_through_translation(self):
        from src.db.errors import is_conflict

        busy = RuntimeError("database is locked")
        assert is_conflict(busy)
        assert is_conflict(translate(busy))


class TestEveryCategoryHasAProducer:
    """A declared category nothing produces is a declaration with no reachable
    consumer — the defect this codebase keeps finding, reintroduced in its own
    taxonomy.

    `STALE_TRANSITION` and `TRANSITION_INTEGRITY_FAILURE` were declared,
    messaged, status-mapped and tested for six commits before anything raised
    them. The unit tests passed because they constructed the failures directly:
    self-authored evidence again, in a shape that looked like coverage.
    """

    def producers(self):
        """Which categories `src/` actually emits, from the syntax tree.

        Derived from source rather than by calling helpers. The first version
        of this check called `errors.stale_transition(rows=2)` to "prove"
        `TRANSITION_INTEGRITY_FAILURE` had a producer — and that helper had no
        callers anywhere in `src/`. The check was importing the vocabulary and
        exercising it itself, which is the same self-authored evidence it was
        written to catch, one level up. `stale_transition` is now deleted.

        Two shapes count as producing a category:

            the SQLSTATE table maps some code to it
            a module names it, outside `db/errors.py` itself
        """
        from src.db.errors import _BY_SQLSTATE

        produced = {mapped[0] for mapped in _BY_SQLSTATE.values()}
        produced.add(PublicCode.DATABASE_INTERNAL_FAILURE)   # the fallback

        for path in sorted(Path("src").rglob("*.py")):
            if str(path) == "src/db/errors.py":
                continue                     # where they are declared
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:                              # pragma: no cover
                continue
            # Attribute *nodes*, not text. A text scan counted the docstring
            # sentence "Carries `PublicCode.TRANSITION_INTEGRITY_FAILURE`" as a
            # producer, so deleting the only real one left the check green —
            # prose satisfying a structural test, which is the failure this
            # codebase has now hit twelve times.
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                if getattr(node.value, "id", "") != "PublicCode":
                    continue
                try:
                    produced.add(PublicCode[node.attr])
                except KeyError:                             # pragma: no cover
                    pass
        return produced

    def test_every_public_code_is_produced_somewhere(self):
        missing = set(PublicCode) - self.producers()
        assert missing == set(), (
            f"these categories are declared and nothing in src/ produces "
            f"them: {sorted(one.value for one in missing)}")

    def test_the_scan_is_not_vacuous(self):
        """A scan that found everything would pass the check above."""
        from src.db.errors import _BY_SQLSTATE

        assert PublicCode.STALE_TRANSITION not in {
            mapped[0] for mapped in _BY_SQLSTATE.values()}, (
            "STALE_TRANSITION is reachable from SQLSTATE alone, so the source "
            "scan proves nothing about it")

    def test_prose_does_not_count_as_a_producer(self):
        """A docstring naming a category must not satisfy the check.

        Constructed rather than asserted about the current source: the scan is
        run against a module whose only mention is in a docstring, and must
        find nothing.
        """
        import tempfile

        source = '''
"""This module mentions PublicCode.STALE_TRANSITION in prose only."""
# and in a comment: PublicCode.STALE_TRANSITION
'''
        tree = ast.parse(source)
        found = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute)
                 and getattr(node.value, "id", "") == "PublicCode"]
        assert found == [], "prose was parsed as a reference"

    def test_the_unclassified_fallback_is_the_only_declared_producer(self):
        """`DATABASE_INTERNAL_FAILURE` is produced by `classify` falling
        through, which no source scan can see — so it is added by hand, and
        that exception is asserted rather than assumed."""
        assert classify(_with_sqlstate("XX000")).code is \
            PublicCode.DATABASE_INTERNAL_FAILURE

    def test_the_transition_refusals_carry_a_category(self):
        from src.workspace.apply import ProposalConflict, TransitionIntegrityError

        assert ProposalConflict.public_code is PublicCode.STALE_TRANSITION
        assert ProposalConflict.retry_disposition is Retry.AFTER_REREAD
        assert TransitionIntegrityError.public_code is \
            PublicCode.TRANSITION_INTEGRITY_FAILURE
        assert TransitionIntegrityError.retry_disposition is Retry.NEVER

    def test_the_application_registers_handlers_for_them(self):
        import src.api as api
        from src.workspace.apply import ProposalConflict, TransitionIntegrityError

        for refusal in (ProposalConflict, TransitionIntegrityError):
            assert refusal in api.app.exception_handlers, refusal.__name__

    def test_a_stale_transition_becomes_a_conflict_response(self):
        from src.web.failure import install
        from src.workspace.apply import ProposalConflict

        app = FastAPI()
        install(app)

        @app.post("/stale")
        def stale():
            raise ProposalConflict(
                "proposal wp-1 was resolved by another request while this one "
                "was applying it")

        response = TestClient(app, raise_server_exceptions=False).post("/stale")
        assert response.status_code == 409
        assert response.json()["code"] == "STALE_TRANSITION"
        assert response.json()["retryable"] is True
        # The refusal's own message is safe but is not the envelope's.
        assert "wp-1" not in response.text

    def test_an_integrity_failure_becomes_a_500(self):
        from src.web.failure import install
        from src.workspace.apply import TransitionIntegrityError

        app = FastAPI()
        install(app)

        @app.post("/broken")
        def broken():
            raise TransitionIntegrityError("changed 2 rows where one was required")

        response = TestClient(app, raise_server_exceptions=False).post("/broken")
        assert response.status_code == 500
        assert response.json()["code"] == "TRANSITION_INTEGRITY_FAILURE"
        assert response.json()["retryable"] is False

    def test_the_unreachable_branch_is_gone(self):
        """`_apply` had a `raise ProposalConflict` after a `raise`, which reads
        as a second branch that can never be taken."""
        import inspect

        from src.workspace import apply as apply_module

        source = inspect.getsource(apply_module._apply)
        assert source.count("raise TransitionIntegrityError") == 1
        after = source.split("raise TransitionIntegrityError", 1)[1]
        assert "raise ProposalConflict" not in after


@pytest.mark.skipif(not POSTGRES_URL,
                    reason="the sequence is the subject; it needs the real "
                           "startup path against the real engine")
class TestTheStartupJourney:
    """`create_app()` -> preflight -> migration -> request -> failure.

    The defect was sequence-dependent, so the permanent proof is too. This
    goes through the production entrypoint rather than a hand-built app: the
    thing being checked is that no stage between process start and first
    request reconfigures logging in a way that kills the private channel, and
    a test that skipped the entrypoint would not be looking at those stages.
    """

    @pytest.fixture
    def served(self, monkeypatch, tmp_path):
        from sqlalchemy import text

        from src.db import migrate
        from src.db.engine import Database

        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        monkeypatch.setenv("QUANTIFY_DEPLOYMENT_PROFILE", "local")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", POSTGRES_URL)

        database = Database(POSTGRES_URL)
        engine = database.sqlalchemy_engine()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()
        migrate.upgrade(database)

        import src.api as api

        app = api.create_app()
        return app

    def test_the_operator_logger_survives_the_whole_startup(self, served):
        assert logging.getLogger("uvicorn.error").disabled is False, (
            "some stage of startup disabled the operator logger; every "
            "database failure this process serves is diagnosed by nothing")

    def test_a_request_through_the_served_app_fails_safely(self, served,
                                                            operator_log):
        """A real request to a real route, on a database with no rows."""
        with TestClient(served, raise_server_exceptions=False) as client:
            response = client.get("/workspace/plans/no-such-plan",
                                  headers={"X-Request-ID": "req-journey"})
        # 404 from the route's own check, which is correct and carries no
        # deployment detail. The point of the assertion is what is absent —
        # checked against the shared list rather than a second one written
        # here, so there is one vocabulary of what may never be disclosed.
        for token in FORBIDDEN:
            assert token.lower() not in response.text.lower(), token

    def test_the_private_channel_is_alive_after_startup(self, served,
                                                         operator_log):
        """Provoked directly rather than through a route, so this is about the
        logger rather than about which route can fail."""
        from src.db.errors import DatabaseFailure
        from src.web.failure import LOG

        assert not LOG.disabled
        LOG.error("database failure %s",
                  DatabaseFailure(Classification(
                      PublicCode.CONSTRAINT_CONFLICT,
                      InternalReason.MISSING_PARENT,
                      Retry.AFTER_REREAD)).private())
        assert operator_log, (
            "the operator logger accepted a record and emitted nothing")
        assert "MISSING_PARENT" in " ".join(operator_log)


class TestTheLoggerCannotTakeTheResponseDownWithIt:
    """The private channel failing must not cost the caller their response.

    A logging handler can be unavailable, a formatter can raise, a field can
    be unserializable. None of that is the caller's problem, and a handler
    whose logging failure propagates turns a clean 409 into an unhandled 500.
    """

    @pytest.fixture
    def client_with_a_broken_logger(self, tmp_path):
        from src.web.failure import install

        app = FastAPI()
        install(app)

        @app.get("/boom")
        def boom():
            from src.workspace.store import WorkspaceStore

            store = WorkspaceStore(tmp_path / "w.db")
            with store._conn() as conn:
                conn.execute(
                    "INSERT INTO plan_run (owner, run_id, plan_id, ran_at, "
                    "result, comparison) VALUES (?,?,?,?,?,?)",
                    ("alice", "r-1", "no-such-plan", "t", "{}", "{}"))

        logger = logging.getLogger("uvicorn.error")

        class _Hostile(logging.Handler):
            def emit(self, record):
                raise RuntimeError("the log sink is unavailable")

        hostile = _Hostile()
        logger.addHandler(hostile)
        try:
            yield TestClient(app, raise_server_exceptions=False)
        finally:
            logger.removeHandler(hostile)

    def test_the_caller_still_gets_the_right_answer(
            self, client_with_a_broken_logger):
        response = client_with_a_broken_logger.get("/boom")
        assert response.status_code == 409
        assert response.json()["code"] == "CONSTRAINT_CONFLICT"

    def test_the_response_is_still_free_of_the_logging_failure(
            self, client_with_a_broken_logger):
        response = client_with_a_broken_logger.get("/boom")
        assert "log sink" not in response.text
        assert "RuntimeError" not in response.text

    def test_the_failure_still_reaches_stderr(self, tmp_path, capsys):
        """Falling back rather than falling silent.

        Saying nothing when the logging subsystem is the thing that failed
        would leave the incident with no record at all — the same silence the
        migration defect produced, arrived at from a different direction.
        """
        from src.web.failure import install

        app = FastAPI()
        install(app)

        @app.get("/boom")
        def boom():
            from src.workspace.store import WorkspaceStore

            store = WorkspaceStore(tmp_path / "w.db")
            with store._conn() as conn:
                conn.execute(
                    "INSERT INTO plan_run (owner, run_id, plan_id, ran_at, "
                    "result, comparison) VALUES (?,?,?,?,?,?)",
                    ("alice", "r-1", "no-such-plan", "t", "{}", "{}"))

        logger = logging.getLogger("uvicorn.error")

        class _Hostile(logging.Handler):
            def emit(self, record):
                raise RuntimeError("the log sink is unavailable")

        hostile = _Hostile()
        logger.addHandler(hostile)
        try:
            response = TestClient(app, raise_server_exceptions=False).get("/boom")
        finally:
            logger.removeHandler(hostile)

        assert response.status_code == 409
        emitted = capsys.readouterr().err
        assert "operator-log-unavailable" in emitted
        assert "MISSING_PARENT" in emitted
