"""Structured payloads, stored as JSONB where that exists.

The point of JSONB is not tidiness. It is that operational questions about
stored payloads — which runs cited an unpinned snapshot, which results carry an
unchecked dimension — become queries instead of a script that reads every row
through the application.

**Both dialects must return the same Python object.** SQLite stores text and
PostgreSQL stores JSONB, so a caller that got a string from one and a dict from
the other would have to know which database it was talking to. That is the
divergence the store exists to prevent, so it is asserted directly rather than
assumed from the fact that both "work".
"""
from __future__ import annotations

import json
import os

import pytest

from src.db.engine import Database, Dialect
from src.db.types import Json, adapt, loads
from src.workspace.store import WorkspaceStore

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

PAYLOAD = {"grant_ref": "G-1", "snapshot": "syn-2026-01",
           "dimensions": [{"name": "concentration", "checked": False},
                          {"name": "sequence", "checked": True}],
           "unicode": "café", "nested": {"depth": {"deeper": 3}}}


@pytest.fixture
def postgres_store():
    if not POSTGRES_URL:
        pytest.skip("set QUANTIFY_TEST_POSTGRES_URL to run against PostgreSQL")
    from sqlalchemy import text

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    return WorkspaceStore(POSTGRES_URL)


class TestTheValueDeclaresItsType:
    """The connection cannot know which column a positional parameter binds
    to, so the value says what it is."""

    def test_sqlite_gets_text(self):
        assert adapt(Json({"a": 1}), "sqlite") == '{"a": 1}'

    def test_postgresql_gets_a_jsonb_parameter(self):
        adapted = adapt(Json({"a": 1}), "postgresql")
        assert type(adapted).__name__ == "Jsonb"

    def test_anything_else_is_left_alone(self):
        for value in ("plain", 3, None, 2.5):
            assert adapt(value, "postgresql") is value


class TestReadsAcceptEitherDialect:
    def test_text_is_parsed(self):
        assert loads('{"a": 1}') == {"a": 1}

    def test_an_already_parsed_object_passes_through(self):
        assert loads({"a": 1}) == {"a": 1}

    def test_absent_returns_the_default(self):
        assert loads(None, []) == []
        assert loads("", []) == []

    def test_corrupt_json_is_not_silently_passed_through(self):
        """A column declared JSON holds JSON. Returning the raw string would
        hand a caller something it cannot use and call it a payload."""
        with pytest.raises(json.JSONDecodeError):
            loads("{not json")


class TestRoundTripParity:
    """The same payload in, the same Python object out, on either engine."""

    def _write_and_read(self, store):
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner="alice", worksheet_id="ws-1",
            created_at="2026-01-01T00:00:00Z", proposal=_Proposal(PAYLOAD))
        return store.get_worksheet_proposal("wp-1", "alice")["payload"]

    def test_on_sqlite(self, tmp_path):
        store = WorkspaceStore(tmp_path / "w.db")
        assert self._write_and_read(store) == PAYLOAD

    def test_on_postgresql(self, postgres_store):
        assert self._write_and_read(postgres_store) == PAYLOAD

    def test_both_dialects_agree(self, tmp_path, postgres_store):
        sqlite_value = self._write_and_read(WorkspaceStore(tmp_path / "w.db"))
        assert sqlite_value == self._write_and_read(postgres_store)


class TestJsonbIsActuallyQueryable:
    """The reason for the type, checked rather than assumed."""

    def test_the_columns_are_jsonb_not_text(self, postgres_store):
        conn = postgres_store.db.connect()
        try:
            rows = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'worksheet_proposal' "
                "AND data_type = 'jsonb'").fetchall()
        finally:
            conn.close()
        assert {r["column_name"] for r in rows} == {"payload", "result_runs"}

    def test_a_nested_field_can_be_queried_in_the_database(self, postgres_store):
        """Text storage cannot do this; it is the whole reason for JSONB."""
        self_payload = dict(PAYLOAD)
        postgres_store.save_worksheet_proposal(
            proposal_id="wp-1", owner="alice", worksheet_id="ws-1",
            created_at="2026-01-01T00:00:00Z", proposal=_Proposal(self_payload))

        conn = postgres_store.db.connect()
        try:
            row = conn.execute(
                "SELECT payload->>'snapshot' AS snapshot FROM worksheet_proposal "
                "WHERE payload->'nested'->'depth'->>'deeper' = '3'").fetchone()
        finally:
            conn.close()
        assert row is not None and row["snapshot"] == "syn-2026-01"

    def test_an_array_member_can_be_found(self, postgres_store):
        postgres_store.save_worksheet_proposal(
            proposal_id="wp-1", owner="alice", worksheet_id="ws-1",
            created_at="2026-01-01T00:00:00Z", proposal=_Proposal(PAYLOAD))
        conn = postgres_store.db.connect()
        try:
            row = conn.execute(
                "SELECT count(*) AS n FROM worksheet_proposal "
                "WHERE payload @> '{\"dimensions\": [{\"checked\": false}]}'"
            ).fetchone()
        finally:
            conn.close()
        assert row["n"] == 1


class _Proposal:
    """Minimal stand-in for the proposal artifact the store serializes."""

    source_revision = 1

    def __init__(self, payload):
        self._payload = payload

    def to_json(self):
        return self._payload
