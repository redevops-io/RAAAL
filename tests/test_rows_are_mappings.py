"""Every row reader must work when a row is only a mapping.

`db/engine.py` says it in its own docstring — "rows are mappings in both
dialects" — and then the two dialects differ in a way that hides the
consequence. `sqlite3.Row` supports positional *and* key access; psycopg's
`dict_row` is a plain dict, where `row[0]` is a lookup for a key named `0`.

So `json.loads(row[0])` passed every test and raised `KeyError: 0` in
production. Opening a saved runtime plan returned INTERNAL_ERROR, and the
suite could not see it because the suite runs on SQLite.

These tests run the readers against rows that are *only* mappings, which is
what production hands them. Nothing here needs PostgreSQL: the defect is not
about PostgreSQL, it is about assuming a row is a sequence.
"""
from __future__ import annotations

import json
import os

import pytest


class OnlyAMapping(dict):
    """A row with no positional access, like psycopg's `dict_row`.

    Subclasses `dict` and refuses integer keys explicitly, so a positional read
    fails the way it fails in production rather than silently returning a
    column that happens to be named 0.
    """

    def __getitem__(self, key):
        if isinstance(key, int):
            raise KeyError(key)
        return super().__getitem__(key)


class MappedCursor:
    def __init__(self, inner):
        self._inner = inner

    def fetchone(self):
        row = self._inner.fetchone()
        return None if row is None else OnlyAMapping(dict(row))

    def fetchall(self):
        return [OnlyAMapping(dict(r)) for r in self._inner.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


class MappedConnection:
    """The real database, handing back rows that are only mappings."""

    def __init__(self, inner):
        self._inner = inner

    def execute(self, sql, params=()):
        return MappedCursor(self._inner.execute(sql, params))

    def commit(self):
        return self._inner.commit()

    def close(self):
        return self._inner.close()


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    """A runtime deployment on SQLite, with mapping-only rows."""
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/w.db")
    # Transcripts are kept only where the deployment says so and the
    # participant agrees. Both gates are declared here so the reader under
    # test has a row to read; neither is what this file is about.
    monkeypatch.setenv("QUANTIFY_PILOT_TRANSCRIPTS", "1")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.workspace import (pilot_consent, pilot_events, pilot_session,
                               pilot_store)

    for module in (pilot_store, pilot_consent, pilot_events, pilot_session):
        original = module._connect
        monkeypatch.setattr(
            module, "_connect",
            lambda _original=original: MappedConnection(_original()))
    return {"store": pilot_store, "consent": pilot_consent,
            "events": pilot_events, "session": pilot_session}


TEXT = "I invest $200 into NVDA every month, on the 15th."


def store_a_plan(store, plan_id="plan-1"):
    """A row, written as SQL.

    Deliberately not through `save`: what is under test is *reading* a row,
    and going through the writer would tie this to the shape of a reading
    object that has nothing to do with the defect.
    """
    connection = store._connect()
    try:
        connection.execute(
            "INSERT INTO pilot_plans "
            "(plan_id, owner, created_at, text, artifact) "
            "VALUES (?, ?, ?, ?, ?)",
            (plan_id, store.PILOT_OWNER(), "2026-08-14T00:00:00Z", TEXT,
             json.dumps({"text": TEXT})))
        connection.commit()
    finally:
        connection.close()
    return plan_id


class TestTheRuntimeStore:
    def test_a_saved_plan_can_be_opened(self, workspace):
        """The exact failure: /pilot/plans/<id> returned INTERNAL_ERROR
        because `load` read the artifact positionally."""
        store = workspace["store"]
        plan_id = store_a_plan(store)

        loaded = store.load(plan_id)
        assert loaded is not None, "the plan was stored and could not be read"
        assert loaded["text"] == TEXT

    def test_the_plan_list_can_be_read(self, workspace):
        store = workspace["store"]
        store_a_plan(store)
        every = store.every_plan()
        assert every and every[0]["text"] == TEXT

    def test_an_absent_plan_is_absent_rather_than_an_error(self, workspace):
        assert workspace["store"].load("plan-nothing") is None


class TestTheStudyInstrumentation:
    """These record what a pilot is for. A reader that raises here loses the
    measurement and, through `record`'s exception guard, loses it silently."""

    def test_consent_can_be_read_back(self, workspace):
        consent = workspace["consent"]
        consent.grant("participant-1")
        assert consent.record_of("participant-1") is not None
        assert consent.state_of("participant-1") == consent.GRANTED

    def test_transcripts_can_be_read_back(self, workspace):
        session = workspace["session"]
        # Prose is only kept where somebody agreed to it being kept, so the
        # consent comes first — recording without it writes nothing, which is
        # the behaviour and not the defect under test.
        workspace["consent"].grant("participant-1")
        session.record("participant-1", "invest $500 monthly", 1)
        assert session.transcript("participant-1")
        assert session.last_prompt("participant-1") == "invest $500 monthly"
        assert "participant-1" in session.every_participant()

    def test_events_can_be_counted(self, workspace):
        """`attempts_by` reads a COUNT positionally, which is the same defect
        wearing an aggregate's clothes — and its exception guard would have
        turned it into a permanent zero rather than an error."""
        from src.workspace.owner import current as owner_of

        events = workspace["events"]
        connection = events._connect()
        try:
            connection.execute(
                "INSERT INTO pilot_events "
                "(owner, event_id, at, kind, plan_id, participant, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (owner_of(), "e-1", "2026-08-14T00:00:00Z",
                 events.PLAN_COMPILED, "plan-1", "participant-1",
                 json.dumps({})))
            connection.commit()
        finally:
            connection.close()
        assert events.attempts_by("participant-1") == 1
        assert events.every_event()


class TestTheShapeItself:
    def test_a_positional_read_really_does_fail_on_these_rows(self):
        """Without this the fixture could be handing back ordinary dicts,
        which support `row[0]` when a column is literally named 0 — and these
        tests would prove nothing."""
        row = OnlyAMapping({"artifact": "{}"})
        assert row["artifact"] == "{}"
        with pytest.raises(KeyError):
            row[0]
