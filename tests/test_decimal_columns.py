"""Exact quantities, on both engines, with the same Python contract.

PostgreSQL stores `NUMERIC(38, 12)` and SQLite stores canonical decimal `TEXT`.
The physical types differ on purpose — SQLite has no exact decimal type, and a
NUMERIC column there carries REAL affinity, which would put the binary
approximation straight back. What must not differ is what a caller receives:

    same persisted economic value
        -> same Python Decimal
        -> same serialized canonical value
        -> same reconciliation behaviour

So the cross-dialect tests assert the Decimal, not the column type. Asserting
identical storage would preserve the wrong property.
"""
from __future__ import annotations

import os
from decimal import Decimal

import pytest

from src.db.decimals import (
    DecimalDrift,
    Money,
    NotADecimal,
    canonical,
    same_value,
    to_decimal,
)
from src.db.engine import Database
from src.mission.rsu_reconcile import ObservedEvent, PlannedEvent
from src.workspace.store import verify_decimal_columns, WorkspaceStore

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

#: Deliberately spans the cases where binary floating point misbehaves.
QUANTITIES = [
    "31800",          # whole
    "152.26",         # delivered shares
    "0.20006",        # concentration ratio
    "3896.10",        # net proceeds, trailing zero significant
    "1.00",           # trailing zeros preserved
    "0.000000000001", # at the declared scale
    "-42.5",          # negative, valid for an adjustment
    "99999999999999999999999.123456789012",  # large, within precision
]


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


def planned(**kwargs):
    return PlannedEvent(event_id="pe-1", grant_ref="grant/g1",
                        expected_date="2026-06-15", employer_asset="ACME",
                        **kwargs)


def observed(**kwargs):
    return ObservedEvent(observation_id="oe-1", observed_date="2026-06-16",
                         effective_date="2026-06-15", grant_ref="grant/g1",
                         employer_asset="ACME", **kwargs)


def store_and_read(store, quantity):
    store.record_planned_event(
        owner="alice", worksheet_id="ws-1",
        event=planned(expected_gross_shares=quantity, expected_value=quantity),
        plan_revision=1, created_at="2026-01-01T00:00:00Z",
        matching_policy_version="m@1")
    row = store.planned_events("ws-1", "alice")[0]
    return row["expected_quantity"], row["expected_value"]


class TestTheGrammar:
    @pytest.mark.parametrize("text,expected", [
        ("1.0", "1.0"),        # trailing zero is precision, preserved
        ("1.00", "1.00"),
        ("01.0", "1.0"),       # leading zeros carry nothing, dropped
        (".5", "0.5"),         # a digit before the point
        ("-0.00", "0.00"),     # negative zero is zero, one spelling only
        ("31800", "31800"),
    ])
    def test_accepted_and_canonicalized(self, text, expected):
        assert canonical(text) == expected

    @pytest.mark.parametrize("text", ["1.", "1e-3", "1E+3", "+1.0", "", "  ",
                                      "1,000", "0x10", "nan", "Infinity"])
    def test_refused(self, text):
        """Each of these has more than one obvious reading, and a stored value
        with two readings is what this grammar exists to prevent."""
        with pytest.raises(NotADecimal):
            canonical(text)

    def test_a_decimal_is_always_accepted_and_expanded(self):
        """A Decimal is already exact; only its spelling is normalized."""
        assert canonical(Decimal("1E+3")) == "1000"
        assert canonical(Decimal("1e-3")) == "0.001"
        assert canonical(Decimal("1.00")) == "1.00"

    def test_an_integer_is_accepted(self):
        assert canonical(31800) == "31800"

    def test_none_stays_none(self):
        """Unknown is not zero, here as everywhere else in this codebase."""
        assert canonical(None) is None
        assert to_decimal(None) is None


class TestFloatsAreRefused:
    @pytest.mark.parametrize("value", [0.1, 152.26, 0.0, -1.5, 1e-3])
    def test_a_float_cannot_enter(self, value):
        with pytest.raises(NotADecimal, match="float"):
            canonical(value)

    def test_a_bool_is_not_a_quantity(self):
        """`bool` is an `int` subclass, so it would otherwise become 1."""
        with pytest.raises(NotADecimal, match="boolean"):
            canonical(True)

    def test_a_float_cannot_enter_a_planned_event(self):
        """Refused where it enters, so the loss is attributed to its cause."""
        with pytest.raises(NotADecimal, match="float"):
            planned(expected_gross_shares=152.26)

    def test_a_float_cannot_enter_an_observed_event(self):
        with pytest.raises(NotADecimal, match="float"):
            observed(gross_shares=152.26)

    def test_a_float_cannot_enter_the_store(self):
        with pytest.raises(NotADecimal, match="float"):
            Money(0.1)


class TestCrossDialectEquality:
    """Same value in, same Decimal out, whichever engine holds it."""

    @pytest.mark.parametrize("quantity", QUANTITIES)
    def test_sqlite_returns_a_decimal(self, tmp_path, quantity):
        store = WorkspaceStore(tmp_path / "w.db")
        got, _ = store_and_read(store, quantity)
        assert isinstance(got, Decimal)
        assert got == Decimal(quantity)

    @pytest.mark.parametrize("quantity", QUANTITIES)
    def test_postgresql_returns_a_decimal(self, postgres_store, quantity):
        got, _ = store_and_read(postgres_store, quantity)
        assert isinstance(got, Decimal)
        assert got == Decimal(quantity)

    @pytest.mark.parametrize("quantity", QUANTITIES)
    def test_both_engines_agree(self, tmp_path, postgres_store, quantity):
        sqlite_value, _ = store_and_read(WorkspaceStore(tmp_path / "w.db"),
                                         quantity)
        postgres_value, _ = store_and_read(postgres_store, quantity)
        assert sqlite_value == postgres_value

    def test_none_round_trips_as_none(self, tmp_path):
        """A missing quantity must not come back as zero."""
        store = WorkspaceStore(tmp_path / "w.db")
        got, value = store_and_read(store, None)
        assert got is None and value is None

    def test_trailing_zeros_survive_the_round_trip(self, tmp_path):
        """`152.20` and `152.2` are one number and two statements about
        precision, and the payload records the precise one."""
        store = WorkspaceStore(tmp_path / "w.db")
        store_and_read(store, "3896.10")
        payload = store.planned_events("ws-1", "alice")[0]["payload"]
        assert payload["expected_gross_shares"] == "3896.10"


class TestTheDenormalizedCopyCannotDisagree:
    """These columns are a second answer to a question the payload answers."""

    def test_a_consistent_write_is_accepted(self, tmp_path):
        store = WorkspaceStore(tmp_path / "w.db")
        store_and_read(store, "152.26")
        assert verify_decimal_columns(store, "planned_event") == []

    def test_a_drifted_column_is_detected(self, tmp_path):
        """Change the column alone: verification must fail."""
        store = WorkspaceStore(tmp_path / "w.db")
        store_and_read(store, "152.26")
        with store._conn() as conn:
            conn.execute("UPDATE planned_event SET expected_quantity = ?",
                         ("152.27",))
        drift = verify_decimal_columns(store, "planned_event")
        assert [d["column"] for d in drift] == ["expected_quantity"]

    def test_a_drifted_payload_is_detected_independently(self, tmp_path):
        """Change the payload alone: the same check must fail, from the other
        side. One check that only looked at the column would miss this."""
        store = WorkspaceStore(tmp_path / "w.db")
        store_and_read(store, "152.26")
        row = store.planned_events("ws-1", "alice")[0]
        payload = dict(row["payload"])
        payload["expected_gross_shares"] = "999.99"
        with store._conn() as conn:
            from src.db.types import Json
            conn.execute("UPDATE planned_event SET payload = ?", (Json(payload),))
        drift = verify_decimal_columns(store, "planned_event")
        assert [d["column"] for d in drift] == ["expected_quantity"]

    def test_a_precision_only_difference_is_drift(self, tmp_path):
        """`152.2` against a payload of `152.26` is a rounded copy, and a
        rounded copy is exactly what a threshold comparison would misread."""
        store = WorkspaceStore(tmp_path / "w.db")
        store_and_read(store, "152.26")
        with store._conn() as conn:
            conn.execute("UPDATE planned_event SET expected_quantity = ?",
                         ("152.2",))
        assert verify_decimal_columns(store, "planned_event")

    def test_verification_reads_the_stored_row_not_the_writer(self, tmp_path):
        """Sharing code with the write would make the two agree by
        construction, as `verify_deleted` avoids for deletion."""
        store = WorkspaceStore(tmp_path / "w.db")
        store_and_read(store, "0.20006")
        assert verify_decimal_columns(store, "planned_event", owner="alice") == []
        assert verify_decimal_columns(store, "planned_event", owner="bob") == []

    def test_a_broken_payload_mapping_refuses_the_write(self, tmp_path):
        """The write-time check guards the field-to-payload-key mapping.

        Through the public API the column and the payload both come from the
        same event, so they agree by construction — the check cannot fire from
        a plain write, and a falsification that removed it left every other test
        passing. What it does catch is the mapping itself breaking: rename the
        payload key and the column becomes a copy of nothing, silently, with the
        row still written.
        """
        store = WorkspaceStore(tmp_path / "w.db")

        class RenamedPayload(PlannedEvent):
            def to_json(self):
                body = dict(super().to_json())
                body["gross_shares"] = body.pop("expected_gross_shares")
                return body

        with pytest.raises(DecimalDrift, match="expected_gross_shares"):
            store.record_planned_event(
                owner="alice", worksheet_id="ws-1",
                event=RenamedPayload(event_id="pe-1", grant_ref="grant/g1",
                                     expected_date="2026-06-15",
                                     employer_asset="ACME",
                                     expected_gross_shares="152.26"),
                plan_revision=1, created_at="2026-01-01T00:00:00Z",
                matching_policy_version="m@1")

    def test_a_payload_that_rounds_refuses_the_write(self, tmp_path):
        """A `to_json` that rounded would make the authoritative copy the less
        precise one, and the hash would certify the rounded value."""
        store = WorkspaceStore(tmp_path / "w.db")

        class RoundingPayload(PlannedEvent):
            def to_json(self):
                body = dict(super().to_json())
                body["expected_gross_shares"] = "152.3"
                return body

        with pytest.raises(DecimalDrift):
            store.record_planned_event(
                owner="alice", worksheet_id="ws-1",
                event=RoundingPayload(event_id="pe-1", grant_ref="grant/g1",
                                      expected_date="2026-06-15",
                                      employer_asset="ACME",
                                      expected_gross_shares="152.26"),
                plan_revision=1, created_at="2026-01-01T00:00:00Z",
                matching_policy_version="m@1")

    def test_nothing_is_written_when_the_mirror_check_refuses(self, tmp_path):
        """A refused write must leave no row, not a row with a bad column."""
        store = WorkspaceStore(tmp_path / "w.db")

        class RoundingPayload(PlannedEvent):
            def to_json(self):
                body = dict(super().to_json())
                body["expected_value"] = "1"
                return body

        with pytest.raises(DecimalDrift):
            store.record_planned_event(
                owner="alice", worksheet_id="ws-1",
                event=RoundingPayload(event_id="pe-1", grant_ref="grant/g1",
                                      expected_date="2026-06-15",
                                      employer_asset="ACME",
                                      expected_value="3896.10"),
                plan_revision=1, created_at="2026-01-01T00:00:00Z",
                matching_policy_version="m@1")
        assert store.planned_events("ws-1", "alice") == []

    def test_a_clean_row_reports_no_drift_on_postgresql(self, postgres_store):
        """The gap that let a broken verifier ship.

        `verify_decimal_columns` had only ever run against SQLite, where the
        canonical text is stored verbatim. On PostgreSQL the NUMERIC column
        pads to its declared scale, so comparing spellings reported drift on
        every clean row — the verifier was failing open on the engine that
        matters, and nothing looked.
        """
        store_and_read(postgres_store, "152.26")
        assert verify_decimal_columns(postgres_store, "planned_event") == []

    @pytest.mark.parametrize("quantity", QUANTITIES)
    def test_no_clean_quantity_reports_drift_on_postgresql(self, postgres_store,
                                                           quantity):
        store_and_read(postgres_store, quantity)
        assert verify_decimal_columns(postgres_store, "planned_event") == []

    def test_a_real_divergence_is_still_caught_on_postgresql(self,
                                                             postgres_store):
        """Loosening the comparison must not stop it detecting anything."""
        store_and_read(postgres_store, "152.26")
        with postgres_store._conn() as conn:
            conn.execute("UPDATE planned_event SET expected_quantity = ?",
                         ("152.27",))
        assert verify_decimal_columns(postgres_store, "planned_event")

    def test_a_rounded_column_is_still_caught_on_postgresql(self,
                                                            postgres_store):
        """152.2 against a payload of 152.26 is a rounded copy, and padding to
        scale must not disguise it."""
        store_and_read(postgres_store, "152.26")
        with postgres_store._conn() as conn:
            conn.execute("UPDATE planned_event SET expected_quantity = ?",
                         ("152.2",))
        assert verify_decimal_columns(postgres_store, "planned_event")

    def test_observed_events_are_covered_too(self, tmp_path):
        store = WorkspaceStore(tmp_path / "w.db")
        store.record_observed_event(
            owner="alice", worksheet_id="ws-1",
            event=observed(gross_shares="152.26", value="3896.10"),
            created_at="2026-01-01T00:00:00Z")
        assert verify_decimal_columns(store, "observed_event") == []
        with store._conn() as conn:
            conn.execute("UPDATE observed_event SET value = ?", ("3896.11",))
        assert verify_decimal_columns(store, "observed_event")


class TestSameValue:
    def test_equal_quantities_agree(self):
        assert same_value(Decimal("152.26"), "152.26")

    def test_a_difference_in_recorded_precision_is_a_difference(self):
        assert not same_value("152.2", "152.20")

    def test_absent_agrees_with_absent(self):
        assert same_value(None, None)

    def test_absent_does_not_agree_with_zero(self):
        assert not same_value(None, "0")


class TestPostgresArithmeticStaysExact:
    """The reason the column is NUMERIC and not a float copy."""

    def test_a_sum_is_exact(self, postgres_store):
        for index, quantity in enumerate(["0.1", "0.2", "0.3"]):
            postgres_store.record_observed_event(
                owner="alice", worksheet_id="ws-1",
                event=ObservedEvent(observation_id=f"oe-{index}",
                                    observed_date="2026-06-16",
                                    effective_date="2026-06-15",
                                    grant_ref="grant/g1", employer_asset="ACME",
                                    value=quantity),
                created_at="2026-01-01T00:00:00Z")
        conn = postgres_store.db.connect()
        try:
            row = conn.execute("SELECT SUM(value) AS total FROM observed_event"
                               ).fetchone()
        finally:
            conn.close()
        # 0.1 + 0.2 + 0.3 is 0.6 exactly here. In binary floating point it is
        # 0.6000000000000001, which a threshold comparison would read as over.
        assert row["total"] == Decimal("0.600000000000")

    def test_ordering_is_numeric_not_lexicographic(self, postgres_store):
        """A text column would order "9" after "100"; NUMERIC does not."""
        for index, quantity in enumerate(["100", "9", "1000"]):
            postgres_store.record_observed_event(
                owner="alice", worksheet_id="ws-1",
                event=ObservedEvent(observation_id=f"oe-{index}",
                                    observed_date="2026-06-16",
                                    effective_date="2026-06-15",
                                    grant_ref="grant/g1", employer_asset="ACME",
                                    value=quantity),
                created_at="2026-01-01T00:00:00Z")
        conn = postgres_store.db.connect()
        try:
            rows = conn.execute(
                "SELECT value FROM observed_event ORDER BY value").fetchall()
        finally:
            conn.close()
        assert [r["value"] for r in rows] == [
            Decimal("9"), Decimal("100"), Decimal("1000")]

    def test_the_columns_are_numeric_not_text(self, postgres_store):
        conn = postgres_store.db.connect()
        try:
            rows = conn.execute(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE data_type = 'numeric' AND table_schema = 'public'"
            ).fetchall()
        finally:
            conn.close()
        assert {(r["table_name"], r["column_name"]) for r in rows} == {
            ("planned_event", "expected_quantity"),
            ("planned_event", "expected_value"),
            ("observed_event", "quantity"),
            ("observed_event", "value")}
