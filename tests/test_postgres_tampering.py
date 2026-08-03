"""Three stored claims, three verifiers, one divergence at a time.

An artifact carries three separable assertions:

    the body is the one that was written        content hash over the payload
    the query columns agree with the body       decimal mirror check
    the conclusion still follows from it        re-derivation

Each test corrupts exactly one and requires the matching verifier to fail *and
the other two to pass*. The cross-check is the load-bearing part: without it a
single broad verifier could be catching all three, and the absence of the other
two would be invisible until the day it stopped catching one.

A fourth case corrupts all three at once and requires all three verifiers to
report independently — proving none of them exits early and suppresses the
others.

Every mutation is applied with direct SQL, below the application boundary. The
store cannot produce these states, which is the point: they are what the
verifiers exist for.
"""
from __future__ import annotations

import os

import pytest

from src.db.engine import Database
from src.db.types import Json
from src.mission.rsu_reconcile import (
    EventReconciliation,
    MatchingPolicy,
    ObservedEvent,
    PlannedEvent,
    ReconciliationStatus,
    reconcile,
)
from src.workspace import reconciliation_view
from src.workspace.store import (
    WorkspaceStore,
    verify_content_hashes,
    verify_decimal_columns,
)

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set QUANTIFY_TEST_POSTGRES_URL; JSONB tampering is a "
           "PostgreSQL-only guarantee")

OWNER = "alice"
WORKSHEET = "ws-1"


def session():
    return WorkspaceStore(POSTGRES_URL)


def execute(sql, params=()):
    conn = Database(POSTGRES_URL).connect()
    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall() if sql.strip().upper().startswith("SELECT") else []
        conn.commit()
        return rows
    finally:
        conn.close()


@pytest.fixture
def artifact():
    """One valid artifact, with all three claims agreeing."""
    from sqlalchemy import text

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    database.create_all()

    store = session()
    store.record_planned_event(
        owner=OWNER, worksheet_id=WORKSHEET,
        event=PlannedEvent(event_id="pe-1", grant_ref="grant/g1",
                           expected_date="2026-06-15", employer_asset="ACME",
                           expected_gross_shares="152.26",
                           expected_delivered_shares="78.00"),
        plan_revision=1, created_at="2026-01-01T00:00:00Z",
        matching_policy_version="m@1")
    store.record_observed_event(
        owner=OWNER, worksheet_id=WORKSHEET,
        event=ObservedEvent(observation_id="oe-1", observed_date="2026-06-16",
                            effective_date="2026-06-15", grant_ref="grant/g1",
                            employer_asset="ACME", gross_shares="152.26",
                            delivered_shares="78.00"),
        created_at="2026-01-01T00:00:00Z")
    derived = reconcile(
        [PlannedEvent(event_id="pe-1", grant_ref="grant/g1",
                      expected_date="2026-06-15", employer_asset="ACME",
                      expected_gross_shares="152.26",
                      expected_delivered_shares="78.00")],
        [ObservedEvent(observation_id="oe-1", observed_date="2026-06-16",
                       effective_date="2026-06-15", grant_ref="grant/g1",
                       employer_asset="ACME", gross_shares="152.26",
                       delivered_shares="78.00")],
        as_of="2026-06-20")
    for one in derived:
        store.record_reconciliation(owner=OWNER, worksheet_id=WORKSHEET,
                                    reconciliation=one)
    return store


# --- the three verifiers, each called the same way -------------------------

def identity_fails(store, table="planned_event"):
    return bool(verify_content_hashes(store, table, owner=OWNER))


def mirror_fails(store, table="planned_event"):
    return bool(verify_decimal_columns(store, table, owner=OWNER))


def derivation_fails(store):
    stored = store.reconciliations(WORKSHEET, OWNER)
    planned = [PlannedEvent(event_id="pe-1", grant_ref="grant/g1",
                            expected_date="2026-06-15", employer_asset="ACME",
                            expected_gross_shares="152.26",
                            expected_delivered_shares="78.00")]
    observed = [ObservedEvent(observation_id="oe-1", observed_date="2026-06-16",
                              effective_date="2026-06-15",
                              grant_ref="grant/g1", employer_asset="ACME",
                              gross_shares="152.26",
                              delivered_shares="78.00")]
    fresh = reconcile(planned, observed, as_of="2026-06-20")
    states = reconciliation_view.verify(stored, fresh)
    return any(state.value != "VERIFIED" for state in states.values())


class TestTheBaselineIsClean:
    """All three must agree before any of them can mean anything."""

    def test_no_verifier_reports_a_problem(self, artifact):
        assert not identity_fails(artifact)
        assert not mirror_fails(artifact)
        assert not derivation_fails(artifact)


class TestPayloadIdentityCorruption:
    """The body is edited; the hash, the mirror and the conclusion are not."""

    def corrupt(self):
        # A field the mirror does not copy, so only the hash notices.
        execute("UPDATE planned_event SET payload = "
                "jsonb_set(payload, '{source_declaration}', '\"forged\"') "
                "WHERE owner = %s", (OWNER,))

    def test_the_identity_verifier_fails(self, artifact):
        self.corrupt()
        assert identity_fails(artifact)

    def test_the_mirror_verifier_still_passes(self, artifact):
        self.corrupt()
        assert not mirror_fails(artifact), (
            "the mirror check reported a payload edit it has no view of — it "
            "is doing the identity check's job and would hide its absence")

    def test_the_derivation_verifier_still_passes(self, artifact):
        self.corrupt()
        assert not derivation_fails(artifact)


class TestMirrorCorruption:
    """Only the denormalized query column moves."""

    def corrupt(self):
        execute("UPDATE planned_event SET expected_quantity = 999.99 "
                "WHERE owner = %s", (OWNER,))

    def test_the_mirror_verifier_fails(self, artifact):
        self.corrupt()
        assert mirror_fails(artifact)

    def test_the_identity_verifier_still_passes(self, artifact):
        """The payload is untouched, so its hash must still verify."""
        self.corrupt()
        assert not identity_fails(artifact)

    def test_the_derivation_verifier_still_passes(self, artifact):
        self.corrupt()
        assert not derivation_fails(artifact)


class TestDerivedStateCorruption:
    """A stored conclusion is changed; its evidence is not."""

    def corrupt(self):
        execute("UPDATE event_reconciliation SET status = 'LATE', "
                "payload = jsonb_set(payload, '{status}', '\"LATE\"') "
                "WHERE owner = %s", (OWNER,))
        # Re-hash so the identity check has nothing to say: this test is about
        # a conclusion that no longer follows, not about a corrupted body.
        rows = execute("SELECT reconciliation_id, payload FROM "
                       "event_reconciliation WHERE owner = %s", (OWNER,))
        from src.runtime.base import canonical_hash
        for row in rows:
            execute("UPDATE event_reconciliation SET content_hash = %s "
                    "WHERE owner = %s AND reconciliation_id = %s",
                    (canonical_hash(row["payload"]), OWNER,
                     row["reconciliation_id"]))

    def test_the_derivation_verifier_fails(self, artifact):
        self.corrupt()
        assert derivation_fails(artifact)

    def test_the_identity_verifier_still_passes(self, artifact):
        """The hash was recomputed over the edited body, so the body *is* what
        the hash says. Only re-derivation can tell that it should not be."""
        self.corrupt()
        assert not identity_fails(artifact, table="event_reconciliation")

    def test_the_mirror_verifier_still_passes(self, artifact):
        self.corrupt()
        assert not mirror_fails(artifact)


class TestAllThreeAtOnce:
    """No verifier may exit early and suppress the others."""

    def corrupt(self):
        TestPayloadIdentityCorruption().corrupt()
        TestMirrorCorruption().corrupt()
        TestDerivedStateCorruption().corrupt()

    def test_every_verifier_reports_independently(self, artifact):
        self.corrupt()
        assert identity_fails(artifact), "the identity verifier went quiet"
        assert mirror_fails(artifact), "the mirror verifier went quiet"
        assert derivation_fails(artifact), "the derivation verifier went quiet"

    def test_each_names_what_it_found(self, artifact):
        """A verifier that only returns a boolean cannot be acted on."""
        self.corrupt()
        identity = verify_content_hashes(artifact, "planned_event", owner=OWNER)
        mirror = verify_decimal_columns(artifact, "planned_event", owner=OWNER)
        assert identity[0]["stored_hash"] != identity[0]["recomputed_hash"]
        assert mirror[0]["column"] == "expected_quantity"
        assert mirror[0]["field"] == "expected_gross_shares"


class TestTheVerifiersAreNotOneCheck:
    """Each has a case only it detects, so none is redundant."""

    def test_identity_alone_catches_a_payload_edit(self, artifact):
        TestPayloadIdentityCorruption().corrupt()
        assert (identity_fails(artifact), mirror_fails(artifact),
                derivation_fails(artifact)) == (True, False, False)

    def test_mirror_alone_catches_a_column_edit(self, artifact):
        TestMirrorCorruption().corrupt()
        assert (identity_fails(artifact), mirror_fails(artifact),
                derivation_fails(artifact)) == (False, True, False)

    def test_derivation_alone_catches_a_rewritten_conclusion(self, artifact):
        TestDerivedStateCorruption().corrupt()
        assert (identity_fails(artifact), mirror_fails(artifact),
                derivation_fails(artifact)) == (False, False, True)
