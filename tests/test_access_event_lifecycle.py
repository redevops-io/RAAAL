"""What happens to a delivery record after the run that cites it exists.

The access-event logic was proven where it is written. The risk that remains is
adjacent: deletion, export, transfer, retry and rollback all iterate table
inventories, and a new table joins those inventories by derivation rather than
by anyone remembering it. Derivation is why it should work; a test is why it is
known to.

Each case here is a lifecycle question that could be answered wrongly without
any of the access-event tests failing:

    deletion      does it remove this owner's events, and only this owner's
    export        exactly once, with digests intact
    transfer      does the table and its constraint survive the round trip
    legacy        do old runs stay explicitly eventless
    redelivery    is identical-only still identical-only
    retry         does one shared access stay one delivery
    rollback      can a failed fan-out leave a run citing an aborted event
    sweep         can any bulk delete remove evidence a run still cites
"""
from __future__ import annotations

import pytest

from src.workspace.store import NotSaveable, WorkspaceStore

POLICY = "PILOT_DATA_POLICY"
OWNER = "alice"
OTHER = "mallory"
AT = "2026-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")


def a_scenario():
    from tests.test_producer_inventory import TestInstanceCompleteness

    return TestInstanceCompleteness().scenario()


def an_access(run_id="run-1", request_id="req-1"):
    from src.market_data.access import resolve

    return resolve(context="a run", accessed_at=AT, run_id=run_id,
                   request_id=request_id)


@pytest.fixture
def workspace(tmp_path):
    """One plan and one cited run for each of two owners."""
    store = WorkspaceStore(tmp_path / "w.db")
    scenario = a_scenario()
    accesses = {}
    for owner in (OWNER, OTHER):
        store.save_plan(plan_id=f"p-{owner}", owner=owner, scenario=scenario,
                        stated_text="x", saved_at=AT)
        access = an_access(run_id=f"run-{owner}", request_id=f"req-{owner}")
        store.record_access_event(access.access_event, owner=owner)
        store.record_run(
            run_id=f"run-{owner}", plan_id=f"p-{owner}", ran_at=AT, owner=owner,
            result={"modelling_scope": {"excludes": []}, "final_value": 1.0,
                    "market_data": access.provenance.to_json()},
            comparison={}, access_event_id=access.access_event_id)
        accesses[owner] = access
    return store, accesses


def events(store, owner):
    with store._conn() as conn:
        return [dict(row) for row in conn.execute(
            "SELECT * FROM market_data_access_event WHERE owner = ?",
            (owner,)).fetchall()]


class TestDeletionReachesTheDeliveryAndStopsAtTheTenant:
    """The table joins the deletion by derivation from the registry and the
    relationship graph. Derivation is why it should work; this is why it is
    known to — the previous ordering was a heuristic that happened to be right
    for the one indirect table that existed."""

    def test_the_owner_s_events_are_deleted(self, workspace):
        from src.workspace.erasure import delete_workspace

        store, _ = workspace
        receipt = delete_workspace(store, OWNER, requested_at=AT)
        assert events(store, OWNER) == []
        assert receipt.counts["market_data_access_event"] == 1

    def test_the_other_tenant_keeps_theirs(self, workspace):
        from src.workspace.erasure import delete_workspace

        store, _ = workspace
        delete_workspace(store, OWNER, requested_at=AT)
        assert len(events(store, OTHER)) == 1

    def test_the_deletion_order_puts_runs_first(self, workspace):
        """RESTRICT would refuse the reverse, so an order that deleted events
        before the runs citing them would fail rather than silently keep
        them — but it would fail on a user's deletion request."""
        from src.db.schema import deletion_order

        order = list(deletion_order())
        assert order.index("plan_run") < order.index("market_data_access_event")

    def test_the_receipt_names_the_table(self, workspace):
        from src.workspace.erasure import delete_workspace

        store, _ = workspace
        receipt = delete_workspace(store, OWNER, requested_at=AT)
        assert "market_data_access_event" in receipt.counts

    def test_the_receipt_reproduces_no_deleted_content(self, workspace):
        """A receipt is handed to the user whose data was removed. Digests are
        not personal content, but a receipt that quoted the record it deleted
        would be reproducing what it just erased."""
        from src.workspace.erasure import delete_workspace

        store, accesses = workspace
        receipt = delete_workspace(store, OWNER, requested_at=AT)
        rendered = str(receipt.to_json())
        assert accesses[OWNER].access_event.frame_digest not in rendered
        assert accesses[OWNER].access_event_id not in rendered

    def test_verification_would_notice_a_skipped_table(self, workspace):
        """`verify_deleted` reads the registry rather than the list the
        deletion iterates — the two used to agree by construction."""
        from src.workspace.erasure import verify_deleted

        store, _ = workspace
        assert "market_data_access_event" in verify_deleted(store, OWNER)


class TestExportCarriesItExactlyOnce:
    def test_it_appears_in_the_bundle(self, workspace):
        from src.db.transfer import export_bundle

        store, _ = workspace
        bundle = export_bundle(store, exported_at=AT)
        assert bundle["manifest"]["counts"]["market_data_access_event"] == 2

    def test_the_digests_survive(self, workspace):
        from src.db.transfer import export_bundle

        store, accesses = workspace
        bundle = export_bundle(store, exported_at=AT)
        carried = {row["frame_digest"]
                   for row in bundle["records"]["market_data_access_event"]}
        assert carried == {accesses[OWNER].access_event.frame_digest}

    def test_a_narrowed_export_carries_one_tenant(self, workspace):
        from src.db.transfer import export_bundle

        store, _ = workspace
        bundle = export_bundle(store, exported_at=AT, owner=OWNER)
        rows = bundle["records"]["market_data_access_event"]
        assert [row["owner"] for row in rows] == [OWNER]

    def test_the_run_reference_survives(self, workspace):
        """An exported run that lost its citation would be a figure whose
        evidence exists in the same bundle and is no longer reachable."""
        from src.db.transfer import export_bundle

        store, accesses = workspace
        bundle = export_bundle(store, exported_at=AT, owner=OWNER)
        runs = bundle["records"]["plan_run"]
        assert [row["access_event_id"] for row in runs] == \
            [accesses[OWNER].access_event_id]


class TestTheTransferRoundTripPreservesTheChain:
    """SQLite -> neutral bundle -> PostgreSQL. The import order is
    `reversed(deletion_order())`, so the delivery must land before the run
    that cites it or the foreign key refuses."""

    @pytest.fixture
    def postgres(self):
        import os

        url = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")
        if not url:
            pytest.skip("the transfer contract is about the deployed engine")
        from sqlalchemy import text

        from src.db import migrate
        from src.db.engine import Database

        database = Database(url)
        engine = database.sqlalchemy_engine()
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA public CASCADE"))
            connection.execute(text("CREATE SCHEMA public"))
        engine.dispose()
        migrate.upgrade(database)
        return WorkspaceStore(url)

    def test_the_events_arrive(self, workspace, postgres):
        from src.db.transfer import apply_import, export_bundle

        store, _ = workspace
        apply_import(postgres.db, export_bundle(store, exported_at=AT))
        assert len(events(postgres, OWNER)) == 1

    def test_the_digest_is_unchanged_across_engines(self, workspace, postgres):
        """Representation-sensitive: JSONB and TEXT are different storage, and
        a digest that changed across the boundary would report every
        transferred run as tampered."""
        from src.db.transfer import apply_import, export_bundle

        store, accesses = workspace
        apply_import(postgres.db, export_bundle(store, exported_at=AT))
        assert events(postgres, OWNER)[0]["frame_digest"] == \
            accesses[OWNER].access_event.frame_digest

    def test_the_chain_still_verifies_after_transfer(self, workspace, postgres):
        from src.db.transfer import apply_import, export_bundle

        store, _ = workspace
        apply_import(postgres.db, export_bundle(store, exported_at=AT))
        assert postgres.verify_access_chain(f"run-{OWNER}", OWNER) == []

    def test_the_constraint_arrived_too(self, workspace, postgres):
        """A transfer that carried the rows and not the constraint would leave
        a database where evidence can be deleted out from under a figure."""
        from src.db.transfer import apply_import, export_bundle

        store, _ = workspace
        apply_import(postgres.db, export_bundle(store, exported_at=AT))
        with pytest.raises(Exception) as refusal:                # noqa: PT011
            with postgres._conn() as conn:
                conn.execute(
                    "DELETE FROM market_data_access_event WHERE owner = ?",
                    (OWNER,))
        assert getattr(refusal.value, "sqlstate", "23503") == "23503"


class TestOldRunsStayLegacy:
    def test_a_run_without_an_event_keeps_none(self, workspace):
        """`NOT_RECORDED` is never inferred, and neither is a delivery. A
        back-filled event would be manufactured from today's configuration —
        the evidence the table exists to provide, invented."""
        store, _ = workspace
        store.record_run(
            run_id="legacy", plan_id=f"p-{OWNER}", ran_at=AT, owner=OWNER,
            result={"modelling_scope": {"excludes": []},
                    "market_data": {"status":
                                    "MARKET_DATA_PROVENANCE_NOT_RECORDED"}},
            comparison={})
        assert store.get_run("legacy", OWNER)["access_event_id"] is None

    def test_it_verifies_as_legacy_rather_than_broken(self, workspace):
        store, _ = workspace
        store.record_run(
            run_id="legacy", plan_id=f"p-{OWNER}", ran_at=AT, owner=OWNER,
            result={"modelling_scope": {"excludes": []},
                    "market_data": {"status":
                                    "MARKET_DATA_PROVENANCE_NOT_RECORDED"}},
            comparison={})
        assert store.verify_access_chain("legacy", OWNER) == []

    def test_reading_a_legacy_run_resolves_nothing(self, workspace,
                                                    monkeypatch):
        import src.market_data.access as access_module

        store, _ = workspace
        store.record_run(
            run_id="legacy", plan_id=f"p-{OWNER}", ran_at=AT, owner=OWNER,
            result={"modelling_scope": {"excludes": []},
                    "market_data": {"status":
                                    "MARKET_DATA_PROVENANCE_NOT_RECORDED"}},
            comparison={})
        monkeypatch.setattr(access_module, "resolve", _refuse)
        assert store.verify_access_chain("legacy", OWNER) == []


def _refuse(*args, **kwargs):
    raise AssertionError("the resolver was called while reading stored state")


class TestRedeliveryIsIdenticalOnly:
    def test_the_same_event_twice_is_one_row(self, workspace):
        store, accesses = workspace
        store.record_access_event(accesses[OWNER].access_event, owner=OWNER)
        assert len(events(store, OWNER)) == 1

    def test_one_field_different_is_a_conflict(self, workspace):
        """Every field, not a chosen subset — the fields most worth editing are
        the ones a careless exclusion would pick."""
        import dataclasses

        store, accesses = workspace
        event = accesses[OWNER].access_event
        for field, value in (("frame_digest", "mdf1:other"),
                             ("provenance_digest", "mdp1:other"),
                             ("row_count", 7),
                             ("snapshot_id", "another-snapshot"),
                             ("accessed_at", "2020-01-01T00:00:00Z"),
                             ("request_id", "another-request")):
            forged = dataclasses.replace(event, **{field: value})
            with pytest.raises(NotSaveable, match="different body"):
                store.record_access_event(forged, owner=OWNER)

    def test_two_owners_may_hold_one_identity(self, workspace):
        """Scoped by (owner, access_event_id): one tenant must not be able to
        make another's write fail, or to learn that their id exists."""
        import dataclasses

        store, accesses = workspace
        borrowed = dataclasses.replace(
            accesses[OTHER].access_event,
            access_event_id=accesses[OWNER].access_event_id)
        store.record_access_event(borrowed, owner=OTHER)
        assert len(events(store, OTHER)) == 2


class TestOneSharedAccessStaysOneDelivery:
    @pytest.fixture
    def fanned(self, tmp_path, monkeypatch):
        from tests.test_access_chain import (
            TestOneResolutionIsOneDeliveryAcrossAFanOut as FanOut,
        )

        harness = FanOut()
        store = harness.prepared.__wrapped__(harness, tmp_path, monkeypatch)
        return harness, store

    def test_accepting_records_one_event(self, fanned):
        harness, store = fanned
        harness.accepted(store)
        assert len(events(store, harness.OWNER)) == 1

    def test_retrying_the_same_access_adds_none(self, fanned):
        """A retry that re-uses the resolved access is the same delivery
        arriving twice, not a second read."""
        harness, store = fanned
        _, access = harness.accepted(store)
        store.record_access_event(access.access_event, owner=harness.OWNER)
        store.record_access_event(access.access_event, owner=harness.OWNER)
        assert len(events(store, harness.OWNER)) == 1

    def test_a_second_acceptance_is_refused_before_a_second_event(self, fanned):
        """Accepting twice is already refused — the point here is that the
        refusal happens without leaving a second delivery behind.

        The *same* staged proposal is accepted again. Re-staging under the same
        id would reset its status and test the fixture rather than the guard.
        """
        from src.workspace.apply import ApplyRefused, accept
        from src.workspace.proposal import from_json as proposal_from_json

        harness, store = fanned
        _, access = harness.accepted(store)
        record = store.get_worksheet_proposal("p1", harness.OWNER)
        with pytest.raises(ApplyRefused):
            accept(store, proposal_id="p1", owner=harness.OWNER,
                   worksheet_id="ws-1",
                   proposal=proposal_from_json(record["payload"]), at="t2",
                   run_candidate=lambda candidate: {}, access=access)
        assert len(events(store, harness.OWNER)) == 1


class TestAFailedFanOutLeavesNothingBehind:
    """`_apply` runs inside one transaction, so a candidate that fails must
    take the delivery record with it. A committed event with no run is inert;
    a committed run citing a rolled-back event is impossible — but only if the
    two are in the same transaction, which is what this checks."""

    @pytest.fixture
    def staged(self, tmp_path, monkeypatch):
        from tests.test_access_chain import (
            TestOneResolutionIsOneDeliveryAcrossAFanOut as FanOut,
        )

        harness = FanOut()
        store = harness.prepared.__wrapped__(harness, tmp_path, monkeypatch)
        return harness, store

    def failing_accept(self, harness, store, fail_on=1, attempts=None):
        from src.workspace.apply import accept
        from src.workspace.intent import plan
        from src.workspace.proposal import propose
        from src.workspace.worksheet import from_json

        from tests.test_access_chain import _RESULT, an_access

        worksheet = from_json(
            store.get_worksheet("ws-1", harness.OWNER)["payload"])
        intent = plan("Try SPY, VTI and VT and keep the best", intent_id="i",
                      source_revision=worksheet.revision, history=[],
                      target_run="run-0")
        proposal = propose(intent, worksheet)
        store.save_worksheet_proposal(
            proposal_id="p1", owner=harness.OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="t0")

        access = an_access(run_id=None)
        body = {**_RESULT, "market_data": access.provenance.to_json()}
        seen = []

        def runner(candidate):
            seen.append(candidate)
            if len(seen) > fail_on:
                raise RuntimeError("this candidate could not be simulated")
            return body

        # A premise witness. "Nothing survived the rollback" is free if
        # nothing was ever written, which is exactly how the telemetry
        # independence claim passed for the life of that suite. Counting the
        # attempted write proves the transaction had something to discard.
        if attempts is not None:
            original = store.record_access_event

            def counted(event, *, owner):
                attempts.append(event.access_event_id)
                return original(event, owner=owner)

            store.record_access_event = counted

        with pytest.raises(Exception):                           # noqa: PT011
            accept(store, proposal_id="p1", owner=harness.OWNER,
                   worksheet_id="ws-1", proposal=proposal, at="t1",
                   run_candidate=runner, access=access)
        return access

    def test_no_delivery_survives_the_rollback(self, staged):
        harness, store = staged
        attempts = []
        self.failing_accept(harness, store, attempts=attempts)
        assert attempts, (
            "no delivery write was attempted, so this case proves nothing "
            "about rollback — it would pass against a fan-out that records "
            "no evidence at all")
        assert events(store, harness.OWNER) == [], (
            "a delivery record outlived the transaction that wrote it; the "
            "run it was written for does not exist")

    def test_no_run_survives_the_rollback(self, staged):
        harness, store = staged
        self.failing_accept(harness, store)
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT run_id FROM plan_run WHERE run_id LIKE 'p1-run-%'"
            ).fetchall()
        assert rows == []

    def test_the_proposal_is_still_open(self, staged):
        """Nothing half-applied: a failed fan-out must leave the proposal
        acceptable rather than consumed."""
        harness, store = staged
        self.failing_accept(harness, store)
        record = store.get_worksheet_proposal("p1", harness.OWNER)
        assert record["status"] == "PROPOSED"


class TestNoSweepCanRemoveCitedEvidence:
    """Orphan cleanup does not exist yet. When it does, the database is what
    stops it removing evidence a run still cites — so the guarantee is tested
    against the constraint rather than against a policy nobody has written."""

    def test_a_bulk_delete_of_all_events_is_refused(self, workspace):
        store, _ = workspace
        with pytest.raises(Exception):                           # noqa: PT011
            with store._conn() as conn:
                conn.execute("DELETE FROM market_data_access_event")
        assert len(events(store, OWNER)) == 1

    def test_a_sweep_of_apparently_unbound_events_is_refused(self, workspace):
        """The shape a naive cleanup would take: delete events whose own
        `run_id` is null. A fan-out delivery has exactly that shape and is
        cited by every candidate."""
        store, accesses = workspace
        with store._conn() as conn:
            conn.execute(
                "UPDATE market_data_access_event SET run_id = NULL "
                "WHERE owner = ?", (OWNER,))
        with pytest.raises(Exception):                           # noqa: PT011
            with store._conn() as conn:
                conn.execute(
                    "DELETE FROM market_data_access_event WHERE run_id IS NULL")
        assert len(events(store, OWNER)) == 1

    def test_a_genuinely_orphan_event_can_be_removed(self, workspace):
        """The counterpart: cleanup must be possible, or the crash-mode choice
        turns inert rows into permanent ones."""
        store, _ = workspace
        orphan = an_access(run_id=None, request_id="req-orphan")
        store.record_access_event(orphan.access_event, owner=OWNER)
        with store._conn() as conn:
            conn.execute(
                "DELETE FROM market_data_access_event WHERE access_event_id = ?",
                (orphan.access_event_id,))
        assert len(events(store, OWNER)) == 1
