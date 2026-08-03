"""Durable forward tracking, and a worksheet that reads rather than re-derives.

    planned + observed persisted
      -> reconciliation persisted
      -> verified against a fresh derivation
      -> three lanes

Stored alone, a status can be edited to say MATCHED beside a variance saying
otherwise. Derived alone, a historical row re-judges itself whenever the rules
change. Both layers, compared, is what makes either trustworthy.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest

from src.mission.rsu_reconcile import (
    MATCHING_POLICY_VERSION,
    MatchingPolicy,
    ObservedEvent,
    PlannedEvent,
    ReconciliationStatus,
    reconcile,
)
from src.workspace.reconciliation_view import (
    RSUReconciliationView,
    VerificationState,
    verify,
)
from src.workspace.store import NotSaveable, WorkspaceStore

OWNER = "pilot"
WORKSHEET = "ws-1"

JUNE = PlannedEvent(
    event_id="plan-jun", grant_ref="grant/g1", expected_date="2026-06-15",
    employer_asset="ACME", expected_gross_shares="100.0",
    expected_withheld_shares="22.0", expected_delivered_shares="78.0",
    source_declaration="declaration/rsu@1", version_pin="pin-abc")


def observation(**overrides) -> ObservedEvent:
    base = dict(observation_id="obs-1", observed_date="2026-06-16",
                effective_date="2026-06-15", grant_ref="grant/g1",
                employer_asset="ACME", gross_shares="100.0",
                withheld_shares="22.0", delivered_shares="78.0",
                evidence_ref="statement/june")
    base.update(overrides)
    return ObservedEvent(**base)


@pytest.fixture
def store(tmp_path):
    return WorkspaceStore(tmp_path / "w.db")


def persist(store, planned=(JUNE,), observed=(), *, as_of):
    for one in planned:
        store.record_planned_event(
            owner=OWNER, worksheet_id=WORKSHEET, event=one, plan_revision=1,
            created_at="t0", matching_policy_version=MATCHING_POLICY_VERSION)
    for one in observed:
        store.record_observed_event(owner=OWNER, worksheet_id=WORKSHEET,
                                    event=one, created_at="t1")

    rows = reconcile(list(planned), list(observed), as_of=as_of)
    for row in rows:
        store.record_reconciliation(owner=OWNER, worksheet_id=WORKSHEET,
                                    reconciliation=row)
    return rows


def rebuild(store, *, as_of):
    """The whole reconciliation, from the database alone."""
    planned = [PlannedEvent(**{k: v for k, v in one["payload"].items()})
               for one in store.planned_events(WORKSHEET, OWNER)]
    observed = [ObservedEvent(**{k: v for k, v in one["payload"].items()})
                for one in store.observed_events(WORKSHEET, OWNER)]
    return planned, observed, reconcile(planned, observed, as_of=as_of)


class TestRecordsPersistIndependently:

    def test_all_three_are_stored(self, store):
        persist(store, observed=[observation()], as_of="2026-06-20")
        assert len(store.planned_events(WORKSHEET, OWNER)) == 1
        assert len(store.observed_events(WORKSHEET, OWNER)) == 1
        assert len(store.reconciliations(WORKSHEET, OWNER)) == 1

    def test_the_planned_event_keeps_its_plan_revision(self, store):
        persist(store, as_of="2026-06-10")
        [row] = store.planned_events(WORKSHEET, OWNER)
        assert row["plan_revision"] == 1

    def test_effective_and_report_dates_stay_apart_in_storage(self, store):
        persist(store, observed=[observation(observed_date="2026-07-02",
                                             effective_date="2026-06-15")],
                as_of="2026-07-10")
        [row] = store.observed_events(WORKSHEET, OWNER)
        assert row["effective_date"] == "2026-06-15"
        assert row["observed_at"] == "2026-07-02"

    def test_a_pending_row_stores_no_observation_id(self, store):
        persist(store, as_of="2026-06-10")
        [row] = store.reconciliations(WORKSHEET, OWNER)
        assert row["observed_event_id"] is None
        assert row["status"] == "PENDING"


class TestImmutabilityAndCorrections:

    def test_an_identical_write_is_redelivery(self, store):
        persist(store, as_of="2026-06-10")
        store.record_planned_event(
            owner=OWNER, worksheet_id=WORKSHEET, event=JUNE, plan_revision=1,
            created_at="t0", matching_policy_version=MATCHING_POLICY_VERSION)
        assert len(store.planned_events(WORKSHEET, OWNER)) == 1

    def test_a_changed_expectation_is_refused(self, store):
        """A prediction that changed after the fact is not a prediction."""
        persist(store, as_of="2026-06-10")
        with pytest.raises(NotSaveable, match="different body"):
            store.record_planned_event(
                owner=OWNER, worksheet_id=WORKSHEET,
                event=replace(JUNE, expected_delivered_shares="70.0"),
                plan_revision=1, created_at="t0",
                matching_policy_version=MATCHING_POLICY_VERSION)

    def test_a_rewritten_observation_is_refused(self, store):
        persist(store, observed=[observation()], as_of="2026-06-20")
        with pytest.raises(NotSaveable, match="supersedes"):
            store.record_observed_event(
                owner=OWNER, worksheet_id=WORKSHEET,
                event=observation(delivered_shares="70.0"), created_at="t2")

    def test_a_correction_is_a_new_record_and_the_original_remains(self, store):
        persist(store, observed=[observation()], as_of="2026-06-20")
        store.record_observed_event(
            owner=OWNER, worksheet_id=WORKSHEET,
            event=observation(observation_id="obs-2", delivered_shares="70.0"),
            created_at="t2", supersedes="obs-1")

        rows = {one["observed_event_id"]: one
                for one in store.observed_events(WORKSHEET, OWNER)}
        assert set(rows) == {"obs-1", "obs-2"}
        assert rows["obs-2"]["supersedes"] == "obs-1"

    def test_a_correction_leaves_the_plan_untouched(self, store):
        persist(store, observed=[observation()], as_of="2026-06-20")
        before = store.planned_events(WORKSHEET, OWNER)[0]["content_hash"]

        store.record_observed_event(
            owner=OWNER, worksheet_id=WORKSHEET,
            event=observation(observation_id="obs-2", delivered_shares="70.0"),
            created_at="t2", supersedes="obs-1")
        assert store.planned_events(WORKSHEET, OWNER)[0]["content_hash"] == before


class TestTenantScope:

    def test_another_owner_sees_none_of_it(self, store):
        persist(store, observed=[observation()], as_of="2026-06-20")
        assert store.planned_events(WORKSHEET, "someone-else") == []
        assert store.observed_events(WORKSHEET, "someone-else") == []
        assert store.reconciliations(WORKSHEET, "someone-else") == []

    def test_two_owners_may_hold_the_same_event_id(self, store):
        """These rows carry employer names and compensation quantities, so a
        write refusal that revealed another tenant's ids would be worse than
        the worksheet case that prompted the rule."""
        persist(store, as_of="2026-06-10")
        store.record_planned_event(
            owner="someone-else", worksheet_id=WORKSHEET, event=JUNE,
            plan_revision=1, created_at="t0",
            matching_policy_version=MATCHING_POLICY_VERSION)
        assert len(store.planned_events(WORKSHEET, "someone-else")) == 1


class TestRebuildFromTheDatabaseAlone:

    def test_the_whole_reconciliation_is_reproducible(self, store):
        original = persist(store, observed=[observation(
            effective_date="2026-06-19")], as_of="2026-06-30")
        _, _, rebuilt = rebuild(store, as_of="2026-06-30")

        assert [one.status for one in rebuilt] == [one.status for one in original]
        assert rebuilt[0].status is ReconciliationStatus.LATE

    def test_a_pending_row_rebuilds_as_pending(self, store):
        persist(store, as_of="2026-06-10")
        _, _, rebuilt = rebuild(store, as_of="2026-06-10")
        assert rebuilt[0].status is ReconciliationStatus.PENDING


class TestStoredIsVerifiedAgainstDerived:

    def test_an_untouched_reconciliation_verifies(self, store):
        persist(store, observed=[observation()], as_of="2026-06-20")
        _, _, fresh = rebuild(store, as_of="2026-06-20")
        states = verify(store.reconciliations(WORKSHEET, OWNER), fresh)
        assert set(states.values()) == {VerificationState.VERIFIED}

    def test_an_edited_status_is_caught(self, store, tmp_path):
        """MATCHED written over LATE, with the variance left in place."""
        persist(store, observed=[observation(effective_date="2026-06-19")],
                as_of="2026-06-30")
        with sqlite3.connect(tmp_path / "w.db") as conn:
            conn.execute("UPDATE event_reconciliation SET status = 'MATCHED'")

        _, _, fresh = rebuild(store, as_of="2026-06-30")
        states = verify(store.reconciliations(WORKSHEET, OWNER), fresh)
        assert set(states.values()) == {
            VerificationState.DERIVATION_MISMATCH}

    def test_an_altered_variance_is_caught(self, store, tmp_path):
        persist(store, observed=[observation(delivered_shares="70.0")],
                as_of="2026-06-20")
        with sqlite3.connect(tmp_path / "w.db") as conn:
            row = conn.execute(
                "SELECT reconciliation_id, payload FROM event_reconciliation"
            ).fetchone()
            payload = json.loads(row[1])
            payload["variances"] = []
            conn.execute(
                "UPDATE event_reconciliation SET payload = ? WHERE "
                "reconciliation_id = ?", (json.dumps(payload), row[0]))

        _, _, fresh = rebuild(store, as_of="2026-06-20")
        states = verify(store.reconciliations(WORKSHEET, OWNER), fresh)
        assert VerificationState.DERIVATION_MISMATCH in states.values()

    def test_an_unverified_row_is_still_shown_as_history(self, store, tmp_path):
        persist(store, observed=[observation(effective_date="2026-06-19")],
                as_of="2026-06-30")
        with sqlite3.connect(tmp_path / "w.db") as conn:
            conn.execute("UPDATE event_reconciliation SET status = 'MATCHED'")

        _, _, fresh = rebuild(store, as_of="2026-06-30")
        view = RSUReconciliationView.from_records(
            store.planned_events(WORKSHEET, OWNER),
            store.observed_events(WORKSHEET, OWNER),
            store.reconciliations(WORKSHEET, OWNER),
            verification=verify(store.reconciliations(WORKSHEET, OWNER), fresh))

        assert len(view.rows) == 1
        assert view.unverified_count == 1
        assert view.rows[0].verification is \
            VerificationState.DERIVATION_MISMATCH


class TestTheViewDecidesNothing:

    def build(self, store, as_of):
        _, _, fresh = rebuild(store, as_of=as_of)
        return RSUReconciliationView.from_records(
            store.planned_events(WORKSHEET, OWNER),
            store.observed_events(WORKSHEET, OWNER),
            store.reconciliations(WORKSHEET, OWNER),
            verification=verify(store.reconciliations(WORKSHEET, OWNER), fresh))

    def test_it_renders_with_the_reconciler_broken(self, store, monkeypatch):
        persist(store, observed=[observation()], as_of="2026-06-20")
        planned = store.planned_events(WORKSHEET, OWNER)
        observed = store.observed_events(WORKSHEET, OWNER)
        rows = store.reconciliations(WORKSHEET, OWNER)

        import src.mission.rsu_reconcile as engine

        def explode(*args, **kwargs):
            raise AssertionError("the view re-reconciled")

        monkeypatch.setattr(engine, "reconcile", explode)
        monkeypatch.setattr(engine, "_could_match", explode)
        monkeypatch.setattr(engine, "_variances", explode)

        view = RSUReconciliationView.from_records(planned, observed, rows)
        assert len(view.rows) == 1

    def test_it_takes_no_policy_and_no_clock(self):
        """Given neither, it cannot re-decide a status even by accident."""
        import inspect

        parameters = inspect.signature(
            RSUReconciliationView.from_records).parameters
        assert set(parameters) == {"planned_events", "observed_events",
                                   "reconciliations", "verification",
                                   "counterfactuals"}

    def test_it_imports_no_matching_logic(self):
        import ast
        import inspect

        from src.workspace import reconciliation_view

        tree = ast.parse(inspect.getsource(reconciliation_view))
        imported = " ".join(
            getattr(node, "module", "") or ""
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom)))
        for package in ("rsu_reconcile", "datetime", "observation"):
            assert package not in imported


class TestTheLanesKeepTheDistinctions:

    def build(self, store, as_of):
        _, _, fresh = rebuild(store, as_of=as_of)
        return RSUReconciliationView.from_records(
            store.planned_events(WORKSHEET, OWNER),
            store.observed_events(WORKSHEET, OWNER),
            store.reconciliations(WORKSHEET, OWNER),
            verification=verify(store.reconciliations(WORKSHEET, OWNER), fresh))

    def test_pending_never_renders_as_missing(self, store):
        persist(store, as_of="2026-06-10")
        [row] = self.build(store, "2026-06-10").rows
        assert row.verdict == "Not yet due"
        assert "missing" not in row.verdict.lower()
        assert row.observed_summary == "No observation yet"

    def test_overdue_never_renders_as_confirmed_absent(self, store):
        persist(store, as_of="2026-06-25")
        [row] = self.build(store, "2026-06-25").rows
        assert row.verdict == "Overdue — nothing reported yet"
        assert "confirmed" not in row.verdict.lower()

    def test_confirmed_absence_says_so_and_cites_evidence(self, store):
        persist(store, observed=[observation(confirms_absence=True,
                                             evidence_ref="payroll/case-12")],
                as_of="2026-07-30")
        [row] = self.build(store, "2026-07-30").rows
        assert row.verdict == "Confirmed not received"
        assert "payroll/case-12" in row.evidence_refs

    def test_the_report_date_is_not_shown_as_the_event_date(self, store):
        persist(store, observed=[observation(observed_date="2026-07-02",
                                             effective_date="2026-06-15")],
                as_of="2026-07-10")
        [row] = self.build(store, "2026-07-10").rows
        assert row.observed_effective_date == "2026-06-15"
        assert row.observed_reported_date == "2026-07-02"
        assert row.verdict == "As planned"

    def test_a_late_row_shows_both_dates(self, store):
        persist(store, observed=[observation(effective_date="2026-06-19")],
                as_of="2026-06-30")
        [row] = self.build(store, "2026-06-30").rows
        assert row.verdict == "Late"
        assert row.planned_date == "2026-06-15"
        assert row.observed_effective_date == "2026-06-19"

    def test_ambiguous_is_not_shown_as_matched(self, store):
        second = replace(JUNE, event_id="plan-jun-b",
                         expected_date="2026-06-17")
        persist(store, planned=[JUNE, second],
                observed=[observation(effective_date="2026-06-16")],
                as_of="2026-06-30")
        view = self.build(store, "2026-06-30")
        assert all(row.verdict == "Could match more than one plan"
                   for row in view.rows)
        assert all(row.candidates for row in view.rows)

    def test_an_unexpected_row_has_no_planned_side(self, store):
        persist(store, planned=[], observed=[observation()], as_of="2026-06-20")
        [row] = self.build(store, "2026-06-20").rows
        assert row.verdict == "Not in the plan"
        assert row.planned_summary == "Not in the plan"

    def test_counterfactuals_stay_out_of_the_lanes(self, store):
        persist(store, observed=[observation()], as_of="2026-06-20")
        view = RSUReconciliationView.from_records(
            store.planned_events(WORKSHEET, OWNER),
            store.observed_events(WORKSHEET, OWNER),
            store.reconciliations(WORKSHEET, OWNER),
            counterfactuals=({"counterfactual_id": "cf-1",
                              "label": "HYPOTHETICAL — this did not happen"},))
        assert len(view.rows) == 1
        assert view.counterfactuals
        assert "cf-1" not in str([row.to_json() for row in view.rows])


class TestTelemetryRemainsExpendable:

    def test_deleting_traces_leaves_the_history_unchanged(self, store,
                                                          tmp_path):
        from src.telemetry import TraceStore

        traces = tmp_path / "trace.db"
        TraceStore(traces)
        persist(store, observed=[observation()], as_of="2026-06-20")
        before = store.reconciliations(WORKSHEET, OWNER)

        traces.unlink()
        assert store.reconciliations(WORKSHEET, OWNER) == before
