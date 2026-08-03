"""Retention classification, export, deletion, and proof that it happened.

The inventory test reads the tables SQLite actually reports, never the
classification registry. Parametrising it from the registry would let a new
table pass by never appearing — the hole the comparison-profile and
diagnostic-destination guards had to close.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.workspace.erasure import (
    DeletionIncomplete,
    delete_workspace,
    export_workspace,
    owner_reference,
    verify_deleted,
)
from src.workspace.retention import (
    RETENTION_POLICY_VERSION,
    SENSITIVE_CATEGORIES,
    WORKSPACE_RECORDS,
    DataClass,
    DeletionBehaviour,
    OwnerScope,
    owner_scoped_tables,
    unclassified,
)
from src.workspace.store import WorkspaceStore

MINE, THEIRS = "owner-a", "owner-b"


def schema_tables(path) -> set:
    """The independent source of truth."""
    with sqlite3.connect(path) as conn:
        return {name for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'")}


@pytest.fixture
def store(tmp_path):
    return WorkspaceStore(tmp_path / "w.db")


def populate(store, owner: str, *, suffix: str = ""):
    """One of everything, so a deletion has something to miss."""
    from src.mission.rsu_reconcile import (MATCHING_POLICY_VERSION,
                                           ObservedEvent, PlannedEvent,
                                           reconcile)
    from src.workspace.worksheet import create

    plan_id = f"plan-{owner}{suffix}"
    worksheet_id = f"ws-{owner}{suffix}"

    from src.mission.compiler import compile_scenario
    from src.mission.scenario import ScenarioSpecification
    from src.mission.spec import Inference, Provenance

    compiled = compile_scenario(
        "I put $500 into SPY every month in my taxable brokerage and never sell.",
        name=plan_id, version=1,
        benchmark_rule="benchmark-policy/public-default@1")
    provenance = compiled.scenario.provenance
    scenario = ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=provenance.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in provenance.inferred),
            contradictions=provenance.contradictions, unresolved=())})

    store.save_plan(plan_id=plan_id, owner=owner, scenario=scenario,
                    stated_text="seed", saved_at="t0")
    store.record_run(run_id=f"run-{owner}{suffix}", plan_id=plan_id,
                     ran_at="t1",
                     result={"modelling_scope": {"excludes": []},
                             "final_value": 1.0},
                     comparison={})
    store.save_worksheet(create(worksheet_id=worksheet_id, owner_id=owner,
                                scenario_ref=plan_id,
                                primary_run_ref=f"run-{owner}{suffix}",
                                created_at="t0"))
    store.record_confirmation_event(
        event_id=f"ev-{owner}{suffix}", owner=owner, occurred_at="t0",
        kind="confirmed", field="dividends", final_value="reinvested",
        provenance="stated")

    planned = PlannedEvent(event_id=f"pe-{owner}{suffix}", grant_ref="grant/g1",
                           expected_date="2026-06-15", employer_asset="ACME",
                           expected_gross_shares="100.0")
    observed = ObservedEvent(observation_id=f"oe-{owner}{suffix}",
                             observed_date="2026-06-16",
                             effective_date="2026-06-15", grant_ref="grant/g1",
                             employer_asset="ACME", gross_shares="100.0")
    store.record_planned_event(owner=owner, worksheet_id=worksheet_id,
                               event=planned, plan_revision=1, created_at="t0",
                               matching_policy_version=MATCHING_POLICY_VERSION)
    store.record_observed_event(owner=owner, worksheet_id=worksheet_id,
                                event=observed, created_at="t1")
    for row in reconcile([planned], [observed], as_of="2026-06-20"):
        store.record_reconciliation(owner=owner, worksheet_id=worksheet_id,
                                    reconciliation=row)

    from src.workspace.intent import plan as plan_intent

    store.append_worksheet_intent(
        worksheet_id=worksheet_id, owner=owner,
        intent=plan_intent("Add 63-day rolling volatility",
                           intent_id=f"i-{owner}{suffix}", source_revision=1),
        created_at="t0", planner_version="1", instruction_hash="h")
    return plan_id, worksheet_id


class TestTheInventoryIsCheckedAgainstTheSchema:

    def test_every_table_the_schema_reports_is_classified(self, store,
                                                          tmp_path):
        missing = unclassified(sorted(schema_tables(tmp_path / "w.db")))
        assert not missing, (
            f"these tables have no retention classification: {missing}")

    def test_a_new_unclassified_table_fails(self, store, tmp_path):
        """The falsification: a table added without a classification."""
        with sqlite3.connect(tmp_path / "w.db") as conn:
            conn.execute("CREATE TABLE scratch_notes "
                         "(owner TEXT, note TEXT)")

        assert unclassified(sorted(schema_tables(tmp_path / "w.db"))) == \
            ("scratch_notes",)

    def test_the_registry_is_not_the_source_of_truth(self):
        """Guards the guard: `unclassified` must compare against something
        outside itself."""
        import inspect

        from src.workspace import retention

        source = inspect.getsource(retention.unclassified)
        assert "tables" in inspect.signature(retention.unclassified).parameters

    def test_every_class_states_all_its_fields(self):
        for record in WORKSPACE_RECORDS.values():
            assert record.retention_policy
            assert record.export_behaviour
            assert isinstance(record.contains_sensitive_financial_data, bool)
            assert isinstance(record.contains_model_content, bool)

    def test_every_indirect_table_declares_an_executable_path(self):
        """Executable metadata, not prose. An ownership rule written in a
        comment is one every consumer re-derives differently."""
        for record in WORKSPACE_RECORDS.values():
            if record.owner_scope is OwnerScope.INDIRECT:
                assert record.ownership_path is not None, record.table
                assert "?" in record.ownership_path.select(record.table)
                assert "?" in record.ownership_path.delete(record.table)

    def test_every_direct_table_names_its_owner_column(self):
        for record in WORKSPACE_RECORDS.values():
            if record.owner_scope is OwnerScope.DIRECT:
                assert record.owner_column, record.table

    def test_no_table_is_scoped_by_assumption(self):
        """Global, direct, or indirect with a path. There is no fourth."""
        for record in WORKSPACE_RECORDS.values():
            if record.owner_scope is OwnerScope.DIRECT:
                assert record.owner_column
            elif record.owner_scope is OwnerScope.INDIRECT:
                assert record.ownership_path
            else:
                assert record.deletion_behaviour is not \
                    DeletionBehaviour.DELETE_WITH_OWNER

    def test_the_deletion_code_special_cases_no_table(self):
        """A new indirectly-owned table must be reached by declaring its path,
        not by editing the deletion function — which is the edit everyone
        forgets."""
        import inspect

        from src.workspace import erasure

        source = inspect.getsource(erasure._rows_for)
        assert "plan_run" not in source

    def test_no_production_table_is_indirectly_owned(self):
        """`plan_run` was the last one, and it has its own owner now.

        An indirect table is one every consumer must remember to join through:
        deletion, export, tenant isolation and auditing each get it right
        separately or not at all. Where a real owner column is possible, it is
        strictly better — so the assertion is that none remain, not that the
        machinery is unused.

        `OwnerScope.INDIRECT` still exists and is still exercised, against a
        table `tests/ownership_fixture.py` owns for the purpose. Holding a
        domain table in a weaker shape to keep that path covered would be
        paying for the test in production.
        """
        indirect = [one.table for one in WORKSPACE_RECORDS.values()
                    if one.owner_scope is OwnerScope.INDIRECT]
        assert indirect == [], (
            f"{indirect} are reachable only through a parent; give them an "
            "owner column or record why they cannot have one")

    def test_every_indirect_scope_still_has_to_name_its_path(self):
        """The rule survives its last production instance disappearing.

        Declaring INDIRECT without a path is what makes a deletion silently
        skip a table, so it is refused whether or not anything uses it today.
        """
        from tests.ownership_fixture import INDIRECT_CHILD

        assert INDIRECT_CHILD.reached_through is not None
        assert "worksheet_proposal.owner" in INDIRECT_CHILD.reached_through

    def test_sensitive_categories_are_named(self):
        for category in ("employer name or ticker", "holdings",
                         "raw user instructions",
                         "model prompts and responses"):
            assert category in SENSITIVE_CATEGORIES


class TestExport:

    def test_it_returns_every_owner_scoped_table(self, store):
        populate(store, MINE)
        payload = export_workspace(store, MINE)
        assert set(payload["tables"]) == {
            one.table for one in owner_scoped_tables()}

    def test_it_includes_the_indirectly_scoped_runs(self, store):
        populate(store, MINE)
        assert export_workspace(store, MINE)["counts"]["plan_run"] == 1

    def test_it_carries_no_raw_owner_identifier(self, store):
        populate(store, MINE)
        assert export_workspace(store, MINE)["owner_reference"] != MINE

    def test_another_owners_records_are_absent(self, store):
        populate(store, MINE)
        populate(store, THEIRS)
        payload = export_workspace(store, THEIRS)
        assert all(row.get("owner", THEIRS) == THEIRS
                   for rows in payload["tables"].values() for row in rows
                   if "owner" in row)


class TestDeletion:

    def test_it_removes_every_classified_table(self, store):
        populate(store, MINE)
        receipt = delete_workspace(store, MINE, requested_at="t9")
        assert receipt.status == "COMPLETE"
        assert verify_deleted(store, MINE) == {}

    def test_it_reaches_the_indirectly_scoped_runs(self, store):
        """A deletion written around `WHERE owner = ?` leaves every run behind
        and reports success."""
        populate(store, MINE)
        assert export_workspace(store, MINE)["counts"]["plan_run"] == 1
        delete_workspace(store, MINE, requested_at="t9")
        assert "plan_run" not in verify_deleted(store, MINE)

    def test_verification_is_independent_of_the_deletion(self, store,
                                                         monkeypatch):
        """The falsification: skip one table and prove verification catches it.

        Verification derived from the deletion code would agree with it by
        construction."""
        populate(store, MINE)

        import src.workspace.erasure as erasure

        original = erasure.owner_scoped_tables
        monkeypatch.setattr(
            erasure, "owner_scoped_tables",
            lambda: tuple(one for one in original()
                          if one.table != "worksheet_intent"))
        erasure.delete_workspace(store, MINE, requested_at="t9")

        monkeypatch.setattr(erasure, "owner_scoped_tables", original)
        assert "worksheet_intent" in erasure.verify_deleted(store, MINE)

    def test_a_deletion_that_removes_nothing_raises(self, store):
        """A delete that silently missed everything looks exactly like one that
        worked, unless something re-reads afterwards."""
        import src.workspace.erasure as erasure

        populate(store, MINE)

        class DeletesNothing:
            """A store whose writes do not land."""

            def __init__(self, real):
                self._real = real

            def _conn(self):
                from contextlib import contextmanager

                @contextmanager
                def scope():
                    with self._real._conn() as conn:
                        yield _SwallowDeletes(conn)

                return scope()

        class _SwallowDeletes:
            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *args, **kwargs):
                if sql.strip().upper().startswith("DELETE"):
                    return self._conn.execute("SELECT 1")
                return self._conn.execute(sql, *args, **kwargs)

        with pytest.raises(DeletionIncomplete, match="left rows"):
            erasure.delete_workspace(DeletesNothing(store), MINE,
                                     requested_at="t9")

    def test_the_records_survive_a_failed_deletion(self, store):
        """A partial deletion must not leave a user half-erased and reported
        complete."""
        populate(store, MINE)
        before = export_workspace(store, MINE)["counts"]

        with pytest.raises(DeletionIncomplete):
            delete_workspace(_NoOpDeletes(store), MINE, requested_at="t9")

        assert export_workspace(store, MINE)["counts"] == before


class _NoOpDeletes:
    """A store wrapper whose DELETE statements do nothing."""

    def __init__(self, real):
        self._real = real

    def _conn(self):
        from contextlib import contextmanager

        @contextmanager
        def scope():
            with self._real._conn() as conn:
                yield _Swallow(conn)

        return scope()


class _Swallow:
    def __init__(self, conn):
        self._inner = conn

    def execute(self, sql, *args, **kwargs):
        if sql.strip().upper().startswith("DELETE"):
            return self._inner.execute("SELECT 1")
        return self._inner.execute(sql, *args, **kwargs)


class TestCrossTenantIsolation:

    def test_deleting_one_owner_leaves_the_other_intact(self, store):
        populate(store, MINE)
        populate(store, THEIRS)

        delete_workspace(store, MINE, requested_at="t9")

        assert verify_deleted(store, MINE) == {}
        surviving = export_workspace(store, THEIRS)
        assert surviving["counts"]["plan"] == 1
        assert surviving["counts"]["worksheet_intent"] == 1
        assert surviving["counts"]["plan_run"] == 1

    def test_identically_named_worksheets_are_not_confused(self, store):
        from src.workspace.worksheet import create

        for owner in (MINE, THEIRS):
            store.save_worksheet(create(worksheet_id="ws-shared",
                                        owner_id=owner, scenario_ref="p",
                                        primary_run_ref="r", created_at="t0"))
        delete_workspace(store, MINE, requested_at="t9")
        assert store.get_worksheet("ws-shared", THEIRS) is not None
        assert store.get_worksheet("ws-shared", MINE) is None


class TestTheReceipt:

    def test_it_names_no_owner_and_no_content(self, store):
        populate(store, MINE)
        receipt = delete_workspace(store, MINE, requested_at="t9")
        rendered = str(receipt.to_json())

        assert MINE not in rendered
        assert "SPY" not in rendered
        assert "ACME" not in rendered

    def test_the_owner_reference_is_stable_and_irreversible(self):
        assert owner_reference(MINE) == owner_reference(MINE)
        assert owner_reference(MINE) != owner_reference(THEIRS)
        assert MINE not in owner_reference(MINE)

    def test_it_records_counts_and_the_policy_version(self, store):
        populate(store, MINE)
        receipt = delete_workspace(store, MINE, requested_at="t9")
        assert receipt.counts["plan"] == 1
        assert receipt.policy_version == RETENTION_POLICY_VERSION


class TestTracesAreIndependent:

    def test_workspace_export_survives_the_trace_store_being_deleted(
            self, store, tmp_path):
        from src.telemetry import TraceStore

        traces = tmp_path / "trace.db"
        TraceStore(traces)
        populate(store, MINE)
        before = export_workspace(store, MINE)["counts"]

        traces.unlink()
        assert export_workspace(store, MINE)["counts"] == before

    def test_deleting_traces_leaves_the_workspace_intact(self, store,
                                                         tmp_path):
        from src.telemetry import TraceStore

        traces = TraceStore(tmp_path / "trace.db")
        traces.start_trace(trace_id="t1", conversation_id="c",
                           request_id="r", tenant=MINE, started_at="t0")
        populate(store, MINE)

        traces.purge_tenant(MINE)
        assert export_workspace(store, MINE)["counts"]["plan"] == 1

    def test_a_workspace_deletion_does_not_require_the_trace_store(self, store):
        populate(store, MINE)
        assert delete_workspace(store, MINE,
                                requested_at="t9").status == "COMPLETE"
