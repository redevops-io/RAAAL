"""Moving a workspace from SQLite to PostgreSQL, and proving it arrived.

The bundle is neither engine's representation. SQLite holds `"152.26"` as text
and PostgreSQL holds `152.260000000000` as NUMERIC, so a byte comparison of the
two would report data loss where there is none, and a copy of either spelling
would make the bundle a dialect artifact.

Every required failure is constructed rather than hoped for, and each one must
leave the target unchanged — a migration that half-applies is worse than one
that refuses, because the target then holds a history nobody wrote.
"""
from __future__ import annotations

from tests.market_fixture import NO_MARKET_DATA
import copy
import os
from decimal import Decimal

import pytest

from src.db.engine import Database
from src.db.transfer import (
    BUNDLE_FORMAT_VERSION,
    BundleUnreadable,
    ExportRefused,
    ImportRefused,
    apply_import,
    digest_of,
    export_bundle,
    plan_import,
    verify_bundle,
    verify_import,
)
from src.db.types import Json
from src.mission.rsu_reconcile import (
    EventReconciliation,
    ObservedEvent,
    PlannedEvent,
    ReconciliationStatus,
)
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set QUANTIFY_TEST_POSTGRES_URL; the import half of this is a "
           "PostgreSQL-only guarantee")

A, B = "alice", "bob"
RESULT = {"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}, "final_value": 1.0, "market_data": NO_MARKET_DATA.to_json()}
AT = "2026-01-01T00:00:00Z"


def scenario_for(name="p-1"):
    from src.mission.compiler import compile_scenario
    from src.mission.scenario import ScenarioSpecification
    from src.mission.spec import Inference, Provenance

    compiled = compile_scenario(
        "I put $2,000 into SPY every month in my Roth IRA, on the first trading "
        "day of the period, reinvesting the dividends, and I never sell.",
        name=name, version=1,
        benchmark_rule="benchmark-policy/public-default@1")
    provenance = compiled.scenario.provenance
    return ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=provenance.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in provenance.inferred),
            contradictions=provenance.contradictions, unresolved=())})


def populate(store, owner):
    """Both tenants use identical ids, so the transfer has to keep them apart."""
    store.save_plan(plan_id="p-1", owner=owner, scenario=scenario_for(),
                    stated_text="seed", saved_at=AT)
    store.record_run(run_id="r-1", plan_id="p-1", ran_at=AT, result=RESULT,
                     comparison={}, owner=owner)
    store.save_worksheet(create(worksheet_id="ws-1", owner_id=owner,
                                scenario_ref="p-1", primary_run_ref="r-1",
                                created_at=AT))
    store.record_planned_event(
        owner=owner, worksheet_id="ws-1",
        event=PlannedEvent(event_id="pe-1", grant_ref="grant/g1",
                           expected_date="2026-06-15", employer_asset="ACME",
                           expected_gross_shares="152.26",
                           expected_value="3896.10"),
        plan_revision=1, created_at=AT, matching_policy_version="m@1")
    store.record_observed_event(
        owner=owner, worksheet_id="ws-1",
        event=ObservedEvent(observation_id="oe-1", observed_date="2026-06-16",
                            effective_date="2026-06-15", grant_ref="grant/g1",
                            employer_asset="ACME", gross_shares="152.26",
                            value="3896.10"),
        created_at=AT)
    store.record_reconciliation(
        owner=owner, worksheet_id="ws-1",
        reconciliation=EventReconciliation(
            reconciliation_id="rc-1", status=ReconciliationStatus.MATCHED,
            planned_ref="pe-1", observed_ref="oe-1",
            derived_at="2026-06-17T00:00:00Z"))


@pytest.fixture
def source(tmp_path):
    store = WorkspaceStore(tmp_path / "source.db")
    populate(store, A)
    populate(store, B)
    return store


@pytest.fixture
def target():
    from sqlalchemy import text

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    database.create_all()
    return database


@pytest.fixture
def bundle(source):
    return export_bundle(source, exported_at=AT, commit="abc123")


def rows(target, sql, params=()):
    conn = target.connect()
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


class TestTheBundleIsNeutral:
    def test_decimals_are_canonical_strings_not_floats(self, bundle):
        record = bundle["records"]["planned_event"][0]
        assert record["expected_quantity"] == "152.26"
        assert isinstance(record["expected_quantity"], str)

    def test_json_is_an_object_not_encoded_text(self, bundle):
        record = bundle["records"]["planned_event"][0]
        assert isinstance(record["payload"], dict)

    def test_the_manifest_records_what_it_was_taken_from(self, bundle):
        manifest = bundle["manifest"]
        assert manifest["format_version"] == BUNDLE_FORMAT_VERSION
        assert manifest["source_commit"] == "abc123"
        assert manifest["source_schema_version"]
        assert manifest["exported_at"] == AT
        assert manifest["canonicalization"]["decimal"]
        assert manifest["canonicalization"]["temporal"]

    def test_it_carries_counts_and_digests_per_table(self, bundle):
        assert bundle["manifest"]["counts"]["worksheet"] == 2
        assert bundle["manifest"]["digests"]["worksheet"]
        assert bundle["manifest"]["bundle_digest"]

    def test_the_digest_does_not_depend_on_row_order(self, bundle):
        forward = bundle["records"]["planned_event"]
        assert digest_of(forward) == digest_of(list(reversed(forward)))

    def test_both_tenants_are_present_with_identical_ids(self, bundle):
        owners = {row["owner"] for row in bundle["records"]["worksheet"]}
        assert owners == {A, B}
        assert {row["worksheet_id"] for row in bundle["records"]["worksheet"]} \
            == {"ws-1"}


class TestTheRoundTrip:
    def test_import_then_verify(self, bundle, target):
        apply_import(target, bundle)
        assert verify_import(target, bundle) == []

    def test_every_tenant_identity_survives(self, bundle, target):
        apply_import(target, bundle)
        for table, column in (("plan", "plan_id"), ("plan_run", "run_id"),
                              ("worksheet", "worksheet_id"),
                              ("planned_event", "planned_event_id"),
                              ("observed_event", "observed_event_id"),
                              ("event_reconciliation", "reconciliation_id")):
            owners = {r["owner"] for r in
                      rows(target, f"SELECT owner FROM {table}")}
            assert owners == {A, B}, table

    def test_decimals_arrive_as_the_same_quantity(self, bundle, target):
        """Not the same spelling — PostgreSQL pads NUMERIC to its scale."""
        apply_import(target, bundle)
        stored = rows(target, "SELECT expected_quantity FROM planned_event "
                              "WHERE owner = ?", (A,))[0]["expected_quantity"]
        assert stored == Decimal("152.26")
        assert str(stored) != "152.26"      # the spelling genuinely differs

    def test_content_hashes_survive(self, bundle, target):
        from src.workspace.store import verify_content_hashes

        apply_import(target, bundle)
        store = WorkspaceStore(POSTGRES_URL)
        for table in ("planned_event", "observed_event",
                      "event_reconciliation"):
            assert verify_content_hashes(store, table) == []

    def test_decimal_mirrors_survive(self, bundle, target):
        from src.workspace.store import verify_decimal_columns

        apply_import(target, bundle)
        store = WorkspaceStore(POSTGRES_URL)
        assert verify_decimal_columns(store, "planned_event") == []
        assert verify_decimal_columns(store, "observed_event") == []

    def test_ownership_relationships_survive(self, bundle, target):
        """A run must still belong to its own tenant's plan."""
        apply_import(target, bundle)
        for owner in (A, B):
            run = rows(target, "SELECT plan_id FROM plan_run WHERE owner = ?",
                       (owner,))[0]
            plan = rows(target, "SELECT plan_id FROM plan WHERE owner = ? "
                                "AND plan_id = ?", (owner, run["plan_id"]))
            assert plan, f"{owner}'s run points at no plan of theirs"

    def test_a_second_import_is_redelivery(self, bundle, target):
        apply_import(target, bundle)
        plan = plan_import(target, bundle)
        assert plan.ready == 0
        assert plan.redelivered > 0
        assert plan.conflicts == []

    def test_one_tenant_can_be_moved_alone(self, source, target):
        narrowed = export_bundle(source, exported_at=AT, owner=A)
        apply_import(target, narrowed)
        assert {r["owner"] for r in rows(target, "SELECT owner FROM worksheet")} \
            == {A}


class TestExportRefusesCorruption:
    """A migration must not copy corruption and call it success."""

    def test_a_tampered_content_hash_stops_the_export(self, source):
        with source._conn() as conn:
            conn.execute("UPDATE planned_event SET content_hash = ?",
                         ("forged",))
        with pytest.raises(ExportRefused, match="content hash"):
            export_bundle(source, exported_at=AT)

    def test_a_disagreeing_decimal_mirror_stops_the_export(self, source):
        with source._conn() as conn:
            conn.execute("UPDATE planned_event SET expected_quantity = ?",
                         ("999.99",))
        with pytest.raises(ExportRefused, match="disagrees with payload"):
            export_bundle(source, exported_at=AT)

    def test_an_unclassified_table_stops_the_export(self, source, monkeypatch):
        from src.workspace import retention

        reduced = {name: record
                   for name, record in retention.WORKSPACE_RECORDS.items()
                   if name != "confirmation_event"}
        monkeypatch.setattr(retention, "WORKSPACE_RECORDS", reduced)
        with pytest.raises(ExportRefused, match="unclassified"):
            export_bundle(source, exported_at=AT)


class TestTheBundleIsCheckedBeforeItIsTrusted:
    def test_an_unknown_format_version_is_refused(self, bundle):
        bundle["manifest"]["format_version"] = "quantify-transfer@99"
        with pytest.raises(BundleUnreadable, match="format"):
            verify_bundle(bundle)

    def test_a_changed_count_is_refused(self, bundle):
        bundle["manifest"]["counts"]["worksheet"] = 99
        with pytest.raises(BundleUnreadable, match="manifest says"):
            verify_bundle(bundle)

    def test_an_edited_record_is_refused(self, bundle):
        bundle["records"]["worksheet"][0]["canonical_hash"] = "forged"
        with pytest.raises(BundleUnreadable, match="digest"):
            verify_bundle(bundle)

    def test_an_edited_digest_is_refused(self, bundle):
        """Editing the record *and* its digest must still fail, because the
        bundle digest covers the digests."""
        bundle["records"]["worksheet"][0]["canonical_hash"] = "forged"
        bundle["manifest"]["digests"]["worksheet"] = digest_of(
            bundle["records"]["worksheet"])
        with pytest.raises(BundleUnreadable, match="bundle digest"):
            verify_bundle(bundle)


class TestImportRefusesAndLeavesNothing:
    def test_a_divergent_body_is_a_conflict(self, bundle, target):
        apply_import(target, bundle)
        diverged = copy.deepcopy(bundle)
        record = diverged["records"]["worksheet"][0]
        record["canonical_hash"] = "different"
        diverged["manifest"]["digests"]["worksheet"] = digest_of(
            diverged["records"]["worksheet"])
        diverged["manifest"]["bundle_digest"] = digest_of(
            [diverged["manifest"]["digests"]])

        plan = plan_import(target, diverged)
        assert plan.conflicts
        with pytest.raises(ImportRefused, match="different body"):
            apply_import(target, diverged, plan)

    def test_the_target_is_unchanged_after_a_refusal(self, bundle, target):
        apply_import(target, bundle)
        before = rows(target, "SELECT canonical_hash FROM worksheet "
                              "ORDER BY owner")
        diverged = copy.deepcopy(bundle)
        diverged["records"]["worksheet"][0]["canonical_hash"] = "different"
        diverged["manifest"]["digests"]["worksheet"] = digest_of(
            diverged["records"]["worksheet"])
        diverged["manifest"]["bundle_digest"] = digest_of(
            [diverged["manifest"]["digests"]])
        with pytest.raises(ImportRefused):
            apply_import(target, diverged)
        assert rows(target, "SELECT canonical_hash FROM worksheet "
                            "ORDER BY owner") == before

    def test_a_failure_partway_writes_nothing(self, bundle, target):
        """A half-applied migration leaves a history nobody wrote."""
        broken = copy.deepcopy(bundle)
        # A run pointing at a plan that is not in the bundle: the foreign key
        # refuses it, and everything inserted before it must go too.
        orphan = dict(broken["records"]["plan_run"][0])
        orphan["run_id"] = "r-orphan"
        orphan["plan_id"] = "p-missing"
        broken["records"]["plan_run"].append(orphan)
        broken["manifest"]["counts"]["plan_run"] += 1
        broken["manifest"]["digests"]["plan_run"] = digest_of(
            broken["records"]["plan_run"])
        broken["manifest"]["bundle_digest"] = digest_of(
            [broken["manifest"]["digests"]])

        with pytest.raises(Exception):
            apply_import(target, broken)
        assert rows(target, "SELECT COUNT(*) AS n FROM worksheet")[0]["n"] == 0
        assert rows(target, "SELECT COUNT(*) AS n FROM plan")[0]["n"] == 0

    def test_a_bundle_for_an_unknown_table_is_refused(self, bundle, target):
        bundle["manifest"]["tables"] = list(bundle["manifest"]["tables"]) + \
            ["not_a_table"]
        bundle["records"]["not_a_table"] = [{"id": 1}]
        bundle["manifest"]["counts"]["not_a_table"] = 1
        bundle["manifest"]["digests"]["not_a_table"] = digest_of([{"id": 1}])
        bundle["manifest"]["bundle_digest"] = digest_of(
            [bundle["manifest"]["digests"]])
        with pytest.raises(ImportRefused, match="does not have"):
            apply_import(target, bundle)


class TestVerificationIsSemanticNotPhysical:
    def test_it_passes_despite_different_spellings(self, bundle, target):
        """The whole reason it compares canonical records: the two engines
        genuinely store these values differently."""
        apply_import(target, bundle)
        assert verify_import(target, bundle) == []

    def test_it_detects_a_missing_row(self, bundle, target):
        apply_import(target, bundle)
        conn = target.connect()
        try:
            conn.execute("DELETE FROM event_reconciliation WHERE owner = ?",
                         (A,))
            conn.commit()
        finally:
            conn.close()
        problems = verify_import(target, bundle)
        assert any("event_reconciliation" in one for one in problems)

    def test_it_detects_a_changed_value(self, bundle, target):
        apply_import(target, bundle)
        conn = target.connect()
        try:
            conn.execute("UPDATE worksheet SET canonical_hash = ? "
                         "WHERE owner = ?", ("changed", A))
            conn.commit()
        finally:
            conn.close()
        problems = verify_import(target, bundle)
        assert any("worksheet" in one for one in problems)


class TestTheDryRunAndTheImportAgree:
    def test_the_plan_is_reused_rather_than_recomputed(self, bundle, target):
        """A dry run that reported one thing while the import did another
        would defeat the only purpose a dry run has."""
        plan = plan_import(target, bundle)
        expected = plan.ready
        applied = apply_import(target, bundle, plan)
        assert applied is plan
        assert rows(target, "SELECT COUNT(*) AS n FROM worksheet")[0]["n"] == 2
        assert expected == plan.ready

    def test_a_dry_run_writes_nothing(self, bundle, target):
        plan_import(target, bundle)
        assert rows(target, "SELECT COUNT(*) AS n FROM worksheet")[0]["n"] == 0

    def test_the_summary_reports_each_category(self, bundle, target):
        summary = plan_import(target, bundle).summary()
        assert set(summary) == {"rows_ready", "redeliveries", "conflicts",
                                "unknown_tables"}
        assert summary["rows_ready"] > 0


class TestTheImportedDatabaseStandsAlone:
    """The exit criterion: a fresh process, historical journeys, no SQLite.

    Verification against the bundle proves the rows arrived. It does not prove
    the application can *use* them — a migration can preserve every byte and
    still land data the running system cannot read back through its own
    surfaces. So these go through the store's public methods only, against a
    connection that has never seen the source file.
    """

    @pytest.fixture
    def migrated(self, bundle, target):
        apply_import(target, bundle)
        assert verify_import(target, bundle) == []
        return WorkspaceStore(POSTGRES_URL)

    def test_the_source_file_is_not_consulted(self, migrated, source):
        """The store under test is pointed at PostgreSQL and nothing else."""
        assert migrated.db.dialect.value == "postgresql"
        assert migrated.path is None

    def test_a_worksheet_reads_back_through_the_public_surface(self, migrated):
        for owner in (A, B):
            worksheet = migrated.get_worksheet("ws-1", owner)
            assert worksheet["revision"] == 1
            assert worksheet["payload"]["scenario_ref"] == "p-1"

    def test_a_plan_and_its_run_read_back(self, migrated):
        for owner in (A, B):
            assert migrated.get_plan("p-1", owner)["plan_id"] == "p-1"
            run = migrated.get_run("r-1", owner)
            assert run["result"]["final_value"] == 1.0
            assert [r["run_id"] for r in migrated.runs_for("p-1", owner)] == ["r-1"]

    def test_the_tracking_history_reads_back_as_decimals(self, migrated):
        for owner in (A, B):
            planned = migrated.planned_events("ws-1", owner)[0]
            assert planned["expected_quantity"] == Decimal("152.26")
            observed = migrated.observed_events("ws-1", owner)[0]
            assert observed["quantity"] == Decimal("152.26")

    def test_the_reconciliation_still_verifies_against_its_evidence(self,
                                                                   migrated):
        """A derived claim that no longer follows from its inputs would be a
        migration that moved rows and lost meaning."""
        from src.mission.rsu_reconcile import reconcile
        from src.workspace import reconciliation_view

        for owner in (A, B):
            stored = migrated.reconciliations("ws-1", owner)
            fresh = reconcile(
                [PlannedEvent(event_id="pe-1", grant_ref="grant/g1",
                              expected_date="2026-06-15",
                              employer_asset="ACME",
                              expected_gross_shares="152.26",
                              expected_value="3896.10")],
                [ObservedEvent(observation_id="oe-1",
                               observed_date="2026-06-16",
                               effective_date="2026-06-15",
                               grant_ref="grant/g1", employer_asset="ACME",
                               gross_shares="152.26", value="3896.10")],
                as_of="2026-06-20")
            states = reconciliation_view.verify(stored, fresh)
            assert all(state.value == "VERIFIED" for state in states.values()), \
                states

    def test_tenants_are_still_isolated_after_the_move(self, migrated):
        assert migrated.get_worksheet("ws-1", "carol") is None
        assert migrated.get_run("r-1", "carol") is None
        assert migrated.list_plans(A)[0]["plan_id"] == "p-1"
        assert len(migrated.list_plans(A)) == 1

    def test_an_export_of_the_migrated_database_matches_the_bundle(self,
                                                                   migrated,
                                                                   bundle):
        """Round trip closed: exporting the target reproduces the digests the
        source produced, through a different engine."""
        again = export_bundle(migrated, exported_at=AT, commit="abc123")
        assert again["manifest"]["digests"] == bundle["manifest"]["digests"]
        assert again["manifest"]["bundle_digest"] == \
            bundle["manifest"]["bundle_digest"]

    def test_new_work_can_continue_on_the_migrated_database(self, migrated):
        """The database is not a museum piece — it has to accept the next
        revision."""
        from src.workspace.worksheet import from_json, revise

        current = from_json(migrated.get_worksheet("ws-1", A)["payload"])
        migrated.save_worksheet(revise(current, reason="after the migration",
                                       created_at="2026-08-01T00:00:00Z"))
        assert migrated.get_worksheet("ws-1", A)["revision"] == 2
        assert migrated.get_worksheet("ws-1", B)["revision"] == 1
