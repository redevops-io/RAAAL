"""No persisted market-derived figure outside the provenance graph.

The reader inventory closed one class: no production request obtains market
data outside the shared gate. This closes the other. Both were needed for the
same reason — fixing the instance that was found is not closing the class, and
the difference is invisible until something enumerates the class.

Two completeness checks, because they fail differently:

    structural   every persisted type is classified — catches a new result type
                 nobody considered
    instance     every actual stored market-derived row has a usable provenance
                 chain — catches a correctly classified producer that failed to
                 populate its field in one branch

The type inventory comes from the schema, not from this registry. Parametrising
the check from the registry would let a new table pass by never appearing, which
is the hole the comparison-profile and diagnostic-destination guards had to
close and the one the reader inventory closed last week.
"""
from __future__ import annotations

import os

import pytest

from src.db.schema import metadata
from src.market_data.producers import (
    PRODUCERS,
    PricingBasis,
    Producer,
    ProvenanceOwnership,
    dig,
    direct_producers,
    referencing_producers,
    unclassified,
)
from src.market_data.provenance import (
    AccessDecision,
    ProvenanceStatus,
    from_json,
    verify,
)

POLICY = "PILOT_DATA_POLICY"


class TestStructuralCompleteness:
    """Every persisted type is classified, from the schema rather than here."""

    def test_no_table_is_unclassified(self):
        missing = unclassified(sorted(metadata.tables))
        assert missing == (), (
            f"these are persisted and not classified as producers: {missing}. "
            "Omission is not NOT_APPLICABLE — decide which it is")

    def test_the_inventory_reads_the_schema_not_the_registry(self):
        """Parametrising from the registry would let a new table pass by never
        appearing."""
        assert set(metadata.tables) == set(PRODUCERS), (
            f"schema-only: {set(metadata.tables) - set(PRODUCERS)}; "
            f"registry-only: {set(PRODUCERS) - set(metadata.tables)}")

    def test_every_classification_records_why(self):
        for producer in PRODUCERS.values():
            assert producer.reason.strip(), producer.table

    def test_direct_producers_name_where_provenance_lives(self):
        for producer in direct_producers():
            assert producer.provenance_path, (
                f"{producer.table} is DIRECT and does not say where its "
                "provenance sits")

    def test_referencing_producers_name_what_they_cite(self):
        for producer in referencing_producers():
            assert producer.reference_path and producer.reference_table, (
                f"{producer.table} is REFERENCED and does not say what it "
                "cites")

    def test_every_reference_target_is_itself_classified(self):
        """A chain that ends outside the graph is not a chain."""
        for producer in referencing_producers():
            assert producer.reference_table in PRODUCERS
            target = PRODUCERS[producer.reference_table]
            assert target.ownership is ProvenanceOwnership.DIRECT, (
                f"{producer.table} cites {target.table}, which is "
                f"{target.ownership.value} rather than DIRECT — the chain "
                "does not terminate in a provenance record")

    def test_at_least_one_producer_is_direct(self):
        """A registry where nothing holds provenance would satisfy everything
        above."""
        assert direct_producers()


class TestThePricingBasisDistinction:
    """A priced field does not make a type market-derived."""

    def test_reported_values_are_not_classified_as_market_derived(self):
        """An observed vest carries a figure payroll reported. Calling it
        market-derived would make it appear to depend on a snapshot it never
        touched."""
        for table in ("observed_event", "planned_event",
                      "event_reconciliation"):
            assert PRODUCERS[table].ownership is \
                ProvenanceOwnership.NOT_APPLICABLE

    def test_those_types_declare_where_a_basis_would_be_stated(self):
        """So the day one prices a difference, the record says so rather than
        the whole type silently becoming market-derived."""
        for table in ("observed_event", "planned_event",
                      "event_reconciliation"):
            assert PRODUCERS[table].pricing_basis_path, table

    def test_the_three_bases_are_distinguishable(self):
        assert PricingBasis.OBSERVED_VALUE is not PricingBasis.MARKET_SNAPSHOT
        assert PricingBasis.NOT_APPLICABLE is not PricingBasis.OBSERVED_VALUE


class TestInstanceCompleteness:
    """Every stored market-derived row has a usable chain."""

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        from src.workspace.store import WorkspaceStore

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        return WorkspaceStore(tmp_path / "w.db")

    def scenario(self, name="p-1"):
        from src.mission.compiler import compile_scenario
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance

        compiled = compile_scenario(
            "I put $2,000 into SPY every month in my Roth IRA, on the first "
            "trading day of the period, reinvesting the dividends, and I never "
            "sell.", name=name, version=1,
            benchmark_rule="benchmark-policy/public-default@1")
        p = compiled.scenario.provenance
        return ScenarioSpecification(**{
            **compiled.scenario.__dict__,
            "provenance": Provenance(
                stated=p.stated,
                inferred=tuple(Inference(i.field, i.value, i.why,
                                         confirmed=True) for i in p.inferred),
                contradictions=p.contradictions, unresolved=())})

    def stored_run(self, store, provenance_payload):
        """Write a run, going under the store's guard when it has to.

        `record_run` now refuses a result with no provenance or an incoherent
        one, which is the point — but these tests are about the *verifier*
        reading rows that already exist, including rows the writer would never
        have produced. A legacy row predates the guard; a tampered one went
        around it. Both have to be constructed below the API, exactly as the
        JSONB tampering tests do.
        """
        from src.db.types import Json

        store.save_plan(plan_id="p-1", owner="alice", scenario=self.scenario(),
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        result = {"modelling_scope": {"excludes": []}, "final_value": 1.0}
        if provenance_payload is not None:
            result["market_data"] = provenance_payload

        with store._conn() as conn:
            conn.execute(
                "INSERT INTO plan_run (owner, run_id, plan_id, ran_at, result, "
                "comparison) VALUES (?,?,?,?,?,?)",
                ("alice", "r-1", "p-1", "2026-01-01T00:00:00Z",
                 Json(result), Json({})))
        return store.get_run("r-1", "alice")

    def stored_through_the_writer(self, store, provenance_payload):
        """The same run, through `record_run`, for the cases it accepts."""
        store.save_plan(plan_id="p-1", owner="alice", scenario=self.scenario(),
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        store.record_run(
            run_id="r-1", plan_id="p-1", owner="alice",
            ran_at="2026-01-01T00:00:00Z",
            result={"modelling_scope": {"excludes": []}, "final_value": 1.0,
                    "market_data": provenance_payload},
            comparison={})
        return store.get_run("r-1", "alice")

    def test_a_run_with_recorded_provenance_has_a_usable_chain(self, store):
        from src.market_data.access import resolve

        _, provenance = resolve(context="a run",
                                accessed_at="2026-01-01T00:00:00Z")
        run = self.stored_through_the_writer(store, provenance.to_json())

        stored = dig(run["result"], "market_data")
        assert stored is not None
        assert verify(stored) == ()
        assert from_json(stored).identifies_data

    def test_a_run_without_provenance_reads_as_not_recorded(self, store):
        """A legacy row. Explicit, and it must not acquire today's snapshot."""
        run = self.stored_run(store, None)
        stored = dig(run["result"], "market_data")
        assert stored is None
        assert from_json(stored).status is ProvenanceStatus.NOT_RECORDED
        assert from_json(stored).snapshot_id is None

    def test_a_run_stamped_with_a_denial_fails_verification(self, store):
        """A denied read cannot authorize a stored result."""
        run = self.stored_run(store, {
            "status": "RECORDED", "snapshot_id": "s-1",
            "content_digest": "mdv1:aaa", "access_decision": "DENIED",
            "accessed_at": "2026-01-01T00:00:00Z"})
        problems = verify(dig(run["result"], "market_data"))
        assert any("DENIED" in one for one in problems)

    def test_a_run_with_a_label_but_no_digest_does_not_identify_data(self,
                                                                     store):
        run = self.stored_run(store, {
            "status": "RECORDED", "snapshot_id": "prices-2026-01",
            "access_decision": "SYNTHETIC_ALLOWED",
            "accessed_at": "2026-01-01T00:00:00Z"})
        stored = dig(run["result"], "market_data")
        assert not from_json(stored).identifies_data
        assert any("content digest" in one for one in verify(stored))

    def test_a_worksheet_resolves_through_its_run(self, store):
        """The chain, not a copy. The worksheet holds no provenance of its
        own — following the reference is what produces the answer."""
        from src.workspace.worksheet import create
        from src.market_data.access import resolve

        _, provenance = resolve(context="a run",
                                accessed_at="2026-01-01T00:00:00Z")
        self.stored_run(store, provenance.to_json())
        store.save_worksheet(create(
            worksheet_id="ws-1", owner_id="alice", scenario_ref="p-1",
            primary_run_ref="r-1", created_at="2026-01-01T00:00:00Z"))

        worksheet = store.get_worksheet("ws-1", "alice")
        assert "market_data" not in str(worksheet["payload"]), (
            "the worksheet carries its own provenance copy; one figure now has "
            "two sources of truth")

        run = store.get_run(worksheet["payload"]["primary_run_ref"], "alice")
        resolved = from_json(dig(run["result"], "market_data"))
        assert resolved.identifies_data
        assert resolved.snapshot_id == provenance.snapshot_id


class TestTheChainSurvivesTheDeploymentMoving:
    """run -> worksheet -> export, across a snapshot change.

    The resolver is patched to raise during reopen and export. If any stage
    reaches for market data instead of following the stored reference, it fails
    loudly here rather than quietly returning today's snapshot.
    """

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        from src.workspace.store import WorkspaceStore

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        return WorkspaceStore(tmp_path / "w.db")

    def build(self, store):
        from src.market_data.access import resolve
        from src.workspace.worksheet import create

        inventory = TestInstanceCompleteness()
        frame, provenance = resolve(context="a run",
                                    accessed_at="2026-01-01T00:00:00Z")
        assert frame is not None
        inventory.stored_run(store, provenance.to_json())
        store.save_worksheet(create(
            worksheet_id="ws-1", owner_id="alice", scenario_ref="p-1",
            primary_run_ref="r-1", created_at="2026-01-01T00:00:00Z"))
        return provenance

    def forbid_market_data(self, monkeypatch):
        """Any access from here on is a defect, so make it fail loudly."""
        import src.market_data.access as access

        def refuse(*args, **kwargs):
            raise AssertionError(
                "market data was accessed while reading a stored result; the "
                "provenance should have come from the record")

        monkeypatch.setattr(access, "resolve", refuse)
        monkeypatch.setattr(access, "resolve_prices", refuse)

    def test_every_field_survives_the_default_changing(self, store,
                                                       monkeypatch):
        original = self.build(store)

        # The deployment moves on, and nothing may consult it.
        monkeypatch.setenv(POLICY,
                           "market-data-egress/pilot-vendor-approved@1")
        self.forbid_market_data(monkeypatch)

        worksheet = store.get_worksheet("ws-1", "alice")
        run = store.get_run(worksheet["payload"]["primary_run_ref"], "alice")
        reopened = from_json(dig(run["result"], "market_data"))

        assert reopened.snapshot_id == original.snapshot_id
        assert reopened.content_digest == original.content_digest
        assert reopened.content_digest_version == original.content_digest_version
        assert reopened.policy_version == original.policy_version
        assert reopened.access_decision == original.access_decision
        assert reopened.access_decision_reason == original.access_decision_reason
        assert reopened.accessed_at == original.accessed_at

    def test_reopening_the_worksheet_touches_no_market_data(self, store,
                                                            monkeypatch):
        self.build(store)
        self.forbid_market_data(monkeypatch)
        assert store.get_worksheet("ws-1", "alice") is not None
        assert store.get_run("r-1", "alice") is not None

    def test_an_export_carries_the_provenance_it_was_given(self, store,
                                                           monkeypatch):
        """A bundle must be verifiable offline, so it embeds rather than
        referencing something the reader cannot reach."""
        from src.db.transfer import export_bundle

        original = self.build(store)
        monkeypatch.setenv(POLICY,
                           "market-data-egress/pilot-vendor-approved@1")
        self.forbid_market_data(monkeypatch)

        bundle = export_bundle(store, exported_at="2026-08-01T00:00:00Z")
        run = bundle["records"]["plan_run"][0]
        embedded = from_json(dig(run["result"], "market_data"))
        assert embedded.snapshot_id == original.snapshot_id
        assert embedded.content_digest == original.content_digest

    def test_the_export_reconstructs_nothing(self, store, monkeypatch):
        from src.db.transfer import export_bundle

        self.build(store)
        monkeypatch.setenv(POLICY,
                           "market-data-egress/pilot-vendor-approved@1")
        self.forbid_market_data(monkeypatch)
        # Raises if any stage reached for the resolver.
        export_bundle(store, exported_at="2026-08-01T00:00:00Z")

    def test_a_legacy_run_stays_legacy_through_the_chain(self, store,
                                                        monkeypatch):
        """The failure this whole design prevents: a row with no provenance
        acquiring today's snapshot on the way out and looking authoritative."""
        from src.db.transfer import export_bundle
        from src.workspace.worksheet import create

        TestInstanceCompleteness().stored_run(store, None)
        store.save_worksheet(create(
            worksheet_id="ws-1", owner_id="alice", scenario_ref="p-1",
            primary_run_ref="r-1", created_at="2026-01-01T00:00:00Z"))
        self.forbid_market_data(monkeypatch)

        bundle = export_bundle(store, exported_at="2026-08-01T00:00:00Z")
        run = bundle["records"]["plan_run"][0]
        carried = from_json(dig(run["result"], "market_data"))
        assert carried.status is ProvenanceStatus.NOT_RECORDED
        assert carried.snapshot_id is None


class TestTheWriterRefusesAnUnattributableRun:
    """The guard that closes the reachability gap.

    The live path resolved a frame and a provenance together, used the frame,
    and dropped the provenance — so every stored run was unattributable while
    the mechanism to attribute it already existed. The store now refuses a
    result that cannot say which data produced it.
    """

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        from src.workspace.store import WorkspaceStore

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        store = WorkspaceStore(tmp_path / "w.db")
        inventory = TestInstanceCompleteness()
        store.save_plan(plan_id="p-1", owner="alice",
                        scenario=inventory.scenario(), stated_text="x",
                        saved_at="2026-01-01T00:00:00Z")
        return store

    def record(self, store, market_data, run_id="r-1"):
        result = {"modelling_scope": {"excludes": []}, "final_value": 1.0}
        if market_data is not _OMITTED:
            result["market_data"] = market_data
        return store.record_run(run_id=run_id, plan_id="p-1", owner="alice",
                                ran_at="2026-01-01T00:00:00Z", result=result,
                                comparison={})

    def test_an_omitted_provenance_is_refused(self, store):
        from src.workspace.store import NotSaveable

        with pytest.raises(NotSaveable, match="no market-data provenance"):
            self.record(store, _OMITTED)

    def test_the_refusal_names_all_three_options(self, store):
        """`None` cannot mean market-derived, not market-derived and unknown at
        once, so the refusal says which of the three to state."""
        from src.workspace.store import NotSaveable

        with pytest.raises(NotSaveable) as caught:
            self.record(store, _OMITTED)
        assert "NOT_APPLICABLE" in str(caught.value)

    def test_nothing_is_written_when_it_is_refused(self, store):
        from src.workspace.store import NotSaveable

        with pytest.raises(NotSaveable):
            self.record(store, _OMITTED)
        assert store.get_run("r-1", "alice") is None

    def test_a_denied_decision_is_refused(self, store):
        """A denied read cannot authorize a stored result."""
        from src.workspace.store import NotSaveable

        with pytest.raises(NotSaveable, match="DENIED"):
            self.record(store, {"status": "RECORDED", "snapshot_id": "s-1",
                                "content_digest": "mdv1:aaa",
                                "access_decision": "DENIED",
                                "accessed_at": "2026-01-01T00:00:00Z"})

    def test_a_label_without_a_digest_is_refused(self, store):
        from src.workspace.store import NotSaveable

        with pytest.raises(NotSaveable, match="content digest"):
            self.record(store, {"status": "RECORDED", "snapshot_id": "s-1",
                                "access_decision": "SYNTHETIC_ALLOWED",
                                "accessed_at": "2026-01-01T00:00:00Z"})

    def test_not_applicable_is_accepted(self, store):
        """A run that used no market data says so, and is stored."""
        from src.market_data.provenance import not_applicable

        assert self.record(store, not_applicable().to_json()) == "r-1"

    def test_not_recorded_is_accepted_for_a_legacy_import(self, store):
        from src.market_data.provenance import not_recorded

        assert self.record(store, not_recorded("legacy").to_json()) == "r-1"

    def test_a_resolver_provenance_is_accepted(self, store):
        from src.market_data.access import resolve

        access = resolve(context="a run", accessed_at="2026-01-01T00:00:00Z")
        assert self.record(store, access.provenance.to_json()) == "r-1"


class TestTheResultCarriesItsOwnProvenance:
    """`MissionResult` holds it, so it travels with the figure."""

    def build(self, market_data=None):
        """A real result from the engine, not a stub.

        Standing in a fake `path` meant chasing whichever attribute `to_json`
        happened to touch next, and a stub that satisfies the serializer today
        proves nothing about the one it will satisfy tomorrow.
        """
        import pandas as pd

        from src.mission.simulate import simulate

        sessions = pd.date_range("2026-01-01", periods=3, freq="D")
        prices = pd.DataFrame({"ACME": [10.0, 11.0, 12.0]}, index=sessions)
        from src.mission.accounting import CashPolicy
        from src.mission.benchmark import buy_and_hold

        result = simulate(prices, flows=[], program=buy_and_hold([]),
                          cash_policy=CashPolicy.idle(),
                          modelling_scope={"excludes": []})
        import dataclasses

        return dataclasses.replace(result, market_data=market_data)

    def test_a_result_without_it_serializes_as_not_recorded(self):
        """Not omitted. An omitted field would mean all three things at once."""
        body = self.build().to_json()
        assert body["market_data"]["status"] == \
            ProvenanceStatus.NOT_RECORDED.value

    def test_a_result_with_it_serializes_the_record(self, monkeypatch):
        from src.market_data.access import resolve

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        access = resolve(context="a run", accessed_at="2026-01-01T00:00:00Z")
        body = self.build(access.provenance).to_json()
        assert body["market_data"]["snapshot_id"] == \
            access.provenance.snapshot_id
        assert body["market_data"]["content_digest"]

    def test_every_serialized_result_carries_the_field(self):
        assert "market_data" in self.build().to_json()


#: Distinguishes "the caller passed None" from "the caller passed nothing".
_OMITTED = object()
