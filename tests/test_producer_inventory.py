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
        store.save_plan(plan_id="p-1", owner="alice", scenario=self.scenario(),
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        result = {"modelling_scope": {"excludes": []}, "final_value": 1.0}
        if provenance_payload is not None:
            result["market_data"] = provenance_payload
        store.record_run(run_id="r-1", plan_id="p-1", owner="alice",
                         ran_at="2026-01-01T00:00:00Z", result=result,
                         comparison={})
        return store.get_run("r-1", "alice")

    def test_a_run_with_recorded_provenance_has_a_usable_chain(self, store):
        from src.market_data.access import resolve

        _, provenance = resolve(context="a run",
                                accessed_at="2026-01-01T00:00:00Z")
        run = self.stored_run(store, provenance.to_json())

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
