"""The chain from a stored figure back to the frame that produced it.

    run -> access_event_id -> event -> frame_digest
                                    -> provenance_digest -> the run's own claim

Three checks, kept apart because two can hold while the third fails and one
boolean would hide which. The store's `verify_access_chain` returns problems
rather than a verdict for the same reason.

**Nothing here consults current configuration.** A run made under one snapshot
must not be reported broken because the deployment's default moved — a verifier
that cries wolf on an ordinary policy change is a verifier that gets switched
off, and then the real tampering passes too.
"""
from __future__ import annotations

import pytest

from src.market_data.provenance import AccessDecision, ProvenanceStatus
from src.workspace.store import NotSaveable, WorkspaceStore

POLICY = "PILOT_DATA_POLICY"
OWNER = "alice"
OTHER = "mallory"
AT = "2026-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")


@pytest.fixture
def store(tmp_path):
    return WorkspaceStore(tmp_path / "w.db")


def a_scenario():
    from tests.test_producer_inventory import TestInstanceCompleteness

    return TestInstanceCompleteness().scenario()


@pytest.fixture
def planned(store):
    scenario = a_scenario()
    store.save_plan(plan_id="p-1", owner=OWNER, scenario=scenario,
                    stated_text="x", saved_at=AT)
    return store, scenario


def an_access(run_id="run-1"):
    from src.market_data.access import resolve

    return resolve(context="a run", accessed_at=AT, run_id=run_id,
                   request_id="req-1")


def a_run(store, access, *, run_id="run-1", owner=OWNER, plan_id="p-1"):
    store.record_access_event(access.access_event, owner=owner)
    return store.record_run(
        run_id=run_id, plan_id=plan_id, ran_at=AT, owner=owner,
        result={"modelling_scope": {"excludes": []}, "final_value": 1.0,
                "market_data": access.provenance.to_json()},
        comparison={}, access_event_id=access.access_event_id)


class TestADeliveryIsRecordedOnce:
    def test_it_round_trips(self, planned):
        store, _ = planned
        access = an_access()
        store.record_access_event(access.access_event, owner=OWNER)
        stored = store.get_access_event(access.access_event_id, OWNER)
        assert stored["frame_digest"] == access.access_event.frame_digest
        assert stored["row_count"] == access.access_event.row_count
        assert stored["selected_columns"] == \
            list(access.access_event.selected_columns)

    def test_the_same_body_is_redelivery(self, planned):
        store, _ = planned
        access = an_access()
        first = store.record_access_event(access.access_event, owner=OWNER)
        second = store.record_access_event(access.access_event, owner=OWNER)
        assert first == second

    def test_a_different_body_under_one_identity_is_a_conflict(self, planned):
        import dataclasses

        store, _ = planned
        access = an_access()
        store.record_access_event(access.access_event, owner=OWNER)
        forged = dataclasses.replace(access.access_event,
                                     frame_digest="mdf1:a-different-frame")
        with pytest.raises(NotSaveable, match="already stored with a different"):
            store.record_access_event(forged, owner=OWNER)

    def test_an_incoherent_event_is_refused(self, planned):
        import dataclasses

        store, _ = planned
        access = an_access()
        empty = dataclasses.replace(access.access_event, row_count=0)
        with pytest.raises(NotSaveable, match="not a coherent delivery"):
            store.record_access_event(empty, owner=OWNER)

    def test_another_tenant_cannot_see_it(self, planned):
        store, _ = planned
        access = an_access()
        store.record_access_event(access.access_event, owner=OWNER)
        assert store.get_access_event(access.access_event_id, OTHER) is None


class TestARunMayNotCiteEvidenceItDoesNotHave:
    def test_an_unrecorded_delivery_is_refused(self, planned):
        store, _ = planned
        with pytest.raises(NotSaveable, match="not in this workspace"):
            store.record_run(
                run_id="run-1", plan_id="p-1", ran_at=AT, owner=OWNER,
                result={"modelling_scope": {"excludes": []},
                        "market_data": an_access().provenance.to_json()},
                comparison={}, access_event_id="mdae-never-recorded")

    def test_a_delivery_for_another_run_is_refused(self, planned):
        """The substitution the digest exists to detect: real evidence, real
        run, wrong pairing."""
        store, _ = planned
        access = an_access(run_id="run-other")
        store.record_access_event(access.access_event, owner=OWNER)
        with pytest.raises(NotSaveable, match="recorded for run"):
            store.record_run(
                run_id="run-1", plan_id="p-1", ran_at=AT, owner=OWNER,
                result={"modelling_scope": {"excludes": []},
                        "market_data": access.provenance.to_json()},
                comparison={}, access_event_id=access.access_event_id)

    def test_another_tenants_delivery_is_refused(self, planned):
        store, _ = planned
        access = an_access()
        store.record_access_event(access.access_event, owner=OTHER)
        with pytest.raises(NotSaveable, match="not in this workspace"):
            store.record_run(
                run_id="run-1", plan_id="p-1", ran_at=AT, owner=OWNER,
                result={"modelling_scope": {"excludes": []},
                        "market_data": access.provenance.to_json()},
                comparison={}, access_event_id=access.access_event_id)

    def test_a_cited_run_stores_the_reference(self, planned):
        store, _ = planned
        access = an_access()
        a_run(store, access)
        assert store.get_run("run-1", OWNER)["access_event_id"] == \
            access.access_event_id


class TestTheChainVerifies:
    def test_a_complete_chain_has_no_problems(self, planned):
        store, _ = planned
        a_run(store, an_access())
        assert store.verify_access_chain("run-1", OWNER) == []

    def test_a_run_citing_nothing_is_not_a_failure(self, planned):
        """Absence is a fact. Runs recorded before deliveries were captured
        cite none, and calling that tampering would make the check useless."""
        store, _ = planned
        access = an_access()
        store.record_run(
            run_id="legacy", plan_id="p-1", ran_at=AT, owner=OWNER,
            result={"modelling_scope": {"excludes": []},
                    "market_data": access.provenance.to_json()},
            comparison={})
        assert store.verify_access_chain("legacy", OWNER) == []

    def test_an_edited_event_body_is_caught(self, planned):
        """Below the store, because the store's own writer refuses this —
        which is the point. The guard is what a direct database edit meets."""
        store, _ = planned
        a_run(store, an_access())
        with store._conn() as conn:
            conn.execute(
                "UPDATE market_data_access_event SET frame_digest = ?",
                ("mdf1:swapped-after-the-fact",))
        problems = store.verify_access_chain("run-1", OWNER)
        assert any("edited since it was written" in one for one in problems)

    def test_a_dangling_reference_makes_the_run_unverifiable(self, planned):
        """The constraint makes this unreachable through the database — see
        `TestDeletingEvidenceIsRefusedWhileARunCitesIt` — so it is constructed
        with foreign keys off.

        The verifier must still answer, because the state is reachable another
        way: a dump restored without constraints, a table rebuilt by a
        migration, an engine where enforcement was never switched on. SQLite
        did not enforce foreign keys here at all until Gate 3 added the pragma,
        which is exactly how long a "guaranteed impossible" state was in fact
        producible.
        """
        store, _ = planned
        a_run(store, an_access())
        with store._conn() as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("DELETE FROM market_data_access_event")
            conn.execute("PRAGMA foreign_keys = ON")
        problems = store.verify_access_chain("run-1", OWNER)
        assert any("not in this workspace" in one for one in problems)

    def test_a_rebound_event_is_caught(self, planned):
        """Run binding, independently of integrity: the event's hash still
        matches its body, and it is evidence about a different execution."""
        store, _ = planned
        a_run(store, an_access())
        with store._conn() as conn:
            conn.execute("UPDATE market_data_access_event SET run_id = ?",
                         ("some-other-run",))
        problems = store.verify_access_chain("run-1", OWNER)
        assert any("recorded for run" in one for one in problems)

    def test_a_swapped_run_provenance_is_caught(self, planned):
        """Declared consistency, independently of the other two: the event is
        intact and bound correctly, and the run's own claim disagrees with it.
        """
        from src.db.types import Json

        store, _ = planned
        access = an_access()
        a_run(store, access)

        other = dict(access.provenance.to_json())
        other["snapshot_id"] = "a-snapshot-this-run-never-read"
        forged = {"modelling_scope": {"excludes": []}, "final_value": 1.0,
                  "market_data": other}
        with store._conn() as conn:
            conn.execute("UPDATE plan_run SET result = ? WHERE run_id = ?",
                         (Json(forged), "run-1"))

        problems = store.verify_access_chain("run-1", OWNER)
        assert any("disagree" in one for one in problems)

    def test_the_three_checks_are_independent(self, planned):
        """Each mutation above trips exactly the check it is about. A single
        verdict would have hidden which, and 'something is wrong' is not
        actionable at three in the morning."""
        store, _ = planned
        a_run(store, an_access())
        assert store.verify_access_chain("run-1", OWNER) == []

    def test_a_missing_run_is_reported_rather_than_passed(self, planned):
        store, _ = planned
        assert store.verify_access_chain("no-such-run", OWNER)

    def test_another_tenant_cannot_verify_this_run(self, planned):
        store, _ = planned
        a_run(store, an_access())
        assert store.verify_access_chain("run-1", OTHER) == \
            ["no run 'run-1' for this owner"]


class TestTheDeliveryOutlivesTheConfiguration:
    def test_a_policy_change_does_not_break_verification(self, planned,
                                                          monkeypatch):
        store, _ = planned
        a_run(store, an_access())
        monkeypatch.setenv(POLICY,
                           "market-data-egress/pilot-vendor-approved@1")
        assert store.verify_access_chain("run-1", OWNER) == []

    def test_verification_resolves_nothing(self, planned, monkeypatch):
        store, _ = planned
        a_run(store, an_access())

        import src.market_data.access as access_module

        def refuse(*args, **kwargs):
            raise AssertionError(
                "verification called the resolver; the answer must come from "
                "the stored record, not from what is configured now")

        monkeypatch.setattr(access_module, "resolve", refuse)
        assert store.verify_access_chain("run-1", OWNER) == []


class TestDeletingEvidenceIsRefusedWhileARunCitesIt:
    def test_the_constraint_holds(self, planned):
        """RESTRICT rather than CASCADE: a stored figure must not become
        unverifiable because something else was deleted."""
        import sqlite3

        store, _ = planned
        access = an_access()
        a_run(store, access)
        with pytest.raises(sqlite3.IntegrityError):
            with store._conn() as conn:
                conn.execute(
                    "DELETE FROM market_data_access_event "
                    "WHERE access_event_id = ?", (access.access_event_id,))

    def test_deleting_the_run_first_releases_it(self, planned):
        store, _ = planned
        access = an_access()
        a_run(store, access)
        with store._conn() as conn:
            conn.execute("DELETE FROM plan_run WHERE run_id = ?", ("run-1",))
            conn.execute(
                "DELETE FROM market_data_access_event WHERE access_event_id = ?",
                (access.access_event_id,))
        assert store.get_access_event(access.access_event_id, OWNER) is None


class TestALiveProducerMustCiteItsDelivery:
    def test_generate_refuses_a_market_derived_run_without_one(self, planned):
        from src.workspace.generate import UnattributableRun, generate

        store, scenario = planned
        access = an_access()
        with pytest.raises(UnattributableRun, match="cites no market-data"):
            generate(store, plan_id="p-1", owner=OWNER, scenario=scenario,
                     run={"modelling_scope": {"excludes": []},
                          "market_data": access.provenance.to_json()},
                     comparison={}, ran_at=AT)

    def test_generate_refuses_a_delivery_for_another_run(self, planned):
        from src.workspace.generate import UnattributableRun, generate

        store, scenario = planned
        with pytest.raises(UnattributableRun, match="recorded for run"):
            generate(store, plan_id="p-1", owner=OWNER, scenario=scenario,
                     run={"modelling_scope": {"excludes": []},
                          "market_data": an_access().provenance.to_json()},
                     comparison={}, ran_at=AT,
                     access=an_access(run_id="a-different-run"))

    def test_a_run_with_no_market_data_needs_no_delivery(self, planned):
        """`NOT_APPLICABLE` consumed no frame. Demanding evidence of a delivery
        that never happened would refuse a legitimate result."""
        from src.market_data.provenance import not_applicable
        from src.workspace.generate import generate

        store, scenario = planned
        worksheet = generate(
            store, plan_id="p-1", owner=OWNER, scenario=scenario,
            run={"modelling_scope": {"excludes": []},
                 "market_data": not_applicable().to_json()},
            comparison={}, ran_at=AT)
        assert worksheet is not None

    def test_generate_records_the_delivery_before_the_run(self, planned):
        from src.workspace.generate import generate, run_id_for

        store, scenario = planned
        identifier = run_id_for("p-1", scenario.content_hash, AT)
        access = an_access(run_id=identifier)
        generate(store, plan_id="p-1", owner=OWNER, scenario=scenario,
                 run={"modelling_scope": {"excludes": []},
                      "market_data": access.provenance.to_json()},
                 comparison={}, ran_at=AT, access=access)
        assert store.verify_access_chain(identifier, OWNER) == []


class TestOneResolutionIsOneDeliveryAcrossAFanOut:
    """SHARED_ACCESS, with the fan-out actually constructed.

    The journey produces a single run, so an assertion there that "every run
    cites the same delivery" holds trivially — it was written first and proved
    nothing, which the falsification pass caught by mutating the fan-out and
    watching nothing fail. Several candidates must exist for the claim to have
    content, so they are built here.

    What is at stake is comparability. The point of running candidates side by
    side is that they were measured on the same data; candidates citing
    separate deliveries would be asserting the opposite, and the comparison
    would look identical either way.
    """

    OWNER = "alice"

    @pytest.fixture
    def prepared(self, tmp_path, monkeypatch):
        from src.mission.compiler import compile_scenario
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance
        from src.workspace.worksheet import create

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        store = WorkspaceStore(tmp_path / "w.db")
        compiled = compile_scenario(
            "I put $2,000 into SPY every month in my Roth IRA, on the first "
            "trading day of the period, reinvesting the dividends, and I "
            "never sell.",
            name="plan-1", version=1,
            benchmark_rule="benchmark-policy/public-default@1")
        source = compiled.scenario.provenance
        scenario = ScenarioSpecification(**{
            **compiled.scenario.__dict__,
            "provenance": Provenance(
                stated=source.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in source.inferred),
                contradictions=source.contradictions, unresolved=())})
        store.save_plan(plan_id="plan-1", owner=self.OWNER, scenario=scenario,
                        stated_text="seed", saved_at="t0")
        store.record_run(run_id="run-0", plan_id="plan-1", ran_at="t0",
                         owner=self.OWNER, result=_RESULT, comparison={})
        store.save_worksheet(create(
            worksheet_id="ws-1", owner_id=self.OWNER, scenario_ref="plan-1",
            primary_run_ref="run-0", created_at="t0"))
        return store

    def accepted(self, store):
        from src.workspace.apply import accept
        from src.workspace.intent import plan
        from src.workspace.proposal import propose
        from src.workspace.worksheet import from_json

        worksheet = from_json(store.get_worksheet("ws-1", self.OWNER)["payload"])
        intent = plan("Try SPY, VTI and VT and keep the best", intent_id="i",
                      source_revision=worksheet.revision, history=[],
                      target_run="run-0")
        proposal = propose(intent, worksheet)
        store.save_worksheet_proposal(
            proposal_id="p1", owner=self.OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="t0")

        access = an_access(run_id=None)
        # Carrying the delivery's own provenance, as `_run` does in
        # production — a candidate that cites a delivery and declares a
        # different source is the disagreement `verify_access_chain` exists to
        # report, and a fixture that produced it would be testing the fixture.
        candidate_result = {**_RESULT,
                            "market_data": access.provenance.to_json()}
        result = accept(store, proposal_id="p1", owner=self.OWNER,
                        worksheet_id="ws-1", proposal=proposal, at="t1",
                        run_candidate=lambda candidate: candidate_result,
                        access=access)
        return result, access

    def test_the_fan_out_produced_several_runs(self, prepared):
        """Without this the rest of the class is vacuous — which is exactly
        how the journey version of this claim passed while proving nothing."""
        result, _ = self.accepted(prepared)
        assert len(result.runs) > 1, (
            f"only {len(result.runs)} candidate run(s); there is no fan-out "
            "here and every assertion below would hold trivially")

    def test_every_candidate_cites_the_same_delivery(self, prepared):
        store = prepared
        result, access = self.accepted(store)
        cited = {store.get_run(run_id, self.OWNER)["access_event_id"]
                 for run_id in result.runs}
        assert cited == {access.access_event_id}

    def test_exactly_one_delivery_row_exists(self, prepared):
        store = prepared
        self.accepted(store)
        with store._conn() as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM market_data_access_event "
                "WHERE owner = ?", (self.OWNER,)).fetchone()["n"]
        assert count == 1, f"{count} deliveries for one resolution"

    def test_every_candidate_verifies(self, prepared):
        store = prepared
        result, _ = self.accepted(store)
        for run_id in result.runs:
            assert store.verify_access_chain(run_id, self.OWNER) == [], run_id

    def test_a_fan_out_delivery_names_no_single_run(self, prepared):
        """It was resolved for the acceptance, not for any one candidate.
        Naming one would make every other citation look like the substitution
        `record_run` refuses."""
        store = prepared
        result, access = self.accepted(store)
        stored = store.get_access_event(access.access_event_id, self.OWNER)
        assert stored["run_id"] is None
        assert len(result.runs) > 1


_RESULT = {"modelling_scope": {"excludes": []}, "final_value": 1.0,
           "market_data": {"status": "NOT_APPLICABLE",
                           "access_decision_reason": "candidate placeholder"}}
