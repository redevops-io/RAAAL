"""Accepting a proposal: durable state, in an order that cannot orphan anything.

    validate source revision
        -> persist artifacts and runs
        -> persist the worksheet revision citing them
        -> resolve the proposal
        -> commit

The load-bearing test instruments the store and asserts the actual write order,
rather than inspecting the source. Source inspection proves what a function
mentions; only the order of writes proves what it did.
"""
from __future__ import annotations

from tests.market_fixture import NO_MARKET_DATA
from pathlib import Path

import pytest

from src.workspace.apply import (
    ApplyRefused,
    ProposalStatus,
    StaleProposal,
    accept,
    reject,
)
from src.workspace.intent import plan
from src.workspace.proposal import propose
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import Block, create, from_json, revise

OWNER = "pilot"
RESULT = {"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}, "final_value": 1.0, "market_data": NO_MARKET_DATA.to_json()}


@pytest.fixture
def store(tmp_path):
    """A stored plan, a run and a worksheet — `get_run` scopes ownership
    through the plan, so a run with no plan is invisible."""
    from src.mission.compiler import compile_scenario
    from src.mission.scenario import ScenarioSpecification
    from src.mission.spec import Inference, Provenance

    store = WorkspaceStore(tmp_path / "w.db")
    compiled = compile_scenario(
        "I put $2,000 into SPY every month in my Roth IRA, on the first trading "
        "day of the period, reinvesting the dividends, and I never sell.",
        name="plan-1", version=1,
        benchmark_rule="benchmark-policy/public-default@1")
    provenance = compiled.scenario.provenance
    scenario = ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=provenance.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in provenance.inferred),
            contradictions=provenance.contradictions, unresolved=())})
    store.save_plan(plan_id="plan-1", owner=OWNER, scenario=scenario,
                    stated_text="seed", saved_at="t0")
    store.record_run(run_id="run-0", plan_id="plan-1", ran_at="t0",
                     result=RESULT, comparison={})
    store.save_worksheet(create(worksheet_id="ws-1", owner_id=OWNER,
                                scenario_ref="plan-1", primary_run_ref="run-0",
                                created_at="t0"))
    return store


@pytest.fixture
def worksheet(store):
    return from_json(store.get_worksheet("ws-1", OWNER)["payload"])


def proposal_for(instruction, worksheet, *, history=()):
    intent = plan(instruction, intent_id="i", source_revision=worksheet.revision,
                  history=list(history), target_run="run-0")
    return propose(intent, worksheet), intent


def stage(store, instruction, worksheet, *, history=(), proposal_id="p1"):
    proposal, intent = proposal_for(instruction, worksheet, history=history)
    store.save_worksheet_proposal(proposal_id=proposal_id, owner=OWNER,
                                  worksheet_id="ws-1", proposal=proposal,
                                  created_at="t0")
    return proposal, intent


class TestWriteOrder:
    """The invariant, proven by watching the store rather than reading code."""

    def test_runs_are_written_before_the_revision(self, store, worksheet):
        proposal, _ = stage(store, "Replace SPY with VTI", worksheet)

        order = []
        original_run, original_sheet = store.record_run, store.save_worksheet
        store.record_run = lambda **kw: (order.append(("run", kw["run_id"])),
                                         original_run(**kw))[1]
        store.save_worksheet = lambda w: (order.append(("revision", w.revision)),
                                          original_sheet(w))[1]

        accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
               proposal=proposal, at="t1", run_candidate=lambda c: RESULT)

        assert [kind for kind, _ in order] == ["run", "revision"], order

    def test_every_candidate_run_precedes_the_revision(self, store, worksheet):
        proposal, _ = stage(store, "Try SPY, VTI and VT and keep the best",
                            worksheet)

        order = []
        original_run, original_sheet = store.record_run, store.save_worksheet
        store.record_run = lambda **kw: (order.append("run"),
                                         original_run(**kw))[1]
        store.save_worksheet = lambda w: (order.append("revision"),
                                          original_sheet(w))[1]

        accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
               proposal=proposal, at="t1", run_candidate=lambda c: RESULT)

        assert order == ["run", "run", "run", "revision"]

    def test_a_scenario_change_without_a_runner_is_refused(self, store,
                                                          worksheet):
        """No run means no revision, enforced rather than assumed."""
        proposal, _ = stage(store, "Replace SPY with VTI", worksheet)
        with pytest.raises(ApplyRefused, match="No run means no revision"):
            accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
                   proposal=proposal, at="t1")


class TestTheSixApplyCases:

    def test_layout_move(self, store, worksheet):
        proposal, _ = stage(store, "Move the provenance panel above results",
                            worksheet)
        result = accept(store, proposal_id="p1", owner=OWNER,
                        worksheet_id="ws-1", proposal=proposal, at="t1")

        assert result.revision == 2 and result.runs == ()
        assert store.get_worksheet("ws-1", OWNER, 1) is not None
        updated = from_json(store.get_worksheet("ws-1", OWNER)["payload"])
        assert list(updated.layout) != list(worksheet.layout)

    def test_single_derived_analysis(self, store, worksheet):
        proposal, _ = stage(store, "Add 63-day rolling volatility", worksheet)
        result = accept(store, proposal_id="p1", owner=OWNER,
                        worksheet_id="ws-1", proposal=proposal, at="t1")

        assert len(result.derived) == 1 and result.runs == ()
        updated = from_json(store.get_worksheet("ws-1", OWNER)["payload"])
        assert updated.scenario_ref == worksheet.scenario_ref

    def test_variant_exploration_records_every_candidate(self, store, worksheet):
        history = [plan("Add 63-day rolling volatility", intent_id="i0",
                        source_revision=1, target_run="run-0")]
        proposal, _ = stage(store, "Try 21, 63 and 126 day windows", worksheet,
                            history=history)
        result = accept(store, proposal_id="p1", owner=OWNER,
                        worksheet_id="ws-1", proposal=proposal, at="t1")

        assert len(result.derived) == 3, "no variant may be silently omitted"
        recorded = [e for e in store.confirmation_events(OWNER)
                    if e["kind"] == "derived_analysis_recorded"]
        assert {e["final_value"] for e in recorded} == {"21", "63", "126"}

    def test_result_aware_selection_keeps_the_rejected_variants(self, store,
                                                                worksheet):
        # Chained, not independent: each classification depends on the one
        # before it, so building the history with separate `plan()` calls
        # gives every entry an empty past and the chain never forms.
        history = []
        for index, text in enumerate(["Add 63-day rolling volatility",
                                      "Try 21, 63 and 126 day windows"]):
            history.append(plan(text, intent_id=f"i{index}", source_revision=1,
                                history=list(history), target_run="run-0"))

        first, _ = stage(store, "Try 21, 63 and 126 day windows", worksheet,
                         history=history[:1], proposal_id="p1")
        accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
               proposal=first, at="t1")

        current = from_json(store.get_worksheet("ws-1", OWNER)["payload"])
        chosen, _ = stage(store, "Keep 63 because it looks smoothest", current,
                          history=history, proposal_id="p2")
        accept(store, proposal_id="p2", owner=OWNER, worksheet_id="ws-1",
               proposal=chosen, at="t2")

        recorded = [e for e in store.confirmation_events(OWNER)
                    if e["kind"] == "derived_analysis_recorded"]
        assert {e["final_value"] for e in recorded} >= {"21", "63", "126"}, (
            "the rejected variants must remain discoverable")
        assert any(e["provenance"] == "AFTER_RESULTS" for e in recorded)

    def test_single_scenario_substitution(self, store, worksheet):
        proposal, _ = stage(store, "Replace SPY with VTI", worksheet)
        result = accept(store, proposal_id="p1", owner=OWNER,
                        worksheet_id="ws-1", proposal=proposal, at="t1",
                        run_candidate=lambda c: RESULT)

        assert len(result.runs) == 1
        assert store.get_run(result.runs[0], OWNER) is not None
        assert store.get_worksheet("ws-1", OWNER, 1)["payload"][
            "primary_run_ref"] == "run-0", "the old revision is unchanged"

    def test_scenario_search_records_all_three(self, store, worksheet):
        proposal, _ = stage(store, "Try SPY, VTI and VT and keep the best",
                            worksheet)
        result = accept(store, proposal_id="p1", owner=OWNER,
                        worksheet_id="ws-1", proposal=proposal, at="t1",
                        run_candidate=lambda c: RESULT)

        assert len(result.runs) == 3
        updated = from_json(store.get_worksheet("ws-1", OWNER)["payload"])
        assert set(result.runs) <= set(updated.benchmark_run_refs), (
            "citing only the activated candidate would hide the search")


class TestRefusals:

    def test_a_stale_proposal_is_refused_not_rebased(self, store, worksheet):
        """An old diff applied to new state changes things nobody reviewed."""
        proposal, _ = stage(store, "Move the provenance panel above results",
                            worksheet)
        store.save_worksheet(revise(worksheet, reason="something else",
                                    created_at="t0.5"))

        with pytest.raises(StaleProposal, match="Re-plan against the current"):
            accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
                   proposal=proposal, at="t1")

    def test_the_same_proposal_cannot_be_accepted_twice(self, store, worksheet):
        proposal, _ = stage(store, "Move the provenance panel above results",
                            worksheet)
        accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
               proposal=proposal, at="t1")

        with pytest.raises(ApplyRefused, match="already ACCEPTED"):
            accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
                   proposal=proposal, at="t2")

    def test_an_unsupported_proposal_is_refused(self, store, worksheet):
        proposal, _ = stage(store, "Move the sparkline widget", worksheet)
        with pytest.raises(ApplyRefused, match="not applicable"):
            accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
                   proposal=proposal, at="t1")

    def test_a_no_op_layout_proposal_is_refused(self, store, worksheet):
        proposal, _ = stage(store, "Move the scope panel below risk", worksheet)
        with pytest.raises(ApplyRefused):
            accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
                   proposal=proposal, at="t1")


class TestFailureRollsEverythingBack:

    def test_a_failing_candidate_run_leaves_no_revision(self, store, worksheet):
        """The second of three fails. Nothing is left behind."""
        proposal, _ = stage(store, "Try SPY, VTI and VT and keep the best",
                            worksheet)
        calls = {"n": 0}

        def flaky(candidate):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("the data provider timed out")
            return RESULT

        with pytest.raises(RuntimeError, match="timed out"):
            accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
                   proposal=proposal, at="t1", run_candidate=flaky)

        assert len(store.worksheet_revisions("ws-1", OWNER)) == 1
        assert [r["run_id"] for r in store.runs_for("plan-1", OWNER)] == ["run-0"], (
            "an accepted edit that produced runs and no revision leaves "
            "history belonging to nothing")

    def test_a_failing_revision_rolls_back_its_runs(self, store, worksheet):
        proposal, _ = stage(store, "Replace SPY with VTI", worksheet)
        original = store.save_worksheet

        def broken(_worksheet):
            raise RuntimeError("disk full")

        store.save_worksheet = broken
        with pytest.raises(RuntimeError, match="disk full"):
            accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
                   proposal=proposal, at="t1", run_candidate=lambda c: RESULT)

        store.save_worksheet = original
        assert [r["run_id"] for r in store.runs_for("plan-1", OWNER)] == ["run-0"]
        assert store.get_worksheet_proposal("p1", OWNER)["status"] == "PROPOSED"


class TestTheProposalItselfIsImmutable:

    def test_acceptance_records_an_outcome_beside_it(self, store, worksheet):
        proposal, _ = stage(store, "Move the provenance panel above results",
                            worksheet)
        before = store.get_worksheet_proposal("p1", OWNER)["payload"]
        accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
               proposal=proposal, at="t1", )

        after = store.get_worksheet_proposal("p1", OWNER)
        assert after["payload"] == before, "the reviewed diff is never rewritten"
        assert after["status"] == ProposalStatus.ACCEPTED.value
        assert after["result_revision"] == 2

    def test_a_rejected_proposal_stays_readable(self, store, worksheet):
        proposal, _ = stage(store, "Move the provenance panel above results",
                            worksheet)
        reject(store, proposal_id="p1", owner=OWNER, at="t1")

        record = store.get_worksheet_proposal("p1", OWNER)
        assert record["status"] == ProposalStatus.REJECTED.value
        assert record["payload"]["changes"]
