"""Traces and decisions, and the rule that nothing financial may need them.

Spans say **when**. Decisions say **why**. The questions asked six months later
are the second kind — "why wasn't this benchmark included?", "why did this
become AFTER_RESULTS?" — and a duration cannot answer either.

The load-bearing test in this file is `TestTelemetryIsExpendable`. Operational
telemetry expires; financial artifacts do not. The moment something in the
workspace needs a span to answer a question about a figure, telemetry has become
a second artifact store with a deletion policy attached to it.
"""
from __future__ import annotations

import pytest

from src.telemetry import (
    DecisionKind,
    Recorder,
    TraceStore,
    new_conversation_id,
    new_request_id,
)
from src.workspace.intent_history import IntentHistory
from src.workspace.intent_service import plan_and_record
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create

OWNER = "pilot"


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace.db"
    WorkspaceStore(path).save_worksheet(
        create(worksheet_id="ws-1", owner_id=OWNER, scenario_ref="plan-1",
               primary_run_ref="run-0", created_at="t0"))
    return path


@pytest.fixture
def traces(tmp_path):
    return tmp_path / "trace.db"


def recorder(traces, conversation=None, worksheet_id="ws-1"):
    return Recorder(TraceStore(traces), tenant=OWNER,
                    conversation_id=conversation or new_conversation_id(),
                    request_id=new_request_id(),
                    worksheet_id=worksheet_id).start()


def request(workspace, traces, instruction, index, conversation=None):
    rec = recorder(traces, conversation)
    planned = plan_and_record(
        WorkspaceStore(workspace), worksheet_id="ws-1", owner=OWNER,
        instruction=instruction, intent_id=f"i{index}",
        proposal_id=f"p{index}", at=f"t{index}", recorder=rec)
    rec.finish()
    return planned, rec


class TestTheStoresAreSeparate:

    def test_telemetry_does_not_live_in_the_workspace(self, workspace, traces):
        request(workspace, traces, "Add 63-day rolling volatility", 0)
        import sqlite3

        tables = {r[0] for r in sqlite3.connect(workspace).execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "span" not in tables
        assert "decision" not in tables
        assert "trace" not in tables

    def test_the_artifact_keeps_only_a_reference(self, workspace, traces):
        planned, rec = request(workspace, traces,
                               "Add 63-day rolling volatility", 0)
        [row] = WorkspaceStore(workspace).worksheet_intents("ws-1", OWNER)
        assert row["trace_id"] == rec.trace_id
        assert planned.trace_id == rec.trace_id


class TestTelemetryIsExpendable:
    """The rule that keeps telemetry from becoming a second artifact store."""

    def test_the_intent_chain_survives_deleting_every_trace(self, workspace,
                                                            traces):
        request(workspace, traces, "Add 21-day rolling volatility", 0)
        request(workspace, traces, "Add 63-day rolling volatility", 1)
        before = IntentHistory.from_store(WorkspaceStore(workspace), "ws-1",
                                          OWNER)

        traces.unlink()

        after = IntentHistory.from_store(WorkspaceStore(workspace), "ws-1",
                                         OWNER)
        assert after.trial_total == before.trial_total
        assert after.verdict.trustworthy
        assert len(after.intents) == 2

    def test_planning_still_works_with_the_trace_store_destroyed(
            self, workspace, traces):
        request(workspace, traces, "Add 21-day rolling volatility", 0)
        traces.unlink()

        planned = plan_and_record(
            WorkspaceStore(workspace), worksheet_id="ws-1", owner=OWNER,
            instruction="Add 63-day rolling volatility", intent_id="i9",
            proposal_id="p9", at="t9")
        assert planned.intent.selection_basis.value == "VARIANT_EXPLORATION"

    def test_a_failing_trace_store_costs_a_trace_and_not_an_edit(
            self, workspace, traces):
        """A locked database, a full disk, a deleted file. Any of them must lose
        a trace rather than a worksheet edit."""
        class Broken:
            def start_trace(self, **kw): raise OSError("disk full")
            def end_trace(self, *a, **kw): raise OSError("disk full")
            def record_span(self, span): raise OSError("disk full")
            def record_decision(self, decision): raise OSError("disk full")

        rec = Recorder(Broken(), tenant=OWNER).start()
        planned = plan_and_record(
            WorkspaceStore(workspace), worksheet_id="ws-1", owner=OWNER,
            instruction="Add 63-day rolling volatility", intent_id="i0",
            proposal_id="p0", at="t0", recorder=rec)

        assert planned.proposal.applicable
        assert rec.failures > 0, (
            "telemetry failures must be counted, not silently swallowed")

    def test_behaviour_is_identical_with_and_without_a_recorder(
            self, tmp_path, workspace, traces):
        """The system must not behave differently when it is being watched."""
        observed, _ = request(workspace, traces,
                              "Try SPY, VTI and VT and keep the best", 0)

        other = tmp_path / "other.db"
        WorkspaceStore(other).save_worksheet(
            create(worksheet_id="ws-1", owner_id=OWNER, scenario_ref="plan-1",
                   primary_run_ref="run-0", created_at="t0"))
        unobserved = plan_and_record(
            WorkspaceStore(other), worksheet_id="ws-1", owner=OWNER,
            instruction="Try SPY, VTI and VT and keep the best",
            intent_id="i0", proposal_id="p0", at="t0")

        assert observed.intent.to_json() == unobserved.intent.to_json()
        assert observed.proposal.to_json() == unobserved.proposal.to_json()


class TestDecisionsAnswerWhy:

    def test_the_classification_records_why_it_landed_there(self, workspace,
                                                            traces):
        _, rec = request(workspace, traces, "Add 63-day rolling volatility", 0)
        recorded = TraceStore(traces).trace(rec.trace_id, OWNER)
        [classification] = [d for d in recorded["decisions"]
                            if d["kind"] == "INTENT_CLASSIFICATION"]

        assert "DERIVED_ANALYSIS" in classification["outcome"]
        assert classification["reason"]
        assert any(e.startswith("planner:")
                   for e in classification["evidence_refs"])

    def test_it_names_what_it_did_not_choose(self, workspace, traces):
        """"It became DERIVED_ANALYSIS" is far less useful beside nothing than
        beside the states it was not."""
        _, rec = request(workspace, traces, "Add 63-day rolling volatility", 0)
        [classification] = [
            d for d in TraceStore(traces).trace(rec.trace_id, OWNER)["decisions"]
            if d["kind"] == "INTENT_CLASSIFICATION"]
        assert "SCENARIO_CHANGE" in classification["alternatives"]
        assert "DERIVED_ANALYSIS" not in classification["alternatives"]

    def test_a_prior_intent_is_cited_as_evidence(self, workspace, traces):
        request(workspace, traces, "Add 21-day rolling volatility", 0)
        _, rec = request(workspace, traces, "Add 63-day rolling volatility", 1)
        [classification] = [
            d for d in TraceStore(traces).trace(rec.trace_id, OWNER)["decisions"]
            if d["kind"] == "INTENT_CLASSIFICATION"]
        assert "intent:i0" in classification["evidence_refs"]

    def test_a_refusal_records_its_reason(self, workspace, traces):
        from src.workspace.worksheet import from_json, revise

        store = WorkspaceStore(workspace)
        current = from_json(store.get_worksheet("ws-1", OWNER)["payload"])
        store.save_worksheet(revise(current, reason="advanced", created_at="t1"))

        rec = recorder(traces)
        with pytest.raises(Exception):
            plan_and_record(store, worksheet_id="ws-1", owner=OWNER,
                            instruction="Add 63-day rolling volatility",
                            intent_id="i0", proposal_id="p0", at="t0",
                            source_revision=1, recorder=rec)

        [refusal] = TraceStore(traces).trace(rec.trace_id, OWNER)["decisions"]
        assert refusal["outcome"] == "REFUSED_STALE"

    def test_a_deterministic_decision_claims_no_confidence(self, workspace,
                                                           traces):
        """A rule match has no confidence; it has a rule. A column of 1.0s
        teaches a reader that the number means something."""
        _, rec = request(workspace, traces, "Add 63-day rolling volatility", 0)
        for decision in TraceStore(traces).trace(rec.trace_id, OWNER)["decisions"]:
            assert decision["confidence"] is None


class TestTheCorrelationSpine:

    def test_one_conversation_gathers_its_requests(self, workspace, traces):
        conversation = new_conversation_id()
        request(workspace, traces, "Add 21-day rolling volatility", 0,
                conversation)
        request(workspace, traces, "Add 63-day rolling volatility", 1,
                conversation)

        assert len(TraceStore(traces).traces_for_conversation(
            conversation, OWNER)) == 2

    def test_a_trace_can_be_found_from_what_it_produced(self, workspace,
                                                        traces):
        """"Show me every model interaction that eventually produced this."""
        planned, rec = request(workspace, traces,
                               "Add 63-day rolling volatility", 0)
        found = TraceStore(traces).traces_producing(
            f"proposal:{planned.proposal_id}", OWNER)
        assert [t["trace_id"] for t in found] == [rec.trace_id]

    def test_another_tenant_sees_none_of_it(self, workspace, traces):
        _, rec = request(workspace, traces, "Add 63-day rolling volatility", 0)
        store = TraceStore(traces)
        assert store.trace(rec.trace_id, "someone-else") is None
        assert store.traces_for_conversation(rec.conversation_id,
                                             "someone-else") == []

    def test_spans_nest(self, workspace, traces):
        _, rec = request(workspace, traces, "Add 63-day rolling volatility", 0)
        spans = TraceStore(traces).trace(rec.trace_id, OWNER)["spans"]
        assert {s["name"] for s in spans} >= {
            "worksheet_load", "history_load", "intent_planning",
            "proposal_generation"}
        assert all(s["duration_ms"] is not None for s in spans)


class TestThePrivacyBoundary:

    def test_no_raw_instruction_reaches_the_trace_store(self, workspace,
                                                        traces):
        secret = "Add 63-day rolling volatility for my Tesla RSUs at Acme Corp"
        _, rec = request(workspace, traces, secret, 0)

        body = traces.read_bytes().decode("utf-8", errors="ignore")
        assert "Tesla" not in body
        assert "Acme" not in body

    def test_an_error_records_its_class_not_its_message(self, traces):
        """A message can quote the input that caused it."""
        rec = recorder(traces)
        with pytest.raises(ValueError):
            with rec.span("thing"):
                raise ValueError("holdings: 4000 shares of TSLA")

        [span] = TraceStore(traces).trace(rec.trace_id, OWNER)["spans"]
        assert span["error"] == "ValueError"
        assert "TSLA" not in str(span)


class TestRetention:

    def test_old_traces_are_purged_with_their_spans_and_decisions(
            self, workspace, traces):
        request(workspace, traces, "Add 63-day rolling volatility", 0)
        store = TraceStore(traces)

        purged = store.purge_before("2999-01-01T00:00:00+00:00")
        assert purged["traces"] == 1
        assert purged["spans"] > 0
        assert purged["decisions"] > 0

    def test_purging_traces_leaves_the_artifacts_untouched(self, workspace,
                                                           traces):
        request(workspace, traces, "Add 21-day rolling volatility", 0)
        request(workspace, traces, "Add 63-day rolling volatility", 1)
        TraceStore(traces).purge_before("2999-01-01T00:00:00+00:00")

        history = IntentHistory.from_store(WorkspaceStore(workspace), "ws-1",
                                           OWNER)
        assert history.trial_total == 2
        assert history.verdict.trustworthy

    def test_a_tenant_deletion_is_not_a_retention_policy(self, workspace,
                                                         traces):
        """A deletion request must not wait for expiry to come round."""
        request(workspace, traces, "Add 63-day rolling volatility", 0)
        store = TraceStore(traces)

        assert store.purge_tenant(OWNER)["traces"] == 1
        assert store.traces_for_conversation("anything", OWNER) == []

    def test_purging_one_tenant_leaves_another(self, workspace, traces):
        _, mine = request(workspace, traces, "Add 63-day rolling volatility", 0)
        store = TraceStore(traces)
        store.start_trace(trace_id="other", conversation_id="c",
                          request_id="r", tenant="someone-else",
                          started_at="t0")

        store.purge_tenant("someone-else")
        assert store.trace(mine.trace_id, OWNER) is not None
