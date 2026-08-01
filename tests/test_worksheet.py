"""The ResearchWorksheet: a saved research result you can return to.

    confirmed scenario -> historical run -> revision 1
                                         -> reopen -> replay stored artifacts

Two properties carry the whole slice:

    it holds references, never copies of results
    opening it never recompiles

The second is the one that has already gone wrong once. The plan page recompiled
its stored prose and simulated the fresh interpretation while displaying the
stored scenario, so after any compiler change it showed one plan and figures
from another. A worksheet makes that worse, because a worksheet is precisely the
thing a user comes back to.
"""
from __future__ import annotations

import pytest

from src.workspace.store import NotSaveable, WorkspaceStore
from src.workspace.worksheet import (
    DEFAULT_LAYOUT,
    REGISTRY,
    Block,
    WorksheetError,
    create,
    from_json,
    revise,
)


@pytest.fixture
def store(tmp_path):
    return WorkspaceStore(tmp_path / "w.db")


@pytest.fixture
def worksheet():
    return create(worksheet_id="ws-1", owner_id="pilot",
                  scenario_ref="scenario/plan@1", primary_run_ref="run-1",
                  benchmark_run_refs=("bench-1", "bench-2"),
                  title="Monthly SPY", created_at="2026-08-01T00:00:00Z")


class TestItHoldsReferencesNotResults:

    def test_no_figure_is_copied_onto_the_worksheet(self, worksheet):
        """A worksheet holding a copy of a figure has two sources of truth for
        it, and the copy is the one that goes stale without saying so."""
        import json

        blob = json.dumps(worksheet.to_json())
        for word in ("return", "twr", "mwr", "drawdown", "final_value"):
            assert word not in blob.lower()

    def test_identity_is_the_references_and_the_layout(self, worksheet):
        renamed = create(worksheet_id="ws-1", owner_id="pilot",
                         scenario_ref="scenario/plan@1", primary_run_ref="run-1",
                         benchmark_run_refs=("bench-2", "bench-1"),
                         title="A completely different title",
                         created_at="2027-01-01T00:00:00Z")
        assert renamed.canonical_hash == worksheet.canonical_hash, (
            "two worksheets over the same artifacts are the same worksheet "
            "whenever they were made and whatever they are called")

    def test_changing_an_artifact_changes_identity(self, worksheet):
        moved = revise(worksheet, reason="dropped a benchmark",
                       benchmark_run_refs=("bench-1",))
        assert moved.canonical_hash != worksheet.canonical_hash

    def test_it_round_trips(self, worksheet):
        assert from_json(worksheet.to_json()).canonical_hash == \
            worksheet.canonical_hash


class TestRevisionsAreImmutable:

    def test_a_revision_names_its_parent(self, worksheet):
        second = revise(worksheet, reason="changed the benchmark set")
        assert second.revision == 2 and second.parent_revision == 1

    def test_an_unexplained_revision_is_refused(self, worksheet):
        """A history of unexplained changes cannot be reviewed, which is the
        only thing keeping revisions is for."""
        with pytest.raises(WorksheetError, match="gives no reason"):
            revise(worksheet, reason="")

    def test_a_later_revision_without_a_parent_is_refused(self):
        """A revision that cannot say what it came from is not a history."""
        from src.workspace.worksheet import ResearchWorksheet

        with pytest.raises(WorksheetError, match="names no parent"):
            ResearchWorksheet(worksheet_id="x", owner_id="p", revision=2,
                              scenario_ref="s")

    def test_storing_the_same_revision_twice_is_idempotent(self, store, worksheet):
        store.save_worksheet(worksheet)
        store.save_worksheet(worksheet)
        assert len(store.worksheet_revisions("ws-1", "pilot")) == 1

    def test_a_stored_revision_cannot_be_overwritten(self, store, worksheet):
        store.save_worksheet(worksheet)
        different = create(worksheet_id="ws-1", owner_id="pilot",
                           scenario_ref="scenario/other@1")
        with pytest.raises(NotSaveable, match="immutable"):
            store.save_worksheet(different)

    def test_every_revision_survives(self, store, worksheet):
        store.save_worksheet(worksheet)
        store.save_worksheet(revise(worksheet, reason="second",
                                    created_at="2026-08-02T00:00:00Z"))
        assert [r["revision"] for r in
                store.worksheet_revisions("ws-1", "pilot")] == [1, 2]
        assert store.get_worksheet("ws-1", "pilot")["revision"] == 2
        assert store.get_worksheet("ws-1", "pilot", 1)["revision"] == 1


class TestOwnership:

    def test_worksheets_are_scoped_by_owner_in_the_query(self, store, worksheet):
        """A get that fetches by id and checks ownership afterwards is one
        early return away from serving someone else's research."""
        store.save_worksheet(worksheet)
        assert store.get_worksheet("ws-1", "someone-else") is None
        assert store.worksheet_revisions("ws-1", "someone-else") == []


class TestTheBlockRegistry:

    def test_it_is_deliberately_small(self):
        """Python cells, markdown and plugin blocks are deferred: a worksheet
        nobody wants to modify does not need an extension mechanism, and
        building one first guarantees the wrong extension points."""
        assert len(REGISTRY) == 7
        assert set(DEFAULT_LAYOUT) == set(REGISTRY)

    def test_every_block_declares_what_it_needs(self):
        for block, spec in REGISTRY.items():
            assert spec.requires, block
            assert spec.title

    def test_results_are_never_shown_before_the_strategy(self):
        """A figure read before what produced it is read as a claim about the
        world rather than about one rule."""
        order = list(DEFAULT_LAYOUT)
        assert order.index(Block.STRATEGY_DEFINITION) < \
            order.index(Block.PERFORMANCE_SUMMARY)

    def test_scope_travels_with_the_result(self):
        """A figure read before its exclusions is read as excluding nothing."""
        order = list(DEFAULT_LAYOUT)
        assert order.index(Block.PERFORMANCE_SUMMARY) < \
            order.index(Block.MODELING_SCOPE)

    def test_comparability_precedes_performance(self):
        order = list(DEFAULT_LAYOUT)
        assert order.index(Block.BENCHMARK_COMPARISON) < \
            order.index(Block.PERFORMANCE_SUMMARY)

    def test_an_unmet_block_says_what_is_missing(self):
        """Named rather than skipped. An omitted panel is invisible; a panel
        that says "no run yet" is a fact."""
        bare = create(worksheet_id="ws-2", owner_id="pilot",
                      scenario_ref="scenario/plan@1")
        unavailable = bare.unavailable_blocks()
        assert Block.PERFORMANCE_SUMMARY.value in unavailable
        assert "primary_run_ref" in unavailable[Block.PERFORMANCE_SUMMARY.value]

    def test_a_complete_worksheet_omits_nothing(self, worksheet):
        assert worksheet.unavailable_blocks() == {}


class TestConfirmationTelemetry:
    """Structure now, conclusions later. The first sessions only happen once."""

    def test_events_record_what_changed_never_why(self, store):
        store.record_confirmation_event(
            event_id="e1", owner="pilot", occurred_at="2026-08-01T00:00:00Z",
            kind="confirmation_field_changed", path="CLARIFY",
            field="account_type", provenance="INFERRED",
            original_value="TAXABLE", final_value="ROTH",
            compiler_version="3", defaults_ref="compiler-defaults/us@1")
        event = store.confirmation_events("pilot")[0]
        assert event["field"] == "account_type"
        assert event["provenance"] == "INFERRED"
        assert event["reason"] is None, (
            "intent cannot be inferred from a value edit — misunderstood, "
            "changed my mind and had not said look identical here")

    def test_a_reason_is_recorded_only_when_asked(self, store):
        store.record_confirmation_event(
            event_id="e2", owner="pilot", occurred_at="t",
            kind="confirmation_field_changed", field="cadence",
            reason="I changed my mind")
        assert store.confirmation_events("pilot")[0]["reason"] == \
            "I changed my mind"

    def test_events_are_scoped_by_owner(self, store):
        store.record_confirmation_event(event_id="e3", owner="pilot",
                                        occurred_at="t", kind="scenario_saved")
        assert store.confirmation_events("someone-else") == []
