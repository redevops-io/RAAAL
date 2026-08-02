"""The live intent path, proven with sequences rather than isolated calls.

The defect this closes: `intent.plan` has always taken history, and the live
application had none to give it. Every request arrived with an empty list, so a
user could try 21-, 63- and 126-day windows across three requests and each one
would classify as `ANALYTICAL_ONLY`. Trial accounting was implemented and could
not run.

Every test here drives the service the way requests do — a fresh store object
per instruction, standing in for a separate request or session — because a
history assembled inside one test function is exactly the fake sequence that
made the old tests pass.
"""
from __future__ import annotations

import pytest

from src.workspace.intent import EditEffect, SelectionBasis
from src.workspace.intent_history import ChainStatus, IntentHistory
from src.workspace.intent_service import (
    StaleInstruction,
    UntrustworthyHistory,
    plan_and_record,
)
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create

OWNER = "pilot"


@pytest.fixture
def workspace(tmp_path):
    """A database path, not a store. Each request opens its own connection."""
    path = tmp_path / "w.db"
    store = WorkspaceStore(path)
    store.save_worksheet(create(worksheet_id="ws-1", owner_id=OWNER,
                                scenario_ref="plan-1", primary_run_ref="run-0",
                                created_at="t0"))
    return path


def request(path, instruction, index, **kw):
    """One request, against its own connection."""
    return plan_and_record(
        WorkspaceStore(path), worksheet_id="ws-1", owner=OWNER,
        instruction=instruction, intent_id=f"i{index}",
        proposal_id=f"p{index}", at=f"t{index}", **kw)


def sequence(path, instructions):
    return [request(path, text, index)
            for index, text in enumerate(instructions)]


def history(path):
    return IntentHistory.from_store(WorkspaceStore(path), "ws-1", OWNER)


class TestAnalyticalEscalationAcrossSessions:
    """Three separate requests. The third must see the first two."""

    INSTRUCTIONS = ["Add 21-day rolling volatility",
                    "Add 63-day rolling volatility",
                    "Add 126-day rolling volatility"]

    def test_the_first_is_analytical(self, workspace):
        assert sequence(workspace, self.INSTRUCTIONS[:1])[0] \
            .intent.selection_basis is SelectionBasis.ANALYTICAL_ONLY

    def test_the_later_ones_escalate_to_variant_exploration(self, workspace):
        planned = sequence(workspace, self.INSTRUCTIONS)
        assert [p.intent.selection_basis for p in planned[1:]] == [
            SelectionBasis.VARIANT_EXPLORATION,
            SelectionBasis.VARIANT_EXPLORATION]

    def test_they_form_one_repetition_family(self, workspace):
        planned = sequence(workspace, self.INSTRUCTIONS)
        keys = {p.intent.repetition_signature.key() for p in planned}
        assert len(keys) == 1

    def test_each_request_sees_the_ones_before_it(self, workspace):
        planned = sequence(workspace, self.INSTRUCTIONS)
        assert [len(p.history.intents) for p in planned] == [0, 1, 2]

    def test_three_windows_count_three_trials(self, workspace):
        """Summed per intent this is two: the first window was analytical when
        it arrived and contributed nothing, and the sequence only became a
        search once the second landed. The value was still evaluated."""
        sequence(workspace, self.INSTRUCTIONS)
        assert history(workspace).trial_total == 3


class TestAfterResultsSelection:

    def test_keeping_one_having_seen_them_is_after_results(self, workspace):
        planned = sequence(workspace, [
            "Add 21-day rolling volatility",
            "Add 63-day rolling volatility",
            "Add 126-day rolling volatility",
            "Keep 63 because it looks smoothest"])
        assert planned[-1].intent.selection_basis is SelectionBasis.AFTER_RESULTS

    def test_it_is_not_analytical_only(self, workspace):
        """The reading a requester has the least incentive to declare."""
        planned = sequence(workspace, ["Add 63-day rolling volatility",
                                       "Keep 63 because it looks smoothest"])
        assert planned[-1].intent.selection_basis is not SelectionBasis.ANALYTICAL_ONLY

    def test_the_selection_does_not_erase_the_alternatives(self, workspace):
        sequence(workspace, ["Add 21-day rolling volatility",
                             "Add 63-day rolling volatility",
                             "Add 126-day rolling volatility",
                             "Keep 63 because it looks smoothest"])
        assert history(workspace).trial_total == 3


class TestRephrasingCannotResetTheChain:
    """The repetition signature keys the analytical decision, not the wording.
    Otherwise repeated tuning hides behind a thesaurus."""

    def test_differently_worded_window_requests_share_a_family(self, workspace):
        planned = sequence(workspace, [
            "Add 63-day rolling volatility",
            "Show 21-day rolling volatility",
            "Plot 126-day rolling volatility"])
        keys = {p.intent.repetition_signature.key() for p in planned}
        assert len(keys) == 1, keys

    def test_rewording_does_not_return_to_analytical_only(self, workspace):
        planned = sequence(workspace, ["Add 63-day rolling volatility",
                                       "Show 21-day rolling volatility"])
        assert planned[-1].intent.selection_basis is SelectionBasis.VARIANT_EXPLORATION

    def test_a_different_metric_is_a_different_family(self, workspace):
        """The signature must not be so loose that unrelated work merges."""
        planned = sequence(workspace, ["Add 63-day rolling volatility",
                                       "Add rolling drawdown"])
        assert (planned[0].intent.repetition_signature.key()
                != planned[1].intent.repetition_signature.key())


class TestReopenAndContinue:

    def test_classification_survives_closing_the_session(self, workspace):
        """Every request here already uses a fresh connection; this asserts the
        property directly rather than relying on that being noticed."""
        sequence(workspace, ["Add 21-day rolling volatility",
                             "Add 63-day rolling volatility"])

        reopened = request(workspace, "Add 126-day rolling volatility", 9)
        assert reopened.intent.selection_basis is SelectionBasis.VARIANT_EXPLORATION
        assert len(reopened.history.intents) == 2

    def test_the_total_is_re_derivable_from_storage_alone(self, workspace):
        sequence(workspace, ["Add 21-day rolling volatility",
                             "Add 63-day rolling volatility",
                             "Add 126-day rolling volatility"])
        assert history(workspace).trial_total == 3


class TestOwnerAndWorksheetScoping:

    def test_another_worksheets_intents_are_not_in_this_chain(self, workspace):
        store = WorkspaceStore(workspace)
        store.save_worksheet(create(worksheet_id="ws-2", owner_id=OWNER,
                                    scenario_ref="plan-1",
                                    primary_run_ref="run-0", created_at="t0"))
        sequence(workspace, ["Add 21-day rolling volatility"])
        plan_and_record(WorkspaceStore(workspace), worksheet_id="ws-2",
                        owner=OWNER, instruction="Add 63-day rolling volatility",
                        intent_id="other", proposal_id="pother", at="t9")

        assert len(history(workspace).intents) == 1

    def test_another_owner_cannot_appear_in_this_chain(self, workspace):
        """Two owners with a same-named worksheet. The other owner's intents
        must not join this chain — they would change the classification of the
        next instruction and inflate a trial total that is not this user's."""
        # Written straight to the store. The `worksheet` table's primary key is
        # (worksheet_id, revision) with no owner, so a second owner cannot hold
        # a same-named worksheet to plan against — a separate boundary problem,
        # recorded rather than worked around. The leak under test is in the
        # intent query, which this reaches directly.
        from src.workspace.intent import plan as plan_intent

        store = WorkspaceStore(workspace)
        for index, text in enumerate(["Add 21-day rolling volatility",
                                      "Add 63-day rolling volatility"]):
            store.append_worksheet_intent(
                worksheet_id="ws-1", owner="someone-else",
                intent=plan_intent(text, intent_id=f"theirs-{index}",
                                   source_revision=1),
                created_at=f"t{index}", planner_version="1",
                instruction_hash="x")

        mine = request(workspace, "Add 126-day rolling volatility", 0)

        assert mine.history.intents == ()
        assert [row["intent_id"] for row in WorkspaceStore(workspace)
                .worksheet_intents("ws-1", OWNER)] == ["i0"]
        # Their two windows must not make my first instruction look like the
        # third step of a search.
        assert mine.intent.selection_basis is SelectionBasis.ANALYTICAL_ONLY
        assert history(workspace).trial_total == 0


class TestStaleRevision:

    def test_an_instruction_against_an_old_revision_is_refused(self, workspace):
        store = WorkspaceStore(workspace)
        from src.workspace.worksheet import from_json, revise

        current = from_json(store.get_worksheet("ws-1", OWNER)["payload"])
        store.save_worksheet(revise(current, reason="advanced", created_at="t1"))

        with pytest.raises(StaleInstruction, match="revision"):
            request(workspace, "Add 63-day rolling volatility", 5,
                    source_revision=current.revision)

    def test_nothing_is_recorded_when_it_is_refused(self, workspace):
        from src.workspace.worksheet import from_json, revise

        store = WorkspaceStore(workspace)
        current = from_json(store.get_worksheet("ws-1", OWNER)["payload"])
        store.save_worksheet(revise(current, reason="advanced", created_at="t1"))

        with pytest.raises(StaleInstruction):
            request(workspace, "Add 63-day rolling volatility", 5,
                    source_revision=current.revision)
        assert history(workspace).intents == ()


class TestHistoryTampering:
    """Editing or removing a prior intent must invalidate the derived total —
    visibly, not by producing a quietly smaller number."""

    def test_editing_a_stored_classification_breaks_the_chain(self, workspace):
        sequence(workspace, ["Add 21-day rolling volatility",
                             "Add 63-day rolling volatility",
                             "Add 126-day rolling volatility"])
        assert history(workspace).verdict.trustworthy

        import json
        import sqlite3

        # Sequence 2, not 1. The first intent is already ANALYTICAL_ONLY with
        # zero trials, so "downgrading" it changes nothing and would leave this
        # test asserting that an unmodified chain verifies.
        with sqlite3.connect(workspace) as conn:
            row = conn.execute("SELECT structured_request FROM worksheet_intent "
                               "WHERE sequence = 2").fetchone()
            payload = json.loads(row[0])
            assert payload["trial_effect"] != 0, "fixture no longer tampers"

            payload["trial_effect"] = 0
            payload["selection_basis"] = "ANALYTICAL_ONLY"
            conn.execute("UPDATE worksheet_intent SET structured_request = ? "
                         "WHERE sequence = 2", (json.dumps(payload),))

        verdict = history(workspace).verdict
        assert not verdict.trustworthy
        assert verdict.status is ChainStatus.BROKEN_LINK

    def test_deleting_an_intent_is_detected_as_a_gap(self, workspace):
        sequence(workspace, ["Add 21-day rolling volatility",
                             "Add 63-day rolling volatility",
                             "Add 126-day rolling volatility"])

        import sqlite3

        with sqlite3.connect(workspace) as conn:
            conn.execute("DELETE FROM worksheet_intent WHERE sequence = 2")

        assert history(workspace).verdict.status is ChainStatus.MISSING_LINK

    def test_nothing_is_planned_against_a_broken_chain(self, workspace):
        """Refused rather than planned against a partial history. Continuing
        would produce a classification derived from a doctored chain that looks
        exactly like one that was not."""
        sequence(workspace, ["Add 21-day rolling volatility",
                             "Add 63-day rolling volatility"])

        import sqlite3

        with sqlite3.connect(workspace) as conn:
            conn.execute("DELETE FROM worksheet_intent WHERE sequence = 1")

        with pytest.raises(UntrustworthyHistory, match="does not verify"):
            request(workspace, "Add 126-day rolling volatility", 7)


class TestItPlansAndNeverApplies:

    def test_the_worksheet_revision_is_unchanged(self, workspace):
        before = WorkspaceStore(workspace).get_worksheet("ws-1", OWNER)["revision"]
        sequence(workspace, ["Add 63-day rolling volatility"])
        after = WorkspaceStore(workspace).get_worksheet("ws-1", OWNER)["revision"]
        assert after == before

    def test_the_proposal_is_recorded_as_proposed(self, workspace):
        planned = sequence(workspace, ["Add 63-day rolling volatility"])[0]
        record = WorkspaceStore(workspace).get_worksheet_proposal(
            planned.proposal_id, OWNER)
        assert record["status"] == "PROPOSED"

    def test_the_intent_is_linked_to_its_proposal(self, workspace):
        planned = sequence(workspace, ["Add 63-day rolling volatility"])[0]
        [row] = WorkspaceStore(workspace).worksheet_intents("ws-1", OWNER)
        assert row["proposal_id"] == planned.proposal_id


class TestThePrivacyBoundary:

    def test_the_raw_instruction_is_not_stored_by_default(self, workspace):
        """The durable record is the classification and the hash. The sentence
        may carry holdings, salary or employer detail and has a shorter life."""
        sequence(workspace, ["Add 63-day rolling volatility for my Tesla RSUs"])
        [row] = WorkspaceStore(workspace).worksheet_intents("ws-1", OWNER)
        assert row["instruction"] is None
        assert row["instruction_hash"]

    def test_the_hash_still_distinguishes_two_instructions(self, workspace):
        planned = sequence(workspace, ["Add 63-day rolling volatility",
                                       "Add rolling drawdown"])
        rows = WorkspaceStore(workspace).worksheet_intents("ws-1", OWNER)
        assert rows[0]["instruction_hash"] != rows[1]["instruction_hash"]
        del planned

    def test_it_can_be_kept_when_explicitly_asked_for(self, workspace):
        request(workspace, "Add 63-day rolling volatility", 0,
                store_instruction=True)
        [row] = WorkspaceStore(workspace).worksheet_intents("ws-1", OWNER)
        assert row["instruction"] == "Add 63-day rolling volatility"
