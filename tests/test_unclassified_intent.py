""""We did not understand this" is a state, not a layout edit.

An unrecognised instruction used to fall through to `LAYOUT_ONLY` with zero
trials. It never applied — `propose` refused it — but it was still a semantic
claim, and the most permissive one available. Downstream, a parser failure and a
genuine "move the scope panel" were the same row, so the question that matters
most for the language work — what fraction of requests does the planner fail to
read? — had no answer.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.workspace.intent import EditEffect, SelectionBasis, plan
from src.workspace.intent_history import IntentHistory
from src.workspace.intent_service import plan_and_record
from src.workspace.proposal import propose
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create

OWNER = "pilot"

UNREADABLE = ["Make it better somehow", "asdf qwer",
              "Do something clever with the numbers"]
GENUINE_LAYOUT = ["Move the scope panel below the summary",
                  "Hide the benchmark block"]


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "w.db"
    WorkspaceStore(path).save_worksheet(
        create(worksheet_id="ws-1", owner_id=OWNER, scenario_ref="plan-1",
               primary_run_ref="run-0", created_at="t0"))
    return path


def request(path, instruction, index):
    return plan_and_record(WorkspaceStore(path), worksheet_id="ws-1",
                           owner=OWNER, instruction=instruction,
                           intent_id=f"i{index}", proposal_id=f"p{index}",
                           at=f"t{index}")


class TestItIsItsOwnState:

    @pytest.mark.parametrize("text", UNREADABLE)
    def test_an_unreadable_instruction_is_unclassified(self, text):
        assert plan(text, intent_id="a", source_revision=1).edit_effect \
            is EditEffect.UNCLASSIFIED

    @pytest.mark.parametrize("text", GENUINE_LAYOUT)
    def test_a_genuine_layout_edit_is_still_layout_only(self, text):
        """The two must stay distinguishable, or the distinction buys nothing."""
        assert plan(text, intent_id="a", source_revision=1).edit_effect \
            is EditEffect.LAYOUT_ONLY

    def test_the_two_are_not_the_same_row(self):
        unreadable = plan(UNREADABLE[0], intent_id="a", source_revision=1)
        layout = plan(GENUINE_LAYOUT[0], intent_id="b", source_revision=1)
        assert unreadable.edit_effect is not layout.edit_effect

    def test_it_is_reported_as_unclassified(self):
        intent = plan(UNREADABLE[0], intent_id="a", source_revision=1)
        assert intent.classified is False
        assert intent.to_json()["classified"] is False


class TestNothingIsAssumedAboutIt:

    def test_the_trial_effect_is_unknown_not_zero(self):
        """An unread instruction may have asked for one chart or forty
        parameters. Zero answers a question nobody can answer."""
        assert plan(UNREADABLE[0], intent_id="a",
                    source_revision=1).trial_effect is None

    def test_it_asks_the_user_rather_than_guessing(self):
        assert plan(UNREADABLE[0], intent_id="a",
                    source_revision=1).requires_user_confirmation is True

    def test_the_comparability_impact_is_stated_as_unknown(self):
        assert "unknown" in plan(UNREADABLE[0], intent_id="a",
                                 source_revision=1).comparability_impact

    def test_it_produces_no_repetition_family(self):
        """A signature invented for an unread instruction would merge unrelated
        requests into one family and inflate its repetition count."""
        signature = plan(UNREADABLE[0], intent_id="a",
                         source_revision=1).repetition_signature
        assert signature.metric == ""
        assert signature.block_type == ""


class TestItStillRefusesToApply:

    @pytest.mark.parametrize("text", UNREADABLE)
    def test_the_proposal_is_inapplicable(self, text):
        worksheet = create(worksheet_id="w", owner_id=OWNER, scenario_ref="s",
                           primary_run_ref="r", created_at="t")
        assert not propose(plan(text, intent_id="a", source_revision=1),
                           worksheet).applicable

    def test_the_refusal_says_it_was_not_recognised(self):
        """Not "a '' change" — a message assembled from a parameter family that
        was never populated, for a request nobody read."""
        worksheet = create(worksheet_id="w", owner_id=OWNER, scenario_ref="s",
                           primary_run_ref="r", created_at="t")
        proposal = propose(plan(UNREADABLE[0], intent_id="a", source_revision=1),
                           worksheet)
        assert "was not recognised" in proposal.unsupported[0].why
        assert "nothing was assumed" in proposal.unsupported[0].why.lower()


class TestAProtectiveSignalSurvivesAnUnreadableTarget:
    """Failing to read *what* an instruction edits is no reason to discard
    evidence about *why* it was chosen."""

    def test_choosing_on_results_is_recorded_even_when_unclassified(self):
        intent = plan("Keep whichever looks best", intent_id="a",
                      source_revision=1)
        assert intent.edit_effect is EditEffect.UNCLASSIFIED
        assert intent.selection_basis is SelectionBasis.AFTER_RESULTS

    def test_a_stated_preference_survives_too(self):
        intent = plan("Do that because that's my rule", intent_id="a",
                      source_revision=1)
        assert intent.selection_basis is SelectionBasis.STATED_PREFERENCE

    def test_only_an_illegible_basis_reports_unknown(self):
        assert plan("asdf qwer", intent_id="a",
                    source_revision=1).selection_basis is SelectionBasis.UNKNOWN


class TestItSurvivesStorage:

    def test_an_unknown_trial_effect_persists_as_null(self, workspace):
        request(workspace, UNREADABLE[0], 0)
        with sqlite3.connect(workspace) as conn:
            [(stored,)] = conn.execute(
                "SELECT trial_effect FROM worksheet_intent").fetchall()
        assert stored is None

    def test_it_rehydrates_as_unknown_not_zero(self, workspace):
        request(workspace, UNREADABLE[0], 0)
        [one] = IntentHistory.from_store(
            WorkspaceStore(workspace), "ws-1", OWNER).intents
        assert one.trial_effect is None

    def test_it_does_not_break_the_chain(self, workspace):
        request(workspace, "Add 21-day rolling volatility", 0)
        request(workspace, UNREADABLE[0], 1)
        request(workspace, "Add 63-day rolling volatility", 2)
        assert IntentHistory.from_store(
            WorkspaceStore(workspace), "ws-1", OWNER).verdict.trustworthy

    def test_it_does_not_reset_the_family_it_interrupts(self, workspace):
        """An unread instruction between two window requests must not make the
        second look like a first."""
        request(workspace, "Add 21-day rolling volatility", 0)
        request(workspace, UNREADABLE[0], 1)
        third = request(workspace, "Add 63-day rolling volatility", 2)
        assert third.intent.selection_basis is SelectionBasis.VARIANT_EXPLORATION


class TestTheTotalSaysWhenItIsIncomplete:

    def test_an_unclassified_request_is_counted_separately(self, workspace):
        request(workspace, "Add 21-day rolling volatility", 0)
        request(workspace, UNREADABLE[0], 1)
        history = IntentHistory.from_store(WorkspaceStore(workspace), "ws-1",
                                           OWNER)
        assert history.unclassified_count == 1
        assert history.total_is_complete is False

    def test_a_fully_classified_chain_is_complete(self, workspace):
        request(workspace, "Add 21-day rolling volatility", 0)
        request(workspace, "Add 63-day rolling volatility", 1)
        history = IntentHistory.from_store(WorkspaceStore(workspace), "ws-1",
                                           OWNER)
        assert history.unclassified_count == 0
        assert history.total_is_complete is True

    def test_the_unknown_is_not_absorbed_into_the_total(self, workspace):
        """It contributes nothing to executed trials — it never ran — and the
        uncertainty is reported rather than hidden inside a whole-looking
        number."""
        request(workspace, "Add 21-day rolling volatility", 0)
        request(workspace, "Add 63-day rolling volatility", 1)
        request(workspace, UNREADABLE[0], 2)
        history = IntentHistory.from_store(WorkspaceStore(workspace), "ws-1",
                                           OWNER)
        assert history.trial_total == 2
        assert not history.total_is_complete


class TestTheMigration:
    """The table shipped one commit ago with `trial_effect INTEGER NOT NULL`.
    An existing database would raise on the first unclassified instruction while
    a fresh checkout worked."""

    def test_an_old_database_is_rebuilt_and_keeps_its_rows(self, tmp_path):
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.executescript("""
        CREATE TABLE worksheet_intent (
            intent_id TEXT PRIMARY KEY, worksheet_id TEXT NOT NULL,
            owner TEXT NOT NULL, source_revision INTEGER NOT NULL,
            sequence INTEGER NOT NULL, instruction TEXT,
            instruction_hash TEXT NOT NULL, structured_request TEXT NOT NULL,
            edit_effect TEXT NOT NULL, selection_basis TEXT NOT NULL,
            repetition_signature TEXT NOT NULL, related_prior TEXT NOT NULL,
            results_visible INTEGER NOT NULL, alternatives INTEGER NOT NULL,
            trial_effect INTEGER NOT NULL, planner_version TEXT NOT NULL,
            chain_hash TEXT NOT NULL, created_at TEXT NOT NULL,
            proposal_id TEXT, status TEXT NOT NULL);
        INSERT INTO worksheet_intent VALUES
          ('kept','ws','p',1,1,NULL,'h','{}','LAYOUT_ONLY','ANALYTICAL_ONLY',
           'k','[]',1,0,0,'1','ch','t0',NULL,'PLANNED');
        """)
        conn.commit()
        conn.close()

        store = WorkspaceStore(path)
        columns = {c[1]: c[3] for c in
                   sqlite3.connect(path).execute(
                       "PRAGMA table_info(worksheet_intent)")}
        assert not columns["trial_effect"], "NOT NULL was not relaxed"
        assert [r["intent_id"] for r in store.worksheet_intents("ws", "p")] \
            == ["kept"]
