"""Turning a classification into a reviewable change, or refusing to.

    WorksheetIntent -> typed diff -> impact -> confirmation -> new revision

The guard that matters: **every proposed edit names the exact block field or
artifact it changes, or is marked unsupported with a reason.** A proposal that
classified successfully and produced prose would be recognition without
representation at the worksheet layer — the same defect as a compiler reading
"hold dividends as cash" and compiling a scenario with no trace of it.
"""
from __future__ import annotations

import pytest

from src.workspace.intent import EditEffect, SelectionBasis, plan
from src.workspace.proposal import WorksheetProposal, propose
from src.workspace.worksheet import Block, create

CASES = ["Move the scope panel below risk", "Add 63-day rolling volatility",
         "Try 21, 63 and 126 day windows", "Keep 63 because it looks smoothest",
         "Replace SPY with VTI", "Try SPY, VTI and VT and keep the best"]


@pytest.fixture
def worksheet():
    return create(worksheet_id="ws-1", owner_id="pilot", scenario_ref="plan-1",
                  primary_run_ref="run-1", benchmark_run_refs=("bench-1",))


@pytest.fixture
def proposals(worksheet):
    history, out = [], []
    for index, text in enumerate(CASES):
        intent = plan(text, intent_id=f"i{index}", source_revision=1,
                      history=history, target_run="run-1")
        history.append(intent)
        out.append(propose(intent, worksheet))
    return out


class TestLayoutOnly:
    """Move the scope panel."""

    def test_it_changes_only_the_order(self, worksheet):
        proposal = propose(plan("Move the provenance panel above results",
                                intent_id="x", source_revision=1), worksheet)
        assert [c.target for c in proposal.changes] == ["layout"]
        assert proposal.proposed_scenario_patch is None
        assert not proposal.touches_money

    def test_it_costs_nothing(self, proposals):
        proposal = proposals[0]
        assert proposal.trial_effect == 0
        assert proposal.rerun_required is False

    def test_the_new_order_is_stated_not_described(self, worksheet):
        """A diff that says "moved down" cannot be applied or reviewed."""
        proposal = propose(plan("Move the provenance panel above results",
                                intent_id="x", source_revision=1), worksheet)
        assert proposal.applicable
        assert proposal.proposed_layout
        assert proposal.changes[0].previous != proposal.changes[0].value

    def test_a_reorder_that_changes_nothing_is_refused(self, proposals):
        """The default layout already puts results above scope, so "move the
        scope panel below risk" is a no-op. A revision that changes nothing
        still costs a revision, and a diff of two identical lists asks a
        reviewer to spot a difference that is not there."""
        proposal = proposals[0]
        assert not proposal.applicable
        assert "already in that order" in proposal.unsupported[0].why


class TestDerivedAnalysis:
    """Add 63-day rolling volatility."""

    def test_it_adds_a_block_and_pins_nothing_new(self, proposals):
        proposal = proposals[1]
        assert proposal.changes[0].operation == "add_block"
        assert proposal.changes[0].value["metric"] == "volatility"
        assert proposal.proposed_scenario_patch is None

    def test_it_is_not_a_strategy_trial(self, proposals):
        assert proposals[1].trial_effect == 0
        assert proposals[1].rerun_required is False

    def test_the_parameters_are_recorded(self, proposals):
        assert proposals[1].changes[0].value["parameter"] == "63"


class TestVariantExploration:
    """Try 21, 63 and 126-day windows."""

    def test_every_variant_becomes_its_own_change(self, proposals):
        """Folding a search into one block showing the chosen window would
        delete the alternatives, which is the record trial accounting keeps."""
        proposal = proposals[2]
        assert len(proposal.changes) == 3
        assert {c.value["parameter"] for c in proposal.changes} == {"21", "63", "126"}

    def test_all_three_count(self, proposals):
        assert proposals[2].trial_effect == 3

    def test_no_winner_is_named(self, proposals):
        proposal = proposals[2]
        assert proposal.selection_basis == SelectionBasis.VARIANT_EXPLORATION.value
        assert all(c.operation == "add_block" for c in proposal.changes)


class TestAfterResultsSelection:
    """Keep 63; it looks smoothest."""

    def test_the_selection_activates_rather_than_replaces(self, proposals):
        """Rejected variants are not deleted — the history is the disclosure."""
        proposal = proposals[3]
        assert proposal.changes[0].operation == "activate"

    def test_the_basis_survives_into_the_proposal(self, proposals):
        assert proposals[3].selection_basis == SelectionBasis.AFTER_RESULTS.value

    def test_it_discloses_that_results_were_visible(self, proposals):
        warning = " ".join(proposals[3].warnings)
        assert "after seeing the results" in warning
        assert "look remarkable" in warning


class TestScenarioSubstitution:
    """Replace SPY with VTI."""

    def test_it_produces_a_typed_scenario_patch(self, proposals):
        proposal = proposals[4]
        assert proposal.proposed_scenario_patch == {
            "methodology.allocation_rule.assets": ["SPY", "VTI"]}
        assert proposal.touches_money

    def test_a_run_is_required_before_the_revision(self, proposals):
        proposal = proposals[4]
        assert proposal.rerun_required
        assert any("before the worksheet revision" in w for w in proposal.warnings)

    def test_it_counts_one_trial(self, proposals):
        assert proposals[4].trial_effect == 1

    def test_comparability_must_be_re_established(self, proposals):
        assert "comparability must be re-established" in \
            proposals[4].comparability_impact


class TestScenarioSearch:
    """Try SPY, VTI and VT, then keep the best."""

    def test_each_candidate_is_its_own_scenario(self, proposals):
        """Setting the holdings to all three would propose a three-asset
        strategy nobody described, and record one trial where three were run."""
        proposal = proposals[5]
        assert len(proposal.changes) == 3
        assert [c.value for c in proposal.changes] == [["SPY"], ["VTI"], ["VT"]]

    def test_all_three_count(self, proposals):
        assert proposals[5].trial_effect == 3

    def test_no_candidate_is_dropped(self, proposals):
        patch = proposals[5].proposed_scenario_patch
        assert patch["methodology.allocation_rule.assets"] == [["SPY"], ["VTI"],
                                                               ["VT"]]

    def test_it_is_marked_as_chosen_on_results(self, proposals):
        assert proposals[5].selection_basis == SelectionBasis.AFTER_RESULTS.value


class TestItRefusesWhatItCannotType:
    """The guard. Classified successfully is not the same as expressible."""

    @pytest.mark.parametrize("instruction", [
        "Move the sparkline widget",
        "Make the worksheet nicer",
        "Change the vibe",
    ])
    def test_an_unexpressible_request_is_refused_not_approximated(
            self, instruction, worksheet):
        proposal = propose(plan(instruction, intent_id="x", source_revision=1),
                           worksheet)
        assert not proposal.applicable
        assert proposal.unsupported
        assert proposal.unsupported[0].why

    def test_a_refusal_says_what_it_could_not_express(self, worksheet):
        proposal = propose(plan("Move the sparkline widget", intent_id="x",
                                source_revision=1), worksheet)
        assert "was not recognised" in proposal.unsupported[0].why

    def test_naming_the_nearest_block_is_refused_on_purpose(self, worksheet):
        """Approximating would move a panel the request never mentioned."""
        proposal = propose(plan("Move the sparkline widget", intent_id="x",
                                source_revision=1), worksheet)
        assert proposal.proposed_layout is None
        assert proposal.changes == ()

    def test_a_partially_expressible_edit_is_not_applied_partially(self):
        """One unsupported entry makes the whole proposal inapplicable."""
        from src.workspace.proposal import Change, Unsupported

        proposal = WorksheetProposal(
            intent_ref="x", source_revision=1, edit_effect="LAYOUT_ONLY",
            selection_basis="ANALYTICAL_ONLY", repetition_signature={},
            changes=(Change(target="layout", operation="reorder", value=[]),),
            unsupported=(Unsupported(what="the rest", why="not expressible"),))
        assert not proposal.applicable

    def test_a_block_absent_from_this_worksheet_is_refused(self):
        from src.workspace.worksheet import ResearchWorksheet

        bare = ResearchWorksheet(
            worksheet_id="ws-2", owner_id="pilot", revision=1,
            scenario_ref="plan-1", layout=(Block.STRATEGY_DEFINITION,))
        proposal = propose(plan("Move the scope panel below risk",
                                intent_id="x", source_revision=1), bare)
        assert not proposal.applicable
        assert "does not contain that block" in proposal.unsupported[0].why


class TestTheProposalIsAnOffer:

    def test_nothing_in_the_module_writes(self):
        import inspect

        from src.workspace import proposal

        source = inspect.getsource(proposal)
        for verb in ("def save", "def apply", "def commit", "store."):
            assert verb not in source, verb

    def test_layout_and_scenario_stay_in_separate_fields(self, proposals,
                                                         worksheet):
        """A financial change hidden inside a presentation request is the one a
        reviewer skims past."""
        layout_only = propose(plan("Move the provenance panel above results",
                                   intent_id="x", source_revision=1), worksheet)
        scenario = proposals[4]
        assert layout_only.proposed_layout is not None
        assert layout_only.proposed_scenario_patch is None
        assert scenario.proposed_scenario_patch is not None
        assert scenario.proposed_layout is None

    def test_it_serializes_whole(self, proposals):
        payload = proposals[4].to_json()
        assert payload["applicable"] is True
        assert payload["touches_money"] is True
        assert payload["repetition_signature"]["key"]
