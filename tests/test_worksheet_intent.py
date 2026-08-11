"""WorksheetIntent: what was asked, classified before anything changes.

The planner classifies — not the model, not the template — because two of the
things it decides are trial accounting and comparability impact. A system that
lets the requester decide how many trials their request counted for has no trial
accounting at all.

Two axes, orthogonal on purpose. What the edit touches is a different question
from why it was chosen, and the six acceptance cases below exist because every
pairing of the two occurs in practice.
"""
from __future__ import annotations

import pytest

from src.workspace.intent import (
    EditEffect,
    SelectionBasis,
    WorksheetIntent,
    classify_effect,
    plan,
    signature_for,
)

#: The six boundaries. Each names a pairing that a single-axis model collapses.
ACCEPTANCE = [
    ("Move the scope panel below risk",
     EditEffect.LAYOUT_ONLY, SelectionBasis.ANALYTICAL_ONLY, 0, False),
    ("Add 63-day rolling volatility",
     EditEffect.DERIVED_ANALYSIS, SelectionBasis.ANALYTICAL_ONLY, 0, False),
    ("Try 21, 63 and 126 day windows",
     EditEffect.DERIVED_ANALYSIS, SelectionBasis.VARIANT_EXPLORATION, 3, False),
    ("Keep 63 because it looks smoothest",
     EditEffect.DERIVED_ANALYSIS, SelectionBasis.AFTER_RESULTS, 1, False),
    ("Replace SPY with VTI",
     EditEffect.SCENARIO_CHANGE, SelectionBasis.STATED_PREFERENCE, 1, True),
    ("Try SPY, VTI and VT and keep the best",
     EditEffect.SCENARIO_CHANGE, SelectionBasis.AFTER_RESULTS, 3, True),
]


def sequence(instructions):
    history = []
    for index, text in enumerate(instructions):
        history.append(plan(text, intent_id=f"i{index}", source_revision=1,
                            history=history, target_run="run/abc"))
    return history


class TestTheSixAcceptanceCases:
    """Run in order, because four of them only mean anything in sequence."""

    @pytest.fixture(scope="class")
    def planned(cls):
        return sequence([case[0] for case in ACCEPTANCE])

    @pytest.mark.parametrize("index", range(len(ACCEPTANCE)))
    def test_case(self, planned, index):
        text, effect, basis, trials, rerun = ACCEPTANCE[index]
        intent = planned[index]
        assert intent.edit_effect is effect, text
        assert intent.selection_basis is basis, text
        assert intent.trial_effect == trials, text
        assert intent.rerun_required is rerun, text


class TestTheAxesAreIndependent:

    def test_two_effects_share_one_basis(self):
        """"Move the panel" and "add a chart" are both analytical, and only one
        creates a derived artifact."""
        layout, derived = sequence(["Move the scope panel below risk",
                                    "Add 63-day rolling volatility"])
        assert layout.selection_basis is derived.selection_basis
        assert layout.edit_effect is not derived.edit_effect

    def test_two_bases_share_one_effect(self):
        """"Replace SPY with VTI" and "try three and keep the best" are both
        scenario changes, and only one counts three trials."""
        stated = plan("Replace SPY with VTI", intent_id="a", source_revision=1)
        chosen = plan("Try SPY, VTI and VT and keep the best", intent_id="b",
                      source_revision=1)
        assert stated.edit_effect is chosen.edit_effect
        assert stated.trial_effect == 1 and chosen.trial_effect == 3


class TestHistoryChangesTheAnswer:
    """Selection basis cannot be read from one instruction."""

    def test_the_same_words_classify_differently_after_a_search(self):
        first = plan("Add 63-day rolling volatility", intent_id="a",
                     source_revision=1)
        assert first.selection_basis is SelectionBasis.ANALYTICAL_ONLY

        history = sequence(["Add 63-day rolling volatility",
                            "Try 21, 63 and 126 day windows"])
        assert history[-1].selection_basis is SelectionBasis.VARIANT_EXPLORATION

    def test_a_follow_up_inherits_what_it_continues(self):
        """"Try 21, 63 and 126 day windows" names no metric and no instrument.
        Read alone it looks like nothing; read after the request it continues it
        is obviously more of the same."""
        # Alone it is UNCLASSIFIED, not LAYOUT_ONLY. Reading an unrecognised
        # instruction as "presentation, zero trials" is a claim, and it is the
        # most permissive one available.
        assert classify_effect("Try 21, 63 and 126 day windows") is \
            EditEffect.UNCLASSIFIED

        history = sequence(["Add 63-day rolling volatility",
                            "Try 21, 63 and 126 day windows"])
        assert history[-1].edit_effect is EditEffect.DERIVED_ANALYSIS

    def test_rephrasing_does_not_escape_the_repetition_family(self):
        """Three differently-worded requests against one metric are one family,
        or repeated tuning hides behind rephrasing."""
        history = sequence(["Add 63-day rolling volatility",
                            "Try 21, 63 and 126 day windows",
                            "Show me the 252 day window"])
        keys = {i.repetition_signature.key() for i in history}
        assert len(keys) == 1
        assert history[-1].related_prior_intents == ("i0", "i1")


class TestTrialAccounting:

    def test_layout_never_counts(self):
        assert plan("Move the scope panel below risk", intent_id="a",
                    source_revision=1).trial_effect == 0

    def test_a_first_diagnostic_does_not_count(self):
        """A drawdown chart is not a strategy trial."""
        assert plan("Add a drawdown chart", intent_id="a",
                    source_revision=1).trial_effect == 0

    def test_every_variant_counts_even_before_one_is_kept(self):
        """A search that has not finished is still a search."""
        history = sequence(["Add 63-day rolling volatility",
                            "Try 21, 63 and 126 day windows"])
        assert history[-1].trial_effect == 3

    def test_a_substitution_is_one_change_not_two(self):
        """"Replace SPY with VTI" names what is leaving and what is arriving.
        Counting both would charge a user for the holding they are removing."""
        assert plan("Replace SPY with VTI", intent_id="a",
                    source_revision=1).trial_effect == 1

    def test_choosing_on_results_is_never_reported_as_analytical(self):
        """The reading a requester has the least incentive to declare."""
        for text in ("Keep 63 because it looks smoothest",
                     "Use whichever performs best",
                     "Show me the best window"):
            intent = plan(text, intent_id="a", source_revision=1)
            assert intent.selection_basis is SelectionBasis.AFTER_RESULTS, text


class TestThePlannerDecidesAndDoesNotAct:

    def test_nothing_in_the_module_writes(self):
        import inspect

        from src.workspace import intent

        source = inspect.getsource(intent)
        for verb in ("def save", "def apply", "def commit", "store.",
                     "def write"):
            assert verb not in source, verb

    def test_a_scenario_change_declares_its_comparability_impact(self):
        intent = plan("Replace SPY with VTI", intent_id="a", source_revision=1)
        assert "comparability must be re-established" in intent.comparability_impact
        assert intent.rerun_required

    def test_a_variant_search_does_not_change_comparability(self):
        """Added trials affect deflation, not whether figures may be read
        together."""
        history = sequence(["Add 63-day rolling volatility",
                            "Try 21, 63 and 126 day windows"])
        assert "deflation" in history[-1].comparability_impact
        assert not history[-1].rerun_required

    def test_the_intent_serializes_whole(self):
        payload = plan("Replace SPY with VTI", intent_id="a",
                       source_revision=2).to_json()
        assert payload["edit_effect"] == "SCENARIO_CHANGE"
        assert payload["repetition_signature"]["key"]
        assert payload["source_revision"] == 2


class TestTheRecognisersAreNotAccidentallyBroad:

    def test_a_short_word_is_not_a_ticker(self):
        """Under IGNORECASE `[A-Z]{2,5}` matches any short word, and "try 21,
        63 and 126 day windows" read "day" as an instrument."""
        assert classify_effect("Try 21, 63 and 126 day windows") is not \
            EditEffect.SCENARIO_CHANGE

    def test_a_lowercase_verb_still_matches(self):
        assert classify_effect("replace SPY with VTI") is EditEffect.SCENARIO_CHANGE
        assert classify_effect("Replace SPY with VTI") is EditEffect.SCENARIO_CHANGE

    def test_a_scenario_input_named_in_words_is_recognised(self):
        for text in ("Change the contribution to $3,000",
                     "Hold dividends as cash instead",
                     "Use my Roth account"):
            assert classify_effect(text) is EditEffect.SCENARIO_CHANGE, text
