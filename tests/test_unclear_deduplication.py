"""A phrase is not "unplaceable" when the page already asks about it properly.

From the live pilot: "What did you mean by 'account type in which the
purchases occur is not specified'?" — offering only "continue without
modelling it" — rendered directly beneath the account-type question and its
five radio buttons. Same concept twice, once answerable and once only
dismissible, with nothing to tell the user which one would take effect.

Asking somebody to abandon a detail the page is simultaneously offering to
settle is worse than either alone.
"""
from __future__ import annotations

import pytest

from src.mission.compiler import _CONCEPT_MARKERS, _covered_by

DUPLICATES = [
    ("whether '200 DMA' refers to a simple or exponential moving average",
     "moving_average_kind"),
    ("whether the trigger is a one-time crossing below the 200 DMA or a "
     "persistent condition (buying repeatedly while price stays below it)",
     "trigger_semantics"),
    ("cadence or frequency of the $1000 purchases is not specified beyond "
     "the trigger condition", "cadence"),
    ("source of the $1000 (new contribution vs. existing cash) is not "
     "specified", "funding_source"),
    ("account type in which the purchases occur is not specified",
     "account_type"),
]

GENUINELY_UNPLACEABLE = [
    "request for calculated total return and accumulated amount over the "
    "past 5 years is a computation request, not a plan attribute",
    "SP500 etf (ticker not specified)",
    "I would like to feel calmer about money",
]


class TestADuplicateIsSuppressed:
    @pytest.mark.parametrize("phrase,field", DUPLICATES)
    def test_the_phrase_maps_to_the_field_that_owns_it(self, phrase, field):
        assert _covered_by(phrase, {field}) == field


class TestOnlyWhenTheControlIsActuallyThere:
    """The safety property, and the one that makes this conservative.

    Suppressing on the marker alone would drop a question whenever the model
    happened to phrase something familiarly, whether or not the user had any
    way to settle it — silently removing the only mention of a detail.
    """

    @pytest.mark.parametrize("phrase,field", DUPLICATES)
    def test_nothing_is_suppressed_when_the_field_is_not_raised(self, phrase, field):
        assert _covered_by(phrase, set()) is None

    def test_nor_when_a_different_field_is_raised(self, phrase="account type in "
                                                  "which the purchases occur is "
                                                  "not specified"):
        assert _covered_by(phrase, {"dividends", "cadence"}) is None


class TestGenuinelyUnplaceableProseSurvives:
    @pytest.mark.parametrize("phrase", GENUINELY_UNPLACEABLE)
    def test_it_is_kept(self, phrase):
        """The other direction. Suppressing everything would remove the
        acknowledgement that lets a user proceed past real extra prose."""
        assert _covered_by(phrase, set(_CONCEPT_MARKERS)) is None


class TestTheCompilerDropsTheDuplicate:
    """Driven with a parse that actually carries the phrases.

    The deterministic compiler emits no `unclear` items for ordinary prose —
    they come from the model. Written against a plain description, this test
    passed with the suppression removed *and* with it forced on, because
    there was nothing to suppress. It now supplies the parse directly.
    """

    TEXT = ("I buy $1000 of VOO every time it crosses below its 200 DMA, "
            "in a taxable account.")

    def compiled_with(self, unclear):
        from src.mission.compiler import ParsedUtterance, compile_scenario

        parsed = ParsedUtterance(text=self.TEXT, unclear=tuple(unclear))
        return compile_scenario(self.TEXT, parsed=parsed).scenario

    def test_a_duplicated_concept_is_not_asked_twice(self):
        phrases = [phrase for phrase, _ in DUPLICATES]
        scenario = self.compiled_with(phrases)
        fields = {one.field for one in scenario.provenance.unresolved}

        for field in fields:
            if field.startswith("unclear:"):
                assert _covered_by(field[len("unclear:"):], fields) is None, (
                    f"{field} duplicates a structured question on the same page")

    def test_the_concepts_are_still_reachable(self):
        """Suppressing must not remove the user's route to the detail. Every
        phrase dropped has to leave its structured question standing."""
        phrases = [phrase for phrase, _ in DUPLICATES]
        scenario = self.compiled_with(phrases)
        fields = {one.field for one in scenario.provenance.unresolved}
        inferred = {one.field for one in scenario.provenance.inferred}
        reachable = fields | inferred

        for phrase, field in DUPLICATES:
            still_asked = f"unclear:{phrase}" in fields
            assert still_asked or field in reachable, (
                f"{field} was suppressed and is now asked nowhere")

    def test_unplaceable_prose_still_becomes_a_question(self):
        scenario = self.compiled_with(GENUINELY_UNPLACEABLE)
        fields = {one.field for one in scenario.provenance.unresolved}
        for phrase in GENUINELY_UNPLACEABLE:
            assert f"unclear:{phrase}" in fields, (
                f"{phrase} was dropped with no control anywhere")
