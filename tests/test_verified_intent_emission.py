"""Today's compiler, emitting the Discovery → Mission contract.

Two things are being proven, and neither is "the compiler works":

    the contract carries what the engine needs, before any model exists
    the reader that produced an intent is recorded, before it is replaced

The second is why this is worth doing now. When Discovery replaces the regex
compiler, every intent already in the record will carry
`produced_by: quantify-compiler@N`, so the two eras are distinguishable. Adding
the field afterwards would mean guessing which intents came from which reader.
"""
from __future__ import annotations

import pytest

from runtime_contracts import Author, OpenReason
from src.mission.compiler import compile_scenario, parse
from src.mission.verified_intent import (
    READER_VERSION,
    derivation,
    executable_check,
    from_compiled,
)

RULE = "benchmark-policy/public-default@1"

CROSSING = ("I buy $1,000 of SPY whenever it crosses below its 200-day moving "
            "average, over the past 5 years.")
SIXTY_FORTY = ("I put $1,000 every year into VTI and BND in a taxable brokerage "
               "account, holding 60/40, over the past 5 years.")
INVERSE_VOL = ("I allocate $100,000 across stocks and bonds and gold by inverse "
               "volatility using 63d vol, over the past 5 years.")


def intent_for(text: str):
    parsed = parse(text)
    result = compile_scenario(text, name="t", version=1, benchmark_rule=RULE,
                              parsed=parsed)
    return from_compiled(result, parsed, utterance_ref="utt-1")


class TestWhoAuthoredEachField:
    """The `execution_timing` defect, made structural. That value was inferred,
    offered back to the user to confirm, and thereby became indistinguishable
    from a choice they had made."""

    def test_the_users_words_are_user_authored(self):
        i = intent_for(CROSSING)
        assert i.fields["trigger_semantics"].author is Author.USER
        assert "crosses below" in i.fields["trigger_semantics"].source_span
        assert i.fields["amount"].author is Author.USER

    def test_a_compiler_default_is_not(self):
        i = intent_for(CROSSING)
        assert i.fields["execution_timing"].author is Author.DEFAULT
        assert i.fields["day_rule"].author is Author.DEFAULT

    def test_the_two_are_enumerable_apart(self):
        i = intent_for(CROSSING)
        assert "trigger_semantics" in i.user_authored
        assert "execution_timing" not in i.user_authored

    def test_a_stated_value_is_never_overwritten_by_a_default(self):
        """USER dominates. Letting an inference win here is the authority
        inversion the contract exists to make impossible."""
        i = intent_for(CROSSING)
        for name in i.user_authored:
            assert i.fields[name].author is Author.USER


class TestOpenIsNotAbsent:
    def test_a_raised_and_unanswered_question_is_open(self):
        i = intent_for(CROSSING)
        assert i.unresolved, "the compiler raised questions and none survived"
        for one in i.unresolved:
            assert one.reason is OpenReason.NOT_ASKED
            assert i.state_of(one.dimension) == "OPEN"

    def test_something_nobody_raised_is_absent(self):
        assert intent_for(CROSSING).state_of("periodic_rebalancing") == "ABSENT"

    def test_an_open_dimension_is_not_also_settled(self):
        i = intent_for(CROSSING)
        settled = set(i.fields)
        assert not settled & {u.dimension for u in i.unresolved}


class TestTheReaderIsRecorded:
    def test_every_intent_names_the_reader_that_produced_it(self):
        i = intent_for(CROSSING)
        assert i.produced_by == READER_VERSION
        assert all(f.produced_by == READER_VERSION for f in i.fields.values())

    def test_the_reader_version_is_not_the_author(self):
        """Two different questions. A user-authored value produced by this
        reader is still user-authored."""
        i = intent_for(CROSSING)
        f = i.fields["trigger_semantics"]
        assert f.author is Author.USER and f.produced_by == READER_VERSION

    def test_the_derivation_names_the_intent_by_hash(self):
        i = intent_for(CROSSING)
        d = derivation(i, compiled_by="quantify-engine@1")
        assert d["compiled_from"] == i.intent_hash
        assert d["compiled_by"] == "quantify-engine@1"
        assert d["manifest_hash"]


class TestTheSameSentenceProducesTheSameIntent:
    def test_identity_is_stable(self):
        assert intent_for(CROSSING).intent_hash == intent_for(CROSSING).intent_hash

    def test_and_a_different_sentence_does_not(self):
        assert intent_for(CROSSING).intent_hash != intent_for(SIXTY_FORTY).intent_hash


class TestMissionRefusesWhatItCannotExecute:
    """The intent is read, never edited. Each dimension the manifest cannot
    run produces a named refusal rather than a substitution."""

    def test_stated_weights_earn_a_named_refusal(self):
        refusals = executable_check(intent_for(SIXTY_FORTY))
        named = {r.dimension for r in refusals}
        assert "stated_weights" in named
        message = next(r.message for r in refusals
                       if r.dimension == "stated_weights")
        assert "60" in message and "40" in message

    def test_an_unsupported_allocation_method_earns_one(self):
        refusals = executable_check(intent_for(INVERSE_VOL))
        assert "allocation_method" in {r.dimension for r in refusals}

    def test_the_refusal_names_what_could_run_without_applying_it(self):
        refusal = next(r for r in executable_check(intent_for(INVERSE_VOL))
                       if r.dimension == "allocation_method")
        assert "equal_weight_at_purchase" in refusal.message
        assert refusal.stated_value != "equal_weight_at_purchase"

    def test_an_executable_plan_earns_none(self):
        """The discriminating half: a gate that refuses everything proves
        nothing."""
        assert executable_check(intent_for(CROSSING)) == ()

    def test_all_refusals_are_returned_not_just_the_first(self):
        text = ("I hold a 60/40 stock and bond split in a taxable brokerage "
                "account and rebalance quarter end, over the past 5 years.")
        named = {r.dimension for r in executable_check(intent_for(text))}
        assert {"stated_weights", "periodic_rebalancing"} <= named


class TestProseDeclarationsReachTheIntent:
    """Found by building this: `stated_weights` and unsupported weighting are
    read from the prose and never become a `Recognition`, so an intent without
    them cannot be refused by the manifest — Mission would never learn they
    were asked for. Coverage still blocked the figure by its own path, so the
    product was safe and the contract was incomplete."""

    def test_a_stated_split_is_in_the_intent_at_all(self):
        assert "stated_weights" in intent_for(SIXTY_FORTY).fields

    def test_it_is_attributed_to_the_user_not_to_a_default(self):
        assert intent_for(SIXTY_FORTY).fields["stated_weights"].author \
            is Author.USER

    def test_the_quoted_phrase_stops_at_the_clause(self):
        """These patterns detect rather than delimit, so a match runs past the
        thing it found. Quoting "rebalance quarter end, over the past 5 years"
        back at a user misdescribes their sentence in a message whose whole job
        is to describe it."""
        text = ("I hold a 60/40 stock and bond split in a taxable brokerage "
                "account and rebalance quarter end, over the past 5 years.")
        phrase = intent_for(text).fields["periodic_rebalancing"].value
        assert "over the past" not in phrase


class TestTodaysCompilerProducesDraftsAndSaysSo:
    """The seal is not a formality here — today's compiler genuinely cannot
    close meaning on most prompts, and the contract makes that visible instead
    of letting a half-understood plan look settled."""

    def test_an_intent_with_open_questions_refuses_to_seal(self):
        from runtime_contracts import NotSealable

        i = intent_for(CROSSING)
        assert i.unresolved, "this prompt should leave questions open"
        with pytest.raises(NotSealable) as raised:
            i.seal()
        # and it names them, so the caller knows what to ask
        for one in i.unresolved:
            assert one.dimension in str(raised.value)

    def test_it_starts_as_a_draft(self):
        from runtime_contracts import IntentState

        assert intent_for(CROSSING).state is IntentState.DRAFT
        assert not intent_for(CROSSING).is_verified

    def test_settling_the_questions_lets_it_seal(self):
        """The discriminating half: a seal nothing can satisfy would be a seal
        everyone routes around."""
        from dataclasses import replace

        i = intent_for(CROSSING)
        assert replace(i, unresolved=()).seal().is_verified

    def test_mission_is_not_asked_to_run_a_draft(self):
        """`is_executable_in_principle` is false until sealed, whatever the
        dimensions say. Closure is the first half of the claim."""
        assert not intent_for(CROSSING).is_executable_in_principle
