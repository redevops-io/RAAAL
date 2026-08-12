"""A clarification conversation must end.

    A successful clarification round must strictly reduce what is unresolved,
    change what would execute, or terminate in an explicit refusal.

Found in the pilot, on the first realistic sentence anybody typed:

    i buy 1000 usd of SP500 etf every time SPY index trades under it's 200DMA
    - only ones on the next business day. what would be my total return and
    the final cash amount over the past 5 years

Discovery read it well — the holding, the observed asset, the trigger, the
execution timing and the five-year window were all settled. One dimension was
not: `amount`, because the numeric reader could read `$1,000` and `1000` and
not `1000 usd`. So the page asked how much, the person answered "1000 usd",
and the page asked how much — the same question, the same answer, forever.

That is worse than a refusal and worse than a wrong figure, because it never
resolves into anything a person can act on. Nobody reaches a third round.

This file is the end-to-end journey rather than a parser fixture, because
every part of it was individually fine: the reading was right, the refusal was
right, the re-ask was right. Only the *sequence* was broken, and no unit test
sees a sequence.
"""
from __future__ import annotations

import pytest

PROMPT = ("i buy 1000 usd of SP500 etf every time SPY index trades under it's "
          "200DMA - only ones on the next business day. what would be my total "
          "return and the final cash amount over the past 5 years")

#: What a person would actually type, in their own words rather than in the
#: schema's. An answer set written in canonical form would prove the runtime
#: converges on input it never receives.
ANSWERS = {"amount": "1000 usd", "assets": "SPY", "starting_capital": "0"}

MAX_ROUNDS = 4


def _reading():
    from src.discovery.hosted_recording import RecordedHostedReader
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.syntax_stanza import RecordedReader
    from src.discovery.witnesses import BOTH
    from src.workspace.pilot import read

    try:
        return read(PROMPT, RecordedHostedReader(), schema=QUANTIFY_SCHEMA,
                    profile=BOTH, syntax_reader=RecordedReader())
    except KeyError:
        pytest.skip("the pilot prompt has no recorded reading; run "
                    "corpus/parser/record_hosted.py")


class TestTheConversationTerminates:
    def test_every_round_changes_the_state(self):
        """The invariant, asserted round by round rather than at the end. A
        journey that converged after wandering through a repeated state would
        pass an end-state check and still be the thing users abandon."""
        from src.workspace.pilot import answer

        reading = _reading()
        seen = {reading.clarification_state()}

        for _round in range(MAX_ROUNDS):
            asking = sorted(reading.open_fields)
            if not asking:
                return
            reply = {name: ANSWERS.get(name, "0") for name in asking}
            reading = answer(reading, reply)

            state = reading.clarification_state()
            assert state not in seen, (
                f"round {_round + 1} returned to a state already seen. The "
                f"person answered {sorted(reply)} and the conversation did "
                "not move: same questions, same refusals, same plan")
            seen.add(state)

        assert not reading.open_fields, (
            f"still asking {sorted(reading.open_fields)} after {MAX_ROUNDS} "
            "rounds; a clarification that does not converge is one nobody "
            "finishes")

    def test_an_answer_the_runtime_cannot_use_is_named_as_rejected(self):
        """The half that makes a stalled round actionable.

        When an answer settles nothing the page must say so. "I could not read
        '1000 usd' as an amount" is a different state from "how much are you
        contributing", and only one of them can be acted on.
        """
        from src.workspace.pilot import answer

        reading = answer(_reading(), {"amount": "as much as I can afford"})
        assert "amount" in reading.rejected_answers, (
            "an unusable answer was accepted silently, so the next render is "
            "the same question with no explanation")
        assert reading.rejected_answers["amount"]

    def test_a_usable_answer_is_not_reported_as_rejected(self):
        """The discriminating half. A check that called every answer rejected
        would pass the test above and describe a runtime that never accepts
        anything."""
        from src.workspace.pilot import answer

        reading = answer(_reading(), {"amount": "1000 usd"})
        assert "amount" not in reading.rejected_answers


class TestTheSentenceIsUnderstoodBeforeItIsQuestioned:
    """The questions asked have to be the ones the sentence leaves open. A
    conversation that converges by asking about things the person already said
    terminates and still wastes their time."""

    def test_what_the_sentence_states_is_not_asked_about(self):
        reading = _reading()
        settled = {f.field for f in reading.settled if f.value is not None}
        for stated in ("observed_assets", "trigger_semantics",
                       "execution_timing", "evaluation_period"):
            assert stated in settled, (
                f"{stated} is in the sentence and was not read from it")
            assert stated not in reading.open_fields

    def test_the_amount_is_read_from_the_sentence(self):
        """`1000 usd` is stated. It was settled and then refused as unreadable,
        which is how a stated figure became a question."""
        from src.mission.from_intent import _decimal

        reading = _reading()
        amount = next((f.value for f in reading.settled
                       if f.field == "amount"), None)
        assert amount is not None, "the amount is stated and was not read"
        assert _decimal(amount) == 1000, (
            f"{amount!r} was read from the sentence and cannot be turned into "
            "a number, so it becomes a question whose answer is itself")
