"""An explicit crossing verb outranks a broader persistent-condition pattern.

Two rules, materially different results. "Every time it crosses below" fires
once per drawdown; "every day it is below" fires on each of them. Over five
years that is not a rounding difference, and the compiler's own comment says
so.

The defect: the persistent pattern was

    \\bwhenever\\b[^.]*\\b(?:is |trades |closes )?(?:below|above)\\b

with the verb group optional, so it matched "whenever … below" whatever sat
between them.

    "whenever it crosses below"   -> persistent, nothing asked
    "when it crosses below"       -> crossing

One word decided a financial rule, and the word the user wrote to say which
rule they meant — *crosses* — was the one discarded. Found by a browser agent
reading the page back: the plan rendered as "buys on every day the condition
holds" for a sentence that says *crosses below*.

**Why precedence rather than a tighter regex.** Widening the crossing pattern
to cover "whenever … crosses" would fix these sentences and leave the shape
intact: the next overlapping matcher added to a flat first-match-wins table
reintroduces it. The rule is now a property of the resolver — an event verb
present means crossing, whatever else also matches — and the table no longer
holds `trigger_semantics` at all.

Note the original ordering was already correct. The crossing entry was listed
first and still lost, because it did not match. That is why this file tests
the *outcome* under a reordered resolver rather than testing list order.
"""
from __future__ import annotations

import pytest

from src.mission.compiler import (
    _CROSSING_LANGUAGE,
    _PERSISTENT_LANGUAGE,
    _RULES,
    parse,
    trigger_semantics,
)

CROSSING = "crossing_event"
PERSISTENT = "persistent_condition"

#: Exact natural language, not fragments. A pattern test on "crosses below"
#: alone would pass while the full sentence still resolved the other way,
#: because the competing matcher keys on a word elsewhere in it.
SENTENCES = (
    ("I buy $1,000 of SPY whenever it crosses below its 200-day moving "
     "average, over the past five years.", CROSSING),
    ("I buy $1,000 of SPY when it crosses below its 200-day moving average.",
     CROSSING),
    ("I buy $1,000 of SPY only on the day it crosses below its 200-day "
     "moving average.", CROSSING),
    ("Buy $500 of VOO each time SPY crosses above its 50-day average.",
     CROSSING),
    ("I buy $1,000 of SPY while it is below its 200-day moving average.",
     PERSISTENT),
    ("I buy $1,000 of SPY whenever it stays below its 200-day moving average.",
     PERSISTENT),
    ("I buy $1,000 of SPY every day it is below its 200-day moving average.",
     PERSISTENT),
    ("I buy $1,000 of SPY whenever it closes below its 200-day average.",
     PERSISTENT),
)

#: Neither reading may be assigned. The first is the pilot user's own wording,
#: and it must keep asking: "trades under" reads as either to a person.
AMBIGUOUS = (
    "I buy $1,000 of SPY whenever it trades under its 200-day moving average.",
    "I buy $1,000 of SPY whenever it crosses below and stays below its "
    "200-day moving average.",
    "I buy $1,000 of SPY each time it drops below its 200-day moving average.",
)


class TestThePrecedenceRule:
    @pytest.mark.parametrize("sentence,expected", SENTENCES)
    def test_the_sentence_resolves_as_written(self, sentence, expected):
        assert trigger_semantics(sentence) == expected, sentence

    @pytest.mark.parametrize("sentence", AMBIGUOUS)
    def test_an_ambiguous_sentence_is_not_assigned(self, sentence):
        """Silence is the correct output. A default here is a guess about
        money, and the question already exists to be asked."""
        assert trigger_semantics(sentence) is None, sentence

    @pytest.mark.parametrize("sentence", [
        one for one, _ in SENTENCES if "cross" in one.lower()
    ] + [one for one in AMBIGUOUS if "cross" in one.lower()])
    def test_a_crossing_verb_is_never_read_as_persistent(self, sentence):
        """The invariant, stated as a property over every sentence containing
        an event verb rather than as one example. `None` is permitted — an
        ambiguous sentence asks — but the state reading may never be assigned
        over the user's own word."""
        assert _CROSSING_LANGUAGE.search(sentence), (
            f"premise failed: no crossing language in {sentence!r}")
        assert trigger_semantics(sentence) != PERSISTENT, sentence

    def test_a_sentence_where_both_vocabularies_fire_asks(self):
        """Constructed so both matchers genuinely hit — verified here rather
        than assumed, because the first version of this test used a sentence
        where the persistent matcher never fired and proved nothing."""
        both = ("I buy $1,000 of SPY whenever it crosses below its 200-day "
                "average and stays below it.")
        assert _CROSSING_LANGUAGE.search(both)
        assert _PERSISTENT_LANGUAGE.search(both)
        assert trigger_semantics(both) is None


class TestTheResolverIsTheOnlyAuthority:
    def test_the_flat_table_no_longer_decides_it(self):
        """A second matcher for this field in the first-match-wins table is
        how the defect would return. Its absence is the fix's shape."""
        assert not [one for one in _RULES if one[0] == "trigger_semantics"]

    def test_the_recognition_reaches_the_parse(self):
        """The resolver deciding correctly is worth nothing if `parse` does
        not carry it — the same reachability shape as the rest of this branch.
        """
        found = [one for one in parse(SENTENCES[0][0]).recognitions
                 if one.field == "trigger_semantics"]
        assert found and found[0].value == CROSSING

    def test_the_recognition_quotes_the_users_words(self):
        """The span is shown back to the user as what was recognised. Quoting
        the persistent matcher's text beside a crossing decision would explain
        the plan by the evidence for the other reading."""
        found = [one for one in parse(SENTENCES[0][0]).recognitions
                 if one.field == "trigger_semantics"][0]
        assert "cross" in found.span.lower(), found.span


@pytest.fixture
def deployment(monkeypatch):
    from src.deploy.context import bind, resolve, unbind

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    try:
        yield
    finally:
        unbind()


def compiled(text):
    from src.workspace.draft import compile_draft

    return compile_draft(text, name="p", context="trigger test").scenario


class TestItReachesTheExecutedRule:
    """The resolver is not the claim. The claim is that the stored, executed
    trigger matches the sentence — the recognition has to survive compilation
    into a `SignalKind`, which is a different piece of code."""

    def test_the_control_compiles_to_a_crossing(self, deployment):
        from src.mission.signals import SignalKind

        scenario = compiled(SENTENCES[0][0])
        assert scenario.funding is not None, "no funding policy at all"
        assert scenario.funding.trigger.kind is \
            SignalKind.CROSSED_BELOW_MOVING_AVERAGE

    def test_a_persistent_sentence_compiles_to_a_state_rule(self, deployment):
        """The control for the control. A build that answered `crossing` for
        everything would pass the test above."""
        from src.mission.signals import SignalKind

        scenario = compiled(SENTENCES[6][0])
        assert scenario.funding is not None
        assert scenario.funding.trigger.kind is \
            SignalKind.BELOW_MOVING_AVERAGE

    def test_an_ambiguous_sentence_still_asks(self, deployment):
        scenario = compiled(AMBIGUOUS[0])
        assert "trigger_semantics" in [one.field for one
                                       in scenario.provenance.unresolved]

    def test_the_two_readings_produce_different_money(self, deployment):
        """The premise for calling this critical rather than cosmetic. If both
        rules paid the same, the defect would be a labelling problem."""
        import src.workspace.routes as routes

        access = routes._market_data("trigger money")
        crossing = routes._run(compiled(SENTENCES[0][0]), access,
                               stated_text=SENTENCES[0][0])
        persistent = routes._run(compiled(SENTENCES[6][0]), access,
                                 stated_text=SENTENCES[6][0])
        assert crossing["result"] is not None, crossing.get("unavailable")
        assert persistent["result"] is not None, persistent.get("unavailable")

        def spent(run):
            ledger = run.get("ledger")
            assert ledger is not None, "no ledger, so nothing to compare"
            return float(ledger.total_contributed), len(ledger.signals)

        crossing_money, crossing_signals = spent(crossing)
        persistent_money, persistent_signals = spent(persistent)
        assert crossing_signals and persistent_signals
        assert persistent_signals > crossing_signals, (
            f"a state rule fired {persistent_signals} times and a crossing "
            f"rule {crossing_signals}; equal counts would mean the two "
            f"readings cost the same and nothing here is critical")
        assert persistent_money > crossing_money, (
            f"${persistent_money:,.0f} against ${crossing_money:,.0f} — the "
            f"reading decides how much of the user's money is committed")


class TestTheRenderedExplanationMatchesTheStoredRule:
    """The defect was visible because the page said one thing and the sentence
    said another. Whatever the stored rule is, the prose must describe it."""

    def sentence_on_the_page(self, text):
        from src.workspace.confirmation import build as build_confirmation
        from src.workspace.draft import compile_draft

        compiled_plan = compile_draft(text, name="p", context="trigger test")
        view = build_confirmation(compiled_plan, text=text, priceable=())
        return " ".join(str(one) for one in getattr(view, "lines", []) or
                        [view]).lower()

    def test_a_crossing_plan_is_not_described_as_persistent(self, deployment):
        rendered = self.sentence_on_the_page(SENTENCES[0][0])
        assert "every day the condition holds" not in rendered, (
            "the page describes a crossing rule as a state rule; this is the "
            "sentence that exposed the defect")
