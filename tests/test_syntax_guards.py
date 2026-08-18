"""Syntax proves an action is present. It never says what the action means.

Four live draws of five read `sell the loser and buy a similar fund to avoid a
wash sale`, reported `sell_action`, and Mission refused it by name. The fifth
read no sell at all and produced an executable plan — for a buy-and-hold engine,
from a sentence whose first word is "sell". The dimension existed and the
refusal existed; nothing downstream had anything to refuse.

The rule these implement:

    if deterministic syntax strongly proves a material dimension is explicitly
    present, Discovery may not seal an intent that omits it

and the line they must not cross: a guard proves presence and stops. It
proposes no value, so a sealed intent can never contain something this layer
invented. That is the difference between a structural witness and a second
domain compiler.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def parser():
    from src.discovery.syntax_stanza import RecordedReader

    return RecordedReader()


class TestAGuardProvesPresenceAndNothingElse:
    def test_it_proposes_no_value(self, parser):
        """The whole boundary in one assertion.

        A guard that proposed `sell_action="sell the loser"` would be reading
        meaning off a parse, and Discovery would have two semantic readers
        disagreeing about domain content instead of one reader and one
        structural check.
        """
        from src.discovery.guards import as_decisions

        parse = parser.parse("sell the loser and buy a similar fund to avoid "
                             "a wash sale")
        decisions = as_decisions(parse, [])
        assert decisions
        for decision in decisions:
            assert decision.value is None
            assert all(e.proposed_value is None for e in decision.syntax)

    def test_it_does_not_proceed(self, parser):
        """`DISAGREE`, so the dimension is a question rather than a silence.

        Reused rather than given a new outcome: the fusion contract already
        means "syntax has something to say and the model does not corroborate
        it", which is exactly this.
        """
        from src.discovery.guards import as_decisions

        parse = parser.parse("sell VTI and buy BND")
        for decision in as_decisions(parse, []):
            assert not decision.proceeds
            assert decision.material


class TestWhatTheGuardsFireOn:
    @pytest.mark.parametrize("text,expected", [
        ("sell the loser and buy a similar fund to avoid a wash sale",
         {"sell_action"}),
        ("sell VTI and buy BND", {"sell_action"}),
        ("withdraw 4% of the portfolio each year, adjusted for inflation",
         {"sell_action"}),
        ("rebalance back to 60/40 every year", {"periodic_rebalancing"}),
    ])
    def test_a_stated_action_is_proved_present(self, parser, text, expected):
        from src.discovery.guards import presence

        assert set(presence(parser.parse(text))) == expected

    @pytest.mark.parametrize("text", [
        "invest $500 monthly into VTI",
        "buy VOO when SPY falls below its 200-day moving average",
        "a 60/40 portfolio",
        "keep the REITs in the Roth",
    ])
    def test_and_a_sentence_without_one_is_left_alone(self, parser, text):
        """Over-firing is the failure that would make this unusable.

        A guard that triggered on ordinary accumulation would turn every plan
        into an unanswerable question, and the usual repair is to weaken the
        guard until it catches nothing.
        """
        from src.discovery.guards import presence

        assert not presence(parser.parse(text))

    def test_position_matters_not_just_the_word(self, parser):
        """"sell" as a noun modifier is not somebody selling.

        Checked because the cheap implementation is a substring search, and it
        would fire on every trigger sentence that mentions a sell signal.
        """
        from src.discovery.guards import GUARDS, PREDICATE_RELATIONS

        guard = {g.dimension: g for g in GUARDS}["sell_action"]
        assert "root" in PREDICATE_RELATIONS
        assert "amod" not in PREDICATE_RELATIONS
        assert "compound" not in PREDICATE_RELATIONS
        assert guard.lemmas


class TestAGuardOnlyCoversWhatVanishingWouldChange:
    def test_every_guarded_dimension_changes_the_plan_if_it_vanishes(self):
        """A guard is justified when a *lost* reading changes what runs.

        This asserted that every guarded dimension was REFUSED or
        NOT_MODELLED, on the reasoning that guarding an executable one would
        manufacture questions about something the engine would have run
        anyway. That reasoning held while `periodic_rebalancing` was refused
        and stopped holding the moment it was not: if the reader goes silent on
        "rebalance annually" the engine does not rebalance, it buys and holds —
        a different strategy, no question asked, no refusal shown. The guard is
        what catches that, and it became *more* necessary when the dimension
        started executing, not less.

        "Would have run it anyway" is only true of a dimension with a default.
        These have none: a lost reading is a lost instruction either way, and
        which way only changes whether the silence costs a refusal or a
        behaviour.
        """
        from src.discovery.guards import GUARDS
        from src.mission.capability import MANIFEST

        manifest = dict(MANIFEST)
        for guard in GUARDS:
            entry = manifest.get(guard.dimension)
            assert entry is not None, guard.dimension
            assert not entry.values or entry.support != "EXECUTED", (
                f"{guard.dimension} executes and has a closed value set, so a "
                "silent reading would fall to a default rather than vanish; "
                "guarding it manufactures a question instead of catching a "
                "reduction")

    def test_every_guarded_dimension_is_in_the_schema(self):
        from src.discovery.guards import GUARDS
        from src.discovery.schema import QUANTIFY_SCHEMA

        names = set(QUANTIFY_SCHEMA.names)
        for guard in GUARDS:
            assert guard.dimension in names

    def test_each_guard_says_why(self):
        """Read by a person, in a refusal. A guard with no reason produces a
        question nobody can act on."""
        from src.discovery.guards import GUARDS

        assert all(len(g.why) > 30 for g in GUARDS)


class TestAnOpenDimensionIsNotMissing:
    def test_a_guard_does_not_duplicate_a_question_already_asked(self, parser):
        """If fusion already has the dimension — settled or open — the guard
        stays quiet. Two entries for one dimension would show a person the same
        question twice and make `open_fields` disagree with itself.
        """
        from discovery_runtime.fusion import Fusion

        from src.discovery.claims import Decision
        from src.discovery.guards import missing

        parse = parser.parse("sell VTI and buy BND")
        already = [Decision(dimension="sell_action", outcome=Fusion.DISAGREE)]
        assert not missing(parse, already)

    def test_but_silence_counts(self, parser):
        from src.discovery.guards import missing

        parse = parser.parse("sell VTI and buy BND")
        assert [g.dimension for g in missing(parse, [])] == ["sell_action"]


class TestTheServingPathCannotSealAroundIt:
    """End to end, which is where the property actually has to hold."""

    @pytest.fixture
    def readers(self):
        from src.discovery.hosted_recording import RecordedHostedReader
        from src.discovery.syntax_stanza import RecordedReader

        return RecordedHostedReader(), RecordedReader()

    def test_a_dropped_sell_becomes_a_question_rather_than_a_plan(
            self, readers, monkeypatch):
        """The fifth draw, reconstructed deterministically.

        The recorded reader is patched to drop `sell_action` — which is exactly
        what the live model did on one draw of five — and the plan must not
        come out executable.
        """
        from dataclasses import replace

        from src.discovery.schema import QUANTIFY_SCHEMA
        from src.discovery.witnesses import BOTH
        from src.workspace.pilot import read

        model, syntax = readers
        text = "sell the loser and buy a similar fund to avoid a wash sale"
        original = model.read

        def without_the_sell(sentence, schema):
            reading = original(sentence, schema)
            return replace(reading, readings=tuple(
                r for r in reading.readings if r.dimension != "sell_action"))

        monkeypatch.setattr(model, "read", without_the_sell)

        reading = read(text, model, schema=QUANTIFY_SCHEMA, profile=BOTH,
                       syntax_reader=syntax)
        assert not reading.executable, (
            "a sentence beginning 'sell' compiled into an executable "
            "buy-and-hold plan because the reader dropped one dimension")
        assert "sell_action" in reading.questions

    def test_and_the_guard_is_what_prevents_it(self, readers, monkeypatch):
        """The mutation. Without the guard the same reading executes, so this
        test is measuring the guard rather than something else that happened
        to stop it."""
        from dataclasses import replace

        import src.workspace.pilot as pilot
        from src.discovery.schema import QUANTIFY_SCHEMA
        from src.discovery.witnesses import BOTH

        model, syntax = readers
        text = "sell the loser and buy a similar fund to avoid a wash sale"
        original = model.read

        def without_the_sell(sentence, schema):
            reading = original(sentence, schema)
            return replace(reading, readings=tuple(
                r for r in reading.readings if r.dimension != "sell_action"))

        monkeypatch.setattr(model, "read", without_the_sell)
        monkeypatch.setattr("src.discovery.guards.as_decisions",
                            lambda parse, decisions: ())

        reading = pilot.read(text, model, schema=QUANTIFY_SCHEMA, profile=BOTH,
                             syntax_reader=syntax)
        assert "sell_action" not in reading.questions, (
            "with the guard disabled the dropped dimension should vanish "
            "again; if it does not, the guard is not what closes this case")

    def test_an_ordinary_plan_is_untouched(self, readers):
        """The cost side. If guards made accumulation unsealable the property
        would be bought by breaking the product."""
        from src.discovery.schema import QUANTIFY_SCHEMA
        from src.discovery.witnesses import BOTH
        from src.workspace.pilot import read

        model, syntax = readers
        reading = read("invest $500 monthly into VTI", model,
                       schema=QUANTIFY_SCHEMA, profile=BOTH,
                       syntax_reader=syntax)
        assert reading.executable
        assert "sell_action" not in reading.questions
