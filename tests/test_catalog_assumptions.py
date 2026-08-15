"""What the catalogue supplies, and the line it must not cross.

Picking a strategy is structured evidence: the product knows which family it
offered. Supplying a stand-in portfolio for a withdrawal rule that names none is
the difference between "answer this before your question can be answered" and a
plan somebody can look at and adjust.

The danger is the same act. A value the product supplies and then records as the
user's is an authority inversion — `Author.USER` dominates every other author and
is never overwritten by a re-read, so the guess would become their word
permanently and invisibly. Most of this file is about that line rather than
about the feature.
"""
from __future__ import annotations

import pytest
from runtime_contracts import Author

from src.discovery.hosted_recording import RecordedHostedReader
from src.discovery.schema import QUANTIFY_SCHEMA
from src.discovery.syntax_stanza import RecordedReader
from src.discovery.witnesses import BOTH
from src.workspace.catalog_assumptions import (ASSUMPTIONS, CATALOG_ASSUMED,
                                               CATALOG_CONFIRMED, applicable,
                                               assume, assumed_in, confirm,
                                               describe, for_group, group_of)
from src.workspace.parameters import ASSUMED, NEEDED, REFUSED, SETTLED
from src.workspace.parameters import editable
from src.workspace.parameters import rows as parameter_rows
from src.workspace.parameters import unanswered
from src.workspace.pilot import answer, read
from src.workspace.strategy_library import LIBRARY

READER = "gpt-5.4-2026-03-05@1"
ENTRIES = [(group, entry) for group in LIBRARY for entry in group.entries]


def reading_for(text: str):
    hosted = RecordedHostedReader()
    hosted.recorded_with = dict(hosted.recorded_with, reader_id=READER)
    return read(text, hosted, schema=QUANTIFY_SCHEMA, profile=BOTH,
                syntax_reader=RecordedReader())


def shape(reading):
    """What the engine did, independent of who supplied the inputs."""
    rows = parameter_rows(reading)
    return (reading.executable,
            tuple(sorted(r.name for r in rows if r.state == REFUSED)),
            tuple(sorted(r.name for r in rows if r.state == NEEDED)))


class TestAnAssumptionNeverEarnsARefusal:
    """The rule that shortened this list, and the reason it is enforced here.

    `stated_weights = 60% VTI and 40% BND` is the most natural-looking
    assumption anybody would write, and it was refused by name in every family
    that used it — this build allocates equally at purchase.
    `periodic_rebalancing = once a year` did the same. Both would have replaced
    a question somebody could answer with a refusal they did not ask for, which
    is strictly worse than asking.
    """

    @pytest.mark.parametrize("group,entry", ENTRIES,
                             ids=[e.key for _g, e in ENTRIES])
    def test_it_does_not_refuse_the_dimension_it_supplied(self, group, entry):
        before = reading_for(entry.text)
        refused_before = {r.name for r in parameter_rows(before)
                          if r.state == REFUSED}
        supplied = {one.dimension for one in applicable(before, entry.key)}
        if not supplied:
            pytest.skip("nothing applies to this entry")

        after = assume(before, entry.key)
        newly = {r.name for r in parameter_rows(after)
                 if r.state == REFUSED} - refused_before
        assert not (newly & supplied), (
            f"{entry.key}: assuming {sorted(newly & supplied)} earned a "
            "refusal naming the same dimension, so the catalogue answered a "
            "question with something the engine will not run")

    def test_every_declared_assumption_is_reachable(self):
        """A family whose key does not exist supplies nothing, silently.

        The dictionary is keyed by group and nothing checks the keys are real,
        so a renamed group would turn its assumptions off with no failure
        anywhere — the page would simply go back to asking.
        """
        known = {group.key for group in LIBRARY}
        assert set(ASSUMPTIONS) <= known, (
            f"{sorted(set(ASSUMPTIONS) - known)} name no catalogue group, so "
            "those assumptions can never apply")


class TestTheEngineDoesNotNoticeWhoSuppliedTheValue:
    """The decisive check, and the one that says the refusals are not ours.

    Assuming turned 37 questions into 15, and refusals from 3 to 23. Read alone
    that looks like the catalogue breaking strategies. It is not: supplying a
    portfolio lets the engine reach the *next* question and answer it honestly
    — "this build only buys; withdrawing is not modelled" — where before it
    stopped at "what do you hold?" and refused afterwards.

    The way to tell those apart is to have a person type the same values by
    hand. If the outcome is identical, the assumption changed who supplied the
    value and nothing else.
    """

    @pytest.mark.parametrize("group,entry", ENTRIES,
                             ids=[e.key for _g, e in ENTRIES])
    def test_assumed_and_typed_reach_the_same_outcome(self, group, entry):
        before = reading_for(entry.text)
        supply = {one.dimension: one.value
                  for one in applicable(before, entry.key)}
        if not supply:
            pytest.skip("nothing applies to this entry")
        assert shape(assume(before, entry.key)) == shape(answer(before, supply)), (
            f"{entry.key}: the catalogue supplying these values reaches a "
            "different outcome than a person typing them, so the assumption "
            "is doing something other than answering")


class TestAnAssumptionIsNeverTheUsersWord:
    """The authority inversion, which is the whole risk of this feature."""

    def entry_that_gets_one(self):
        for _group, entry in ENTRIES:
            before = reading_for(entry.text)
            if applicable(before, entry.key):
                return entry, before
        raise AssertionError("no catalogue entry receives an assumption")

    def test_the_value_is_authored_default_not_user(self):
        entry, before = self.entry_that_gets_one()
        after = assume(before, entry.key)
        supplied = {one.dimension for one in applicable(before, entry.key)}
        for name in supplied:
            field = after.intent.fields[name]
            assert field.author is Author.DEFAULT, (
                f"{name} was supplied by the catalogue and recorded as "
                f"{field.author.value}")
            assert not field.author.dominates, (
                f"{name} was recorded with an author that no re-read may "
                "overwrite, so our guess would outlive every later reading")

    def test_it_is_marked_in_the_settled_record(self):
        entry, before = self.entry_that_gets_one()
        after = assume(before, entry.key)
        assumed = {s.field for s in after.settled
                   if s.provenance == CATALOG_ASSUMED}
        assert assumed == {one.dimension
                           for one in applicable(before, entry.key)}
        for one in after.settled:
            if one.provenance == CATALOG_ASSUMED:
                assert "assumed by the catalogue, not stated" in one.detail
                assert one.witnesses == ["catalogue"]

    def test_typed_prose_receives_nothing(self):
        """Only a picked strategy. Inferring a family from free text would put
        a second classifier in charge of what to assume on somebody's behalf."""
        typed = reading_for("I withdraw $20,000 from the portfolio each year.")
        assert applicable(typed, "") == ()
        assert assume(typed, "") is typed

    def test_a_stated_value_is_never_replaced(self):
        """Assumptions filter on what the reading asked, not on the family.

        A family-wide assumption applied to a sentence that states the same
        dimension would overwrite what somebody said with what we guessed.
        """
        for group, entry in ENTRIES:
            before = reading_for(entry.text)
            asked = set(getattr(before, "questions", ()) or ())
            for one in applicable(before, entry.key):
                assert one.dimension in asked, (
                    f"{entry.key}: {one.dimension} was assumed and the reading "
                    "never asked for it")


class TestConfirmingChangesAuthorityAndNotHistory:
    def test_a_confirmed_value_becomes_the_users(self):
        """`assets`, named rather than taken as the first assumed dimension.

        It used to confirm `assumed_in(...)[0]`, which sorts — so once `amount`
        joined the assumed set it was confirming a contribution of "SPY". That
        passed while any string settled any field, and stopped the moment
        answers were canonicalised: a value that is not a number no longer
        settles an amount. The test was wrong and the change reported it.
        """
        entry, before = self.__class__.pick()
        assumed = assume(before, entry.key)
        assert "assets" in assumed_in(assumed)
        confirmed = confirm(assumed, {"assets": "SPY"})

        assert confirmed.intent.fields["assets"].author is Author.USER
        assert confirmed.intent.fields["assets"].value == "SPY"

    def test_the_record_still_says_it_began_as_a_guess(self):
        """Authoritative without rewriting provenance.

        A confirmed assumption that erased its own history would leave no
        evidence the product ever guessed — and "the user agreed" would become
        indistinguishable from "we stopped asking", which is the distinction
        the settled record exists to hold.
        """
        entry, before = self.__class__.pick()
        confirmed = confirm(assume(before, entry.key), {"assets": "SPY"})
        provenances = [s.provenance for s in confirmed.settled
                       if s.field == "assets"]
        assert CATALOG_ASSUMED in provenances, (
            "confirming erased the record that the value was assumed")
        assert CATALOG_CONFIRMED in provenances
        assert provenances.index(CATALOG_ASSUMED) < \
            provenances.index(CATALOG_CONFIRMED), "history is out of order"

    def test_a_confirmed_value_stops_being_reported_as_assumed(self):
        entry, before = self.__class__.pick()
        assumed = assume(before, entry.key)
        assert "assets" in assumed_in(assumed)
        assert "assets" not in assumed_in(confirm(assumed, {"assets": "SPY"}))

    @staticmethod
    def pick():
        for _group, entry in ENTRIES:
            before = reading_for(entry.text)
            if any(one.dimension == "assets"
                   for one in applicable(before, entry.key)):
                return entry, before
        raise AssertionError("no entry receives an assumed holding")


class TestThePageSaysWhatIsOurs:
    def test_an_assumed_row_is_shown_and_editable(self):
        entry, before = TestConfirmingChangesAuthorityAndNotHistory.pick()
        after = assume(before, entry.key)
        rows = {r.name: r for r in parameter_rows(after)}

        assert rows["assets"].state == ASSUMED, (
            "an assumed value rendered as SETTLED would read as something the "
            "person said")
        assert rows["assets"].value, "an assumed row with no value shows nothing"
        assert rows["assets"] in editable(after), (
            "an assumed value nobody can change is a decision taken for them")

    def test_it_does_not_block_the_run(self):
        entry, before = TestConfirmingChangesAuthorityAndNotHistory.pick()
        after = assume(before, entry.key)
        assert not [r for r in unanswered(after) if r.name == "assets"], (
            "an assumed row counted as unanswered puts the page back to "
            "offering to fill in something already filled in")

    def test_the_result_says_what_it_rests_on(self):
        entry, before = TestConfirmingChangesAuthorityAndNotHistory.pick()
        said = describe(assume(before, entry.key))
        assert said and "assumed" in said and "assets" in said, (
            "a figure resting on values nobody chose must say so where the "
            "figure is; the table is above it and gets scrolled past")
        assert describe(before) is None, (
            "a reading with no assumptions must say nothing rather than "
            "reassure")

    def test_the_input_is_empty_so_a_click_cannot_confirm(self):
        """The authority inversion, at the last place it can happen.

        If the input carried the assumed value, pressing "run it" would post it
        back and `answer` would record our guess as USER — the strongest author
        there is — for anybody who did not read the row. Empty means silence
        stays silence, and only a typed value speaks.
        """
        from pathlib import Path

        template = (Path(__file__).resolve().parent.parent / "src" /
                    "workspace" / "templates" / "pilot.html").read_text()
        block = template.split('row.state == "ASSUMED"')[1].split("{% else %}")[0]
        assert 'value=""' in block, (
            "the assumed row's input is pre-filled, so pressing the button "
            "records the catalogue's guess as the user's own word")
        assert "row.value" in block, "the assumed value is not shown at all"


class TestTheGainIsReal:
    """Without this the feature could be all mechanism and no effect."""

    def test_it_removes_most_of_the_asking(self):
        asked_before = asked_after = 0
        for group, entry in ENTRIES:
            before = reading_for(entry.text)
            asked_before += len(unanswered(before))
            asked_after += len(unanswered(assume(before, entry.key)))
        assert asked_after < asked_before / 2, (
            f"{asked_before} questions before, {asked_after} after — the "
            "catalogue is supplying almost nothing")

    def test_no_offered_strategy_goes_silent(self):
        """The one outcome nobody can act on, checked after assuming.

        `test_catalogue_sweep` checks this before. Supplying values moves
        strategies between running, asking and refusing, and a strategy that
        landed in none of the three would be a page that does nothing.
        """
        for group, entry in ENTRIES:
            after = assume(reading_for(entry.text), entry.key)
            rows = parameter_rows(after)
            assert after.executable or any(
                r.state in (NEEDED, REFUSED) for r in rows), (
                f"{entry.key}: after assuming, it neither ran, asked, nor "
                "refused")
