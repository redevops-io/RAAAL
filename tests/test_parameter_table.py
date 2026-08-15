"""The parameter table: what it shows, and what it must not.

The page carried four lists — understood, needed, chosen, not mentioned — and
left a person to assemble the plan from them. One table replaces them, and the
properties that make it better than four lists are the ones asserted here.

The one that matters most is what is *absent*. The manifest declares nineteen
dimensions; a sentence about monthly contributions uses five. A table that
listed all nineteen would be asking somebody to disregard fourteen rows, which
is how a form teaches people to skim the one row that mattered.
"""
from __future__ import annotations

import pytest

from src.workspace.parameters import (CHOSEN, NEEDED, REFUSED, SETTLED, rows,
                                      unanswered)


class Settled:
    def __init__(self, field, value, provenance="MODEL_ONLY_ACCEPTED",
                 witnesses=("model",)):
        self.field, self.value = field, value
        self.provenance, self.witnesses = provenance, witnesses


class Refusal:
    def __init__(self, dimension, detail, stated_value=""):
        self.dimension, self.detail = dimension, detail
        self.stated_value = stated_value


class Reading:
    def __init__(self, settled=(), questions=(), refusals=(),
                 applied_defaults=()):
        self.settled = settled
        self.questions = questions
        self.refusals = refusals
        self.applied_defaults = applied_defaults


def by_name(reading):
    return {p.name: p for p in rows(reading)}


class TestOnlyWhatThisStrategyNeeds:
    def test_it_shows_no_dimension_the_sentence_never_touched(self):
        """The whole point of one table over four lists."""
        found = by_name(Reading(
            settled=(Settled("amount", "200"), Settled("cadence", "monthly")),
            questions=("assets",)))

        assert set(found) == {"amount", "cadence", "assets"}
        for absent in ("withdrawal_ordering", "annuitisation", "leverage",
                       "tax_loss_harvesting"):
            assert absent not in found, (
                f"{absent} appeared in a table for a contribution schedule")

    def test_a_picked_strategy_leaves_fewer_blanks_than_a_typed_one(self):
        """The catalogue sentences were written to state their parameters, so
        picking one should be less work than typing. If that stops being true
        the catalogue has stopped earning its place."""
        picked = Reading(settled=(Settled("amount", "500"),
                                  Settled("assets", "VTI"),
                                  Settled("cadence", "monthly"),
                                  Settled("day_rule", "calendar_day:15")))
        typed = Reading(settled=(Settled("amount", "500"),),
                        questions=("assets", "cadence", "day_rule"))

        assert len(unanswered(picked)) == 0
        assert len(unanswered(typed)) == 3
        assert len(unanswered(picked)) < len(unanswered(typed))


class TestABlankTellsYouWhatToType:
    def test_a_needed_parameter_carries_examples(self):
        needed = by_name(Reading(questions=("day_rule",)))["day_rule"]
        assert needed.state == NEEDED
        assert needed.examples, "a blank with no example is a prompt to guess"
        assert "the 15th" in needed.examples

    def test_the_examples_are_things_the_engine_runs(self):
        """A blank whose example is refused would be an invitation to a
        refusal — the page suggesting something and then declining it.

        Asserted on `cadence` because it is the dimension where the engine's
        value is also the word a person writes. `day_rule` is not: nobody
        types `first_session_of_period`, they type "the first trading day",
        and the reader translates. So this checks the case where the claim is
        checkable rather than pretending it generalises.
        """
        from src.mission.capability import dimension

        needed = by_name(Reading(questions=("cadence",)))["cadence"]
        declared = dimension("cadence")
        assert needed.examples
        assert declared is not None and declared.values
        # Every example names a cadence the manifest executes.
        for example in needed.examples:
            assert any(value in example for value in declared.values), (
                f"{example!r} is offered beside a blank and names no cadence "
                f"this build runs: {declared.values}")

    def test_no_blank_is_ever_offered_without_an_example(self):
        """The property, over every dimension rather than the one I checked.

        `assets` rendered as a blank with a paragraph explaining what it means
        and nothing showing what to type — which is the state this whole table
        exists to remove. Six dimensions were in that position; the schema
        describes them and gives no example.
        """
        from src.discovery.schema import QUANTIFY_SCHEMA

        every = Reading(questions=tuple(d.name
                                        for d in QUANTIFY_SCHEMA.dimensions))
        without = [p.name for p in rows(every) if not p.examples]
        assert not without, (
            f"{without} would render as a blank with nothing showing what to "
            "type, which is a prompt to guess")

    def test_it_says_what_the_dimension_is(self):
        needed = by_name(Reading(questions=("day_rule",)))["day_rule"]
        assert needed.describes, (
            "a parameter named and not explained is a field somebody guesses")


class TestEveryStateIsDistinguishable:
    def test_a_refusal_carries_its_reason_rather_than_a_blank(self):
        found = by_name(Reading(refusals=(
            Refusal("evaluation_period",
                    "this build evaluates over the whole price history"),)))
        row = found["evaluation_period"]
        assert row.state == REFUSED
        assert "whole price history" in row.detail

    def test_a_default_says_the_engine_chose_it(self):
        row = by_name(Reading(applied_defaults=("day_rule",)))["day_rule"]
        assert row.state == CHOSEN
        assert "default" in row.provenance

    def test_a_settled_value_keeps_its_provenance_unsummarised(self):
        """MODEL_ONLY_ACCEPTED is not AGREE. A table printing "agreed" while
        one reader ran would claim corroboration it never had."""
        row = by_name(Reading(
            settled=(Settled("amount", "200"),)))["amount"]
        assert row.state == SETTLED
        assert "MODEL_ONLY_ACCEPTED" in row.provenance
        assert "model" in row.provenance

    def test_needed_rows_come_first(self):
        """The only rows that are work."""
        ordered = rows(Reading(
            settled=(Settled("amount", "200"),),
            questions=("assets",),
            applied_defaults=("day_rule",)))
        assert ordered[0].state == NEEDED

    def test_a_dimension_appears_once(self):
        """A parameter both settled and defaulted would otherwise be two rows
        saying different things about the same field."""
        ordered = rows(Reading(settled=(Settled("day_rule", "x"),),
                               applied_defaults=("day_rule",),
                               questions=("day_rule",)))
        assert [p.name for p in ordered] == ["day_rule"]


class TestThePageRendersIt:
    def test_the_answer_form_posts_where_it_renders_in_place(self):
        """Answering used to post to `/pilot/save`, which saved and redirected
        to a plan page — a second screen for a question asked on the first,
        and where a 500 was found. `/pilot/answer` re-renders the same page
        with the table filled in."""
        from pathlib import Path

        template = (Path(__file__).resolve().parent.parent / "src" /
                    "workspace" / "templates" / "pilot.html").read_text()
        assert 'action="/pilot/answer"' in template
        assert 'name="answer_' in template

    def test_the_picker_is_lifted_off_the_prompt(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "src" / "workspace"
        template = (root / "templates" / "pilot.html").read_text()
        styles = (root / "templates" / "base.html").read_text()
        assert 'class="picker"' in template
        assert ".picker {" in styles, (
            "the selector has no spacing of its own, so it reads as part of "
            "the box it fills")
