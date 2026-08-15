"""Every catalogue strategy, checked for the inconsistencies a person would hit.

`test_strategy_library` asks whether each entry *resolves* — runs, or refuses by
name. That is a property of the engine. This asks whether the resulting **page**
is coherent, which is a different question and the one users keep answering for
us:

  * a button offering to fill in blanks above a table with no blanks, which
    loops when pressed. Found by somebody typing a sentence, after the table
    had been reviewed and shipped.
  * a blank with a paragraph explaining the dimension and nothing showing what
    to type.
  * a plan that neither runs, nor asks, nor refuses.

None of these are visible to a test that checks a reading executes. They are
visible here, over the whole catalogue at once, for free — the readings are
recorded, so this costs no provider call and runs in the ordinary suite.

The browser sweep in `ui-agent/` asks the same questions of the deployed page,
where a template that is right can still be wired wrong. This is the half that
runs on every commit.
"""
from __future__ import annotations

import pytest

from src.workspace.parameters import CHOSEN, NEEDED, REFUSED, SETTLED
from src.workspace.parameters import rows as parameter_rows
from src.workspace.parameters import unanswered
from src.workspace.strategy_library import LIBRARY


def readers() -> list:
    """Every reader the corpus holds readings for."""
    import json
    from pathlib import Path

    document = json.loads(
        (Path(__file__).resolve().parent.parent / "corpus" / "parser"
         / "hosted.json").read_text())
    return sorted({r["reader_id"] for r in document["readings"]})


def reading_for(text: str, reader_id: str):
    from src.discovery.hosted_recording import RecordedHostedReader
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.syntax_stanza import RecordedReader
    from src.discovery.witnesses import BOTH
    from src.workspace.pilot import read

    hosted = RecordedHostedReader()
    hosted.recorded_with = dict(hosted.recorded_with, reader_id=reader_id)
    return read(text, hosted, schema=QUANTIFY_SCHEMA, profile=BOTH,
                syntax_reader=RecordedReader())


ENTRIES = [(group.key, entry) for group in LIBRARY for entry in group.entries]
CASES = [(group, entry, reader)
         for group, entry in ENTRIES for reader in readers()]
IDS = [f"{e.key}-{r}" for _g, e, r in CASES]


@pytest.mark.parametrize("group,entry,reader", CASES, ids=IDS)
class TestEveryOfferedStrategyProducesACoherentPage:
    """One offered strategy, under one reader, as a page rather than a result."""

    def test_the_table_and_the_button_agree(self, group, entry, reader):
        """The defect a user found by typing.

        The button is gated on the rows that need answering. If a question
        exists and the table does not show it, the page asks somebody to fill
        in something invisible and loops when they press it.
        """
        reading = reading_for(entry.text, reader)
        rows = parameter_rows(reading)
        needed = [row for row in rows if row.state == NEEDED]

        assert bool(unanswered(reading)) == bool(needed), (
            f"{entry.key}: the button and the table disagree about whether "
            "anything needs answering")

        for name in getattr(reading, "questions", ()) or ():
            assert any(row.name == name for row in needed), (
                f"{entry.key}: {name!r} is a question the table does not show, "
                "so it cannot be answered from the page")

    def test_no_parameter_appears_twice(self, group, entry, reader):
        rows = parameter_rows(reading_for(entry.text, reader))
        names = [row.name for row in rows]
        assert len(names) == len(set(names)), (
            f"{entry.key}: {[n for n in names if names.count(n) > 1]} appears "
            "more than once, saying different things about one field")

    def test_every_blank_shows_what_to_type(self, group, entry, reader):
        for row in parameter_rows(reading_for(entry.text, reader)):
            if row.state == NEEDED:
                assert row.examples, (
                    f"{entry.key}: {row.name!r} is blank with no example, "
                    "which is a prompt to guess")
                assert row.describes, (
                    f"{entry.key}: {row.name!r} is blank with no explanation")

    def test_every_refusal_says_why(self, group, entry, reader):
        for row in parameter_rows(reading_for(entry.text, reader)):
            if row.state == REFUSED:
                assert row.detail.strip(), (
                    f"{entry.key}: {row.name!r} is refused with no reason, "
                    "which tells somebody something is wrong and not what")

    def test_the_plan_is_never_silent(self, group, entry, reader):
        """Runs, asks, or refuses. The fourth outcome is the one nobody can
        act on and nobody can report."""
        reading = reading_for(entry.text, reader)
        if reading.executable:
            return
        rows = parameter_rows(reading)
        assert any(row.state in (NEEDED, REFUSED) for row in rows), (
            f"{entry.key}: did not run, asked nothing and refused nothing")

    def test_a_settled_row_says_where_it_came_from(self, group, entry, reader):
        for row in parameter_rows(reading_for(entry.text, reader)):
            if row.state in (SETTLED, CHOSEN):
                assert row.provenance.strip(), (
                    f"{entry.key}: {row.name!r} carries a value and no "
                    "provenance, so a reader cannot tell what settled it")


class TestTheSweepIsActuallySweeping:
    def test_it_covers_every_offered_strategy(self):
        assert len({entry.key for _g, entry in ENTRIES}) == sum(
            len(group.entries) for group in LIBRARY)

    def test_it_covers_more_than_one_reader(self):
        """A property shown under the corpus reader alone is a property about
        a reader no user meets."""
        assert len(readers()) > 1
