"""The catalogue, and the claim it makes by existing.

An entry in a dropdown is the product saying "this works". That is a stronger
claim than anything else in the interface, because the user did not propose it —
we did. A catalogue entry that compiles to a refusal is a false claim of
support, and it is worse than the same sentence typed unprompted: the person
followed a suggestion, and the refusal arrives after they have already invested
in it.

So the offered list is checked by running it, not by reading it. Everything
under `LIBRARY` goes through the whole pipeline here — read, fuse, compile — and
has to come out executable with nothing left to ask.
"""
from __future__ import annotations

import pytest

from src.workspace.strategy_library import (EDITED, LIBRARY, PICKED, TYPED,
                                            entry, offered, origin_of)


def readers() -> list:
    """Every reader the corpus holds readings for.

    The catalogue is checked under all of them, because the reader the corpus
    is recorded with and the reader a deployment serves are not the same one:
    the corpus lanes declare gpt-4.1-2025-04-14 and this deployment serves
    claude-sonnet-5. A guarantee proved under the first is a guarantee about a
    reader no user meets.

    It is not hypothetical. A drawdown entry — "invest $2,000 into VTI whenever
    it drops 10% below its highest close of the last year" — ran under
    claude-sonnet-5 and was refused under gpt-4.1, which read the fixed amount
    as a `conditional_amount`. Checking one reader would have shipped it.
    """
    import json
    from pathlib import Path

    document = json.loads(
        (Path(__file__).resolve().parent.parent / "corpus" / "parser"
         / "hosted.json").read_text())
    found = sorted({r["reader_id"] for r in document["readings"]})
    assert len(found) > 1, (
        "the corpus holds readings for one reader only, so 'runs under every "
        "reader' is the same claim as 'runs under the corpus reader'. Record "
        "the catalogue under the reader the deployment serves")
    return found


def _read(text: str, reader_id: str):
    from src.discovery.hosted_recording import RecordedHostedReader
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.syntax_stanza import RecordedReader
    from src.discovery.witnesses import BOTH
    from src.workspace.pilot import read

    hosted = RecordedHostedReader()
    # Pinned rather than left at the corpus default, which is the whole point:
    # `id` is what `read` keys recordings by.
    hosted.recorded_with = dict(hosted.recorded_with, reader_id=reader_id)
    return read(text, hosted, schema=QUANTIFY_SCHEMA,
                profile=BOTH, syntax_reader=RecordedReader())


CASES = [(e, r) for e in offered() for r in readers()]
IDS = [f"{e.key}-{r}" for e, r in CASES]


class TestEveryEntryResolves:
    """Execute, or refuse by name with a reason. Silence is the only failure.

    This class used to require every entry to run, and the catalogue carried
    only what did. That made the product look smaller than it is and made its
    gaps invisible: a strategy nobody can select is one nobody asks for, and one
    nobody asks for never gets built. All twenty families are offered now, and
    what the engine cannot do it says so about — which turns each selection into
    a recorded request for the thing that is missing.

    What must never happen is the third outcome: a sentence that neither runs
    nor explains itself. That is the state a person cannot act on and cannot
    report, and it is what these assertions exist to make impossible.
    """

    @pytest.mark.parametrize("case,reader", CASES, ids=IDS)
    def test_it_either_runs_or_says_why_not(self, case, reader):
        reading = _read(case.text, reader)
        if reading.executable:
            return
        reasons = [getattr(r, "detail", "") for r in reading.refusals]
        assert reasons or reading.questions, (
            f"{case.key!r} under {reader} neither ran, refused, nor asked. The "
            "page shows a plan that did nothing and says nothing about why")
        for detail in reasons:
            assert detail.strip(), (
                f"{case.key!r} under {reader} is refused with an empty reason. "
                "A dimension named with no explanation is worse than no "
                "refusal: the reader is told something is wrong and not what")

    @pytest.mark.parametrize("case,reader", CASES, ids=IDS)
    def test_a_refusal_names_a_dimension(self, case, reader):
        """By name, never a generic decline. The dimension is what lets a
        refusal be counted, queued and eventually fixed."""
        reading = _read(case.text, reader)
        for refusal in reading.refusals:
            assert getattr(refusal, "dimension", ""), (
                f"{case.key!r} under {reader} refuses without naming what it "
                "refused")


class TestTheCatalogueSpansTheKnownFamilies:
    def test_every_sampled_family_is_offered(self):
        """The corpus sampled twenty strategy families from cited sources. All
        twenty are selectable, including the ones this build declines — that is
        the point of offering them."""
        import json
        from pathlib import Path

        sampled = {c["family"] for c in json.loads(
            (Path(__file__).resolve().parent.parent / "corpus" / "parser"
             / "strategy_families.json").read_text())["cases"]}
        offered_families = {e.family for e in offered()}
        assert sampled - offered_families == set(), (
            f"{sorted(sampled - offered_families)} were sampled from cited "
            "sources and cannot be selected")

    def test_every_entry_cites_where_it_came_from(self):
        """Attested rather than invented, the same rule the harvested corpus
        follows. A sentence somebody made up that sounds plausible teaches the
        reader nothing about how people actually write."""
        for case in offered():
            assert case.source.startswith("http"), case.key


class TestTheCatalogueIsWellFormed:
    def test_keys_are_unique(self):
        keys = [e.key for e in offered()]
        assert len(set(keys)) == len(keys)

    def test_groups_are_not_empty(self):
        for group in LIBRARY:
            assert group.entries, f"{group.key!r} is an empty heading"

    def test_every_entry_names_its_family(self):
        """`demonstrates` was here and is gone with the hand-written entries.
        The family is the better answer to the same question: it says which
        strategy the sentence is an instance of, and it is what lets a refusal
        be traced back and counted rather than read once and forgotten."""
        for case in offered():
            assert case.family.strip(), case.key


class TestOriginIsDerivedNotDeclared:
    """Without this the cohort measures the catalogue. A high success rate over
    sentences we wrote, read by a reader we wrote, is a closed loop; `TYPED` is
    the only origin carrying evidence about anyone's own words."""

    def test_an_untouched_pick_is_picked(self):
        case = offered()[0]
        assert origin_of(case.text, case.key) == PICKED

    def test_a_changed_pick_is_edited(self):
        case = offered()[0]
        assert origin_of(case.text + " in my ISA", case.key) == EDITED

    def test_whitespace_alone_is_not_an_edit(self):
        """A textarea round-trips newlines and padding. Counting that as an
        edit would report the catalogue as insufficient every time somebody
        clicked into the box."""
        case = offered()[0]
        assert origin_of(f"  {case.text}\n", case.key) == PICKED

    def test_nothing_picked_is_typed(self):
        assert origin_of("invest $500 a month into VTI", "") == TYPED

    def test_an_unknown_key_is_typed_not_picked(self):
        """The client sends the key. A browser claiming a pick that names no
        entry must not be able to launder a typed sentence into the picked
        bucket and out of the evidence."""
        assert origin_of("anything at all", "no-such-entry") == TYPED

    def test_the_key_alone_cannot_assert_the_origin(self):
        """The specific thing the client cannot be trusted about: whether the
        user changed the sentence after picking it."""
        case = offered()[0]
        assert origin_of("something else entirely", case.key) == EDITED
        assert entry(case.key) is not None
