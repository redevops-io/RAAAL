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
                                            UNSUPPORTED, entry, offered,
                                            origin_of, unsupported)


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


class TestEverythingOfferedRuns:
    """The property the catalogue exists to keep. Nothing here is skipped for
    a missing recording: an entry with no recorded reading is an entry the
    product offers and nothing has ever run, which is the defect."""

    @pytest.mark.parametrize("case,reader", CASES, ids=IDS)
    def test_it_reads_without_a_question(self, case, reader):
        reading = _read(case.text, reader)
        assert not reading.questions, (
            f"{case.key!r} under {reader} is offered and comes back asking "
            f"for {sorted(reading.questions)}. A suggested sentence that does "
            "not settle sends the user into the clarification loop on a "
            "sentence they did not write")

    @pytest.mark.parametrize("case,reader", CASES, ids=IDS)
    def test_it_is_not_refused(self, case, reader):
        reading = _read(case.text, reader)
        refused = [getattr(r, "dimension", "") for r in reading.refusals]
        assert not refused, (
            f"{case.key!r} under {reader} is offered and the engine refuses "
            f"{refused}. "
            "Offering a strategy this build will not run is a claim of support "
            "the product makes on its own initiative")

    @pytest.mark.parametrize("case,reader", CASES, ids=IDS)
    def test_it_compiles_to_something_executable(self, case, reader):
        reading = _read(case.text, reader)
        assert reading.executable, (
            f"{case.key!r} under {reader} reads cleanly and does not compile "
            "to a runnable "
            "plan")
        assert reading.compiled is not None


def heading_for(name: str) -> str:
    return next(h for h, n in UNSUPPORTED.items() if n == name)


class TestTheRefusedListStaysTiedToTheEngine:
    def test_every_heading_names_a_real_dimension(self):
        """`unsupported()` raises rather than skipping, so this is the test
        that a heading pointing at a renamed dimension is caught here and not
        by a user reading a blank reason."""
        assert len(unsupported()) == len(UNSUPPORTED)

    def test_every_reason_is_the_engines_own_words(self):
        """Not paraphrased into the catalogue. The refusal a user reads has to
        be the one the engine would give, or the selector becomes a second
        account of the boundary that drifts from the first."""
        from src.mission.capability import MANIFEST

        for heading, reason in unsupported():
            assert reason, f"{heading!r} shows an empty reason"
            name = UNSUPPORTED[heading].partition(":")[0]
            dimension = MANIFEST[name]
            assert reason in (dimension.why,
                              *dimension.refuses.values()), heading

    def test_nothing_refused_is_also_offered(self):
        """The contradiction that would matter most. If a dimension is listed
        as unsupported and an offered sentence settles it, one of the two is
        lying to the user."""
        from src.mission.capability import MANIFEST

        for name in set(UNSUPPORTED.values()):
            dimension_name, _, value = name.partition(":")
            dimension = MANIFEST[dimension_name]
            if value:
                # A boundary inside an executed dimension. The claim is
                # narrower — this *value* is refused — and that is what gets
                # checked, or the heading could name any value it liked.
                assert value in dimension.refuses, (
                    f"{heading_for(name)!r} says {value!r} is refused and the "
                    f"manifest does not refuse it")
                continue
            assert dimension.support != "EXECUTED", (
                f"{name!r} is shown to users as not supported and the "
                "manifest says it executes; the catalogue is understating "
                "what the engine does")

    def test_the_headings_are_in_a_users_words(self):
        """Somebody looking for momentum searches for "momentum", not for
        `selection_rule`. Internal vocabulary in this list would make the
        entries unfindable by the people they exist for."""
        from src.discovery.schema import QUANTIFY_SCHEMA

        names = {d.name for d in QUANTIFY_SCHEMA.dimensions}
        for heading in UNSUPPORTED:
            words = set(heading.lower().replace("—", " ").split())
            assert not (words & names), (
                f"{heading!r} uses schema vocabulary; the internal name is "
                "already carried in the mapping's value")


class TestTheCatalogueIsWellFormed:
    def test_keys_are_unique(self):
        keys = [e.key for e in offered()]
        assert len(set(keys)) == len(keys)

    def test_groups_are_not_empty(self):
        for group in LIBRARY:
            assert group.entries, f"{group.key!r} is an empty heading"

    def test_every_entry_says_what_it_demonstrates(self):
        """The catalogue is small and will be edited by someone who was not
        here. An entry with no stated purpose is one nobody can tell the
        difference between keeping and deleting."""
        for case in offered():
            assert case.demonstrates.strip()


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
