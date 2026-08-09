"""The recordings must still be what the parser produces.

`RecordedReader` makes the corpus fast, and buys that speed with a risk: a
fixture that has drifted from the parser keeps every test above it green while
measuring a parse nobody would get today. That is the same shape as a test whose
subject was deleted, which this project has already shipped once.

So this file re-runs the live parser and diffs. It needs the model, which is
half a gigabyte, so it does not run on every push — the `parser` CI job runs it,
and `tests/test_parser_corpus.py` runs everywhere on the recordings.

**It skips, and that is the one place skipping is right** — but the skip is
loud and the corpus test asserts a count that this file cannot influence, so a
permanently-skipped check cannot be mistaken for a passing one. The failure mode
being guarded is a fixture rotting silently, and a machine without the model has
no fixtures to rot: it is reading the same recording either way.
"""
from __future__ import annotations

import pytest

from corpus.parser.loader import load
from src.discovery.syntax_stanza import RecordedReader, StanzaReader


def live_parser_available(language: str = "en") -> bool:
    try:
        import stanza  # noqa: F401
    except ImportError:
        return False
    try:
        StanzaReader(language)._load()
    except Exception:                                     # noqa: BLE001
        return False
    return True


needs_model = pytest.mark.skipif(
    not live_parser_available(),
    reason="no Stanza model; the `parser` CI job runs this. Fixtures are "
           "read-only here, so there is nothing this machine can rot")


@needs_model
class TestTheRecordingsMatchTheParser:
    def test_every_recorded_english_parse_is_reproducible(self):
        """The whole point. If a parser upgrade moves an attachment, that is a
        reading changing, and it must appear as a diff rather than as a silent
        difference between what tests measure and what production would do."""
        recorded, live = RecordedReader(), StanzaReader("en")
        drifted = []

        for case in load():
            if case.language != "en" or not recorded.has(case.text, "en"):
                continue
            was = recorded.parse(case.text, "en")
            now = live.parse(case.text, "en")
            if _edges(was) != _edges(now):
                drifted.append(case.id)

        assert not drifted, (
            f"{len(drifted)} recorded parses no longer match the parser: "
            f"{drifted[:8]}. Re-record with "
            "`python corpus/parser/record_parses.py`, and read the diff — a "
            "changed attachment is a changed reading, not a formatting change")

    def test_the_recording_says_which_parser_made_it(self):
        """Provenance, so the diff above can be explained rather than only
        observed. A recording of unknown origin cannot be checked against
        anything."""
        recorded = RecordedReader()
        assert recorded.recorded_with, "no provenance in the recording"
        for language, provenance in recorded.recorded_with.items():
            assert provenance["parser"].startswith("stanza@")
            assert language in provenance["model"]


def _edges(parse) -> list:
    """Attachment structure only — the thing this layer scores.

    Character offsets and surface forms are excluded deliberately: a tokenizer
    that splits a hyphen differently is a real change but not a change of
    attachment, and folding the two together would make every upgrade look like
    a semantic regression.
    """
    return [(sentence.sentence_id, token.index, token.lemma, token.head,
             token.relation)
            for sentence in parse.sentences for token in sentence.tokens]
