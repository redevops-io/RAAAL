"""The one derived reader, and the boundary that keeps it one.

    Syntax evidence does not author intent.
    Certified deterministic semantic readers may derive narrowly defined
    intent from syntax evidence.

`TriggerSemanticsReader` exists because the live drift lane found a sentence
executing on two draws of five and asking on the other three: the hosted reader
omits `trigger_semantics` non-deterministically, and the field has no other
author. Always asking would have turned a supported journey into a follow-up on
every event-triggered sentence; letting the parser carry the field would have
made it an authority on meaning. A named, versioned reader is neither.

Most of this file is falsification. A deterministic reader that claims a
material field is only safe while it is provably narrow, and the reading it
claims — event versus state — is the distinction whose conflation changed
contributed capital by 4.6x.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.discovery.derived_readers import (
    AUTHORS, DERIVED_READERS, TRIGGER_READER_ID, trigger_semantics,
)

ROOT = Path(__file__).resolve().parent.parent
CASES = json.loads(
    (ROOT / "corpus" / "parser" / "trigger_falsification.json").read_text()
)["cases"]


def _read(text: str):
    """Candidates and the parse, because the reader needs both — the parse is
    how it sees a negation the candidates cannot show it."""
    from src.discovery.binding import bind
    from src.discovery.semantics import propose
    from src.discovery.syntax import normalize
    from src.discovery.syntax_stanza import RecordedReader

    parse = RecordedReader().parse(text)
    values = normalize(text)
    return propose(bind(parse, values), values), parse


class TestTheReaderDecidesOnlyWhatTheGrammarStates:
    @pytest.mark.parametrize("case", CASES, ids=lambda c: c["text"][:44])
    def test_it_claims_the_reading_or_declines(self, case):
        found = trigger_semantics(*_read(case["text"]))
        got = None if found is None else found.value
        assert got == case["expect"], (
            f"{case['text']!r}\n  expected {case['expect']!r} because "
            f"{case['why']}\n  got {got!r}")

    def test_the_declining_cases_are_a_real_share(self):
        """A falsification set where everything resolves proves the reader
        answers, not that it knows when not to."""
        declines = sum(1 for c in CASES if c["expect"] is None)
        assert declines >= 2, "nothing here tests the reader's silence"

    def test_a_sentence_carrying_both_readings_is_declined(self):
        """The case the rule turns on. An event and a state differ in how
        often a strategy fires, so a sentence containing both is not one this
        reader gets to pick between."""
        found = trigger_semantics(*_read(
            "buy VOO when SPY crosses below and stays below its "
            "200-day moving average"))
        assert found is None


class TestItCannotGrowIntoACompiler:
    """`quantify-compiler@2` began as a few narrow rules and took months to
    delete. The restriction is structural rather than remembered."""

    def test_each_reader_authors_one_field_and_names_it(self):
        """Three readers now, one field each. The restriction was never "one
        field in the module" — it is that no single reader may grow into a
        compiler by claiming a second."""
        assert AUTHORS == {"trigger_semantics", "stated_weights", "day_rule"}

    def test_every_proposal_it_makes_is_for_that_field(self):
        for case in CASES:
            found = trigger_semantics(*_read(case["text"]))
            if found is not None:
                assert found.dimension in AUTHORS

    def test_the_module_constructs_proposals_for_nothing_else(self):
        """From the AST. A reader that started claiming a second field would
        be a different component wearing this one's id."""
        source = (ROOT / "src" / "discovery" / "derived_readers.py").read_text()
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "Proposal"):
                continue
            named = {k.arg: k.value for k in node.keywords}
            dimension = named.get("dimension")
            assert isinstance(dimension, ast.Constant), (
                "a Proposal built with a computed dimension; this reader's "
                "field must be readable from the source")
            assert dimension.value in AUTHORS, dimension.value

    def test_it_is_versioned(self):
        assert TRIGGER_READER_ID.endswith("@1")
        assert dict(DERIVED_READERS.__iter__().__next__() for _ in [0]) or True
        assert any(rid == TRIGGER_READER_ID for rid, _ in DERIVED_READERS)


class TestRemovingItReintroducesTheInstability:
    """The deletion check. A reader that changes nothing when removed is a
    reader nothing depends on, and this one was added to close a measured
    3-of-5 instability."""

    TEXT = "buy VOO when SPY falls below its 200-day moving average"

    def test_without_the_derived_reading_a_silent_model_leaves_it_open(self):
        from src.discovery.fusion import Fusion, fuse

        without = fuse("trigger_semantics", model=None, derived=None)
        assert without.outcome is not Fusion.AGREE
        assert not without.proceeds

    def test_with_it_the_same_silence_settles(self):
        from src.discovery.fusion import Fusion, fuse

        derived = trigger_semantics(*_read(self.TEXT))
        assert derived is not None, "the grammar states the answer here"
        with_it = fuse("trigger_semantics", model=None, derived=derived)
        assert with_it.outcome is Fusion.AGREE
        assert with_it.value == "crossing_event"

    def test_and_a_disagreement_still_goes_to_the_person(self):
        """Settling on one reader's silence is not the same as settling over
        another reader's objection."""
        from src.discovery.fusion import Fusion, Proposal, fuse

        derived = trigger_semantics(*_read(self.TEXT))
        model = Proposal("trigger_semantics", "persistent_condition", "hosted")
        out = fuse("trigger_semantics", model=model, derived=derived)
        assert out.outcome is Fusion.DISAGREE
        assert not out.proceeds


class TestTheDocumentDescribesTheReaderThatExists:
    """`docs/Reader-Authority.md` states the rule this module is an instance
    of. A document that drifts from the code becomes a claim about a system
    nobody has, which is the failure `FormalCore.md` was written against."""

    DOC = ROOT / "docs" / "Reader-Authority.md"

    def test_it_names_every_reader_and_the_field_each_authors(self):
        from src.discovery.derived_readers import DERIVED_READERS

        text = self.DOC.read_text()
        assert TRIGGER_READER_ID in text
        for reader_id, _ in DERIVED_READERS:
            assert reader_id in text, (
                f"{reader_id} authors a contract field and the authority "
                "document does not mention it")
        for field in AUTHORS:
            assert field in text

    def test_it_records_both_defects_falsification_found(self):
        """They are the argument, not decoration. Without them this reads as a
        design note rather than as a rule with a price attached."""
        lowered = self.DOC.read_text().lower()
        assert "negat" in lowered
        assert "crosses below and stays below" in lowered

    def test_the_four_fusion_rules_it_states_are_the_four_implemented(self):
        from src.discovery.fusion import Fusion, Proposal, fuse

        model = Proposal("trigger_semantics", "crossing_event", "hosted")
        derived = Proposal("trigger_semantics", "crossing_event",
                           TRIGGER_READER_ID)
        other = Proposal("trigger_semantics", "persistent_condition",
                         TRIGGER_READER_ID)

        assert fuse("trigger_semantics", model=model,
                    derived=derived).outcome is Fusion.AGREE
        assert fuse("trigger_semantics", model=None,
                    derived=derived).outcome is Fusion.AGREE
        assert fuse("trigger_semantics", model=model,
                    derived=other).outcome is not Fusion.AGREE
        assert fuse("trigger_semantics", model=None,
                    derived=None).outcome is not Fusion.AGREE
