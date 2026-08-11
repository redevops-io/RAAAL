"""Both witnesses through fusion, on recorded evidence.

The hosted reader runs on every utterance rather than only when the
deterministic path is silent, because a reader consulted only as a fallback
inherits the other's authority wherever the other speaks. Both are replayed
from recordings — one call per sentence, made once — so this suite needs no
network and cannot be moved by a provider changing its mind between runs.

**A recording is not an answer.** It replays what the model proposed; fusion
still decides. `TestARecordingIsNotAuthority` holds that line, because "it is
in the fixture" is the most natural way for a stored reading to quietly become
a settled field.
"""
from __future__ import annotations

import pytest

from src.discovery.fusion import Fusion, same_value
from src.discovery.hosted_recording import RecordedHostedReader, proposals
from src.discovery.pipeline import read
from src.discovery.reader import ReadingSet
from src.discovery.schema import QUANTIFY_SCHEMA
from src.discovery.syntax_stanza import RecordedReader

HOSTED = RecordedHostedReader()
PARSES = RecordedReader()

YEAR_END = "I contribute monthly and rebalance at year end"
BOTH_CADENCES = "contribute $500 monthly, rebalanced annually"
AMBIGUOUS = "rebalance to 70/30"


def run(text: str):
    return read(text, PARSES.parse(text), HOSTED.read(text, QUANTIFY_SCHEMA),
                QUANTIFY_SCHEMA)


class TestTheRecordingsAreUsable:
    def test_both_witnesses_have_recordings_for_the_acceptance_sentences(self):
        for text in (YEAR_END, BOTH_CADENCES, AMBIGUOUS):
            assert HOSTED.has(text), f"no model recording for {text!r}"
            assert PARSES.has(text), f"no parse recording for {text!r}"

    def test_a_recording_carries_the_identity_of_what_produced_it(self):
        """Model, prompt version and schema version. A reply produced under a
        different prompt or schema is a reply to a different question, and a
        recording that cannot say which is one the drift lane cannot check."""
        provenance = HOSTED.recorded_with
        assert provenance["model"] and provenance["reader_id"]
        assert provenance["prompt_version"].startswith("quantify-hosted-prompt@")
        assert provenance["schema_version"] == QUANTIFY_SCHEMA.version

    def test_reading_under_a_different_schema_is_refused(self):
        """Not silently replayed. The recorded answer was to a different
        question."""
        from dataclasses import replace

        other = replace(QUANTIFY_SCHEMA, version="quantify-discovery-schema@99")
        with pytest.raises(ValueError):
            HOSTED.read(YEAR_END, other)

    def test_a_missing_recording_raises_rather_than_calling_out(self):
        """A quiet fallback to a live call would hide that the model was never
        consulted for a sentence, behind a green run."""
        with pytest.raises(KeyError):
            HOSTED.read("a sentence nobody recorded", QUANTIFY_SCHEMA)


class TestARecordingIsNotAuthority:
    def test_a_recorded_reading_still_goes_through_fusion(self):
        """Every field of every decision has an outcome. If a recording could
        settle a field by existing, there would be values here with no decision
        behind them."""
        result = run(BOTH_CADENCES)
        assert result.decisions
        for decision in result.decisions:
            assert decision.outcome in set(Fusion)

    def test_a_recorded_reading_can_still_be_refused(self):
        """The property that makes the previous test mean something: a stored
        proposal is not automatically settled."""
        result = run(AMBIGUOUS)
        assert any(not d.proceeds for d in result.decisions), (
            "every recorded reading proceeded, so nothing here distinguishes "
            "fusion from a passthrough")

    def test_a_failed_reading_produces_no_proposals(self):
        """A transport failure is not a reading. Scoring one as silence would
        let a timeout look like a model with nothing to say."""
        assert proposals(ReadingSet(reader_id="x", failed="timeout")) == ()


class TestNeitherWitnessIsPrivileged:
    def test_the_model_alone_proceeds(self):
        """`objective` has no deterministic producer at all, and the model's
        reading of it still settles — silence from syntax is not an argument."""
        result = run(BOTH_CADENCES)
        assert result.by_field["objective"].outcome is Fusion.AGREE

    def test_no_contract_field_is_currently_proposed_by_syntax_alone(self):
        """Recorded because it is the result of the alignment pass, not a gap.

        Before contract names were canonical, `rebalancing_cadence` looked like
        a real `DISAGREE` with the model silent. It was not: the schema calls
        that dimension `periodic_rebalancing`, and the model could only ever
        answer in schema terms. Once both witnesses speak the same vocabulary,
        the deterministic path proposes no contract field the model missed —
        on any recorded sentence.

        So the syntax-alone policy is exercised synthetically in
        `test_fusion.py` and has no live instance. That is a fact about these
        sentences rather than about the rule, and it is asserted so that the
        first sentence which *does* produce one is noticed.
        """
        for text in (BOTH_CADENCES, YEAR_END, AMBIGUOUS):
            alone = [d.dimension for d in run(text).decisions if d.model is None]
            assert not alone, (
                f"{text!r} now proposes {alone} from syntax alone. Check it is "
                "a real reading the model missed rather than a vocabulary "
                "mismatch, then make it the DISAGREE acceptance case")

    def test_agreement_carries_the_model_s_value(self):
        result = run(BOTH_CADENCES)
        assert result.by_field["cadence"].value == "monthly"


class TestAmbiguityNeedsBothReadingsOnTheTable:
    """The narrowing, and the pair that shows it discriminates."""

    def test_a_sentence_carrying_both_readings_is_ambiguous(self):
        """`rebalance to 70/30` can mean restore an existing 70/30 target or
        change the target to 70/30. Both readings need a target, and this
        sentence has one — `stated_weights` is proposed alongside."""
        result = run(AMBIGUOUS)
        assert result.by_field["periodic_rebalancing"].outcome is (
            Fusion.AMBIGUOUS_BY_LANGUAGE)
        assert "stated_weights" in result.by_field

    def test_the_same_word_without_a_target_is_not(self):
        """`rebalanced annually` carries the word and only one reading: the
        competing one needs a target and there is none. Firing here would be
        firing on the vocabulary rather than on the ambiguity."""
        result = run(BOTH_CADENCES)
        assert result.by_field["periodic_rebalancing"].outcome is Fusion.AGREE


class TestTheContractVocabularyIsCanonical:
    def test_every_decision_is_about_a_contract_dimension(self):
        """Readers, mappers, fusion and corpus assertions all speak contract
        field names at this boundary. The moment two witnesses name the same
        thing differently they can never agree about it."""
        dimensions = {d.name for d in QUANTIFY_SCHEMA.dimensions}
        for text in (BOTH_CADENCES, YEAR_END, AMBIGUOUS):
            for decision in run(text).decisions:
                assert decision.dimension in dimensions, decision.dimension

    def test_intermediate_semantics_are_kept_rather_than_dropped(self):
        """`rebalancing_cadence=annual` is a real reading of a real sentence.
        The contract has no field for it, so it does not enter fusion — but
        discarding it would lose the evidence that the contract might need
        one."""
        result = run(BOTH_CADENCES)
        assert any(c.field == "rebalancing_cadence" for c in result.intermediate)
        assert all(not c.is_contract_field for c in result.intermediate)


class TestFormattingIsNotDisagreement:
    def test_an_amount_agrees_across_two_spellings(self):
        """The model reads `$500`; the deterministic path normalises to `500`.
        A string comparison called that a conflict, which is the failure
        `Dimension.compare_as` has existed to prevent since the first shadow
        run — fusion was simply not using it."""
        assert run(BOTH_CADENCES).by_field["amount"].outcome is Fusion.AGREE

    def test_the_comparison_cannot_make_two_amounts_equal(self):
        """The discriminating opposite. A coercion that made everything agree
        would be worse than the string comparison it replaced."""
        assert same_value("$500", "500", "NUMBER")
        assert not same_value("$500", "$600", "NUMBER")

    def test_text_stays_exact(self):
        """No synonym table. "annual" and "yearly" stay contested, because
        resolving them needs a table nobody can audit."""
        assert not same_value("annual", "yearly", "TEXT")


class TestTheObservedParserFailureDoesNotReachFusion:
    """The acceptance case that was required, and the finding that it does not
    hold — recorded rather than worked around.

    Stanza attaches `year end` to *contribute* in this sentence. The intent was
    that the model would read it correctly and fusion would report a real
    `DISAGREE`. It does not, because **neither witness produces a reading for
    that dimension at all**: the deterministic path has no `day_rule` candidate
    (there is no normalised value in "at year end"), and the model returned no
    `day_rule` either.

    So the parser's error is real and currently unreachable — nothing consumes
    the attachment it got wrong. That is the reachability defect class this
    project already knows: a wrong component and a correct result, because
    nothing calls it. Making it reach fusion needs a producer for worded day
    rules, which is a normalisation question and not a fusion one.
    """

    def test_the_parse_still_has_the_error(self):
        parse = PARSES.parse(YEAR_END)
        sentence = parse.sentences[0]
        end = next(t for t in sentence.tokens if t.text == "end")
        governors = [g.lemma for g in sentence.governor_chain(end)]
        assert governors[0] == "contribute", (
            f"the parser now attaches 'year end' to {governors[0]!r}; if that "
            "is 'rebalance', re-read this whole class")

    def test_and_no_decision_is_made_about_it(self):
        result = run(YEAR_END)
        assert "day_rule" not in result.by_field, (
            "a day_rule decision exists now — the error has become reachable, "
            "and this class should become the DISAGREE acceptance case it was "
            "written to be")

    def test_the_model_did_not_read_a_day_rule_either(self):
        """Which is why this is not a fusion defect. Neither witness spoke."""
        reading = HOSTED.read(YEAR_END, QUANTIFY_SCHEMA)
        assert "day_rule" not in {r.dimension for r in reading.readings}
