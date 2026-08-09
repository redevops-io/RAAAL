"""The closure report is the metric, so it needs the same scrutiny as the code.

`AWAITING_A_PARSER` is a number that can be moved by writing rules. The report
answers the question that cannot be: *why* is each pending case still pending.
This file holds it to that.

The defect it is written from is one the report itself shipped with. When no
candidate matched the field a case asserts, `classify` fell back to
`candidates[0]` and reported `MAPPED_AND_AGREED` — so "when SPY crosses below
its 200-day average" was answered with a 200-day *holding period* and counted
as a success for trigger semantics. Six of thirteen agreements were that shape.
It is the comparator-manufactures-agreement defect this project has hit before,
reproduced inside the instrument built to measure the pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

from corpus.parser.closure import STATES, classify
from corpus.parser.loader import load
from src.discovery.syntax_stanza import RecordedReader

REPORT = json.loads(
    (Path(__file__).resolve().parent.parent
     / "corpus" / "parser" / "closure.json").read_text())
ROWS = REPORT["rows"]
RECORDED = RecordedReader()


def pending():
    return [c for c in load()
            if c.tier == "semantics"
            or (c.tier == "dependency" and not RECORDED.has(c.text, c.language))]


class TestTheReportDescribesTheCorpusItClaimsTo:
    def test_it_covers_every_pending_case(self):
        assert {row["id"] for row in ROWS} == {case.id for case in pending()}

    def test_it_is_reproducible(self):
        """The committed file must be what the code produces. A hand-edited
        report is a report that says whatever its last editor wanted."""
        recorded = RecordedReader()
        for case in pending():
            fresh = classify(case, recorded)
            stored = next(r for r in ROWS if r["id"] == case.id)
            assert fresh["state"] == stored["state"], case.id

    def test_every_row_lands_in_exactly_one_declared_state(self):
        for row in ROWS:
            assert row["state"] in STATES, row["id"]

    def test_every_row_says_why(self):
        """A state without a reason is a bucket, and a bucket is what this
        report exists instead of."""
        for row in ROWS:
            assert row.get("reason", "").strip(), row["id"]


class TestAgreementIsNotAssumed:
    def test_every_agreement_carries_the_value_it_produced(self):
        """Fusion's outcome and the corpus's expectation are different axes,
        and a state name carrying only the first is a green number for a wrong
        value."""
        for row in ROWS:
            if row["state"] == "MAPPED_AND_AGREED":
                assert "value" in row and "matches_expected" in row, row["id"]

    def test_no_agreement_is_recorded_with_the_wrong_value(self):
        wrong = [row["id"] for row in ROWS
                 if row["state"] == "MAPPED_AND_AGREED"
                 and not row["matches_expected"]]
        assert not wrong, (
            f"{wrong} agreed with a value the case does not expect. Either the "
            "mapping is wrong or the case is; both need saying out loud")

    def test_an_agreement_is_for_the_field_the_case_asserts(self):
        """The defect this file was written from. A candidate for some other
        field is not an answer to this case, however confident."""
        for row in ROWS:
            if row["state"] == "MAPPED_AND_AGREED":
                case = next(c for c in pending() if c.id == row["id"])
                assert row["field"] == case.asserts.get("field"), (
                    f"{row['id']} was answered with {row['field']!r} and "
                    f"asserts {case.asserts.get('field')!r}")

    def test_cases_answered_for_another_field_say_so(self):
        """They are `NO_FIELD_MAPPING` and they name what *was* proposed, so
        the next person can see how close the pipeline got rather than only
        that it stopped."""
        for row in ROWS:
            if row["state"] == "NO_FIELD_MAPPING" and "proposed" in row:
                assert row["proposed"], row["id"]
                assert "none for" in row["reason"] or "role pair" in row["reason"]


class TestTheStatesMeanDifferentThings:
    def test_no_literal_and_no_field_mapping_are_kept_apart(self):
        """They were one state until the counts made the difference matter, and
        they want opposite work: "weight by inverse volatility" has nothing to
        normalise, while "contribute a fixed $500" has a literal and a binding
        and is missing only a rule saying what the value means. One needs a
        reader; the other needs field derivation."""
        states = {row["state"] for row in ROWS}
        assert "STILL_UNSUPPORTED" not in states, (
            "the merged state is back; a pile is not a queue")
        assert {"NO_LITERAL", "NO_FIELD_MAPPING"} & states

    def test_every_state_names_who_owns_the_remaining_work(self):
        """"36 unsupported" is a pile. "25 waiting on the semantic reader, 15
        on field derivation" is a queue with owners."""
        for state in {row["state"] for row in ROWS}:
            assert REPORT["owner"][state].strip()

    def test_the_note_says_neither_is_a_defect_of_these_layers(self):
        assert "not defects" in REPORT["note"] or "not a defect" in REPORT["note"]

    def test_the_counts_add_up(self):
        assert sum(REPORT["by_state"].values()) == REPORT["pending"] == len(ROWS)

    def test_no_parse_recorded_is_kept_separate(self):
        """"The Spanish model was never fetched" and "this sentence has nothing
        to normalise" are different facts with different repairs."""
        no_parse = [r for r in ROWS if r["state"] == "NO_PARSE_RECORDED"]
        assert no_parse, "expected the non-English cases to be visible here"
        for row in no_parse:
            assert row["language"] != "en"
