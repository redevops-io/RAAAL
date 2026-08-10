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
        from src.discovery.hosted_recording import RecordedHostedReader

        recorded, hosted = RecordedReader(), RecordedHostedReader()
        for case in pending():
            fresh = classify(case, recorded, hosted)
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
            if row["state"] in ("AGREE", "MODEL_ONLY_ACCEPTED"):
                assert "value" in row and "matches_expected" in row, row["id"]
                assert row["compared_as"] in ("TEXT", "NUMBER", "SET")

    #: Cases where the *model* is wrong and the corpus is right, named so the
    #: failure is attributed rather than absorbed. Not a tolerance: each entry
    #: says which value the model produced and why the case stands.
    #:
    #: The expectation is never edited to match a draw. "the first trading day
    #: of the month" names a session, and `calendar_first_rolled_forward` is
    #: the 1st rolled off a holiday — a different day and a different figure.
    #: Rewriting the case would make the corpus assert whatever the reader last
    #: said, which is the one thing a regression corpus must not do.
    MODEL_IS_WRONG = {
        "sema-timing-day_rule-001":
            "read calendar_first_rolled_forward for 'the first trading day of "
            "the month', which names a session, not the 1st rolled forward",
        # `-002` was listed here too, on the assumption the same confusion
        # applied to the last trading day. It does not, and the staleness
        # check above caught the guess immediately.
    }

    def test_no_agreement_is_recorded_with_the_wrong_value(self):
        wrong = [row["id"] for row in ROWS
                 if row["state"] in ("AGREE", "MODEL_ONLY_ACCEPTED")
                 and not row["matches_expected"]]
        unexplained = [one for one in wrong if one not in self.MODEL_IS_WRONG]
        assert not unexplained, (
            f"{unexplained} agreed with a value the case does not expect. "
            "Either the mapping is wrong or the case is; both need saying out "
            "loud")

    def test_every_named_model_error_is_still_happening(self):
        """A list of known model errors that have quietly been fixed is a list
        that hides the next one. If a draw gets these right, remove them."""
        wrong = {row["id"] for row in ROWS
                 if row["state"] in ("AGREE", "MODEL_ONLY_ACCEPTED")
                 and not row["matches_expected"]}
        stale = set(self.MODEL_IS_WRONG) - wrong
        assert not stale, (
            f"{sorted(stale)} are listed as model errors and now read "
            "correctly; remove them so the list keeps meaning something")

    def test_an_agreement_is_for_the_field_the_case_asserts(self):
        """The defect this file was written from. A candidate for some other
        field is not an answer to this case, however confident."""
        for row in ROWS:
            if row["state"] in ("AGREE", "MODEL_ONLY_ACCEPTED"):
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


class TestBothDirectionsOfTheAsymmetry:
    """Fusion handles two witnesses, not one filling the other's gaps.

    Both directions are live in the corpus, which is what makes the policy
    evidence rather than a design note.
    """

    def test_the_model_alone_can_settle_a_field(self):
        """The common direction: the model reads dimensions nothing
        normalises, and syntax being silent is not an argument against it."""
        accepted = [r for r in ROWS if r["state"] == "MODEL_ONLY_ACCEPTED"]
        assert accepted, "no model-only witness"
        for row in accepted:
            assert row["witnesses"] == ["model"]

    def test_syntax_alone_cannot(self):
        """The reciprocal, and the one that was being mislabelled. It was
        reported as MODEL_ONLY_UNRESOLVED — the count was right and the label
        named the wrong witness, which is worse than either alone."""
        alone = [r for r in ROWS if r["state"] == "SYNTAX_ONLY_UNRESOLVED"]
        assert alone, (
            "no syntax-only witness in the corpus. The policy is then only "
            "exercised synthetically, and that should be said out loud")
        for row in alone:
            assert row["witnesses"] == ["syntax"]
            assert row["value"] is None, "an unresolved field carries no value"


class TestTheCorpusCannotBecomeASecondSchema:
    def test_expectations_are_validated_against_the_contract(self):
        """The preflight, and the class it closes: wrong field names, wrong
        value vocabularies, wrong unit coercion — caught three times in three
        passes, each time only because a second witness disagreed."""
        import json as _json
        import tempfile
        from pathlib import Path as _Path

        import pytest as _pytest

        from corpus.parser.loader import CorruptCorpus, load as _load

        for broken in ({"field": "not_a_dimension", "value": "x"},
                       {"field": "dividend_policy", "value": "cash"}):
            document = {"schema": "quantify-parser-corpus@1", "count": 1,
                        "cases": [{"id": "t-001", "tier": "semantics",
                                   "property": "t", "text": "x", "language": "en",
                                   "asserts": broken, "origin": "constructed",
                                   "note": ""}]}
            with tempfile.TemporaryDirectory() as folder:
                path = _Path(folder) / "cases.json"
                path.write_text(_json.dumps(document))
                with _pytest.raises(CorruptCorpus):
                    _load(path)

    def test_a_schema_gap_is_allowed_through(self):
        """The escape that means something: the reading is right and the
        contract has no value for it. That is the finding, not an error."""
        gaps = [r for r in ROWS if r["state"] == "SCHEMA_GAP"]
        assert gaps
        for row in gaps:
            assert "no such value" in row["reason"]


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

    def test_every_state_names_who_owns_the_remaining_work(self):
        """"36 unsupported" is a pile. "25 waiting on the semantic reader, 15
        on field derivation" is a queue with owners."""
        for state in {row["state"] for row in ROWS}:
            assert REPORT["owner"][state].strip()

    def test_the_note_says_neither_is_a_defect_of_these_layers(self):
        assert "not defects" in REPORT["note"] or "not a defect" in REPORT["note"]

    def test_the_counts_add_up(self):
        assert sum(REPORT["by_state"].values()) == REPORT["cases"] == len(ROWS)

    def test_agreement_by_two_witnesses_is_kept_apart_from_one(self):
        """A field settled by both readers independently and one settled by the
        model alone are different evidence. Collapsing them would let the report
        claim agreement it never observed."""
        for row in ROWS:
            if row["state"] == "AGREE":
                assert len(row["witnesses"]) > 1, row["id"]
            if row["state"] == "MODEL_ONLY_ACCEPTED":
                assert row["witnesses"] == ["model"], row["id"]

    def test_reclassified_cases_keep_their_history(self):
        """Excluding intermediates from the pending count is correcting the
        boundary being measured, not improving the number — so the report says
        they used to be counted."""
        excluded = REPORT["excluded_from_pending"]
        assert excluded["previously_counted_by_awaiting_a_parser"] is True
        assert excluded["count"] == len(excluded["ids"])

    def test_no_case_is_waiting_on_a_model_nobody_fetched(self):
        """The multilingual fixtures are deferred rather than pending. A case
        that cannot run is not a case that is waiting; it is out of scope, and
        counting it as pending measured intent rather than capability."""
        assert not [r for r in ROWS if r["state"] == "NO_PARSE_RECORDED"]

    def test_an_intermediate_nobody_computes_is_not_excluded_from_the_queue(self):
        """The defect this state was split out for. Classifying a case as
        intermediate on its asserted field alone, then excluding it from the
        pending count, put four cases outside the queue with nothing verifying
        them."""
        for row in ROWS:
            if row["state"] == "INTERMEDIATE_SEMANTIC":
                assert row.get("matches_expected"), row["id"]
