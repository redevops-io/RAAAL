"""The ledger must account for every row, from a run entitled to be read.

Two properties, and the second is the one that was violated repeatedly:

    completeness   every adjudicated row carries one of the declared
                   dispositions, and the summary loses none of them
    eligibility    the run the ledger was built from passed its validity gate

The second exists because three separate measurements were quoted before this
check did: one where a dry-run probe had overwritten the real result, one where
16 replies were truncated by the harness's own token ceiling, and one where the
comparator was supplying a capability to a reader and then scoring the two as
agreeing about it. In each, the values in the file looked entirely normal.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
LEDGER = CORPUS / "ledger.json"


@pytest.fixture(scope="module")
def ledger():
    if not LEDGER.exists():
        pytest.skip("no ledger built; run `python corpus/ledger.py --write`")
    return json.loads(LEDGER.read_text())


class TestEveryRowIsAccountedFor:
    def test_no_row_lacks_a_disposition(self, ledger):
        """There is deliberately no UNADJUDICATED bucket. A bucket like that
        fills up and is then read as 'nothing to see'."""
        missing = [r for r in ledger["rows"] if not r.get("disposition")]
        assert not missing, f"{len(missing)} rows carry no disposition"

    def test_every_disposition_is_declared(self, ledger):
        declared = set(ledger["dispositions"])
        used = {r["disposition"] for r in ledger["rows"]}
        assert used <= declared, f"undeclared: {sorted(used - declared)}"

    def test_the_summary_loses_no_rows(self, ledger):
        """It printed 73 of 80 once, under a heading that did not say so."""
        assert sum(ledger["counts"].values()) == len(ledger["rows"])

    def test_every_row_states_its_reason_and_how_it_was_reached(self, ledger):
        """An unexplained disposition is a vote. `evidence_source` separates a
        mechanical rule from a human judgement without anyone having to ask."""
        for row in ledger["rows"]:
            assert row["reason"], f"{row['prompt_id']}/{row['dimension']}"
            assert row["evidence_source"], f"{row['prompt_id']}"

    def test_every_row_carries_its_run_identity(self, ledger):
        """So a row can never be read against a run it did not come from."""
        for row in ledger["rows"]:
            assert row["schema_fingerprint"] and row["run_id"]


class TestTheLedgerRefusesAnIneligibleRun:
    def test_it_checks_mode_truncation_and_reader_enabled(self):
        from corpus.ledger import check_validity

        assert check_validity({"provenance": {
            "mode": "full", "truncated": 0,
            "readers": {"claude-sonnet-5@1": {"enabled": True}}}}, "x") == []

        assert check_validity({"provenance": {
            "mode": "dryrun", "truncated": 0,
            "readers": {"claude-sonnet-5@1": {"enabled": True}}}}, "x")
        assert check_validity({"provenance": {
            "mode": "full", "truncated": 16,
            "readers": {"claude-sonnet-5@1": {"enabled": True}}}}, "x")
        assert check_validity({"provenance": {
            "mode": "full", "truncated": 0,
            "readers": {"claude-sonnet-5@1": {"enabled": False}}}}, "x")

    def test_a_file_with_no_provenance_is_refused(self):
        """Every result file predating the manifest is ineligible by
        construction, which is the correct treatment of the runs whose numbers
        were withdrawn."""
        from corpus.ledger import check_validity

        assert check_validity({}, "x")


class TestModelOnlyRowsAreNotAdjudicatedHere:
    def test_they_are_counted_separately(self, ledger):
        """There is nothing to adjudicate when the compiler has no opinion.
        Their correctness is the canonical expectations' job."""
        assert ledger["exceeds"] > 0
        states = {r["comparison_state"] for r in ledger["rows"]}
        assert states <= {"CONTESTED", "ONE_SIDED"}
