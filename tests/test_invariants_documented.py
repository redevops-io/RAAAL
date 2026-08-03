"""The invariants are checkable claims, not a reading list.

`Architecture.md` names six. Each was written after a defect appeared enough
times to be a shape, and each names a lane that must exist. A document listing
invariants nothing enforces would be the same failure it warns about — a
declaration with no reachable consumer.

This checks the cheap, real things: that each named invariant has a lane, that
the journey lane exists and runs against the deployed engine, and that the
document's claims about the three lanes match the suite.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ARCHITECTURE = Path("docs/Architecture.md")

#: Each invariant, and a test file that enforces it. Not decoration: an
#: invariant with no lane is a paragraph.
INVARIANTS = {
    "Reachability of enforcement": "tests/test_data_access.py",
    "Ownership reachability": "tests/test_postgres_deletion.py",
    "Constructed invalid state": "tests/test_transition_invariant.py",
    "Coverage evidence": "tests/test_tenancy_invariant.py",
    "Single resolution": "tests/test_single_resolution.py",
    "Journey completeness": "tests/test_provenance_journey.py",
}

#: The three lanes the document claims find different defect classes.
LANES = {
    "mechanism": "tests/test_provenance.py",
    "journey": "tests/test_provenance_journey.py",
    "deployment": "tests/test_container_reachability.py",
}


class TestEveryInvariantHasALane:
    @pytest.mark.parametrize("invariant", sorted(INVARIANTS))
    def test_it_is_named_in_the_document(self, invariant):
        assert invariant in ARCHITECTURE.read_text(), invariant

    @pytest.mark.parametrize("invariant", sorted(INVARIANTS))
    def test_something_enforces_it(self, invariant):
        lane = Path(INVARIANTS[invariant])
        assert lane.exists(), f"{invariant} names {lane}, which does not exist"
        assert lane.read_text().strip(), lane

    def test_the_document_lists_exactly_these(self):
        """A seventh added to the document without a lane, or a lane deleted
        while the document still claims it, both fail here."""
        body = ARCHITECTURE.read_text()
        table = body.split("| Invariant | The question it asks |", 1)[1]
        table = table.split("\n\n", 1)[0]
        named = {line.split("|")[1].strip() for line in table.splitlines()
                 if line.startswith("|") and "---" not in line}
        assert named == set(INVARIANTS), (
            f"document names {named - set(INVARIANTS)}, "
            f"lanes name {set(INVARIANTS) - named}")


class TestTheThreeLanesExist:
    @pytest.mark.parametrize("lane", sorted(LANES))
    def test_the_lane_exists(self, lane):
        assert Path(LANES[lane]).exists(), lane

    def test_the_journey_runs_against_the_deployed_engine(self):
        """A journey on SQLite would not have found either defect it found."""
        body = Path(LANES["journey"]).read_text()
        assert "QUANTIFY_TEST_POSTGRES_URL" in body
        assert "skipif" in body

    def test_the_deployment_lane_builds_the_real_image(self):
        body = Path(LANES["deployment"]).read_text()
        assert "docker" in body.lower()
        assert "Dockerfile" in body

    def test_the_lanes_are_distinct_files(self):
        """One file serving two lanes would mean one of them is not run."""
        assert len(set(LANES.values())) == len(LANES)


class TestTheDocumentIsHonestAboutFit:
    """A taxonomy claimed to be exhaustive and not becoming a way of not
    looking is the specific risk here, so the caveats are load-bearing."""

    def test_it_records_that_the_set_is_not_a_partition(self):
        body = ARCHITECTURE.read_text()
        assert "not a partition" in body

    def test_it_records_the_defect_that_is_not_on_the_list(self):
        body = ARCHITECTURE.read_text()
        assert "is not fixing the class" in body

    def test_it_records_that_defects_span_invariants(self):
        body = ARCHITECTURE.read_text()
        assert "more than one" in body
