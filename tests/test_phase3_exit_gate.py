"""Phase 3 is complete when these hold — checked, not asserted.

Nine conditions. Each is here because leaving it implicit is how a phase gets
declared finished on the strength of a summary, and this project has already
withdrawn three sets of numbers that looked finished.

The gate matters most for what it makes possible: Phase 4 means *removing the
legacy reader*, and that is only safe if no correctness fixture depends on it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"


def matrix(corpus: str) -> dict:
    path = CORPUS / "shadow" / f"matrix-{corpus}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not built")
    return json.loads(path.read_text())


class TestPhase3ExitGate:
    def test_1_the_schema_the_matrices_were_built_under_is_recorded(self):
        """The frozen fingerprint is a record of Phase 3, not of today.

        It used to assert `frozen == current`, which was right while the schema
        was still frozen. `objective` has since been widened, and re-pointing
        this at @3 without re-running the shadow corpora would leave numbers
        computed under @2 wearing an @3 label — the same defect as a reader
        whose behaviour changed under an unchanged id.

        So the frozen file keeps saying what the matrices were built under, the
        matrices are checked against it, and the drift is declared in
        `shadow/STALE.md` rather than absorbed.
        """
        frozen = (CORPUS / "schema-frozen.txt").read_text().strip()
        assert "@2" in frozen
        for corpus in ("strategies", "catalogue"):
            assert matrix(corpus)["schema"] == frozen, (
                f"matrix-{corpus} was built under a different schema than the "
                "frozen record claims")

    def test_1a_any_drift_from_that_schema_is_declared(self):
        """Either the schema still matches the matrices, or a staleness note
        names the current fingerprint. A note that has itself gone stale is
        worse than none, so it must name today's schema exactly."""
        import sys

        from src.discovery import QUANTIFY_SCHEMA

        sys.path.insert(0, str(CORPUS))
        from shadow_run import schema_fingerprint

        current = schema_fingerprint(QUANTIFY_SCHEMA)
        frozen = (CORPUS / "schema-frozen.txt").read_text().strip()
        if current == frozen:
            return

        stale = CORPUS / "shadow" / "STALE.md"
        assert stale.exists(), (
            f"the schema moved to {current} and nothing says the shadow "
            "matrices are stale; declare it or re-run shadow_run.py")
        said = stale.read_text()
        assert current in said, (
            f"the staleness note does not name the current schema {current}; "
            "it was written for an earlier drift and is itself out of date")
        assert frozen in said

    @pytest.mark.parametrize("corpus", ["strategies", "catalogue"])
    def test_2_both_corpora_are_valid_runs(self, corpus):
        checks = matrix(corpus)["validity"]
        failed = [name for name, ok in checks.items() if not ok]
        assert not failed, f"{corpus} failed: {failed}"

    @pytest.mark.parametrize("corpus", ["strategies", "catalogue"])
    def test_3_both_were_measured_under_the_frozen_schema(self, corpus):
        frozen = (CORPUS / "schema-frozen.txt").read_text().strip()
        assert matrix(corpus)["provenance"]["schema_fingerprint"] == frozen

    def test_4_every_material_row_is_adjudicated(self):
        ledger = json.loads((CORPUS / "ledger.json").read_text())
        assert all(r["disposition"] for r in ledger["rows"])

    def test_5_there_is_no_residual_bucket(self):
        ledger = json.loads((CORPUS / "ledger.json").read_text())
        assert "UNADJUDICATED" not in ledger["dispositions"]
        assert "UNADJUDICATED" not in ledger["counts"]

    def test_6_schema_gaps_are_represented_canonically(self):
        """Both gaps found in Phase 3 have a relation that states them."""
        from src.discovery import QUANTIFY_SCHEMA

        assert {"portfolio_sleeves", "account_transition"} <= \
            QUANTIFY_SCHEMA.relation_kinds

    @pytest.mark.parametrize("corpus", ["strategies", "catalogue"])
    def test_7_relations_appear_in_the_matrix(self, corpus):
        assert any(k.startswith("rel:") for k in matrix(corpus)["matrix"])

    def test_8_both_hosted_reader_failure_classes_have_fixtures(self):
        cases = json.loads(
            (CORPUS / "expected" / "discovery.json").read_text())["cases"]
        classes = {c.get("failure_class") for c in cases}
        winners = {c.get("adjudicated_winner") for c in cases}
        assert "NON_DETERMINISM" in classes, "no consistency fixture"
        assert "COMPILER" in winners, "no recall-miss fixture"

    def test_9_no_correctness_fixture_depends_on_a_readers_identity(self):
        """The condition that makes Phase 4 a deletion rather than a rename."""
        cases = json.loads(
            (CORPUS / "expected" / "discovery.json").read_text())["cases"]
        for case in cases:
            blob = json.dumps(case["expected"]).lower()
            for identity in ("compiler", "model", "reader", "winner",
                             "sonnet", "claude", "quantify-"):
                assert identity not in blob, (
                    f"{case['ref']} expects something about a reader, so the "
                    "legacy reader cannot be deleted without changing it")
