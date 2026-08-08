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
    def test_1_the_schema_is_frozen(self):
        frozen = (CORPUS / "schema-frozen.txt").read_text().strip()
        from src.discovery import QUANTIFY_SCHEMA
        import sys
        sys.path.insert(0, str(CORPUS))
        from shadow_run import schema_fingerprint

        assert frozen == schema_fingerprint(QUANTIFY_SCHEMA)
        assert "@2" in frozen

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
