"""The PostgreSQL-only boundary, kept honest.

`docs/Runbook.md` claims a set of properties are proven only
against PostgreSQL. A document making that claim is worth exactly as much as
the gating behind it, so this checks the gating rather than the prose.

The failure this prevents is quiet: a guarantee whose test stops being gated
starts passing on SQLite, the document still says PostgreSQL proved it, and
nobody finds out until the property is needed.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

DOCUMENT = Path("docs/Runbook.md")

#: Files holding a PostgreSQL-only guarantee. Listed here so a file that loses
#: its gate is caught, rather than discovered by it quietly passing on SQLite.
GATED = (
    "test_postgres_concurrency.py",
    "test_postgres_tenancy.py",
    "test_postgres_deletion.py",
    "test_postgres_tampering.py",
    "test_tenancy_invariant.py",
    "test_immutability.py",
    "test_referential_policy.py",
    "test_decimal_columns.py",
    "test_json_columns.py",
    "test_migration_parity.py",
    "test_transfer.py",
)


class TestEveryGuaranteeIsActuallyGated:
    @pytest.mark.parametrize("name", GATED)
    def test_the_file_reads_the_gate_variable(self, name):
        body = (Path("tests") / name).read_text()
        assert "QUANTIFY_TEST_POSTGRES_URL" in body, (
            f"{name} holds a PostgreSQL-only guarantee and does not read the "
            "gate, so it either runs on SQLite or does not run at all")

    @pytest.mark.parametrize("name", GATED)
    def test_the_file_skips_rather_than_passes(self, name):
        """Skipping is the honest outcome. Passing without PostgreSQL would
        report a guarantee nothing established."""
        body = (Path("tests") / name).read_text()
        assert "skip" in body

    def test_no_gated_file_has_been_quietly_ungated(self):
        present = {path.name for path in Path("tests").glob("test_*.py")
                   if "QUANTIFY_TEST_POSTGRES_URL" in path.read_text()}
        assert set(GATED) <= present, (
            f"these lost their PostgreSQL gate: {set(GATED) - present}")


class TestTheDocumentDescribesWhatExists:
    def test_it_exists(self):
        assert DOCUMENT.exists()

    def test_it_states_the_isolation_level_the_protocol_relies_on(self):
        body = DOCUMENT.read_text()
        assert "READ COMMITTED" in body

    def test_it_records_the_defects_that_earned_the_boundary(self):
        """A boundary asserted without evidence reads as caution. These were
        found, not anticipated."""
        body = DOCUMENT.read_text()
        for defect in ("foreign_keys = ON", "INSERT OR REPLACE",
                       "NUMERIC", "OwnershipPath"):
            assert defect in body, f"the document does not record {defect}"

    def test_it_does_not_claim_sqlite_proves_concurrency(self):
        body = DOCUMENT.read_text()
        does_not = body.split("## What SQLite does not prove")[1]
        for claim in ("Row locking", "Concurrent writer",
                      "Alembic migration parity"):
            assert claim in does_not
