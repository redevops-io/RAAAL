"""Adjudicated shadow rows, as permanent fixtures.

`corpus/adjudicated.json` records ten contested readings and which reader was
right. Left as a report it would be read once; as tests it survives.

Three defect classes, and they are asserted differently on purpose:

    COMPILER   the legacy reader is wrong. Asserted as a *known* defect, so it
               is visible without failing the build — this reader is being
               replaced, and failing on its known faults would mean a red
               suite until Phase 4.
    MODEL      the hosted reader is wrong. Asserted only where it can be
               checked offline; a live call in the suite would make the build
               depend on a provider.
    SCHEMA     neither reader is wrong and the schema cannot represent the
               sentence. Asserted as a *gap*, so closing it fails this test
               and forces the fixture to be revisited.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
ADJUDICATED = json.loads((CORPUS / "adjudicated.json").read_text())
PROMPTS = {r["ref"]: r["prompt"]
           for r in json.loads((CORPUS / "strategies.json").read_text())}


def rows(defect: str):
    return [r for r in ADJUDICATED["rows"] if r["defect"] == defect]


class TestTheAdjudicationIsWellFormed:
    def test_every_row_names_a_defect_class_and_a_reason(self):
        for row in ADJUDICATED["rows"]:
            assert row["defect"] in ("COMPILER", "MODEL", "SCHEMA")
            assert row["why"], f"{row['ref']} has no reason"
            assert row["ref"] in PROMPTS, f"{row['ref']} is not in the corpus"

    def test_the_summary_matches_the_rows(self):
        """A summary that drifts from its own rows is the failure this project
        has already had twice."""
        counted = {}
        for row in ADJUDICATED["rows"]:
            counted[row["defect"]] = counted.get(row["defect"], 0) + 1
        for defect, n in ADJUDICATED["summary"].items():
            if defect == "reading":
                continue
            assert counted.get(defect) == n, f"{defect}: {counted.get(defect)} != {n}"

    def test_a_row_nobody_could_be_right_about_has_no_correct_value(self):
        for row in rows("SCHEMA"):
            assert row["correct"] is None and row["right"] == "neither"


class TestKnownCompilerDefects:
    """Recorded, not failing. This reader is being replaced; a red suite until
    Phase 4 would train everyone to ignore it."""

    @pytest.mark.parametrize(
        "ref", [r["ref"] for r in rows("COMPILER")
                if r["dimension"] == "evaluation_period"])
    def test_cadence_plus_window_is_still_read_as_rolling(self, ref):
        """Five of the ten contested rows, one defect: 'each month ... past 5
        years' is a cadence and a window, and the ROLLING detector takes it for
        many windows. The same class as the collision closed earlier, so that
        fix was narrower than it looked."""
        from src.discovery import QUANTIFY_SCHEMA
        from src.discovery.readers_quantify import CompilerReader

        read = CompilerReader().read(PROMPTS[ref], QUANTIFY_SCHEMA)
        window = read.value_of("evaluation_period")
        assert window is not None
        if str(window.value) != "rolling:unresolved":
            pytest.fail(
                f"{ref} no longer reads as rolling — it reads "
                f"{window.value!r}. If that is the fix, delete this test and "
                "move the row to a passing fixture.")

    def test_the_truncation_defect_is_still_there(self):
        """WM-0044: a three-way description read as its first token."""
        from src.discovery import QUANTIFY_SCHEMA
        from src.discovery.readers_quantify import CompilerReader

        read = CompilerReader().read(PROMPTS["WM-0044"], QUANTIFY_SCHEMA)
        assets = read.value_of("assets")
        if assets is None or str(assets.value) != "US":
            pytest.fail("WM-0044 assets no longer reads 'US'; re-adjudicate")


class TestTheSchemaGapIsOpen:
    def test_account_type_still_cannot_hold_a_conversion(self):
        """'Convert my traditional IRA to a Roth' names two accounts and the
        schema has one field. Both readers report a true fact and the schema
        forces them to disagree.

        This fails when the gap is closed, which is the point: the fixture has
        to be revisited rather than silently becoming stale.
        """
        from src.discovery import QUANTIFY_SCHEMA

        names = QUANTIFY_SCHEMA.names
        assert not ({"account_from", "account_to", "conversion"} & names), (
            "the schema now represents a conversion — re-adjudicate WM-0073 "
            "and remove this test")


class TestTheModelRecallMiss:
    def test_it_is_recorded_with_the_sentence_that_shows_it(self):
        """WM-0017 names VTI and BND and the hosted reader returned one.

        Not executed against the provider: a suite that calls a model depends
        on a provider being up, and a flaky red build is worse than a recorded
        finding. The shadow run is where this is re-measured.
        """
        miss = next(r for r in rows("MODEL"))
        assert miss["ref"] == "WM-0017"
        assert "VTI" in PROMPTS[miss["ref"]] and "BND" in PROMPTS[miss["ref"]]
        assert miss["correct"] == "VTI, BND"
