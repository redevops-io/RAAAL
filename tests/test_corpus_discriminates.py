"""The corpus is the instrument, so the instrument gets checked.

`corpus/harness.py validate` proves each labelled expectation could fail.
Running it here makes that a build-time property rather than something someone
remembers to do — a corpus of expectations nothing can violate is a list of
opinions, and it would pass quietly forever.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"


def test_both_corpora_are_in_the_repository():
    """Phase 0's gate: they reproduce from a clean clone. The vendor manifest
    was gitignored and its tests passed only where someone had fetched it by
    hand; a corpus outside the repo is the same failure with a longer fuse."""
    assert (CORPUS / "strategies.json").exists()
    assert (CORPUS / "catalogue.json").exists()
    assert len(json.loads((CORPUS / "strategies.json").read_text())) == 35
    assert len(json.loads((CORPUS / "catalogue.json").read_text())) == 144


def test_every_labelled_expectation_discriminates():
    from corpus.harness import validate

    assert validate() == 0, (
        "a labelled expectation cannot be violated, so it measures nothing")


def test_the_validator_checks_a_meaningful_share_of_the_corpus():
    """Guards the guard. `validate` skips CLARIFY and UNKNOWN rows, and a
    version that skipped everything would also return 0."""
    rows = json.loads((CORPUS / "strategies.json").read_text())
    checkable = [r for r in rows if r["expectation"] in ("EXECUTE", "REFUSE")]
    assert len(checkable) >= 20, (
        f"only {len(checkable)} rows are checkable; the validator would be "
        "reporting success over almost nothing")


def test_a_harness_error_is_not_a_refusal():
    """An evaluator that reports its own crash as a product refusal
    manufactures evidence."""
    from corpus.harness import HARNESS_ERROR, REFUSE

    assert HARNESS_ERROR != REFUSE
