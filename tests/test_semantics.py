"""Tier 3, for the cases the deterministic path can actually answer.

Every case here runs the whole pipeline — normalise, parse from the recording,
bind, propose, fuse — and asserts the field and the value the corpus expects.
Nothing is read from `closure.json` except *which* cases to run; the values are
re-derived, because a test that compared a report against itself would pass
whatever the report happened to say.

The rest of tier 3 is not skipped and not deleted. It stays counted in
`AWAITING_A_PARSER`, and `closure.json` says why each one is still there — which
is the difference between "50 cases we have not got to" and "36 with no literal
to normalise, 6 with nothing to bind, 4 with no model fetched".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus.parser.loader import load
from src.discovery.binding import bind
from src.discovery.fusion import Fusion, Proposal, fuse
from src.discovery.semantics import propose
from src.discovery.syntax import normalize
from src.discovery.syntax_stanza import RecordedReader

RECORDED = RecordedReader()
CLOSURE = json.loads(
    (Path(__file__).resolve().parent.parent
     / "corpus" / "parser" / "closure.json").read_text())

#: The ids the report says the deterministic path answers. Used only to choose
#: the cases; every value below is recomputed.
#: Cases the pipeline answers *and* answers correctly. Both witnesses agreeing
#: and one witness alone are different evidence, and both are run here — what
#: is excluded is anything whose value the corpus does not expect, because a
#: case the pipeline answers wrongly is not a case it answers.
ANSWERABLE = [row["id"] for row in CLOSURE["rows"]
              if row["state"] in ("AGREE", "MODEL_ONLY_ACCEPTED")
              and row.get("matches_expected")]

CASES = [c for c in load() if c.id in ANSWERABLE]


def run(case):
    """Both witnesses, end to end, exactly as the pipeline runs them."""
    from src.discovery.hosted_recording import RecordedHostedReader
    from src.discovery.pipeline import read
    from src.discovery.schema import QUANTIFY_SCHEMA

    hosted = RecordedHostedReader()
    result = read(case.text, RECORDED.parse(case.text, case.language),
                  hosted.read(case.text, QUANTIFY_SCHEMA), QUANTIFY_SCHEMA,
                  language=case.language)
    wanted = case.asserts["field"]
    decision = result.by_field.get(wanted)
    assert decision is not None, (
        f"no decision for {wanted!r}; the closure report says this case is "
        "answerable, so one of the two is out of date")
    return result, decision


def test_the_report_and_this_file_agree_on_what_is_answerable():
    """If the report claims more than this file runs, cases are being counted
    as handled without anything asserting them."""
    assert len(CASES) == len(ANSWERABLE) > 0


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_the_pipeline_produces_the_field_and_the_value(case):
    from src.discovery.fusion import REQUIREMENTS, Requirement, same_value

    _, decision = run(case)
    assert decision.outcome is Fusion.AGREE, decision.detail
    rule = REQUIREMENTS.get(decision.dimension, Requirement()).compare_as
    assert same_value(decision.value, case.asserts["value"], rule), (
        f"{case.text!r} -> {decision.value!r}, expected "
        f"{case.asserts['value']!r} (compared as {rule}). {case.note}")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_every_settled_field_names_a_witness(case):
    """A value with no reader behind it is a field nobody can trace back to a
    sentence, which is the audit property the whole runtime promises."""
    _, decision = run(case)
    assert decision.model is not None or decision.syntax


class TestTheTwoCadencesSeparate:
    """The sentence the layer was built for, now going all the way through."""

    TEXT = "contribute $500 monthly, rebalanced annually"

    def candidates(self):
        values = normalize(self.TEXT)
        return propose(bind(RECORDED.parse(self.TEXT), values), values)

    def test_the_contribution_cadence_fills_cadence(self):
        by_field = {c.field: c.value for c in self.candidates()}
        assert by_field["cadence"] == "monthly"

    def test_the_rebalancing_cadence_fills_a_different_field(self):
        """Not `cadence`. A reader that collected both and picked one would
        have a fifty-fifty chance of contributing annually."""
        by_field = {c.field: c.value for c in self.candidates()}
        assert by_field["rebalancing_cadence"] == "annual"

    def test_neither_field_takes_the_other_s_value(self):
        by_field = {c.field: c.value for c in self.candidates()}
        assert by_field["cadence"] != by_field["rebalancing_cadence"]
