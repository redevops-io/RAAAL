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
ANSWERABLE = [row["id"] for row in CLOSURE["rows"]
              if row["state"] == "MAPPED_AND_AGREED"]

CASES = [c for c in load() if c.id in ANSWERABLE]


def run(case):
    """The production path, end to end."""
    values = normalize(case.text, case.language)
    bindings = bind(RECORDED.parse(case.text, case.language), values)
    candidates = propose(bindings, values)
    wanted = case.asserts["field"]
    match = next((c for c in candidates if c.field == wanted), None)
    assert match is not None, (
        f"no candidate for {wanted!r}; the closure report says this case is "
        "answerable, so one of the two is out of date")
    decision = fuse(match.field,
                    model=Proposal(match.field, match.value,
                                   "deterministic-stand-in@1",
                                   match.source_span))
    return match, decision


def test_the_report_and_this_file_agree_on_what_is_answerable():
    """If the report claims more than this file runs, cases are being counted
    as handled without anything asserting them."""
    assert len(CASES) == len(ANSWERABLE) > 0


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_the_pipeline_produces_the_field_and_the_value(case):
    match, decision = run(case)
    assert decision.outcome is Fusion.AGREE, decision.detail
    assert str(match.value) == str(case.asserts["value"]), (
        f"{case.text!r} -> {match.value!r}, expected "
        f"{case.asserts['value']!r}. {case.note}")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_the_candidate_names_what_it_came_from(case):
    """A candidate with no source is a field nobody can trace back to a
    sentence, which is the audit property the whole runtime promises."""
    match, _ = run(case)
    assert match.source_value_id and match.binding_id
    assert match.evidence and any("->" in e for e in match.evidence)


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
