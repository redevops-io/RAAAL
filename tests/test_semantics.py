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
from discovery_runtime.fusion import Fusion
from src.discovery.claims import Proposal
from src.discovery.semantics import INTERMEDIATE_FIELDS, propose
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
    from src.discovery.adapter import two_witness_run
    from src.discovery.hosted_recording import RecordedHostedReader
    from src.discovery.schema import QUANTIFY_SCHEMA

    hosted = RecordedHostedReader()
    result = two_witness_run(case.text, RECORDED.parse(case.text, case.language),
                             hosted.read(case.text, QUANTIFY_SCHEMA),
                             QUANTIFY_SCHEMA, language=case.language)
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


#: Cases whose expected value moved with a hosted re-recording rather than with
#: any code change. Listed rather than edited: rewriting the expectation to
#: match the latest draw would make the corpus assert whatever the model last
#: said, which is the opposite of a regression corpus.
#:
#: `sema-assets-stay_as_written-003` expects "an SPX ETF" and the current draw
#: returns "SPX ETF". The property the case exists for — that an asset is never
#: resolved to a ticker on the user's behalf — still holds; the leading article
#: is over-specification that only showed up once the model moved.
#: Emptied. Both entries were the same defect wearing two case ids: the model
#: dropped a leading article and the SET comparison treated it as a different
#: holding. Fixed where it belonged — `same_value` now ignores a leading
#: article for SET dimensions — rather than by listing whichever cases the last
#: draw happened to break.
DRIFTED: set = set()

#: Cases that were once answered correctly and are not answered now.
#:
#: `CASES` is derived from `closure.json`, which is regenerated. That means a
#: case the pipeline stops agreeing about does not fail — it stops being
#: collected, and the suite goes green with one fewer thing tested. This is the
#: corpus selecting itself down to whatever it can currently pass, which is the
#: failure mode a regression corpus exists to make impossible.
#:
#: It was not a hypothetical. Re-recording under schema `@6` moved
#: `sema-window-moving_average-013` from AGREE to DISAGREE and moved two
#: `day_rule` cases the other way. The total went 41 → 42, the suite stayed
#: green, and a case had silently left the tested set under cover of a number
#: going up. `corpus/parser/answerable.json` is the recorded set; this dict is
#: the only way out of it, and each entry has to say what happened.
#:
#: `sema-window-moving_average-013` was listed here and has been removed. The
#: unit ambiguity — syntax reading `12` from "the 12-month moving average" and
#: the hosted reader reading `252` — was a property of *that* reader.
#: gpt-4.1-2025-04-14 reads it as 12 and agrees with syntax, so the case is
#: answered again and the staleness rule required the entry out. The dimension
#: still has no unit; nothing about the schema improved. A reader that happens
#: to agree is not the same as an ambiguity being resolved, which is why the
#: queue entry in docs/Benchmark-Queue.md stays.
#: `sema-negation-changes_the_value-002` was listed here and is answered again.
#:
#: It is the reason the corpus moved to gpt-5.4-2026-03-05. Under
#: gpt-4.1-2025-04-14 `buy the index rather than through an ETF` read as
#: `assets='ETF'` — the model dropped the negation and returned the instrument
#: the sentence rejects, which reverses what is held. That is a wrong
#: executable meaning, the most dangerous class in the benchmark taxonomy, and
#: gpt-5.4 reads it correctly. The expectation was never edited to match the
#: reader that got it wrong; the reader changed.
LEFT_THE_ANSWERABLE_SET = {
    "sema-trigger-semantics-004":
        "`as long as SPY is under trend` settles no `trigger_semantics` under "
        "gpt-5.4-2026-03-05: the reader answers `observed_assets` and is "
        "silent on the dimension the case asserts, and the deterministic "
        "TriggerSemanticsReader declines too — `under trend` names no crossing "
        "and no persistence in the structure it reads. The outcome is safe. "
        "Nothing is settled, so the pipeline asks rather than guessing between "
        "`crossing_event` and `persistent_condition`, which are different "
        "strategies and would fire on different days. What is lost is a case "
        "that used to be answered, and the loss is a precision one on a "
        "phrasing with no explicit signal in it. Not repaired by editing the "
        "expectation: the sentence does mean the persistent reading to a "
        "person, and a build that cannot get there should say so.",
    "sema-window-moving_average-013":
        "the unit gap again, and the third reader to land on a different side "
        "of it. Syntax reads `12` from `the 12-month moving average`; gpt-5.4 "
        "returns `12-month`, which the normaliser reads as a duration of 360 "
        "days. Both are defensible readings of a field that never says what it "
        "counts, so fusion refuses to settle it — the safe outcome, and why "
        "this is not on the dangerous list. Deliberately not fixed in the "
        "comparison layer: making `12-month` equal `12` there would be "
        "choosing the unit in the wrong place and on no authority. It is a "
        "schema change, and it is queued in docs/Benchmark-Queue.md.",
}


class TestTheCorpusCannotSelectItselfDown:
    """The set of cases this file runs is derived, not declared. Without this,
    the cheapest way to make the semantics tier pass is for a case to stop
    agreeing."""

    RECORDED = json.loads(
        (Path(__file__).resolve().parent.parent / "corpus" / "parser"
         / "answerable.json").read_text())

    def test_every_case_once_answered_is_answered_or_accounted_for(self):
        gone = set(self.RECORDED["ids"]) - set(ANSWERABLE)
        unexplained = sorted(gone - set(LEFT_THE_ANSWERABLE_SET))
        assert not unexplained, (
            f"{unexplained} used to be answered correctly and are not in the "
            "collected set any more. They did not fail — they stopped being "
            "collected, which is why nothing else caught this. Either fix the "
            "regression or name it in LEFT_THE_ANSWERABLE_SET")

    def test_the_exception_list_does_not_outlive_its_reasons(self):
        """The same staleness rule the model-error list carries. An entry that
        starts passing again and stays listed is a case excluded from the suite
        for a reason that no longer holds."""
        stale = sorted(set(LEFT_THE_ANSWERABLE_SET) & set(ANSWERABLE))
        assert not stale, (
            f"{stale} are listed as having left the answerable set and are "
            "answered again; remove them so the list keeps meaning something")

    def test_every_recorded_id_is_a_real_case(self):
        """`answerable.json` can be edited, and the cheapest way to bury a
        regression is to delete its id from the recorded set. A test cannot
        stop that — the file is committed and a reviewer has to look. What a
        test *can* stop is the quieter version: ids drifting out of sync with
        the corpus until the guard compares against phantoms and passes
        because both sides are empty of anything real.
        """
        known = {case.id for case in load()}
        phantom = sorted(set(self.RECORDED["ids"]) - known)
        assert not phantom, (
            f"{phantom} are recorded as answerable and name no case in the "
            "corpus")
        assert self.RECORDED["count"] == len(self.RECORDED["ids"]) > 0


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_the_pipeline_produces_the_field_and_the_value(case, request):
    from src.discovery.adapter import same_value_for
    from src.discovery.vocabulary import REQUIREMENTS, Requirement

    if case.id in DRIFTED:
        request.node.add_marker(pytest.mark.xfail(
            strict=False,
            reason="expected value moved with a hosted re-recording; see "
                   "DRIFTED in this file"))

    _, decision = run(case)
    assert decision.outcome is Fusion.AGREE, decision.detail
    rule = REQUIREMENTS.get(decision.dimension, Requirement()).compare_as
    # The dimension is passed, not just the rule. `12m` is twelve million
    # for an amount and twelve periods for a window, and a comparison that
    # does not know which is being asked reports one of them wrongly.
    assert same_value_for(decision.dimension, decision.value,
                          case.asserts["value"]), (
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


class TestIntermediateSemanticsAreVerifiedAtTheMapperBoundary:
    """The six cases the closure report classifies `INTERMEDIATE_SEMANTIC`.

    They were worse than pending: excluded from the queue *and* untested, which
    is the shape of a thing that quietly rots. This is their lifecycle —
    normalise, bind, propose, assert the candidate — and it stops at
    `propose()`.

    No fusion and no contract-field assertion, by definition: these are
    semantics this pipeline computes and the contract does not carry, so there
    is no second witness to agree with and no field for a decision to be about.
    Running them through fusion would manufacture a `DISAGREE` against a
    silence that could never have been anything else.
    """

    #: Only the ones the mapper actually produces. The four it does not are
    #: `INTERMEDIATE_NOT_PRODUCED` and stay in the pending queue — being
    #: outside the contract is not a reason for a case to go unverified.
    CASES = [c for c in load()
             if c.id in {row["id"] for row in CLOSURE["rows"]
                         if row["state"] == "INTERMEDIATE_SEMANTIC"}]

    def test_the_report_and_this_file_agree_on_which_they_are(self):
        """If the report excluded more from the queue than this file runs,
        cases would again be out of the queue and verified by nothing — which
        is exactly what this test surface found on its first run."""
        classified = {row["id"] for row in CLOSURE["rows"]
                      if row["state"] == "INTERMEDIATE_SEMANTIC"}
        assert {c.id for c in self.CASES} == classified
        assert classified

    @pytest.mark.parametrize(
        "case", CASES, ids=lambda c: c.id)
    def test_the_mapper_proposes_the_intermediate_candidate(self, case):
        values = normalize(case.text, case.language)
        candidates = propose(bind(RECORDED.parse(case.text, case.language),
                                  values), values)
        wanted = case.asserts["field"]
        match = next((c for c in candidates if c.field == wanted), None)
        assert match is not None, (
            f"no {wanted!r} candidate for {case.text!r}; the report classifies "
            "this case as intermediate semantics, so something should compute "
            "it")
        assert str(match.value) == str(case.asserts["value"]), (
            f"{case.text!r} -> {match.value!r}, expected "
            f"{case.asserts['value']!r}. {case.note}")

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
    def test_it_is_marked_as_outside_the_contract(self, case):
        """The property that keeps this tier honest. If one of these ever
        becomes a contract field, it should leave this file rather than be
        asserted in two places."""
        values = normalize(case.text, case.language)
        candidates = propose(bind(RECORDED.parse(case.text, case.language),
                                  values), values)
        match = next(c for c in candidates
                     if c.field == case.asserts["field"])
        assert not match.is_contract_field
