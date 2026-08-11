"""The strategy benchmark, and the properties that keep it honest.

It is a counterexample generator, not a fifth reopen trigger. What it finds
activates the existing four: a wrong executable meaning or a silent reduction
is work on the day it appears; an unsupported strategy appearing forty times is
the counted demand the fourth trigger asks for.

Most of this file is about the classifier rather than the corpus, because a
benchmark that manufactures findings is worse than none — it spends attention
on defects that are not there and earns distrust of the ones that are. Two such
defects were found while building it and are pinned below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "corpus" / "benchmark"
SUITE = BENCH / "suite.json"
FINDINGS = BENCH / "findings.json"


@pytest.fixture(scope="module")
def suite():
    if not SUITE.exists():
        pytest.skip("suite.json is absent")
    return json.loads(SUITE.read_text())


@pytest.fixture(scope="module")
def findings():
    if not FINDINGS.exists():
        pytest.skip("findings.json is absent; run corpus/benchmark/run.py")
    return json.loads(FINDINGS.read_text())


class TestTheCorpusIsWellFormed:
    def test_every_class_declares_what_it_expects(self, suite):
        for entry in suite["classes"]:
            assert entry["disposition"] in ("EXECUTES", "REFUSES", "CLARIFIES")
            assert len(entry["phrasings"]) >= 3, (
                f"{entry['id']} has too few phrasings to be an equivalence "
                "class; one phrasing cannot disagree with itself")

    def test_a_refusing_class_names_what_should_be_refused(self, suite):
        """Otherwise "it refused" passes for a refusal of the wrong thing."""
        for entry in suite["classes"]:
            if entry["disposition"] == "REFUSES":
                assert entry["refuses"], entry["id"]

    def test_every_prompt_has_a_recording(self, suite):
        """A prompt with no recorded reading would silently drop out of the
        run rather than be measured."""
        hosted = json.loads(
            (BENCH.parent / "parser" / "hosted.json").read_text())
        recorded = {r["text"] for r in hosted["readings"]}
        missing = sorted(p for p in suite["prompts"] if p not in recorded)
        assert not missing, f"{len(missing)} prompts unrecorded: {missing[:3]}"

    def test_the_corpus_declares_that_it_is_authored(self, suite):
        assert "Authored" in suite["provenance_note"]


class TestTheScoringCannotDegenerate:
    def test_there_is_no_pass_rate(self, findings):
        """A score that counted executions would reward the silent reduction
        this project spent months removing."""
        assert "pass_rate" not in findings
        assert "score" not in findings
        assert "deliberately" in findings["scoring_note"]

    def test_correct_refusals_are_counted_as_correct(self, findings):
        assert findings["correct"]["CORRECT_REFUSAL"] > 0

    def test_the_queue_is_ranked_by_danger_then_recurrence(self, findings):
        from corpus.benchmark.run import DANGEROUS

        rows = findings["queue"]
        keys = [(row["kind"] not in DANGEROUS, -row["instances"])
                for row in rows]
        assert keys == sorted(keys), "the queue is not ranked as it claims"

    def test_every_finding_names_a_layer(self, findings):
        """"two equivalent prompts disagreed" is useless until you know which
        stage disagreed."""
        for finding in findings["findings"]:
            assert finding["layer"] in ("Discovery", "Fusion", "Mission",
                                        "Surface")


class TestTheClassifierDoesNotManufactureFindings:
    """Two defects found while building this, pinned so they cannot return."""

    def test_a_question_is_not_counted_as_a_capability_refusal(self, findings):
        """`UNRESOLVED_INPUT` is Mission saying "the intent names nothing to
        hold" — a question wearing a refusal's shape. Counting it as a
        capability refusal reported a reader asking something reasonable as a
        reader refusing a supported strategy."""
        sys.path.insert(0, str(BENCH.parent.parent))
        from corpus.benchmark.run import _disposition

        asked = {"refusals": [], "needs_input": ["assets"], "questions":
                 ["assets"], "executable": False}
        assert _disposition(asked) == "CLARIFIES"

        refused = {"refusals": ["sell_action"], "needs_input": [],
                   "questions": [], "executable": False}
        assert _disposition(refused) == "REFUSES"

    def test_two_refused_prompts_can_still_differ(self, findings):
        """`60/40` and `40/60` are both refused, so neither has a compiled
        plan. Comparing only the plan made "the plan did not change" trivially
        true and reported a conflation the reader had not made — it tells them
        apart in `stated_weights`."""
        from corpus.benchmark.run import _semantic_identity

        left = {"refusals": ["stated_weights"], "executable": False,
                "plan": None, "settled": {"stated_weights": "60/40"},
                "needs_input": [], "questions": []}
        right = {**left, "settled": {"stated_weights": "40/60"}}
        assert _semantic_identity(left) != _semantic_identity(right)

    def test_and_surface_form_does_not_count_as_meaning(self, findings):
        """The other direction: a `SAME` relation compares what executes, so
        `$1,000` and `$1k` differing as strings is not itself the finding —
        their compiling to different plans is."""
        from corpus.benchmark.run import _executable_identity

        a = {"refusals": [], "executable": True, "plan": "abc",
             "settled": {"amount": "$1,000"}, "needs_input": [], "questions": []}
        b = {**a, "settled": {"amount": "$1k"}}
        assert _executable_identity(a) == _executable_identity(b)


class TestTheFindingsAreRealFindings:
    """Spot-checks on the ones this run surfaced, so a later change that fixes
    or breaks them shows up here rather than in a count."""

    def test_a_rotation_strategy_executes_as_something_else(self, findings):
        """"hold whichever performed best" reads as two holdings and a monthly
        cadence, and executes — the selection is gone. The presence guards did
        not fire because no disposal verb appears in the sentence."""
        rotation = [f for f in findings["findings"]
                    if f["class"] == "momentum-rotation"
                    and f["kind"] == "SILENT_REDUCTION"]
        assert len(rotation) >= 3

    def test_holding_order_changes_the_compiled_plan(self, findings):
        """`VTI and BND` against `BND and VTI` for an equal-weight strategy.
        The economics cannot depend on the order they were named in."""
        assert any(f["class"] == "holding order, equal weight"
                   for f in findings["findings"])

    def test_thousands_shorthand_changes_the_compiled_plan(self, findings):
        assert any(f["class"] == "thousands shorthand"
                   for f in findings["findings"])
