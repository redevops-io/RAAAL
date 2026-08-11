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

    def test_a_rotation_strategy_no_longer_executes_as_something_else(
            self, findings):
        """The benchmark's first finding, closed, and kept as a regression.

        "hold whichever performed best" used to read as two holdings and a
        monthly cadence and *execute* — the selection silently gone, the person
        shown a buy-and-hold backtest and told it was their rotation strategy.

        It was not closed by teaching the syntax guard the words `rotate`,
        `stronger` and `whichever`. Those are witnesses of a missing semantic,
        not the semantic itself, and a guard built from them would pass this
        test while the next synonym reduced silently. It was closed by giving
        Discovery a `selection_rule` dimension to represent the concept and
        Mission a `NOT_MODELLED` entry to refuse it by name.
        """
        rotation = [f for f in findings["findings"]
                    if f["class"] == "momentum-rotation"
                    and f["kind"] == "SILENT_REDUCTION"]
        assert rotation == [], (
            f"rotation is silently reducing again: {rotation}")

    def test_and_it_is_refused_by_name_rather_than_merely_not_executing(
            self, findings, suite):
        """The half that makes the fix a fix. A sentence that stopped executing
        because some unrelated guard tripped would satisfy the test above and
        still leave the person with no idea which part of their strategy this
        build cannot do."""
        entry = next(e for e in suite["classes"] if e["id"] == "momentum-rotation")
        for phrasing in entry["phrasings"]:
            point = findings["checkpoints"][phrasing]
            assert not point["executable"], f"{phrasing!r} still executes"
            assert "selection_rule" in point["refusals"], (
                f"{phrasing!r} does not execute, but names "
                f"{point['refusals']} rather than the selection it dropped")

    def test_holding_order_no_longer_changes_the_compiled_plan(self, findings):
        """`VTI and BND` against `BND and VTI` for an equal-weight strategy.
        The economics cannot depend on the order they were named in.

        The diagnosis was worse than the symptom. `_assets` split on commas
        only, so "VTI and BND" was *one* holding with a name no market has,
        weighted at 100% — the two prompts were not two orderings of a
        portfolio, they were two different single-instrument portfolios.
        Discovery's fusion had already agreed the sentence named two assets;
        Mission split them with a rule that had drifted from Discovery's.
        """
        assert not any(f["class"] == "holding order, equal weight"
                       for f in findings["findings"])

    def test_thousands_shorthand_no_longer_executes_a_different_plan(
            self, findings):
        """`$1k` compiled with `amount = 0`: every other field correct, and a
        plan that invested nothing. It is now refused by name, which is a
        recognition gap rather than a danger — see the taxonomy tests."""
        from corpus.benchmark.run import UNSTABLE_EXECUTION

        assert not any(f["class"] == "thousands shorthand"
                       and f["kind"] == UNSTABLE_EXECUTION
                       for f in findings["findings"])


class TestTheTaxonomyCannotBeUsedToHideThings:
    """`UNSTABLE_SAFE` was introduced in the same change that took dangerous
    instances to zero. That is exactly the circumstance in which a new category
    deserves checking, so these are the checks."""

    def test_a_downgraded_finding_is_still_reported(self, findings):
        """Reclassified, not deleted. A category that removes a finding from
        the headline *and* from the queue is a category for making numbers
        look better."""
        from corpus.benchmark.run import UNSTABLE_SAFE

        safe = [q for q in findings["queue"] if q["kind"] == UNSTABLE_SAFE]
        assert safe, ("`$1k` against `$1,000` is a real recognition gap and is "
                      "not in the queue")
        assert sum(q["instances"] for q in safe) >= 1

    def test_the_downgrade_requires_that_nothing_executed_wrongly(self, findings):
        """The condition, asserted rather than trusted: every `UNSTABLE_SAFE`
        pair must have at most one side that compiled a plan. Two executable
        plans for one meaning is the dangerous shape and must stay dangerous."""
        from corpus.benchmark.run import UNSTABLE_SAFE

        suite = json.loads(SUITE.read_text())
        by_name = {r["name"]: r for r in suite["metamorphic"]}
        for entry in findings["queue"]:
            if entry["kind"] != UNSTABLE_SAFE:
                continue
            relation = by_name[entry["area"]]
            executed = [side for side in ("from", "to")
                        if findings["checkpoints"][relation[side]].get("plan")]
            assert len(executed) <= 1, (
                f"{entry['area']} is classified UNSTABLE_SAFE and both sides "
                f"compiled a plan: {executed}")

    def test_unstable_safe_is_not_counted_as_dangerous(self, findings):
        from corpus.benchmark.run import DANGEROUS, UNSTABLE_SAFE

        assert UNSTABLE_SAFE not in DANGEROUS
        assert findings["dangerous_instances"] == sum(
            q["instances"] for q in findings["queue"]
            if q["kind"] in DANGEROUS)
