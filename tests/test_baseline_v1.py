"""The frozen baseline, checked against the tree it claims to describe.

A baseline recording what somebody believed at the time is not a baseline. Each
version below is read out of the code, so the before/after point the pilot
starts from stays a fact rather than a memory.

When a pilot observation legitimately moves one of these, the expected change
is to update this file *and* the document together — which is the point: the
move becomes visible instead of silent.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent / "docs" / "Baseline-v1.md"


def _recorded() -> str:
    if not DOC.exists():
        pytest.skip("docs/Baseline-v1.md is absent")
    return DOC.read_text()


class TestTheRecordedVersionsAreTheRealOnes:
    def test_the_discovery_schema_fingerprint(self):
        import sys

        sys.path.insert(0, str(DOC.parent.parent / "corpus"))
        from shadow_run import schema_fingerprint

        from src.discovery import QUANTIFY_SCHEMA

        assert schema_fingerprint(QUANTIFY_SCHEMA) in _recorded()

    def test_the_schema_version_is_the_one_in_the_tree(self):
        """Read out, not typed. A baseline naming `@3` would name an artifact
        that has not existed since."""
        from src.discovery import QUANTIFY_SCHEMA

        assert QUANTIFY_SCHEMA.version in _recorded()

    def test_the_version_it_was_frozen_at_is_still_recorded(self):
        """The baseline moved: `@5` was frozen, `@6` is current, and the
        benchmark is why. Renumbering the line in place would have destroyed
        the before/after point this document exists to provide, so the old
        fingerprint stays and the movement is narrated.

        This assertion is the thing that makes that non-optional. Without it,
        the cheapest way to make the fingerprint test pass on the *next* schema
        change is to overwrite one line, and the history goes with it.
        """
        text = _recorded()
        assert "ca8f3b7785ff5d70" in text, (
            "the fingerprint this baseline was frozen at is gone; the schema "
            "line was rewritten instead of the movement being recorded")
        assert "moved after this baseline was frozen" in text

    @pytest.mark.parametrize("module,name", [
        ("src.mission.capability", "MANIFEST_SCHEMA"),
        ("src.discovery.hosted_recording", "PROMPT_VERSION"),
        ("src.discovery.pipeline", "PIPELINE_VERSION"),
        ("src.evaluation.runner", "DRAWDOWN_SEMANTICS"),
        ("src.mission.mwr_contract", "CONTRACT_VERSION"),
    ])
    def test_each_declared_version(self, module, name):
        import importlib

        value = getattr(importlib.import_module(module), name)
        assert value in _recorded(), f"{name} is {value!r}, not what is recorded"

    def test_the_theorem_count(self):
        """Counted, not estimated. A baseline claiming more proof than exists
        is the failure `FormalCore.md` was written against."""
        formal = DOC.parent.parent / "formal" / "Quantify"
        theorems = sum(len(re.findall(r"^theorem ", p.read_text(), re.M))
                       for p in formal.rglob("*.lean"))
        found = re.search(r"(\d+) theorems", _recorded())
        assert found and int(found.group(1)) == theorems, (
            f"{theorems} theorems in the tree, {found and found.group(1)} "
            "recorded")


class TestTheOpenItemsAreStillOpen:
    """A baseline that listed a resolved problem would misdescribe the
    starting point in the more flattering direction."""

    def test_mission_still_computes_no_drawdown_or_volatility(self):
        import inspect

        from src.mission.simulate import MissionResult

        emitted = inspect.getsource(MissionResult.to_json)
        assert "drawdown" not in emitted
        assert "volatil" not in emitted

    def test_the_gate_is_still_closed_on_provenance(self):
        """Not on a semantic condition. If this starts failing because a
        semantic condition broke, the baseline is no longer describing the
        system it was frozen from."""
        from src.mission.prelean_gate import verdict

        gate = verdict()
        if gate.open:
            pytest.skip("the gate has been opened by a real CI run")
        # `and this build is` is the staleness blocker, and it fired for real:
        # the drift artifact was recorded against schema `@5`, the benchmark
        # moved the schema to `@6`, and the gate refused to count evidence
        # gathered about a build that no longer exists. That is the artifact
        # being self-describing rather than a semantic condition breaking, so
        # it belongs in this list beside provenance and age.
        # `not declared` joined the list when the gate stopped guessing which
        # reader to expect. It is an operational blocker like the others — the
        # deployment has not said which provider it serves — rather than a
        # semantic condition failing.
        # `closure report was produced by` joined when the gate began checking
        # that its second evidence file came from the same reader as the
        # first. Operational like the rest: a report needs regenerating, not a
        # semantic condition failing.
        allowed = ("not the scheduled lane", "days old", "and this build is",
                   "not declared", "closure report was produced by")
        assert all(any(a in b for a in allowed) for b in gate.blockers), \
            gate.blockers


class TestTheReopenTriggersAreWrittenDown:
    def test_all_four_are_recorded(self):
        """Written down rather than remembered, so "should we build this?" has
        an answer that does not depend on who is asked."""
        text = _recorded().lower()
        for trigger in ("wrong executable meaning", "silent reduction",
                        "journey blocked", "roadmap trigger"):
            assert trigger in text, trigger
