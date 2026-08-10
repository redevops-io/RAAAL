"""The gate that decides whether formal verification may start.

    zero UNSTABLE_EXECUTABLE   in the live drift artifact
    zero SILENTLY_REDUCED      in the serving closure report

Neither condition is "Discovery is deterministic", and that is the point. Lean
proves deterministic operators obey their contract; it cannot prove the
contract describes what somebody asked for. So what must hold first is that an
unsupported or ambiguous intent never reaches the engine wearing an executable
shape — otherwise Lean proves the wrong strategy perfectly.

Everything here runs on fabricated artifacts. The gate reads files, so it can
be tested exhaustively without a provider, which is the whole reason it reads
files rather than measuring.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def versions():
    from src.mission.prelean_gate import _current_versions

    return _current_versions()


def _drift(tmp_path, versions, *, unsafe=(), draws=3, age_days=0,
           overrides=None):
    at = datetime.now(timezone.utc) - timedelta(days=age_days)
    provenance = {**versions, "recorded_at": at.isoformat(),
                  "draws_per_prompt": draws}
    provenance.update(overrides or {})
    path = tmp_path / "drift.json"
    path.write_text(json.dumps({
        "provenance": provenance,
        "by_classification": {"STABLE_REFUSAL": 30,
                              "UNSTABLE_EXECUTABLE": len(unsafe)},
        "execution_unsafe": list(unsafe)}))
    return path


def _closure(tmp_path, *, reduced=0):
    path = tmp_path / "closure.json"
    path.write_text(json.dumps({
        "witness": "claude-sonnet-5@1",
        "by_state": {"REFUSED": 30, "SILENTLY_REDUCED": reduced}}))
    return path


class TestTheGateOpensOnlyOnBothConditions:
    def test_clean_evidence_opens_it(self, tmp_path, versions):
        from src.mission.prelean_gate import verdict

        gate = verdict(drift_path=_drift(tmp_path, versions),
                       closure_path=_closure(tmp_path))
        assert gate.open, gate.blockers

    def test_execution_unsafe_instability_closes_it(self, tmp_path, versions):
        from src.mission.prelean_gate import verdict

        gate = verdict(
            drift_path=_drift(tmp_path, versions,
                              unsafe=["withdraw 4% of the portfolio each year"]),
            closure_path=_closure(tmp_path))
        assert not gate.open
        assert any("UNSTABLE_EXECUTABLE" in b for b in gate.blockers)

    def test_a_silent_reduction_closes_it(self, tmp_path, versions):
        from src.mission.prelean_gate import verdict

        gate = verdict(drift_path=_drift(tmp_path, versions),
                       closure_path=_closure(tmp_path, reduced=2))
        assert not gate.open
        assert any("collapse into an executable plan" in b
                   for b in gate.blockers)

    def test_both_are_reported_not_just_the_first(self, tmp_path, versions):
        """A gate that stopped at the first blocker would be reopened twice,
        each time revealing the next reason — the same rule `refusals_for`
        already follows for capabilities."""
        from src.mission.prelean_gate import verdict

        gate = verdict(drift_path=_drift(tmp_path, versions, unsafe=["x"]),
                       closure_path=_closure(tmp_path, reduced=1))
        assert len(gate.blockers) >= 2


class TestSafeInstabilityDoesNotBlock:
    def test_determinism_is_not_the_requirement(self, tmp_path, versions):
        """A reader that refuses on one draw and asks on the next is safe:
        nothing executes either way and the person is asked something.

        Requiring determinism would block Lean on behaviour that cannot put a
        wrong number in front of anybody, and would make the gate unreachable
        for a stochastic reader — which is the only kind there is.
        """
        from src.mission.prelean_gate import verdict

        path = tmp_path / "drift.json"
        path.write_text(json.dumps({
            "provenance": {**versions,
                           "recorded_at": datetime.now(timezone.utc).isoformat(),
                           "draws_per_prompt": 3},
            "by_classification": {"UNSTABLE_SAFE": 9, "STABLE_REFUSAL": 27},
            "execution_unsafe": []}))
        gate = verdict(drift_path=path, closure_path=_closure(tmp_path))
        assert gate.open, gate.blockers


class TestStaleEvidenceIsNotEvidence:
    def test_an_old_artifact_is_refused(self, tmp_path, versions):
        """"We ran the drift lane once" must not become a permanent guarantee.
        The provider can change under a fixed model id, so this evidence has a
        shelf life whether or not anything in the repository moved."""
        from src.mission.prelean_gate import VALID_FOR_DAYS, verdict

        gate = verdict(
            drift_path=_drift(tmp_path, versions, age_days=VALID_FOR_DAYS + 1),
            closure_path=_closure(tmp_path))
        assert not gate.open
        assert any("days old" in b for b in gate.blockers)

    def test_a_fresh_one_is_accepted(self, tmp_path, versions):
        from src.mission.prelean_gate import VALID_FOR_DAYS, verdict

        gate = verdict(
            drift_path=_drift(tmp_path, versions, age_days=VALID_FOR_DAYS - 1),
            closure_path=_closure(tmp_path))
        assert gate.open, gate.blockers

    @pytest.mark.parametrize("field", ["schema_fingerprint", "prompt_version",
                                       "pipeline_version"])
    def test_a_version_that_moved_invalidates_it(self, tmp_path, versions,
                                                 field):
        """Measured against a different schema, prompt or pipeline, the numbers
        describe a system that no longer exists."""
        from src.mission.prelean_gate import verdict

        gate = verdict(
            drift_path=_drift(tmp_path, versions,
                              overrides={field: "something-else"}),
            closure_path=_closure(tmp_path))
        assert not gate.open
        assert any(field in b for b in gate.blockers)

    def test_a_single_draw_cannot_prove_stability(self, tmp_path, versions):
        """The longitudinal lane runs one draw and writes its own file. Pointing
        the gate at it would turn a provider-drift check into a stability claim
        that one sample cannot support."""
        from src.mission.prelean_gate import verdict

        gate = verdict(drift_path=_drift(tmp_path, versions, draws=1),
                       closure_path=_closure(tmp_path))
        assert not gate.open
        assert any("at least 3" in b for b in gate.blockers)

    def test_a_missing_artifact_does_not_read_as_success(self, tmp_path):
        """Absence of evidence, distinguished from evidence of stability.

        The failure this guards is a gate that opens because nothing ran.
        """
        from src.mission.prelean_gate import verdict

        gate = verdict(drift_path=tmp_path / "nothing.json",
                       closure_path=_closure(tmp_path))
        assert not gate.open
        assert any("no drift artifact" in b for b in gate.blockers)


class TestTheClassifier:
    """Over the fused artifact, never over raw model output."""

    def test_identical_draws_are_stable(self):
        from corpus.parser.drift_lane import STABLE_REFUSAL, classify

        draws = [{"class": "REFUSAL", "identity": "refused:sell_action"}] * 3
        assert classify(draws) == STABLE_REFUSAL

    def test_refusal_versus_clarification_is_safe(self):
        from corpus.parser.drift_lane import UNSTABLE_SAFE, classify

        assert classify([
            {"class": "REFUSAL", "identity": "refused:sell_action"},
            {"class": "CLARIFICATION", "identity": "asks:assets"},
        ]) == UNSTABLE_SAFE

    def test_executable_versus_refusal_is_unsafe(self):
        from corpus.parser.drift_lane import UNSTABLE_EXECUTABLE, classify

        assert classify([
            {"class": "EXECUTABLE", "identity": "intent:abc"},
            {"class": "REFUSAL", "identity": "refused:sell_action"},
        ]) == UNSTABLE_EXECUTABLE

    def test_two_executable_draws_with_different_plans_are_unsafe(self):
        """The case an outcome-class comparison misses entirely, and the worst
        one: both draws "worked", and two people typing the same sentence get
        different strategies, each confidently."""
        from corpus.parser.drift_lane import UNSTABLE_EXECUTABLE, classify

        assert classify([
            {"class": "EXECUTABLE", "identity": "intent:aaa"},
            {"class": "EXECUTABLE", "identity": "intent:bbb"},
        ]) == UNSTABLE_EXECUTABLE

    def test_and_the_same_plan_twice_is_stable(self):
        from corpus.parser.drift_lane import STABLE_EXECUTABLE, classify

        assert classify([
            {"class": "EXECUTABLE", "identity": "intent:aaa"},
            {"class": "EXECUTABLE", "identity": "intent:aaa"},
        ]) == STABLE_EXECUTABLE

    def test_clarifications_asking_different_things_are_not_stable(self):
        """Same class, different questions. Calling it stable would report a
        reader that cannot decide what it needs as one that always agrees."""
        from corpus.parser.drift_lane import UNSTABLE_SAFE, classify

        assert classify([
            {"class": "CLARIFICATION", "identity": "asks:assets"},
            {"class": "CLARIFICATION", "identity": "asks:assets,cadence"},
        ]) == UNSTABLE_SAFE


class TestTheLaneStaysOutOfTheOrdinarySuite:
    def test_nothing_in_tests_imports_the_live_runner(self):
        """`run()` calls the provider by design. A test that imported it would
        put provider calls and a 500MB parser load into every commit — and the
        usual repair is to mock them, at which point the lane measures a mock.
        """
        import ast
        from pathlib import Path

        offenders = []
        for path in Path(__file__).resolve().parent.rglob("test_*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and \
                        (node.module or "").endswith("drift_lane"):
                    names = {a.name for a in node.names}
                    if "run" in names or "main" in names:
                        offenders.append(path.name)
        assert not offenders, f"{offenders} import the live drift runner"
