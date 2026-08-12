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
import os
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(autouse=True)
def _a_declared_deployment(monkeypatch):
    """Every test here asks "given this deployment, is this artifact
    admissible", which presumes a deployment that has said what it is.

    Without this they inherited whatever the shell had, and the gate's own
    undeclared-provider blocker fired in all of them — a real signal about the
    environment, arriving as noise in tests about the evidence. The undeclared
    case gets its own test below rather than contaminating these.
    """
    from src.deploy import context

    monkeypatch.setenv("QUANTIFY_PARSER_PROVIDER", "OPENAI")
    resolved = context.resolve(dict(os.environ))
    monkeypatch.setattr(context, "current", lambda: resolved)


@pytest.fixture
def versions():
    from src.mission.prelean_gate import _current_versions

    return _current_versions()


def _drift(tmp_path, versions, *, unsafe=(), draws=3, age_days=0,
           overrides=None):
    at = datetime.now(timezone.utc) - timedelta(days=age_days)
    provenance = {**versions, "recorded_at": at.isoformat(),
                  "draws_per_prompt": draws,
                  "producer": "github-actions", "mode": "schedule"}
    provenance.update(overrides or {})
    path = tmp_path / "drift.json"
    path.write_text(json.dumps({
        "provenance": provenance,
        "by_classification": {"STABLE_REFUSAL": 30,
                              "UNSTABLE_EXECUTABLE": len(unsafe)},
        "execution_unsafe": list(unsafe)}))
    return path


def _closure(tmp_path, *, reduced=0, witness=None):
    """The witness defaults to the reader this deployment serves with.

    It was the literal `claude-sonnet-5@1`, which made every test here assert
    that a gate fed a *Claude* closure report opens — and it did, because the
    gate printed that field without checking it. Once the gate started
    comparing, these fixtures were the first thing to fail, correctly: they
    described an experiment run by two different readers.
    """
    from src.mission.prelean_gate import _current_versions

    path = tmp_path / "closure.json"
    path.write_text(json.dumps({
        "witness": witness or _current_versions()["hosted_model_id"],
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
                           "draws_per_prompt": 3,
                           "producer": "github-actions", "mode": "schedule"},
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


class TestTheWatchSet:
    """Six prompts whose draws disagree and where nothing ever executes.

    They do not block. They are named because the transition that would matter
    is invisible in a total: UNSTABLE_SAFE 6 -> 5 and UNSTABLE_EXECUTABLE 0 -> 1
    reads as noise unless something says which six were being watched.
    """

    def _drift_with(self, tmp_path, versions, results):
        import json as _json
        from datetime import datetime, timezone

        path = tmp_path / "drift.json"
        path.write_text(_json.dumps({
            "provenance": {**versions,
                           "recorded_at": datetime.now(timezone.utc).isoformat(),
                           "draws_per_prompt": 3,
                           "producer": "github-actions", "mode": "schedule"},
            "by_classification": {},
            "execution_unsafe": [r["text"] for r in results
                                 if r["classification"] == "UNSTABLE_EXECUTABLE"],
            "silently_reduced_any_draw": [],
            "results": results}))
        return path

    def test_a_watched_prompt_staying_safe_does_not_block(self, tmp_path,
                                                          versions):
        from src.mission.prelean_gate import WATCHED, verdict

        drift = self._drift_with(tmp_path, versions, [
            {"text": WATCHED[0], "classification": "UNSTABLE_SAFE"}])
        gate = verdict(drift_path=drift, closure_path=_closure(tmp_path))
        assert gate.open, gate.blockers

    def test_but_crossing_into_executable_blocks_and_names_it(self, tmp_path,
                                                              versions):
        from src.mission.prelean_gate import WATCHED, verdict

        drift = self._drift_with(tmp_path, versions, [
            {"text": WATCHED[0], "classification": "UNSTABLE_EXECUTABLE"}])
        gate = verdict(drift_path=drift, closure_path=_closure(tmp_path))
        assert not gate.open
        assert any("unstable-but-safe" in b for b in gate.blockers)
        assert any(WATCHED[0][:30] in b for b in gate.blockers)

    def test_the_watch_set_is_not_empty(self):
        """An empty watch set would pass every check above while watching
        nothing, which is how a guard quietly stops guarding."""
        from src.mission.prelean_gate import WATCHED

        assert len(WATCHED) >= 6

    def test_every_watched_prompt_is_in_the_corpus(self):
        """A watched sentence nobody runs is never classified, so it can never
        cross anything."""
        import json as _json
        from pathlib import Path

        from src.mission.prelean_gate import CORPUS, WATCHED

        cases = _json.loads(
            Path(CORPUS / "strategy_families.json").read_text())["cases"]
        texts = {c["text"] for c in cases}
        missing = [t for t in WATCHED if t not in texts]
        assert not missing, f"{missing} are watched and not in the corpus"


class TestTheSpendCeilingIsReal:
    """A ceiling nothing reads is not a ceiling.

    The drift workflow pinned `QUANTIFY_PARSER_MAX_TOKENS` as a spend control
    while `HostedReader.max_tokens` kept its hardcoded default, so the budget
    was a comment in YAML — the exact failure that workflow's own header warns
    about, committed in the workflow itself.
    """

    def test_the_ceiling_is_resolved_from_the_environment(self):
        from src.deploy.context import PARSER_MAX_TOKENS_VAR, resolve

        assert resolve({PARSER_MAX_TOKENS_VAR: "2048"}).model.max_tokens == 2048

    @pytest.mark.parametrize("raw", ["", "nonsense", "0", "-5"])
    def test_an_unusable_value_keeps_the_default(self, raw):
        """A malformed budget must not stop a deployment serving, and a zero
        ceiling would refuse every request while looking like a setting."""
        from src.deploy.context import PARSER_MAX_TOKENS_VAR, resolve

        assert resolve({PARSER_MAX_TOKENS_VAR: raw}).model.max_tokens == 8000

    def test_the_reader_is_built_with_it(self, monkeypatch):
        """The half that was missing. Resolving the value and never passing it
        on would satisfy the test above and change nothing about the bill."""
        from src.deploy import context as deploy_context

        resolved = deploy_context.resolve({
            "QUANTIFY_PARSER_MODE": "RUNTIME",
            "QUANTIFY_PILOT_READER": "HOSTED",
            "QUANTIFY_PARSER_MODEL": "claude-sonnet-5",
            "ANTHROPIC_API_KEY": "unused",
            "QUANTIFY_PARSER_MAX_TOKENS": "2048",
        })
        monkeypatch.setattr(deploy_context, "current", lambda: resolved)

        from src.workspace.pilot_routes import configured_reader

        assert configured_reader().max_tokens == 2048

    def test_the_workflow_pins_a_variable_the_code_reads(self):
        """Checked structurally, because the defect was a name in YAML that
        matched nothing in Python."""
        import re
        from pathlib import Path

        from src.deploy import context

        workflow = (Path(__file__).resolve().parent.parent / ".github"
                    / "workflows" / "drift-lane.yml").read_text()
        pinned = set(re.findall(r"^\s{2}(QUANTIFY_\w+):", workflow, re.M))
        known = {getattr(context, name) for name in dir(context)
                 if name.endswith("_VAR")}
        known |= {"QUANTIFY_PARSER_MODEL", "QUANTIFY_DATABASE_URL"}
        unread = pinned - known
        assert not unread, f"{sorted(unread)} are pinned and read by nothing"


class TestLocalEvidenceIsNotACIGuarantee:
    """A laptop run and the scheduled lane are different claims.

    Both are useful and only one is a guarantee. Without this, a seven-day-old
    local artifact keeps the gate open while CI has never successfully spoken
    to the provider — and the gate's whole purpose is to say the *deployment*
    is safe, not that somebody's checkout was.
    """

    def _local(self, tmp_path, versions):
        at = datetime.now(timezone.utc)
        path = tmp_path / "drift.json"
        path.write_text(json.dumps({
            "provenance": {**versions, "recorded_at": at.isoformat(),
                           "draws_per_prompt": 3, "producer": "local",
                           "mode": "local"},
            "by_classification": {"STABLE_REFUSAL": 36},
            "execution_unsafe": [], "silently_reduced_any_draw": []}))
        return path

    def test_a_local_artifact_does_not_open_the_gate(self, tmp_path, versions):
        from src.mission.prelean_gate import verdict

        gate = verdict(drift_path=self._local(tmp_path, versions),
                       closure_path=_closure(tmp_path))
        assert not gate.open
        assert any("not the scheduled lane" in b for b in gate.blockers)
        assert gate.evidence["producer"] == "local"

    def test_but_development_may_ask_without_it(self, tmp_path, versions):
        """`require_ci=False` is for a person checking their own work. It is a
        parameter rather than a default so that using it is a decision somebody
        made, and shows up in the call."""
        from src.mission.prelean_gate import verdict

        gate = verdict(drift_path=self._local(tmp_path, versions),
                       closure_path=_closure(tmp_path), require_ci=False)
        assert gate.open, gate.blockers

    def test_an_artifact_predating_the_field_is_not_treated_as_ci(
            self, tmp_path, versions):
        """Absent provenance reads as unknown, not as trusted. The artifact on
        disk when this check was added had no `producer` at all, and defaulting
        it either way would have decided the question by omission."""
        from src.mission.prelean_gate import verdict

        at = datetime.now(timezone.utc)
        path = tmp_path / "drift.json"
        path.write_text(json.dumps({
            "provenance": {**versions, "recorded_at": at.isoformat(),
                           "draws_per_prompt": 3},
            "by_classification": {}, "execution_unsafe": [],
            "silently_reduced_any_draw": []}))
        gate = verdict(drift_path=path, closure_path=_closure(tmp_path))
        assert not gate.open
        assert gate.evidence["producer"] == "unknown"

    def test_the_lane_records_who_produced_it(self, monkeypatch):
        """The other half. A gate that demanded provenance the producer never
        wrote would be permanently closed for a reason no run could fix."""
        import sys
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).resolve().parent.parent
                               / "corpus" / "parser"))
        import drift_lane

        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_WORKFLOW", "Discovery drift lane")
        monkeypatch.setenv("GITHUB_RUN_ID", "12345")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")

        class _Reader:
            id = "claude-sonnet-5@1"
            model = "claude-sonnet-5"

        provenance = drift_lane._provenance(_Reader(), _Reader(), ["a"], 3)
        assert provenance["producer"] == "github-actions"
        assert provenance["workflow"] == "Discovery drift lane"
        assert provenance["run_id"] == "12345"
        assert provenance["mode"] == "workflow_dispatch"

    def test_and_says_local_when_it_is(self, monkeypatch):
        import sys
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).resolve().parent.parent
                               / "corpus" / "parser"))
        import drift_lane

        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

        class _Reader:
            id = "claude-sonnet-5@1"
            model = "claude-sonnet-5"

        assert drift_lane._provenance(_Reader(), _Reader(), ["a"], 3)[
            "producer"] == "local"



class TestTheGateKnowsWhichReaderItIsCheckingFor:
    """The pin on `hosted_model_id` closed "the gate does not check the
    reader". This closes the one underneath it: "the gate decides for itself
    which reader to expect"."""

    def test_an_undeclared_provider_blocks(self, tmp_path, versions,
                                           monkeypatch):
        """Found while capturing the gate's state before a real dispatch. The
        same artifact passed the reader check locally and would have failed it
        in CI, differing only by an environment variable nobody had set — a
        verdict about the checker's shell wearing the clothes of a verdict
        about the evidence.
        """
        from src.deploy import context
        from src.mission.prelean_gate import verdict

        monkeypatch.delenv("QUANTIFY_PARSER_PROVIDER", raising=False)
        resolved = context.resolve({k: v for k, v in os.environ.items()
                                    if k != "QUANTIFY_PARSER_PROVIDER"})
        monkeypatch.setattr(context, "current", lambda: resolved)

        path = _drift(tmp_path, versions)
        out = verdict(drift_path=path, require_ci=True)
        assert not out.open
        assert any("not declared" in b for b in out.blockers), out.blockers

    def test_and_a_declared_one_does_not(self, tmp_path, versions):
        from src.mission.prelean_gate import verdict

        path = _drift(tmp_path, versions)
        out = verdict(drift_path=path, require_ci=True)
        assert not any("not declared" in b for b in out.blockers), out.blockers



class TestTheClosureReportIsCheckedNotJustPrinted:
    """The gate blocks on `silently_reduced` from the closure report, which
    makes that report evidence — and it was being combined with a drift
    artifact from a different reader to reach one verdict."""

    def test_a_closure_report_from_another_reader_blocks(self, tmp_path,
                                                         versions):
        from src.mission.prelean_gate import verdict

        out = verdict(drift_path=_drift(tmp_path, versions),
                      closure_path=_closure(tmp_path,
                                            witness="some-other-reader@1"),
                      require_ci=True)
        assert not out.open
        assert any("closure report was produced by" in b
                   for b in out.blockers), out.blockers

    def test_and_one_from_the_serving_reader_does_not(self, tmp_path, versions):
        from src.mission.prelean_gate import verdict

        out = verdict(drift_path=_drift(tmp_path, versions),
                      closure_path=_closure(tmp_path), require_ci=True)
        assert not any("closure report was produced by" in b
                       for b in out.blockers), out.blockers
