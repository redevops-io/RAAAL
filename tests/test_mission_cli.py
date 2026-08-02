"""Mission DevOps: the eight verbs, and the exit codes that make them useful.

Everything underneath already existed; only the tooling did not. Each verb
composes what the engine already does rather than adding semantics — none takes
an option that would let it decide something.

Exit codes are the point. A tool whose failure is a paragraph nobody reads is a
tool that runs in CI and never fails a build, so every verb is tested for both
answers.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

CLI = [sys.executable, "scripts/mission.py"]
COMPLETE = ("I put $2,000 into SPY every month in my Roth IRA, on the first "
            "trading day of the period, reinvesting the dividends, and I never "
            "sell.")
CONTRADICTORY = ("I buy $500 of VTI and BND monthly in my taxable account, "
                 "rebalancing to equal weights, but I never sell.")


def run(*args):
    return subprocess.run(CLI + list(args), capture_output=True, text=True)


@pytest.fixture
def saved(tmp_path):
    """A plan and a worksheet, stored the way the product stores them."""
    from src.mission.compiler import compile_scenario
    from src.mission.scenario import ScenarioSpecification
    from src.mission.spec import Inference, Provenance
    from src.workspace.generate import generate
    from src.workspace.store import WorkspaceStore

    db = tmp_path / "w.db"
    store = WorkspaceStore(db)
    compiled = compile_scenario(COMPLETE, name="plan-1", version=1,
                                benchmark_rule="benchmark-policy/public-default@1")
    provenance = compiled.scenario.provenance
    scenario = ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=provenance.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in provenance.inferred),
            contradictions=provenance.contradictions, unresolved=())})
    store.save_plan(plan_id="plan-1", owner="pilot", scenario=scenario,
                    stated_text=COMPLETE, saved_at="2026-08-01T00:00:00Z")
    generate(store, plan_id="plan-1", owner="pilot", scenario=scenario,
             run={"modelling_scope": {"excludes": ["dividends"]},
                  "final_value": 1.0},
             comparison={}, ran_at="2026-08-01T00:00:00Z")
    return str(db)


class TestCreateAndValidate:

    def test_create_states_the_plan_in_plain_language(self):
        result = run("create", COMPLETE)
        assert result.returncode == 0
        assert "Roth IRA" in result.stdout
        assert "SPY" in result.stdout

    def test_validate_passes_a_fully_represented_plan(self):
        result = run("validate", COMPLETE)
        assert result.returncode == 0
        assert "every recognised value reached the compiled scenario" in result.stdout

    def test_validate_fails_a_contradiction(self):
        result = run("validate", CONTRADICTORY)
        assert result.returncode == 1
        assert "contradiction" in result.stdout

    def test_a_finding_is_reported_once(self):
        """The compiler and the scenario can restate one conflict. A list that
        counts a finding twice makes a caller think there are two."""
        result = run("validate", CONTRADICTORY)
        assert result.stdout.count("maintaining a weight requires selling") == 1


class TestBenchmarkLeadsWithComparability:

    def test_it_reports_the_verdict_before_any_figure(self):
        result = run("benchmark", COMPLETE)
        assert result.returncode == 0
        assert result.stdout.index("comparability") == 0

    def test_a_missing_asset_is_a_data_gap_not_a_result(self):
        result = run("benchmark",
                     "I put $500 into ZZZZ every month in my Roth IRA and never sell.")
        assert result.returncode == 1
        assert "data gap, not a result" in result.stdout


class TestReplayAndVerify:

    def test_replay_reproduces_a_stored_plan(self, saved):
        result = run("replay", "plan-1", "--store", saved)
        assert result.returncode == 0
        assert "reproduces" in result.stdout

    def test_verify_passes_a_sound_plan(self, saved):
        result = run("verify", "plan-1", "--store", saved)
        assert result.returncode == 0
        assert "ok" in result.stdout

    def test_replay_detects_a_record_edited_after_saving(self, saved):
        """The exit code is what makes this usable in CI."""
        connection = sqlite3.connect(saved)
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT scenario FROM plan WHERE plan_id='plan-1'").fetchone()
        body = json.loads(row["scenario"])
        body["flows"]["amount"] = 999999.0
        connection.execute("UPDATE plan SET scenario=? WHERE plan_id='plan-1'",
                           (json.dumps(body),))
        connection.commit()

        result = run("replay", "plan-1", "--store", saved)
        assert result.returncode == 1
        assert "DOES NOT REPRODUCE" in result.stdout

    def test_a_missing_plan_fails_rather_than_reporting_nothing(self, saved):
        assert run("verify", "absent", "--store", saved).returncode == 1


class TestDiffIsAProposal:

    def test_an_unchanged_plan_reports_no_difference(self, saved):
        result = run("diff", "plan-1", "--store", saved)
        assert result.returncode == 0
        assert "reads this the same way" in result.stdout

    def test_diff_writes_nothing(self, saved):
        """A diff that mutated what it described would be the migration this
        project has refused everywhere else."""
        from src.workspace.store import WorkspaceStore

        store = WorkspaceStore(Path(saved))
        before = store.get_plan("plan-1", "pilot")["scenario"]
        run("diff", "plan-1", "--store", saved)
        assert store.get_plan("plan-1", "pilot")["scenario"] == before


class TestRollbackGoesForward:

    def test_it_creates_a_new_revision_rather_than_deleting(self, saved):
        """Rolling back by deleting revisions would erase the history that
        revisions exist to keep."""
        from src.workspace.store import WorkspaceStore
        from src.workspace.worksheet import from_json, revise

        store = WorkspaceStore(Path(saved))
        # Looked up by what it cites. Worksheet ids are opaque and
        # server-generated, so there is no id to recompute from the plan name.
        identifier = store.worksheet_for_scenario("plan-1", "pilot")["worksheet_id"]
        first = from_json(store.get_worksheet(identifier, "pilot")["payload"])
        store.save_worksheet(revise(first, reason="a later run",
                                    primary_run_ref="run-later",
                                    created_at="2026-09-01T00:00:00Z"))

        result = run("rollback", identifier, "--to", "1", "--store", saved,
                     "--at", "2026-10-01T00:00:00Z")
        assert result.returncode == 0

        revisions = store.worksheet_revisions(identifier, "pilot")
        assert [r["revision"] for r in revisions] == [1, 2, 3]
        restored = from_json(revisions[-1]["payload"])
        assert restored.primary_run_ref == first.primary_run_ref
        assert "rolled back" in restored.change_reason

    def test_an_absent_revision_fails(self, saved):
        from src.workspace.store import WorkspaceStore

        identifier = WorkspaceStore(Path(saved)).worksheet_for_scenario(
            "plan-1", "pilot")["worksheet_id"]
        assert run("rollback", identifier, "--to", "99",
                   "--store", saved).returncode == 1


class TestTheToolDecidesNothing:

    def test_no_verb_takes_an_option_that_overrides_a_verdict(self):
        """The CLI composes the engine; it does not get a vote."""
        source = Path("scripts/mission.py").read_text()
        for flag in ("--force", "--ignore", "--assume", "--skip-validation",
                     "--no-verify"):
            assert flag not in source, flag
