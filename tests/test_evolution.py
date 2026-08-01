"""EvolutionBench: replay, reinterpret and migrate kept apart.

Three operations that look alike and must never be confused:

    replay          run the Mission that was saved, as it was saved
    reinterpret     compile the original words again, under today's compiler
    migrate         adopt the new interpretation, deliberately and on the record

A plan saved last year and opened today is a replay. Recompiling its text under
a compiler that has since learned three more fields is a different Mission,
however faithfully it reads the same sentence.

Not hypothetical: `dividend_policy`, `moving_average_estimator` and
`funding_source` all became canonical after this project had already saved
plans, so every one of those plans has a stored body that lacks them.
"""
from __future__ import annotations

import pytest

from src.loadtest.evolutionbench import SPINE, build_history, replay, run, summarize
from src.mission.compiler import compile_scenario
from src.mission.evolution import (
    COMPILER_CHANGELOG,
    COMPILER_VERSION,
    as_compiled_by,
    diff_stored_against,
    propose_migration,
    rebuild_scenario,
)

BR = "benchmark-policy/public-default@1"


@pytest.fixture(scope="module")
def history():
    return build_history()


@pytest.fixture(scope="module")
def checkpoints():
    return run()


class TestTheBenchExercisesARealMigration:

    def test_stored_bodies_are_written_as_the_compiler_of_the_day_would(self, history):
        """Compiling both sides with today's compiler is how a version
        benchmark measures nothing while appearing to pass. It did, on the
        first run: seven checkpoints, zero differences.
        """
        first = history[0].stored
        assert "funding_source" not in (first["flows"] or {})
        assert "dividend_policy" not in (
            first["methodology"]["holdings_policy"] or {})

    def test_the_downgrade_is_driven_by_the_changelog(self):
        """So the reconstruction and the record cannot drift apart."""
        current = compile_scenario(
            "I put $500 into VTI monthly, holding the dividends as cash, and "
            "never sell.", name="s", version=1, benchmark_rule=BR)
        old = as_compiled_by(current.scenario.to_json(), "1")
        assert "dividend_policy" not in old["methodology"]["holdings_policy"]

        at_two = as_compiled_by(current.scenario.to_json(), "2")
        assert "dividend_policy" in at_two["methodology"]["holdings_policy"]
        assert "funding_source" not in at_two["flows"]

    def test_every_checkpoint_reinterprets_differently(self, checkpoints):
        """All seven revisions predate the three fields, so all seven differ."""
        assert summarize(checkpoints)["reinterpretations_differing"] == len(SPINE)


class TestReplayIsStableAcrossCompilerUpgrades:
    """The invariant. A historical replay hash must not move when the compiler
    changes; only a reinterpretation may."""

    def test_replay_derives_its_identity_from_the_stored_body(self, history):
        for revision in history:
            first, _us = replay(revision)
            second, _us = replay(revision)
            assert first == second

    def test_replay_does_not_consult_the_compiler(self, history):
        """Structural: `replay` takes a revision and nothing else. A function
        with nowhere to put a compiler provably cannot run one."""
        import inspect

        assert set(inspect.signature(replay).parameters) == {"revision"}

    def test_a_tampered_stored_body_is_detected(self, history):
        """Verified, not trusted — the same rule the evidence ledger follows."""
        import copy

        revision = copy.deepcopy(history[0])
        before, _ = replay(revision)
        revision.stored["flows"]["amount"] = 999999.0
        after, _ = replay(revision)
        assert before != after


class TestTheDiffIsTyped:

    def test_a_newly_represented_field_is_reported_as_added(self, checkpoints):
        added = [c for cp in checkpoints if cp.diff
                 for c in cp.diff.added]
        assert any(c.path.endswith("dividend_policy") for c in added)
        assert any(c.path.endswith("funding_source") for c in added)

    def test_rule_and_schedule_identity_are_reported_separately(self, checkpoints):
        """They have different consequences. A changed schedule invalidates
        flow-matched benchmark comparisons even when the rule is untouched."""
        first = checkpoints[0].diff
        assert first.rule_identity_changed
        assert first.schedule_identity_changed
        assert first.affects_comparability

    def test_the_explanation_names_what_the_compiler_learned(self, checkpoints):
        text = " ".join(checkpoints[0].diff.explain())
        assert "version 1" in text and f"version {COMPILER_VERSION}" in text
        assert "dividend_policy" in text

    def test_an_unchanged_plan_produces_no_diff(self):
        text = ("I put $500 into VTI monthly, holding the dividends as cash, "
                "and never sell.")
        current = compile_scenario(text, name="s", version=1,
                                   benchmark_rule=BR)
        diff = diff_stored_against(current.scenario.to_json(), current.scenario,
                                   stored_compiler=COMPILER_VERSION)
        assert diff.is_empty
        assert propose_migration("plan", diff) is None


class TestMigrationIsProposedNeverPerformed:

    def test_a_proposal_is_an_explanation_and_an_offer(self, checkpoints):
        from src.mission.evolution import MigrationProposal

        diff = checkpoints[0].diff
        proposal = propose_migration("plan", diff)
        assert isinstance(proposal, MigrationProposal)
        assert proposal.recommended is True
        assert proposal.required is False, (
            "a plan whose stored form still runs may keep running; the new "
            "interpretation is an option, not a debt")

    def test_nothing_in_the_module_writes(self):
        """Adopting a new interpretation changes what a saved plan means, and
        only its owner can agree to that."""
        import inspect

        from src.mission import evolution

        source = inspect.getsource(evolution)
        for verb in ("def save", "def write", "def update", "def apply_migration"):
            assert verb not in source


class TestAStoredPlanCanBeRebuilt:

    def test_a_body_round_trips_into_a_scenario(self):
        text = ("I put $900 into VTI and BND every month in my Roth IRA, on "
                "the first trading day of the period, holding the dividends as "
                "cash, and I never sell.")
        current = compile_scenario(text, name="plan", version=1,
                                   benchmark_rule=BR)
        rebuilt = rebuild_scenario(current.scenario.to_json())
        assert rebuilt is not None
        assert rebuilt.content_hash == current.scenario.content_hash

    def test_an_old_body_rebuilds_with_the_defaults_it_ran_under(self):
        """A plan saved before `funding_source` existed simulated as though the
        buy came from the contribution. The rebuild must match what it did, not
        what today's compiler would decide."""
        text = ("I put $500 into VTI monthly, reinvesting the dividends, and "
                "never sell.")
        current = compile_scenario(text, name="plan", version=1,
                                   benchmark_rule=BR)
        old = as_compiled_by(current.scenario.to_json(), "1")
        rebuilt = rebuild_scenario(old)
        assert rebuilt.flow_schedule.funding_source == "contribution"
        assert rebuilt.holdings_policy.dividend_policy == "reinvested"

    def test_an_unrebuildable_body_returns_none(self):
        """So a caller can say the replay is approximate rather than quietly
        serving a fresh interpretation under an old plan's name."""
        assert rebuild_scenario({"name": "x"}) is None


class TestThePlanPageReplays:

    def test_opening_a_saved_plan_runs_the_stored_scenario(self):
        """This route used to recompile the text and simulate the result while
        displaying the stored scenario, so after any compiler change the page
        showed one plan and the figures came from another."""
        import inspect

        from src.workspace import routes

        source = inspect.getsource(routes.plan_detail)
        assert "scenario_from_stored" in source
        assert "migration_for" in source
