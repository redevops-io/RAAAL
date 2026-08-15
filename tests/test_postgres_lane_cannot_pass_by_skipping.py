"""The lane's own guard, exercised rather than described.

`QUANTIFY_REQUIRE_POSTGRES` turns a skipped database guarantee into a failed
run, so the persistence lane cannot report green having quietly run the SQLite
suite. That mechanism is itself a claim, and an unchecked guard is the thing it
exists to prevent: the digest check was skipped on SQLite for its whole life and
nobody noticed, which is what "a guard nobody tested" looks like from inside.

**The first version of this file tested a guard that was not running.** It built
a temporary conftest that imported the hooks by name, and they never fired —
and the reason they never fired is worth more than the test was. `tests/conftest.py`
already defined `pytest_sessionstart` and `pytest_sessionfinish` for the
tree-freeze guard, so a second pair of functions with those names further up the
module was silently discarded: Python keeps the last definition. The lane
reported green while checking nothing, which is precisely the failure it was
written to prevent, one level up.

So this runs the real pytest against real test files in a subprocess, and reads
the exit status. Nothing here imports a hook.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: A fast file with no database of its own, so what is being measured is the
#: lane guard and not the suite it guards.
SAMPLE = "tests/test_conventions.py"

#: Never connected to. Case 2 needs the URL merely *present*, so that the
#: precondition passes and the run reaches the end-of-session check.
UNUSED_URL = "postgresql://unused:unused@127.0.0.1:1/none"


def run_pytest(**environment) -> subprocess.CompletedProcess:
    env = {**os.environ}
    env.pop("QUANTIFY_REQUIRE_POSTGRES", None)
    env.pop("QUANTIFY_TEST_POSTGRES_URL", None)
    env.update(environment)
    return subprocess.run(
        [sys.executable, "-m", "pytest", SAMPLE, "-q", "-p", "no:randomly"],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=600)


class TestTheGuardBites:
    def test_the_flag_without_a_database_refuses_to_start(self):
        """Before collection, so the lane cannot run the wrong suite at all."""
        result = run_pytest(QUANTIFY_REQUIRE_POSTGRES="1")
        assert result.returncode != 0, (
            "the lane started without a database, so it would have run the "
            "SQLite suite and reported the persistence lane green")
        assert "QUANTIFY_TEST_POSTGRES_URL is not" in (
            result.stdout + result.stderr)

    def test_a_guarantee_that_never_ran_fails_the_run(self):
        """The half that a skip-watcher cannot see.

        This selects one unrelated file. Nothing skips; the named guarantees
        simply are not there. A rename, a stray `-k` or a deleted file produces
        exactly this, and it must not be green.
        """
        result = run_pytest(QUANTIFY_REQUIRE_POSTGRES="1",
                            QUANTIFY_TEST_POSTGRES_URL=UNUSED_URL)
        assert result.returncode != 0, (
            "the lane passed without running a single guarantee it names")
        assert "did not check what it exists to check" in result.stdout
        assert "test_the_stored_digest_is_the_resolver_digest" in result.stdout

    def test_the_sample_itself_passes(self):
        """Otherwise the two failures above prove nothing about the guard."""
        result = run_pytest()
        assert result.returncode == 0, result.stdout[-1500:]


class TestItStaysOutOfTheWayOtherwise:
    def test_without_the_flag_the_suite_is_unchanged(self):
        """The guard is for one lane.

        If it fired everywhere, a developer with no database would face a wall
        and would turn it off — and a check that blocks everything teaches
        people to ignore checks just as reliably as one that blocks nothing.
        """
        result = run_pytest()
        assert "did not check what it exists to check" not in result.stdout


class TestTheSkipHalf:
    """Checked directly, because provoking a real database skip needs the flag
    set *and* no URL, which the precondition already refuses."""

    def test_a_database_skip_becomes_a_problem(self, monkeypatch):
        from tests import conftest

        monkeypatch.setenv("QUANTIFY_REQUIRE_POSTGRES", "1")
        monkeypatch.setattr(
            conftest, "_skipped_for_postgres",
            [("tests/test_provenance_journey.py::test_x",
              "needs a reachable PostgreSQL")])
        monkeypatch.setattr(conftest, "_ran",
                            set(conftest.MUST_RUN_ON_POSTGRES))

        problems = conftest._lane_problems()
        assert problems and "skipped for want of a database" in problems[0]

    def test_nothing_is_a_problem_when_the_flag_is_off(self, monkeypatch):
        from tests import conftest

        monkeypatch.delenv("QUANTIFY_REQUIRE_POSTGRES", raising=False)
        monkeypatch.setattr(
            conftest, "_skipped_for_postgres",
            [("tests/test_provenance_journey.py::test_x",
              "needs a reachable PostgreSQL")])
        assert conftest._lane_problems() == []


class TestTheNamedListIsReal:
    def test_every_named_guarantee_exists(self):
        """A list of node ids nothing checks is a list that rots.

        Each entry names a file, a class and a function. If one is renamed the
        lane starts failing with "did not run" — correct, and far harder to
        diagnose than being told here that the name is stale.
        """
        import ast

        from tests.conftest import MUST_RUN_ON_POSTGRES

        for nodeid in MUST_RUN_ON_POSTGRES:
            path, class_name, function = nodeid.split("::")
            source = REPO_ROOT / path
            assert source.exists(), f"{path} does not exist ({nodeid})"
            names = {node.name for node in ast.walk(ast.parse(source.read_text()))
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            assert class_name in names, f"{class_name} is not in {path}"
            assert function in names, f"{function} is not in {path}"


class TestNoHookIsSilentlyShadowed:
    """The defect that made the first version of this file useless.

    Two functions with one name in one module: the later wins and the earlier
    vanishes without a warning from anything. It cost a working guard once
    already, and the next person adding a hook to this conftest will not know.
    """

    def test_each_pytest_hook_is_defined_once(self):
        import ast
        from collections import Counter

        source = (REPO_ROOT / "tests" / "conftest.py").read_text()
        defined = Counter(
            node.name for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("pytest_"))
        repeated = {name: count for name, count in defined.items() if count > 1}
        assert repeated == {}, (
            f"{repeated} defined more than once in tests/conftest.py; the "
            "earlier definition never runs and nothing reports it")
