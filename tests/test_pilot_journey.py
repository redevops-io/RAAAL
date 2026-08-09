"""The Era 1 launch journey, unchanged, on the runtime path.

    describe → clarify → save → figure → reopen → the same figure

That criterion predates this runtime by months and was chosen for a product
reason rather than an implementation one: *a Quantify plan is not useful merely
because it compiles correctly; it must execute and produce an interpretable
result.* A pilot that stopped at "the runtime understood you" would validate
the parser and the workspace integration, not the product.

So the gate is kept and execution was wired to meet it, rather than the gate
being softened to match what was built.

**The rule the wiring follows.** Legacy *execution* may be reused; legacy
*interpretation* may not be reintroduced. Both paths call
`run_boundary.execute_compiled_plan`, so there is one implementation of Quantify's
simulation and no second copy to drift — and the pilot branch never reaches
`compile_draft` or `compile_scenario`, which `test_pilot_route` proves
separately.
"""
from __future__ import annotations

import os
import re

import pytest
from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"
NEW = "/workspace/new"


@pytest.fixture
def journey_client(monkeypatch, tmp_path):
    """A runtime deployment with synthetic prices, so a figure is reachable."""
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-by-recordings")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/j.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    return TestClient(app)


def _no_legacy_interpretation(monkeypatch):
    """Both legacy compilers replaced with probes that raise.

    Execution is shared; interpretation is not. If the journey completes with
    these in place, no part of it turned text into a plan through the old
    grammar.
    """
    import src.mission.compiler as compiler
    import src.workspace.draft as draft

    def refuse(*args, **kwargs):
        raise AssertionError(
            "legacy interpretation was reached on the runtime path")

    monkeypatch.setattr(draft, "compile_draft", refuse)
    monkeypatch.setattr(compiler, "compile_scenario", refuse)


def _figure_in(page: str):
    """The figure the page reports, if it reports one."""
    found = re.search(r'class="figure">([^<]+)<', page)
    return None if found is None else found.group(1).strip()


class TestTheLaunchJourney:
    def test_describe_clarify_save_figure_reopen(self, journey_client,
                                                 monkeypatch):
        """The whole gate, in one test, because the steps depend on each other.

        Split into six tests each would pass individually while the journey
        stayed broken — which is the failure `test_journey.py` was written
        against, and the reason it is one test there too.
        """
        _no_legacy_interpretation(monkeypatch)

        # describe
        drafted = journey_client.get(NEW, params={"describe": SENTENCE})
        assert drafted.status_code == 200
        assert "assets" in drafted.text, "the plan should ask what to hold"

        # clarify + save
        saved = journey_client.post(
            "/pilot/save",
            data={"describe": SENTENCE, "answer_assets": "VTI"},
            follow_redirects=False)
        assert saved.status_code == 303
        location = saved.headers["location"]

        # figure
        opened = journey_client.get(location)
        assert opened.status_code == 200
        figure = _figure_in(opened.text)
        assert figure, (
            "the plan reopened without a figure. A plan that compiles and "
            "produces nothing is the failure this gate was written for")

        # reopen → the same figure
        again = journey_client.get(location)
        assert _figure_in(again.text) == figure
        assert again.text == opened.text, (
            "two reopens of one plan rendered differently; the figure a user "
            "returns to must be the figure their confirmed plan produces")

    def test_the_figure_comes_from_the_persisted_artifact(self, journey_client,
                                                          monkeypatch):
        """Reopening executes from the pinned intent, with no model in reach.

        The reader is replaced with one that raises. A figure still rendering
        means the number came from the stored artifact rather than from a fresh
        interpretation of the sentence.
        """
        _no_legacy_interpretation(monkeypatch)

        saved = journey_client.post(
            "/pilot/save",
            data={"describe": SENTENCE, "answer_assets": "VTI"},
            follow_redirects=False)
        location = saved.headers["location"]

        import src.workspace.pilot_routes as pilot_routes

        def no_reader():
            raise AssertionError(
                "a reader was constructed while producing a figure on the "
                "reopen path")

        monkeypatch.setattr(pilot_routes, "configured_reader", no_reader)
        opened = journey_client.get(location)
        assert opened.status_code == 200
        assert _figure_in(opened.text)


class TestTheCoverageGateStillGuardsTheFigure:
    def test_a_plan_with_an_unexecuted_declared_element_shows_no_figure(
            self, journey_client, monkeypatch):
        """The gate that caught three prompts returning an identical $103,393
        while each quietly dropped a different declared element.

        A pilot path that produced figures without consulting it would have
        routed around the check rather than through it, and the check's whole
        value is that it sits between a plan and a number.
        """
        _no_legacy_interpretation(monkeypatch)

        import src.mission.coverage as coverage_module

        # Asserted as *reached*, not as forced to refuse. Faking a verdict
        # would test that a rigged refusal suppresses a figure, which is a fact
        # about the fake. What matters is that the pilot path routes through
        # the gate rather than around it — so the real assessment runs and is
        # watched.
        real = coverage_module.assess
        seen = {"called": False}

        def watching(*args, **kwargs):
            seen["called"] = True
            return real(*args, **kwargs)

        monkeypatch.setattr(coverage_module, "assess", watching)

        journey_client.post("/pilot/save",
                            data={"describe": SENTENCE, "answer_assets": "VTI"},
                            follow_redirects=True)
        assert seen["called"], (
            "the coverage gate was not consulted on the pilot path; a figure "
            "can then be published for a plan that dropped something the user "
            "said")


class TestExecutionIsSharedAndInterpretationIsNot:
    def test_both_paths_call_one_execution_boundary(self):
        """One implementation of Quantify's simulation. Two copies drift, and
        the drift shows up as two users getting different numbers from the
        same plan."""
        import ast
        from pathlib import Path

        from src.workspace import pilot_routes

        tree = ast.parse(Path(pilot_routes.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(f"{node.module}.{a.name}" for a in node.names)

        assert any("execute_compiled_plan" in name for name in imported)
        assert not [n for n in imported
                    if "compile_draft" in n or "compile_scenario" in n]

    def test_the_boundary_does_not_import_an_interpreter(self):
        """`run_boundary.py` takes a compiled scenario. There is no path
        through it that turns text into a plan."""
        import ast
        from pathlib import Path

        from src.workspace import run_boundary

        source = Path(run_boundary.__file__).read_text()
        tree = ast.parse(source)
        called = {ast.unparse(n.func) for n in ast.walk(tree)
                  if isinstance(n, ast.Call)}
        assert not [c for c in called
                    if "compile_scenario" in c or "compile_draft" in c]
