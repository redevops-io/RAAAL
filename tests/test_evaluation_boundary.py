"""The evaluator must never reinterpret user intent.

The target architecture puts interpretation and calculation on opposite sides of
one line:

    free text -> Discovery ----\\
                                >-- VerifiedIntent -> Mission Runtime
    catalogue -> structured ---/         -> Executable Strategy Specification
                                         -> Evaluation Service -> result

Discovery decides what somebody meant. Evaluation decides what a rule is worth.
An evaluator that can still reach a reader is one that can quietly change the
strategy it was asked to price — and because it would do so while producing a
perfectly well-formed number, nothing downstream would notice.

The boundary is also about to become a deployment: evaluation and market data
move to a separate service and a separate repository. A dependency that is
merely unused today becomes an import error there, and an import error found at
deploy time is one found in the worst place.

**Checked on the import graph rather than on the prose.** A grep for "reader"
matches this docstring. `ast` reads what the module actually imports, which is
the only form of this claim that cannot pass by describing itself.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

#: What must not appear on the calculating side. `discovery` reads language;
#: `workspace` serves pages and owns the database.
INTERPRETERS = ("discovery", "workspace")

#: The packages that are the evaluation service, and are already clean.
CALCULATES = ("evaluation", "market_data")


def imported_packages(package: str) -> dict:
    """Which sibling packages this one imports, and from where.

    Relative and absolute forms both counted: `from ..discovery.syntax import
    normalize` and `import src.discovery.syntax` are the same dependency, and a
    check that saw only one of them would be satisfied by rewriting the import.
    """
    found = {}
    for path in sorted((SRC / package).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [("." * node.level) + (node.module or "")]
            else:
                continue
            for name in names:
                for sibling in INTERPRETERS:
                    if (name.startswith(f"src.{sibling}")
                            or name.lstrip(".").startswith(sibling)
                            and name.startswith(".")):
                        where = f"{path.relative_to(SRC)}:{node.lineno}"
                        found.setdefault(sibling, []).append(where)
    return found


@pytest.mark.parametrize("package", CALCULATES)
def test_the_calculating_side_cannot_reach_an_interpreter(package):
    found = imported_packages(package)
    assert found == {}, (
        f"src/{package} imports an interpreter: "
        + "; ".join(f"{name} at {', '.join(sites)}"
                    for name, sites in sorted(found.items()))
        + ". An evaluator that can reach a reader can change the strategy it "
          "was asked to price, and would do so while returning a well-formed "
          "number")


def test_the_check_would_notice_a_crossing():
    """Without this the test above passes on a typo in the package name.

    `mission` genuinely does import `discovery`, so it doubles as a positive
    control: if `imported_packages` ever stops seeing that, it is not seeing
    anything.
    """
    found = imported_packages("mission")
    assert found.get("discovery"), (
        "the import scanner found no discovery import in src/mission, which "
        "has several — so a clean result from it means nothing")


class TestTheDecompositionHeld:
    """Step 7's exit condition, as a property rather than a plan.

        Given a plan, a frame and a policy, the extracted evaluator produces the
        seven streams and the publish/refuse decision without importing
        `workspace`.

    `_run` did four jobs its name mentioned none of. Calculation and the gate
    that decides whether a figure may be shown moved into `evaluation.core`;
    benchmarks, the comparison payload, comparability verdicts and provenance
    stayed, because none of them changes the figure or whether it is
    publishable — and moving them would have taken the application along.
    """

    def test_the_evaluator_can_calculate_without_the_application(self):
        found = imported_packages("evaluation")
        assert found == {}, (
            f"src/evaluation imports {found}. The exit condition is that the "
            "evaluator computes without the web layer, and an import is the "
            "part that fails at deploy time rather than at test time")

    def test_the_gate_moved_with_the_calculation(self):
        """A gate the caller can forget to ask is a gate that will be forgotten.

        Reconciliation catches the engine doing something other than the plan,
        and coverage catches a declared element that quietly did not run —
        three prompts once returned an identical figure while each dropped a
        different one. Both must be able to withhold the number, so both live
        with the calculation rather than beside it.
        """
        from src.evaluation.core import evaluate_plan
        from src.evaluation.core import Evaluated

        source = (SRC / "evaluation" / "core.py").read_text()
        assert "coverage_module.assess" in source, "coverage does not gate here"
        assert "reconcile" in source, "reconciliation does not gate here"
        assert "publishable" in [f for f in Evaluated.__dataclass_fields__]

    def test_publishable_is_not_merely_having_a_result(self):
        """A run can compute a perfectly good result and still be
        unpublishable. Collapsing the two would let a figure through on the
        strength of having been calculated.

        Checked by constructing the state rather than by reading the source.
        The first version searched `inspect.getsource` for `result is not None`
        and matched the docstring that explains why it must not appear — a
        check reading prose as code, in a test about not doing that.
        """
        from src.evaluation.core import Evaluated

        computed_but_withheld = Evaluated(
            publishable=False, refusal="the ledger and the totals disagree",
            result=object())
        assert computed_but_withheld.result is not None
        assert computed_but_withheld.publishable is False, (
            "a result that exists is being treated as a result that may be "
            "shown, so a failing reconciliation would publish a figure")

    def test_what_stayed_behind_stayed_behind(self):
        """The other half of the classification, checked so the next person
        does not "finish" the move by dragging the page along."""
        source = (SRC / "evaluation" / "core.py").read_text()
        for application in ("comparison_payload", "comparability_records",
                            "_benchmark_specs", "RunConditions"):
            assert application not in source, (
                f"{application} is presentation or comparison shaping and has "
                "moved into the evaluator; it changes no figure and gates "
                "nothing")


class TestTheCompilerNoLongerReadsLanguage:
    """It did, and this is the test that used to say so.

    `from_intent` scanned spans for negation words, split holdings on an
    English "and", stripped currency words from figures, and ran Discovery's
    normaliser to turn "annually" into a period — all inside the compiler,
    after interpretation was supposed to have ended. The earlier version of
    this class asserted those imports were present, so the shape of the problem
    stayed measured; it also said that if they went away, it should be replaced
    by exactly this.

    Two consequences closed together. A sealed intent whose meaning depended on
    code across the boundary could compile differently as that code moved,
    which is what pinning an intent exists to prevent. And the evaluation
    service could not have taken `discovery.syntax` with it.
    """

    def test_from_intent_imports_no_interpreter(self):
        found = imported_packages("mission")
        sites = [where for where in found.get("discovery", ())
                 if "from_intent" in where]
        assert sites == [], (
            f"src/mission/from_intent.py reads Discovery again at {sites}. "
            "A value parsed here is a second opinion about a question the seal "
            "already closed, and it cannot travel to the evaluation service")

    def test_the_gate_reads_a_version_and_not_a_sentence(self):
        """Not every crossing is the same crossing.

        `prelean_gate` imports a reader to ask for its *id*, so a stale drift
        artifact cannot be cited against a reader it was never produced under.
        That is a version check. It travels differently from `from_intent`, and
        conflating the two would make the real finding look bigger and the fix
        look harder than it is.
        """
        source = (SRC / "mission" / "prelean_gate.py").read_text()
        tree = ast.parse(source)
        reads = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute) and node.attr == "read"]
        assert not reads, (
            "prelean_gate calls .read() on something; it is supposed to ask a "
            "reader for its identity and never for an interpretation")
