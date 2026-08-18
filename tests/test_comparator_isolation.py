"""The historical comparator must not be measured through today's architecture.

Three things the closure report keeps separate, and it is worth naming them
because two of them were briefly the same thing:

    serving path          what the product does now — the hosted reader and
                          the deterministic syntax witness, in the profile the
                          deployment declares
    closure/drift         evidence *about* that serving behaviour
    compiler comparator   a historical baseline, reconstructing what
                          `quantify-compiler@2` read, pinned as a defect report

`measure(witness="compiler")` is the third. Its numbers only mean anything
against the same numbers taken before, and `test_strategy_families` pins
`SILENTLY_REDUCED = 17` for exactly that reason.

**This happened.** The serving measurement was changed to ask `pilot.read`
instead of reconstructing its answer — correct, and the reason the closure lane
had drifted twice — and the comparator was routed through it too. The frozen
17 became 1. That is not the compiler improving; it is a historical baseline
being retroactively measured through an architecture that did not exist when
the defects were found, which is the one thing a comparator must never do. The
file's own history already recorded the number moving twice for instrumental
reasons while the reader stayed constant, and this would have been the third.

So the separation is asserted structurally rather than left to the comment that
now sits beside the branch. Checked on the syntax tree and by execution, since
a comment is advice and a call graph is a fact.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

LANE = (pathlib.Path(__file__).resolve().parent.parent
        / "corpus" / "parser" / "strategy_closure.py")


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(LANE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"{LANE.name} has no {name}")


def _serving_machinery(node: ast.AST) -> list:
    """Serving-path machinery reached from inside a function.

    Named exactly, because the blunt version had two false positives and both
    were things the comparator is supposed to do:

        `reader.read(text, schema)`   the comparator calling its own reader,
                                      which is the whole reconstruction
        `from src.workspace.pilot import _relation_fields`
                                      folding relation kinds through the
                                      serving path's helper instead of keeping
                                      a second copy — deliberate, and the
                                      opposite of the defect

    What is forbidden is `pilot.read` itself and the `_serving` wrapper around
    it. `in` is not identity and a module is not a function; a check that
    cannot tell `reader.read` from `pilot.read` would force the comparator to
    be written around it.
    """
    found = []
    for inner in ast.walk(node):
        if isinstance(inner, ast.ImportFrom):
            module = inner.module or ""
            if module.endswith("workspace.pilot"):
                for alias in inner.names:
                    if alias.name == "read":
                        found.append(f"{module}.read")
            if module.startswith("discovery_runtime"):
                found.append(module)
        elif isinstance(inner, ast.Import):
            for alias in inner.names:
                if alias.name.startswith("discovery_runtime"):
                    found.append(alias.name)
        elif isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
            if inner.func.id in ("_serving", "configured_syntax_reader"):
                found.append(inner.func.id)
    return sorted(set(found))


def test_the_reconstruction_path_reaches_no_serving_machinery():
    """`_read` is the comparator's own way of asking, and stays that way."""
    clashes = _serving_machinery(_function("_read"))
    assert not clashes, (
        f"the reconstruction path uses {clashes}. It exists to model a reader "
        "that nothing in src/ constructs, so measuring it through the serving "
        "path measures that reader through a path it never takes.")


def test_the_serving_path_is_asked_only_for_the_serving_witness():
    """The branch, read off the tree rather than trusted.

    `measure` must choose: the serving witness is asked, the comparator is
    reconstructed. A version that called `_serving` unconditionally is what
    moved the frozen baseline.
    """
    measure = _function("measure")

    calls = [n for n in ast.walk(measure)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    called = {n.func.id for n in calls}
    assert {"_serving", "_read"} <= called, (
        f"`measure` calls {sorted(called & {'_serving', '_read'})}; it needs "
        "both — one path per witness — and calling only one means some witness "
        "is being measured the other's way")

    # And the choice is made on the witness, not on something incidental.
    guarded = [n for n in ast.walk(measure) if isinstance(n, ast.If)]
    assert any("witness" in ast.dump(n.test) or "SERVING" in ast.dump(n.test)
               for n in guarded), (
        "nothing in `measure` branches on which witness is being measured")


def test_the_comparator_still_reports_its_pinned_defect_count():
    """The property the separation exists to protect, checked end to end.

    Deterministic and provider-free, so this runs anywhere. If the comparator
    is ever routed through the serving path again, this is the number that
    moves — from 17 to 1 the last time.
    """
    import json
    import subprocess
    import sys

    artifact = LANE.parent / "strategy_closure_compiler.json"
    if not artifact.exists():
        subprocess.run([sys.executable, str(LANE), "--compiler"],
                       cwd=LANE.parent.parent.parent, check=True)

    report = json.loads(artifact.read_text())
    assert report["witness"] == "quantify-compiler@2", report["witness"]
    assert report["by_state"].get("SILENTLY_REDUCED") == 17, (
        f"the comparator reports {report['by_state']}. It is deterministic, so "
        "a move is either a change to that reader or a change to how it is "
        "measured — and the second is what this file exists to prevent.")


def test_the_comparator_reports_none_of_the_serving_only_states():
    """`NEEDS_INPUT` and `ASKED_NOT_REFUSED` come from asking `pilot.read`.

    They cannot arise from the reconstruction, which sees declared dimensions
    and a manifest and never a compiled plan. Finding one in the comparator's
    report is the clearest possible sign it went down the serving path.
    """
    import json

    artifact = LANE.parent / "strategy_closure_compiler.json"
    if not artifact.exists():
        pytest.skip("the comparator report has not been produced")

    states = set(json.loads(artifact.read_text())["by_state"])
    serving_only = states & {"NEEDS_INPUT", "ASKED_NOT_REFUSED"}
    assert not serving_only, (
        f"the comparator reports {sorted(serving_only)}, which only the "
        "serving path produces")


def test_the_three_measurements_are_named_apart_in_the_lane():
    """A reader of the file can tell which is which.

    Weaker than the checks above and kept because the next person to change
    this needs the reason, not just a failing test.
    """
    text = LANE.read_text()
    assert "historical" in text or "comparator" in text
    assert "quantify-compiler" in text


def test_the_guard_catches_the_mistake_that_was_made(tmp_path):
    """The mutation, and it is the diff that actually happened.

    A `measure` that calls `_serving` for every witness — which is how the
    frozen 17 became 1 — and a `_read` that imports `pilot.read`.
    """
    planted = tmp_path / "lane.py"
    planted.write_text(
        "def _read(reader, schema, text):\n"
        "    from src.workspace.pilot import read\n"
        "    return read(text, reader, schema=schema)\n")
    tree = ast.parse(planted.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_read")
    assert _serving_machinery(node) == ["src.workspace.pilot.read"]


def test_the_guard_permits_what_the_comparator_legitimately_does(tmp_path):
    """The other half, and the reason the first version was wrong.

    Calling the reader it was handed, and folding relations through the
    serving path's helper rather than keeping a second copy of that logic.
    Both were reported as violations by a check that matched names loosely.
    """
    planted = tmp_path / "lane.py"
    planted.write_text(
        "def _read(reader, schema, text):\n"
        "    from src.workspace.pilot import _relation_fields\n"
        "    result = reader.read(text, schema)\n"
        "    return {**{r.dimension: r.value for r in result.readings},\n"
        "            **_relation_fields(result)}\n")
    tree = ast.parse(planted.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_read")
    assert _serving_machinery(node) == [], (
        "the comparator calling its own reader, or sharing the relation "
        "helper, was reported as reaching serving machinery")
