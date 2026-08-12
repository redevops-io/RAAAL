"""One place decides which provider serves.

The selection lived in four modules and a fifth that needed it never got it:
`corpus/parser/drift_lane.py` built `HostedReader()` directly, so after the
serving provider moved to OpenAI the lane looked for `ANTHROPIC_API_KEY`, found
none, and refused to run — in a CI job whose environment held a valid OpenAI
key and whose workflow correctly declared `QUANTIFY_PARSER_PROVIDER: OPENAI`.

It cost a dispatch to find, and it would not have been found by reading: every
individual file was self-consistent. A rule duplicated four times is a rule
that will be applied three times.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SELECTOR = ROOT / "src" / "discovery" / "readers_quantify.py"

#: Where the provider classes may legitimately be named: the module that
#: defines them and the one selector that chooses between them.
MAY_NAME_A_PROVIDER_CLASS = {SELECTOR}

SOURCES = [p for p in list((ROOT / "src").rglob("*.py"))
           + list((ROOT / "corpus").rglob("*.py"))
           if "__pycache__" not in p.parts]


def _constructs(tree, name: str) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == name for n in ast.walk(tree))


def test_there_are_sources_to_check():
    assert len(SOURCES) > 20


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_nothing_builds_a_provider_reader_for_itself(path):
    """Structural, not a grep: an import is allowed, a *construction* is not.

    Checking for the class name in the text would flag docstrings that explain
    the rule — the same false positive this repository's oracle check produced
    once already.
    """
    if path in MAY_NAME_A_PROVIDER_CLASS:
        return
    tree = ast.parse(path.read_text())
    for cls in ("HostedReader", "OpenAIReader"):
        assert not _constructs(tree, cls), (
            f"{path.relative_to(ROOT)} constructs {cls}() directly. Which "
            "provider serves is a deployment decision resolved in one place; "
            "building it here means this file keeps its own opinion and stops "
            "agreeing with the others the moment the provider changes. Use "
            "`configured_hosted_reader()`")


def test_the_selector_reads_the_declared_provider():
    """And that it is the *declared* one, not a guess from which key exists."""
    tree = ast.parse(SELECTOR.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "configured_hosted_reader")

    # From the AST, not the text. The first version matched the substring
    # "environ" and found it in the word *environment* inside this function's
    # own docstring — a check failing on its own explanation, which is a
    # defect this repository has produced twice before.
    reads_environ = any(
        isinstance(n, ast.Attribute) and n.attr == "environ"
        for n in ast.walk(fn))
    assert not reads_environ, (
        "the selector reads the environment directly; the provider is "
        "resolved in deploy.context and nowhere else")

    calls = {n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "current" in calls, (
        "the selector does not consult the resolved deployment context")
