"""`pilot.read()` reaches the official runtime, and cannot reach internal fusion.

The differential proves the runtime is *substitutable* for what serves. This
proves what *serves* — a different claim, which is why the two artifacts are
kept apart. A green differential says nothing about which implementation the
request actually went through.

Checked three ways, because each alone can certify a cutover that did not
happen:

    structurally   the serving module no longer imports the legacy fusion
    dynamically    a real read is watched, and the runtime's fuse is called
    negatively     the legacy fuse is replaced with one that raises, and a read
                   must still succeed

The third is the one that cannot be satisfied by accident. An import-graph test
that happened to see nothing would pass the first; a call-counter attached to
the wrong function would pass the second.
"""
from __future__ import annotations

import ast
import os
import pathlib

import pytest

PILOT = pathlib.Path(__file__).resolve().parent.parent / "src" / "workspace" / "pilot.py"

DECLARED = {"QUANTIFY_PILOT_READER": "recorded",
            "QUANTIFY_PARSER_MODE": "RUNTIME",
            "QUANTIFY_PARSER_MODEL": "claude-sonnet-5",
            "ANTHROPIC_API_KEY": "unused",
            "PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}

RECORDED = "invest $500 monthly into VTI"

#: Names that would mean the serving module still makes its own fusion
#: decisions rather than reading the runtime's.
LEGACY_FUSION = {"fuse", "fuse_with_bindings", "Proposal", "Decision"}


@pytest.fixture()
def declared(monkeypatch):
    from src.deploy import context as deploy_context

    settings = deploy_context.resolve({**os.environ, **DECLARED})
    monkeypatch.setattr(deploy_context, "current", lambda: settings)
    return settings


def _read(with_syntax=False):
    from src.discovery.hosted_recording import RecordedHostedReader
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.syntax_stanza import RecordedReader
    from src.discovery.witnesses import BOTH, MODEL_ONLY
    from src.workspace.pilot import read

    return read(RECORDED, RecordedHostedReader(), schema=QUANTIFY_SCHEMA,
                profile=BOTH if with_syntax else MODEL_ONLY,
                syntax_reader=RecordedReader() if with_syntax else None)


def test_the_serving_module_imports_no_legacy_fusion():
    """Structural, on the syntax tree.

    A grep matches this docstring and the list above it. The property is which
    names the module imports, which is a fact about the tree.
    """
    tree = ast.parse(PILOT.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and "fusion" in (node.module or ""):
            if (node.module or "").startswith("discovery_runtime"):
                continue
            imported.update(a.name for a in node.names)
    clashes = imported & LEGACY_FUSION
    assert not clashes, (
        f"pilot.py imports {sorted(clashes)} from the internal fusion. The "
        "serving path reads decisions now; it does not make them.")


@pytest.mark.parametrize("with_syntax", [False, True], ids=["single", "two"])
def test_a_real_read_calls_the_runtimes_fusion(declared, monkeypatch,
                                               with_syntax):
    """Dynamic, watched where the serving path actually resolves the name.

    Patched on the package attribute, not on `discovery_runtime.reader`: the
    adapter does `from discovery_runtime import fuse` inside the call, so the
    lookup happens per request against `discovery_runtime.fuse`. Watching the
    submodule instead saw nothing and reported the cutover as incomplete when
    it was the watcher that was in the wrong place.
    """
    import discovery_runtime

    calls = []
    original = discovery_runtime.fuse

    def watched(*args, **kwargs):
        calls.append(args[0] if args else "")
        return original(*args, **kwargs)

    monkeypatch.setattr(discovery_runtime, "fuse", watched)
    reading = _read(with_syntax)

    assert reading.intent is not None, "the read produced no intent"
    assert calls, (
        "no call reached discovery_runtime's fuse; the serving path is not "
        "going through the runtime")


@pytest.mark.parametrize("with_syntax", [False, True], ids=["single", "two"])
def test_a_read_succeeds_with_the_legacy_fusion_disabled(declared, monkeypatch,
                                                         with_syntax):
    """Negative, and the one that cannot pass by accident.

    The internal `fuse` is replaced with a function that raises. A read that
    still succeeds cannot have called it — no import-graph oversight and no
    mis-attached counter can fake that.
    """
    import src.discovery.fusion as legacy

    def refuse(*_args, **_kwargs):
        raise AssertionError(
            "the serving path called the internal fusion. The cutover is not "
            "complete, or something reintroduced a fallback.")

    monkeypatch.setattr(legacy, "fuse", refuse)
    monkeypatch.setattr(legacy, "fuse_with_bindings", refuse, raising=False)

    reading = _read(with_syntax)
    assert reading.intent is not None
    assert reading.settled, "the read settled nothing"


def test_no_fallback_to_the_legacy_implementation():
    """No `try: runtime / except: internal` anywhere in the serving module.

    A fallback is worse than a failure: it makes two implementations serve
    interchangeably and which one answered a given request unknowable.
    """
    tree = ast.parse(PILOT.read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom):
                module = inner.module or ""
                if "discovery.fusion" in module or "discovery.pipeline" in module:
                    offenders.append(f"pilot.py:{node.lineno}")
    assert not offenders, (
        f"a legacy implementation is imported inside a try at {offenders}")
