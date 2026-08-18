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
def test_a_read_succeeds_with_no_legacy_implementation_present(declared,
                                                               with_syntax):
    """Negative, and the one that cannot pass by accident.

    This used to replace the internal `fuse` with a function that raises. The
    internal implementation is deleted now, so the stronger form is available:
    the module is not importable at all, and a read still succeeds. Nothing can
    be calling what does not exist.
    """
    import importlib

    for gone in ("src.discovery.fusion", "src.discovery.pipeline"):
        with pytest.raises(ImportError):
            importlib.import_module(gone)

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


#: The modules the migration deleted, by their absolute dotted path. Absolute
#: rather than a suffix or a pattern: the first version of this check matched
#: any module ending in `fusion` that mentioned `discovery`, which flagged
#: every `discovery_runtime.fusion` import in the codebase — the correct
#: imports — as violations. An assertion has to name the exact surface it
#: means, and "ends with" is not "is".
DELETED = ("src.discovery.fusion", "src.discovery.pipeline")


def _resolved(node, module_path, root):
    """An ImportFrom as an absolute dotted module, relative levels included.

    A relative import carries no package in the tree; `from .fusion import x`
    inside `src/discovery/` and inside `src/mission/` are different modules and
    the node looks identical. Resolving against the importing file is the only
    way to tell them apart.
    """
    package = module_path.relative_to(root.parent).with_suffix("").parts[:-1]
    if node.level == 0:
        return node.module or ""
    base = package[:len(package) - node.level + 1]
    return ".".join((*base, node.module)) if node.module else ".".join(base)


def test_nothing_in_production_imports_a_deleted_module():
    """Repo-wide, not just the serving module.

    `pilot.py` was the module that had to change, so it is the module the rest
    of this file checks. But an import anywhere under `src/` would fail at load
    time in production and pass every test that does not import that module —
    which is the failure mode of checking one file.
    """
    import ast

    root = pathlib.Path(__file__).resolve().parent.parent / "src"
    offenders = []
    for file in sorted(root.rglob("*.py")):
        if "__pycache__" in str(file):
            continue
        try:
            tree = ast.parse(file.read_text())
        except SyntaxError:
            continue
        where = f"{file.relative_to(root.parent)}"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if _resolved(node, file, root) in DELETED:
                    offenders.append(f"{where}:{node.lineno}")
            elif isinstance(node, ast.Import):
                if any(a.name in DELETED for a in node.names):
                    offenders.append(f"{where}:{node.lineno}")
    assert not offenders, (
        f"{offenders} import a module the migration deleted. It would raise at "
        "import time in production and pass every test that does not load it.")


def test_the_walk_catches_a_planted_import(tmp_path):
    """The mutation, both forms — and the false positive that motivated it.

    A relative import from inside the discovery package is a violation; the
    same text from a sibling package is a different module; and an import of
    `discovery_runtime.fusion` is the correct thing this check previously
    reported as a violation.
    """
    import ast

    root = tmp_path / "src"
    (root / "discovery").mkdir(parents=True)
    (root / "mission").mkdir(parents=True)

    inside = root / "discovery" / "reader.py"
    inside.write_text("from .fusion import fuse\n")
    sibling = root / "mission" / "compile.py"
    sibling.write_text("from .fusion import fuse\n")
    correct = root / "discovery" / "adapter.py"
    correct.write_text("from discovery_runtime.fusion import Fusion\n")

    def offends(path):
        tree = ast.parse(path.read_text())
        return [_resolved(n, path, root) for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom)
                and _resolved(n, path, root) in DELETED]

    assert offends(inside) == ["src.discovery.fusion"]
    assert offends(sibling) == [], (
        "a sibling package's own `.fusion` was read as the deleted module")
    assert offends(correct) == [], (
        "the runtime's fusion was reported as the deleted one, which is the "
        "false positive this check was rewritten to remove")
