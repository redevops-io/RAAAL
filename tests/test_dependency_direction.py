"""Which package may import which, enforced on the import graph.

**Read from the AST, never from source text.** A check that greps for
`from ..deploy` matches its own docstring, every comment that mentions the rule,
and the frozen list below — and reports a codebase in violation of a rule it is
in fact obeying. The import graph is the structure; the text is prose about it.

**A ratchet, not a wish.** Three rules below are already broken in places, and a
guard that fails on the day it lands gets deleted or marked xfail within a week.
So each known violation is frozen with its current count, and the test fails in
both directions:

    a new violating edge            -> fail; the rule was broken somewhere new
    an existing edge growing        -> fail; the debt got deeper
    an edge shrinking or removed    -> fail; update the baseline

That last one is the part that matters. A baseline nobody has to touch when
things improve goes stale silently, and a stale baseline permits exactly the
imports somebody already paid to remove.

The rules themselves:

**Nothing imports `deploy`.** It is the composition root — it resolves the
environment and wires concrete implementations into the things that need them.
Seven packages import it today, almost all reaching `deploy.context.current()`
for configuration. That is a service locator: it makes a module's real
dependencies invisible at its boundary, and it is why the market-data store
cannot be constructed in a test without a deployment existing.

**`discovery` does not import `mission`.** Discovery reads words and produces a
sealed `VerifiedIntent`; Mission binds that intent to an executable plan.
Mission depends on Discovery. Both directions exist today, which means the
layering is currently a description rather than a constraint.

**Nothing imports `workspace` or `web`.** They are the UI. A lower layer
importing them inverts the whole stack.
"""
from __future__ import annotations

import ast
import collections
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

#: Packages that may not be imported, and by whom they are imported anyway.
#: Counts are a ceiling. Lower the number when you remove an import; delete the
#: entry when you remove the last one.
FROZEN = {
    ("agentic", "deploy"): 1,
    ("db", "deploy"): 2,
    ("discovery", "deploy"): 1,
    ("market_data", "deploy"): 4,
    ("mission", "deploy"): 1,
    ("telemetry", "deploy"): 1,
    ("workspace", "deploy"): 28,
    ("discovery", "mission"): 4,
    ("db", "workspace"): 2,
    ("deploy", "workspace"): 4,
    ("web", "workspace"): 1,
    ("workspace", "web"): 1,
}

#: The packages nothing below them may import.
FORBIDDEN_TARGETS = {"deploy", "workspace", "web"}

#: Layering that is not merely "do not import X", but a direction.
FORBIDDEN_EDGES = {("discovery", "mission")}


def _package_of(path: pathlib.Path) -> str:
    parts = path.relative_to(SRC).parts
    return parts[0] if len(parts) > 1 else "<top>"


def import_graph() -> collections.Counter:
    """Every cross-package import in `src/`, counted, from the syntax tree."""
    edges: collections.Counter = collections.Counter()
    for file in SRC.rglob("*.py"):
        if "__pycache__" in str(file):
            continue
        package = _package_of(file)
        try:
            tree = ast.parse(file.read_text())
        except SyntaxError:                    # not this test's job to report
            continue
        for node in ast.walk(tree):
            for target in _targets(node, package):
                if target and target != package:
                    edges[(package, target)] += 1
    return edges


def _targets(node: ast.AST, package: str):
    """The package a single import statement reaches, if it reaches one.

    Relative imports carry the level: `..market_data` leaves the current
    package, `.sibling` does not. Absolute `src.x` names its package directly.
    """
    if isinstance(node, ast.ImportFrom):
        if node.level and node.level >= 2:
            yield (node.module or "").split(".")[0]
        elif node.module and node.module.startswith("src."):
            yield node.module.split(".")[1]
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith("src."):
                yield alias.name.split(".")[1]


def violations() -> collections.Counter:
    found: collections.Counter = collections.Counter()
    for (importer, target), count in import_graph().items():
        if target in FORBIDDEN_TARGETS or (importer, target) in FORBIDDEN_EDGES:
            found[(importer, target)] = count
    return found


def test_no_new_forbidden_import():
    """An edge that is not in the baseline is a rule broken somewhere new."""
    new = {edge: n for edge, n in violations().items() if edge not in FROZEN}
    assert not new, (
        "new forbidden imports:\n" + "\n".join(
            f"  src/{a}/ imports src/{b}/ ({n}×)" for (a, b), n in new.items())
        + "\n\nIf this is deliberate, the import is still wrong — the layer "
          "below should be handed what it needs, not reach up for it.")


def test_forbidden_imports_do_not_grow():
    """The debt may shrink. It may not deepen."""
    found = violations()
    deeper = {e: (FROZEN[e], found[e]) for e in FROZEN
              if found.get(e, 0) > FROZEN[e]}
    assert not deeper, "\n".join(
        f"src/{a}/ imports src/{b}/ {now}× (baseline {was})"
        for (a, b), (was, now) in deeper.items())


def test_baseline_has_no_stale_entries():
    """Improvements must be recorded, or the baseline re-permits them.

    A frozen list that nobody updates when an import is removed keeps allowing
    it, so the next reintroduction passes silently. This is the direction the
    ratchet actually turns in.
    """
    found = violations()
    stale = {e: (FROZEN[e], found.get(e, 0)) for e in FROZEN
             if found.get(e, 0) < FROZEN[e]}
    assert not stale, ("the baseline is out of date — lower or delete these:\n"
                       + "\n".join(f"  ('{a}', '{b}'): {now},  # was {was}"
                                   for (a, b), (was, now) in stale.items()))


@pytest.mark.parametrize("package", sorted(FORBIDDEN_TARGETS))
def test_the_graph_is_actually_read(package):
    """The guard fails if the reader stops finding imports.

    `import_graph` returning nothing would make every test above pass, and the
    suite would report a clean layering because it read no code at all. This
    asserts the reader still sees the package's own outbound imports, which are
    legitimate and must always be non-empty.
    """
    edges = import_graph()
    outbound = sum(n for (importer, _), n in edges.items() if importer == package)
    assert outbound > 0, (
        f"the import reader found no imports from src/{package}/, so the "
        "layering checks above are passing on an empty graph")


# --------------------------------------------------------------------------
# The canonical runtimes, and which way they may be depended on.
#
#     Quantify domain code
#         v
#     discovery-runtime
#         v
#     runtime-contracts
#
# One direction. A canonical runtime that imported Quantify would stop being
# canonical the moment it did — it would carry finance vocabulary into a
# package whose whole claim is that it does not have any, and the other
# consumer of the same contract would inherit it.

CANONICAL_PACKAGES = ("runtime_contracts", "discovery_runtime")

#: The vendored checkout, if it is present. The submodule is pinned but not
#: wired in, so these run against whatever is on disk and skip nothing when it
#: is absent — an assertion that silently disappears with its subject is how a
#: guard stops guarding.
VENDORED = pathlib.Path(__file__).resolve().parent.parent / "vendor"


def _imports_of(root: pathlib.Path):
    """Every module name imported anywhere under a directory tree."""
    names = set()
    for file in root.rglob("*.py"):
        if "__pycache__" in str(file) or "/.git/" in str(file):
            continue
        try:
            tree = ast.parse(file.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Absolute only. `from .mission import x` inside
                # runtime_contracts.models names that package's own submodule
                # and has nothing to do with Quantify's `mission` — reading the
                # name without the level reported three false violations in
                # the contracts package and one in the runtime, all of them
                # relative imports of a sibling.
                if node.level == 0 and node.module:
                    names.add((node.module.split(".")[0], str(file)))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add((alias.name.split(".")[0], str(file)))
    return names


def test_the_canonical_runtimes_do_not_import_quantify():
    """The direction, checked on whichever canonical packages are on disk.

    `src` is the obvious name to look for and not the only one: a runtime that
    reached for `workspace` or `mission` directly would be just as wrong and
    would not contain the string "src".
    """
    import runtime_contracts

    quantify_packages = {p.name for p in SRC.iterdir() if p.is_dir()
                         and not p.name.startswith("__")} | {"src"}

    roots = [pathlib.Path(runtime_contracts.__file__).parent]
    vendored = VENDORED / "discovery-runtime" / "discovery_runtime"
    if vendored.exists():
        roots.append(vendored)

    offences = []
    for root in roots:
        for name, where in _imports_of(root):
            if name in quantify_packages:
                offences.append(f"{where} imports {name}")
    assert not offences, (
        "a canonical runtime imports Quantify:\n  " + "\n  ".join(offences)
        + "\n\nThe dependency runs the other way. A runtime that imports the "
          "application is not application-neutral, and the other consumer of "
          "the same contract inherits whatever it picked up.")


def test_the_contracts_package_does_not_import_the_discovery_runtime():
    """Contracts sit below the runtime, not beside it.

    `discovery-runtime` depends on `runtime-contracts`. The reverse would be a
    cycle between two separately released packages, which is resolvable only
    by releasing them together forever.
    """
    import runtime_contracts

    root = pathlib.Path(runtime_contracts.__file__).parent
    offences = [f"{where} imports discovery_runtime"
                for name, where in _imports_of(root)
                if name == "discovery_runtime"]
    assert not offences, "\n".join(offences)


def test_quantify_has_no_fallback_to_a_local_discovery_runtime():
    """No optional import that silently substitutes a local implementation.

    A `try: from discovery_runtime import X / except ImportError: from
    .something import X` would make the canonical dependency optional, and the
    deployment that quietly ran the local half would be indistinguishable from
    the one that did not — which is the failure the whole migration exists to
    remove.
    """
    offenders = []
    for file in SRC.rglob("*.py"):
        if "__pycache__" in str(file):
            continue
        try:
            tree = ast.parse(file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            imported = any(
                isinstance(inner, (ast.Import, ast.ImportFrom))
                and any(a.name.split(".")[0] in CANONICAL_PACKAGES
                        for a in getattr(inner, "names", []))
                or (isinstance(inner, ast.ImportFrom)
                    and (inner.module or "").split(".")[0] in CANONICAL_PACKAGES)
                for inner in ast.walk(node))
            caught = any(
                isinstance(h.type, ast.Name) and h.type.id == "ImportError"
                for h in node.handlers)
            if imported and caught:
                offenders.append(
                    f"{file.relative_to(SRC.parent)}:{node.lineno}")
    assert not offenders, (
        "a canonical runtime import is wrapped in an ImportError fallback: "
        + ", ".join(offenders))
