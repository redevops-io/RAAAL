"""The intent boundary's types come from `runtime-contracts`, and only there.

Quantify does not define `VerifiedIntent`, `Author`, `IntentState`,
`Unresolved` or their neighbours. It imports them. That is what makes a plan
sealed here readable by another consumer of the same contract, and what makes
`intent_from_json` on a stored artifact produce the intent that was pinned
rather than a local approximation of it.

**Checked on the syntax tree, not in the source text.** A grep for
`class VerifiedIntent` matches this docstring, the list below, and every
comment that mentions the rule — and would report a violation in a file whose
only crime is describing the constraint.

**Names collide, and a collision is the thing worth catching.** Until this
guard existed `src/mission/spec.py` defined `Unresolved` — a question to put to
a person, with `field` / `question` / `why_it_matters` — while
`runtime_contracts.Unresolved` is the discovery boundary's record of an open
dimension, with `dimension` / `reason` / `detail` / `evidence` /
`result_changing`. Two incompatible types, one name, both inside
`src/mission/`: `compiler` bound one and `verified_intent` the other. Nothing
had gone wrong yet only because no single module had reason to import both.
The local type is `OpenQuestion` now.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

#: The boundary. Not every name the package exports — `Check`, `Verdict` and
#: `State` are ordinary words that several unrelated modules use for their own
#: purposes, and forbidding them would be a naming policy rather than a
#: contract guard. These are the types the sealed-intent boundary is made of,
#: where a local definition means two systems disagreeing about what a
#: verified intent *is*.
BOUNDARY_TYPES = {
    "VerifiedIntent",
    "Author",
    "IntentState",
    "Unresolved",
    "DecisionEvidence",
    "Amendment",
    "IntentField",
    "IntentRelation",
    "RelationMember",
    "OpenReason",
    "ReaderKind",
    "NotSealable",
    "CorruptIntent",
}


def _definitions():
    """Every class defined under `src/`, by name, from the AST."""
    found = []
    for file in SRC.rglob("*.py"):
        if "__pycache__" in str(file):
            continue
        try:
            tree = ast.parse(file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found.append((node.name, file, node.lineno))
    return found


def test_no_boundary_type_is_defined_locally():
    """The whole point: these are imported, never re-declared."""
    clashes = [(name, str(f.relative_to(SRC.parent)), line)
               for name, f, line in _definitions() if name in BOUNDARY_TYPES]
    assert not clashes, (
        "boundary contract types defined inside Quantify:\n"
        + "\n".join(f"  {f}:{line}  {name}" for name, f, line in clashes)
        + "\n\nImport them from `runtime_contracts`. A local definition means "
          "a plan sealed here is not the same artifact another consumer of "
          "the contract would read back.")


def test_the_boundary_types_all_exist_in_the_contract():
    """The list names real types, so the guard above cannot pass vacuously.

    A misspelling in `BOUNDARY_TYPES` would silently stop guarding that type
    while the test still went green — the failure mode of every allow/deny
    list nobody re-derives.
    """
    import runtime_contracts

    missing = sorted(n for n in BOUNDARY_TYPES
                     if not hasattr(runtime_contracts, n))
    assert not missing, (
        f"{missing} are guarded but are not exported by runtime_contracts, so "
        "the guard is watching for names that cannot occur")


def test_the_scanner_actually_reads_classes():
    """An empty scan would make the guard pass by reading nothing."""
    found = _definitions()
    assert len(found) > 100, (
        f"only {len(found)} classes found under src/; the AST scan is not "
        "reading the tree, so the guard above proves nothing")


def test_the_dependency_is_pinned_to_a_tag_not_a_branch():
    """A branch ref is a mutable authority.

    The contract that defines what a sealed intent *is* must not be able to
    change under a fixed name. This is the same rule the container image
    follows by digest, and the reason `tunnel_origin` stopped being a default:
    an identifier that can mean something different tomorrow is not a pin.
    """
    root = SRC.parent
    declared = []
    for name in ("requirements.txt", "requirements-core.txt"):
        path = root / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if "runtime-contracts" in line:
                declared.append((name, line.strip()))

    assert declared, "runtime-contracts is not declared in any requirements file"
    for name, line in declared:
        assert "@v" in line or "==" in line, (
            f"{name} pins runtime-contracts to something that is not a tag or "
            f"a version: {line}")
        assert not line.rstrip().endswith("@main"), (
            f"{name} pins runtime-contracts to a branch: {line}")


def test_the_installed_version_matches_the_pin():
    """Declared and installed can disagree, and then the pin is a comment."""
    from importlib.metadata import version

    root = SRC.parent
    text = (root / "requirements-core.txt").read_text()
    line = next(l for l in text.splitlines() if "runtime-contracts" in l)
    pinned = line.rsplit("@v", 1)[-1].strip() if "@v" in line else ""
    assert pinned, f"could not read a pinned version from: {line}"

    installed = version("runtime-contracts")
    assert installed == pinned, (
        f"runtime-contracts is pinned to {pinned} and {installed} is "
        "installed. Every claim about contract compatibility in this suite is "
        "about the installed one.")


@pytest.mark.parametrize("name", sorted(BOUNDARY_TYPES))
def test_each_boundary_type_is_importable(name):
    """Named individually so a failure says which type went missing."""
    import runtime_contracts

    assert hasattr(runtime_contracts, name)
