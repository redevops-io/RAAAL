"""The vendored contract must disappear the moment the real one is depended on.

`src/contracts/` exists because the canonical `VerifiedIntent` lives on a
feature branch rather than a tagged release, and depending on a mutable ref is
the anti-pattern this migration criticises elsewhere. That is a good reason and
it expires.

The failure this guards against is not the copy existing — it is the copy
existing *alongside* the real dependency. Two authoritative copies kept "just in
case" is how a contract acquires a second version that nobody declared, and the
one that drifts is always the one nobody is reading.

So this is written to be unfailable today and unavoidable later:

    today            no `runtime_contracts` package, copy present   -> passes
    dependency added copy still present                             -> FAILS
    dependency added copy deleted                                   -> passes

The deletion therefore happens in the same change as the dependency, because
the suite will not go green otherwise.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "src" / "contracts"


def real_package_available() -> bool:
    return importlib.util.find_spec("runtime_contracts") is not None


def declared_as_a_dependency() -> bool:
    """Whether anything in this repo asks for the real package."""
    for name in ("pyproject.toml", "requirements.txt", "requirements-core.txt"):
        path = ROOT / name
        if path.exists() and "runtime-contracts" in path.read_text():
            return True
    return False


class TestTheCopyAndTheDependencyAreMutuallyExclusive:
    def test_they_never_coexist(self):
        """The exit criterion, enforced rather than remembered."""
        if not (real_package_available() or declared_as_a_dependency()):
            pytest.skip("the real package is neither installed nor declared; "
                        "the vendored copy is the expected state")

        assert not VENDORED.exists(), (
            "runtime-contracts is now depended on, so `src/contracts/` is a "
            "second authoritative copy of a contract two runtimes must agree "
            "on. Delete it, delete scripts/check_vendored_contracts.py, and "
            "point the imports at the real package — in this change, not a "
            "later one.")

    def test_the_drift_check_goes_with_it(self):
        if VENDORED.exists():
            pytest.skip("copy still present, so its check should be too")
        assert not (ROOT / "scripts" / "check_vendored_contracts.py").exists(), (
            "the copy is gone and its drift check is not; a check with nothing "
            "to check passes forever and reads as a guarantee")


class TestWhileTheCopyExistsItStaysHonest:
    def test_it_is_marked_as_not_ours(self):
        if not VENDORED.exists():
            pytest.skip("no copy")
        note = (VENDORED / "__init__.py").read_text()
        assert "not ours" in note.lower()
        assert "runtime-contracts" in note

    def test_the_reason_it_still_exists_is_current(self):
        """Twice already the recorded reason has outlived the fact — archived,
        then private. A stale reason is worse than none: it argues for keeping
        something on grounds that no longer hold."""
        note = (VENDORED / "__init__.py").read_text().lower()
        assert "archived" not in note.split("the reason for the copy")[-1][:400] \
            or "were gone" in note or "both earlier reasons" in note
        assert "main" in note and "tag" in note, (
            "the note should say what unblocks deletion, not only why the copy "
            "exists")
