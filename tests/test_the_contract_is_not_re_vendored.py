"""The Discovery → Mission contract is depended on, never copied back.

This file used to guard a temporary vendored copy under `src/contracts/`. That
copy is gone: `runtime-contracts` landed on `main`, was tagged, and is now a
pinned dependency. What replaced the copy is the thing worth guarding.

The failure mode is not dramatic. It is somebody needing one extra field on a
Friday, copying `intent.py` into this repository to add it, and shipping. The
result compiles, the tests pass, and this runtime now speaks a contract the
other runtime has never seen — a private fork of a shared boundary, which is
precisely what the contract package exists to prevent.

Three properties, each of which failed once in some form during this migration:

    the pin is a tag        a branch can be rebased under a consumer
    the pin resolves        a declared dependency nobody installed is a wish
    nothing redefines it    a second definition is a second version

The old class asserting the copy's own README stayed honest is deleted rather
than left skipping. A test whose subject no longer exists passes forever, and a
green line that checks nothing reads exactly like a guarantee.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ("requirements.txt", "requirements-core.txt")

#: `runtime-contracts @ git+…@<ref>` — the ref is what this file cares about.
PIN = re.compile(r"^runtime-contracts\s*@\s*git\+\S+@(?P<ref>\S+)\s*$", re.M)
TAG = re.compile(r"^v\d+\.\d+\.\d+")


def _pins():
    for name in REQUIREMENTS:
        path = ROOT / name
        if path.exists():
            for match in PIN.finditer(path.read_text()):
                yield name, match.group("ref")


class TestThePinIsStable:
    def test_every_requirements_file_asks_for_it(self):
        """Two files install this project — the image builds from the core one.
        A pin in only one of them is a dependency that exists on a developer's
        machine and not in the container that runs the pilot."""
        pinned = {name for name, _ in _pins()}
        assert pinned == set(REQUIREMENTS), (
            f"runtime-contracts is pinned in {sorted(pinned)} but not in "
            f"{sorted(set(REQUIREMENTS) - pinned)}")

    def test_it_is_a_tag_and_not_a_moving_ref(self):
        """The whole reason the copy outlived its other justifications.

        A branch pin means the contract this runtime was tested against can be
        rewritten without a commit here — the same fragility this project
        criticised in `mission-sdk`'s bare-commit pin on `agentic-os`. If the
        contract changes, that should be a diff in this repository.
        """
        for name, ref in _pins():
            assert TAG.match(ref), (
                f"{name} pins runtime-contracts at {ref!r}, which is not a "
                "version tag. A ref that can move under this repository is not "
                "a contract, it is a subscription")

    def test_the_pinned_version_is_the_one_installed(self):
        """A declared dependency nobody installed is a wish, and the suite
        would still pass on whatever happens to be on the machine."""
        import runtime_contracts

        for _, ref in _pins():
            assert runtime_contracts.__version__ == ref.lstrip("v"), (
                f"pinned {ref}, installed {runtime_contracts.__version__} — "
                "the tests are measuring a different contract than the one "
                "this repository declares")


class TestNothingHereRedefinesIt:
    def test_the_real_package_is_importable(self):
        assert importlib.util.find_spec("runtime_contracts") is not None

    def test_no_local_module_defines_the_boundary_types(self):
        """A copy under any name is the same defect as a copy under the old one.

        Checked by definition rather than by path, so renaming the directory
        does not evade it.
        """
        owned = ("VerifiedIntent", "IntentField", "IntentRelation", "Amendment",
                 "DecisionEvidence", "MissionProposal", "MissionOutcome")
        pattern = re.compile(r"^class (%s)\b" % "|".join(owned), re.M)

        offenders = []
        for path in sorted((ROOT / "src").rglob("*.py")):
            for name in pattern.findall(path.read_text(errors="ignore")):
                offenders.append(f"{path.relative_to(ROOT)} defines {name}")
        assert not offenders, (
            "these types are owned by runtime-contracts and a second "
            "definition is a second version: " + "; ".join(offenders) +
            ". If the contract is missing something, change it there and bump "
            "the pin — slower, and the only way both runtimes stay agreed")

    def test_nothing_imports_a_local_contracts_module(self):
        stale = re.compile(r"^\s*from\s+(\.{1,2}contracts|src\.contracts)\b", re.M)
        offenders = [
            str(path.relative_to(ROOT))
            for folder in ("src", "tests", "scripts", "corpus")
            if (ROOT / folder).exists()
            for path in sorted((ROOT / folder).rglob("*.py"))
            if stale.search(path.read_text(errors="ignore"))]
        assert not offenders, (
            "still importing a vendored contract: " + ", ".join(offenders))
