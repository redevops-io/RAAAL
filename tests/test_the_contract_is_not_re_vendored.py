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

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ("requirements.txt", "requirements-core.txt")

#: The runtime-family packages this repository pins by git tag.
#:
#: `agentic-os` joined when the cross-runtime replay gate needed both runtimes
#: in one process. The rules below were written for the contracts and apply
#: unchanged — a moving ref is no safer for a runtime than for a contract.
FAMILY = ("runtime-contracts", "agentic-os")

#: `<package> @ git+…@<ref>` — the ref is what this file cares about.
PIN = re.compile(r"^(?P<package>[\w-]+)\s*@\s*git\+\S+@(?P<ref>\S+)\s*$", re.M)
TAG = re.compile(r"^v\d+\.\d+\.\d+")

MODULE = {"runtime-contracts": "runtime_contracts", "agentic-os": "agentic_os"}

#: Packages deliberately absent from the image the pilot runs, with the reason
#: and the test that keeps the reason true.
#:
#: An entry here is a claim that the serving path cannot reach the package —
#: not that somebody would rather not install it — and the third field is what
#: makes that claim checkable after the person who made it has moved on.
NOT_IN_THE_IMAGE = {
    "agentic-os": (
        ("requirements-core.txt",),
        "it pins runtime-contracts v0.2.2 against Quantify's v0.2.4, so an "
        "image installing both is unsatisfiable; the serving path reaches no "
        "module under src/agentic and no agentic_os import at all",
        "test_serving_image_closure.py"),
}


def _pins():
    for name in REQUIREMENTS:
        path = ROOT / name
        if path.exists():
            for match in PIN.finditer(path.read_text()):
                if match.group("package") in FAMILY:
                    yield name, match.group("package"), match.group("ref")


class TestThePinIsStable:
    def test_every_requirements_file_asks_for_each_of_them(self):
        """Two files install this project — the image builds from the core one.
        A pin in only one of them is a dependency that exists on a developer's
        machine and not in the container that runs the pilot.

        That is the direction this was written for and it still holds. The
        other direction — a package deliberately kept *out* of the image —
        needs an argument, not an exemption, so `NOT_IN_THE_IMAGE` names the
        package, the reason, and the test that proves the reason.
        """
        for package in FAMILY:
            pinned = {name for name, pkg, _ in _pins() if pkg == package}
            absent, _, _ = NOT_IN_THE_IMAGE.get(package, ((), "", ""))
            expected = set(REQUIREMENTS) - set(absent)
            assert pinned == expected, (
                f"{package} is pinned in {sorted(pinned)} and expected in "
                f"{sorted(expected)}")

    @pytest.mark.parametrize("package", sorted(NOT_IN_THE_IMAGE))
    def test_a_package_kept_out_of_the_image_has_a_proof(self, package):
        """A declared exception is not a justified one.

        `agentic-os@v0.2.3` pins `runtime-contracts@v0.2.2` against Quantify's
        `v0.2.4`, so an image installing both is unsatisfiable — the build
        failed with `ResolutionImpossible`, and had done since the contracts
        pin moved, unnoticed because nothing built the image.

        Removing it is only safe while the serving path cannot reach it, and
        that is a property somebody has to keep checking. So the exception
        names the test that checks it, and this asserts that test exists and
        is about this package.
        """
        _, reason, proof = NOT_IN_THE_IMAGE[package]
        assert len(reason.split()) >= 12, f"{package}: {reason!r}"

        path = ROOT / "tests" / proof
        assert path.exists(), (
            f"{package} is kept out of the image on the strength of {proof}, "
            "which does not exist")
        text = path.read_text()
        assert MODULE[package] in text or package in text, (
            f"{proof} does not mention {package}, so it is not the proof that "
            "the serving path cannot reach it")

    def test_each_is_a_tag_and_not_a_moving_ref(self):
        """The whole reason the vendored copy outlived its other justifications.

        A branch pin means the code this repository was tested against can be
        rewritten without a commit here — the same fragility this project
        criticised in `mission-sdk`'s bare-commit pin on `agentic-os`. If they
        change, that should be a diff here.
        """
        for name, package, ref in _pins():
            assert TAG.match(ref), (
                f"{name} pins {package} at {ref!r}, which is not a version "
                "tag. A ref that can move under this repository is not a "
                "dependency, it is a subscription")

    def test_the_pinned_version_is_the_one_installed(self):
        """A declared dependency nobody installed is a wish, and the suite
        would still pass on whatever happens to be on the machine."""
        import importlib

        for name, package, ref in _pins():
            installed = importlib.import_module(MODULE[package]).__version__
            assert installed == ref.lstrip("v"), (
                f"{name} pins {package} {ref}, {installed} is installed — the "
                "tests are measuring something other than what this repository "
                "declares")


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
