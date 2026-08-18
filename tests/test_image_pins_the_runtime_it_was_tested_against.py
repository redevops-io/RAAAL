"""The image installs the runtime this repository is tested against.

`requirements-core.txt` is what the production image installs, and it pinned
`discovery-runtime@v0.1.7` while the submodule — the copy every test in this
suite runs against — was at `v0.1.9`. The suite would have been green on a
runtime the deployment did not contain, including the `merge_readings` fix
that stops a second reader's dimension being dropped.

Two declarations of one version, and the one that serves was the stale one.
That is the shape of nearly every defect this migration surfaced, so it gets a
check rather than a convention.

**Read from the files, not from the environment.** `importlib.metadata` reports
what happens to be installed in *this* checkout, which is the submodule — so
comparing the two would compare the submodule with itself.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE = ROOT / "requirements-core.txt"
SUBMODULE = ROOT / "vendor" / "discovery-runtime"


def _pinned_in_requirements(package: str) -> str:
    """The tag `requirements-core.txt` installs for a package."""
    pattern = re.compile(rf"^{re.escape(package)}\s*@\s*git\+\S+@(\S+)\s*$", re.M)
    found = pattern.search(CORE.read_text())
    assert found, f"{package} is not pinned by tag in requirements-core.txt"
    return found.group(1)


def _submodule_tag() -> str:
    if not (SUBMODULE / ".git").exists():
        pytest.skip("the discovery-runtime submodule is not checked out")
    out = subprocess.run(["git", "describe", "--tags", "--exact-match"],
                         cwd=SUBMODULE, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.fail(
            "the submodule is not on a tag. The serving path must depend on a "
            f"tagged release, not a loose commit: {out.stderr.strip()}")
    return out.stdout.strip()


def test_the_image_installs_the_runtime_the_suite_runs_against():
    """From the submodule, so it is the same code and not the same label.

    It was `git+https://...@v0.1.7` while the submodule was at v0.1.9 — the
    suite green against a runtime the deployment did not contain. Pinning the
    tag would have fixed the number; installing the submodule fixes the
    question, because a tag can be moved and a checked-out commit cannot.

    It also has to be this way: the repository is private and a credential-less
    build cannot clone it, which is how the Dockerfile came to be correct in
    git and not buildable.
    """
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "pip install --no-cache-dir ./vendor/discovery-runtime" in dockerfile, (
        "the image does not install the vendored runtime")
    assert "COPY vendor/discovery-runtime" in dockerfile, (
        "the submodule is never copied into the build context")

    # *Both* requirements files, not just the one the image reads.
    #
    # This checked requirements-core.txt only, and requirements.txt went on
    # pinning v0.1.7 while the submodule was v0.1.9 — so CI would install one
    # runtime and the suite report on another. A guard that covers the file
    # somebody happened to be looking at is how the second file drifts.
    for name in ("requirements-core.txt", "requirements.txt"):
        text = (ROOT / name).read_text()
        assert not re.search(r"^discovery-runtime\s*@\s*git\+", text, re.M), (
            f"{name} fetches discovery-runtime from git; two installs of one "
            "package is how a build comes to hold a different version than "
            "the one the tests import")

    assert "./vendor/discovery-runtime" in (ROOT / "requirements.txt").read_text(), (
        "requirements.txt does not install the vendored runtime, so the test "
        "environment and the image install different things")

    # And the submodule is a release, not a commit somebody happened to be on.
    assert _submodule_tag().startswith("v")


def test_the_contracts_pin_agrees_with_the_runtime_that_was_released_against_it():
    """The runtime declares which contracts it was built for; the image must
    install that one, or the two halves of the contract disagree in
    production and nowhere else."""
    import tomllib

    pyproject = tomllib.loads((SUBMODULE / "pyproject.toml").read_text())
    declared = [d for d in pyproject["project"]["dependencies"]
                if d.startswith("runtime-contracts")]
    assert declared, "the runtime declares no contracts dependency"

    wanted = re.search(r"@(\S+)$", declared[0]).group(1)
    assert _pinned_in_requirements("runtime-contracts") == wanted, (
        f"discovery-runtime {_submodule_tag()} was released against "
        f"runtime-contracts {wanted} and the image installs "
        f"{_pinned_in_requirements('runtime-contracts')}")


def test_both_pins_are_tags_rather_than_branches_or_commits():
    """A branch moves under a pin and a bare commit is not a release.

    `discovery-runtime` is not here because it is not fetched — the image
    installs the vendored submodule, whose tag is asserted above.
    """
    for package in ("runtime-contracts",):
        pin = _pinned_in_requirements(package)
        assert re.fullmatch(r"v\d+\.\d+\.\d+", pin), (
            f"{package} is pinned to {pin!r}, which is not a release tag")


def test_the_submodule_can_be_fetched_without_a_credential():
    """CI has no SSH key, and it needs the submodule to build.

    `.gitmodules` used `git@github.com:` while the repository was private, so
    `actions/checkout` — which does not fetch submodules by default anyway —
    could not have got it even when asked. The vendoring fix worked on a
    developer machine, where the submodule is already checked out, and moved
    the failure into the pipeline.
    """
    import configparser

    parser = configparser.ConfigParser()
    parser.read_string((ROOT / ".gitmodules").read_text())
    urls = [parser[s]["url"] for s in parser.sections() if "url" in parser[s]]
    assert urls, "no submodules are declared"
    for url in urls:
        assert url.startswith("https://"), (
            f"{url} needs a credential a CI checkout does not have")


def test_every_workflow_that_installs_the_runtime_fetches_the_submodule():
    """`actions/checkout` does not fetch submodules unless told to.

    Without it `vendor/discovery-runtime` is an empty directory and `pip
    install ./vendor/discovery-runtime` fails — on a path that exists locally,
    which is the shape that makes this invisible until a pipeline runs.
    """
    workflows = ROOT / ".github" / "workflows"
    if not workflows.exists():
        pytest.skip("no workflows in this checkout")

    missing = []
    for path in sorted(workflows.glob("*.yml")):
        text = path.read_text()
        installs = ("requirements.txt" in text or "requirements-core.txt" in text
                    or "docker build" in text)
        if installs and "submodules:" not in text:
            missing.append(path.name)
    assert not missing, (
        f"{missing} install this project and do not fetch submodules, so the "
        "runtime they build against would be an empty directory")
