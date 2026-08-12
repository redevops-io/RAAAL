"""What the image is allowed to contain.

The served image was 19.3GB. Three of them filled a 29GB volume and the deploy
refused, correctly, on a full disk — so the symptom appeared in the deployment
lane, one layer away from the cause, which was that `COPY . .` had nothing
telling it what to leave behind.

There was no `.dockerignore`, and there could not have been one: `.gitignore`
listed it. A file that fixes the build was itself unable to be committed, which
is why this went unnoticed through every previous deploy. That is the specific
regression these tests exist to stop coming back.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCKERIGNORE = ROOT / ".dockerignore"


def test_the_build_context_is_constrained_at_all():
    assert DOCKERIGNORE.exists(), (
        "no .dockerignore, so `COPY . .` takes the entire working tree "
        "including everything git ignores")


def test_the_file_that_fixes_the_build_can_be_committed():
    """The actual defect. `.dockerignore` was in `.gitignore`, so the fix for a
    19.3GB image was a file the repository refused to track. Nothing else here
    would have caught it: the tests below all pass against a `.dockerignore`
    that exists locally and reaches no build server."""
    ignored = subprocess.run(
        ["git", "check-ignore", "-v", ".dockerignore"],
        cwd=ROOT, capture_output=True, text=True)
    assert ignored.returncode != 0, (
        f"`.dockerignore` is ignored by {ignored.stdout.strip()}. It is build "
        "configuration, not local state — ignoring it means the build server "
        "never receives it and builds the whole tree")

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ".dockerignore"],
        cwd=ROOT, capture_output=True, text=True)
    assert tracked.returncode == 0, (
        "`.dockerignore` is untracked; a build from a clean checkout would "
        "have no build context rules at all")


#: The directories that actually caused this, with what they cost. Named
#: individually rather than checked as a total, because a size assertion
#: would pass on a machine that simply had not built the proofs yet.
MUST_BE_EXCLUDED = {
    ".venv/": "a 5.6GB virtualenv; the image installs its own dependencies",
    "formal/.lake/": "5.1GB of compiled Lean; the proofs are not run in the "
                     "image",
    ".git/": "history the image never reads — the deployment identity arrives "
             "as an environment variable",
}


@pytest.mark.parametrize("path,why", sorted(MUST_BE_EXCLUDED.items()))
def test_the_expensive_directories_are_excluded(path, why):
    rules = {line.strip() for line in DOCKERIGNORE.read_text().splitlines()
             if line.strip() and not line.startswith("#")}
    assert path in rules, f"{path} is not excluded: {why}"


def test_docker_agrees_that_they_are_excluded():
    """The rules are checked above by reading them, which assumes this file
    means to Docker what it reads like. This asks Docker.

    Skipped rather than failed where there is no daemon: the property is about
    the image, and a machine that cannot build one cannot answer.
    """
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip("no docker daemon")

    built = subprocess.run(
        ["docker", "build", "-q", "-f", "-", "."],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
        input="FROM busybox\nCOPY . /ctx\nRUN du -sm /ctx | cut -f1 > /size\n")
    assert built.returncode == 0, built.stderr[-2000:]

    image = built.stdout.strip()
    try:
        size = subprocess.run(["docker", "run", "--rm", image, "cat", "/size"],
                              capture_output=True, text=True, timeout=120)
        megabytes = int(size.stdout.strip())
    finally:
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True)

    # Generous: the point is to catch a multi-gigabyte tree arriving, not to
    # police the context to the megabyte.
    assert megabytes < 500, (
        f"the build context is {megabytes}MB. Something large is no longer "
        "excluded; `docker build` sends this to the daemon and `COPY . .` "
        "bakes it into the image")
