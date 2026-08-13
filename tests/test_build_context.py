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


def test_what_the_deploy_runs_in_the_container_is_in_the_image():
    """The other direction, and the one that actually broke a deploy.

    The tests above ask "is the expensive thing excluded". Nothing asked "is
    the thing the deployment needs still present", so excluding `tests/` passed
    every check here and failed on a host with `No module named 'conftest'` —
    after the image had been pulled, migrations had run, and every identity
    assertion had already passed.

    The deploy runs its launch journeys inside the container against paths under
    `/app`. Any path it reaches for has to survive the build context, so this
    reads the playbook rather than trusting a comment to stay true.
    """
    import re

    role = (ROOT / "infra" / "ansible" / "roles" / "quantify" / "tasks"
            / "main.yml")
    if not role.exists():
        pytest.skip("no ansible role in this checkout")

    rules = {line.strip().rstrip("/")
             for line in DOCKERIGNORE.read_text().splitlines()
             if line.strip() and not line.startswith("#")}

    needed = set(re.findall(r"/app/([A-Za-z0-9_][A-Za-z0-9_./-]*)",
                            role.read_text()))
    assert needed, "the playbook reaches into no /app path; this test is stale"

    for path in sorted(needed):
        top = path.split("/")[0]
        assert top not in rules, (
            f"the deploy runs code from /app/{path} inside the container and "
            f"`{top}/` is excluded from the image. The build succeeds, every "
            "identity check passes, and the playbook fails on the host")


def test_no_tracked_data_is_excluded_from_the_image():
    """Runtime data the application reads must survive the build context.

    `data/snapshots/` was excluded on the reasoning that snapshots are mounted
    at runtime. Nothing mounts them — the host mounts telemetry and nothing
    else — so the image read sentences, compiled plans and sealed intents, then
    reported "market data is not available in this deployment" for every one.
    The product's entire output, removed by a build rule, while every
    deployment check passed and the launch journey passed with it.

    Tracked is the line. Files under `data/` that git carries are source the
    application resolves; `cache/`, `history/` and the `.db` files are local
    and are meant to go. `infra/` is tracked and excluded deliberately, which
    is why this rule is scoped to the data the runtime reads rather than to
    everything in the repository.
    """
    tracked = subprocess.run(["git", "ls-files", "data/"],
                             cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode != 0 or not tracked.stdout.strip():
        pytest.skip("no tracked data in this checkout")

    rules = {line.strip().rstrip("/")
             for line in DOCKERIGNORE.read_text().splitlines()
             if line.strip() and not line.startswith("#")}

    directories = {"/".join(path.split("/")[:2])
                   for path in tracked.stdout.split()}
    excluded = sorted(d for d in directories if d in rules or f"{d}/" in rules)
    assert not excluded, (
        f"{excluded} are tracked in git and excluded from the image. The "
        "application reads them at runtime and nothing mounts them, so the "
        "container starts, passes every check, and cannot do its job")


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
