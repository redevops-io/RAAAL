"""No workflow runs without somebody asking it to.

Every lane in `.github/workflows` is `workflow_dispatch` only, and that is a
decision rather than an oversight: they were failing on every push while
gating nothing, and failing checks that block nothing teach people to ignore
checks. The deployment is dispatch-only too and has never run from a commit.

It was reintroduced within a day. A test lane added here carried `on: push` and
`pull_request` and referenced a `requirements-dev.txt` that has never existed in
this repository, so it failed at the install step on every push — a red cross
beside a green suite, which is worse than no lane at all because it teaches the
same lesson twice as fast.

So the policy is checked rather than remembered. Turning a lane on is then a
line in this file and a line in that one, in the same diff, which is what makes
it a decision.

**Also checked: that every `run:` step installs something that exists.** The
trigger was only half the defect. A lane nobody can run is not safer than a
lane that fires — it is the same lane, discovered later, and usually at the
moment somebody needs it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))

#: Triggers that fire without a person. `schedule` is here too: a nightly lane
#: that fails is the same red cross arriving while nobody is looking.
AUTOMATIC = ("push", "pull_request", "pull_request_target", "schedule",
             "repository_dispatch")


def triggers(path: Path) -> set:
    """The `on:` keys, read without a YAML parser.

    Deliberately literal: this must fail when somebody adds `push:` under `on:`,
    including inside a comment block that was meant to be a note. A commented
    trigger is not a trigger, so comment lines are stripped first — which is the
    one piece of leniency here, and it is why the workflows can keep their
    "the way back is to restore this" blocks.
    """
    lines = [line for line in path.read_text().splitlines()
             if not line.lstrip().startswith("#")]
    found, inside = set(), False
    for line in lines:
        if re.match(r"^on:\s*$", line):
            inside = True
            continue
        if re.match(r"^on:\s*\S", line):          # `on: push` on one line
            found.update(re.findall(r"[a-z_]+", line.split(":", 1)[1]))
            continue
        if inside:
            if line and not line.startswith((" ", "\t")):
                inside = False
                continue
            key = re.match(r"^\s{1,4}([a-z_]+):", line)
            if key:
                found.add(key.group(1))
    return found


@pytest.mark.parametrize("path", WORKFLOWS, ids=[p.name for p in WORKFLOWS])
def test_no_workflow_fires_without_being_asked(path):
    automatic = triggers(path) & set(AUTOMATIC)
    assert not automatic, (
        f"{path.name} fires on {sorted(automatic)}. Every lane here is "
        "dispatch-only because they were failing on every push while gating "
        "nothing. If this one is meant to gate, say so here in the same diff "
        "that turns it on")


@pytest.mark.parametrize("path", WORKFLOWS, ids=[p.name for p in WORKFLOWS])
def test_every_requirements_file_it_installs_exists(path):
    """A lane nobody can run is not a lane that is switched off.

    `pip install -r requirements-dev.txt` failed every push for a file that has
    never existed here. Nothing checked, because nothing reads a workflow.

    Comments stripped first, for the reason the workflow keeps them: the file
    explains what the broken install line used to be, and a scan over raw text
    matched that sentence and reported the defect it describes as still
    present. A check that reads prose as configuration will always find
    whatever it warns about.
    """
    body = "\n".join(line for line in path.read_text().splitlines()
                     if not line.lstrip().startswith("#"))
    missing = [name for name in re.findall(r"-r\s+(\S+\.txt)", body)
               if not (REPO_ROOT / name).exists()]
    assert missing == [], (
        f"{path.name} installs {missing}, which do not exist in this "
        "repository, so the lane fails before it runs anything")


def test_the_reader_finds_the_dispatch_trigger():
    """Without this, a parser that found nothing would pass everything.

    Every workflow here is dispatch-only, so every one must report that trigger.
    A refactor of `on:` blocks that this stopped understanding would otherwise
    look like a repository with no automatic triggers, which is exactly what it
    is meant to prove.
    """
    assert WORKFLOWS, "no workflows found; this test is reading nothing"
    for path in WORKFLOWS:
        assert "workflow_dispatch" in triggers(path), (
            f"{path.name}: the trigger reader found "
            f"{sorted(triggers(path))}, which does not include the one every "
            "workflow in this directory has")


def test_it_would_notice_an_automatic_trigger(tmp_path):
    """The mutation, on the exact shape that shipped."""
    sample = tmp_path / "sample.yml"
    sample.write_text(
        "name: Sample\n\n# push:  <- a note, not a trigger\n\n"
        "on:\n  push:\n    branches: [master]\n  workflow_dispatch:\n\n"
        "jobs: {}\n")
    assert triggers(sample) & set(AUTOMATIC) == {"push"}

    commented = tmp_path / "commented.yml"
    commented.write_text(
        "name: Sample\n\n#   push:\n#     branches: [master]\n\n"
        "on:\n  workflow_dispatch:\n\njobs: {}\n")
    assert triggers(commented) & set(AUTOMATIC) == set(), (
        "a commented-out trigger was read as a real one, which would stop the "
        "workflows recording how to turn themselves back on")


def test_a_comment_about_a_broken_install_is_not_a_broken_install(tmp_path):
    """The mutation for the other half, and it caught itself.

    The install check first read raw text, so the workflow's own note — "the
    first version installed requirements-dev.txt" — was reported as the defect
    it was describing. Prose read as configuration always finds what it warns
    about.
    """
    sample = tmp_path / "sample.yml"
    sample.write_text(
        "name: Sample\n"
        "# it used to run: pip install -r requirements-nonexistent.txt\n"
        "on:\n  workflow_dispatch:\njobs:\n  a:\n    steps:\n"
        "      - run: pip install -r requirements.txt\n")
    body = "\n".join(line for line in sample.read_text().splitlines()
                     if not line.lstrip().startswith("#"))
    assert re.findall(r"-r\s+(\S+\.txt)", body) == ["requirements.txt"]
