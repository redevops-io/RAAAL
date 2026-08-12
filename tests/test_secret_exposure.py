"""Whether a repository secret can be reached by someone without write access.

This repository is public. Actions secrets are encrypted and never appear in
the tree, and GitHub does not pass them to workflows triggered by a pull
request *from a fork* — which is the protection people usually mean when they
say a secret is safe in a public repo.

That protection has a known hole, and it is not theoretical: `pull_request_target`
and `workflow_run` run in the **base** repository's context and **do** receive
secrets, while being triggerable by an outsider's pull request. A workflow that
uses either and then checks out the pull request's code is the standard
public-repo credential-exfiltration pattern.

No workflow here does that today. This file is what keeps it that way, because
the alternative is remembering — and the cost of forgetting once is a live
provider key in someone else's hands.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOWS = sorted((Path(__file__).resolve().parent.parent
                    / ".github" / "workflows").glob("*.yml"))

#: Triggers that hand secrets to a run an outsider can start.
DANGEROUS_TRIGGERS = ("pull_request_target", "workflow_run")


def _uses_secret(text: str) -> bool:
    return bool(re.search(r"\$\{\{\s*secrets\.", text))


def test_there_are_workflows_to_check():
    """A glob that matched nothing would make every assertion below vacuous."""
    assert WORKFLOWS


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_workflow_both_takes_a_secret_and_runs_for_outsiders(path):
    text = path.read_text()
    if not _uses_secret(text):
        return
    for trigger in DANGEROUS_TRIGGERS:
        assert not re.search(rf"^\s*{trigger}\s*:", text, re.M), (
            f"{path.name} maps a repository secret and triggers on "
            f"`{trigger}`, which runs in this repository's context for a pull "
            "request anybody can open. In a public repository that is a "
            "credential handed to a stranger")


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_a_secret_bearing_pull_request_workflow_is_inert_without_it(path):
    """Banning `pull_request` outright was the first version of this test, and
    it was stricter than the property. GitHub does not give secrets to fork
    pull requests, so such a run simply arrives with the variable empty — the
    question is what happens then.

    The answer must be a run that stops and says the check did not happen.
    `parser-corpus.yml` did stop, and said `no key configured` while exiting
    zero, so a check that had never once executed reported success on every
    pull request for months. That is not a secret-exposure bug; it is the
    silent-skip bug wearing the same clothes, and it is why this asserts on
    the message rather than only on the exit.
    """
    text = path.read_text()
    if not _uses_secret(text):
        return
    triggers = set(re.findall(r"^\s{2}(\w+)\s*:", text, re.M))
    if "pull_request" not in triggers:
        return
    assert re.search(r"if \[ -z \"\$[A-Z_]*API_KEY\"", text), (
        f"{path.name} maps a secret and runs on pull_request without checking "
        "whether the secret arrived. A fork pull request gets an empty "
        "variable and would call a provider with no credential")
    assert "NOT checked" in text or "not checked" in text, (
        f"{path.name} handles a missing secret without saying the check was "
        "skipped. A step that no-ops quietly is a green tick for work that "
        "did not happen")


def test_only_the_expected_workflow_carries_a_provider_credential():
    """A new workflow reaching for the key should be a decision somebody
    makes, not something noticed later in a billing alert."""
    carrying = {p.name for p in WORKFLOWS
                if re.search(r"(OPENAI|ANTHROPIC)_API_KEY", p.read_text())}
    # Two, and both are deliberate. The drift lane is the measurement; the
    # parser corpus re-asks a subset of eight sentences on a pull request and
    # diffs them against the recordings, which is a much smaller spend and a
    # different question — has the provider moved under a fixed id.
    #
    # Written as an exact set rather than a maximum, so a third workflow
    # reaching for the key is a decision somebody makes rather than something
    # noticed later in a billing alert.
    assert carrying == {"drift-lane.yml", "parser-corpus.yml"}, (
        f"{sorted(carrying)} map a provider credential. Each one spends real "
        "money on a schedule somebody has to have chosen")
