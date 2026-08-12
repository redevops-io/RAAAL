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
    # What happens next is asserted by
    # `test_a_job_that_must_call_a_provider_fails_without_its_key`, which
    # checks the exit status rather than the wording. An earlier version of
    # this test matched on the message text and broke when the message was
    # improved — pinning prose is how a test starts measuring itself.


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


class TestTheWorkflowsAreHardenedForAPublicRepository:
    """The repository was made private after a look at these. Going back to
    public should rest on these properties holding, not on remembering that
    they held once.
    """

    def test_every_action_is_pinned_to_a_commit(self):
        """A tag is a pointer its owner can move. `liskin/gh-workflow-keepalive@v1`
        is a third party, and a job that carries a credential running code
        someone else can change under a fixed name is the supply-chain shape.

        GitHub's own actions get the same treatment: `actions/checkout@v4` is
        also mutable, and the reason to trust it is a judgement about a vendor
        rather than a property of the reference.
        """
        for path in WORKFLOWS:
            for ref in re.findall(r"uses:\s*(\S+)", path.read_text()):
                if ref.startswith("./"):
                    continue
                assert re.search(r"@[0-9a-f]{40}$", ref), (
                    f"{path.name} uses {ref}, which is a movable reference")

    def test_every_workflow_declares_its_permissions(self):
        """The repository default is `read`, and a default is a setting
        somebody can change in a UI without touching a reviewed file."""
        for path in WORKFLOWS:
            text = path.read_text()
            has_top = re.search(r"^permissions:", text, re.M)
            has_job = re.search(r"^\s{4}permissions:", text, re.M)
            assert has_top or has_job, (
                f"{path.name} declares no permissions and inherits whatever "
                "the repository setting happens to be")

    def test_no_event_data_is_expanded_into_a_shell(self):
        """`${{ github.event.<field> }}` inside a `run:` block is substituted
        before the shell sees it, so the value becomes script.

        Read from the parsed YAML rather than by matching text. The first
        version used a regex for `run:` blocks, over-captured past the end of
        one, and reported an `env:` assignment in the next step as if it were
        shell — a false positive that would have been "fixed" by editing
        perfectly safe configuration.

        `github.event_name` is deliberately not covered: it is one of a fixed
        set GitHub assigns and cannot carry a payload.
        """
        import yaml

        for path in WORKFLOWS:
            document = yaml.safe_load(path.read_text())
            for job in (document.get("jobs") or {}).values():
                for step in job.get("steps", []):
                    script = step.get("run")
                    if not script:
                        continue
                    found = re.findall(r"\$\{\{\s*github\.event\.[^}]*\}\}",
                                       script)
                    assert not found, (
                        f"{path.name} expands {found} in the script of step "
                        f"{step.get('name', '?')!r}; those fields carry text a "
                        "contributor chooses. Pass it through `env:`")

    def test_a_job_that_must_call_a_provider_fails_without_its_key(self):
        """A green job that skipped the only thing it exists to do is the
        PASS/VACUOUS problem in CI form."""
        for path in WORKFLOWS:
            text = path.read_text()
            if "API_KEY" not in text:
                continue
            for guard in re.findall(r'if \[ -z "\$[A-Z_]*API_KEY" \];(.*?)fi',
                                    text, re.S):
                assert "exit 0" not in guard, (
                    f"{path.name} exits successfully when its provider key is "
                    "absent, so a run that made no call reports as evidence")
