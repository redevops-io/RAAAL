"""Can the running service say which revision it is?

    master SHA -> deployment run -> deployed artifact
               -> serving SHA    -> the AGPL source link for that SHA

Every link was broken. `BuildManifest` has required four deployment facts
since it was written, nothing has ever supplied them, and so every deployment
answered `observable: false` — a service that could not say what it was. The
consequences were two, and they look unrelated until the cause is named:

- A pilot observation could only be joined to code by timestamp, so a cohort
  spanning a deploy would be one population wearing two behaviours.
- The AGPL §13 offer pointed at a repository rather than a revision, which is
  an offer for whatever the default branch holds when somebody clicks it.

**What these tests do not prove.** That a deployment happened. The workflow now
supplies the facts and has zero registered runners to execute on, so the supply
side is unexercised — `test_the_supply_side_is_declared_but_unproven` says so
rather than leaving a green tick to imply otherwise.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / ".github" / "workflows" / "daily-deploy.yml"


@pytest.fixture
def deployed(monkeypatch):
    """A context that knows what it is, as a real deployment would."""
    from src.deploy import context

    resolved = context.resolve({
        "QUANTIFY_COMMIT": "e675471abcdef", "QUANTIFY_RELEASE_REF": "master",
        "QUANTIFY_IMAGE_DIGEST": "sha256:deadbeef",
        "QUANTIFY_SNAPSHOT_ID": "31603973383"})
    monkeypatch.setattr(context, "current", lambda: resolved)
    return resolved


class TestTheServiceCanSayWhichRevisionItIs:
    def test_the_manifest_is_observable_when_the_facts_are_present(self,
                                                                   deployed):
        assert deployed.build.observable
        assert not deployed.build.missing

    def test_and_says_so_honestly_when_they_are_not(self):
        """False is the useful answer. A manifest that filled the gaps would
        describe a build that does not exist and be indistinguishable from a
        correct one."""
        from src.deploy.context import resolve

        build = resolve({}).build
        assert not build.observable
        assert "QUANTIFY_COMMIT" in build.missing

    def test_the_commit_stays_out_of_the_public_view(self, deployed):
        """Written the other way round first, and the existing boundary was
        right: the public view is "what a client needs to know it is
        compatible, and nothing more", and it deliberately carries none of the
        deployment facts.

        The AGPL entitlement does not require widening it. `source_url()`
        reads the private view and renders a link that carries the revision,
        so the user is given the corresponding source without this object
        becoming the place operational facts leak from.
        """
        public = deployed.build.public()
        for name in ("commit", "release_ref", "image_digest", "snapshot_id"):
            assert name not in public
        assert public["observable"] is True

    def test_but_the_private_view_has_it(self, deployed):
        assert deployed.build.private()["commit"] == "e675471abcdef"


class TestTheSourceOfferNamesTheRevisionServing:
    def test_it_points_at_the_running_commit(self, deployed):
        from src.api import source_url

        assert source_url().endswith("/tree/e675471abcdef")

    def test_it_falls_back_to_the_repository_when_unknown(self, monkeypatch):
        """Too broad rather than silently wrong, and the manifest reports
        `observable: false` in the same case so the two statements agree."""
        from src.api import SOURCE_REPOSITORY, source_url
        from src.deploy import context

        monkeypatch.setattr(context, "current", lambda: context.resolve({}))
        assert source_url() == SOURCE_REPOSITORY

    def test_the_notice_is_computed_rather_than_frozen_at_import(self,
                                                                  deployed):
        """It was a module-level constant built when the process started —
        the moving-target problem one level in."""
        from src.api import license_notice

        assert license_notice()["source"].endswith("/tree/e675471abcdef")


class TestAnObservationCanBeJoinedToItsCode:
    def test_every_event_carries_the_serving_commit(self, deployed):
        from src.workspace.pilot_events import _profile

        assert _profile()["serving_commit"] == "e675471abcdef"

    def test_and_carries_an_empty_one_rather_than_inventing_it(self,
                                                               monkeypatch):
        from src.deploy import context
        from src.workspace.pilot_events import _profile

        monkeypatch.setattr(context, "current", lambda: context.resolve({}))
        assert _profile()["serving_commit"] == ""


class TestTheSupplySide:
    """`daily-deploy.yml` is gone. It rsynced research artifacts to a Proxmox
    directory on a self-hosted runner that does not exist, and it deployed
    nothing. `deploy-aws.yml` replaces it, and the deployment facts now travel
    the route `infra/` already built: Terraform variable -> Ansible variable
    -> `production.env.j2` -> `QUANTIFY_COMMIT` in the running container.
    """

    WORKFLOW = ROOT / ".github" / "workflows" / "deploy-aws.yml"

    def test_the_commit_it_deploys_is_the_commit_it_builds(self):
        """Not a tag and not a branch. Either would let two deployments claim
        one identity, which is the property the whole chain rests on."""
        text = self.WORKFLOW.read_text()
        assert "build_commit=${COMMIT}" in text
        assert 'COMMIT=${GITHUB_SHA}' in text

    def test_the_image_is_pinned_by_digest_not_tag(self):
        """Terraform refuses an unpinned image, so a workflow passing a tag
        would fail there — but it would fail late and for a reason that reads
        as a Terraform problem. A tag that can move under a fixed name is the
        same defect as a model alias under a fixed reader id."""
        text = self.WORKFLOW.read_text()
        assert "imageDigest" in text
        assert "@${DIGEST}" in text

    def test_the_proof_runs_and_is_not_conditional(self):
        """A deployment job that ends when Ansible exits has reported that a
        playbook ran, not that the service serves the revision it was given."""
        import yaml

        steps = yaml.safe_load(self.WORKFLOW.read_text())["jobs"]["deploy"]["steps"]
        proof = [s for s in steps
                 if "verify_deployment_identity" in str(s.get("run", ""))]
        assert len(proof) == 1, "the deployment does not prove what it deployed"
        assert "if" not in proof[0], (
            "the proof is conditional, so a deployment can succeed without it")
        assert proof[0]["run"].strip().startswith("set -euo pipefail"), (
            "the proof step does not fail the job when the proof fails")

    def test_the_proof_is_preserved(self):
        """The durable half of `cohort event -> serving_commit -> proof ->
        revision`. The running service will not be serving this revision in
        three months; the artifact still says what it was."""
        text = self.WORKFLOW.read_text()
        assert "upload-artifact" in text
        assert "deployment-proof" in text

    def test_it_is_still_unproven_against_the_live_service(self):
        """The honest state, and the reason the cohort has not been invited.

        The live service reports `observable: true` — the AWS deployment does
        supply the facts — and offers source for the repository rather than a
        revision, because it is running `3eaa5eb`, merged as PR #3, from before
        `source_url()` existed. Running the proof against it returns MISMATCH,
        which is the conjunctive condition working: the service can identify
        itself and the identity it offers is not the one deployed.

        Delete this test when the proof passes against quantify.club and the
        artifact is preserved beside the cohort evidence.
        """
        assert (ROOT / "evidence" / "deployment-proof.txt").exists() is False, (
            "a deployment proof exists; run it against the live service, and "
            "if it passes, replace this test with one that reads the "
            "preserved proof")
