"""Which market data produced a stored number, recorded with the number.

The invariant:

    every persisted market-derived value either traces to a specific realized
    snapshot and access decision, or explicitly states that its provenance was
    not recorded — and it never inherits the deployment's current snapshot
    after the fact

The last clause is the load-bearing one. A run made under snapshot A and
reopened after the default moved to B must still say A. Reading the answer from
the environment would produce a confident wrong answer, which is worse than
none: nothing about it looks reconstructed.

`NOT_RECORDED` is a fact about the record, not a gap to be filled. It is never
inferred from the current environment, the default snapshot, a cache, the
nearest timestamp or matching asset coverage.
"""
from __future__ import annotations

import pytest

from src.market_data.access import resolve
from src.market_data.provenance import (
    AccessDecision,
    MarketDataProvenance,
    ProvenanceStatus,
    from_json,
    not_applicable,
    not_recorded,
    verify,
)

POLICY = "PILOT_DATA_POLICY"
AT = "2026-08-01T00:00:00Z"


class TestARecordedProvenanceIdentifiesTheData:
    def test_it_names_the_snapshot_and_its_digest(self, monkeypatch):
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        _, provenance = resolve(context="a run", accessed_at=AT)
        assert provenance.status is ProvenanceStatus.RECORDED
        assert provenance.snapshot_id
        assert provenance.content_digest
        assert provenance.identifies_data

    def test_a_snapshot_id_alone_does_not_identify_data(self):
        """Two snapshots can share a friendly label and hold different objects.
        The digest is what tells them apart."""
        labelled_only = MarketDataProvenance(
            status=ProvenanceStatus.RECORDED, snapshot_id="prices-2026-01",
            access_decision=AccessDecision.SYNTHETIC_ALLOWED, accessed_at=AT)
        assert not labelled_only.identifies_data

    def test_it_carries_the_access_decision(self, monkeypatch):
        """A snapshot id says which data; it does not say the read was
        permitted."""
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        _, provenance = resolve(context="a run", accessed_at=AT)
        assert provenance.access_decision is AccessDecision.SYNTHETIC_ALLOWED
        assert provenance.permitted

    def test_it_carries_the_licence_class_and_review_state(self, monkeypatch):
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        _, provenance = resolve(context="a run", accessed_at=AT)
        assert provenance.license_class
        assert provenance.license_review_status

    def test_it_records_when_the_data_was_read(self, monkeypatch):
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        _, provenance = resolve(context="a run", accessed_at=AT)
        assert provenance.accessed_at == AT

    def test_the_context_reaches_the_record(self, monkeypatch):
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        _, provenance = resolve(context="a distinctive context", accessed_at=AT)
        assert provenance.access_decision_reason == "a distinctive context"


class TestConfigurationDriftDoesNotRewriteHistory:
    """The case that reading current configuration would get wrong."""

    def test_a_stored_provenance_survives_the_default_moving(self,
                                                             monkeypatch):
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        _, original = resolve(context="a run", accessed_at=AT)
        stored = original.to_json()

        # The deployment moves on: a different policy, a different snapshot.
        monkeypatch.setenv(POLICY,
                           "market-data-egress/pilot-vendor-approved@1")
        reopened = from_json(stored)

        assert reopened.snapshot_id == original.snapshot_id
        assert reopened.policy_version == "SYNTHETIC_ONLY"
        assert reopened.content_digest == original.content_digest

    def test_reading_it_consults_nothing_but_the_record(self, monkeypatch):
        """`from_json` must not look at the environment at all."""
        stored = {"status": "RECORDED", "snapshot_id": "snapshot-a",
                  "content_digest": "mdv1:aaa", "policy_version": "policy-a",
                  "access_decision": "SYNTHETIC_ALLOWED", "accessed_at": AT}
        monkeypatch.delenv(POLICY, raising=False)
        assert from_json(stored).snapshot_id == "snapshot-a"
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        assert from_json(stored).snapshot_id == "snapshot-a"


class TestHistoricalAbsenceIsStatedNotInferred:
    def test_a_record_with_no_provenance_reads_as_not_recorded(self):
        assert from_json(None).status is ProvenanceStatus.NOT_RECORDED
        assert from_json({}).status is ProvenanceStatus.NOT_RECORDED

    def test_it_names_no_snapshot(self, monkeypatch):
        """The failure this prevents: a legacy row acquiring today's snapshot
        id and becoming indistinguishable from a real record."""
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        legacy = from_json(None)
        assert legacy.snapshot_id is None
        assert legacy.content_digest is None
        assert not legacy.identifies_data

    def test_absence_that_names_something_is_a_defect(self):
        reconstructed = {"status": "MARKET_DATA_PROVENANCE_NOT_RECORDED",
                         "snapshot_id": "prices-synthetic-1"}
        problems = verify(reconstructed)
        assert any("reconstructed" in one for one in problems)

    def test_not_applicable_is_distinct_from_not_recorded(self):
        """An omitted field and an inapplicable one look identical and mean
        different things."""
        assert not_applicable().status is ProvenanceStatus.NOT_APPLICABLE
        assert not_recorded("x").status is ProvenanceStatus.NOT_RECORDED
        assert not_applicable().status is not from_json(None).status


class TestADeniedReadProducesNoData:
    def test_a_denial_yields_no_prices(self, monkeypatch):
        from src.market_data.loader import synthetic_snapshot
        import src.market_data.access as access
        import src.market_data.pilot_policy as policy_module

        monkeypatch.setattr(access, "approved_snapshot", synthetic_snapshot)
        monkeypatch.setenv(POLICY, "market-data-egress/pilot-vendor-approved@1")
        monkeypatch.setattr(
            policy_module, "authorise",
            lambda snapshot, *, context, **_: (_ for _ in ()).throw(
                policy_module.PilotDataDenied("review open")))

        frame, provenance = resolve(context="a run", accessed_at=AT)
        assert frame is None
        assert provenance.access_decision is AccessDecision.DENIED
        assert not provenance.permitted

    def test_a_result_stored_against_a_denial_fails_verification(self):
        """It should not be producible. If tampering creates one, the record
        is rejected rather than trusted."""
        problems = verify({"status": "RECORDED", "snapshot_id": "s-1",
                           "content_digest": "mdv1:aaa",
                           "access_decision": "DENIED", "accessed_at": AT})
        assert any("DENIED" in one for one in problems)


class TestVerificationReadsTheRecord:
    def test_a_complete_record_passes(self, monkeypatch):
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        _, provenance = resolve(context="a run", accessed_at=AT)
        assert verify(provenance.to_json()) == ()

    def test_a_missing_digest_is_caught(self):
        problems = verify({"status": "RECORDED", "snapshot_id": "s-1",
                           "access_decision": "SYNTHETIC_ALLOWED",
                           "accessed_at": AT})
        assert any("content digest" in one for one in problems)

    def test_a_missing_access_decision_is_caught(self):
        problems = verify({"status": "RECORDED", "snapshot_id": "s-1",
                           "content_digest": "mdv1:aaa", "accessed_at": AT})
        assert any("access decision" in one for one in problems)

    def test_a_missing_access_time_is_caught(self):
        problems = verify({"status": "RECORDED", "snapshot_id": "s-1",
                           "content_digest": "mdv1:aaa",
                           "access_decision": "SYNTHETIC_ALLOWED"})
        assert any("access time" in one for one in problems)

    def test_it_does_not_consult_the_environment(self, monkeypatch):
        """Verification asks whether the record is coherent, not whether it
        matches what this deployment is configured with today.

        The record deliberately names a *different* policy and snapshot from
        anything currently configured. Without that, an environment check
        added to `verify` would never evaluate and the falsification would
        pass — which is what happened the first time.
        """
        record = {"status": "RECORDED", "snapshot_id": "some-other-snapshot",
                  "content_digest": "mdv1:aaa",
                  "policy_version": "a-policy-this-deployment-does-not-use",
                  "access_decision": "SYNTHETIC_ALLOWED", "accessed_at": AT}
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        assert verify(record) == (), (
            "verification rejected a coherent record because it disagreed "
            "with the current deployment — a run made under an earlier policy "
            "is not thereby invalid")
        monkeypatch.delenv(POLICY, raising=False)
        assert verify(record) == ()


class TestNoDataMeansNoRecordedProvenance:
    def test_an_unconfigured_policy_records_the_absence(self, monkeypatch):
        monkeypatch.delenv(POLICY, raising=False)
        frame, provenance = resolve(context="a run", accessed_at=AT)
        assert frame is None
        assert provenance.status is ProvenanceStatus.NOT_RECORDED
        assert "policy" in provenance.access_decision_reason

    def test_an_unresolvable_snapshot_records_the_absence(self, monkeypatch):
        """Was written when the approved policy had no snapshot to resolve, so
        setting the policy was enough to reach this branch. A vendor snapshot
        exists now and resolves, so the condition has to be made rather than
        assumed — otherwise this test silently stops testing what it names."""
        import src.market_data.access as access

        monkeypatch.setenv(POLICY, "market-data-egress/pilot-vendor-approved@1")
        monkeypatch.setattr(access, "approved_snapshot", lambda: None)
        frame, provenance = resolve(context="a run", accessed_at=AT)
        assert frame is None
        assert provenance.status is ProvenanceStatus.NOT_RECORDED
        assert "no snapshot" in provenance.access_decision_reason

    def test_an_unloadable_snapshot_records_the_absence(self, monkeypatch):
        import src.market_data.loader as loader

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        monkeypatch.setattr(
            loader, "load_prices",
            lambda snapshot: (_ for _ in ()).throw(FileNotFoundError("gone")))
        frame, provenance = resolve(context="a run", accessed_at=AT)
        assert frame is None
        assert provenance.status is ProvenanceStatus.NOT_RECORDED
        assert "could not be loaded" in provenance.access_decision_reason

    def test_a_refused_snapshot_is_recorded_rather_than_absent(self, monkeypatch):
        """The distinction this class is easy to flatten: no frame is not one
        state. A snapshot nobody could resolve leaves nothing to record. A
        snapshot that resolved and was then refused leaves a decision, and a
        decision that records itself as an absence is a refusal nobody can
        later find.

        This branch used to be reached by accident — the vendor policy resolved
        no snapshot at all, so the test above landed here and asserted the
        wrong half. Reached deliberately now.
        """
        from src.market_data.provenance import AccessDecision
        import src.market_data.pilot_policy as pilot_policy
        from src.market_data.pilot_policy import PilotDataDenied

        monkeypatch.setenv(POLICY, "market-data-egress/pilot-vendor-approved@1")

        def refuse(snapshot, **kwargs):
            raise PilotDataDenied("refused for this test")

        monkeypatch.setattr(pilot_policy, "authorise", refuse)
        frame, provenance = resolve(context="a run", accessed_at=AT)
        assert frame is None
        assert provenance.status is ProvenanceStatus.RECORDED
        assert provenance.access_decision is AccessDecision.DENIED
        assert provenance.snapshot_id


class TestTheVendorSnapshotIsActuallyServable:
    """Every check on the vendor snapshot passed while no run could use it.

    `approved_snapshot()` returned it, the licensing record verified, and the
    disclosure rendered the attribution — three green checks around a snapshot
    whose manifest said `license_review_status: RESOLVED`, a word the loader
    does not know. `review_complete` was therefore False, the approved policy
    denied every run, and `egress_policy` was keyed by names no `Egress` value
    matches so every route took the DENY default.

    Both failures were closed-direction and silent. What none of the checks
    asked was the only question that decides whether the work is done: does a
    run come back holding prices.
    """

    def test_a_run_under_the_approved_policy_receives_prices(self, monkeypatch):
        from src.market_data.provenance import AccessDecision

        monkeypatch.setenv(POLICY, "market-data-egress/pilot-vendor-approved@1")
        frame, provenance = resolve(context="a run", accessed_at=AT)
        assert frame is not None and not frame.empty
        assert provenance.status is ProvenanceStatus.RECORDED
        assert provenance.access_decision is AccessDecision.PILOT_VENDOR_APPROVED

    def test_the_prices_are_the_snapshot_the_provenance_names(self, monkeypatch):
        """Otherwise the record is true about a snapshot and the figure came
        from somewhere else."""
        from src.market_data.access import approved_snapshot

        monkeypatch.setenv(POLICY, "market-data-egress/pilot-vendor-approved@1")
        frame, provenance = resolve(context="a run", accessed_at=AT)
        snapshot = approved_snapshot()
        assert provenance.snapshot_id == snapshot.snapshot_id
        assert len(frame) == int(snapshot.raw["sessions"])
        assert len(frame.columns) == int(snapshot.raw["assets"])

    def test_customer_results_are_permitted_and_exports_are_not(self, monkeypatch):
        """The egress block, read the way the code reads it. With the invented
        keys it had, every one of these was DENY — including the route the
        licensing record explicitly permits."""
        from src.market_data.access import approved_snapshot
        from src.market_data.loader import Decision, Egress

        monkeypatch.setenv(POLICY, "market-data-egress/pilot-vendor-approved@1")
        snapshot = approved_snapshot()
        assert snapshot.decision_for(Egress.CUSTOMER_RESULT) is Decision.ALLOW
        assert snapshot.decision_for(Egress.PUBLIC_EXPORT) is Decision.DENY
        assert snapshot.decision_for(Egress.CASE_BUNDLE) is Decision.DENY
        # "no automated path" in the record: the permission is for a person.
        assert snapshot.decision_for(Egress.MODEL_PROVIDER_UPLOAD) is Decision.DENY


class TestTheDataAndItsProvenanceComeTogether:
    def test_resolve_returns_one_object_carrying_both(self, monkeypatch):
        """A separate 'and also fetch the provenance' call is one a producer
        can forget, and the figure it forgot on looks like one it did not.

        A pair was the first shape; a single object is harder to split by
        accident as it is threaded through several functions, which is where
        the live path lost it.
        """
        from src.market_data.access import MarketDataAccess

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        access = resolve(context="a run", accessed_at=AT)
        assert isinstance(access, MarketDataAccess)
        assert access.usable
        assert access.provenance.status is ProvenanceStatus.RECORDED

    def test_it_still_unpacks_for_callers_that_want_one_half(self,
                                                             monkeypatch):
        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        frame, provenance = resolve(context="a run", accessed_at=AT)
        assert frame is not None and provenance is not None

    def test_the_prices_only_helper_still_exists_for_non_storing_callers(
            self, monkeypatch):
        from src.market_data.access import resolve_prices

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        assert resolve_prices(context="a page render") is not None


class TestAManifestWrittenInAnUnknownVocabularyIsRejected:
    """Both defects above were introduced by hand, in a file that looked
    plausible, and neither produced a message. The manifest is the interface
    between a licensing decision and the code that enforces it, and an
    interface where a typo means "deny everything, quietly" cannot be
    proofread into correctness.

    These stay fatal rather than defaulting, because the two failure modes are
    not symmetric: a rejected manifest stops a deploy, and an accepted one that
    means nothing stops every run for a reason nobody can find.
    """

    def _manifest(self, tmp_path, **overrides):
        import yaml
        body = {
            "dataset_id": "market-data/prices",
            "snapshot_id": "s-1",
            "kind": "vendor",
            "schema_version": "1",
            "license_review_status": "CONFIRMED",
            "egress_policy": {"customer_result": "ALLOW"},
        }
        body.update(overrides)
        path = tmp_path / "m.yaml"
        path.write_text(yaml.safe_dump(body))
        return path

    def test_a_valid_manifest_still_loads(self, tmp_path):
        """The discriminating half. A check that rejects everything proves
        nothing about the check."""
        from src.market_data.loader import load_manifest

        snapshot = load_manifest(self._manifest(tmp_path))
        assert snapshot.review_complete

    def test_the_review_status_typo_is_named(self, tmp_path):
        """`RESOLVED` — the word actually written, which read as an open
        review and denied every run."""
        from src.market_data.loader import load_manifest

        with pytest.raises(ValueError) as raised:
            load_manifest(self._manifest(tmp_path,
                                         license_review_status="RESOLVED"))
        assert "RESOLVED" in str(raised.value)
        assert "CONFIRMED" in str(raised.value)

    def test_an_invented_egress_route_is_named(self, tmp_path):
        from src.market_data.loader import load_manifest

        with pytest.raises(ValueError) as raised:
            load_manifest(self._manifest(
                tmp_path,
                egress_policy={"derived_results_to_end_users": "ALLOW"}))
        assert "derived_results_to_end_users" in str(raised.value)
        assert "customer_result" in str(raised.value)

    def test_a_boolean_where_a_decision_belongs_is_named(self, tmp_path):
        """`derived_results_to_end_users: true` was meant to permit a route.
        Read as a policy it is not a decision at all."""
        from src.market_data.loader import load_manifest

        with pytest.raises(ValueError) as raised:
            load_manifest(self._manifest(
                tmp_path, egress_policy={"customer_result": True}))
        assert "customer_result" in str(raised.value)
        assert "ALLOW" in str(raised.value)

    def test_an_omitted_route_is_still_a_refusal_not_an_error(self, tmp_path):
        """Silence must keep meaning DENY. If omission became an error, every
        manifest would have to list every route, and a route added later would
        break files that were never wrong."""
        from src.market_data.loader import Decision, Egress, load_manifest

        snapshot = load_manifest(self._manifest(tmp_path))
        assert snapshot.decision_for(Egress.PUBLIC_EXPORT) is Decision.DENY
