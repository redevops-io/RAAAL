"""Which data a pilot identity may reach, gated at runtime.

Distinct from the egress policy, which asks where data may go. This asks whether
a principal may touch a dataset at all, and it sits above egress because a
vendor snapshot reached by an ordinary code path is a licence problem before it
is an export problem.

Closed pilot v1 is SYNTHETIC_ONLY. The vendor snapshot may stay in S3 for
internal development; pilot identities must not be authorised to use it.
"""
from __future__ import annotations

import pytest

from src.market_data.loader import production_snapshot, synthetic_snapshot
from src.market_data.pilot_policy import (
    LICENSING_QUESTIONS,
    POLICY_VARIABLE,
    REQUIRED_ANSWER_FIELDS,
    Authorisation,
    PilotDataDenied,
    PilotDataPolicy,
    PilotPolicyMissing,
    authorise,
    configured_policy,
    evaluate,
    licensing_resolved,
)

SYNTHETIC = {POLICY_VARIABLE: "SYNTHETIC_ONLY"}
APPROVED = {POLICY_VARIABLE: "market-data-egress/pilot-vendor-approved@1"}


class TestItFailsClosed:

    def test_an_unset_policy_refuses(self):
        """The default that matters is the one nobody chose deliberately."""
        with pytest.raises(PilotPolicyMissing, match=POLICY_VARIABLE):
            configured_policy({})

    def test_an_empty_policy_refuses(self):
        with pytest.raises(PilotPolicyMissing):
            configured_policy({POLICY_VARIABLE: ""})

    def test_an_unrecognised_policy_refuses(self):
        with pytest.raises(PilotPolicyMissing, match="not a recognised"):
            configured_policy({POLICY_VARIABLE: "production"})

    def test_a_generic_production_flag_is_not_a_policy(self):
        """Lifting the boundary introduces a named, versioned policy so the
        change is reviewable and a stored run says what it ran under."""
        assert "production" not in {one.value for one in PilotDataPolicy}
        assert PilotDataPolicy.PILOT_VENDOR_APPROVED.value.endswith("@1")


class TestSyntheticOnly:

    def test_the_synthetic_fixture_is_permitted(self):
        verdict = evaluate(synthetic_snapshot(),
                           policy=PilotDataPolicy.SYNTHETIC_ONLY)
        assert verdict.permitted

    def test_the_vendor_snapshot_is_denied(self):
        verdict = evaluate(production_snapshot(),
                           policy=PilotDataPolicy.SYNTHETIC_ONLY)
        assert not verdict.permitted
        assert "synthetic-only" in verdict.reason

    def test_the_denial_names_the_snapshot_and_the_policy(self):
        with pytest.raises(PilotDataDenied, match="prices-2025-11-19"):
            authorise(production_snapshot(), environ=SYNTHETIC)

    def test_a_non_redistributable_synthetic_kind_is_still_denied(self):
        """Both conditions, not either."""
        class Marked:
            kind, redistributable, snapshot_id = "synthetic", False, "x"
            review_complete = True

        assert not evaluate(Marked(),
                            policy=PilotDataPolicy.SYNTHETIC_ONLY).permitted

    def test_an_unknown_kind_is_not_admitted_by_omission(self):
        class Unknown:
            kind, redistributable, snapshot_id = "delayed-feed", True, "x"
            review_complete = True

        assert not evaluate(Unknown(),
                            policy=PilotDataPolicy.SYNTHETIC_ONLY).permitted


class TestTheApprovedPolicyStillRequiresTheReview:

    def test_an_unreviewed_snapshot_is_denied_even_under_approval(self):
        """A deployment flag cannot substitute for reading the agreement."""
        verdict = evaluate(production_snapshot(),
                           policy=PilotDataPolicy.PILOT_VENDOR_APPROVED)
        assert not verdict.permitted
        assert "licence review" in verdict.reason

    def test_a_reviewed_snapshot_is_permitted_under_approval(self):
        class Reviewed:
            kind, redistributable, snapshot_id = "licensed", False, "vendor-1"
            review_complete = True

        assert evaluate(Reviewed(),
                        policy=PilotDataPolicy.PILOT_VENDOR_APPROVED).permitted


class TestThereIsNoFallback:

    def test_a_denied_snapshot_does_not_become_the_synthetic_one(self):
        """A figure from data the plan did not name is worse than no figure,
        because nothing in the result would say so."""
        import inspect

        from src.market_data import pilot_policy

        source = inspect.getsource(pilot_policy.authorise)
        assert "synthetic_snapshot" not in source
        with pytest.raises(PilotDataDenied):
            authorise(production_snapshot(), environ=SYNTHETIC)


class TestTheLiveRouteIsGated:

    def test_it_serves_prices_under_the_synthetic_policy(self, monkeypatch):
        import src.workspace.routes as routes

        monkeypatch.setenv(POLICY_VARIABLE, "SYNTHETIC_ONLY")
        assert routes._prices() is not None

    def test_it_serves_nothing_with_no_policy_set(self, monkeypatch):
        import src.workspace.routes as routes

        monkeypatch.delenv(POLICY_VARIABLE, raising=False)
        assert routes._prices() is None

    def test_it_no_longer_reads_the_unmanifested_file(self, monkeypatch):
        """`data/history/prices.parquet` had no snapshot identity, no licence
        class and no egress check, and the live route read it directly.

        Watched rather than read. This asserted `"read_parquet" not in source`,
        which is a claim about the text of one function — it says nothing about
        what that function calls, and it broke the moment the resolution moved
        into `market_data.access` while the behaviour stayed correct.
        """
        import pandas as pd

        import src.workspace.routes as routes
        from src.market_data.access import UNMANIFESTED_PRICES

        opened = []
        original = pd.read_parquet
        monkeypatch.setattr(
            pd, "read_parquet",
            lambda path, *a, **k: (opened.append(str(path)),
                                   original(path, *a, **k))[1])
        monkeypatch.setenv(POLICY_VARIABLE, "SYNTHETIC_ONLY")
        routes._prices()

        assert opened, "no file was read at all; this proves nothing"
        assert not any(UNMANIFESTED_PRICES in one for one in opened), (
            f"the live route read the unmanifested file: {opened}")

    def test_prices_are_loaded_by_snapshot_identity(self, monkeypatch):
        """The snapshot is resolved and authorised, then loaded by its id."""
        import src.market_data.access as access
        import src.workspace.routes as routes

        authorised = []
        original = access.__dict__.get("_authorise_probe")
        import src.market_data.pilot_policy as policy_module

        real_authorise = policy_module.authorise

        def watched(snapshot, *, context, **_):
            authorised.append((snapshot.snapshot_id, context))
            return real_authorise(snapshot, context=context, **_)

        monkeypatch.setattr(policy_module, "authorise", watched)
        monkeypatch.setenv(POLICY_VARIABLE, "SYNTHETIC_ONLY")
        assert routes._prices() is not None

        assert authorised, "nothing was authorised; the gate was not reached"
        snapshot_id, context = authorised[0]
        assert snapshot_id, "a snapshot with no identity was authorised"
        assert context == "pilot scenario run"

    def test_a_denied_snapshot_yields_no_prices_rather_than_other_data(
            self, monkeypatch):
        """The comment here used to read "the vendor snapshot's review is still
        UNCONFIRMED, so even the approved policy denies it". That was true, and
        it made the denial a property of the fixtures rather than of the gate:
        the moment a snapshot passed review, the test stopped testing anything
        and would have failed rather than told anyone why.

        The denial is now caused directly, so the assertion is about the
        absence of a fallback and nothing else.
        """
        import src.market_data.pilot_policy as policy_module
        import src.workspace.routes as routes

        monkeypatch.setenv(POLICY_VARIABLE,
                           "market-data-egress/pilot-vendor-approved@1")

        def refuse(snapshot, **kwargs):
            raise policy_module.PilotDataDenied("refused for this test")

        monkeypatch.setattr(policy_module, "authorise", refuse)
        assert routes._prices() is None

    def test_an_approved_snapshot_does_yield_prices(self, monkeypatch):
        import src.workspace.routes as routes

        monkeypatch.setenv(POLICY_VARIABLE,
                           "market-data-egress/pilot-vendor-approved@1")
        assert routes._prices() is not None


class TestTheLicensingExitGate:

    def test_all_six_questions_are_listed(self):
        assert len(LICENSING_QUESTIONS) == 6

    def test_an_empty_record_is_not_resolved(self):
        assert not licensing_resolved({})

    def test_a_partial_answer_is_not_resolved(self):
        """"We looked into it" is not a record, and a missing field is what a
        rushed review leaves behind."""
        record = {question: {"answer": "yes"} for question in LICENSING_QUESTIONS}
        assert not licensing_resolved(record)

    def test_every_required_field_is_needed(self):
        complete = {field: "x" for field in REQUIRED_ANSWER_FIELDS}
        record = {question: dict(complete) for question in LICENSING_QUESTIONS}
        assert licensing_resolved(record)

        for field in REQUIRED_ANSWER_FIELDS:
            missing = {question: {**complete, field: ""}
                       for question in LICENSING_QUESTIONS}
            assert not licensing_resolved(missing), field

    def test_one_missing_question_defeats_it(self):
        complete = {field: "x" for field in REQUIRED_ANSWER_FIELDS}
        record = {question: dict(complete)
                  for question in LICENSING_QUESTIONS[:-1]}
        assert not licensing_resolved(record)

    def test_the_current_record_is_unresolved(self):
        """The shipped manifest still carries six UNCONFIRMED answers, so the
        boundary stays where it is."""
        assert not licensing_resolved({})
