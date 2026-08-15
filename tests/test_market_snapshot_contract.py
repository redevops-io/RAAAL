"""The snapshot contract, and the invariant everything downstream rests on.

Step 8. `MarketSnapshot` is immutable and content-addressed, and the hard
invariant is that resolving hash `H` returns the canonical observations whose
digest is `H`.

Three properties, each of which has already failed once somewhere in this
system:

  * **the request is part of the identity.** The same dataset resolved with
    dividends reinvested returns the total-return twin instead of the price
    series — different bytes, same dataset. A snapshot naming only the dataset
    does not determine what arrives, which is the defect that started this
    whole thread.
  * **immutable and independently serialized are separate.** A frozen
    dataclass handing back the objects it holds is immutable only in the sense
    that its fields cannot be reassigned. An `EvaluationResult` did this, and
    the mutation test that should have caught a changed field watched both
    sides change together.
  * **nothing is inferred.** Corporate-action treatment, calendar, adapter and
    its version are declared or explicitly `NOT_DECLARED` — a snapshot silent
    about corporate actions is one somebody assumes handled them.
"""
from __future__ import annotations

import os

import pytest

from src.market_data.snapshot_contract import (NOT_DECLARED, PRICE_ONLY,
                                               SNAPSHOT_CONTRACT_VERSION,
                                               TOTAL_RETURN, MarketSnapshot,
                                               SourceAdapter, describe,
                                               from_json)


@pytest.fixture(autouse=True)
def deployment(monkeypatch):
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)


def delivered(*, reinvested: bool = False):
    """One real delivery, and the snapshot describing it."""
    from src.market_data.access import resolve

    access = resolve(context="snapshot-contract", reinvested=reinvested)
    snapshot = describe(
        _source(), access.frame,
        resolution={"reinvested": reinvested, "version": "mdr1"},
        adapter=SourceAdapter("local-parquet", "1"))
    return access, snapshot


def _source():
    from src.market_data.loader import synthetic_snapshot

    return synthetic_snapshot()


class TestTheHardInvariant:
    def test_resolving_the_hash_returns_the_bytes_it_names(self):
        """Asked of the bytes, not of a stored copy of the same claim."""
        access, snapshot = delivered()
        assert snapshot.verify(access.frame) == ()

    def test_it_notices_different_bytes(self):
        """Without this, `verify` passes on anything and the invariant is a
        sentence in a docstring."""
        access, snapshot = delivered()
        moved = access.frame.copy()
        moved.iloc[0, 0] = float(moved.iloc[0, 0]) + 1.0

        problems = snapshot.verify(moved)
        assert problems, "an altered frame verified against its own snapshot"
        assert "digest" in problems[0]

    def test_it_notices_missing_symbols(self):
        access, snapshot = delivered()
        fewer = access.frame.drop(columns=[access.frame.columns[0]])
        problems = snapshot.verify(fewer)
        assert any("symbols" in one for one in problems)

    def test_it_notices_a_different_span(self):
        access, snapshot = delivered()
        shorter = access.frame.iloc[:-5]
        problems = snapshot.verify(shorter)
        assert any("sessions" in one for one in problems)

    def test_no_observations_is_a_problem_and_not_a_pass(self):
        _access, snapshot = delivered()
        assert snapshot.verify(None), (
            "a snapshot verified against nothing, so an unresolvable snapshot "
            "would report as intact")


class TestTheRequestIsPartOfTheIdentity:
    def test_reinvested_and_not_are_different_snapshots(self):
        """The defect that started the data-lake thread, now structural.

        Same dataset, same sessions, same symbols — and different observations,
        because one is the total-return twin. A snapshot identity naming only
        the dataset would call these one thing.
        """
        _price, price_only = delivered(reinvested=False)
        _total, total_return = delivered(reinvested=True)

        assert price_only.dataset_id == total_return.dataset_id
        assert price_only.snapshot_hash != total_return.snapshot_hash
        assert price_only.descriptor_hash != total_return.descriptor_hash

    def test_the_corporate_action_treatment_follows_the_request(self):
        """Read from what arrived rather than assumed. A reinvested resolution
        is served by a series in which distributions are already credited."""
        _a, price_only = delivered(reinvested=False)
        _b, total_return = delivered(reinvested=True)

        assert price_only.corporate_actions == PRICE_ONLY
        assert total_return.corporate_actions == TOTAL_RETURN

    def test_the_request_survives_into_the_record(self):
        _access, snapshot = delivered(reinvested=True)
        assert snapshot.resolution["reinvested"] is True


class TestImmutableIsNotTheSameAsIndependentlySerialized:
    """The Step 7 lesson, designed in rather than rediscovered."""

    def test_editing_the_wire_body_does_not_edit_the_snapshot(self):
        _access, snapshot = delivered(reinvested=True)
        body = snapshot.to_json()

        body["resolution"]["reinvested"] = False
        body["symbols"].append("INVENTED")
        body["session_range"]["sessions"] = 1

        assert snapshot.resolution["reinvested"] is True
        assert "INVENTED" not in snapshot.symbols
        assert snapshot.session_range.sessions != 1

    def test_the_fields_cannot_be_reassigned_either(self):
        """Both halves. Frozen alone was never the property that mattered, and
        neither is copying alone."""
        import dataclasses

        _access, snapshot = delivered()
        with pytest.raises(dataclasses.FrozenInstanceError):
            snapshot.snapshot_hash = "mdf1:something-else"

    def test_it_round_trips(self):
        _access, snapshot = delivered(reinvested=True)
        assert from_json(snapshot.to_json()) == snapshot

    def test_a_record_missing_a_field_is_refused(self):
        """A reader built by splatting a dict would accept a record that
        described less than it claimed, and a field that does not survive the
        wire silently stops being part of the identity."""
        _access, snapshot = delivered()
        body = snapshot.to_json()
        del body["corporate_actions"]
        with pytest.raises(KeyError):
            from_json(body)


class TestNothingIsInferred:
    def test_an_undeclared_adapter_says_so(self):
        """Rather than naming one. A default would attribute this data to code
        that may not have produced it, which is worse than admitting nobody
        wrote it down."""
        snapshot = describe(_source(), delivered()[0].frame,
                            resolution={"reinvested": False})
        assert snapshot.source_adapter.name == NOT_DECLARED
        assert snapshot.source_adapter.version == NOT_DECLARED

    def test_a_declared_adapter_travels(self):
        _access, snapshot = delivered()
        assert snapshot.source_adapter.name == "local-parquet"
        assert snapshot.source_adapter.version == "1"

    def test_the_licence_and_calendar_come_from_the_source(self):
        _access, snapshot = delivered()
        source = _source()
        assert snapshot.license_class == source.license_class
        assert snapshot.license_review_status == source.license_review_status
        assert snapshot.calendar == source.calendar

    def test_every_declared_field_is_populated_or_explicitly_absent(self):
        """No blanks. An empty string reads as "nothing to say"; NOT_DECLARED
        reads as "nobody established it", and only one of those is true."""
        _access, snapshot = delivered()
        for name, value in snapshot.to_json().items():
            if isinstance(value, str):
                assert value, f"{name} is blank rather than {NOT_DECLARED}"


class TestTheDescriptorAndTheDataHaveSeparateAddresses:
    def test_redescribing_the_same_bytes_keeps_the_content_address(self):
        """A licence correction is not a change to the market.

        Collapsing the two addresses would make re-reviewing a licence look
        like the observations had moved.
        """
        import dataclasses

        access, snapshot = delivered()
        corrected = dataclasses.replace(snapshot,
                                        license_review_status="RECONFIRMED")

        assert corrected.snapshot_hash == snapshot.snapshot_hash
        assert corrected.descriptor_hash != snapshot.descriptor_hash
        assert corrected.verify(access.frame) == ()

    def test_the_contract_version_is_in_the_descriptor(self):
        _access, snapshot = delivered()
        assert snapshot.version == SNAPSHOT_CONTRACT_VERSION
        assert SNAPSHOT_CONTRACT_VERSION in str(snapshot.to_json())


class TestItNamesTheSameBytesAsTheDeliveryRecord:
    """The snapshot contract and the access event must agree, or they are two
    claims about one delivery and nothing says which is right.

    `MarketDataAccessEvent.frame_digest` already records what a run consumed.
    `MarketSnapshot.snapshot_hash` records what a snapshot *is*. If those could
    differ for one delivery, a run could cite a snapshot it did not read.
    """

    def test_the_snapshot_hash_is_the_delivered_frames_digest(self):
        access, snapshot = delivered()
        assert access.access_event is not None
        assert snapshot.snapshot_hash == access.access_event.frame_digest

    def test_and_it_still_holds_for_the_total_return_twin(self):
        """The pair that used to be indistinguishable. Both the event and the
        snapshot must move together when the request changes, or one of them
        is describing the other's data."""
        access, snapshot = delivered(reinvested=True)
        assert snapshot.snapshot_hash == access.access_event.frame_digest

        price_access, price_snapshot = delivered(reinvested=False)
        assert price_snapshot.snapshot_hash != snapshot.snapshot_hash
        assert (price_access.access_event.frame_digest
                != access.access_event.frame_digest)

    def test_the_recorded_request_matches_the_one_in_the_snapshot(self):
        access, snapshot = delivered(reinvested=True)
        recorded = access.access_event.resolution
        assert recorded is not None
        assert bool(recorded.reinvested) is bool(snapshot.resolution["reinvested"])
