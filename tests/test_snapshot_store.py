"""The descriptor store, and the classification that lets it carry no owner.

Step 8, metadata. `market_snapshot` is the first `SHARED_REFERENCE` table: it
describes the world rather than a user, so it holds no tenant column at all —
a rule stricter than the tenant one rather than an exemption from it.

The two-hash split is what this store exists to express. Keyed by
`descriptor_hash`, indexed by `snapshot_hash`, so one set of observations can
carry several descriptions over time — a licence re-reviewed, an adapter
corrected — without any of it looking like the market moved.
"""
from __future__ import annotations

import dataclasses
import os

import pytest

from src.db.mutability import TABLE_MUTABILITY, Ownership
from src.db.tenancy import (TENANT_COLUMNS, reference_violations,
                            shared_reference_tables, tenant_owned_tables)
from src.market_data.snapshot_contract import SourceAdapter, describe
from src.market_data.snapshot_store import (DescriptorConflict, descriptor,
                                            descriptors_for, record)


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    from src.db import migrate
    from src.db.engine import Database
    from src.deploy import context as deploy_context

    url = f"sqlite:///{tmp_path}/snapshots.db"
    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", url)
    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)
    migrate.upgrade(Database(url))


def a_snapshot(*, reinvested: bool = False, review: str = "CONFIRMED"):
    from src.market_data.access import resolve
    from src.market_data.loader import synthetic_snapshot

    access = resolve(context="store", reinvested=reinvested)
    built = describe(synthetic_snapshot(), access.frame,
                     resolution={"reinvested": reinvested, "version": "mdr1"},
                     adapter=SourceAdapter("local-parquet", "1"))
    return dataclasses.replace(built, license_review_status=review)


class TestTheClassificationIsRealAndStricter:
    def test_the_table_is_declared_shared_reference(self):
        assert TABLE_MUTABILITY["market_snapshot"].ownership \
            is Ownership.SHARED_REFERENCE
        assert "market_snapshot" in shared_reference_tables()
        assert "market_snapshot" not in tenant_owned_tables()

    def test_a_shared_table_carrying_an_owner_is_a_violation(self):
        """The stricter half, exercised on the shape it forbids.

        Declaring a table shared must not be a way to opt out of tenancy — it
        is a different and harder rule, and a check that never fires would make
        it the easy escape hatch instead.
        """
        found = reference_violations(
            {"market_snapshot": [{"name": "descriptor_hash"},
                                 {"name": "owner"}]})
        assert found, "a reference table with an `owner` column passed"
        assert "owner" in found[0].detail

    def test_the_real_schema_carries_no_tenant_column(self):
        from src.db.schema import metadata

        columns = {c.name for c in metadata.tables["market_snapshot"].columns}
        assert not (columns & set(TENANT_COLUMNS)), (
            f"{sorted(columns & set(TENANT_COLUMNS))} identify somebody, in a "
            "table declared to describe the world")

    def test_ownership_defaults_to_tenant_owned(self):
        """The safe direction: a table nobody classified is checked rather than
        exempted."""
        from src.db.mutability import Mutability, TableClass

        assert Mutability(table="x", kind=TableClass.IMMUTABLE_ARTIFACT
                          ).ownership is Ownership.TENANT_OWNED


class TestOneSetOfBytesManyDescriptions:
    def test_a_licence_re_review_is_a_new_descriptor_not_a_new_snapshot(self):
        """The separation, doing the job it was drawn for."""
        first = a_snapshot(review="CONFIRMED")
        second = a_snapshot(review="RECONFIRMED")

        assert first.snapshot_hash == second.snapshot_hash
        assert first.descriptor_hash != second.descriptor_hash

        record(first, recorded_at="2026-08-15T00:00:00Z")
        record(second, recorded_at="2026-08-16T00:00:00Z")

        history = descriptors_for(first.snapshot_hash)
        assert len(history) == 2
        assert [one["license_review_status"] for one in history] == \
            ["CONFIRMED", "RECONFIRMED"]

    def test_the_observations_identity_never_moves(self):
        first = a_snapshot(review="CONFIRMED")
        record(first, recorded_at="2026-08-15T00:00:00Z")
        record(a_snapshot(review="RECONFIRMED"),
               recorded_at="2026-08-16T00:00:00Z")

        for one in descriptors_for(first.snapshot_hash):
            assert one["snapshot_hash"] == first.snapshot_hash

    def test_a_different_request_is_a_different_snapshot(self):
        price = a_snapshot(reinvested=False)
        total = a_snapshot(reinvested=True)
        record(price, recorded_at="2026-08-15T00:00:00Z")
        record(total, recorded_at="2026-08-15T00:00:01Z")

        assert descriptors_for(price.snapshot_hash) != \
            descriptors_for(total.snapshot_hash)
        assert len(descriptors_for(price.snapshot_hash)) == 1


class TestRecordingIsIdempotentByAddress:
    def test_recording_the_same_descriptor_twice_says_the_same_thing(self):
        one = a_snapshot()
        assert record(one, recorded_at="2026-08-15T00:00:00Z") \
            == record(one, recorded_at="2026-08-16T00:00:00Z")
        assert len(descriptors_for(one.snapshot_hash)) == 1

    def test_a_descriptor_address_holding_a_different_body_is_a_conflict(self):
        """Exactly one of them can be right, so neither is written through.

        Picking a side would silently rewrite what somebody recorded about a
        licence or an adapter.
        """
        one = a_snapshot()
        record(one, recorded_at="2026-08-15T00:00:00Z")

        impostor = dataclasses.replace(one, snapshot_hash="mdf1:something-else")
        impostor = dataclasses.replace(impostor)
        with pytest.raises(DescriptorConflict):
            _forced_write(impostor, one.descriptor_hash)


def _forced_write(snapshot, address: str):
    """Write a body under an address that is not its own.

    Only a test can construct this — the address is derived from the body — and
    it is what the conflict check exists for: a collision, or a bug that
    computed the address from something other than what it stored.
    """
    class _AtAddress:
        def __init__(self, inner, at):
            self._inner, self.descriptor_hash = inner, at
            self.snapshot_hash = inner.snapshot_hash

        def to_json(self):
            return self._inner.to_json()

    return record(_AtAddress(snapshot, address),
                  recorded_at="2026-08-17T00:00:00Z")


class TestReadingBack:
    def test_a_descriptor_round_trips_through_the_store(self):
        from src.market_data.snapshot_contract import from_json

        one = a_snapshot(reinvested=True)
        record(one, recorded_at="2026-08-15T00:00:00Z")
        assert from_json(descriptor(one.descriptor_hash)) == one

    def test_an_unknown_descriptor_is_absent_rather_than_an_error(self):
        assert descriptor("mds1:nothing") is None

    def test_the_request_survives_storage(self):
        """The field that determines which bytes exist. A store that dropped it
        would hold descriptions of observations nobody could ask for again."""
        one = a_snapshot(reinvested=True)
        record(one, recorded_at="2026-08-15T00:00:00Z")
        assert descriptor(one.descriptor_hash)["resolution"]["reinvested"] is True
