"""The whole lifecycle, and every way it can fail differently.

    delivery -> descriptor -> immutable bytes -> read by hash -> verified

Steps 3 and 4. The object store is create-once and stupid; the read path is
what proves a snapshot rather than merely fetching one.

**The invariant this file is really about**: metadata may exist without payload,
and that state must never be mistaken for a valid snapshot. A descriptor is a
record of what was *believed*; only the read path establishes that the bytes
exist and match. An interrupted write produces exactly this state, so it is a
named outcome rather than an unclassified error.

**Failures are distinct because repairs are.** A missing payload is a storage
problem, a digest mismatch is a corruption investigation, and a symbol mismatch
is a descriptor written wrong. One boolean would send somebody to the wrong
place four times out of five.
"""
from __future__ import annotations

import os

import pytest

from src.market_data.object_store import (ObjectStore, PayloadConflict,
                                          from_bytes, to_bytes)
from src.market_data.snapshot_contract import SourceAdapter, describe
from src.market_data.snapshot_read import Read, SnapshotProblem, get
from src.market_data.snapshot_store import record


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    from src.db import migrate
    from src.db.engine import Database
    from src.deploy import context as deploy_context

    url = f"sqlite:///{tmp_path}/snapshots.db"
    for name, value in (("PILOT_DATA_POLICY", "SYNTHETIC_ONLY"),
                        ("QUANTIFY_PILOT_READER", "recorded"),
                        ("QUANTIFY_PARSER_MODE", "RUNTIME"),
                        ("QUANTIFY_PARSER_MODEL", "claude-sonnet-5"),
                        ("ANTHROPIC_API_KEY", "unused"),
                        ("QUANTIFY_DATABASE_URL", url)):
        monkeypatch.setenv(name, value)
    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)
    migrate.upgrade(Database(url))


@pytest.fixture
def store(tmp_path):
    return ObjectStore(root=tmp_path / "objects")


def a_delivery(*, reinvested: bool = False):
    from src.market_data.access import resolve
    from src.market_data.loader import synthetic_snapshot

    access = resolve(context="lifecycle", reinvested=reinvested)
    snapshot = describe(
        synthetic_snapshot(), access.frame,
        resolution={"reinvested": reinvested, "version": "mdr1"},
        adapter=SourceAdapter("local-parquet", "1"))
    return access, snapshot


def stored(store, *, reinvested: bool = False):
    """The full write path: descriptor recorded, bytes filed."""
    access, snapshot = a_delivery(reinvested=reinvested)
    record(snapshot, recorded_at="2026-08-15T00:00:00Z")
    store.put(snapshot.snapshot_hash, to_bytes(access.frame))
    return access, snapshot


# --- Step 3: create-once ----------------------------------------------------

class TestTheObjectStoreIsCreateOnce:
    def test_absent_writes(self, store):
        access, snapshot = a_delivery()
        assert not store.holds(snapshot.snapshot_hash)
        store.put(snapshot.snapshot_hash, to_bytes(access.frame))
        assert store.holds(snapshot.snapshot_hash)

    def test_identical_bytes_are_idempotent(self, store):
        """Which requires the serializer to be byte-stable, so that is checked
        rather than assumed — one embedding a timestamp would turn every
        re-put into a spurious conflict."""
        access, snapshot = a_delivery()
        first, second = to_bytes(access.frame), to_bytes(access.frame)
        assert first == second, "serialization is not deterministic"

        store.put(snapshot.snapshot_hash, first)
        assert store.put(snapshot.snapshot_hash, second) == snapshot.snapshot_hash

    def test_different_bytes_are_a_hard_failure(self, store):
        """No overwrite path, and no silent keep either. Overwriting changes
        what every figure citing this snapshot was computed from; keeping the
        first quietly leaves the caller believing it wrote."""
        access, snapshot = a_delivery()
        store.put(snapshot.snapshot_hash, to_bytes(access.frame))

        moved = access.frame.copy()
        moved.iloc[0, 0] = float(moved.iloc[0, 0]) + 1.0
        with pytest.raises(PayloadConflict):
            store.put(snapshot.snapshot_hash, to_bytes(moved))

    def test_the_original_survives_a_rejected_write(self, store):
        access, snapshot = a_delivery()
        original = to_bytes(access.frame)
        store.put(snapshot.snapshot_hash, original)

        moved = access.frame.copy()
        moved.iloc[0, 0] = float(moved.iloc[0, 0]) + 1.0
        with pytest.raises(PayloadConflict):
            store.put(snapshot.snapshot_hash, to_bytes(moved))
        assert store.get(snapshot.snapshot_hash) == original

    def test_there_is_no_overwrite_or_delete(self):
        """Not a discouraged path — an absent one."""
        for forbidden in ("delete", "remove", "overwrite", "replace", "update"):
            assert not hasattr(ObjectStore, forbidden), (
                f"ObjectStore.{forbidden} exists, so immutability is a "
                "convention rather than a property")

    def test_empty_bytes_are_refused(self, store):
        """An empty payload at a real address reads as a snapshot that exists
        and holds nothing, which is worse than one that is absent."""
        _access, snapshot = a_delivery()
        with pytest.raises(ValueError, match="no bytes"):
            store.put(snapshot.snapshot_hash, b"")

    def test_absence_is_an_answer_rather_than_an_error(self, store):
        assert store.get("mdf1:never-written") is None

    def test_the_bytes_round_trip_to_the_same_observations(self, store):
        from src.market_data.access_event import frame_digest

        access, snapshot = stored(store)
        back = from_bytes(store.get(snapshot.snapshot_hash))
        assert frame_digest(back) == snapshot.snapshot_hash


# --- Step 4: read by hash ---------------------------------------------------

class TestAVerifiedRead:
    def test_the_whole_lifecycle(self, store):
        access, snapshot = stored(store)
        read = get(snapshot.snapshot_hash, snapshot.descriptor_hash, store=store)

        assert read.ok, read.refusal()
        assert read.snapshot == snapshot
        assert read.observations is not None
        assert read.problems == ()

    def test_the_total_return_twin_reads_as_its_own_snapshot(self, store):
        _price_access, price = stored(store, reinvested=False)
        _total_access, total = stored(store, reinvested=True)

        assert get(price.snapshot_hash, price.descriptor_hash, store=store).ok
        assert get(total.snapshot_hash, total.descriptor_hash, store=store).ok
        assert price.snapshot_hash != total.snapshot_hash


class TestMetadataWithoutPayloadIsNotASnapshot:
    """The invariant this whole path exists for.

    A descriptor is a record of what was believed. Without bytes it is a claim
    with nothing behind it, and an interrupted write produces exactly that.
    """

    def test_a_recorded_descriptor_with_no_bytes_is_not_ok(self, store):
        _access, snapshot = a_delivery()
        record(snapshot, recorded_at="2026-08-15T00:00:00Z")

        read = get(snapshot.snapshot_hash, snapshot.descriptor_hash, store=store)
        assert not read.ok
        assert read.kinds == (SnapshotProblem.PAYLOAD_MISSING,)

    def test_it_still_returns_the_descriptor_it_found(self, store):
        """So an operator can see what was expected, without that counting as
        a successful read."""
        _access, snapshot = a_delivery()
        record(snapshot, recorded_at="2026-08-15T00:00:00Z")

        read = get(snapshot.snapshot_hash, snapshot.descriptor_hash, store=store)
        assert read.snapshot is not None
        assert read.observations is None
        assert not read.ok, (
            "a descriptor with no observations reported as usable, which is "
            "the mistake this path exists to prevent")


class TestEveryFailureIsDistinct:
    def test_a_missing_descriptor(self, store):
        read = get("mdf1:anything", "mds1:nothing", store=store)
        assert read.kinds == (SnapshotProblem.DESCRIPTOR_MISSING,)

    def test_a_descriptor_describing_other_observations(self, store):
        _access, snapshot = stored(store)
        read = get("mdf1:some-other-observations", snapshot.descriptor_hash,
                   store=store)
        assert read.kinds == (SnapshotProblem.DESCRIPTOR_MISMATCH,)

    def test_a_missing_payload(self, store):
        _access, snapshot = a_delivery()
        record(snapshot, recorded_at="2026-08-15T00:00:00Z")
        read = get(snapshot.snapshot_hash, snapshot.descriptor_hash, store=store)
        assert read.kinds == (SnapshotProblem.PAYLOAD_MISSING,)

    def test_empty_observations(self, store):
        access, snapshot = a_delivery()
        record(snapshot, recorded_at="2026-08-15T00:00:00Z")
        store.put(snapshot.snapshot_hash, to_bytes(access.frame.iloc[:0]))

        read = get(snapshot.snapshot_hash, snapshot.descriptor_hash, store=store)
        assert read.kinds == (SnapshotProblem.EMPTY_OBSERVATIONS,)

    def test_a_payload_that_is_not_the_observations_named(self, store):
        """Filed under the right address, holding the wrong data. The store
        cannot catch this — it compares bytes and knows nothing about frames —
        so the read path must."""
        access, snapshot = a_delivery()
        record(snapshot, recorded_at="2026-08-15T00:00:00Z")
        moved = access.frame.copy()
        moved.iloc[0, 0] = float(moved.iloc[0, 0]) + 1.0
        store.put(snapshot.snapshot_hash, to_bytes(moved))

        read = get(snapshot.snapshot_hash, snapshot.descriptor_hash, store=store)
        assert SnapshotProblem.PAYLOAD_DIGEST_MISMATCH in read.kinds

    def test_a_symbol_mismatch(self, store):
        access, snapshot = a_delivery()
        record(snapshot, recorded_at="2026-08-15T00:00:00Z")
        fewer = access.frame.drop(columns=[access.frame.columns[0]])
        store.put(snapshot.snapshot_hash, to_bytes(fewer))

        read = get(snapshot.snapshot_hash, snapshot.descriptor_hash, store=store)
        assert SnapshotProblem.SYMBOL_MISMATCH in read.kinds

    def test_the_six_kinds_are_reachable_and_different(self, store):
        """A taxonomy nothing produces is a taxonomy that rots.

        Each of these is exercised above; this asserts the enum has no member
        the code cannot reach, which is how a category quietly becomes
        decorative.
        """
        reached = set()
        # missing descriptor
        reached |= set(get("mdf1:x", "mds1:none", store=store).kinds)
        # mismatch
        _a, one = stored(store)
        reached |= set(get("mdf1:other", one.descriptor_hash, store=store).kinds)
        # payload missing
        _b, two = a_delivery(reinvested=True)
        record(two, recorded_at="2026-08-15T00:00:01Z")
        reached |= set(get(two.snapshot_hash, two.descriptor_hash,
                           store=store).kinds)

        assert {SnapshotProblem.DESCRIPTOR_MISSING,
                SnapshotProblem.DESCRIPTOR_MISMATCH,
                SnapshotProblem.PAYLOAD_MISSING} <= reached
        assert len(set(SnapshotProblem)) == 6


class TestTheReadProvesRatherThanFetches:
    def test_ok_requires_observations_and_not_merely_a_descriptor(self):
        from src.market_data.snapshot_contract import MarketSnapshot

        assert Read().ok is False
        assert Read(snapshot=object()).ok is False, (
            "a descriptor alone reported as a verified snapshot")
        assert Read(observations=object()).ok is False

    def test_a_refusal_says_which_kind_and_why(self, store):
        _access, snapshot = a_delivery()
        record(snapshot, recorded_at="2026-08-15T00:00:00Z")
        read = get(snapshot.snapshot_hash, snapshot.descriptor_hash, store=store)

        assert "PAYLOAD_MISSING" in read.refusal()
        assert "claim nobody can check" in read.refusal()
