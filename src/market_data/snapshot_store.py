"""Where snapshot descriptors are kept, and what may not change about them.

Metadata and index only. The observations themselves are bytes and belong in an
object store keyed by `snapshot_hash`; this table holds what is believed *about*
those bytes and is keyed by `descriptor_hash`.

**Two hashes, two lifetimes.** The same observations can carry several
descriptions over time — a licence re-reviewed, an adapter version corrected,
provenance filled in — and each is a distinct record of what was believed and
when. So a second write for one `snapshot_hash` is ordinary and expected, while
a second write for one `descriptor_hash` with a different body is a conflict:
the descriptor is addressed by its own content, so a differing body under the
same address means one of them is not what it says.

**No tenant column, by classification.** `market_snapshot` is the first
`SHARED_REFERENCE` table. A snapshot describes the world rather than a user, and
the rule for such a table is stricter than the tenant rule rather than an
exemption from it: no `owner`, no `participant`, nothing identifying anybody. A
column added to satisfy a check is one a future query will scope by and a future
reader will trust.

**Recording is not verifying.** This stores a description. Whether the bytes it
describes exist, and whether they digest to the hash it names, is the read
path's question — and conflating the two would let a descriptor stand as
evidence that the data behind it is intact.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Sequence

from ..db.types import Json, loads


class DescriptorConflict(ValueError):
    """One descriptor address, two different descriptions.

    A descriptor hash is taken over the description, so this means the stored
    body and the offered one disagree while claiming the same address — and
    exactly one of them can be right. Raised rather than resolved: picking a
    side would silently rewrite what somebody recorded about a licence or an
    adapter.
    """


def _connect():
    from ..db.engine import Database
    from ..deploy.context import current

    return Database(current().database.url).connect()


def record(snapshot, *, recorded_at: str) -> str:
    """Store one description of one set of observations. Idempotent by address.

    Re-recording an identical descriptor is a no-op, because it says the same
    thing. Re-recording a *different* descriptor under the same hash raises —
    that is a hash collision or a bug, and neither should be written through.
    """
    body = snapshot.to_json()
    address = snapshot.descriptor_hash

    connection = _connect()
    try:
        existing = connection.execute(
            "SELECT snapshot_hash, resolution, license_review_status "
            "FROM market_snapshot WHERE descriptor_hash = ?",
            (address,)).fetchone()
        if existing is not None:
            if existing["snapshot_hash"] != snapshot.snapshot_hash:
                raise DescriptorConflict(
                    f"descriptor {address} is stored against observations "
                    f"{existing['snapshot_hash']} and was offered against "
                    f"{snapshot.snapshot_hash}. A descriptor is addressed by "
                    "its own content, so these cannot both be it")
            return address                       # said the same thing already

        span = body["session_range"]
        adapter = body["source_adapter"]
        connection.execute(
            """INSERT INTO market_snapshot
               (descriptor_hash, snapshot_hash, snapshot_id, dataset_id,
                symbols, range_start, range_end, sessions, resolution,
                corporate_actions, calendar, source_adapter,
                source_adapter_version, source_uri, data_as_of, license_class,
                license_review_status, content_digest_version,
                contract_version, recorded_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (address, snapshot.snapshot_hash, body["snapshot_id"],
             body["dataset_id"], Json(list(body["symbols"])),
             span["start"], span["end"], int(span["sessions"]),
             Json(body["resolution"]), body["corporate_actions"],
             body["calendar"], adapter["name"], adapter["version"],
             body["source_uri"], body["data_as_of"], body["license_class"],
             body["license_review_status"], body["content_digest_version"],
             body["version"], recorded_at))
        connection.commit()
    finally:
        connection.close()
    return address


def descriptor(descriptor_hash: str) -> Optional[Dict[str, Any]]:
    """One description, in the contract's own shape, or None."""
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT * FROM market_snapshot WHERE descriptor_hash = ?",
            (descriptor_hash,)).fetchone()
    finally:
        connection.close()
    return _as_contract(row) if row is not None else None


def descriptors_for(snapshot_hash: str) -> Sequence[Dict[str, Any]]:
    """Every description of one set of observations, oldest first.

    Plural on purpose, and this is the index the two-hash split exists for: it
    answers "what has been believed about this data, and when did that change",
    which a store keyed by content could not express without overwriting the
    earlier answer.
    """
    connection = _connect()
    try:
        rows = connection.execute(
            "SELECT * FROM market_snapshot WHERE snapshot_hash = ? "
            "ORDER BY recorded_at", (snapshot_hash,)).fetchall()
    finally:
        connection.close()
    return [_as_contract(row) for row in rows]


def _as_contract(row) -> Dict[str, Any]:
    """The stored row as the contract's JSON, so one reader rebuilds it.

    Reassembled here rather than by each caller, for the reason the access
    event does the same: a body compared against a hash must be the stored
    body, not whatever shape a particular caller happened to build.
    """
    return {
        "version": row["contract_version"],
        "snapshot_hash": row["snapshot_hash"],
        "snapshot_id": row["snapshot_id"],
        "dataset_id": row["dataset_id"],
        "symbols": loads(row["symbols"], []),
        "session_range": {"start": row["range_start"], "end": row["range_end"],
                          "sessions": row["sessions"]},
        "resolution": loads(row["resolution"], {}),
        "corporate_actions": row["corporate_actions"],
        "calendar": row["calendar"],
        "source_adapter": {"name": row["source_adapter"],
                           "version": row["source_adapter_version"]},
        "source_uri": row["source_uri"],
        "data_as_of": row["data_as_of"],
        "license_class": row["license_class"],
        "license_review_status": row["license_review_status"],
        "content_digest_version": row["content_digest_version"],
    }


def latest_descriptor(snapshot_hash: str) -> Optional[str]:
    """The address of the current description of these observations.

    "Current" means most recently recorded. A snapshot legitimately carries
    several descriptors — a licence re-reviewed, an adapter corrected — and a
    consumer has no basis for choosing between them, so the owner of the data
    offers one. The older ones stay readable by address; this is which to
    verify against by default, not which is the only one.
    """
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT descriptor_hash FROM market_snapshot "
            "WHERE snapshot_hash = ? ORDER BY recorded_at DESC LIMIT 1",
            (snapshot_hash,)).fetchone()
    finally:
        connection.close()
    return row["descriptor_hash"] if row is not None else None
