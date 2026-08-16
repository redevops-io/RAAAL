"""Immutable observation bytes, addressed by the snapshot hash. Create-once.

    put(snapshot_hash, payload)   absent          -> written
                                  identical bytes -> idempotent success
                                  different bytes -> hard failure
    get(snapshot_hash)            bytes, or None

**There is no overwrite path.** Not a discouraged one — none. A snapshot hash
names a specific set of observations, so bytes arriving under an existing hash
either are those observations or are something else pretending to be them, and
the second must never be written. Every stored figure that cites a snapshot
would silently start describing different data.

**The store is deliberately stupid.** It compares bytes and knows nothing about
frames, symbols or digests. Whether the payload actually digests to the hash it
was filed under is the read path's question — a store that verified content
would need to understand every format it ever holds, and a store that
half-verified would be trusted for the half it did not do.

**Filesystem now, object storage later.** The interface is `put`/`get` over
bytes because that is all S3 offers too; moving is then a change of backend
rather than of contract. What is *not* deferred is the create-once rule, which
belongs here rather than in a bucket policy somebody can edit.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class PayloadConflict(ValueError):
    """One snapshot hash, two different sets of bytes.

    Either a hash collision or a bug that computed the address from something
    other than what it stored. Raised rather than resolved: overwriting would
    change what every stored figure citing this snapshot was computed from, and
    keeping the first silently would leave the caller believing it wrote.
    """


@dataclass(frozen=True)
class ObjectStore:
    """Content-addressed bytes under a root directory."""

    root: Path

    def _path(self, snapshot_hash: str) -> Path:
        # The hash as written, with the separator made safe. Sharded two deep
        # so a directory listing stays usable at scale — and derived from the
        # address rather than from a counter, so the same snapshot always lands
        # in the same place on any machine.
        safe = snapshot_hash.replace(":", "_").replace("/", "_")
        return self.root / safe[:2] / safe[2:4] / f"{safe}.parquet"

    def put(self, snapshot_hash: str, payload: bytes) -> str:
        """Write once. Returns the address, whether or not this call wrote it."""
        if not snapshot_hash:
            raise ValueError("a payload with no address cannot be stored")
        if not payload:
            raise ValueError(
                f"{snapshot_hash} was offered with no bytes. An empty payload "
                "under a real address is worse than an absent one: it reads as "
                "a snapshot that exists and holds nothing")

        target = self._path(snapshot_hash)
        if target.exists():
            if target.read_bytes() == payload:
                return snapshot_hash              # said the same thing already
            raise PayloadConflict(
                f"{snapshot_hash} already holds different bytes "
                f"({len(payload)} offered, {target.stat().st_size} stored). A "
                "snapshot hash names one set of observations; writing over it "
                "would change what every figure citing it was computed from")

        target.parent.mkdir(parents=True, exist_ok=True)
        # Written beside and moved into place, so a crash mid-write leaves no
        # half-file at an address that is supposed to be complete.
        staging = target.with_suffix(".partial")
        staging.write_bytes(payload)
        os.replace(staging, target)
        return snapshot_hash

    def get(self, snapshot_hash: str) -> Optional[bytes]:
        """The bytes, or None. Absence is an answer, not an error."""
        target = self._path(snapshot_hash)
        return target.read_bytes() if target.exists() else None

    def holds(self, snapshot_hash: str) -> bool:
        return self._path(snapshot_hash).exists()


def to_bytes(observations) -> bytes:
    """Observations as bytes, deterministically.

    Byte-stable across writes, which is what lets `put` compare bytes rather
    than trust a caller's claim about them — checked in the tests rather than
    assumed, because a serializer that embedded a timestamp would turn every
    re-put into a spurious conflict.
    """
    import io

    buffer = io.BytesIO()
    observations.to_parquet(buffer, engine="pyarrow", compression="zstd")
    return buffer.getvalue()


def from_bytes(payload: bytes):
    """Bytes back to observations. The digest must survive this."""
    import io

    import pandas as pd

    return pd.read_parquet(io.BytesIO(payload))


def byte_digest(payload: bytes) -> str:
    """A digest of the stored bytes, distinct from the frame's own.

    Two different things: `frame_digest` describes the *values* and is stable
    across serialization formats; this describes the file. A store comparing
    files needs the second, and a reader checking the data needs the first.
    """
    return "obj1:" + hashlib.sha256(payload).hexdigest()
