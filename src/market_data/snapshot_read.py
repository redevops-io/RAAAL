"""Read a snapshot by hash, and prove it before returning it.

    get(snapshot_hash, descriptor_hash) -> Read

**Metadata may exist without payload, and that state must never read as a valid
snapshot.** The descriptor store records what is *believed* about some
observations; this path proves the bytes exist and match. A descriptor standing
alone is a claim with nothing behind it, and the single most likely way for one
to appear is an interrupted write — so it is a named outcome rather than an
error nobody classified.

**Every failure is distinct, because each points somewhere different.**

    DESCRIPTOR_MISSING        nobody recorded this description
    DESCRIPTOR_MISMATCH       the descriptor describes other observations
    PAYLOAD_MISSING           the description exists and the bytes do not
    EMPTY_OBSERVATIONS        bytes that decode to nothing
    PAYLOAD_DIGEST_MISMATCH   the bytes are not the observations named
    SYMBOL_MISMATCH           the right bytes, describing different instruments

A single `ok: bool` would send whoever reads it to the wrong place four times
out of five: a missing payload is a storage repair, a digest mismatch is a
corruption investigation, and a symbol mismatch is a descriptor that was written
wrong. Collapsing them would make the common case — an interrupted write —
indistinguishable from the alarming one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence, Tuple


class SnapshotProblem(str, Enum):
    """Why a snapshot could not be read, specifically enough to act on."""

    DESCRIPTOR_MISSING = "DESCRIPTOR_MISSING"
    DESCRIPTOR_MISMATCH = "DESCRIPTOR_MISMATCH"
    PAYLOAD_MISSING = "PAYLOAD_MISSING"
    EMPTY_OBSERVATIONS = "EMPTY_OBSERVATIONS"
    PAYLOAD_DIGEST_MISMATCH = "PAYLOAD_DIGEST_MISMATCH"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"


@dataclass(frozen=True)
class Read:
    """What a read produced, or precisely why it produced nothing."""

    problems: Tuple[Tuple[SnapshotProblem, str], ...] = ()
    snapshot: Optional[Any] = None
    observations: Optional[Any] = None

    @property
    def ok(self) -> bool:
        """Verified, not merely fetched.

        A `Read` with a descriptor and no observations is not a partial
        success — it is a claim with nothing behind it, and returning it as
        usable is exactly the mistake this whole path exists to prevent.
        """
        return (not self.problems and self.snapshot is not None
                and self.observations is not None)

    @property
    def kinds(self) -> Tuple[SnapshotProblem, ...]:
        return tuple(kind for kind, _why in self.problems)

    def refusal(self) -> str:
        return "; ".join(f"{kind.value}: {why}" for kind, why in self.problems)


def _problem(kind: SnapshotProblem, why: str, **carried) -> Read:
    return Read(problems=((kind, why),), **carried)


def get(snapshot_hash: str, descriptor_hash: str, *, store,
        read_descriptor=None) -> Read:
    """Fetch by hash and verify against the description.

    Ordered, and the order is the point: there is no sense checking a digest
    before knowing which observations were expected, and no sense checking
    symbols against bytes that are not the right bytes. Each stage answers a
    question the next one depends on, so the first failure is reported and the
    rest are not guessed at.
    """
    from .object_store import from_bytes
    from .snapshot_contract import from_json

    if read_descriptor is None:
        from .snapshot_store import descriptor as read_descriptor

    body = read_descriptor(descriptor_hash)
    if body is None:
        return _problem(
            SnapshotProblem.DESCRIPTOR_MISSING,
            f"no descriptor {descriptor_hash} is recorded, so there is nothing "
            "saying what these observations should be")

    snapshot = from_json(body)
    if snapshot.snapshot_hash != snapshot_hash:
        return _problem(
            SnapshotProblem.DESCRIPTOR_MISMATCH,
            f"descriptor {descriptor_hash} describes observations "
            f"{snapshot.snapshot_hash} and {snapshot_hash} was asked for. "
            "Verifying one against the other would confirm the wrong data",
            snapshot=snapshot)

    payload = store.get(snapshot_hash)
    if payload is None:
        # The state this path exists to name. A descriptor is a record of what
        # was believed; without bytes it is a belief with nothing behind it,
        # and the likeliest cause is a write that did not finish.
        return _problem(
            SnapshotProblem.PAYLOAD_MISSING,
            f"the descriptor for {snapshot_hash} is recorded and the "
            "observations are not stored. Metadata without payload is not a "
            "snapshot; it is a claim nobody can check",
            snapshot=snapshot)

    observations = from_bytes(payload)
    if observations is None or len(observations.index) == 0:
        return _problem(
            SnapshotProblem.EMPTY_OBSERVATIONS,
            f"{snapshot_hash} decodes to no sessions. Bytes that hold nothing "
            "are not the observations this descriptor names",
            snapshot=snapshot, observations=observations)

    # `verify` asks the bytes, and its answers are turned into named kinds here
    # rather than returned as prose. A caller acting on "the digest differs"
    # and one acting on "the symbols differ" are doing different jobs.
    problems = []
    for why in snapshot.verify(observations):
        if "digest" in why:
            problems.append((SnapshotProblem.PAYLOAD_DIGEST_MISMATCH, why))
        elif "symbols" in why:
            problems.append((SnapshotProblem.SYMBOL_MISMATCH, why))
        else:
            problems.append((SnapshotProblem.PAYLOAD_DIGEST_MISMATCH, why))

    if problems:
        return Read(problems=tuple(problems), snapshot=snapshot,
                    observations=observations)
    return Read(snapshot=snapshot, observations=observations)
