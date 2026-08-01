"""Two different questions, two different digests.

    content_digest(frame)     is this the same data?
    file_sha256(path)         is this the same object?

They are not interchangeable, and using one where the other belongs is how an
integrity check quietly stops checking anything.

A Parquet file embeds its writer version, so the same DataFrame written by two
Arrow releases produces different bytes. Pinning file bytes for a committed
fixture would therefore break on a dependency upgrade with no data change — a
failure that says "the data moved" when it did not, which trains people to
regenerate the expected value until it passes.

For a remote object the reverse holds. The object is immutable and nobody
rewrites it, so the bytes *are* the identity, and verifying the bytes also
catches a truncated download that a content digest computed after a lenient
parse might not.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

DIGEST_VERSION = "mdv1"
"""Bumped only when the canonical form below changes. A digest whose rules moved
silently is a digest that agrees with everything."""

_NULL = "\x00"
"""Distinct from any formatted number, so a missing value and a zero cannot
collide. They mean different things: "not listed yet" is not "worth nothing"."""

_TEXT = "\x01"
"""Prefixes a non-numeric cell. Without it the string "1.0" and the number 1.0
canonicalize identically, and a column that silently changed dtype would keep
its digest — which is exactly the change worth catching."""


class IntegrityError(ValueError):
    """Data did not match what was pinned. Never downgraded to a warning."""


def canonical_lines(frame: pd.DataFrame, *, places: int = 6) -> Iterable[str]:
    """The frame as text, in one fixed order and one fixed precision.

    Columns are sorted and the index is rendered as ISO dates, so neither column
    order nor a timezone-aware reload changes the answer. Floats are formatted
    rather than hashed as raw bytes — the same value can have more than one
    binary representation across platforms, which is the cross-language
    divergence the canonicalization rules elsewhere in this project refuse
    floats over.
    """
    yield f"{DIGEST_VERSION}\t{len(frame)}\t{frame.shape[1]}"
    columns = sorted(frame.columns, key=str)
    yield "\t".join(str(c) for c in columns)
    for timestamp, row in zip(frame.index, frame[columns].to_numpy(dtype=object)):
        cells = (_cell(v, places) for v in row)
        yield f"{_label(timestamp)}\t" + "\t".join(cells)


def _cell(value: Any, places: int) -> str:
    """One cell, canonically.

    Not every frame here is a price matrix: weights and timelines carry labels
    and categories alongside numbers, and a digest that only handles floats
    silently excludes exactly those datasets from integrity checking.
    """
    if value is None:
        return _NULL
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _NULL if pd.isna(value) else f"{float(value):.{places}f}"
    try:
        if pd.isna(value):
            return _NULL
    except (TypeError, ValueError):
        pass                      # arrays and other non-scalars are not "na"
    return _TEXT + str(value)


def _label(value: Any) -> str:
    """Index labels as text. Dates render as ISO so a timezone-aware reload,
    or a reload as object dtype, does not change the answer."""
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return _TEXT + str(value)


def content_digest(frame: pd.DataFrame, *, places: int = 6) -> str:
    """Identity of the *data*, independent of how it was serialized."""
    digest = hashlib.sha256()
    for line in canonical_lines(frame, places=places):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return f"{DIGEST_VERSION}:{digest.hexdigest()}"


def file_sha256(path: Path | str, *, chunk: int = 1 << 20) -> str:
    """Identity of the *object*. Streamed, because these are not always small."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(frame: pd.DataFrame, *, expected_content_digest: Optional[str],
           source: str) -> None:
    """Refuse a mismatch. Never repair, never warn and continue.

    A loader that accepts the newest object when the pinned one has moved is not
    a loader with a hash check; it is a loader with a log line. Every result
    computed after that point silently describes different data than the run
    that recorded the pin.
    """
    if not expected_content_digest:
        return
    actual = content_digest(frame)
    if actual != expected_content_digest:
        raise IntegrityError(
            f"{source} does not match its pinned content digest.\n"
            f"  expected {expected_content_digest}\n"
            f"  actual   {actual}\n"
            "The pinned snapshot is what the recorded results were computed "
            "against. Loading this would make every figure describe data the "
            "run never saw.")
