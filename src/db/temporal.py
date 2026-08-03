"""Two temporal grammars, kept apart, validated before anything is stored.

Timestamps stay strings through this migration — they participate in content
hashes and in the effective-date/observed-date distinction, and retyping them
belongs in its own change rather than mixed into an engine cutover. But a string
column means the database compares bytes. So the canonical form has to be
established here, at the application boundary, and PostgreSQL must never be
asked what one of these strings *means* — only whether two of them are equal or
ordered.

    a date        YYYY-MM-DD
    a timestamp   YYYY-MM-DDTHH:MM:SS[.ffffff]Z

**The two are not interchangeable, and neither accepts the other.** A vest
settles on a date; a report arrives at an instant. `effective_date` taking a
datetime would make an on-time vest sortable against a report time, and
`observed_at` taking a bare date would silently mean midnight UTC — a fact
nobody stated. The reconciliation system's whole point is that these are
different kinds of thing, so the types refuse each other rather than coercing.

**UTC only.** Preserving arbitrary offsets would let two equal instants have two
stored spellings, and a byte-comparing database would call them different.
`2026-08-02T20:00:00-04:00` and `2026-08-03T00:00:00Z` are one instant and
normalize to one string.

**Naive datetimes are refused, not assumed.** A datetime without a timezone is
an instant nobody has identified. Guessing UTC would turn a local wall-clock
reading into a specific wrong instant, and the error would be silent and
off by hours.
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any, Optional, Union

#: Bumped if either grammar changes. Recorded so a stored value can be read
#: against the rules it was written under.
TEMPORAL_GRAMMAR_VERSION = "temporal/utc-iso@1"

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Offsets are accepted as *input* and normalized away; `Z` is the only form
#: that is ever stored. Lowercase `z` is refused: with one stored spelling, a
#: second accepted input spelling is a way for two equal instants to be written
#: differently by two callers.
TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?(Z|[+-]\d{2}:\d{2})$")

Dateish = Union[str, dt.date]
Timestampish = Union[str, dt.datetime]


class NotCanonicalTemporal(ValueError):
    """A date or timestamp that cannot be stored as written."""


def canonical_date(value: Optional[Any]) -> Optional[str]:
    """`YYYY-MM-DD`, or None.

    Refuses a datetime: an event that happened on a day did not happen at an
    instant this code is entitled to invent.
    """
    if value is None:
        return None

    if isinstance(value, dt.datetime):
        raise NotCanonicalTemporal(
            f"{value!r} is a datetime where a date is required. A date and an "
            "instant are different kinds of fact here — an effective date "
            "compared against a report time would make an on-time event look "
            "late. Pass a date, or the date part explicitly.")

    if isinstance(value, dt.date):
        return value.isoformat()

    if not isinstance(value, str):
        raise NotCanonicalTemporal(
            f"{type(value).__name__} is not a date: {value!r}")

    if not DATE.match(value):
        raise NotCanonicalTemporal(
            f"{value!r} is not a canonical date. Expected exactly "
            "YYYY-MM-DD — zero-padded, no time part, no surrounding "
            "whitespace.")
    try:
        dt.date.fromisoformat(value)
    except ValueError as exc:
        raise NotCanonicalTemporal(
            f"{value!r} is not a real calendar date") from exc
    return value


def canonical_timestamp(value: Optional[Any]) -> Optional[str]:
    """`YYYY-MM-DDTHH:MM:SS[.ffffff]Z`, or None.

    An aware datetime is normalized to UTC. A naive one is refused, as is a
    bare date: both would have this code choosing an instant on the caller's
    behalf.
    """
    if value is None:
        return None

    if isinstance(value, dt.datetime):
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise NotCanonicalTemporal(
                f"{value!r} has no timezone. A naive datetime is an instant "
                "nobody has identified; assuming UTC would turn a local "
                "wall-clock reading into a specific wrong instant, silently "
                "and by hours.")
        return _format(value.astimezone(dt.timezone.utc))

    if isinstance(value, dt.date):
        raise NotCanonicalTemporal(
            f"{value!r} is a date where a timestamp is required. Reading it as "
            "midnight UTC would state a time nobody supplied.")

    if not isinstance(value, str):
        raise NotCanonicalTemporal(
            f"{type(value).__name__} is not a timestamp: {value!r}")

    if not TIMESTAMP.match(value):
        raise NotCanonicalTemporal(
            f"{value!r} is not a canonical timestamp. Expected "
            "YYYY-MM-DDTHH:MM:SS[.ffffff] followed by 'Z' or a numeric offset "
            "— no bare date, no missing timezone, no lowercase 'z', no "
            "surrounding whitespace.")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NotCanonicalTemporal(
            f"{value!r} is not a real instant") from exc
    return _format(parsed.astimezone(dt.timezone.utc))


def _format(moment: dt.datetime) -> str:
    """One spelling per instant.

    A fractional part appears only when there is one. Emitting `.000000`
    always would make an equal instant compare unequal to a value written
    before the fraction existed.
    """
    if moment.microsecond:
        return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    return moment.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def same_instant(left: Optional[Any], right: Optional[Any]) -> bool:
    """Whether two timestamps denote the same instant, after normalization."""
    return canonical_timestamp(left) == canonical_timestamp(right)
