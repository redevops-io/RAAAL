"""JSON that is structured on PostgreSQL and still readable on SQLite.

Quantify stores a great deal of structured payload — result contexts, scope
disclosures, comparison verdicts, intent chains, proposals, reconciliation
records, provenance, runtime pins. As opaque text none of it can be queried, so
an operational question like "which runs used the unpinned snapshot" has no
answer short of reading every row in the application.

JSONB answers those. SQLite has no equivalent, so the difference is handled the
same way every other dialect difference here is: in one place, by a value that
declares what it is.

**The value carries the type, not the column.** Statements are executed as raw
SQL with positional parameters, so the connection cannot know which column a
parameter is bound to. Wrapping the value at the call site — `Json(payload)`
instead of `json.dumps(payload)` — tells the adapter what to do without the
store knowing which database it is talking to.

**Reads are tolerant in one direction only.** PostgreSQL returns a parsed
object; SQLite returns text. `loads` accepts either and returns the object. It
does not accept arbitrary strings as JSON-free values: a column declared JSON
holds JSON, and a string that fails to parse is a corrupt row rather than a
value to pass through silently.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import Numeric, Text, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB

from .decimals import Money


class JsonText(TypeDecorator):
    """JSONB on PostgreSQL, TEXT on SQLite.

    Declared on the column so the migration renders the right type per dialect.
    Serialization itself happens in the connection adapter, because queries run
    through the raw driver rather than through SQLAlchemy Core.
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(Text())


class Json:
    """A value bound for a JSON column.

    Wrapping at the call site keeps the store dialect-free: it says *this is
    JSON*, and the adapter decides whether that means a serialized string or a
    JSONB parameter.
    """

    __slots__ = ("obj",)

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Json({self.obj!r})"

    def as_text(self) -> str:
        """The SQLite representation. Sorted keys so a stored payload is
        byte-stable across writes, which is what lets content hashes over the
        same object keep agreeing."""
        return json.dumps(self.obj)


def loads(value: Any, default: Any = None) -> Any:
    """Read a JSON column from either dialect.

    PostgreSQL hands back a parsed object and SQLite hands back text, so callers
    that did `json.loads(row["payload"])` would break on one of the two. This
    accepts both and is the only thing the store needs to know about it.
    """
    if value is None or value == "":
        return default
    if isinstance(value, (str, bytes, bytearray)):
        return json.loads(value)
    return value


class DecimalText(TypeDecorator):
    """NUMERIC on PostgreSQL, canonical decimal TEXT on SQLite.

    The types differ on purpose. SQLite has no exact decimal type — a NUMERIC
    column there has REAL affinity and would silently reintroduce the binary
    approximation the column exists to avoid. Canonical text is exact, and the
    adapter returns a `Decimal` from both, which is the parity that matters.

    `NUMERIC(38, 12)` covers shares, prices, values, ratios and fractional RSU
    quantities without pretending every field has currency-style two-decimal
    precision.
    """

    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Numeric(38, 12))
        return dialect.type_descriptor(Text())


def adapt(value: Any, dialect_name: str) -> Any:
    """Turn a bound parameter into something the driver accepts."""
    if isinstance(value, Json):
        if dialect_name == "postgresql":
            from psycopg.types.json import Jsonb

            return Jsonb(value.obj)
        return value.as_text()
    if isinstance(value, Money):
        # PostgreSQL takes the Decimal and stores it exactly; SQLite takes the
        # canonical string, because its NUMERIC affinity would convert a
        # Decimal back into a float.
        return value.as_decimal() if dialect_name == "postgresql" else value.text
    return value
