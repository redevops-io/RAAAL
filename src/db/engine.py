"""One connection interface over two dialects, so callers write neither.

The store's 36 methods hold ordinary SQL. Making them dialect-aware would have
put a conditional at every call site, and the conditional that gets forgotten is
the one on the path nobody tested — which is the shape of every defect this
codebase has spent its time removing.

Instead the dialect difference lives here, in three places and no others:

    parameter style   `?` is rewritten to `%s` for psycopg
    upsert            `INSERT OR REPLACE` becomes `ON CONFLICT DO UPDATE`
    introspection     `PRAGMA table_info` becomes a SQLAlchemy inspector

**The conflict target comes from the model, not from a list.** A hand-kept map
of table to primary key would be a second schema, and it would drift from the
first silently — an upsert with the wrong conflict target does not fail, it
inserts a duplicate.

**Rows are mappings in both dialects.** `sqlite3.Row` and psycopg's `dict_row`
both support `row["column"]` and `dict(row)`, so calling code sees one shape.
"""
from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

from sqlalchemy import create_engine, inspect

from .schema import metadata, primary_key_columns
from .types import adapt

#: Where a deployed instance finds its database. Absent, the target falls back
#: to a local SQLite file — correct for tests and a developer checkout, and
#: refused for a deployed pilot by `require_postgresql`.
DATABASE_URL_VAR = "QUANTIFY_DATABASE_URL"

DEFAULT_SQLITE_PATH = Path("data/workspace.db")


class Dialect(str, Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class UnsupportedTarget(ValueError):
    """A database target this build cannot open."""


def _is_url(target: Union[str, Path]) -> bool:
    return isinstance(target, str) and "://" in target


def resolve_target(target: Optional[Union[str, Path]] = None) -> str:
    """Turn a path, a URL or nothing into a SQLAlchemy URL.

    A path stays a path so that `WorkspaceStore(tmp_path / "w.db")` — the idiom
    in twenty test files — keeps meaning what it has always meant.
    """
    if target is None:
        # Ask the deployment, not the environment. This line used to read
        # `QUANTIFY_DATABASE_URL` itself, which made the engine a second
        # resolver: the preflight decided the deployment was PostgreSQL and
        # this decided, separately and by the same means, that it was too —
        # until `WorkspaceStore.__init__` stopped passing its argument through
        # and the two answers silently parted. One resolver, so there is
        # nothing left to disagree.
        from ..deploy.context import current

        return current().database.url
    if _is_url(target):
        return str(target)
    return f"sqlite:///{Path(target)}"


def dialect_of(url: str) -> Dialect:
    if url.startswith("sqlite"):
        return Dialect.SQLITE
    if url.startswith(("postgresql", "postgres")):
        return Dialect.POSTGRESQL
    raise UnsupportedTarget(
        f"{url.split('://')[0]!r} is not a database this build can open. "
        "Supported: sqlite (tests and local development), postgresql (deployed)")


# --------------------------------------------------------------------------
# statement translation


_PLACEHOLDER = re.compile(r"\?(?=(?:[^']*'[^']*')*[^']*$)")

#: Spans the whole statement, VALUES included. The conflict clause must follow
#: VALUES, so a pattern stopping at the column list cannot place it correctly.
_UPSERT = re.compile(
    r"INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)\s*\(([^)]*)\)\s*VALUES\s*\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL)

_UNTRANSLATED = re.compile(r"INSERT\s+OR\s+(REPLACE|IGNORE)", re.IGNORECASE)

#: SQLite's JSON accessor. PostgreSQL has no `json_extract`, so a query using
#: one failed on every deployed request while passing every SQLite test — the
#: same shape as `INSERT OR REPLACE`, and found the same way: by driving a real
#: request into a real PostgreSQL database.
#:
#: Only the top-level `$.field` form is translated. A deeper path has more than
#: one reasonable PostgreSQL spelling, and guessing between them would be a
#: silent difference in what a query matches.
_JSON_EXTRACT = re.compile(
    r"json_extract\s*\(\s*(\w+)\s*,\s*'\$\.(\w+)'\s*\)", re.IGNORECASE)

_UNTRANSLATED_JSON = re.compile(r"json_extract\s*\(", re.IGNORECASE)


class UntranslatableStatement(ValueError):
    """A SQLite-only statement with no PostgreSQL rewrite.

    Raised rather than passed through: PostgreSQL would reject it anyway, and a
    clear failure here names the statement instead of surfacing a syntax error
    from the driver with no indication of which query produced it.
    """


def to_postgres(sql: str) -> str:
    """Rewrite SQLite-flavoured SQL for PostgreSQL.

    `INSERT OR REPLACE` has no PostgreSQL spelling. The equivalent is an upsert
    whose conflict target is the primary key, and whose update assigns every
    non-key column — which is what `OR REPLACE` does: it replaces the row.
    """
    def upsert(match: "re.Match[str]") -> str:
        table, columns, values = match.group(1), match.group(2), match.group(3)
        names = [name.strip() for name in columns.split(",") if name.strip()]
        keys = primary_key_columns(table)
        assignments = ", ".join(f"{name} = EXCLUDED.{name}"
                                for name in names if name not in keys)
        conflict = ", ".join(keys)
        # Every column belongs to the key: there is nothing left to assign, and
        # the stored row is already identical to the one being written.
        action = f"DO UPDATE SET {assignments}" if assignments else "DO NOTHING"
        return (f"INSERT INTO {table} ({' '.join(columns.split())}) "
                f"VALUES ({' '.join(values.split())}) "
                f"ON CONFLICT ({conflict}) {action}")

    sql = _JSON_EXTRACT.sub(r"\1->>'\2'", sql)
    if _UNTRANSLATED_JSON.search(sql):
        raise UntranslatableStatement(
            "`json_extract` outside the top-level `$.field` form has no single "
            f"PostgreSQL spelling here:\n{sql.strip()}")

    sql = _UPSERT.sub(upsert, sql)
    if _UNTRANSLATED.search(sql):
        raise UntranslatableStatement(
            "`INSERT OR REPLACE/IGNORE` outside the `... (columns) VALUES (...)` "
            f"form has no rewrite here:\n{sql.strip()}")
    return _PLACEHOLDER.sub("%s", sql)


#: Statements issued while a recorder is active, in order. Populated after
#: dialect translation, because the statement the store *writes* is not the one
#: PostgreSQL *receives* — `INSERT OR REPLACE` becomes `ON CONFLICT`, and it is
#: the rewrite that either overwrites an immutable body or does not.
_recorders: List[List[str]] = []


def _record(sql: str) -> None:
    for sink in _recorders:
        sink.append(sql)


@contextmanager
def capture_statements() -> Iterator[List[str]]:
    """Collect every statement issued inside this block.

    Used by `tests/test_immutability.py` to check what the store actually sends
    rather than what its source contains — the distinction that nine
    prose-matching failures in this codebase have turned on.
    """
    sink: List[str] = []
    _recorders.append(sink)
    try:
        yield sink
    finally:
        _recorders.remove(sink)


class _Cursor:
    """A cursor that reads the same way under either driver."""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def __iter__(self) -> Iterator[Any]:
        return iter(self._cursor)

    def fetchall(self) -> Sequence[Any]:
        return self._cursor.fetchall()

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class Connection:
    """A driver connection with the dialect differences already applied."""

    def __init__(self, raw: Any, dialect: Dialect) -> None:
        self._raw = raw
        self.dialect = dialect

    def execute(self, sql: str, params: Sequence[Any] = ()) -> _Cursor:
        bound = tuple(adapt(value, self.dialect.value) for value in params)
        if self.dialect is Dialect.POSTGRESQL:
            issued = to_postgres(sql)
            _record(issued)
            return _Cursor(self._raw.execute(issued, bound))
        _record(sql)
        return _Cursor(self._raw.execute(sql, bound))

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    def close(self) -> None:
        self._raw.close()


class Database:
    """A database target, its dialect, and how to open it."""

    def __init__(self, target: Optional[Union[str, Path]] = None) -> None:
        self.url = resolve_target(target)
        self.dialect = dialect_of(self.url)
        if self.dialect is Dialect.SQLITE:
            self.path: Optional[Path] = Path(self.url.split("///", 1)[1])
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.path = None

    # -- schema ----------------------------------------------------------

    def sqlalchemy_engine(self):
        """Used for DDL and introspection only; queries use the raw driver."""
        url = self.url
        if self.dialect is Dialect.POSTGRESQL:
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        return create_engine(url)

    def create_all(self) -> None:
        """Create anything missing. Existing tables are left alone.

        For PostgreSQL this is a convenience for tests; a deployed instance gets
        its schema from Alembic, and `require_migration_head` refuses to start
        against a database that has not been migrated.
        """
        engine = self.sqlalchemy_engine()
        try:
            metadata.create_all(engine, checkfirst=True)
        finally:
            engine.dispose()

    def columns(self, table: str) -> Sequence[dict]:
        """Column metadata, replacing `PRAGMA table_info`."""
        engine = self.sqlalchemy_engine()
        try:
            return inspect(engine).get_columns(table)
        finally:
            engine.dispose()

    def existing_tables(self) -> Tuple[str, ...]:
        engine = self.sqlalchemy_engine()
        try:
            return tuple(sorted(inspect(engine).get_table_names()))
        finally:
            engine.dispose()

    # -- connections -----------------------------------------------------

    def connect(self) -> Connection:
        if self.dialect is Dialect.SQLITE:
            raw = sqlite3.connect(self.path)
            raw.row_factory = sqlite3.Row
            # Off by default in SQLite, which means a declared foreign key is
            # decorative there. The shipped schema had one on `plan_run` and it
            # had never once been enforced — so the constraint said something
            # about the data that was not true, on the engine the tests run on.
            raw.execute("PRAGMA foreign_keys = ON")
            return Connection(raw, self.dialect)

        import psycopg
        from psycopg.rows import dict_row

        raw = psycopg.connect(self.url, row_factory=dict_row, autocommit=False)
        return Connection(raw, self.dialect)
