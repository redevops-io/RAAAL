"""Which schema this database is at, and whether the code agrees.

`QUANTIFY_MIGRATION_HEAD` was a required deployment fact with nothing to
produce it. It is produced here, from the migration scripts themselves, and it
is checked against the database at startup.

**A mismatch fails closed.** Code expecting a column the database has not grown
yet fails at the first request that touches it — which may be hours after
deployment, on a user's request, in a code path nobody was watching. Refusing to
start says the same thing at the only moment it is cheap to hear.

**Unknown is not equal.** A database with no `alembic_version` row has never
been migrated; that is reported as `None` rather than assumed to be current.
The distinction matters because the two failure modes need different fixes: one
needs a migration run, the other needs investigating.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from alembic.config import Config
from alembic.script import ScriptDirectory

from .engine import Database

#: Repository root, from this file rather than the working directory: a service
#: started from elsewhere would otherwise find no migrations and conclude the
#: schema was empty.
_ROOT = Path(__file__).resolve().parents[2]

ALEMBIC_INI = _ROOT / "alembic.ini"
MIGRATIONS = _ROOT / "migrations"


class MigrationStateUnknown(RuntimeError):
    """The database cannot say which schema it is at."""


class MigrationMismatch(RuntimeError):
    """The database and the code disagree about the schema."""


def alembic_config(database: Optional[Database] = None) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(MIGRATIONS))
    if database is not None:
        # Set explicitly so `env.py` migrates *this* database. Without it the
        # environment variable decides, and a call naming a database would
        # migrate a different one while reporting success.
        url = database.url
        if url.startswith(("postgresql://", "postgres://")):
            url = f"postgresql+psycopg://{url.split('://', 1)[1]}"
        config.set_main_option("sqlalchemy.url", url)
    return config


def code_head() -> str:
    """The revision this build's migrations end at."""
    script = ScriptDirectory.from_config(alembic_config())
    heads: Sequence[str] = script.get_heads()
    if len(heads) != 1:
        raise MigrationStateUnknown(
            f"the migration history has {len(heads)} heads ({', '.join(heads)}). "
            "Branched history has no single 'current schema', so nothing can "
            "check a database against it")
    return heads[0]


def applied_revision(database: Database) -> Optional[str]:
    """The revision this database is at, or None if it has never been migrated.

    None is a real answer and is kept distinct from a stale revision: an
    unmigrated database and an out-of-date one need different remedies.
    """
    if "alembic_version" not in database.existing_tables():
        return None
    conn = database.connect()
    try:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        conn.close()
    return row["version_num"] if row else None


def upgrade(database: Database, revision: str = "head") -> None:
    """Run migrations against this database."""
    from alembic import command

    command.upgrade(alembic_config(database), revision)


def downgrade(database: Database, revision: str) -> None:
    from alembic import command

    command.downgrade(alembic_config(database), revision)


def stamp(database: Database, revision: str = "head") -> None:
    from alembic import command

    command.stamp(alembic_config(database), revision)


def require_migration_head(database: Database) -> str:
    """Refuse to run against a database at a different schema than this code.

    Called at startup rather than at first use. A service that starts happily
    and fails on the request that touches the missing column has moved the
    failure onto a user, and onto whichever code path happened to get there
    first.
    """
    expected = code_head()
    actual = applied_revision(database)
    if actual is None:
        raise MigrationMismatch(
            f"this database has never been migrated; the code expects "
            f"{expected}. Run `alembic upgrade head` before starting.")
    if actual != expected:
        raise MigrationMismatch(
            f"the database is at schema {actual} and this build expects "
            f"{expected}. Starting would fail at the first request touching a "
            "column that does not exist yet.")
    return expected
