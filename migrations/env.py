"""Alembic's entry point, pointed at the model rather than at a URL.

The database URL comes from `QUANTIFY_DATABASE_URL`, the same variable the
application reads. A URL configured separately in `alembic.ini` would let
migrations run against one database while the application talked to another,
and both would report success.

`target_metadata` is `src.db.schema.metadata` — the single definition both
dialects render from. Autogenerate compares the live database against it, which
is what makes `tests/test_migration_parity.py` able to fail on drift.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.engine import resolve_target  # noqa: E402
from src.db.schema import metadata  # noqa: E402

config = context.config

if config.config_file_name is not None:
    # `disable_existing_loggers` defaults to True, which switches off every
    # logger already configured in this process — including `uvicorn.error`,
    # which is where `web.failure` writes the private half of every database
    # error. Migrations run in-process at startup, so the default meant a
    # deployment that had migrated served perfectly sanitised public errors
    # and logged nothing at all about them: the public channel clean, the
    # operator channel silently dead, which is the exact failure mode the
    # error taxonomy exists to prevent.
    #
    # Found by asserting on the operator channel rather than only on the
    # response body. A test that checked the response alone would have passed.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = metadata


def _url() -> str:
    """Which database to migrate.

    A URL set on the config wins. `src.db.migrate` sets it to migrate a
    specific database — a test's temporary file, or an instance being upgraded —
    and without this precedence those calls would silently migrate whatever
    `QUANTIFY_DATABASE_URL` happened to name instead. That is not a hypothetical:
    the parity tests first ran against the developer's real `data/workspace.db`.

    Otherwise the environment decides, using the same variable the application
    reads, so migrations and the service cannot disagree about the target.
    """
    configured = config.get_main_option("sqlalchemy.url", None)
    url = configured or resolve_target(os.environ.get("QUANTIFY_DATABASE_URL"))
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


def render_item(type_, obj, autogen_context):
    """Render `JsonText` as a concrete type rather than as an import.

    A migration is a historical record of what was done. Referring to
    `src.db.types.JsonText` would make every past migration mean whatever that
    class means today — so a later change to the type would silently rewrite
    history, and a fresh migrate would produce a different schema than the one
    that actually shipped.

    `Text().with_variant(JSONB(), "postgresql")` says the same thing in terms
    that cannot drift: JSONB where it exists, TEXT where it does not.
    """
    from src.db.types import DecimalText, JsonText

    if type_ != "type":
        return False
    if isinstance(obj, JsonText):
        autogen_context.imports.add("import sqlalchemy as sa")
        autogen_context.imports.add(
            "from sqlalchemy.dialects import postgresql")
        return ('sa.Text().with_variant('
                'postgresql.JSONB(astext_type=sa.Text()), "postgresql")')
    if isinstance(obj, DecimalText):
        autogen_context.imports.add("import sqlalchemy as sa")
        return ('sa.Text().with_variant('
                'sa.Numeric(precision=38, scale=12), "postgresql")')
    return False


def run_migrations_offline() -> None:
    context.configure(
        url=_url(), target_metadata=target_metadata, literal_binds=True,
        dialect_opts={"paramstyle": "named"}, compare_type=True,
        render_item=render_item, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(
        section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata,
            compare_type=True, render_item=render_item,
            # SQLite cannot ALTER most things in place. Batch mode rebuilds the
            # table instead, which is what the three repair routines in
            # `store.py` were hand-rolling before this existed.
            render_as_batch=connection.dialect.name == "sqlite")
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
