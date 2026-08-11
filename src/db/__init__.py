"""Where Quantify's records live, and on which engine.

PostgreSQL is the deployed authority. SQLite remains for tests, local
development and the standalone demo, and nowhere else — two equally
authoritative implementations would be another parity problem, and the parity
problems this codebase has already found were all of the same kind: a rule
enforced on one path and not on the path that runs.

    PostgreSQL   deployed pilot, production, and every test that proves
                 production behaviour — locking, concurrency, migrations,
                 transaction isolation
    SQLite       unit tests, local development, disposable environments
"""
from .engine import (
    DATABASE_URL_VAR,
    capture_statements,
    Database,
    Dialect,
    UnsupportedTarget,
    UntranslatableStatement,
    dialect_of,
    resolve_target,
    to_postgres,
)
from .schema import SCHEMA_VERSION, metadata, primary_key_columns, table_names

__all__ = ["DATABASE_URL_VAR", "capture_statements", "Database", "Dialect", "SCHEMA_VERSION",
           "UnsupportedTarget", "UntranslatableStatement", "dialect_of",
           "metadata", "primary_key_columns", "resolve_target", "table_names",
           "to_postgres"]
