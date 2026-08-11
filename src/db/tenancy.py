"""Tenant identity, checked at the three layers where it can be lost.

Five tables once let one tenant's write overwrite another's row, and they were
found one at a time. They were not five bugs; they were one invariant violated
in five places, and nothing enumerated the invariant. Then a correct key
migration invalidated a correct consumer — `OwnershipPath` still joined on the
scalar identity those keys used to have — which showed that fixing the schema
is not the end of it.

So there are three checks, reading three different things:

    schema      what the deployed database permits
    writes      what production writers ask it to do
    consumers   whether a reader or a join preserves the identity it references

None subsumes the others. A schema can be correct while a writer uses a
conflict target that ignores it. A writer can be correct while a join matches
half a key. And both can be correct while the behaviour still crosses tenants,
which is what the Alice/Bob collision fixtures are for.

**Neither side is generated from the other.** The schema check reads PostgreSQL
metadata, not `src/db/schema.py` — a model and its migrations can share an
omission, and have. The write and consumer checks read captured statements
after dialect adaptation, not source text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .mutability import TABLE_MUTABILITY

#: The column that carries tenant identity everywhere it appears.
OWNER = "owner"


@dataclass(frozen=True)
class TenancyViolation:
    """One place tenant identity is missing, named so it can be acted on."""

    table: str
    layer: str
    """schema | write | consumer"""

    detail: str
    statement: str = ""

    def __str__(self) -> str:  # pragma: no cover - diagnostics
        body = f"[{self.layer}] {self.table}: {self.detail}"
        return f"{body}\n    {self.statement.strip()[:200]}" if self.statement \
            else body


# --------------------------------------------------------------------------
# 1. schema


def schema_violations(
        columns: Mapping[str, Sequence[Mapping]],
        primary_keys: Mapping[str, Sequence[str]],
        unique_constraints: Mapping[str, Sequence[Sequence[str]]],
        foreign_keys: Mapping[str, Sequence[Mapping]],
        tenant_owned: Iterable[str]) -> List[TenancyViolation]:
    """What the deployed schema permits, from PostgreSQL's own metadata."""
    found: List[TenancyViolation] = []
    owned = set(tenant_owned)

    for table in sorted(owned):
        table_columns = {c["name"]: c for c in columns.get(table, ())}
        if OWNER not in table_columns:
            found.append(TenancyViolation(
                table, "schema", "tenant-owned and has no `owner` column"))
            continue
        if table_columns[OWNER].get("nullable", True):
            found.append(TenancyViolation(
                table, "schema",
                "`owner` is nullable, so a row can exist belonging to nobody"))

        key = tuple(primary_keys.get(table, ()))
        if OWNER not in key:
            found.append(TenancyViolation(
                table, "schema",
                f"primary key {key} omits `owner`, so two tenants cannot hold "
                "the same id and one write can replace the other's row"))

        for unique in unique_constraints.get(table, ()):
            # A unique constraint is an identity too: one that omits the owner
            # refuses a second tenant the same value, which both breaks
            # legitimate use and answers a question about another tenant.
            if OWNER not in tuple(unique):
                found.append(TenancyViolation(
                    table, "schema",
                    f"unique constraint {tuple(unique)} omits `owner`"))

        for reference in foreign_keys.get(table, ()):
            target = reference.get("referred_table")
            if target not in owned:
                continue
            constrained = tuple(reference.get("constrained_columns", ()))
            if OWNER not in constrained:
                found.append(TenancyViolation(
                    table, "schema",
                    f"foreign key {constrained} -> {target} omits `owner`, so "
                    "a row can reference another tenant's parent"))
    return found


# --------------------------------------------------------------------------
# 2. writes


_INSERT = re.compile(
    r"INSERT\s+INTO\s+(\w+)\s*\(([^)]*)\)", re.IGNORECASE | re.DOTALL)
_CONFLICT_TARGET = re.compile(
    r"ON\s+CONFLICT\s*\(([^)]*)\)", re.IGNORECASE)
_UPDATE = re.compile(r"UPDATE\s+(\w+)\s+SET\b", re.IGNORECASE)
_WHERE = re.compile(r"\bWHERE\b(.*?)(?:\bRETURNING\b|$)",
                    re.IGNORECASE | re.DOTALL)


def _names(clause: str) -> List[str]:
    return [one.strip().lower() for one in clause.split(",") if one.strip()]


def write_violations(statements: Iterable[str],
                     tenant_owned: Iterable[str]) -> List[TenancyViolation]:
    """What production writers ask the schema to do.

    Would have caught all five original overwrite defects even had the schema
    still permitted them, because a conflict target omitting `owner` is visible
    in the statement regardless of what the key says.
    """
    found: List[TenancyViolation] = []
    owned = set(tenant_owned)

    for sql in statements:
        insert = _INSERT.search(sql)
        if insert:
            table, columns = insert.group(1).lower(), _names(insert.group(2))
            if table in owned:
                if OWNER not in columns:
                    found.append(TenancyViolation(
                        table, "write",
                        "INSERT does not supply `owner`", sql))
                target = _CONFLICT_TARGET.search(sql)
                if target is not None and OWNER not in _names(target.group(1)):
                    found.append(TenancyViolation(
                        table, "write",
                        f"conflict target ({target.group(1).strip()}) omits "
                        "`owner`, so a second tenant's write resolves against "
                        "the first tenant's row", sql))

        update = _UPDATE.search(sql)
        if update:
            table = update.group(1).lower()
            if table in owned:
                where = _WHERE.search(sql)
                if where is None or not re.search(rf"\b{OWNER}\b",
                                                  where.group(1),
                                                  re.IGNORECASE):
                    found.append(TenancyViolation(
                        table, "write",
                        "UPDATE is not scoped by `owner`", sql))
    return found


# --------------------------------------------------------------------------
# 3. consumers


_JOIN = re.compile(
    r"JOIN\s+(\w+)\s+ON\s+(.*?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT|$))",
    re.IGNORECASE | re.DOTALL)
_SUBQUERY_IN = re.compile(
    r"\(([^()]*?)\)\s+IN\s*\(\s*SELECT\s+(.*?)\s+FROM\s+(\w+)",
    re.IGNORECASE | re.DOTALL)
_SCALAR_IN = re.compile(
    r"(\w+)\s+IN\s*\(\s*SELECT\s+(\w+)\s+FROM\s+(\w+)",
    re.IGNORECASE | re.DOTALL)


def consumer_violations(statements: Iterable[str],
                        primary_keys: Mapping[str, Sequence[str]],
                        tenant_owned: Iterable[str]) -> List[TenancyViolation]:
    """Whether a join or subquery preserves the identity it references.

    The check the `OwnershipPath` defect needed. A composite key is no
    protection if the consumer still matches on the scalar identifier the key
    used to be: the query stays valid, keeps returning rows, and returns
    another tenant's.
    """
    found: List[TenancyViolation] = []
    owned = set(tenant_owned)

    for sql in statements:
        for table, predicate in _JOIN.findall(sql):
            table = table.lower()
            if table not in owned:
                continue
            key = tuple(primary_keys.get(table, ()))
            if OWNER not in key:
                continue
            if not re.search(rf"\b{table}\.{OWNER}\b", predicate, re.IGNORECASE):
                found.append(TenancyViolation(
                    table, "consumer",
                    f"JOIN matches on {predicate.strip()[:80]!r} and does not "
                    f"include {table}.{OWNER}; the referenced identity is "
                    f"{key}", sql))

        for columns, selected, table in _SUBQUERY_IN.findall(sql):
            table = table.lower()
            if table not in owned:
                continue
            key = tuple(primary_keys.get(table, ()))
            if OWNER in key and OWNER not in _names(selected):
                found.append(TenancyViolation(
                    table, "consumer",
                    f"subquery selects ({selected.strip()}) from {table}, "
                    f"whose identity is {key}", sql))

        for column, selected, table in _SCALAR_IN.findall(sql):
            table = table.lower()
            if table not in owned:
                continue
            key = tuple(primary_keys.get(table, ()))
            if OWNER in key and selected.lower() != OWNER:
                found.append(TenancyViolation(
                    table, "consumer",
                    f"subquery matches a single column ({selected}) against "
                    f"{table}, whose identity is {key}", sql))
    return found


def tenant_owned_tables() -> Tuple[str, ...]:
    """Every table carrying tenant records.

    Derived from the mutability classification rather than restated: a table
    that exists is classified there, so this cannot fall behind a new one.
    """
    return tuple(sorted(TABLE_MUTABILITY))
