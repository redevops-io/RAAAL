"""Moving a workspace between engines, without trusting either one's storage.

    SQLite -> validated neutral bundle -> PostgreSQL -> independent verification

**Neither database's physical representation is the interchange format.** SQLite
holds `"152.26"` as text and PostgreSQL holds `152.260000000000` as NUMERIC.
Copying either spelling would make the bundle a dialect artifact, and a
comparison of the two would report differences that are not differences. The
bundle carries canonical application values — decimal strings, canonical UTC
timestamps, JSON objects rather than encoded JSON text — and both sides are
verified against *that*.

**A migration must not faithfully copy corruption and call it success.** Export
refuses if any content hash, decimal mirror, ownership path or foreign key is
already broken at the source. Moving a corrupt row is not preservation; it
launders a defect into a database that has no record of where it came from.

**Import never overwrites.** Absent is an insert, an identical body is
redelivery, and a divergent body is a conflict that aborts the whole import —
the same rule the artifact tables enforce at runtime. An overwrite-style upsert
would silently resolve exactly the case that means the bundle and the target
disagree about history.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .decimals import canonical, to_decimal
from .engine import Database
from .mutability import TABLE_MUTABILITY
from .schema import SCHEMA_VERSION, deletion_order, metadata, primary_key_columns
from .types import Json, loads

#: Bumped when the bundle's shape changes. An importer reading an unknown
#: version stops rather than guessing which fields moved.
BUNDLE_FORMAT_VERSION = "quantify-transfer@1"

#: Columns holding JSON, per table. Read from the model so a new JSON column
#: joins the export by being declared rather than by being remembered.
def _json_columns(table: str) -> Sequence[str]:
    from .types import JsonText

    return tuple(column.name for column in metadata.tables[table].columns
                 if isinstance(column.type, JsonText))


def _decimal_columns(table: str) -> Sequence[str]:
    from .types import DecimalText

    return tuple(column.name for column in metadata.tables[table].columns
                 if isinstance(column.type, DecimalText))


class ExportRefused(RuntimeError):
    """The source is not in a state worth copying."""


class ImportRefused(RuntimeError):
    """The target cannot accept this bundle."""


class BundleUnreadable(RuntimeError):
    """The bundle is not one this build knows how to read."""


@dataclass
class TransferPlan:
    """What an import would do, computed before anything is written.

    The dry run and the real import consume this same object, so a dry run
    cannot report one thing and the import do another — they would otherwise
    each rediscover the plan and could disagree.
    """

    to_insert: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    redeliveries: Dict[str, int] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    unknown_tables: List[str] = field(default_factory=list)

    @property
    def ready(self) -> int:
        return sum(len(rows) for rows in self.to_insert.values())

    @property
    def redelivered(self) -> int:
        return sum(self.redeliveries.values())

    def summary(self) -> Dict[str, Any]:
        return {"rows_ready": self.ready, "redeliveries": self.redelivered,
                "conflicts": len(self.conflicts),
                "unknown_tables": list(self.unknown_tables)}


# --------------------------------------------------------------------------
# canonical record form


def _canonical_row(table: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    """One row as the application means it, not as a database spelled it."""
    record: Dict[str, Any] = {}
    json_columns = set(_json_columns(table))
    decimal_columns = set(_decimal_columns(table))

    for name in sorted(row.keys()):
        value = row[name]
        if name in json_columns:
            record[name] = loads(value)
        elif name in decimal_columns:
            # One spelling per quantity, whichever engine produced it.
            #
            # SQLite returns the canonical text it stored — "152.26" — and
            # PostgreSQL returns NUMERIC(38, 12) padded to scale,
            # Decimal("152.260000000000"). Both are the same quantity and
            # neither is more correct, so the bundle records the minimal form
            # and the digests match across a round trip. The recorded precision
            # is not lost: it lives in the payload, which is authoritative and
            # is carried verbatim. These columns are its query-friendly mirror,
            # and PostgreSQL cannot preserve their precision in any case.
            record[name] = _bundle_decimal(value)
        elif isinstance(value, Decimal):
            record[name] = canonical(value)
        else:
            record[name] = value
    return record


def _bundle_decimal(value: Any) -> Optional[str]:
    """The minimal canonical spelling of a quantity, or None."""
    if value is None:
        return None
    number = to_decimal(value)
    if number is None:
        return None
    normalized = number.normalize()
    # `normalize` may produce an exponent (1E+3); `f` expands it.
    return format(normalized, "f")


def digest_of(records: Sequence[Mapping[str, Any]]) -> str:
    """A content digest over canonical records.

    Order-independent by construction: rows are hashed individually and the
    hashes sorted, so a differing `ORDER BY` between engines is not a
    difference in content.
    """
    import hashlib

    per_row = sorted(
        hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":"),
                       default=str).encode()).hexdigest()
        for record in records)
    return hashlib.sha256("".join(per_row).encode()).hexdigest()


# --------------------------------------------------------------------------
# export


def _validate_source(store) -> List[str]:
    """Everything that must already be true before a row is copied."""
    from ..workspace.retention import WORKSPACE_RECORDS
    from ..workspace.store import (
        HASHED_ARTIFACTS,
        MIRRORED_DECIMALS,
        verify_content_hashes,
        verify_decimal_columns,
    )

    problems: List[str] = []

    for table in HASHED_ARTIFACTS:
        for bad in verify_content_hashes(store, table):
            problems.append(
                f"{table} {bad['id']}: stored content hash does not match its "
                "payload")
    for table in MIRRORED_DECIMALS:
        for bad in verify_decimal_columns(store, table):
            problems.append(
                f"{table}.{bad['column']}: disagrees with payload field "
                f"{bad['field']} ({bad['stored']!r} vs {bad['payload']!r})")

    present = set(store.db.existing_tables())
    unclassified = present - set(WORKSPACE_RECORDS) - {"alembic_version"}
    unclassified = {name for name in unclassified if name in metadata.tables}
    if unclassified:
        problems.append(
            f"tables present and unclassified: {sorted(unclassified)}")
    return problems


def _data_boundary() -> Dict[str, Any]:
    """What the exported figures are, stated in the bundle itself."""
    from ..deploy.context import current
    from ..market_data.pilot_policy import PilotDataPolicy

    policy = current().market_data.policy
    synthetic = policy is PilotDataPolicy.SYNTHETIC_ONLY
    return {
        "policy": getattr(policy, "value", None),
        "synthetic": synthetic,
        "notice": ("Pilot mode uses synthetic market data. Results are for "
                   "product evaluation only and are not based on licensed "
                   "live market data.") if synthetic else "",
    }


def export_bundle(store, *, commit: str = "", exported_at: str,
                  owner: Optional[str] = None) -> Dict[str, Any]:
    """Read a workspace into a neutral bundle, or refuse.

    `owner` narrows the export to one tenant; omitted, it takes everything.
    """
    problems = _validate_source(store)
    if problems:
        raise ExportRefused(
            "the source workspace is not consistent, and copying it would move "
            "the inconsistency into a database with no record of where it came "
            "from:\n  " + "\n  ".join(problems))

    records: Dict[str, List[Dict[str, Any]]] = {}
    counts: Dict[str, int] = {}
    digests: Dict[str, str] = {}

    with store._conn() as conn:
        for table in deletion_order():
            sql = f"SELECT * FROM {table}"
            params: Sequence[Any] = ()
            if owner is not None and "owner" in metadata.tables[table].columns:
                sql += " WHERE owner = ?"
                params = (owner,)
            rows = [_canonical_row(table, dict(row))
                    for row in conn.execute(sql, params).fetchall()]
            records[table] = rows
            counts[table] = len(rows)
            digests[table] = digest_of(rows)

    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "source_commit": commit,
        "source_schema_version": SCHEMA_VERSION,
        "exported_at": exported_at,
        "owner_scope": owner,
        "tables": list(deletion_order()),
        "counts": counts,
        "digests": digests,
        "canonicalization": {
            "decimal": "decimal/plain@1",
            "temporal": "temporal/utc-iso@1",
        },
        # The data boundary travels with the bundle. An export is the one
        # artifact that outlives the screen it was read on: a file of
        # realistic-looking series with no statement of what they are is the
        # most likely thing to be mistaken for historical analysis, and it is
        # the copy nobody can add a caveat to afterwards.
        "market_data": _data_boundary(),
    }
    manifest["bundle_digest"] = digest_of([manifest["digests"]])
    return {"manifest": manifest, "records": records}


def verify_bundle(bundle: Mapping[str, Any]) -> None:
    """Refuse a bundle this build cannot read, or one that has been edited."""
    manifest = bundle.get("manifest") or {}
    version = manifest.get("format_version")
    if version != BUNDLE_FORMAT_VERSION:
        raise BundleUnreadable(
            f"bundle format {version!r} is not {BUNDLE_FORMAT_VERSION!r}. "
            "Fields may have moved, and guessing which would import wrong data "
            "under a correct-looking manifest")

    for table, expected in (manifest.get("counts") or {}).items():
        actual = len(bundle["records"].get(table, ()))
        if actual != expected:
            raise BundleUnreadable(
                f"{table}: manifest says {expected} rows and the bundle holds "
                f"{actual}. The manifest was written at export and one of the "
                "two has been changed since")

    for table, expected in (manifest.get("digests") or {}).items():
        actual = digest_of(bundle["records"].get(table, ()))
        if actual != expected:
            raise BundleUnreadable(
                f"{table}: content digest does not match the manifest")

    if digest_of([manifest["digests"]]) != manifest.get("bundle_digest"):
        raise BundleUnreadable("the bundle digest does not cover its digests")


# --------------------------------------------------------------------------
# import


def plan_import(target: Database, bundle: Mapping[str, Any]) -> TransferPlan:
    """What the import would do, without doing any of it."""
    verify_bundle(bundle)
    plan = TransferPlan()

    existing_tables = set(target.existing_tables())
    conn = target.connect()
    try:
        for table in bundle["manifest"]["tables"]:
            rows = bundle["records"].get(table, [])
            if table not in existing_tables:
                if rows:
                    plan.unknown_tables.append(table)
                continue

            keys = primary_key_columns(table)
            insert: List[Dict[str, Any]] = []
            redelivered = 0
            for record in rows:
                predicate = " AND ".join(f"{name} = ?" for name in keys)
                found = conn.execute(
                    f"SELECT * FROM {table} WHERE {predicate}",
                    tuple(record[name] for name in keys)).fetchone()
                if found is None:
                    insert.append(record)
                    continue
                if _canonical_row(table, dict(found)) == record:
                    redelivered += 1
                else:
                    plan.conflicts.append({
                        "table": table,
                        "identity": {name: record[name] for name in keys},
                        "reason": "a row with this identity is already stored "
                                  "with a different body"})
            plan.to_insert[table] = insert
            plan.redeliveries[table] = redelivered
    finally:
        conn.close()
    return plan


def apply_import(target: Database, bundle: Mapping[str, Any],
                 plan: Optional[TransferPlan] = None) -> TransferPlan:
    """Write a planned import, all of it or none.

    Takes the plan the dry run produced rather than recomputing one, so the
    two cannot disagree about what was going to happen.
    """
    plan = plan if plan is not None else plan_import(target, bundle)
    if plan.conflicts:
        raise ImportRefused(
            f"{len(plan.conflicts)} row(s) already exist with a different "
            "body. Nothing was written; resolve the divergence rather than "
            "letting an import decide which history is real")
    if plan.unknown_tables:
        raise ImportRefused(
            f"the bundle carries rows for tables this database does not have: "
            f"{plan.unknown_tables}. Migrate the target first")

    json_by_table = {table: set(_json_columns(table))
                     for table in bundle["manifest"]["tables"]}

    conn = target.connect()
    try:
        # The manifest lists tables in *deletion* order — dependents first,
        # which is what `RESTRICT` requires on the way out. Insertion is the
        # exact opposite: a reconciliation cannot be written before the events
        # it references exist. Reversing the same declared order keeps one
        # source for both directions rather than a second hand-kept list.
        for table in reversed(list(bundle["manifest"]["tables"])):
            for record in plan.to_insert.get(table, ()):
                names = sorted(record)
                values = [Json(record[name]) if name in json_by_table[table]
                          and record[name] is not None else record[name]
                          for name in names]
                conn.execute(
                    f"INSERT INTO {table} ({', '.join(names)}) "
                    f"VALUES ({', '.join('?' for _ in names)})",
                    tuple(values))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return plan


def verify_import(target: Database, bundle: Mapping[str, Any]
                  ) -> List[str]:
    """Compare the target against the bundle, semantically.

    Physical spellings differ between engines by design, so this compares
    canonical records and content digests rather than raw values — a byte
    comparison would report `152.26` against `152.260000000000` as data loss.
    """
    problems: List[str] = []
    manifest = bundle["manifest"]
    owner = manifest.get("owner_scope")

    conn = target.connect()
    try:
        for table in manifest["tables"]:
            sql = f"SELECT * FROM {table}"
            params: Sequence[Any] = ()
            if owner is not None and "owner" in metadata.tables[table].columns:
                sql += " WHERE owner = ?"
                params = (owner,)
            rows = [_canonical_row(table, dict(row))
                    for row in conn.execute(sql, params).fetchall()]

            if len(rows) != manifest["counts"][table]:
                problems.append(
                    f"{table}: {len(rows)} rows in the target, "
                    f"{manifest['counts'][table]} in the bundle")
            actual = digest_of(rows)
            if actual != manifest["digests"][table]:
                problems.append(
                    f"{table}: content digest differs from the bundle")
    finally:
        conn.close()
    return problems
