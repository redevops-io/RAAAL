"""Remove a snapshot and every retained version of it.

    python3 scripts/retire_market_data_snapshot.py prices-2025-11-19 --dry-run
    python3 scripts/retire_market_data_snapshot.py prices-2025-11-19 --confirm

Exists because versioning makes ordinary deletion misleading. `aws s3 rm` on a
versioned bucket writes a **delete marker**: the object disappears from listings
while every prior version is retained and retrievable. Someone told to "delete
the data" would do exactly that, see it gone, and report it removed — with the
bytes still there.

This enumerates every version *and* every delete marker under the snapshot
prefix, removes them, and then verifies the prefix is empty rather than assuming
the calls worked. It emits an audit record, because "we deleted it" is a claim
and the record is the evidence.

The likely reason to need this is a licence review concluding that private cloud
storage was not permitted. In that case the remedy is deletion and a written
record of it — not editing the history that shows the upload happened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

BUCKET = os.environ.get("QUANTIFY_MARKET_DATA_BUCKET", "quantify-club-market-data")
REGION = "us-east-1"
PREFIX = "market-data"
AUDIT_DIR = Path("data/manifests/retired")


def enumerate_versions(s3, snapshot_id: str) -> List[Dict[str, Any]]:
    """Every version and every delete marker under the snapshot.

    Both, deliberately. A prefix holding only delete markers still holds
    objects, and a cleanup that removed versions but left markers would report
    an empty listing over a non-empty prefix — the same illusion this script
    exists to dispel.
    """
    found: List[Dict[str, Any]] = []
    paginator = s3.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        for kind in ("Versions", "DeleteMarkers"):
            for entry in page.get(kind, []):
                if f"/{snapshot_id}/" not in entry["Key"]:
                    continue
                found.append({
                    "key": entry["Key"],
                    "version_id": entry["VersionId"],
                    "kind": "delete-marker" if kind == "DeleteMarkers" else "version",
                    "size": entry.get("Size"),
                    "last_modified": str(entry.get("LastModified")),
                })
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_id")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="enumerate what would be removed and change nothing")
    group.add_argument("--confirm", action="store_true",
                       help="actually remove every version and marker")
    parser.add_argument("--reason", default="",
                        help="why, for the audit record")
    args = parser.parse_args()

    import boto3                                                # noqa: PLC0415

    s3 = boto3.client("s3", region_name=REGION)
    entries = enumerate_versions(s3, args.snapshot_id)

    print(f"bucket   s3://{BUCKET}")
    print(f"snapshot {args.snapshot_id}")
    print(f"found    {len(entries)} object version(s) and marker(s)\n")
    for entry in entries:
        size = "" if entry["size"] is None else f"{entry['size'] / 1024:.0f} KiB"
        print(f"  {entry['kind']:13} {entry['key']}")
        print(f"                version {entry['version_id']}  {size}")

    if not entries:
        print("\nnothing to remove.")
        return 0

    if args.dry_run:
        print(f"\nDRY RUN — nothing removed. Re-run with --confirm to delete "
              f"all {len(entries)} of the above, permanently.")
        return 0

    deleted, failures = [], []
    for batch_start in range(0, len(entries), 1000):
        batch = entries[batch_start:batch_start + 1000]
        response = s3.delete_objects(
            Bucket=BUCKET,
            Delete={"Objects": [{"Key": e["key"], "VersionId": e["version_id"]}
                                for e in batch], "Quiet": False})
        deleted.extend(response.get("Deleted", []))
        failures.extend(response.get("Errors", []))

    remaining = enumerate_versions(s3, args.snapshot_id)
    print(f"\ndeleted {len(deleted)}, failed {len(failures)}, "
          f"remaining {len(remaining)}")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc)
    record = {
        "snapshot_id": args.snapshot_id,
        "bucket": BUCKET,
        "retired_at": stamp.isoformat(),
        "reason": args.reason or "not stated",
        "enumerated": len(entries),
        "deleted": len(deleted),
        "failed": len(failures),
        "remaining_after": len(remaining),
        "verified_empty": not remaining,
        "errors": failures,
        # Keys and versions, never bytes. The record says what was removed; it
        # is not a copy of what was removed.
        "removed": [{"key": e["key"], "version_id": e["version_id"]}
                    for e in entries],
    }
    audit = AUDIT_DIR / f"{args.snapshot_id}-{stamp.date()}.json"
    audit.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"audit record written to {audit}")

    if remaining or failures:
        print("\nPREFIX IS NOT EMPTY. Do not report this snapshot as removed.",
              file=sys.stderr)
        return 1
    print("\nverified: no versions or delete markers remain under this snapshot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
