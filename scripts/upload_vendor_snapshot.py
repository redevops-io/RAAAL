"""Upload a built vendor snapshot to S3 and emit the records the pin step reads.

Approach A, phase 2b. `build_catalog_snapshot.py` writes two parquets locally —
the market series `{id}.parquet` and the dividend-reinvested twin
`{id}.total-return.parquet`. This uploads BOTH to the versioned private bucket at
the keys the loader derives (`src/market_data/loader._twin_s3_uri`), and prints
the object versions + hashes as the JSON `pin_vendor_snapshot.py` consumes.

**Not `provision_market_data.py`.** That one uploads the dashboard's
`data/history/*` under per-snapshot keys. This uploads the *vendor* snapshot to
the single fixed key `QUANTIFY_VENDOR_PRICES_URI` (and its `.total-return`
sibling): the bucket is versioned, so each day is a new *version* of that key and
the manifest pins the exact version — the immutability the loader enforces.

Usage:
    QUANTIFY_VENDOR_PRICES_URI=s3://bucket/vendor/prices.parquet \
    python scripts/upload_vendor_snapshot.py \
        --snapshot-dir data/snapshots --snapshot-id prices-yahoo-2026-08-31 \
        --out uploads.json                       # add --dry-run to change nothing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# scripts/ is on sys.path at REPO_ROOT when run as a script; import the app's own
# integrity helpers so the digest matches exactly what the loader verifies.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.market_data.integrity import content_digest, file_sha256  # noqa: E402

REGION = "us-east-1"


class UploadError(RuntimeError):
    pass


def _split_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise UploadError(f"QUANTIFY_VENDOR_PRICES_URI must be an s3:// URI, got {uri!r}")
    without = uri[len("s3://"):]
    bucket, _, key = without.partition("/")
    if not bucket or not key:
        raise UploadError(f"{uri!r} is not s3://bucket/key")
    return bucket, key


def _twin_key(price_key: str) -> str:
    """The twin's key: the price key with the total-return suffix, mirroring
    `loader._twin_s3_uri` so the loader finds what this uploads."""
    if not price_key.endswith(".parquet"):
        raise UploadError(f"the price key must end .parquet, got {price_key!r}")
    return price_key[: -len(".parquet")] + ".total-return.parquet"


@dataclass(frozen=True)
class ObjectPlan:
    role: str            # "market" | "total_return"
    local: Path
    bucket: str
    key: str
    sha256: str
    content_digest: Optional[str]   # only the market series carries one


def plan_upload(snapshot_dir: Path, snapshot_id: str, price_uri: str) -> list[ObjectPlan]:
    """What will be uploaded and its hashes — computed from local files, no S3.

    Separated from the upload so the keys and digests are testable offline: a
    wrong twin key or a mismatched digest is a bug caught here, not in production.
    """
    import pandas as pd

    bucket, price_key = _split_s3_uri(price_uri)
    market_local = snapshot_dir / f"{snapshot_id}.parquet"
    twin_local = snapshot_dir / f"{snapshot_id}.total-return.parquet"
    for path in (market_local, twin_local):
        if not path.exists():
            raise UploadError(f"missing built file {path} — run build_catalog_snapshot.py first")

    market_frame = pd.read_parquet(market_local)
    return [
        ObjectPlan("market", market_local, bucket, price_key,
                   file_sha256(market_local), content_digest(market_frame)),
        ObjectPlan("total_return", twin_local, bucket, _twin_key(price_key),
                   file_sha256(twin_local), None),
    ]


def _put(s3, plan: ObjectPlan, snapshot_id: str) -> str:
    metadata = {"snapshot-id": snapshot_id, "sha256": plan.sha256,
                "license-class": "restricted"}
    if plan.content_digest:
        metadata["content-digest"] = plan.content_digest
    with open(plan.local, "rb") as handle:
        response = s3.put_object(
            Bucket=plan.bucket, Key=plan.key, Body=handle,
            ServerSideEncryption="AES256", ChecksumAlgorithm="SHA256",
            Metadata=metadata)
    return response["VersionId"]


def run(snapshot_dir: Path, snapshot_id: str, price_uri: str, *, dry_run: bool) -> Dict[str, Any]:
    plans = plan_upload(snapshot_dir, snapshot_id, price_uri)

    s3 = None
    if not dry_run:
        import boto3
        s3 = boto3.client("s3", region_name=REGION)

    records: Dict[str, Any] = {}
    for plan in plans:
        version = "DRY-RUN" if dry_run else _put(s3, plan, snapshot_id)
        print(f"  {'[dry-run] ' if dry_run else ''}s3://{plan.bucket}/{plan.key}")
        print(f"      sha256  {plan.sha256}")
        if plan.content_digest:
            print(f"      content {plan.content_digest}")
        print(f"      version {version}")
        record: Dict[str, Any] = {"object_version_id": version, "sha256": plan.sha256}
        if plan.content_digest:
            record["content_digest"] = plan.content_digest
        records[plan.role] = record
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=REPO_ROOT / "data" / "snapshots")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--price-uri", default=os.environ.get("QUANTIFY_VENDOR_PRICES_URI", ""),
                        help="s3://bucket/key of the price parquet; default $QUANTIFY_VENDOR_PRICES_URI")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, help="write the upload records JSON here (for the pin step)")
    args = parser.parse_args()

    if not args.price_uri:
        print("set QUANTIFY_VENDOR_PRICES_URI or pass --price-uri", file=sys.stderr)
        return 1

    try:
        records = run(args.snapshot_dir, args.snapshot_id, args.price_uri, dry_run=args.dry_run)
    except UploadError as err:
        print(f"upload refused: {err}", file=sys.stderr)
        return 1

    if args.out:
        args.out.write_text(json.dumps(records, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
