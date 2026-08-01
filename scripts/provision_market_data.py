"""Create the private market-data bucket and upload a pinned snapshot.

    python3 scripts/provision_market_data.py --dry-run     # show the plan
    python3 scripts/provision_market_data.py               # do it

Idempotent. Re-running against an existing bucket re-checks the settings rather
than assuming them, because a bucket created correctly once can be changed
later, and "we set that up properly" is a memory rather than a fact.

Every object key contains its snapshot id, so nothing is ever overwritten. That
is the property the whole market-data design rests on: an ordinary path that can
be rewritten means the same commit against the same URI can produce a different
result, and every recorded figure silently starts describing data the run never
saw.

The bucket name is written to a gitignored env file rather than the manifest.
The manifest carries snapshot identity, object version and hashes — the things
that must be reviewable — while the bucket name stays out of a repository that
is published under AGPL.

**This uploads licensed data.** Private storage does not cure a licence that
forbids server-side redistribution or multi-user serving; see `licensing_review`
in data/manifests/prices-production.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.market_data.integrity import content_digest, file_sha256  # noqa: E402

REGION = "us-east-1"
BUCKET = os.environ.get("QUANTIFY_MARKET_DATA_BUCKET", "quantify-club-market-data")
PREFIX = "market-data"
SOURCE = Path("data/history")
ENV_FILE = Path(".env.market-data")

#: What gets uploaded, and under which dataset. Only these — the directory also
#: holds a 6 MB summary JSON that is a rendering artifact, not an input.
DATASETS = {
    "prices.parquet": "prices",
    "weights.parquet": "weights",
    "timeline.parquet": "timeline",
    "fomo_indicator.parquet": "fomo-indicator",
}

#: Refuse plaintext transport. A bucket policy is checked by AWS on every
#: request; a convention is checked by whoever remembers it.
TLS_ONLY = {
    "Version": "2012-10-17",
    "Statement": [{
        "Sid": "DenyInsecureTransport",
        "Effect": "Deny",
        "Principal": "*",
        "Action": "s3:*",
        "Resource": [f"arn:aws:s3:::{BUCKET}", f"arn:aws:s3:::{BUCKET}/*"],
        "Condition": {"Bool": {"aws:SecureTransport": "false"}},
    }],
}


def ensure_bucket(s3, *, dry_run: bool) -> None:
    from botocore.exceptions import ClientError

    try:
        s3.head_bucket(Bucket=BUCKET)
        exists = True
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket"}:
            raise
        exists = False

    steps = [
        ("create bucket", lambda: s3.create_bucket(Bucket=BUCKET)) if not exists
        else ("bucket exists", None),
        ("block all public access", lambda: s3.put_public_access_block(
            Bucket=BUCKET,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True, "IgnorePublicAcls": True,
                "BlockPublicPolicy": True, "RestrictPublicBuckets": True})),
        # Versioning must be on *before* the first upload. Enabling it later
        # leaves the earliest objects without version ids, and those are exactly
        # the ones the oldest recorded runs point at.
        ("enable versioning", lambda: s3.put_bucket_versioning(
            Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})),
        ("enable default encryption", lambda: s3.put_bucket_encryption(
            Bucket=BUCKET,
            ServerSideEncryptionConfiguration={"Rules": [{
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"},
                "BucketKeyEnabled": True}]})),
        ("deny non-TLS requests", lambda: s3.put_bucket_policy(
            Bucket=BUCKET, Policy=json.dumps(TLS_ONLY))),
    ]

    for label, action in steps:
        if action is None or dry_run:
            print(f"  [{'dry-run' if dry_run else 'skip'}] {label}")
            continue
        action()
        print(f"  [done] {label}")


def upload(s3, snapshot_id: str, *, dry_run: bool) -> List[Dict[str, str]]:
    records = []
    for filename, dataset in DATASETS.items():
        path = SOURCE / filename
        if not path.exists():
            print(f"  [absent] {path} — skipped")
            continue

        # The snapshot id is in the key, so a later snapshot is a new object
        # rather than a new version of this one. Versioning is the safety net,
        # not the addressing scheme.
        key = f"{PREFIX}/{dataset}/{snapshot_id}/{filename}"
        digest = file_sha256(path)
        try:
            frame = pd.read_parquet(path)
            content = content_digest(frame)
            shape = f"{len(frame)}x{frame.shape[1]}"
        except Exception as exc:                                # noqa: BLE001
            content, shape = f"unreadable: {exc}", "?"

        print(f"  {'[dry-run] ' if dry_run else ''}s3://{BUCKET}/{key}")
        print(f"      {path.stat().st_size / 1024:.0f} KiB  {shape}")
        print(f"      sha256  {digest}")
        print(f"      content {content}")

        version_id = "DRY-RUN"
        if not dry_run:
            with open(path, "rb") as handle:
                response = s3.put_object(
                    Bucket=BUCKET, Key=key, Body=handle,
                    ServerSideEncryption="AES256",
                    ChecksumAlgorithm="SHA256",
                    Metadata={"snapshot-id": snapshot_id,
                              "content-digest": content,
                              "sha256": digest,
                              "license-class": "restricted"})
            version_id = response["VersionId"]
            print(f"      version {version_id}")

        records.append({
            "dataset": dataset, "key": key, "uri": f"s3://{BUCKET}/{key}",
            "object_version_id": version_id, "sha256": digest,
            "content_digest": content,
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and change nothing")
    parser.add_argument("--snapshot-id", default=None,
                        help="defaults to prices-<latest session in the data>")
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"no local snapshot at {SOURCE}. Produce one first:\n"
              "  python3 -m src.history --start 2015-01-01 "
              "--end $(date +%F) --step 5", file=sys.stderr)
        return 1

    snapshot_id = args.snapshot_id
    if snapshot_id is None:
        frame = pd.read_parquet(SOURCE / "prices.parquet")
        snapshot_id = f"prices-{frame.index.max().date()}"

    print(f"bucket    s3://{BUCKET}  ({REGION})")
    print(f"snapshot  {snapshot_id}")
    print(f"mode      {'DRY RUN — nothing will change' if args.dry_run else 'LIVE'}\n")

    s3 = None
    if not args.dry_run:
        import boto3                                            # noqa: PLC0415
        s3 = boto3.client("s3", region_name=REGION)

    print("bucket configuration:")
    if args.dry_run:
        for label in ("create bucket", "block all public access",
                      "enable versioning", "enable default encryption",
                      "deny non-TLS requests"):
            print(f"  [dry-run] {label}")
    else:
        ensure_bucket(s3, dry_run=False)

    print("\nobjects:")
    records = upload(s3, snapshot_id, dry_run=args.dry_run)

    prices = next((r for r in records if r["dataset"] == "prices"), None)
    if prices and not args.dry_run:
        ENV_FILE.write_text(
            "# Written by scripts/provision_market_data.py. Gitignored: the\n"
            "# bucket name stays out of a repository published under AGPL,\n"
            "# while the manifest carries the reviewable snapshot identity.\n"
            f'export QUANTIFY_MARKET_DATA_URI="{prices["uri"]}"\n'
            f'export QUANTIFY_MARKET_DATA_VERSION_ID="{prices["object_version_id"]}"\n'
            f'export QUANTIFY_MARKET_DATA_SHA256="{prices["sha256"]}"\n')
        print(f"\nwrote {ENV_FILE}")

    print("\nmanifest values for data/manifests/prices-production.yaml:")
    if prices:
        print(f"  snapshot_id: {snapshot_id}")
        print(f"  object_version_id: {prices['object_version_id']}")
        print(f"  sha256: {prices['sha256']}")
        print(f"  content_digest: \"{prices['content_digest']}\"")

    print("\nThe licensing review in that manifest is still UNCONFIRMED. "
          "Private storage does not cure a licence that forbids server-side\n"
          "redistribution or multi-user serving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
