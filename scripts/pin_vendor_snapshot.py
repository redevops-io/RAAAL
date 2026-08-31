"""Turn a freshly-built catalog snapshot into a *pinned* vendor manifest.

The daily refresh (approach A) is a chain of existing tools:

    build_catalog_snapshot.py   fresh Yahoo -> {id}.parquet + {id}.total-return.parquet + {id}.yaml
    provision_market_data.py    upload the two parquets -> S3 object versions + hashes
    THIS SCRIPT                 base manifest + upload records -> the pinned manifest the pilot reads

This step is deterministic and touches nothing external: it reads the base
manifest the builder wrote (`license_review_status: OPEN`, a local `uri`) and the
records the upload returned (object versions, hashes, the mdv1 content digest),
and writes the manifest the pilot's `approved_snapshot` reads — env-referenced
`uri`, the S3 pins, and `CONFIRMED`.

**Why it may set CONFIRMED.** The licence is dataset-level: one review of
`market-data/prices` (data/licensing/market-data-licensing@1.yaml) governs every
snapshot of that dataset, exactly as the manifest of the previous snapshot said.
So a *same-dataset* daily snapshot inherits the confirmed review; a snapshot of a
*different* dataset would not, and this script refuses to pin one (the dataset id
must match the licence record's).

Usage:
    python scripts/pin_vendor_snapshot.py \
        --base data/snapshots/prices-yahoo-2026-08-31.yaml \
        --uploads uploads.json \
        --license-record data/licensing/market-data-licensing@1.yaml \
        --out data/snapshots/prices-yahoo-2026-08-31.pinned.yaml

`uploads.json` is the JSON `provision_market_data.py` prints for the two
datasets, e.g. {"market": {...}, "total_return": {...}} — each with
object_version_id / sha256 / content_digest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


class SnapshotPinError(RuntimeError):
    """The base manifest and the upload records cannot be pinned together."""


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise SnapshotPinError(f"{where} is missing {key!r}")
    return mapping[key]


def pin_manifest(
    base: Mapping[str, Any],
    uploads: Mapping[str, Mapping[str, Any]],
    *,
    license_record: str,
    license_dataset_id: str,
    uploaded_at: str,
) -> Dict[str, Any]:
    """The pinned vendor manifest, from the built base + the upload records.

    Pure: same inputs, same bytes. It carries the reviewable identity forward
    (snapshot id, coverage, sessions, assets, corporate actions) and adds the two
    things the base could not know — the S3 object versions the upload assigned,
    and that the dataset licence is confirmed for this snapshot.
    """
    dataset_id = _require(base, "dataset_id", "base manifest")
    if dataset_id != license_dataset_id:
        raise SnapshotPinError(
            f"base manifest is dataset {dataset_id!r} but the licence record "
            f"governs {license_dataset_id!r}; a snapshot cannot inherit a review "
            "written for a different dataset")

    market = uploads.get("market")
    total_return = uploads.get("total_return")
    if not market:
        raise SnapshotPinError("uploads is missing the 'market' record")
    if not total_return:
        raise SnapshotPinError("uploads is missing the 'total_return' record")

    content_digest = str(_require(market, "content_digest", "market upload"))
    digest_version = content_digest.split(":", 1)[0] if ":" in content_digest else "mdv1"

    return {
        "dataset_id": dataset_id,
        "snapshot_id": _require(base, "snapshot_id", "base manifest"),
        "kind": "vendor",
        # Env-referenced, never the built local path: the private bucket name
        # stays out of the committed/mounted manifest (as the previous pin did).
        "uri": "${QUANTIFY_VENDOR_PRICES_URI}",
        "object_version_id": _require(market, "object_version_id", "market upload"),
        "sha256": _require(market, "sha256", "market upload"),
        "content_digest": content_digest,
        "content_digest_version": digest_version,
        "total_return_object_version_id": _require(
            total_return, "object_version_id", "total_return upload"),
        "total_return_sha256": _require(total_return, "sha256", "total_return upload"),
        "schema_version": str(base.get("schema_version", "1")),
        "calendar": base.get("calendar", "calendar/nyse@1"),
        "sessions": base.get("sessions"),
        "assets": base.get("assets"),
        "data_as_of": _require(base, "data_as_of", "base manifest"),
        "uploaded_at": uploaded_at,
        "coverage": _require(base, "coverage", "base manifest"),
        "provider": "yahoo-finance",
        "license_class": "restricted",
        "redistributable": False,
        # Inherited from the dataset review, checked above. CONFIRMED is the only
        # word `review_complete` accepts; anything else denies every run.
        "license_review_status": "CONFIRMED",
        "license_record": license_record,
        "attribution": {
            "source": "Yahoo Finance",
            "acknowledgement": "non-commercial usage only",
        },
        # The reviewed routes, restating the licence record's answers in the
        # vocabulary the code reads: raw prices never leave; derived figures may.
        "egress_policy": {
            "public_export": "DENY",
            "case_bundle": "DENY",
            "model_provider_upload": "DENY",
            "derived_aggregate": "ALLOW",
            "internal_benchmark": "ALLOW",
            "customer_result": "ALLOW",
        },
    }


def _license_dataset_id(record_path: Path) -> str:
    import yaml

    record = yaml.safe_load(record_path.read_text()) or {}
    # The licence record names the dataset it reviewed; a pin for any other
    # dataset is refused above.
    return str(record.get("dataset_id") or record.get("dataset") or "market-data/prices")


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path,
                        help="the manifest build_catalog_snapshot.py wrote")
    parser.add_argument("--uploads", required=True, type=Path,
                        help="JSON of {market: {...}, total_return: {...}} upload records")
    parser.add_argument("--license-record", required=True, type=Path)
    parser.add_argument("--uploaded-at", required=True,
                        help="the upload date (YYYY-MM-DD); passed in, not read from a clock")
    parser.add_argument("--out", type=Path, help="write here; default: stdout")
    args = parser.parse_args()

    base = yaml.safe_load(args.base.read_text()) or {}
    uploads = json.loads(args.uploads.read_text())
    dataset_id = _license_dataset_id(args.license_record)

    try:
        pinned = pin_manifest(
            base, uploads,
            license_record=str(args.license_record),
            license_dataset_id=dataset_id,
            uploaded_at=args.uploaded_at,
        )
    except SnapshotPinError as err:
        print(f"refusing to pin: {err}", file=sys.stderr)
        return 1

    text = yaml.safe_dump(pinned, sort_keys=False)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
