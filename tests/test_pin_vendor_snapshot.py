"""The pin step (approach A, phase 2): a built base manifest + S3 upload records
become the manifest the pilot serves. Deterministic, no network."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# The script lives under scripts/, not an importable package.
_SPEC = importlib.util.spec_from_file_location(
    "pin_vendor_snapshot",
    Path(__file__).resolve().parents[1] / "scripts" / "pin_vendor_snapshot.py",
)
pin_vendor_snapshot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pin_vendor_snapshot)  # type: ignore[union-attr]
pin_manifest = pin_vendor_snapshot.pin_manifest
SnapshotPinError = pin_vendor_snapshot.SnapshotPinError


BASE = {
    "dataset_id": "market-data/prices",
    "snapshot_id": "prices-yahoo-2026-08-31",
    "kind": "licensed",
    "uri": "data/snapshots/prices-yahoo-2026-08-31.parquet",
    "schema_version": 1,
    "calendar": "calendar/nyse@1",
    "sessions": 2681,
    "assets": 48,
    "data_as_of": "2026-08-31",
    "coverage": {"start": "2016-01-04", "end": "2026-08-31"},
    "provider": "yahoo",
    "license_review_status": "OPEN",
}

UPLOADS = {
    "market": {
        "object_version_id": "MKT_VERSION_123",
        "sha256": "a" * 64,
        "content_digest": "mdv1:" + "b" * 64,
    },
    "total_return": {
        "object_version_id": "TR_VERSION_456",
        "sha256": "c" * 64,
    },
}


def _pin(base=BASE, uploads=UPLOADS, dataset="market-data/prices"):
    return pin_manifest(
        base, uploads,
        license_record="data/licensing/market-data-licensing@1.yaml",
        license_dataset_id=dataset,
        uploaded_at="2026-09-01",
    )


class TestPinsTheUploadedSnapshot:
    def test_carries_the_s3_versions_and_hashes(self):
        pinned = _pin()
        assert pinned["object_version_id"] == "MKT_VERSION_123"
        assert pinned["sha256"] == "a" * 64
        assert pinned["content_digest"] == "mdv1:" + "b" * 64
        assert pinned["content_digest_version"] == "mdv1"
        assert pinned["total_return_object_version_id"] == "TR_VERSION_456"
        assert pinned["total_return_sha256"] == "c" * 64

    def test_flips_the_manifest_to_a_served_vendor_pin(self):
        pinned = _pin()
        # kind + the env uri + the confirmed review are what turn a built file
        # into one the pilot may serve.
        assert pinned["kind"] == "vendor"
        assert pinned["uri"] == "${QUANTIFY_VENDOR_PRICES_URI}"
        assert pinned["license_review_status"] == "CONFIRMED"
        assert pinned["license_record"].endswith("market-data-licensing@1.yaml")
        # And it never leaks a local path.
        assert "data/snapshots" not in pinned["uri"]

    def test_preserves_the_reviewable_identity(self):
        pinned = _pin()
        assert pinned["snapshot_id"] == "prices-yahoo-2026-08-31"
        assert pinned["coverage"] == {"start": "2016-01-04", "end": "2026-08-31"}
        assert pinned["sessions"] == 2681 and pinned["assets"] == 48
        assert pinned["data_as_of"] == "2026-08-31"

    def test_is_deterministic(self):
        assert _pin() == _pin()


class TestRefusesWhatItCannotStandBehind:
    def test_refuses_a_dataset_the_licence_did_not_review(self):
        with pytest.raises(SnapshotPinError, match="different dataset"):
            _pin(dataset="market-data/some-other-set")

    def test_refuses_a_missing_total_return_twin(self):
        with pytest.raises(SnapshotPinError, match="total_return"):
            _pin(uploads={"market": UPLOADS["market"]})

    def test_refuses_a_market_upload_with_no_version(self):
        broken = {"market": {"sha256": "a" * 64, "content_digest": "mdv1:x"},
                  "total_return": UPLOADS["total_return"]}
        with pytest.raises(SnapshotPinError, match="object_version_id"):
            _pin(uploads=broken)
