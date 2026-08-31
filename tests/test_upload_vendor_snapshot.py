"""The vendor-snapshot upload PLAN (approach A, phase 2b): keys, twin derivation
and hashes, computed from local files with no S3."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "upload_vendor_snapshot",
    Path(__file__).resolve().parents[1] / "scripts" / "upload_vendor_snapshot.py",
)
upload_vendor_snapshot = importlib.util.module_from_spec(_SPEC)
# Register before exec so the frozen dataclass can resolve its own module's
# namespace (dataclasses look it up in sys.modules by __module__).
sys.modules[_SPEC.name] = upload_vendor_snapshot
_SPEC.loader.exec_module(upload_vendor_snapshot)  # type: ignore[union-attr]
plan_upload = upload_vendor_snapshot.plan_upload
UploadError = upload_vendor_snapshot.UploadError


def _snapshot(tmp_path: Path, snapshot_id: str = "prices-yahoo-2026-08-31"):
    frame = pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.5], "TLT": [90.0, 89.5, 90.2]},
        index=pd.to_datetime(["2026-08-27", "2026-08-28", "2026-08-31"]),
    )
    tr = frame * 1.01  # a different series — the reinvested twin
    (tmp_path / f"{snapshot_id}.parquet").write_bytes(b"")  # placeholder, overwritten below
    frame.to_parquet(tmp_path / f"{snapshot_id}.parquet")
    tr.to_parquet(tmp_path / f"{snapshot_id}.total-return.parquet")
    return snapshot_id


class TestPlan:
    def test_market_and_twin_go_to_the_derived_keys(self, tmp_path):
        sid = _snapshot(tmp_path)
        plans = plan_upload(tmp_path, sid, "s3://the-bucket/vendor/prices.parquet")
        by_role = {p.role: p for p in plans}

        assert by_role["market"].bucket == "the-bucket"
        assert by_role["market"].key == "vendor/prices.parquet"
        # The twin's key is the price key with the total-return suffix — exactly
        # what loader._twin_s3_uri derives, so the loader finds what we upload.
        assert by_role["total_return"].key == "vendor/prices.total-return.parquet"

    def test_only_the_market_series_carries_a_content_digest(self, tmp_path):
        sid = _snapshot(tmp_path)
        by_role = {p.role: p for p in plan_upload(tmp_path, sid, "s3://b/k/prices.parquet")}
        assert by_role["market"].content_digest and by_role["market"].content_digest.startswith("mdv1:")
        # The twin is checked by its own bytes only, never against the price digest.
        assert by_role["total_return"].content_digest is None
        assert by_role["total_return"].sha256 and by_role["market"].sha256

    def test_refuses_a_non_s3_uri(self, tmp_path):
        sid = _snapshot(tmp_path)
        with pytest.raises(UploadError, match="s3://"):
            plan_upload(tmp_path, sid, "/local/prices.parquet")

    def test_refuses_when_the_twin_is_missing(self, tmp_path):
        sid = "prices-yahoo-2026-08-31"
        pd.DataFrame({"SPY": [1.0]}).to_parquet(tmp_path / f"{sid}.parquet")
        with pytest.raises(UploadError, match="missing built file"):
            plan_upload(tmp_path, sid, "s3://b/k/prices.parquet")
