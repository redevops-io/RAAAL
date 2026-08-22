"""The serving adapter honours the deployment's data policy (vendor path).

The adapter used to hardcode the synthetic fixture, so a deployment on the
approved vendor policy served invented series while its banner promised vendor
data. These tests pin the fix: the policy decides which snapshot the adapter
serves, at the one place that turns a snapshot into observations, and the vendor
policy fails closed rather than falling back to synthetic.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.market_data.adapters import AdapterRefused, LocalParquetAdapter

VENDOR = "market-data-egress/pilot-vendor-approved@1"


def _context(monkeypatch, policy: str):
    """Resolve a deployment context with the given data policy and install it."""
    from src.deploy import context as deploy_context

    for name, value in (("PILOT_DATA_POLICY", policy),
                        ("QUANTIFY_PILOT_READER", "recorded"),
                        ("QUANTIFY_PARSER_MODE", "RUNTIME"),
                        ("QUANTIFY_PARSER_MODEL", "claude-sonnet-5"),
                        ("ANTHROPIC_API_KEY", "unused"),
                        ("QUANTIFY_SNAPSHOT_ROOT", "/tmp/snap")):
        monkeypatch.setenv(name, value)
    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)


class _FakeSnapshot:
    snapshot_id = "prices-yahoo-test"
    uri = "s3://bucket/market-data/prices/prices-yahoo-test.parquet"
    license_class = "restricted"
    license_review_status = "CONFIRMED"
    dataset_id = "market-data/prices"
    data_as_of = "2026-08-22"
    calendar = "calendar/nyse@1"


def test_synthetic_policy_serves_the_fixture_offline(monkeypatch):
    _context(monkeypatch, "SYNTHETIC_ONLY")
    seen = {}

    def fake_load_prices(snapshot, *, reinvested=False, allow_network=False, **kw):
        seen["allow_network"] = allow_network
        seen["snapshot_id"] = getattr(snapshot, "snapshot_id", "")
        return pd.DataFrame({"VTI": [1.0, 2.0]})

    monkeypatch.setattr("src.market_data.loader.load_prices", fake_load_prices)
    LocalParquetAdapter().fetch(["VTI"])
    assert seen["allow_network"] is False            # synthetic is local, no network
    assert seen["snapshot_id"] != "prices-yahoo-test"


def test_vendor_policy_serves_the_approved_snapshot_over_the_network(monkeypatch):
    _context(monkeypatch, VENDOR)
    seen = {}

    def fake_load_prices(snapshot, *, reinvested=False, allow_network=False, **kw):
        seen["allow_network"] = allow_network
        seen["snapshot"] = snapshot
        return pd.DataFrame({"VTI": [1.0, 2.0]})

    monkeypatch.setattr("src.market_data.loader.load_prices", fake_load_prices)
    monkeypatch.setattr("src.market_data.access.approved_snapshot",
                        lambda: _FakeSnapshot())

    fetched = LocalParquetAdapter().fetch(["VTI"])
    assert seen["allow_network"] is True             # vendor snapshot is fetched from S3
    assert seen["snapshot"].snapshot_id == "prices-yahoo-test"
    # the vendor snapshot's own provenance travels with the observations
    assert fetched.license_class == "restricted"
    assert fetched.license_review_status == "CONFIRMED"
    assert fetched.dataset_id == "market-data/prices"


def test_vendor_policy_fails_closed_when_no_snapshot_is_approved(monkeypatch):
    _context(monkeypatch, VENDOR)
    # the licensing record is incomplete / the manifest is missing → None
    monkeypatch.setattr("src.market_data.access.approved_snapshot", lambda: None)

    with pytest.raises(AdapterRefused) as refusal:
        LocalParquetAdapter().fetch(["VTI"])
    # it refuses rather than silently serving synthetic under a vendor policy
    assert "synthetic" in str(refusal.value).lower()


# --- loader: the S3 dividend-reinvested twin (freeze the fetch semantics) ----

import pandas as _pd
from src.market_data.loader import SnapshotUnavailable, load_prices
from src.market_data.integrity import IntegrityError, file_sha256


def _s3_snapshot(**over):
    from src.market_data.loader import Snapshot
    base = dict(
        dataset_id="market-data/prices", snapshot_id="prices-yahoo-test",
        kind="vendor",
        uri="s3://bucket/market-data/prices/prices-yahoo-test/prices-yahoo-test.parquet",
        schema_version="1", object_version_id="v-price",
        sha256="x", content_digest="mdv1:x")
    base.update(over)
    return Snapshot(**base)


def _write_twin(cache_dir):
    frame = _pd.DataFrame({"SPY": [1.0, 2.0], "VTI": [3.0, 4.0]})
    d = cache_dir / "market-data_prices"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "prices-yahoo-test.total-return.parquet"
    frame.to_parquet(p)
    return p


def test_s3_reinvested_refuses_without_a_pinned_twin_version(tmp_path):
    snap = _s3_snapshot(total_return_object_version_id=None)
    with pytest.raises(SnapshotUnavailable) as exc:
        load_prices(snap, reinvested=True, allow_network=True, cache_dir=tmp_path)
    assert "total_return_object_version_id" in str(exc.value)


def test_s3_reinvested_serves_the_cached_twin_and_checks_its_bytes(tmp_path):
    p = _write_twin(tmp_path)
    snap = _s3_snapshot(total_return_object_version_id="v-tr",
                        total_return_sha256=file_sha256(p))
    frame = load_prices(snap, reinvested=True, allow_network=False, cache_dir=tmp_path)
    assert list(frame.columns) == ["SPY", "VTI"]           # served the twin, not prices


def test_s3_reinvested_refuses_a_twin_whose_bytes_do_not_match(tmp_path):
    _write_twin(tmp_path)
    snap = _s3_snapshot(total_return_object_version_id="v-tr",
                        total_return_sha256="deadbeef")     # wrong pin
    with pytest.raises(IntegrityError):
        load_prices(snap, reinvested=True, allow_network=False, cache_dir=tmp_path)


def test_s3_reinvested_fetches_the_twin_when_absent(tmp_path, monkeypatch):
    captured = {}

    def fake_fetch_object(uri, version_id, destination):
        captured["uri"] = uri
        captured["version_id"] = version_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        _pd.DataFrame({"SPY": [9.0]}).to_parquet(destination)

    monkeypatch.setattr("src.market_data.loader._fetch_object", fake_fetch_object)
    snap = _s3_snapshot(total_return_object_version_id="v-tr")   # no sha → no byte check
    frame = load_prices(snap, reinvested=True, allow_network=True, cache_dir=tmp_path)
    assert captured["version_id"] == "v-tr"
    assert captured["uri"].endswith(".total-return.parquet")    # twin key, not price key
    assert list(frame.columns) == ["SPY"]
