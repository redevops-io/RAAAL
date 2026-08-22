"""A consumer pod obtains prices from the data service, not from S3.

`resolve()` runs on every pod, but only quantify-data holds the S3 credentials
and the vendor URI. Under the vendor policy a pod that loaded prices locally —
web, most of all — would fail closed with "has no URI", so when a deployment
names a market-data service (`market_data_service_url`, from `QUANTIFY_DATA_URL`)
`resolve()` fetches the frame over HTTP instead.

The tests here fix the shape of that split rather than a price:

    resolve() delegates when a service URL is set, and loads locally when it is
    not — the synthetic path every existing test runs on is unchanged
    the client decodes a parquet body and keeps failure kinds across the wire
    the /prices endpoint serves a decodable frame under SYNTHETIC_ONLY

The provenance is not under test because it does not change: it is read from the
manifest by `resolve()`, not from whatever produced the bytes, so which pod
fetched the frame is not a fact any figure records.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest


def _resolved(monkeypatch, tmp_path, *, data_url: str = ""):
    """A deployment context, with or without a market-data service URL.

    Mirrors the workspace fixture the other market-data suites use: the same
    synthetic policy every existing test runs on, resolved once and installed as
    `current` so the gate reads it rather than the process environment.
    """
    from src.db import migrate
    from src.db.engine import Database
    from src.deploy import context as deploy_context

    url = f"sqlite:///{tmp_path}/w.db"
    env = {"PILOT_DATA_POLICY": "SYNTHETIC_ONLY",
           "QUANTIFY_PILOT_READER": "recorded",
           "QUANTIFY_PARSER_MODE": "RUNTIME",
           "QUANTIFY_PARSER_MODEL": "claude-sonnet-5",
           "ANTHROPIC_API_KEY": "unused",
           "QUANTIFY_DATABASE_URL": url}
    if data_url:
        env["QUANTIFY_DATA_URL"] = data_url
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    resolved = deploy_context.resolve({**os.environ, **env})
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)
    migrate.upgrade(Database(url))
    return resolved


# --- resolve() delegates or loads, on one setting ---------------------------

class TestResolveDelegatesWhenAServiceIsNamed:
    def test_a_service_url_routes_through_the_client_not_load_prices(
            self, monkeypatch, tmp_path):
        """The whole point: with a URL set, the frame comes from the client and
        `load_prices` is never called — which is what a pod with no S3 needs."""
        import src.market_data.access as access
        import src.market_data.loader as loader
        from src.market_data.loader import load_prices, synthetic_snapshot

        # The real synthetic frame, captured before `load_prices` is stubbed, so
        # the delegated frame passes the content-digest check the local path
        # also makes — the fetch stands in for the network, not for the data.
        real_frame = load_prices(synthetic_snapshot())

        _resolved(monkeypatch, tmp_path, data_url="http://quantify-data:8000")

        calls = {"client": [], "load": 0}

        class _FakeClient:
            def prices(self, *, reinvested=False):
                calls["client"].append(reinvested)
                return real_frame.copy()

        def _no_local(*_a, **_k):
            calls["load"] += 1
            raise AssertionError("load_prices must not run on a consumer pod")

        monkeypatch.setattr(access, "_market_data_client",
                            lambda base: _FakeClient())
        monkeypatch.setattr(loader, "load_prices", _no_local)

        got = access.resolve(context="a delegating pod")

        assert calls["client"] == [False], "the client was not asked for prices"
        assert calls["load"] == 0, "load_prices ran on a pod that has no S3"
        assert got.usable, "the delegated frame did not flow through"
        assert got.provenance.snapshot_id == synthetic_snapshot().snapshot_id
        assert got.access_event is not None

    def test_a_reinvested_request_crosses_without_a_digest_check(
            self, monkeypatch, tmp_path):
        """The reinvested twin is a different series with a different digest, so
        it is delegated but not digest-checked — the same asymmetry the local
        twin path has. A frame that would fail the price digest still flows."""
        import src.market_data.access as access
        import src.market_data.loader as loader

        _resolved(monkeypatch, tmp_path, data_url="http://quantify-data:8000")

        # A frame that deliberately does not match the price series' digest. If
        # the reinvested path verified it, this would raise; it must not.
        twin = pd.DataFrame({"VTI": [1.0, 2.0, 3.0]},
                            index=pd.to_datetime(
                                ["2020-01-02", "2020-01-03", "2020-01-06"]))

        class _FakeClient:
            def prices(self, *, reinvested=False):
                assert reinvested is True
                return twin.copy()

        monkeypatch.setattr(access, "_market_data_client",
                            lambda base: _FakeClient())
        monkeypatch.setattr(loader, "load_prices", lambda *a, **k: (_ for _ in
                            ()).throw(AssertionError("must not load locally")))

        got = access.resolve(context="a reinvesting pod", reinvested=True)
        assert got.usable

    def test_no_service_url_loads_locally(self, monkeypatch, tmp_path):
        """The data service itself, and every deployment before the cutover: an
        empty URL keeps the local `load_prices` path, unchanged."""
        import src.market_data.access as access
        import src.market_data.loader as loader
        from src.market_data.loader import load_prices, synthetic_snapshot

        _resolved(monkeypatch, tmp_path, data_url="")

        seen = {"load": 0}
        original = load_prices

        def _tracked(snapshot=None, **kwargs):
            seen["load"] += 1
            return original(snapshot, **kwargs)

        monkeypatch.setattr(loader, "load_prices", _tracked)
        monkeypatch.setattr(access, "_market_data_client",
                            lambda base: (_ for _ in ()).throw(
                                AssertionError("no client when no URL is set")))

        got = access.resolve(context="the data service")
        assert seen["load"] == 1, "the local loader was not used"
        assert got.usable
        assert got.provenance.snapshot_id == synthetic_snapshot().snapshot_id


# --- the client keeps failure kinds across the wire -------------------------

class TestTheClientPricesMethod:
    def _frame(self):
        from src.market_data.loader import load_prices, synthetic_snapshot

        return load_prices(synthetic_snapshot())

    def test_a_parquet_body_decodes_to_the_frame(self):
        from src.market_data.client import HttpMarketData
        from src.market_data.object_store import to_bytes

        frame = self._frame()
        body = to_bytes(frame)
        client = HttpMarketData(post=lambda *_a: (200, {}),
                                fetch=lambda *_a: (200, body, {}))
        got = client.prices(reinvested=False)
        pd.testing.assert_frame_equal(got, frame)

    def test_the_reinvested_flag_is_sent_as_a_query_param(self):
        from src.market_data.client import HttpMarketData
        from src.market_data.object_store import to_bytes

        seen = {}

        def fetch(url, params):
            seen["url"], seen["params"] = url, params
            return 200, to_bytes(self._frame()), {}

        HttpMarketData(post=lambda *_a: (200, {}), fetch=fetch,
                       base="http://d").prices(reinvested=True)
        assert seen["url"] == "http://d/prices"
        assert seen["params"] == {"reinvested": 1}

    def test_a_non_200_with_a_kind_is_a_classified_refusal(self):
        from src.market_data.client import HttpMarketData, MarketDataRefused
        from src.market_data.service_contract import Failure, failure

        client = HttpMarketData(
            post=lambda *_a: (200, {}),
            fetch=lambda *_a: (
                409, __import__("json").dumps(
                    failure(Failure.PAYLOAD_MISSING, "gone")).encode(), {}))
        with pytest.raises(MarketDataRefused) as refused:
            client.prices()
        assert refused.value.kind is Failure.PAYLOAD_MISSING

    def test_a_non_200_without_a_kind_is_unreachable(self):
        """An outage carries no failure kind, and guessing which refusal it was
        would invent a reason for the user."""
        from src.market_data.client import (HttpMarketData,
                                            MarketDataUnreachable)

        client = HttpMarketData(post=lambda *_a: (200, {}),
                                fetch=lambda *_a: (503, b"", {}))
        with pytest.raises(MarketDataUnreachable):
            client.prices()

    def test_a_corrupt_body_is_a_payload_corrupt_refusal(self):
        from src.market_data.client import HttpMarketData, MarketDataRefused
        from src.market_data.service_contract import Failure

        client = HttpMarketData(
            post=lambda *_a: (200, {}),
            fetch=lambda *_a: (200, b"this is not parquet", {}))
        with pytest.raises(MarketDataRefused) as refused:
            client.prices()
        assert refused.value.kind is Failure.PAYLOAD_CORRUPT
        assert "did not decode" in refused.value.detail


# --- the endpoint serves a decodable frame ----------------------------------

class TestThePricesEndpoint:
    @pytest.fixture
    def service(self, monkeypatch, tmp_path):
        """The real app over an in-process ASGI transport, on synthetic data.

        Mirrors `tests/test_market_data_service.py`: a `LocalParquetAdapter` and
        a temp `ObjectStore` are injected even though `/prices` uses neither, so
        the app is constructed exactly as it is deployed.
        """
        from fastapi.testclient import TestClient

        from src.market_data.adapters import LocalParquetAdapter
        from src.market_data.object_store import ObjectStore
        from src.market_data.server import create_app

        _resolved(monkeypatch, tmp_path, data_url="")
        store = ObjectStore(root=tmp_path / "objects")
        app = create_app(adapter=LocalParquetAdapter(), store=store)
        with TestClient(app) as http:
            yield http

    def test_it_returns_a_decodable_frame_under_synthetic_only(self, service):
        from src.market_data.loader import synthetic_snapshot
        from src.market_data.object_store import from_bytes

        answer = service.get("/prices", params={"reinvested": 0})
        assert answer.status_code == 200

        frame = from_bytes(answer.content)
        assert not frame.empty
        assert "VTI" in frame.columns

        snapshot = synthetic_snapshot()
        assert answer.headers["x-snapshot-id"] == snapshot.snapshot_id
        assert answer.headers["x-content-digest"] == snapshot.content_digest

    def test_the_served_frame_verifies_against_the_snapshot_digest(self, service):
        """What the consumer's delegated path checks: the served price frame is
        the one the snapshot's content digest names."""
        from src.market_data.integrity import verify
        from src.market_data.loader import synthetic_snapshot
        from src.market_data.object_store import from_bytes

        frame = from_bytes(service.get("/prices").content)
        verify(frame,
               expected_content_digest=synthetic_snapshot().content_digest,
               source="the /prices endpoint")

    def test_it_loads_with_network_allowed(self, monkeypatch, service):
        """The regression that broke the first deploy: the endpoint must load
        with `allow_network=True`. This service holds the S3 role; without the
        flag a vendor snapshot is 'not cached … and network access was not
        requested' and every consumer got a 409. Synthetic is local so the flag
        is invisible in behaviour here — assert it is passed, not inferred."""
        import src.market_data.loader as loader

        seen = {}
        real = loader.load_prices

        def watched(snapshot, **kw):
            seen.update(kw)
            return real(snapshot, **kw)

        monkeypatch.setattr(loader, "load_prices", watched)
        assert service.get("/prices").status_code == 200
        assert seen.get("allow_network") is True, seen
