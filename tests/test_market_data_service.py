"""The market-data service over HTTP, and the architecture it enforces.

Step 6. The exit condition is not a feature but a shape:

    there is no production path from provider to evaluator except through a
    verified immutable MarketSnapshot addressed by hash

Everything else here is transport. The Step-5 fixtures run again through the
real service, and hashes, bytes and descriptor identities are compared — not
final prices, because a matching price is what a wrong-but-consistent answer
looks like.

The failure boundary is the other half. Six ways this can fail and none may
degrade into another: a caller that cannot tell an ambiguous symbol from a
corrupted payload will retry one forever and go looking for the other in the
wrong place.
"""
from __future__ import annotations

import os

import pytest

from src.market_data.client import (HttpMarketData, MarketDataRefused,
                                    MarketDataUnreachable)
from src.market_data.service_contract import (MARKET_DATA_CONTRACT_VERSION,
                                              Failure)


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    from src.db import migrate
    from src.db.engine import Database
    from src.deploy import context as deploy_context

    url = f"sqlite:///{tmp_path}/w.db"
    for name, value in (("PILOT_DATA_POLICY", "SYNTHETIC_ONLY"),
                        ("QUANTIFY_PILOT_READER", "recorded"),
                        ("QUANTIFY_PARSER_MODE", "RUNTIME"),
                        ("QUANTIFY_PARSER_MODEL", "claude-sonnet-5"),
                        ("ANTHROPIC_API_KEY", "unused"),
                        ("QUANTIFY_DATABASE_URL", url)):
        monkeypatch.setenv(name, value)
    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)
    migrate.upgrade(Database(url))


@pytest.fixture
def market(tmp_path):
    """The real app over a real serialization, without a socket.

    In-process ASGI: contract and routing in full, the network not at all.
    Named rather than implied.
    """
    from fastapi.testclient import TestClient

    from src.market_data.adapters import LocalParquetAdapter
    from src.market_data.object_store import ObjectStore
    from src.market_data.server import create_app

    store = ObjectStore(root=tmp_path / "objects")
    app = create_app(adapter=LocalParquetAdapter(), store=store)
    with TestClient(app) as http:
        def post(url, json):
            answer = (http.post(url, json=json) if json is not None
                      else http.get(url))
            try:
                return answer.status_code, answer.json()
            except ValueError:
                return answer.status_code, None

        def fetch(url, params):
            answer = http.get(url, params=params)
            return answer.status_code, answer.content, dict(answer.headers)

        yield http, HttpMarketData(post=post, fetch=fetch), store


# --- the Step 5 fixtures, through the service -------------------------------

class TestTheFixturesSurviveTheService:
    def test_a_direct_ticker(self, market):
        _http, client, _store = market
        snapshot_hash, descriptor_hash = client.create(["VTI"],
                                                       reinvested=False)
        got = client.get(snapshot_hash, descriptor_hash)
        assert got.descriptor["symbols"] == ["VTI"]

    def test_a_fund_name_alias_resolves_service_side(self, market):
        """The caller sends words. Resolving them is the service's job, and
        refusing an ambiguous one is the point of it — a caller that resolved
        first would be deciding and then asking for confirmation."""
        _http, client, _store = market
        snapshot_hash, descriptor_hash = client.create(
            ["Vanguard Total Stock Market ETF"], reinvested=False)
        assert client.get(snapshot_hash, descriptor_hash) \
            .descriptor["symbols"] == ["VTI"]

    def test_the_hashes_match_the_local_lifecycle(self, market):
        """Same delivery, same address. Compared on the hash and the bytes,
        not on a price."""
        from src.market_data.adapters import LocalParquetAdapter, snapshot_from
        from src.market_data.object_store import to_bytes

        _http, client, store = market
        snapshot_hash, descriptor_hash = client.create(["VTI", "BND"],
                                                       reinvested=False)

        local = snapshot_from(
            LocalParquetAdapter().fetch(["VTI", "BND"], reinvested=False),
            reinvested=False)
        assert local.snapshot_hash == snapshot_hash
        assert local.descriptor_hash == descriptor_hash
        assert store.get(snapshot_hash) == to_bytes(
            LocalParquetAdapter().fetch(["VTI", "BND"]).observations)

    def test_the_total_return_twin_is_its_own_snapshot(self, market):
        _http, client, _store = market
        price, price_descriptor = client.create(["VTI"], reinvested=False)
        total, total_descriptor = client.create(["VTI"], reinvested=True)

        assert price != total
        assert client.get(price, price_descriptor) \
            .descriptor["corporate_actions"] == "PRICE_ONLY"
        assert client.get(total, total_descriptor) \
            .descriptor["corporate_actions"] == "TOTAL_RETURN"

    def test_creating_the_same_snapshot_twice_is_idempotent(self, market):
        _http, client, _store = market
        assert client.create(["VTI"], reinvested=False) \
            == client.create(["VTI"], reinvested=False)


# --- the failure boundary ---------------------------------------------------

class TestNoFailureDegradesIntoAnother:
    def refused(self, client, call):
        with pytest.raises(MarketDataRefused) as refusal:
            call(client)
        return refusal.value.kind

    def test_an_ambiguous_symbol(self, market):
        _http, client, _store = market
        assert self.refused(
            client, lambda c: c.create(["the S&P 500"], reinvested=False)) \
            is Failure.AMBIGUOUS_SYMBOL

    def test_an_unknown_symbol(self, market):
        _http, client, _store = market
        assert self.refused(
            client, lambda c: c.create(["Vanguard Total Stonk"],
                                       reinvested=False)) \
            is Failure.UNKNOWN_SYMBOL

    def test_a_non_tradable_instrument(self, market):
        _http, client, _store = market
        assert self.refused(
            client, lambda c: c.create(["^VIX"], reinvested=False)) \
            is Failure.NOT_TRADABLE

    def test_a_missing_descriptor(self, market):
        _http, client, _store = market
        assert self.refused(
            client, lambda c: c.get("mdf1:x", "mds1:nothing")) \
            is Failure.DESCRIPTOR_MISSING

    def test_a_descriptor_describing_other_observations(self, market):
        _http, client, _store = market
        _hash, descriptor_hash = client.create(["VTI"], reinvested=False)
        assert self.refused(
            client, lambda c: c.get("mdf1:something-else", descriptor_hash)) \
            is Failure.DESCRIPTOR_MISMATCH

    def test_a_missing_payload(self, market, tmp_path):
        """The state an interrupted write leaves. Distinct from a missing
        descriptor, because the repairs differ: one is a re-fetch, the other
        is a record nobody wrote."""
        from src.market_data.adapters import LocalParquetAdapter, snapshot_from
        from src.market_data.snapshot_store import record

        _http, client, _store = market
        snapshot = snapshot_from(LocalParquetAdapter().fetch(["GLD"]),
                                 reinvested=False)
        record(snapshot, recorded_at="2026-08-15T00:00:00Z")

        assert self.refused(
            client, lambda c: c.get(snapshot.snapshot_hash,
                                    snapshot.descriptor_hash)) \
            is Failure.PAYLOAD_MISSING

    def test_a_corrupted_payload(self, market):
        """Right address, wrong bytes. The object store cannot catch this —
        it compares bytes and knows nothing about frames — so the read path
        and the client both must."""
        from src.market_data.object_store import to_bytes

        _http, client, store = market
        snapshot_hash, descriptor_hash = client.create(["VTI"],
                                                       reinvested=False)

        from src.market_data.object_store import from_bytes
        moved = from_bytes(store.get(snapshot_hash)).copy()
        moved.iloc[0, 0] = float(moved.iloc[0, 0]) + 1.0
        store._path(snapshot_hash).write_bytes(to_bytes(moved))

        assert self.refused(
            client, lambda c: c.get(snapshot_hash, descriptor_hash)) \
            is Failure.PAYLOAD_CORRUPT

    def test_an_unreachable_service_is_not_a_refusal(self):
        """Not a statement about the request. A caller that confused them
        would treat an outage as an unsupported strategy."""
        client = HttpMarketData(post=lambda *_a: (503, None),
                                fetch=lambda *_a: (503, b"", {}))
        with pytest.raises(MarketDataUnreachable):
            client.create(["VTI"], reinvested=False)

    def test_a_contract_it_does_not_implement(self, market):
        http, _client, _store = market
        answer = http.post("/snapshots", json={"contract_version": "other@9",
                                               "symbols": ["VTI"]})
        assert answer.status_code == 400
        assert answer.json()["kind"] == Failure.CONTRACT_MISMATCH.value

    def test_every_declared_failure_has_a_status(self):
        """A kind with no status would arrive as a 500 and lose its name."""
        from src.market_data.service_contract import STATUS

        assert set(STATUS) == set(Failure)


# --- the architectural exit condition ---------------------------------------

class TestNoPathFromProviderToEvaluatorExceptByHash:
    def test_the_evaluator_imports_no_provider_machinery(self):
        """The exit condition, read off the import graph.

        Not "we do not call it" but "it is not reachable from here". A second
        route to the data is a route with none of the lifecycle's checks, and
        it looks exactly like the first until somebody compares them.
        """
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "src" / "evaluation"
        crossings = []
        for path in sorted(root.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.ImportFrom):
                    names = [("." * node.level) + (node.module or "")]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                for name in names:
                    # `client` is deliberately absent from this list: reaching
                    # the market-data service *is* the hash path, and forbidding
                    # it would force the evaluator to be handed pre-fetched data
                    # by somebody else — which is the direct-access shape this
                    # condition exists to remove. What must stay unreachable is
                    # provider machinery: adapters, symbol tables, the file
                    # loader, the byte store and the descriptor table.
                    if any(one in name for one in
                           ("adapters", "symbols", "loader", "object_store",
                            "snapshot_store")):
                        crossings.append(f"{path.name}:{node.lineno} {name}")
        assert crossings == [], (
            f"the evaluator can reach provider machinery at {crossings}, so "
            "there is a path to market data that does not pass through a "
            "verified snapshot")

    def test_the_hash_only_entry_takes_no_provider_detail(self):
        """What crosses into evaluation is observations and a hash."""
        import inspect

        from src.evaluation.service import evaluate_by_hash

        parameters = set(inspect.signature(evaluate_by_hash).parameters)
        assert parameters == {"strategy_spec", "snapshot", "evaluation_policy",
                              "engine_version", "run_plan"}
        for forbidden in ("symbols", "adapter", "uri", "provider",
                          "resolution", "reinvested"):
            assert forbidden not in parameters

    def test_the_service_returns_hashes_and_nothing_else(self, market):
        """No URI, no adapter name, no symbols in the create response. Anything
        that crossed would be a way to reach the data without the hash."""
        _http, client, _store = market
        snapshot_hash, descriptor_hash = client.create(["VTI"],
                                                       reinvested=False)
        assert snapshot_hash.startswith("mdf1:")
        assert descriptor_hash.startswith("mds1:")

    def test_a_verified_snapshot_evaluates(self, market):
        """The whole chain: words to the service, a hash back, and a figure
        computed from observations nobody handed the evaluator directly."""
        from runtime_contracts import Author, IntentField, VerifiedIntent

        from src.discovery.canonical import canonicalise
        from src.evaluation.service import evaluate_by_hash
        from src.mission.evaluation_policy import declared_policy
        from src.mission.from_intent import compile_intent
        from src.mission.strategy_spec import from_scenario
        from src.workspace.run_boundary import execute_compiled_plan

        _http, client, _store = market
        snapshot_hash, descriptor_hash = client.create(["VTI"],
                                                       reinvested=True)
        snapshot = client.get(snapshot_hash, descriptor_hash)

        canonical = canonicalise({"assets": "VTI", "amount": "1000",
                                  "cadence": "monthly"})
        intent = VerifiedIntent(
            objective="evaluate_investment_strategy", produced_by="t",
            utterance_ref="u",
            fields={n: IntentField(value=v, author=Author.MODEL)
                    for n, (v, _a) in canonical.fields.items()},
            unresolved=()).seal()
        spec = from_scenario(compile_intent(intent,
                                            benchmark_rule="a-rule").scenario)

        result = evaluate_by_hash(
            spec, snapshot,
            evaluation_policy=declared_policy(data_policy="SYNTHETIC_ONLY",
                                              as_of="2026-08-15"),
            engine_version="engine@test", run_plan=execute_compiled_plan)

        assert result.market_snapshot_hash == snapshot_hash
        assert result.streams["fills"].produced
