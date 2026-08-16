"""Adapters, symbol resolution, and the five cases that matter.

Step 5. Two hard rules, both checked rather than described:

  1. an adapter never writes to evaluation — everything becomes a
     `MarketSnapshot` and travels the ordinary lifecycle;
  2. symbol resolution never guesses — an ambiguous name produces a refusal
     naming the candidates, not the closest ticker.

The second is the one that costs money quietly. A plan priced against the wrong
instrument is wrong in a way that looks entirely correct: right shape, right
dates, wrong company. Nothing downstream can catch it, because every downstream
check is about consistency and a wrong-but-consistent answer passes them all.
"""
from __future__ import annotations

import os

import pytest

from src.market_data.adapters import (AdapterRefused, LocalParquetAdapter,
                                      snapshot_from)
from src.market_data.symbols import (INSTRUMENTS, InstrumentKind, Outcome,
                                     resolve, resolve_all)


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


# --- the five conformance fixtures ------------------------------------------

class TestDirectTicker:
    def test_it_resolves_to_itself(self):
        found = resolve("VTI")
        assert found.resolved
        assert found.instrument.symbol == "VTI"
        assert found.instrument.kind is InstrumentKind.ETF

    def test_case_does_not_change_the_instrument(self):
        assert resolve("vti").instrument.symbol == "VTI"


class TestFundOrCompanyNameAlias:
    def test_a_fund_name_resolves(self):
        found = resolve("Vanguard Total Stock Market ETF")
        assert found.resolved and found.instrument.symbol == "VTI"

    def test_a_shortened_alias_resolves(self):
        assert resolve("total bond market").instrument.symbol == "BND"

    def test_a_company_name_resolves_to_the_equity(self):
        found = resolve("Berkshire Hathaway")
        assert found.resolved
        assert found.instrument.symbol == "BRK-B"
        assert found.instrument.kind is InstrumentKind.EQUITY

    def test_a_leading_article_does_not_prevent_it(self):
        assert resolve("the total stock market").instrument.symbol == "VTI"


class TestIndexVersusTradableFund:
    def test_an_index_resolves_and_is_not_tradable(self):
        """Resolving is not the same as being buyable, and the identity says
        which. An engine handed an index prices something nobody can hold."""
        found = resolve("^VIX")
        assert found.resolved
        assert found.instrument.kind is InstrumentKind.INDEX
        assert found.instrument.tradable is False

    def test_the_etf_beside_it_is_tradable(self):
        assert resolve("SPY").instrument.tradable is True

    def test_an_adapter_refuses_to_fetch_an_index(self):
        with pytest.raises(AdapterRefused, match="index"):
            LocalParquetAdapter().fetch(["^VIX"])

    def test_the_refusal_names_why_rather_than_substituting_the_etf(self):
        """The tempting repair — quietly swapping SPY for ^GSPC — would answer
        a question about an index with a figure about a fund."""
        with pytest.raises(AdapterRefused) as refused:
            LocalParquetAdapter().fetch(["^GSPC"])
        assert "nobody can hold" in str(refused.value)
        assert "SPY" not in str(refused.value)


class TestTotalReturnVersusPriceTwin:
    def test_the_two_are_different_snapshots(self):
        adapter = LocalParquetAdapter()
        price = snapshot_from(adapter.fetch(["VTI"], reinvested=False),
                              reinvested=False)
        total = snapshot_from(adapter.fetch(["VTI"], reinvested=True),
                              reinvested=True)

        assert price.snapshot_hash != total.snapshot_hash
        assert price.corporate_actions == "PRICE_ONLY"
        assert total.corporate_actions == "TOTAL_RETURN"

    def test_both_carry_the_same_instrument(self):
        adapter = LocalParquetAdapter()
        for reinvested in (False, True):
            fetched = adapter.fetch(["VTI"], reinvested=reinvested)
            assert fetched.symbols() == ("VTI",)


class TestAmbiguousSymbolOrName:
    def test_a_family_name_refuses_and_lists_what_it_could_mean(self):
        found = resolve("the S&P 500")
        assert found.outcome is Outcome.AMBIGUOUS
        symbols = {one.symbol for one in found.candidates}
        assert {"^GSPC", "VOO", "SPY", "RSP"} <= symbols
        assert "cannot be bought" in found.detail

    def test_it_does_not_return_a_closest_match(self):
        found = resolve("the S&P 500")
        assert found.instrument is None, (
            "an ambiguous name produced an instrument, so a lookup chose a "
            "holding on the user's behalf")

    def test_an_unknown_name_is_unresolved_rather_than_approximated(self):
        found = resolve("Vanguard Total Stonk Market")
        assert found.outcome is Outcome.UNRESOLVED
        assert found.instrument is None
        assert "closest match" in found.detail

    def test_an_adapter_refuses_an_ambiguous_name(self):
        with pytest.raises(AdapterRefused) as refused:
            LocalParquetAdapter().fetch(["the S&P 500"])
        assert "not resolved" in str(refused.value)

    def test_resolving_several_reports_every_failure(self):
        """A caller given only the successes would build a portfolio quietly
        missing whatever could not be named."""
        found, failed = resolve_all(["VTI", "the S&P 500", "nonsense"])
        assert [one.symbol for one in found] == ["VTI"]
        assert len(failed) == 2


# --- the two hard rules -----------------------------------------------------

class TestAdaptersNeverReachEvaluation:
    def test_the_module_imports_no_evaluator_and_no_web_layer(self):
        """Structural rather than promised. A second path to a figure is a
        path with none of the checks on it, and it looks exactly like the
        first until somebody compares them."""
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "src"
                  / "market_data" / "adapters.py").read_text()
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                imported.add(("." * node.level) + (node.module or ""))
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        forbidden = [one for one in imported
                     if "evaluation" in one or "workspace" in one
                     or "mission" in one]
        assert forbidden == [], (
            f"{forbidden} — an adapter with a route to the evaluator is a "
            "second way to a figure, with none of the lifecycle's checks")

    def test_the_only_exit_is_a_market_snapshot(self):
        from src.market_data.snapshot_contract import MarketSnapshot

        fetched = LocalParquetAdapter().fetch(["VTI", "BND"])
        built = snapshot_from(fetched, reinvested=False)
        assert isinstance(built, MarketSnapshot)


class TestAdapterOutputTravelsTheOrdinaryLifecycle:
    def test_it_records_stores_and_reads_back_verified(self, tmp_path):
        """The point of Step 5: nothing about an adapter's output is special.

        It becomes a descriptor, becomes immutable bytes, and is read back and
        verified by exactly the path anything else uses.
        """
        from src.market_data.object_store import ObjectStore, to_bytes
        from src.market_data.snapshot_read import get
        from src.market_data.snapshot_store import record

        store = ObjectStore(root=tmp_path / "objects")
        fetched = LocalParquetAdapter().fetch(["VTI", "BND"], reinvested=True)
        snapshot = snapshot_from(fetched, reinvested=True)

        record(snapshot, recorded_at="2026-08-15T00:00:00Z")
        store.put(snapshot.snapshot_hash, to_bytes(fetched.observations))

        read = get(snapshot.snapshot_hash, snapshot.descriptor_hash, store=store)
        assert read.ok, read.refusal()
        assert read.snapshot.symbols == ("BND", "VTI")

    def test_the_adapter_and_its_version_reach_the_descriptor(self):
        snapshot = snapshot_from(LocalParquetAdapter().fetch(["VTI"]),
                                 reinvested=False)
        assert snapshot.source_adapter.name == "local-parquet"
        assert snapshot.source_adapter.version == "1"

    def test_licensing_metadata_reaches_the_descriptor(self):
        snapshot = snapshot_from(LocalParquetAdapter().fetch(["VTI"]),
                                 reinvested=False)
        assert snapshot.license_class
        assert snapshot.license_review_status
        assert snapshot.license_class != "NOT_DECLARED"


class TestTheRegistryIsCoherent:
    def test_every_instrument_has_a_distinct_symbol(self):
        symbols = [one.symbol for one in INSTRUMENTS]
        assert len(symbols) == len(set(symbols))

    def test_every_index_is_marked_untradable(self):
        for one in INSTRUMENTS:
            assert one.tradable is (one.kind is not InstrumentKind.INDEX)

    def test_no_alias_silently_resolves_two_ways(self):
        """An alias mapping to several instruments must produce AMBIGUOUS, and
        this checks none of them resolves anyway."""
        for one in INSTRUMENTS:
            for alias in one.aliases:
                found = resolve(alias)
                assert found.outcome in (Outcome.RESOLVED, Outcome.AMBIGUOUS)
                if found.outcome is Outcome.RESOLVED:
                    assert found.instrument.symbol == one.symbol
