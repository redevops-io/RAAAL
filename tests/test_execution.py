"""Tests for the Alpaca execution module."""
from __future__ import annotations

import pytest

from src.execution import AlpacaExecutor, OrderResult, RebalanceResult


class TestAlpacaExecutor:
    def test_init_defaults(self):
        executor = AlpacaExecutor(mode="paper", dry_run=True)
        assert executor.mode == "paper"
        assert executor.dry_run is True
        assert "paper" in executor._base_url

    def test_init_live(self):
        executor = AlpacaExecutor(mode="live", dry_run=True)
        assert executor.mode == "live"
        assert "paper" not in executor._base_url

    def test_ticker_map(self):
        executor = AlpacaExecutor(dry_run=True)
        assert executor._map_ticker("BTC-USD") == "BTC/USD"
        assert executor._map_ticker("SPY") == "SPY"

    def test_dry_run_rebalance(self):
        executor = AlpacaExecutor(mode="paper", dry_run=True)
        result = executor.rebalance({"SPY": 0.6, "TLT": 0.3, "BIL": 0.1})
        assert isinstance(result, RebalanceResult)
        assert result.mode == "paper"
        assert len(result.orders) == 0
        assert len(result.errors) == 0

    def test_connect_without_credentials(self):
        executor = AlpacaExecutor(
            mode="paper",
            api_key="",
            secret_key="",
        )
        with pytest.raises(RuntimeError, match="credentials"):
            executor._connect()


class TestOrderResult:
    def test_creation(self):
        order = OrderResult(
            ticker="SPY",
            side="buy",
            qty=10.0,
            notional=5000.0,
            status="filled",
            order_id="abc123",
        )
        assert order.ticker == "SPY"
        assert order.status == "filled"


class TestRebalanceResult:
    def test_creation(self):
        from datetime import datetime, timezone

        result = RebalanceResult(
            timestamp=datetime.now(timezone.utc),
            target_weights={"SPY": 1.0},
            orders=[],
            portfolio_value=100000.0,
            mode="paper",
        )
        assert result.portfolio_value == 100000.0
        assert result.mode == "paper"
