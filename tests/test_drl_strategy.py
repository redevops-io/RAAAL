"""Tests for the DRL strategy module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import ordered_tickers
from src.drl_strategy import (
    DRLAgent,
    _normalize_weight_dict,
    adaptive_rotation_strategy,
    drl_portfolio_strategy,
)

TICKERS = ordered_tickers()


@pytest.fixture
def sample_data():
    """Synthetic prices and returns."""
    np.random.seed(99)
    dates = pd.bdate_range("2022-01-03", periods=200)
    data = {}
    for t in TICKERS:
        cumret = np.cumsum(np.random.randn(200) * 0.008)
        data[t] = 100 * np.exp(cumret)
    prices = pd.DataFrame(data, index=dates)
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns


class TestNormalizeWeightDict:
    def test_sums_to_one(self):
        w = {"SPY": 0.6, "TLT": 0.3, "BIL": 0.1}
        result = _normalize_weight_dict(w)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_negative_clipped(self):
        w = {"SPY": -0.5, "TLT": 0.5, "BIL": 0.5}
        result = _normalize_weight_dict(w)
        assert result["SPY"] == 0.0
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_all_zero_defaults_to_cash(self):
        w = {"SPY": 0.0, "TLT": 0.0}
        result = _normalize_weight_dict(w)
        assert result.get("BIL", 0.0) == 1.0


class TestDRLPortfolioStrategy:
    def test_returns_valid_weights(self, sample_data):
        prices, returns = sample_data
        context = {"fomo_fobi": {"score": 0.3, "state": "neutral"}}
        weights = drl_portfolio_strategy(
            prices, returns, "risk_on", context,
        )
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in weights.values())

    def test_all_tickers_present(self, sample_data):
        prices, returns = sample_data
        weights = drl_portfolio_strategy(prices, returns, None, {})
        for t in TICKERS:
            assert t in weights


class TestAdaptiveRotationStrategy:
    def test_returns_valid_weights(self, sample_data):
        prices, returns = sample_data
        weights = adaptive_rotation_strategy(
            prices, returns, "risk_on", {},
        )
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_empty_returns_defaults_defensive(self):
        empty_prices = pd.DataFrame()
        empty_returns = pd.DataFrame()
        weights = adaptive_rotation_strategy(
            empty_prices, empty_returns, "risk_off", {},
        )
        assert isinstance(weights, dict)
        total = sum(weights.values())
        assert total > 0


class TestDRLAgent:
    def test_init(self):
        agent = DRLAgent(agent_type="ppo")
        assert agent.agent_type == "ppo"

    def test_predict_without_training(self, sample_data):
        """Without training, should return equal weights."""
        _, returns = sample_data
        agent = DRLAgent(agent_type="ppo")
        weights = agent.predict_weights(returns)
        assert isinstance(weights, dict)
        assert len(weights) == len(TICKERS)
