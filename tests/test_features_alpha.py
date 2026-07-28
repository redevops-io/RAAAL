"""Tests for the Alpha158-inspired feature module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features_alpha import (
    adx_features,
    bollinger_features,
    build_alpha_features,
    build_universe_features,
    macd_features,
    moving_average_features,
    return_features,
    rolling_autocorrelation,
    rolling_kurtosis,
    rsi_features,
    volatility_features,
)


@pytest.fixture
def sample_prices():
    """Generate a synthetic price DataFrame."""
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-03", periods=300)
    tickers = ["SPY", "TLT", "GLD"]
    data = {}
    for t in tickers:
        cumret = np.cumsum(np.random.randn(300) * 0.01)
        data[t] = 100 * np.exp(cumret)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_returns(sample_prices):
    return np.log(sample_prices / sample_prices.shift(1)).dropna()


class TestMACDFeatures:
    def test_output_shape(self, sample_prices):
        result = macd_features(sample_prices["SPY"])
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"macd", "macd_signal", "macd_hist"}
        assert len(result) == len(sample_prices)

    def test_no_all_nan(self, sample_prices):
        result = macd_features(sample_prices["SPY"])
        assert not result["macd"].isna().all()


class TestRSIFeatures:
    def test_default_windows(self, sample_prices):
        result = rsi_features(sample_prices["SPY"])
        assert "rsi_14" in result.columns
        assert "rsi_30" in result.columns

    def test_custom_windows(self, sample_prices):
        result = rsi_features(sample_prices["SPY"], windows=[7, 21])
        assert "rsi_7" in result.columns
        assert "rsi_21" in result.columns

    def test_rsi_range(self, sample_prices):
        result = rsi_features(sample_prices["SPY"])
        valid = result["rsi_14"].dropna()
        assert valid.min() >= 0.0
        assert valid.max() <= 100.0


class TestBollingerFeatures:
    def test_output_columns(self, sample_prices):
        result = bollinger_features(sample_prices["SPY"])
        expected = {"boll_upper", "boll_lower", "boll_pct_b", "boll_bw"}
        assert set(result.columns) == expected

    def test_upper_above_lower(self, sample_prices):
        result = bollinger_features(sample_prices["SPY"])
        valid = result.dropna()
        assert (valid["boll_upper"] >= valid["boll_lower"]).all()


class TestADXFeatures:
    def test_output_columns(self, sample_prices):
        close = sample_prices["SPY"]
        high = close * 1.005
        low = close * 0.995
        result = adx_features(high, low, close)
        assert "adx" in result.columns
        assert "atr" in result.columns


class TestRollingKurtosis:
    def test_output(self, sample_returns):
        result = rolling_kurtosis(sample_returns["SPY"])
        assert "kurt_21" in result.columns
        assert "kurt_63" in result.columns


class TestRollingAutocorrelation:
    def test_output(self, sample_returns):
        result = rolling_autocorrelation(sample_returns["SPY"])
        assert "autocorr_1" in result.columns
        assert "autocorr_5" in result.columns


class TestReturnFeatures:
    def test_horizons(self, sample_prices):
        result = return_features(sample_prices["SPY"])
        assert "ret_1d" in result.columns
        assert "logret_21d" in result.columns


class TestVolatilityFeatures:
    def test_output(self, sample_returns):
        result = volatility_features(sample_returns["SPY"])
        assert "vol_21" in result.columns


class TestMovingAverageFeatures:
    def test_output(self, sample_prices):
        result = moving_average_features(sample_prices["SPY"])
        assert "sma_50" in result.columns
        assert "close_over_sma_200" in result.columns


class TestBuildAlphaFeatures:
    def test_builds_for_ticker(self, sample_prices):
        result = build_alpha_features(sample_prices, "SPY")
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > 20
        assert all(c.startswith("SPY_") for c in result.columns)

    def test_missing_ticker(self, sample_prices):
        result = build_alpha_features(sample_prices, "UNKNOWN")
        assert result.empty or len(result.columns) == 0


class TestBuildUniverseFeatures:
    def test_multi_ticker(self, sample_prices):
        result = build_universe_features(
            sample_prices, tickers=["SPY", "TLT"]
        )
        spy_cols = [c for c in result.columns if c.startswith("SPY_")]
        tlt_cols = [c for c in result.columns if c.startswith("TLT_")]
        assert len(spy_cols) > 0
        assert len(tlt_cols) > 0
