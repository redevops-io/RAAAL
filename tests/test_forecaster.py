"""Tests for the return forecaster module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecaster import (
    DriftReport,
    ForecastResult,
    ReturnForecaster,
    detect_drift,
)


@pytest.fixture
def synthetic_data():
    """Generate synthetic prices and returns for testing."""
    np.random.seed(123)
    dates = pd.bdate_range("2021-01-04", periods=300)
    from src.config import ordered_tickers

    tickers = ordered_tickers()
    price_data = {}
    for t in tickers:
        cumret = np.cumsum(np.random.randn(300) * 0.01)
        price_data[t] = 100 * np.exp(cumret)
    prices = pd.DataFrame(price_data, index=dates)
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns


class TestDriftDetection:
    def test_no_drift_when_similar(self):
        hist = pd.Series(np.random.randn(100) * 0.01)
        recent = pd.Series(np.random.randn(10) * 0.01)
        report = detect_drift(recent, hist, threshold=3.0)
        assert isinstance(report, DriftReport)
        assert report.metric == "zscore"

    def test_drift_detected_on_shift(self):
        hist = pd.Series(np.random.randn(100) * 0.01)
        recent = pd.Series(np.ones(10) * 10)
        report = detect_drift(recent, hist, threshold=2.0)
        assert report.is_drifted

    def test_empty_inputs(self):
        report = detect_drift(pd.Series(dtype=float), pd.Series(dtype=float))
        assert report.is_drifted is False


class TestReturnForecaster:
    def test_lightgbm_init(self):
        fc = ReturnForecaster(backend="lightgbm")
        assert fc.backend_name == "lightgbm"

    def test_predict_without_training(self, synthetic_data):
        """Predict should return fallback mu when model is untrained."""
        prices, returns = synthetic_data
        fc = ReturnForecaster(backend="lightgbm")
        result = fc.predict(prices, returns)
        assert isinstance(result, ForecastResult)
        assert isinstance(result.mu, pd.Series)
        # Should have one value per ticker
        from src.config import ordered_tickers

        assert len(result.mu) == len(ordered_tickers())

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            ReturnForecaster(backend="unknown_model")

    def test_forecast_result_fields(self, synthetic_data):
        prices, returns = synthetic_data
        fc = ReturnForecaster(backend="lightgbm")
        result = fc.predict(prices, returns)
        assert result.backend == "lightgbm"
        assert result.drift_report is not None
        assert "needs_retrain" in result.metadata
