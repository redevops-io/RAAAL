"""Tests for the expanded strategy suite."""
import numpy as np
import pandas as pd
import pytest

from src.ensemble_regime import train_ensemble_models
from src.features import compute_returns
from src.strategies import StrategySuite


@pytest.fixture
def synthetic_market_data():
    """Create synthetic price and return history for the full universe."""
    from src.config import AUX_SERIES, UNIVERSE

    np.random.seed(7)
    dates = pd.date_range(start="2022-01-01", periods=260, freq="B")
    prices = pd.DataFrame(index=dates)
    for idx, asset in enumerate(UNIVERSE):
        drift = 0.05 + 0.01 * idx
        noise = np.random.normal(scale=0.4 + 0.05 * idx, size=len(dates))
        prices[asset.ticker] = 100 + np.cumsum(drift + noise)
    for extra in AUX_SERIES:
        prices[extra] = 20 + np.cumsum(np.random.normal(scale=0.3, size=len(dates)))
    returns = compute_returns(prices)
    return prices, returns


@pytest.fixture
def sample_timeline():
    """Timeline with rule-based regimes and diagnostics for ML training."""
    dates = pd.date_range(start="2023-01-01", periods=90, freq="B")
    regimes = ["risk_on"] * 30 + ["risk_off"] * 30 + ["inflation"] * 30
    data = {
        "regime": regimes,
        "spy_price": np.linspace(400, 445, len(dates)) + np.sin(np.linspace(0, 6, len(dates))) * 5,
        "vix": [18.0] * 30 + [28.0] * 30 + [20.0] * 30,
        "gold_price_oz": np.linspace(1800, 1850, len(dates)),
        "diag_spy_ma200": np.linspace(395, 430, len(dates)),
        "diag_credit_signal": np.concatenate([
            np.full(30, 0.3),
            np.full(30, -0.4),
            np.full(30, -0.1),
        ]),
        "diag_spy_tlt_corr": np.concatenate([
            np.full(30, -0.1),
            np.full(30, -0.35),
            np.full(30, 0.25),
        ]),
        "diag_commodity_mom": np.concatenate([
            np.full(30, 0.05),
            np.full(30, -0.05),
            np.full(30, 0.12),
        ]),
        "diag_tip_mom": np.concatenate([
            np.full(30, 0.01),
            np.full(30, 0.02),
            np.full(30, -0.03),
        ]),
        "exp_return": np.concatenate([
            np.full(30, 0.08),
            np.full(30, 0.03),
            np.full(30, 0.05),
        ]),
        "exp_vol": np.concatenate([
            np.full(30, 0.15),
            np.full(30, 0.25),
            np.full(30, 0.18),
        ]),
        "sharpe": np.concatenate([
            np.full(30, 0.6),
            np.full(30, 0.2),
            np.full(30, 0.4),
        ]),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def ensemble_models(sample_timeline):
    return train_ensemble_models(sample_timeline, test_size=0.2, random_state=0)


def test_strategy_suite_rule_based_produces_weights(synthetic_market_data):
    prices, returns = synthetic_market_data
    suite = StrategySuite()
    results = suite.evaluate(prices, returns, detection_mode="rule_based")
    assert set(results.keys()) == set(suite.available_strategies())
    for outcome in results.values():
        total_weight = sum(outcome.weights.values())
        assert pytest.approx(1.0, abs=1e-6) == total_weight
        assert all(weight >= -1e-6 for weight in outcome.weights.values())
        assert np.isfinite(outcome.metrics["exp_vol"])
        assert np.isfinite(outcome.metrics["sharpe"])


def test_strategy_suite_ml_detection_support(synthetic_market_data, sample_timeline, ensemble_models):
    prices, returns = synthetic_market_data
    suite = StrategySuite()
    results = suite.evaluate(
        prices,
        returns,
        detection_mode="ml",
        timeline=sample_timeline,
        ensemble_models=ensemble_models,
    )
    regime_momentum = results["regime_momentum"]
    assert regime_momentum.regime_used in {"risk_on", "risk_off", "inflation"}
    assert np.isfinite(regime_momentum.metrics["sharpe"])


def test_strategy_suite_without_regime_detection_defaults_to_baseline(synthetic_market_data):
    prices, returns = synthetic_market_data
    suite = StrategySuite(default_regime="risk_on")
    results = suite.evaluate(
        prices,
        returns,
        detection_mode="none",
        strategy_names=["regime_momentum", "risk_parity", "relative_value"],
    )
    assert results["regime_momentum"].regime_used == "risk_on"
    assert results["risk_parity"].regime_used is None
    assert results["relative_value"].regime_used is None
    for name, outcome in results.items():
        assert pytest.approx(1.0, abs=1e-6) == sum(outcome.weights.values())
        assert np.isfinite(outcome.metrics["exp_return"])
