"""Tests for HRP, network analysis, and ensemble learning modules."""
import pandas as pd
import pytest

from src.config import UNIVERSE
from src.ensemble_regime import prepare_features, train_ensemble_models
from src.hrp import compute_hrp_weights
from src.network import build_correlation_network, compute_centrality_measures, compute_network_metrics


@pytest.fixture
def sample_returns():
    """Create sample returns data for testing."""
    import numpy as np
    
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    tickers = [asset.ticker for asset in UNIVERSE]
    
    # Generate correlated returns
    np.random.seed(42)
    n_assets = len(tickers)
    cov = np.random.rand(n_assets, n_assets)
    cov = (cov + cov.T) / 2  # Make symmetric
    cov = cov + n_assets * np.eye(n_assets)  # Make positive definite
    
    returns = np.random.multivariate_normal(np.zeros(n_assets), cov / 10000, size=len(dates))
    df = pd.DataFrame(returns, index=dates, columns=tickers)
    return df


@pytest.fixture
def sample_timeline():
    """Create sample timeline data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
    data = {
        "regime": ["risk_on"] * 20 + ["risk_off"] * 15 + ["inflation"] * 15,
        "spy_price": range(400, 450),
        "vix": [15.0] * 20 + [25.0] * 15 + [18.0] * 15,
        "gold_price_oz": range(1800, 1850),
        "diag_spy_ma200": range(390, 440),
        "diag_credit_signal": [0.5] * 50,
        "diag_spy_tlt_corr": [-0.3] * 50,
        "diag_commodity_mom": [0.1] * 50,
        "diag_tip_mom": [-0.2] * 50,
        "exp_return": [0.08] * 50,
        "exp_vol": [0.15] * 50,
        "sharpe": [0.5] * 50,
    }
    df = pd.DataFrame(data, index=dates)
    return df


def test_hrp_weights_sum_to_one(sample_returns):
    """Test that HRP weights sum to 1."""
    weights = compute_hrp_weights(sample_returns)
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "HRP weights should sum to 1"


def test_hrp_weights_all_positive(sample_returns):
    """Test that HRP weights are all non-negative."""
    weights = compute_hrp_weights(sample_returns)
    assert all(w >= 0 for w in weights.values()), "HRP weights should be non-negative"


def test_hrp_includes_all_assets(sample_returns):
    """Test that HRP includes all assets."""
    weights = compute_hrp_weights(sample_returns)
    expected_tickers = {asset.ticker for asset in UNIVERSE}
    assert set(weights.keys()) == expected_tickers, "HRP should include all assets"


def test_network_construction(sample_returns):
    """Test correlation network construction."""
    G = build_correlation_network(sample_returns, threshold=0.3)
    assert G.number_of_nodes() == len(UNIVERSE), "Network should have all assets as nodes"
    assert G.number_of_edges() >= 0, "Network should have edges"


def test_centrality_measures(sample_returns):
    """Test centrality measure computation."""
    G = build_correlation_network(sample_returns, threshold=0.3)
    centrality_df = compute_centrality_measures(G)
    
    assert not centrality_df.empty, "Centrality measures should be computed"
    assert "degree" in centrality_df.columns, "Should include degree centrality"
    assert "betweenness" in centrality_df.columns, "Should include betweenness centrality"
    assert "eigenvector" in centrality_df.columns, "Should include eigenvector centrality"


def test_network_metrics(sample_returns):
    """Test comprehensive network metrics."""
    metrics = compute_network_metrics(sample_returns, threshold=0.3)
    
    assert "graph" in metrics, "Should include graph object"
    assert "num_nodes" in metrics, "Should include node count"
    assert "num_edges" in metrics, "Should include edge count"
    assert "density" in metrics, "Should include network density"
    assert "centrality" in metrics, "Should include centrality measures"
    assert "communities" in metrics, "Should include community detection"


def test_ensemble_feature_preparation(sample_timeline):
    """Test feature preparation for ensemble models."""
    X, y = prepare_features(sample_timeline)
    
    assert not X.empty, "Features should be extracted"
    assert len(X) == len(y), "Features and labels should have same length"
    assert "spy_price" in X.columns, "Should include SPY price feature"
    assert "vix" in X.columns, "Should include VIX feature"


def test_ensemble_model_training(sample_timeline):
    """Test ensemble model training."""
    models_result = train_ensemble_models(sample_timeline, test_size=0.3, random_state=42)
    
    assert "random_forest" in models_result, "Should include Random Forest model"
    assert "gradient_boosting" in models_result, "Should include Gradient Boosting model"
    assert "label_encoder" in models_result, "Should include label encoder"
    assert "rf_accuracy" in models_result, "Should include RF accuracy"
    assert "gb_accuracy" in models_result, "Should include GB accuracy"
    assert "feature_importance" in models_result, "Should include feature importance"
    
    # Check accuracy is reasonable (above random guessing)
    assert models_result["rf_accuracy"] > 0.3, "RF accuracy should be above random"
    assert models_result["gb_accuracy"] > 0.3, "GB accuracy should be above random"


def test_hrp_with_minimal_data():
    """Test HRP with minimal data (edge case)."""
    dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
    tickers = [asset.ticker for asset in UNIVERSE]
    data = {ticker: [0.01, -0.01] for ticker in tickers}
    returns = pd.DataFrame(data, index=dates)
    
    weights = compute_hrp_weights(returns)
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "HRP should handle minimal data"
