"""Tests for the FOMO/FOBI indicator construction."""
import numpy as np
import pandas as pd

from src.fomo_fobi import compute_fomo_fobi_indicator, required_tickers


def _synthetic_prices() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    base = 100 + np.cumsum(np.random.normal(scale=0.5, size=len(dates)))
    data = {}
    for ticker in required_tickers():
        drift = np.random.uniform(-0.1, 0.1)
        noise = np.random.normal(scale=0.4, size=len(dates))
        series = base * (1 + drift) + np.cumsum(noise)
        data[ticker] = series
    # Ensure denominators are positive
    for ticker in data:
        data[ticker] = np.maximum(5.0, data[ticker])
    return pd.DataFrame(data, index=dates)


def test_indicator_returns_score_and_state():
    prices = _synthetic_prices()
    indicator = compute_fomo_fobi_indicator(prices)
    assert "fomo_fobi_score" in indicator.columns
    assert "fomo_fobi_state" in indicator.columns
    assert indicator.shape[0] == prices.shape[0]
    # We should eventually see at least one finite score once lookback windows fill
    assert np.isfinite(indicator["fomo_fobi_score"].dropna()).any()


def test_indicator_thresholds_change_state():
    prices = _synthetic_prices()
    indicator = compute_fomo_fobi_indicator(prices)
    score = indicator["fomo_fobi_score"].dropna()
    if score.empty:
        return
    max_state = indicator.loc[score.idxmax(), "fomo_fobi_state"]
    min_state = indicator.loc[score.idxmin(), "fomo_fobi_state"]
    assert max_state in {"FOMO", "neutral"}
    assert min_state in {"FOBI", "neutral"}