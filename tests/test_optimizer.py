import numpy as np
import pandas as pd

from src.config import REGIME_CONSTRAINTS, UNIVERSE
from src.optimizer import optimize_weights


def _toy_returns():
    dates = pd.date_range("2022-01-01", periods=200)
    data = {}
    for idx, asset in enumerate(UNIVERSE):
        base = 0.0005 + 0.0001 * idx
        noise = np.sin(np.arange(200) / (idx + 1)) * 0.0002
        data[asset.ticker] = base + noise
    return pd.DataFrame(data, index=dates)


def test_optimize_weights_bounds_respected():
    returns = _toy_returns()
    weights = optimize_weights(returns, "risk_on", rf_rate=0.0)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    guardrails = REGIME_CONSTRAINTS["risk_on"]
    inverse_sum = sum(weights[ticker] for ticker in weights if any(asset.ticker == ticker and asset.is_inverse for asset in UNIVERSE))
    assert inverse_sum <= guardrails["inverse_cap"] + 1e-6
    cash_weight = weights.get("BIL", 0.0)
    assert guardrails["cash_min"] - 1e-6 <= cash_weight <= guardrails["cash_max"] + 1e-6
