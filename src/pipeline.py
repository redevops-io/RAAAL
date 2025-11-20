"""End-to-end orchestration for the regime-adjusted allocation MVP."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict
import pandas as pd

from .config import AUX_SERIES, TICKER_INDEX, UNIVERSE
from .data_loader import download_prices
from .features import compute_returns, exponential_cov, exponential_mean
from .optimizer import optimize_weights
from .portfolio_utils import build_rationales, portfolio_metrics, rf_from_sgov, weights_array
from .regime import RegimeResult, detect_regime

STATE_PATH = Path("reports/state.json")


@dataclass
class AllocationResult:
    weights: Dict[str, float]
    expected_returns: pd.Series
    covariance: pd.DataFrame
    rf_rate: float
    regime: RegimeResult
    metrics: Dict[str, float]
    rationales: Dict[str, str]
    rebalance_triggered: bool


def _load_state() -> Dict[str, float] | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return None


def _save_state(state: Dict[str, float]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run_allocation(
    start: datetime,
    end: datetime,
    force_refresh: bool = False,
) -> AllocationResult:
    tickers = [asset.ticker for asset in UNIVERSE] + AUX_SERIES
    prices = download_prices(tickers, start=start, end=end, force_refresh=force_refresh)
    returns = compute_returns(prices)

    regime = detect_regime(prices, returns)
    base_returns = returns[[asset.ticker for asset in UNIVERSE]]
    mu = exponential_mean(base_returns)
    cov = exponential_cov(base_returns)
    rf_rate = rf_from_sgov(prices)

    prev_state = _load_state()
    prev_weights = None
    last_regime = None
    if prev_state:
        prev_weights = weights_array(prev_state.get("weights", {}))
        last_regime = prev_state.get("regime")

    weights = optimize_weights(base_returns, regime.name, rf_rate=rf_rate, prev_weights=prev_weights)
    metrics = portfolio_metrics(weights, mu, cov, rf_rate)
    rationales = build_rationales(weights, mu, regime.name, regime.diagnostics)

    rebalance_triggered = last_regime != regime.name if last_regime else True

    _save_state(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "regime": regime.name,
            "weights": weights,
        }
    )

    return AllocationResult(
        weights=weights,
        expected_returns=mu,
        covariance=cov,
        rf_rate=rf_rate,
        regime=regime,
        metrics=metrics,
        rationales=rationales,
        rebalance_triggered=rebalance_triggered,
    )
