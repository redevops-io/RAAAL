"""End-to-end orchestration for the regime-adjusted allocation MVP."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from .config import AUX_SERIES, TICKER_INDEX, UNIVERSE
from .data_loader import download_prices
from .features import compute_returns, exponential_cov, exponential_mean
from .optimizer import optimize_weights
from .portfolio_utils import (
    build_rationales,
    portfolio_metrics,
    rf_from_sgov,
    weights_array,
)
from .regime import RegimeResult, detect_regime

logger = logging.getLogger(__name__)

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
    use_forecaster: bool = False,
    forecaster_backend: str = "lightgbm",
    use_sentiment: bool = False,
) -> AllocationResult:
    """Run the full allocation pipeline.

    Parameters
    ----------
    use_forecaster : bool
        If True, use ML forecaster (LightGBM/LSTM/Transformer) for μ
        instead of naive exponential mean.
    forecaster_backend : str
        Which forecaster backend: "lightgbm", "lstm", "transformer".
    use_sentiment : bool
        If True, run the NLP sentiment engine and inject scores
        into the FOMO/FOBI indicator.
    """
    tickers = [asset.ticker for asset in UNIVERSE] + AUX_SERIES
    prices = download_prices(
        tickers, start=start, end=end,
        force_refresh=force_refresh,
    )
    returns = compute_returns(prices)

    regime = detect_regime(prices, returns)
    base_tickers = [asset.ticker for asset in UNIVERSE]
    base_returns = returns[base_tickers]
    cov = exponential_cov(base_returns)
    rf_rate = rf_from_sgov(prices)

    # --- Expected returns (μ) ---
    if use_forecaster:
        try:
            from .forecaster import ReturnForecaster

            fc = ReturnForecaster(backend=forecaster_backend)
            fallback = exponential_mean(base_returns)
            result = fc.rolling_retrain(prices, returns)
            mu = result.mu
            if result.drift_report and result.drift_report.is_drifted:
                logger.warning(
                    "Forecaster drift detected (z=%.2f) — "
                    "blending with exponential mean",
                    result.drift_report.value,
                )
                mu = 0.7 * mu + 0.3 * fallback
        except Exception as exc:
            logger.warning(
                "Forecaster unavailable (%s) — "
                "falling back to exponential mean",
                exc,
            )
            mu = exponential_mean(base_returns)
    else:
        mu = exponential_mean(base_returns)

    # --- Sentiment (optional) ---
    nlp_components = None
    if use_sentiment:
        try:
            from .sentiment import SentimentEngine

            engine = SentimentEngine(scorer="auto")
            nlp_components = engine.as_fomo_components(
                tickers=base_tickers[:5],
            )
            logger.info(
                "Sentiment engine returned %d components",
                len(nlp_components),
            )
        except Exception as exc:
            logger.warning(
                "Sentiment engine unavailable (%s)", exc,
            )

    prev_state = _load_state()
    prev_weights = None
    last_regime = None
    if prev_state:
        prev_weights = weights_array(
            prev_state.get("weights", {}),
        )
        last_regime = prev_state.get("regime")

    weights = optimize_weights(
        base_returns, regime.name,
        rf_rate=rf_rate, prev_weights=prev_weights,
    )
    metrics = portfolio_metrics(weights, mu, cov, rf_rate)
    rationales = build_rationales(
        weights, mu, regime.name, regime.diagnostics,
    )

    rebalance_triggered = (
        last_regime != regime.name if last_regime else True
    )

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
