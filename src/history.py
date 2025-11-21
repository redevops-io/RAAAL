"""Historical regime + allocation analysis for visualization."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import AUX_SERIES, UNIVERSE
from .data_loader import download_prices
from .features import compute_returns, exponential_cov, exponential_mean
from .hrp import compute_hrp_weights
from .optimizer import optimize_weights, optimize_weights_unrestricted
from .portfolio_utils import build_rationales, portfolio_metrics, rf_from_sgov, weights_array
from .regime import detect_regime

HISTORY_DIR = Path("data/history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
TIMELINE_PATH = HISTORY_DIR / "timeline.parquet"
WEIGHTS_PATH = HISTORY_DIR / "weights.parquet"
PRICES_PATH = HISTORY_DIR / "prices.parquet"
SUMMARY_JSON = HISTORY_DIR / "history_summary.json"

# SPDR GLD share corresponds to roughly 1/10 ounce of gold after fees.
GLD_SHARE_TO_OUNCE = 10.0


@dataclass
class HistoryRunResult:
    timeline: pd.DataFrame
    weights: pd.DataFrame
    prices: pd.DataFrame
    performance: Dict[str, float]


def _evaluation_dates(returns: pd.DataFrame, warmup: int, step: int) -> List[pd.Timestamp]:
    if returns.shape[0] <= warmup:
        return []
    eligible = returns.index[warmup:]
    return list(eligible[::step])


def run_historical_analysis(
    start: datetime,
    end: datetime,
    warmup_days: int = 252,
    step: int = 5,
    force_refresh: bool = False,
) -> HistoryRunResult:
    tickers = [asset.ticker for asset in UNIVERSE] + AUX_SERIES
    prices = download_prices(tickers, start=start, end=end, force_refresh=force_refresh)
    returns = compute_returns(prices)

    eval_dates = _evaluation_dates(returns, warmup_days, step)
    if not eval_dates:
        raise ValueError("Not enough data to run historical analysis. Try reducing warmup window.")

    timeline_rows: List[Dict[str, float]] = []
    weight_rows: List[Dict[str, float]] = []

    prev_weights_vec = None
    prev_standard_vec = None
    baseline_regime = "risk_on"

    for date in eval_dates:
        prices_window = prices.loc[:date]
        returns_window = returns.loc[:date]
        base_returns = returns_window[[asset.ticker for asset in UNIVERSE]]

        regime = detect_regime(prices_window, returns_window)
        # Regime-aware statistics: exponentially weighted with regime-specific decay
        mu = exponential_mean(base_returns)
        cov = exponential_cov(base_returns)
        rf_rate = rf_from_sgov(prices_window)

        # Standard (uniform-weighted) statistics for non-regime strategies
        mu_standard = base_returns.mean()  # Simple arithmetic mean
        cov_standard = base_returns.cov()  # Sample covariance

        weights = optimize_weights(
            base_returns,
            regime.name,
            rf_rate=rf_rate,
            prev_weights=prev_weights_vec,
        )
        standard_weights = optimize_weights(
            base_returns,
            baseline_regime,
            rf_rate=rf_rate,
            prev_weights=prev_standard_vec,
        )
        metrics = portfolio_metrics(weights, mu, cov, rf_rate)
        rationales = build_rationales(weights, mu, regime.name, regime.diagnostics)
        regime_unrestricted_weights = optimize_weights_unrestricted(mu, cov, rf_rate)
        unrestricted_metrics = portfolio_metrics(regime_unrestricted_weights, mu, cov, rf_rate)
        standard_unrestricted_weights = optimize_weights_unrestricted(mu_standard, cov_standard, rf_rate)  # Uses uniform weighting
        
        # HRP weights (2 variants)
        # 1. Base HRP using uniform-weighted returns with standard covariance (baseline regime)
        hrp_weights = compute_hrp_weights(base_returns, cov_standard)
        hrp_metrics = portfolio_metrics(hrp_weights, mu, cov, rf_rate)
        
        # 2. HRP with regime-aware approach: uses regime-specific exponentially-weighted covariance
        hrp_regime_weights = compute_hrp_weights(base_returns, cov)
        
        # 3 & 4: Apply regime restrictions to both HRP variants
        from .optimizer import _bounds
        hrp_restricted = {}
        hrp_regime_restricted = {}
        bounds_baseline = _bounds(baseline_regime)
        bounds_regime = _bounds(regime.name)
        
        for i, asset in enumerate(UNIVERSE):
            ticker = asset.ticker
            # HRP restricted: apply baseline bounds to base HRP
            low, high = bounds_baseline[i]
            hrp_restricted[ticker] = max(low, min(high, hrp_weights.get(ticker, 0.0)))
            # HRP regime restricted: apply regime-specific bounds
            low_r, high_r = bounds_regime[i]
            hrp_regime_restricted[ticker] = max(low_r, min(high_r, hrp_regime_weights.get(ticker, 0.0)))
        
        # Renormalize after clipping
        hrp_restricted_sum = sum(hrp_restricted.values())
        if hrp_restricted_sum > 0:
            hrp_restricted = {k: v / hrp_restricted_sum for k, v in hrp_restricted.items()}
        hrp_regime_restricted_sum = sum(hrp_regime_restricted.values())
        if hrp_regime_restricted_sum > 0:
            hrp_regime_restricted = {k: v / hrp_regime_restricted_sum for k, v in hrp_regime_restricted.items()}

        vix_value = float(prices_window["^VIX"].iloc[-1]) if "^VIX" in prices_window.columns else float("nan")
        gld_value = float(prices_window["GLD"].iloc[-1]) if "GLD" in prices_window.columns else float("nan")
        gold_oz_price = gld_value * GLD_SHARE_TO_OUNCE if pd.notna(gld_value) else float("nan")
        timeline_rows.append(
            {
                "date": date,
                "regime": regime.name,
                "spy_price": float(prices_window["SPY"].iloc[-1]),
                "vix": vix_value,
                "gold_price_oz": gold_oz_price,
                **{f"diag_{k}": v for k, v in regime.diagnostics.items()},
                **metrics,
                "sharpe_unrestricted": unrestricted_metrics.get("sharpe", float("nan")),
                "sharpe_hrp": hrp_metrics.get("sharpe", float("nan")),
                "rf_daily": rf_rate,
            }
        )
        for ticker, weight in weights.items():
            weight_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "weight": weight,
                    "unrestricted_weight": regime_unrestricted_weights.get(ticker, 0.0),
                    "standard_weight": standard_weights.get(ticker, 0.0),
                    "standard_unrestricted_weight": standard_unrestricted_weights.get(ticker, 0.0),
                    "hrp_weight": hrp_weights.get(ticker, 0.0),
                    "hrp_restricted_weight": hrp_restricted.get(ticker, 0.0),
                    "hrp_regime_weight": hrp_regime_weights.get(ticker, 0.0),
                    "hrp_regime_restricted_weight": hrp_regime_restricted.get(ticker, 0.0),
                }
            )

        prev_weights_vec = weights_array(weights)
        prev_standard_vec = weights_array(standard_weights)

    timeline_df = pd.DataFrame(timeline_rows).set_index("date").sort_index()
    weights_df = pd.DataFrame(weight_rows)
    asset_returns = returns[[asset.ticker for asset in UNIVERSE]]
    strategy_map = {
        "standard_restricted": "standard_weight",
        "standard_unrestricted": "standard_unrestricted_weight",
        "regime_restricted": "weight",
        "regime_unrestricted": "unrestricted_weight",
        "hrp_unrestricted": "hrp_weight",
        "hrp_restricted": "hrp_restricted_weight",
        "hrp_regime_unrestricted": "hrp_regime_weight",
        "hrp_regime_restricted": "hrp_regime_restricted_weight",
    }
    performance = {
        label: _strategy_total_return(weights_df, asset_returns, column)
        for label, column in strategy_map.items()
    }
    for label, value in performance.items():
        timeline_df[f"total_return_{label}"] = value

    return HistoryRunResult(timeline=timeline_df, weights=weights_df, prices=prices, performance=performance)


def _strategy_total_return(weights: pd.DataFrame, asset_returns: pd.DataFrame, column: str) -> float:
    if column not in weights.columns:
        return float("nan")
    weight_history = (
        weights.pivot(index="date", columns="ticker", values=column)
        .sort_index()
        .reindex(columns=asset_returns.columns, fill_value=0.0)
    )
    if weight_history.empty:
        return float("nan")
    weight_history = weight_history.reindex(asset_returns.index).ffill()
    valid_mask = weight_history.notna().any(axis=1)
    weight_history = weight_history.loc[valid_mask]
    if weight_history.empty:
        return float("nan")
    aligned_returns = asset_returns.loc[weight_history.index].fillna(0.0)
    daily_returns = (weight_history.fillna(0.0) * aligned_returns).sum(axis=1)
    total_return = float((1.0 + daily_returns).prod() - 1.0)
    # Annualize: (1 + total_return)^(252 / n_days) - 1
    n_days = len(daily_returns)
    if n_days == 0:
        return float("nan")
    annualized_return = float((1.0 + total_return) ** (252.0 / n_days) - 1.0)
    return annualized_return


def save_history(result: HistoryRunResult) -> Dict[str, str]:
    result.timeline.to_parquet(TIMELINE_PATH)
    result.weights.to_parquet(WEIGHTS_PATH)
    result.prices.to_parquet(PRICES_PATH)

    payload = {
        "timeline": result.timeline.reset_index().to_dict(orient="list"),
        "weights": result.weights.to_dict(orient="list"),
        "performance": result.performance,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, default=str))

    return {
        "timeline": str(TIMELINE_PATH),
        "weights": str(WEIGHTS_PATH),
        "prices": str(PRICES_PATH),
        "summary": str(SUMMARY_JSON),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical regime-adjusted allocation analysis")
    parser.add_argument("--start", required=True, type=datetime.fromisoformat, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=datetime.fromisoformat, help="End date YYYY-MM-DD")
    parser.add_argument("--warmup", type=int, default=252, help="Warmup window in trading days")
    parser.add_argument("--step", type=int, default=5, help="Evaluate every N trading days")
    parser.add_argument("--refresh", action="store_true", help="Force data refresh from Yahoo")
    args = parser.parse_args()

    result = run_historical_analysis(
        start=args.start,
        end=args.end,
        warmup_days=args.warmup,
        step=args.step,
        force_refresh=args.refresh,
    )
    paths = save_history(result)
    print("Historical analysis saved:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
