"""Historical regime + allocation analysis for visualization."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import (
    AUX_SERIES,
    EXECUTION_LAG_DAYS,
    FOMO_COMPONENT_WEIGHTS,
    TRANSACTION_COST_BPS,
    UNIVERSE,
)
from .data_loader import download_prices
from .features import compute_returns, exponential_cov, exponential_mean
from .ensemble_regime import load_ensemble_models
from .fomo_fobi import compute_fomo_fobi_indicator
from .hrp import compute_hrp_weights
from .optimizer import optimize_weights, optimize_weights_unrestricted
from .portfolio_utils import build_rationales, portfolio_metrics, rf_from_sgov, weights_array
from .regime import detect_regime
from .reproducibility import (
    DEFAULT_SEED,
    build_run_manifest,
    frame_digest,
    seed_everything,
)
from .strategies import StrategySuite
from .nowcasting import compute_nowcasts

HISTORY_DIR = Path("data/history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
TIMELINE_PATH = HISTORY_DIR / "timeline.parquet"
WEIGHTS_PATH = HISTORY_DIR / "weights.parquet"
PRICES_PATH = HISTORY_DIR / "prices.parquet"
SUMMARY_JSON = HISTORY_DIR / "history_summary.json"
FOMO_PATH = HISTORY_DIR / "fomo_indicator.parquet"
MANIFEST_PATH = HISTORY_DIR / "run_manifest.json"

# SPDR GLD share corresponds to roughly 1/10 ounce of gold after fees.
GLD_SHARE_TO_OUNCE = 10.0


@dataclass
class HistoryRunResult:
    timeline: pd.DataFrame
    weights: pd.DataFrame
    prices: pd.DataFrame
    performance: Dict[str, float]
    strategy_columns: Dict[str, str]
    fomo_indicator: pd.DataFrame | None = None


def _evaluation_dates(returns: pd.DataFrame, warmup: int, step: int) -> List[pd.Timestamp]:
    if returns.shape[0] <= warmup:
        return []
    eligible = returns.index[warmup:]
    dates = list(eligible[::step])
    # Always include the most recent date for up-to-date dashboard
    if dates and eligible[-1] not in dates:
        dates.append(eligible[-1])
    return dates


def run_historical_analysis(
    start: datetime,
    end: datetime,
    warmup_days: int = 252,
    step: int = 5,
    force_refresh: bool = False,
    use_forecaster: bool = False,
    forecaster_backend: str = "lightgbm",
    use_sentiment: bool = False,
    sentiment_backend: str = "auto",
    seed: int = DEFAULT_SEED,
    protocol=None,
) -> HistoryRunResult:
    """Run the legacy multi-strategy engine.

    `protocol` is an `EvaluationProtocol`. When supplied, the engine **replays**
    under it rather than reconstructing its own settings: the trading calendar,
    execution lag, cost model and annualization all come from the artifact, and
    the same code paths the evaluation runner uses are called with the same
    values. Passing ``None`` keeps the historical defaults from `config`, which
    remain only so that pre-protocol invocations still run — they are not a
    second source of truth.
    """
    logger = logging.getLogger(__name__)
    seed_everything(seed)
    tickers = [asset.ticker for asset in UNIVERSE] + AUX_SERIES
    prices = download_prices(tickers, start=start, end=end, force_refresh=force_refresh)

    if protocol is not None:
        from .evaluation.runner import apply_calendar, resolve_calendar

        before = len(prices)
        prices = apply_calendar(prices, protocol)
        logger.info(
            "Calendar %s: %d of %d rows are sessions",
            resolve_calendar(protocol).calendar_id, len(prices), before,
        )

    returns = compute_returns(prices)

    # --- Sentiment engine (scrapes once, caches for 1 h) ---
    nlp_components = None
    if use_sentiment:
        try:
            from .sentiment import SentimentEngine

            engine = SentimentEngine(scorer=sentiment_backend)
            base_tickers = [a.ticker for a in UNIVERSE][:5]
            nlp_components = engine.as_fomo_components(tickers=base_tickers)
            logger.info(
                "Sentiment engine: %s",
                {k: round(v, 4) for k, v in nlp_components.items()},
            )
        except Exception as exc:
            logger.warning("Sentiment engine unavailable (%s)", exc)

    try:
        fomo_indicator = compute_fomo_fobi_indicator(
            prices, nlp_components=nlp_components,
        )
    except ValueError as exc:
        print(f"Warning: could not compute FOMO/FOBI indicator — {exc}")
        fomo_indicator = pd.DataFrame(index=prices.index)
    indicator_aligned = fomo_indicator.reindex(prices.index).ffill() if not fomo_indicator.empty else None

    eval_dates = _evaluation_dates(returns, warmup_days, step)
    if not eval_dates:
        raise ValueError("Not enough data to run historical analysis. Try reducing warmup window.")

    timeline_rows: List[Dict[str, float]] = []
    weight_rows: List[Dict[str, float]] = []

    prev_weights_vec = None
    prev_standard_vec = None
    baseline_regime = "risk_on"
    strategy_suite = StrategySuite()
    # Gate the ensemble on the FIRST evaluation date: a model is only admissible
    # for this backtest if it was trained entirely before the earliest date it
    # would be asked to predict. A model trained on the full timeline — which is
    # what the dashboard produces — is refused, and `ml` mode is simply absent.
    # Losing a strategy mode is the correct outcome; the alternative is publishing
    # look-ahead-contaminated results under an `ml` label.
    ensemble_models = load_ensemble_models(as_of=eval_dates[0])
    if not ensemble_models:
        logger.info("ML regime mode disabled — no look-ahead-clean ensemble available.")
    strategy_modes = ["rule_based", "none"] + (["ml"] if ensemble_models else [])
    strategy_column_map: Dict[str, str] = {}

    # --- ML Forecaster (train once, rolling-retrain inside loop) ---
    forecaster = None
    all_features = None
    if use_forecaster:
        try:
            from .forecaster import ReturnForecaster
            from .features_alpha import build_universe_features

            forecaster = ReturnForecaster(backend=forecaster_backend)
            all_features = build_universe_features(prices)
            logger.info(
                "ML forecaster (%s) enabled — features shape %s",
                forecaster_backend, all_features.shape,
            )
        except Exception as exc:
            logger.warning(
                "Forecaster init failed (%s) — using exponential mean",
                exc,
            )

    for date in eval_dates:
        prices_window = prices.loc[:date]
        returns_window = returns.loc[:date]
        base_returns = returns_window[[asset.ticker for asset in UNIVERSE]]

        regime = detect_regime(prices_window, returns_window)
        # Regime-aware statistics: exponentially weighted with regime-specific decay
        cov = exponential_cov(base_returns)
        rf_rate = rf_from_sgov(prices_window)

        # --- Expected returns (μ): ML forecaster or exponential mean ---
        if forecaster is not None and all_features is not None:
            try:
                features_window = all_features.loc[:date]
                result = forecaster.rolling_retrain(
                    prices_window, returns_window,
                    features=features_window,
                )
                mu = result.mu
                fallback = exponential_mean(base_returns)
                if result.drift_report and result.drift_report.is_drifted:
                    logger.debug(
                        "Forecaster drift at %s (z=%.2f) — blending",
                        date.date(), result.drift_report.value,
                    )
                    mu = 0.7 * mu + 0.3 * fallback
            except Exception as exc:
                logger.debug("Forecaster error at %s: %s", date.date(), exc)
                mu = exponential_mean(base_returns)
        else:
            mu = exponential_mean(base_returns)

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

        if timeline_rows:
            timeline_so_far = pd.DataFrame(timeline_rows).set_index("date").sort_index()
        else:
            timeline_so_far = None
        indicator_point = None
        if indicator_aligned is not None and date in indicator_aligned.index:
            indicator_point = indicator_aligned.loc[date]

        indicator_context = {}
        if indicator_point is not None:
            indicator_context = {
                "fomo_fobi": {
                    "score": float(indicator_point.get("fomo_fobi_score", float("nan"))),
                    "state": str(indicator_point.get("fomo_fobi_state", "neutral")),
                    "probability": float(indicator_point.get("fomo_probability", float("nan"))),
                    "components": {
                        name: float(indicator_point.get(f"component_{name}_z", float("nan")))
                        for name in FOMO_COMPONENT_WEIGHTS
                    },
                }
            }
        nowcasts = compute_nowcasts(prices_window)
        indicator_context.setdefault("nowcasts", nowcasts)

        strategy_evaluations = []
        for mode in strategy_modes:
            eval_kwargs: Dict[str, object] = {}
            if mode == "ml":
                if not ensemble_models or timeline_so_far is None or timeline_so_far.empty:
                    continue
                eval_kwargs = {"timeline": timeline_so_far, "ensemble_models": ensemble_models}
            results = strategy_suite.evaluate(
                prices_window,
                base_returns,
                detection_mode=mode,
                extra_context=indicator_context,
                **eval_kwargs,
            )
            strategy_evaluations.append((mode, results))
            for strat_name in results:
                key = f"{mode}:{strat_name}"
                column = f"strategy_{mode}_{strat_name}_weight"
                strategy_column_map.setdefault(key, column)

        vix_value = float(prices_window["^VIX"].iloc[-1]) if "^VIX" in prices_window.columns else float("nan")
        gld_value = float(prices_window["GLD"].iloc[-1]) if "GLD" in prices_window.columns else float("nan")
        gold_oz_price = gld_value * GLD_SHARE_TO_OUNCE if pd.notna(gld_value) else float("nan")
        timeline_entry = {
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
            **{
                f"strategy_{mode}_{strat}_sharpe": outcome.metrics.get("sharpe", float("nan"))
                for mode, results in strategy_evaluations
                for strat, outcome in results.items()
            },
        }
        for name, value in nowcasts.items():
            timeline_entry[f"nowcast_{name}"] = float(value)

        if indicator_point is not None:
            timeline_entry["fomo_fobi_score"] = float(indicator_point.get("fomo_fobi_score", float("nan")))
            timeline_entry["fomo_probability"] = float(indicator_point.get("fomo_probability", float("nan")))
            timeline_entry["fomo_fobi_state"] = str(indicator_point.get("fomo_fobi_state", "neutral"))
            for name in FOMO_COMPONENT_WEIGHTS:
                column = f"component_{name}_z"
                timeline_entry[f"fomo_component_{name}_z"] = float(indicator_point.get(column, float("nan")))
        else:
            timeline_entry["fomo_fobi_score"] = float("nan")
            timeline_entry["fomo_probability"] = float("nan")
            timeline_entry["fomo_fobi_state"] = "neutral"
            for name in FOMO_COMPONENT_WEIGHTS:
                timeline_entry[f"fomo_component_{name}_z"] = float("nan")

        timeline_rows.append(timeline_entry)
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
                    **{
                        f"strategy_{mode}_{strat}_weight": outcome.weights.get(ticker, 0.0)
                        for mode, results in strategy_evaluations
                        for strat, outcome in results.items()
                    },
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
    for key, column in strategy_column_map.items():
        mode, strat = key.split(":", 1)
        label = f"strategy_{mode}_{strat}"
        strategy_map[label] = column
    performance = {
        label: _strategy_total_return(weights_df, asset_returns, column, protocol)
        for label, column in strategy_map.items()
    }
    for label, value in performance.items():
        timeline_df[f"total_return_{label}"] = value

    strategy_columns = {label: column for label, column in strategy_map.items() if label.startswith("strategy_")}

    return HistoryRunResult(
        timeline=timeline_df,
        weights=weights_df,
        prices=prices,
        performance=performance,
        strategy_columns=strategy_columns,
        fomo_indicator=fomo_indicator,
    )


def strategy_daily_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    column: str,
    *,
    execution_lag: int = EXECUTION_LAG_DAYS,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> pd.Series:
    """Net daily returns for one strategy weight column.

    Two properties this guarantees, both of which the pre-2026-07 implementation
    violated and which every published figure depends on:

    * **Causality.** Weights derived from data through date *d* are shifted by
      ``execution_lag`` before meeting returns, so they earn *d+1* onward. Applying
      them on *d* itself credits the strategy with a day it could not have traded.
    * **Costs.** Turnover between consecutive holdings is charged at ``cost_bps``
      of notional traded. A gross-only backtest is not a publishable number.

    Returns an empty Series when the column is absent or carries no usable weights.
    """
    if column not in weights.columns:
        return pd.Series(dtype=float)

    weight_history = (
        weights.pivot(index="date", columns="ticker", values=column)
        .sort_index()
        .reindex(columns=asset_returns.columns, fill_value=0.0)
    )
    if weight_history.empty:
        return pd.Series(dtype=float)

    # Hold each rebalance until the next one, then lag into executable territory.
    held = weight_history.reindex(asset_returns.index).ffill()
    executable = held.shift(execution_lag)

    # Drop the leading span where no rebalance has happened yet.
    valid_mask = executable.notna().any(axis=1)
    executable = executable.loc[valid_mask].fillna(0.0)
    if executable.empty:
        return pd.Series(dtype=float)

    aligned_returns = asset_returns.loc[executable.index].fillna(0.0)
    gross = (executable * aligned_returns).sum(axis=1)

    # Turnover is charged on the day the trade happens — the first row trades in
    # from cash, so compare against a zero holding rather than dropping it.
    turnover = executable.diff().abs().sum(axis=1)
    turnover.iloc[0] = executable.iloc[0].abs().sum()
    costs = turnover * (cost_bps / 10_000.0)

    net = gross - costs
    net.name = column
    return net


def _annualize(daily_returns: pd.Series, periods_per_year: float = 252.0) -> float:
    """Annualize a compounded return series.

    `periods_per_year` must match the observation frequency actually present. A
    252 assumption applied to a 365-observation year understates the result; the
    evaluation protocol declares the calendar so callers do not have to guess.
    """
    n_days = len(daily_returns)
    if n_days == 0:
        return float("nan")
    total_return = float((1.0 + daily_returns).prod() - 1.0)
    return float((1.0 + total_return) ** (periods_per_year / n_days) - 1.0)


def _strategy_total_return(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    column: str,
    protocol=None,
) -> float:
    """Annualized net return for one weight column, replayed under a protocol.

    When `protocol` is supplied, the execution lag, cost model and annualization
    all come from it — the same object the evaluation runner uses, so the legacy
    engine cannot drift from the runner by holding its own defaults.
    """
    if protocol is None:
        daily_returns = strategy_daily_returns(weights, asset_returns, column)
        return _annualize(daily_returns) if not daily_returns.empty else float("nan")

    from .evaluation.runner import periods_per_year

    daily_returns = strategy_daily_returns(
        weights,
        asset_returns,
        column,
        execution_lag=protocol.transaction_costs.execution_lag_days,
        cost_bps=protocol.transaction_costs.bps,
    )
    if daily_returns.empty:
        return float("nan")
    return _annualize(daily_returns, periods_per_year=periods_per_year(protocol))


def save_history(
    result: HistoryRunResult,
    params: Dict[str, object] | None = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, str]:
    result.timeline.to_parquet(TIMELINE_PATH)
    result.weights.to_parquet(WEIGHTS_PATH)
    result.prices.to_parquet(PRICES_PATH)
    if result.fomo_indicator is not None and not result.fomo_indicator.empty:
        result.fomo_indicator.to_parquet(FOMO_PATH)

    # The manifest is what makes a published number checkable: it pins the code,
    # the environment, the parameters, and content hashes of both the input price
    # snapshot and the outputs derived from it.
    manifest = build_run_manifest(
        run_id=pd.Timestamp.now("UTC").strftime("run_%Y%m%dT%H%M%SZ"),
        seed=seed,
        params=params or {},
        inputs={"prices": frame_digest(result.prices)},
        outputs={
            "timeline": frame_digest(result.timeline),
            "weights": frame_digest(result.weights),
        },
    )
    manifest.write(MANIFEST_PATH)

    payload = {
        "timeline": result.timeline.reset_index().to_dict(orient="list"),
        "weights": result.weights.to_dict(orient="list"),
        "performance": result.performance,
        "strategy_columns": result.strategy_columns,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, default=str))

    return {
        "timeline": str(TIMELINE_PATH),
        "weights": str(WEIGHTS_PATH),
        "prices": str(PRICES_PATH),
        "summary": str(SUMMARY_JSON),
        "manifest": str(MANIFEST_PATH),
        "fomo_indicator": str(FOMO_PATH) if result.fomo_indicator is not None and not result.fomo_indicator.empty else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical regime-adjusted allocation analysis")
    parser.add_argument("--start", required=True, type=datetime.fromisoformat, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=datetime.fromisoformat, help="End date YYYY-MM-DD")
    parser.add_argument("--warmup", type=int, default=252, help="Warmup window in trading days")
    parser.add_argument("--step", type=int, default=5, help="Evaluate every N trading days")
    parser.add_argument("--refresh", action="store_true", help="Force data refresh from Yahoo")
    parser.add_argument("--use-forecaster", action="store_true", help="Use ML forecaster for expected returns")
    parser.add_argument("--forecaster-backend", default="lightgbm", choices=["lightgbm", "lstm", "transformer"], help="Forecaster backend")
    parser.add_argument("--use-sentiment", action="store_true", help="Enable NLP sentiment engine")
    parser.add_argument("--sentiment-backend", default="auto", choices=["auto", "vader", "fingpt"], help="Sentiment scorer backend")
    parser.add_argument(
        "--protocol",
        default="standard@1",
        help=(
            "Evaluation protocol to replay under. Supplies the trading calendar, "
            "execution lag, cost model and annualization. Pass 'none' for the "
            "pre-protocol defaults (not recommended: they are not a source of truth)."
        ),
    )
    args = parser.parse_args()

    protocol = None
    if args.protocol and args.protocol.lower() != "none":
        from .evaluation import ProtocolRegistry

        protocol = ProtocolRegistry().resolve(args.protocol)
        print(f"Replaying under {protocol.protocol_id} "
              f"(calendar {protocol.walk_forward.calendar}, "
              f"{protocol.transaction_costs.bps}bps, "
              f"lag {protocol.transaction_costs.execution_lag_days}d)")

    result = run_historical_analysis(
        start=args.start,
        end=args.end,
        warmup_days=args.warmup,
        step=args.step,
        force_refresh=args.refresh,
        use_forecaster=args.use_forecaster,
        forecaster_backend=args.forecaster_backend,
        use_sentiment=args.use_sentiment,
        sentiment_backend=args.sentiment_backend,
        protocol=protocol,
    )
    paths = save_history(result, params={"protocol": protocol.protocol_id if protocol else None})
    print("Historical analysis saved:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


def strategy_cumulative_returns(weights: pd.DataFrame, asset_returns: pd.DataFrame, column: str) -> pd.Series:
    """Cumulative growth series for a strategy weight column, net of costs.

    Shares :func:`strategy_daily_returns` with the headline metric so the growth
    curve and the annualized figure can never disagree about lag or costs.
    """
    daily_returns = strategy_daily_returns(weights, asset_returns, column)
    if daily_returns.empty:
        return pd.Series(dtype=float)
    cumulative = (1.0 + daily_returns).cumprod()
    cumulative = cumulative / cumulative.iloc[0]
    cumulative.name = column
    return cumulative


if __name__ == "__main__":
    main()
