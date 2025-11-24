"""Strategy testing utilities for the expanded strategy families."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import UNIVERSE
from .ensemble_regime import prepare_features, predict_regime_ensemble
from .features import exponential_cov, exponential_mean
from .portfolio_utils import portfolio_metrics, rf_from_sgov
from .regime import detect_regime

TICKERS = [asset.ticker for asset in UNIVERSE]
CASH_TICKER = "BIL" if "BIL" in TICKERS else TICKERS[-1]

StrategyFn = Callable[[pd.DataFrame, pd.DataFrame, Optional[str], Dict[str, object]], Dict[str, float]]


@dataclass
class StrategySpec:
    name: str
    category: str
    requires_regime: bool
    fn: StrategyFn


@dataclass
class StrategyResult:
    name: str
    category: str
    weights: Dict[str, float]
    metrics: Dict[str, float]
    regime_used: Optional[str]


def _base_weight_dict() -> Dict[str, float]:
    return {ticker: 0.0 for ticker in TICKERS}


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    normalized = _base_weight_dict()
    for ticker, value in weights.items():
        if ticker in normalized:
            normalized[ticker] = max(float(value), 0.0)
    total = sum(normalized.values())
    if total <= 0:
        normalized[CASH_TICKER] = 1.0
        return normalized
    return {ticker: value / total for ticker, value in normalized.items()}


def _trailing_simple_returns(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    if returns.empty:
        return pd.Series(0.0, index=TICKERS)
    window = min(window, len(returns))
    windowed = returns.tail(window)
    total_log = windowed.reindex(columns=TICKERS).sum(axis=0).fillna(0.0)
    simple = np.expm1(total_log)
    return pd.Series(simple, index=TICKERS)


def _annualized_volatility(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    if returns.empty:
        return pd.Series(1.0, index=TICKERS)
    window = min(window, len(returns))
    windowed = returns.tail(window)
    vol = windowed.reindex(columns=TICKERS).std().replace(0.0, np.nan)
    vol = vol * np.sqrt(252)
    mean_vol = vol.mean()
    if np.isnan(mean_vol) or mean_vol == 0:
        mean_vol = 1.0
    return vol.fillna(mean_vol)


def _recent_covariance(returns: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(np.eye(len(TICKERS)) * 1e-4, index=TICKERS, columns=TICKERS)
    window = min(window, len(returns))
    windowed = returns.tail(window)
    cov = windowed.reindex(columns=TICKERS).fillna(0.0).cov()
    if cov.empty:
        cov = pd.DataFrame(np.eye(len(TICKERS)) * 1e-4, index=TICKERS, columns=TICKERS)
    return cov


def _portfolio_vol(weights: Dict[str, float], cov: pd.DataFrame) -> float:
    vec = np.array([weights.get(ticker, 0.0) for ticker in TICKERS])
    try:
        variance = float(vec.T @ cov.values @ vec)
    except ValueError:
        variance = 0.0
    return float(np.sqrt(abs(variance)) * np.sqrt(252))


def _latest_zscore(series: pd.Series, window: int = 63) -> float:
    if series.empty:
        return 0.0
    window = min(window, len(series))
    sample = series.tail(window)
    std = sample.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((sample.iloc[-1] - sample.mean()) / std)


def _blend_weights(weight_sets: Sequence[Dict[str, float]]) -> Dict[str, float]:
    blended = _base_weight_dict()
    for weights in weight_sets:
        for ticker, value in weights.items():
            if ticker in blended:
                blended[ticker] += value
    return _normalize_weights(blended)


REGIME_BUCKETS = {
    "risk_on": ["SPY", "HYG", "LQD", "GLD"],
    "risk_off": ["TLT", "BIL", "GLD", "SH"],
    "inflation": ["DBC", "GLD", "HYG", "BIL"],
}

EQUITY_FACTOR_BUCKETS = {
    "value": ["LQD", "HYG"],
    "momentum": ["SPY", "DBC"],
    "quality": ["TLT", "LQD"],
    "low_vol": ["TLT", "BIL"],
    "size": ["SPY"],
}

MACRO_FACTOR_BUCKETS = {
    "growth": ["SPY", "HYG"],
    "inflation": ["DBC", "GLD"],
    "liquidity": ["BIL", "LQD"],
    "real_rates": ["TLT", "TBT"],
    "credit": ["LQD", "HYG"],
}


def momentum_time_series_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    momentum = _trailing_simple_returns(returns, window=63).clip(lower=0.0)
    return _normalize_weights(momentum.to_dict())


def momentum_cross_sectional_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
    top_n: int = 3,
) -> Dict[str, float]:
    momentum = _trailing_simple_returns(returns, window=63)
    top_assets = momentum.sort_values(ascending=False).head(top_n)
    positive = top_assets[top_assets > 0]
    if positive.empty:
        return _normalize_weights({CASH_TICKER: 1.0})
    weights = {ticker: value for ticker, value in positive.items()}
    return _normalize_weights(weights)


def momentum_dual_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    momentum = _trailing_simple_returns(returns, window=126)
    cash_return = momentum.get(CASH_TICKER, 0.0)
    eligible = momentum[momentum > cash_return]
    if eligible.empty:
        return _normalize_weights({CASH_TICKER: 1.0})
    adjusted = {ticker: value - cash_return for ticker, value in eligible.items()}
    return _normalize_weights(adjusted)


def momentum_regime_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    bucket_key = regime or "risk_on"
    bucket = REGIME_BUCKETS.get(bucket_key, REGIME_BUCKETS["risk_on"])
    momentum = _trailing_simple_returns(returns, window=63)
    weights = {ticker: momentum.get(ticker, 0.0) for ticker in bucket}
    weights[CASH_TICKER] = max(0.1, weights.get(CASH_TICKER, 0.0))
    return _normalize_weights(weights)


def relative_value_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    momentum = _trailing_simple_returns(returns, window=63)
    contrarian = (-momentum).clip(lower=0.0)
    if contrarian.sum() == 0:
        contrarian = pd.Series(1.0, index=TICKERS)
    return _normalize_weights(contrarian.to_dict())


def pairs_trading_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    momentum = _trailing_simple_returns(returns, window=42)
    pair = [ticker for ticker in ["SPY", "TLT"] if ticker in momentum.index]
    if len(pair) < 2:
        return _normalize_weights({CASH_TICKER: 1.0})
    laggard = momentum[pair].idxmin()
    leader = momentum[pair].idxmax()
    weights = {
        laggard: 0.45,
        leader: 0.15,
        "LQD": 0.2,
        CASH_TICKER: 0.2,
    }
    return _normalize_weights(weights)


def stat_arbitrage_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    if not {"LQD", "HYG"}.issubset(returns.columns):
        return _normalize_weights({CASH_TICKER: 1.0})
    spread = returns["LQD"] - returns["HYG"]
    zscore = _latest_zscore(spread, window=63)
    if zscore > 0:
        weights = {"HYG": 0.35, "SPY": 0.25, CASH_TICKER: 0.4}
    else:
        weights = {"LQD": 0.35, "TLT": 0.25, CASH_TICKER: 0.4}
    return _normalize_weights(weights)


def reversal_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    short_momentum = _trailing_simple_returns(returns, window=21)
    laggards = short_momentum.sort_values().head(max(1, len(short_momentum) // 2))
    weights = {ticker: abs(value) for ticker, value in laggards.items()}
    weights[CASH_TICKER] = weights.get(CASH_TICKER, 0.1)
    return _normalize_weights(weights)


def risk_parity_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    vol = _annualized_volatility(returns, window=63).replace(0.0, np.nan)
    inv_vol = 1.0 / vol
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _normalize_weights(inv_vol.to_dict())


def minimum_variance_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    cov = _recent_covariance(returns, window=126)
    ones = np.ones(len(TICKERS))
    try:
        inv_cov = np.linalg.pinv(cov.values)
        raw = inv_cov @ ones
    except np.linalg.LinAlgError:
        raw = np.ones(len(TICKERS))
    weights = {ticker: max(value, 0.0) for ticker, value in zip(TICKERS, raw)}
    return _normalize_weights(weights)


def max_diversification_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    vol = _annualized_volatility(returns, window=63)
    corr = returns.reindex(columns=TICKERS).tail(126).corr().abs()
    scores = {}
    for ticker in TICKERS:
        avg_corr = corr[ticker].mean() if ticker in corr else 1.0
        scores[ticker] = float(vol.get(ticker, 0.0)) / (avg_corr + 1e-6)
    return _normalize_weights(scores)


def equal_risk_contribution_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    rp = risk_parity_strategy(prices, returns, regime, context)
    mv = minimum_variance_strategy(prices, returns, regime, context)
    return _blend_weights([rp, mv])


def volatility_targeting_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
    target_vol: float = 0.10,
) -> Dict[str, float]:
    base = risk_parity_strategy(prices, returns, regime, context)
    cov = _recent_covariance(returns, window=126)
    current_vol = _portfolio_vol(base, cov)
    if current_vol <= 1e-6:
        return base
    scale = min(1.0, target_vol / current_vol)
    scaled = {ticker: weight * scale for ticker, weight in base.items()}
    scaled[CASH_TICKER] = scaled.get(CASH_TICKER, 0.0) + (1.0 - scale)
    return _normalize_weights(scaled)


def equity_factor_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    momentum = _trailing_simple_returns(returns, window=126)
    scores = _base_weight_dict()
    for tickers in EQUITY_FACTOR_BUCKETS.values():
        available = [ticker for ticker in tickers if ticker in momentum.index]
        if not available:
            continue
        factor_score = float(momentum[available].mean())
        if factor_score <= 0:
            continue
        per_asset = factor_score / len(available)
        for ticker in available:
            scores[ticker] += per_asset
    return _normalize_weights(scores)


def macro_factor_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    momentum = _trailing_simple_returns(returns, window=84)
    scores = _base_weight_dict()
    for tickers in MACRO_FACTOR_BUCKETS.values():
        available = [ticker for ticker in tickers if ticker in momentum.index]
        if not available:
            continue
        factor_score = float(momentum[available].mean())
        if factor_score < 0:
            continue
        per_asset = factor_score / len(available)
        for ticker in available:
            scores[ticker] += per_asset
    return _normalize_weights(scores)


def multi_factor_blend_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    equity = equity_factor_strategy(prices, returns, regime, context)
    macro = macro_factor_strategy(prices, returns, regime, context)
    return _blend_weights([equity, macro])


def fomo_fobi_overlay_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    indicator = (context or {}).get("fomo_fobi", {})
    state = indicator.get("state", "neutral").upper()
    score = indicator.get("score", 0.0)
    intensity = float(min(max(abs(score) / 2.0, 0.0), 1.5))

    if state == "FOMO":
        cash_weight = min(0.8, 0.4 + 0.25 * intensity)
        duration_weight = min(0.3, 0.15 + 0.1 * intensity)
        gold_weight = max(0.05, 1.0 - cash_weight - duration_weight)
        weights = {
            CASH_TICKER: cash_weight,
            "TLT": duration_weight,
            "GLD": gold_weight,
        }
    elif state == "FOBI":
        equity_weight = min(1.1, 0.55 + 0.35 * intensity)
        credit_weight = max(0.05, 0.15 + 0.1 * intensity)
        cyclicals = max(0.05, 0.15)
        defensive = max(0.05, 1.0 - equity_weight - credit_weight - cyclicals)
        weights = {
            "SPY": equity_weight,
            "HYG": credit_weight,
            "DBC": cyclicals,
            "LQD": defensive,
        }
    else:
        weights = {
            "SPY": 0.55,
            "LQD": 0.2,
            "TLT": 0.15,
            "GLD": 0.1,
        }
    return _normalize_weights(weights)


DEFAULT_STRATEGIES: List[StrategySpec] = [
    StrategySpec("time_series_momentum", "momentum", False, momentum_time_series_strategy),
    StrategySpec("cross_sectional_momentum", "momentum", False, momentum_cross_sectional_strategy),
    StrategySpec("dual_momentum", "momentum", False, momentum_dual_strategy),
    StrategySpec("regime_momentum", "momentum", True, momentum_regime_strategy),
    StrategySpec("relative_value", "relative_value_mean_reversion", False, relative_value_strategy),
    StrategySpec("pairs_trading", "relative_value_mean_reversion", False, pairs_trading_strategy),
    StrategySpec("stat_arb_credit", "relative_value_mean_reversion", False, stat_arbitrage_strategy),
    StrategySpec("reversal", "relative_value_mean_reversion", False, reversal_strategy),
    StrategySpec("risk_parity", "risk_based", False, risk_parity_strategy),
    StrategySpec("minimum_variance", "risk_based", False, minimum_variance_strategy),
    StrategySpec("max_diversification", "risk_based", False, max_diversification_strategy),
    StrategySpec("equal_risk_contribution", "risk_based", False, equal_risk_contribution_strategy),
    StrategySpec("volatility_targeting", "risk_based", False, volatility_targeting_strategy),
    StrategySpec("equity_factors", "factor_based", False, equity_factor_strategy),
    StrategySpec("macro_factors", "factor_based", False, macro_factor_strategy),
    StrategySpec("multi_factor_blend", "factor_based", False, multi_factor_blend_strategy),
    StrategySpec("fomo_fobi_overlay", "sentiment", False, fomo_fobi_overlay_strategy),
]


class StrategySuite:
    """Evaluate multiple strategies under different regime assumptions."""

    def __init__(
        self,
        strategies: Optional[Sequence[StrategySpec]] = None,
        default_regime: str = "risk_on",
    ) -> None:
        catalog = list(strategies) if strategies else list(DEFAULT_STRATEGIES)
        self._specs: Dict[str, StrategySpec] = {spec.name: spec for spec in catalog}
        self.default_regime = default_regime

    def available_strategies(self) -> List[str]:
        return list(self._specs.keys())

    def evaluate(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        detection_mode: str = "rule_based",
        strategy_names: Optional[Sequence[str]] = None,
        timeline: Optional[pd.DataFrame] = None,
        ensemble_models: Optional[Dict[str, object]] = None,
        extra_context: Optional[Dict[str, object]] = None,
    ) -> Dict[str, StrategyResult]:
        selected_specs = self._select_specs(strategy_names)
        regime_name, regime_meta, regime_prob = self._resolve_regime(
            prices,
            returns,
            detection_mode,
            timeline,
            ensemble_models,
        )
        base_returns = returns.reindex(columns=TICKERS).dropna(how="all")
        if base_returns.empty:
            raise ValueError("returns data is empty; cannot evaluate strategies")
        mu = exponential_mean(base_returns)
        cov = exponential_cov(base_returns)
        rf_rate = rf_from_sgov(prices)
        context = {
            "regime_metadata": regime_meta,
            "regime_probabilities": regime_prob or {},
        }
        if extra_context:
            context.update(extra_context)
        results: Dict[str, StrategyResult] = {}
        for spec in selected_specs:
            effective_regime = regime_name or (self.default_regime if spec.requires_regime else None)
            weights = spec.fn(prices, base_returns, effective_regime, context)
            metrics = portfolio_metrics(weights, mu, cov, rf_rate)
            results[spec.name] = StrategyResult(
                name=spec.name,
                category=spec.category,
                weights=weights,
                metrics=metrics,
                regime_used=effective_regime,
            )
        return results

    def _select_specs(self, strategy_names: Optional[Sequence[str]]) -> List[StrategySpec]:
        if not strategy_names:
            return list(self._specs.values())
        unknown = [name for name in strategy_names if name not in self._specs]
        if unknown:
            raise ValueError(f"Unknown strategies requested: {unknown}")
        return [self._specs[name] for name in strategy_names]

    def _resolve_regime(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        detection_mode: str,
        timeline: Optional[pd.DataFrame],
        ensemble_models: Optional[Dict[str, object]],
    ) -> tuple[Optional[str], Dict[str, object], Optional[Dict[str, float]]]:
        detection_mode = detection_mode or "rule_based"
        if detection_mode == "rule_based":
            regime = detect_regime(prices, returns)
            return regime.name, {"rule_based": regime}, None
        if detection_mode == "ml":
            if timeline is None or ensemble_models is None:
                raise ValueError("timeline and ensemble_models are required for ML regime detection")
            features, _ = prepare_features(timeline)
            latest_features = features.tail(1)
            pred_regime, probabilities = predict_regime_ensemble(latest_features, ensemble_models)
            return pred_regime, {"ml_features": latest_features}, probabilities
        if detection_mode == "none":
            return None, {}, None
        raise ValueError(f"Unsupported detection mode: {detection_mode}")
