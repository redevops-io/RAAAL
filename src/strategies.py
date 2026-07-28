"""Strategy testing utilities for the expanded strategy families."""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_RF, MANDATE_CONSTRAINTS, UNIVERSE
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


# ---------------------------------------------------------------------------
# IBS Hybrid Switch  (regime-conditional mean-reversion)
# ---------------------------------------------------------------------------
# Inspired by the Internal Bar Strength (IBS) dip-buying system:
#   Calm market  (VIX < threshold & SPY > SMA-200) → overweight equities
#   Volatile mkt (VIX >= threshold | SPY < SMA-200) → IBS dip-buy on pullbacks
#
# IBS = (Close − Low) / (High − Low)   ∈ [0, 1]
# Low IBS ≈ close near day's low → oversold → mean-reversion buy signal.
#
# Because RAAAL uses daily *close-only* data from yfinance, we approximate
# intraday bar structure with a rolling min/max proxy:
#   IBS_proxy = (close − rolling_low_5d) / (rolling_high_5d − rolling_low_5d)
#
# Regime filter mirrors the original: VIX < 22 AND SPY above 200-day SMA.
# ---------------------------------------------------------------------------

_IBS_VIX_THRESHOLD = 22.0
_IBS_SMA_WINDOW = 200
_IBS_RANGE_WINDOW = 5      # rolling H/L window for IBS proxy
_IBS_BUY_THRESHOLD = 0.2   # IBS below this → dip buy
_IBS_RSI_WINDOW = 2         # ultra-short RSI confirmation


def _compute_ibs_proxy(
    spy_prices: pd.Series,
    window: int = _IBS_RANGE_WINDOW,
) -> pd.Series:
    """Approximate IBS using rolling high/low over *window* days."""
    rolling_high = spy_prices.rolling(window, min_periods=1).max()
    rolling_low = spy_prices.rolling(window, min_periods=1).min()
    span = rolling_high - rolling_low
    # Avoid division by zero on flat stretches
    span = span.replace(0.0, np.nan)
    ibs = (spy_prices - rolling_low) / span
    return ibs.fillna(0.5)


def _compute_rsi(series: pd.Series, window: int = _IBS_RSI_WINDOW) -> float:
    """RSI(2) of the most recent bar — used as confirmation."""
    delta = series.diff()
    gain = delta.clip(lower=0.0).rolling(window, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window, min_periods=1).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    last = rsi.iloc[-1] if not rsi.empty else 50.0
    return float(last) if not np.isnan(last) else 50.0


def ibs_hybrid_switch_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    """IBS Hybrid Switch: regime-gated mean-reversion dip-buying.

    Calm regime  → overweight equity + growth assets (momentum hold).
    Volatile     → if IBS is low (dip), buy equities; else park in cash/bonds.
    """
    # --- Regime filter: VIX + SMA-200 ---
    vix_col = "^VIX"
    spy_col = "SPY"
    if spy_col not in prices.columns:
        return _normalize_weights({CASH_TICKER: 1.0})

    spy = prices[spy_col].dropna()
    if len(spy) < _IBS_SMA_WINDOW:
        # Not enough data — fall back to balanced
        return _normalize_weights({
            spy_col: 0.4, "TLT": 0.3,
            "GLD": 0.15, CASH_TICKER: 0.15,
        })

    sma200 = spy.rolling(_IBS_SMA_WINDOW).mean().iloc[-1]
    spy_last = spy.iloc[-1]
    spy_above_sma = spy_last > sma200

    vix_last = 20.0  # default if VIX unavailable
    if vix_col in prices.columns:
        vix_series = prices[vix_col].dropna()
        if not vix_series.empty:
            vix_last = float(vix_series.iloc[-1])

    calm_regime = (vix_last < _IBS_VIX_THRESHOLD) and spy_above_sma

    if calm_regime:
        # ------ CALM: hold leveraged / momentum-biased equity ------
        # In the real system this maps to 2× leveraged ETFs.
        # In RAAAL's universe, overweight SPY + growth-correlated assets.
        return _normalize_weights({
            spy_col: 0.60,
            "HYG": 0.15,   # high-yield as equity proxy
            "DBC": 0.10,   # commodities momentum
            "GLD": 0.10,
            CASH_TICKER: 0.05,
        })

    # ------ VOLATILE: IBS dip-buying logic ------
    ibs = _compute_ibs_proxy(spy)
    ibs_last = float(ibs.iloc[-1])
    rsi2 = _compute_rsi(spy)

    if ibs_last < _IBS_BUY_THRESHOLD and rsi2 < 30.0:
        # Strong dip signal → aggressive equity entry
        return _normalize_weights({
            spy_col: 0.55,
            "HYG": 0.10,
            "TLT": 0.15,
            CASH_TICKER: 0.20,
        })
    elif ibs_last < 0.4:
        # Mild dip → cautious equity tilt
        return _normalize_weights({
            spy_col: 0.30,
            "TLT": 0.30,
            "GLD": 0.15,
            CASH_TICKER: 0.25,
        })
    else:
        # No dip → defensive / wait
        return _normalize_weights({
            "TLT": 0.30,
            "GLD": 0.15,
            CASH_TICKER: 0.55,
        })


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
    StrategySpec("ibs_hybrid_switch", "regime_mean_reversion", False, ibs_hybrid_switch_strategy),
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

# DRL strategies are optional — only available when gymnasium + SB3 are installed.
try:
    from .drl_strategy import (
        adaptive_rotation_strategy,
        drl_portfolio_strategy,
    )
    DEFAULT_STRATEGIES += [
        StrategySpec("drl_portfolio", "reinforcement_learning", False, drl_portfolio_strategy),
        StrategySpec("adaptive_rotation", "reinforcement_learning", True, adaptive_rotation_strategy),
    ]
except ImportError:  # gymnasium / stable-baselines3 not installed
    pass


# ===========================================================================
# Strategy capability registry — the authoritative capability graph used by
# the Decision Planner and the learning selector.
#
# The three user objectives are RANKING/SELECTION policies over THIS registry
# (see src/agentic/selection.py). No allocation is ever produced outside a
# registered `implementation`. Adding a new optimizer/strategy requires a new
# registry entry that passes the same research/benchmark/promotion process.
# ===========================================================================


def sharpe_optimizer_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    """The existing guardrailed Sharpe optimizer, exposed as a registered capability.

    This is the ONLY optimizer in the production registry (Amendments §5); it keeps
    its role as the max-return-to-risk RAAAL strategy and is not repurposed to
    fabricate the other objectives.
    """
    from .optimizer import optimize_weights  # local import avoids import cycle

    rf = float((context or {}).get("rf", DEFAULT_RF))
    reg = regime if regime in {"risk_on", "risk_off", "inflation"} else "risk_on"
    try:
        return _normalize_weights(optimize_weights(returns, reg, rf_rate=rf))
    except Exception:  # pragma: no cover - solver edge cases fall back to cash
        return _normalize_weights({CASH_TICKER: 1.0})


# Members of the approved Strategy-Lab composite (governed ensemble, Amendments §7).
COMPOSITE_MEMBERS: Tuple[str, ...] = (
    "dual_momentum",
    "sharpe_optimizer",
    "risk_parity",
    "multi_factor_blend",
)


def raaal_composite_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    """Approved ensemble: an EQUAL-WEIGHT blend of named member strategies.

    Reuses the Strategy-Lab composite idea but governs it explicitly — the members
    and weighting rule are declared (COMPOSITE_MEMBERS, equal-weight). The planner
    may never invent an opaque blend.
    """
    member_weights = []
    for member_id in COMPOSITE_MEMBERS:
        try:
            member_weights.append(run_capability(member_id, prices, returns, regime, context))
        except Exception:  # pragma: no cover - skip a failing member
            continue
    if not member_weights:
        return _normalize_weights({CASH_TICKER: 1.0})
    return _blend_weights(member_weights)


@dataclass(frozen=True)
class StrategyCapability:
    """One registered, research-backed strategy the planner/learning may select."""

    id: str
    family: str
    implementation: str                      # dotted path, e.g. "src.strategies.momentum_dual_strategy"
    supported_objectives: Tuple[str, ...]    # subset of config.OBJECTIVES
    allowed_regimes: Tuple[str, ...] = ("risk_on", "risk_off", "inflation")
    evidence_requirements: Tuple[str, ...] = ("prices", "returns", "regime")
    data_freshness_days: int = 3
    hard_constraints: Tuple[str, ...] = ("mandate", "cash_floor", "turnover_cap")
    risk_profile: str = "moderate"           # defensive | balanced | moderate | aggressive
    expected_behavior: str = ""
    benchmark_status: str = "validated"      # validated | provisional | unbenchmarked
    research_refs: Tuple[str, ...] = ()
    known_failure_modes: Tuple[str, ...] = ()
    min_history: int = 126
    txn_cost_sensitivity: str = "medium"     # low | medium | high
    promotion_status: str = "approved"       # experimental | shadow | approved | deprecated
    is_ensemble: bool = False
    ensemble_members: Tuple[str, ...] = ()
    fallback_rank: int = 100

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "family": self.family,
            "implementation": self.implementation,
            "supported_objectives": list(self.supported_objectives),
            "allowed_regimes": list(self.allowed_regimes),
            "risk_profile": self.risk_profile,
            "expected_behavior": self.expected_behavior,
            "benchmark_status": self.benchmark_status,
            "research_refs": list(self.research_refs),
            "known_failure_modes": list(self.known_failure_modes),
            "promotion_status": self.promotion_status,
            "is_ensemble": self.is_ensemble,
            "ensemble_members": list(self.ensemble_members),
            "min_history": self.min_history,
            "txn_cost_sensitivity": self.txn_cost_sensitivity,
        }


_ALL = ("risk_on", "risk_off", "inflation")
_RES_REFS = (
    "Vuletic (2025), Multi-asset financial markets (repo monograph)",
    "AI in Asset Management (CFA Research Foundation monograph, repo)",
)

# Authoritative registry. supported_objectives decide which objective column may
# select a capability; promotion_status gates what the LIVE path may use.
STRATEGY_REGISTRY: List[StrategyCapability] = [
    # --- return-seeking (momentum) ---
    StrategyCapability("time_series_momentum", "momentum", "src.strategies.momentum_time_series_strategy",
        ("max_total_return", "max_return_to_risk"), risk_profile="aggressive", min_history=63,
        txn_cost_sensitivity="high", expected_behavior="Buys trailing 3m winners; strong in trends, whipsaws in chop.",
        research_refs=_RES_REFS, known_failure_modes=("regime turning points", "high turnover"), fallback_rank=20),
    StrategyCapability("cross_sectional_momentum", "momentum", "src.strategies.momentum_cross_sectional_strategy",
        ("max_total_return", "max_return_to_risk"), risk_profile="aggressive", min_history=63,
        txn_cost_sensitivity="high", expected_behavior="Concentrates in the top-3 momentum assets.",
        research_refs=_RES_REFS, known_failure_modes=("concentration", "reversals"), fallback_rank=22),
    StrategyCapability("dual_momentum", "momentum", "src.strategies.momentum_dual_strategy",
        ("max_total_return", "max_return_to_risk"), risk_profile="aggressive", min_history=126,
        txn_cost_sensitivity="high", expected_behavior="Absolute+relative momentum vs cash; rotates to cash in downtrends.",
        research_refs=_RES_REFS, known_failure_modes=("sharp V-shaped recoveries",), fallback_rank=10),
    StrategyCapability("regime_momentum", "momentum", "src.strategies.momentum_regime_strategy",
        ("max_total_return", "max_return_to_risk"), risk_profile="aggressive", min_history=63,
        txn_cost_sensitivity="medium", expected_behavior="Momentum within the regime's preferred bucket.",
        research_refs=_RES_REFS, known_failure_modes=("regime misclassification",), fallback_rank=18),
    # --- relative value / mean reversion ---
    StrategyCapability("relative_value", "relative_value_mean_reversion", "src.strategies.relative_value_strategy",
        ("max_return_to_risk",), risk_profile="moderate", min_history=63, txn_cost_sensitivity="high",
        expected_behavior="Contrarian tilt toward recent laggards.", research_refs=_RES_REFS,
        known_failure_modes=("persistent trends",), fallback_rank=40),
    StrategyCapability("pairs_trading", "relative_value_mean_reversion", "src.strategies.pairs_trading_strategy",
        ("max_return_to_risk",), risk_profile="moderate", min_history=63, txn_cost_sensitivity="high",
        expected_behavior="SPY/TLT mean-reversion pair.", research_refs=_RES_REFS,
        known_failure_modes=("correlation breakdown",), fallback_rank=44),
    StrategyCapability("stat_arb_credit", "relative_value_mean_reversion", "src.strategies.stat_arbitrage_strategy",
        ("max_return_to_risk",), risk_profile="moderate", min_history=63, txn_cost_sensitivity="medium",
        expected_behavior="LQD/HYG credit-spread reversion.", research_refs=_RES_REFS,
        known_failure_modes=("credit stress",), fallback_rank=46),
    StrategyCapability("reversal", "relative_value_mean_reversion", "src.strategies.reversal_strategy",
        ("max_return_to_risk",), risk_profile="moderate", min_history=42, txn_cost_sensitivity="high",
        expected_behavior="Short-term (1m) reversal toward laggards.", research_refs=_RES_REFS,
        known_failure_modes=("momentum regimes",), fallback_rank=48),
    StrategyCapability("ibs_hybrid_switch", "regime_mean_reversion", "src.strategies.ibs_hybrid_switch_strategy",
        ("max_total_return", "max_return_to_risk"), risk_profile="aggressive", min_history=200,
        txn_cost_sensitivity="medium", expected_behavior="Calm: overweight equity; volatile: IBS dip-buy or defend.",
        research_refs=_RES_REFS, known_failure_modes=("gap-down regimes",), fallback_rank=16),
    # --- defensive / risk-based ---
    StrategyCapability("risk_parity", "risk_based", "src.strategies.risk_parity_strategy",
        ("min_risk", "max_return_to_risk"), risk_profile="defensive", min_history=63, txn_cost_sensitivity="low",
        expected_behavior="Inverse-vol weighting; balanced risk.", research_refs=_RES_REFS,
        known_failure_modes=("all-asset drawdowns",), fallback_rank=6),
    StrategyCapability("minimum_variance", "risk_based", "src.strategies.minimum_variance_strategy",
        ("min_risk",), risk_profile="defensive", min_history=126, txn_cost_sensitivity="low",
        expected_behavior="Minimizes portfolio variance; concentrates in low-vol assets.", research_refs=_RES_REFS,
        known_failure_modes=("estimation error in cov",), fallback_rank=4),
    StrategyCapability("max_diversification", "risk_based", "src.strategies.max_diversification_strategy",
        ("min_risk", "max_return_to_risk"), risk_profile="defensive", min_history=126, txn_cost_sensitivity="low",
        expected_behavior="Maximizes diversification ratio (vol / avg correlation).", research_refs=_RES_REFS,
        known_failure_modes=("correlation regime shifts",), fallback_rank=8),
    StrategyCapability("equal_risk_contribution", "risk_based", "src.strategies.equal_risk_contribution_strategy",
        ("min_risk",), risk_profile="defensive", min_history=126, txn_cost_sensitivity="low",
        expected_behavior="Equalizes each asset's risk contribution.", research_refs=_RES_REFS,
        known_failure_modes=("cov estimation",), fallback_rank=7),
    StrategyCapability("volatility_targeting", "risk_based", "src.strategies.volatility_targeting_strategy",
        ("min_risk", "max_return_to_risk"), risk_profile="defensive", min_history=126, txn_cost_sensitivity="medium",
        expected_behavior="Scales risk-parity to a 10% vol target, parking the rest in cash.", research_refs=_RES_REFS,
        known_failure_modes=("vol spikes lag",), fallback_rank=9),
    # --- factor ---
    StrategyCapability("equity_factors", "factor_based", "src.strategies.equity_factor_strategy",
        ("max_total_return", "max_return_to_risk"), risk_profile="moderate", min_history=126, txn_cost_sensitivity="medium",
        expected_behavior="Value/momentum/quality/low-vol/size factor tilts.", research_refs=_RES_REFS,
        known_failure_modes=("factor crowding",), fallback_rank=30),
    StrategyCapability("macro_factors", "factor_based", "src.strategies.macro_factor_strategy",
        ("max_total_return", "max_return_to_risk"), risk_profile="moderate", min_history=84, txn_cost_sensitivity="medium",
        expected_behavior="Growth/inflation/liquidity/real-rate/credit factor tilts.", research_refs=_RES_REFS,
        known_failure_modes=("macro regime shifts",), fallback_rank=32),
    StrategyCapability("multi_factor_blend", "factor_based", "src.strategies.multi_factor_blend_strategy",
        ("max_total_return", "max_return_to_risk"), risk_profile="moderate", min_history=126, txn_cost_sensitivity="medium",
        expected_behavior="Equal blend of equity + macro factor tilts.", research_refs=_RES_REFS,
        known_failure_modes=("compound factor decay",), fallback_rank=28),
    # --- sentiment overlay ---
    StrategyCapability("fomo_fobi_overlay", "sentiment", "src.strategies.fomo_fobi_overlay_strategy",
        ("max_return_to_risk", "min_risk"), risk_profile="defensive", min_history=63, txn_cost_sensitivity="medium",
        expected_behavior="Risk-off on FOMO extremes, risk-on on FOBI (capitulation).",
        research_refs=("alpha salient metrics (Cosemans-Frehen salience, repo)",) + _RES_REFS,
        known_failure_modes=("indicator lag",), fallback_rank=36),
    # --- the guardrailed optimizer, as a first-class capability ---
    StrategyCapability("sharpe_optimizer", "optimizer", "src.strategies.sharpe_optimizer_strategy",
        ("max_return_to_risk",), risk_profile="balanced", min_history=126, txn_cost_sensitivity="low",
        expected_behavior="SLSQP Sharpe maximization with regime guardrails + turnover penalty.",
        research_refs=_RES_REFS, known_failure_modes=("mean estimation error",), fallback_rank=2),
    # --- approved governed ensemble ---
    StrategyCapability("raaal_composite", "ensemble", "src.strategies.raaal_composite_strategy",
        ("max_return_to_risk", "max_total_return"), risk_profile="balanced", min_history=126,
        txn_cost_sensitivity="medium", expected_behavior="Equal-weight blend of dual_momentum + sharpe_optimizer + "
        "risk_parity + multi_factor_blend.", research_refs=_RES_REFS, is_ensemble=True,
        ensemble_members=COMPOSITE_MEMBERS, fallback_rank=12),
]

# DRL capabilities are registered but NOT approved for the live path (shadow only).
if any(spec.name == "drl_portfolio" for spec in DEFAULT_STRATEGIES):
    STRATEGY_REGISTRY += [
        StrategyCapability("drl_portfolio", "reinforcement_learning", "src.drl_strategy.drl_portfolio_strategy",
            ("max_total_return", "max_return_to_risk"), risk_profile="aggressive", benchmark_status="provisional",
            promotion_status="shadow", expected_behavior="RL portfolio policy.", fallback_rank=80),
        StrategyCapability("adaptive_rotation", "reinforcement_learning", "src.drl_strategy.adaptive_rotation_strategy",
            ("max_total_return",), risk_profile="aggressive", benchmark_status="provisional",
            promotion_status="shadow", expected_behavior="RL regime rotation.", fallback_rank=82),
    ]

CAPABILITY_BY_ID: Dict[str, StrategyCapability] = {c.id: c for c in STRATEGY_REGISTRY}


def strategy_capabilities() -> List[Dict[str, object]]:
    """The registry as JSON-serializable metadata (for the API / EXPLAIN / UI)."""
    return [c.as_dict() for c in STRATEGY_REGISTRY]


def registry_for_objective(
    objective: str,
    regime: Optional[str] = None,
    promotion: Sequence[str] = ("approved",),
) -> List[StrategyCapability]:
    """Eligible capabilities for an objective + regime + promotion status (the pre-selection prune)."""
    out = []
    for c in STRATEGY_REGISTRY:
        if objective not in c.supported_objectives:
            continue
        if c.promotion_status not in promotion:
            continue
        if regime is not None and regime not in c.allowed_regimes:
            continue
        out.append(c)
    return sorted(out, key=lambda c: c.fallback_rank)


def _resolve_implementation(cap: StrategyCapability) -> StrategyFn:
    module_path, _, fn_name = cap.implementation.rpartition(".")
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"capability {cap.id}: {cap.implementation} is not callable")
    return fn  # type: ignore[return-value]


def run_capability(
    capability_id: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Produce weights from a REGISTERED capability's implementation (the only way weights are made)."""
    cap = CAPABILITY_BY_ID.get(capability_id)
    if cap is None:
        raise KeyError(f"unknown strategy capability: {capability_id}")
    fn = _resolve_implementation(cap)
    return _normalize_weights(fn(prices, returns, regime, context or {}))


def apply_mandate(
    weights: Dict[str, float],
    mandate: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """Clip weights to the HARD mandate (long-only, inverse cap, crypto cap, cash floor) and renormalize.

    Returns (constrained_weights, binding_constraints). Applied to every recommendation so no allocation
    can violate the mandate regardless of which strategy produced it.
    """
    m = mandate or MANDATE_CONSTRAINTS
    binding: List[str] = []
    w = {t: max(0.0, float(weights.get(t, 0.0))) for t in TICKERS}  # long_only

    inverse = [a.ticker for a in UNIVERSE if a.is_inverse]
    inv_sum = sum(w[t] for t in inverse)
    inv_cap = float(m.get("inverse_exposure_cap", 1.0))
    if inv_sum > inv_cap and inv_sum > 0:
        scale = inv_cap / inv_sum
        for t in inverse:
            w[t] *= scale
        binding.append(f"inverse_exposure_cap {inv_cap:.0%}")

    crypto_cap = float(m.get("crypto_cap", 1.0))
    for a in UNIVERSE:
        if a.asset_class == "crypto" and w[a.ticker] > crypto_cap:
            w[a.ticker] = crypto_cap
            binding.append(f"crypto_cap {crypto_cap:.0%}")

    total = sum(w.values())
    if total <= 0:
        w = {t: 0.0 for t in TICKERS}
        w[CASH_TICKER] = 1.0
        return w, ["minimum_cash (forced all-cash)"]
    w = {t: v / total for t, v in w.items()}

    floor = float(m.get("minimum_cash", 0.0))
    if w.get(CASH_TICKER, 0.0) < floor:
        cash = floor
        others = {t: v for t, v in w.items() if t != CASH_TICKER}
        os = sum(others.values())
        if os > 0:
            for t in others:
                w[t] = others[t] / os * (1.0 - cash)
        w[CASH_TICKER] = cash
        binding.append(f"minimum_cash {floor:.0%}")
    return w, binding


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
