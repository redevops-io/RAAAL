"""Rule-based regime detection aligned with the MVP brief."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .config import MA_LONG, REGIME_RULES
from .features import (
    commodity_momentum,
    corr_spy_tlt,
    credit_spread_momentum,
    tip_momentum,
)


@dataclass
class RegimeResult:
    name: str
    matches: Dict[str, float]
    diagnostics: Dict[str, float]


VIX_RISK_ON = 18.0
VIX_RISK_OFF = 22.0


def _moving_average(prices: pd.Series, window: int) -> float:
    ma = prices.rolling(window=window).mean().dropna()
    if ma.empty:
        return float("nan")
    return float(ma.iloc[-1])


def detect_regime(prices: pd.DataFrame, returns: pd.DataFrame) -> RegimeResult:
    latest_prices = prices.iloc[-1]
    spy_price = float(latest_prices.get("SPY", float("nan")))
    spy_ma = _moving_average(prices["SPY"], MA_LONG)
    spy_above_ma = spy_price > spy_ma

    vix_price = float(latest_prices.get("^VIX", float("nan")))
    credit_signal = credit_spread_momentum(returns)
    corr_signal = corr_spy_tlt(returns)
    commod_signal = commodity_momentum(returns)
    tip_signal = tip_momentum(prices)

    matches = {
        "risk_on": float(spy_above_ma) + float(vix_price < VIX_RISK_ON) + float(credit_signal > 0) + float(corr_signal <= 0.2),
        "risk_off": float(not spy_above_ma) + float(vix_price > VIX_RISK_OFF) + float(credit_signal < 0) + float(corr_signal <= -0.3),
        "inflation": float(corr_signal > 0) + float(tip_signal < 0) + float(commod_signal > 0),
    }

    # Prioritize the regime with most satisfied conditions; tie-breaker risk_off -> inflation -> risk_on
    ordered = sorted(
        matches.items(),
        key=lambda kv: (kv[1], kv[0] == "risk_on", kv[0] == "inflation"),
        reverse=True,
    )
    candidate, _ = ordered[0]

    diagnostics = {
        "spy_price": spy_price,
        "spy_ma200": spy_ma,
        "vix": vix_price,
        "credit_signal": credit_signal,
        "spy_tlt_corr": corr_signal,
        "commodity_mom": commod_signal,
        "tip_mom": tip_signal,
    }

    return RegimeResult(name=candidate, matches=matches, diagnostics=diagnostics)
