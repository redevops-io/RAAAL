"""Central place for ETF universe and regime-specific constraints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Asset:
    ticker: str
    label: str
    lower: float
    upper: float
    is_inverse: bool = False
    asset_class: str = "misc"


# Ordered universe (controls optimizer weight vector ordering)
UNIVERSE: List[Asset] = [
    Asset("SPY", "S&P 500", 0.0, 0.55, asset_class="equity"),
    Asset("SH", "Inverse S&P 500", 0.0, 0.2, is_inverse=True, asset_class="equity"),
    Asset("TLT", "Long Treasuries", 0.0, 0.5, asset_class="bonds"),
    Asset("TBT", "Short Treasuries", 0.0, 0.2, is_inverse=True, asset_class="bonds"),
    Asset("LQD", "Investment Grade Credit", 0.0, 0.35, asset_class="credit"),
    Asset("DBC", "Broad Commodities", 0.0, 0.2, asset_class="commodities"),
    Asset("GLD", "Gold", 0.0, 0.15, asset_class="commodities"),
    Asset("BTC-USD", "Bitcoin", 0.0, 0.1, asset_class="crypto"),
    Asset("HYG", "High Yield Credit", 0.0, 0.2, asset_class="credit"),
    Asset("BIL", "Cash Proxy", 0.05, 0.8, asset_class="cash"),
]

# Additional series needed for features/regime detection
AUX_SERIES: List[str] = [
    "^VIX",
    "^VVIX",
    "TIP",
    "RSP",  # S&P 500 equal weight (breadth proxy)
    "QQQ",  # Nasdaq 100 (mega-cap tech leadership)
    "MGK",  # Mega-cap growth (Magnificent 7 proxy)
    "IWM",  # Russell 2000 (small-cap breadth)
    "BRK-B",  # Berkshire Hathaway cash proxy
]

# Composite FOMO/FOBI indicator defaults
FOMO_COMPONENT_WEIGHTS: Dict[str, float] = {
    "breadth": 0.18,
    "mega_cap": 0.12,
    "tech_leadership": 0.1,
    "small_cap_leadership": 0.08,
    "dispersion": 0.1,
    "cash_shortage": 0.12,
    "liquidity_stress": 0.1,
    "berkshire_cash": 0.07,
    "vol_complacency": 0.08,
    "options_hedging": 0.05,
}

FOMO_SCORE_THRESHOLDS = {
    "fomo": 0.75,
    "fobi": -0.75,
}

FOMO_LONG_LOOKBACK = 252
FOMO_SHORT_LOOKBACK = 63

# Lookback windows (trading days)
FAST_LOOKBACK = 21
MED_LOOKBACK = 63
SLOW_LOOKBACK = 252
MA_LONG = 200

# Regime thresholds extracted from the MVP brief
REGIME_RULES = {
    "risk_on": {
        "spy_ma": "> 200d",
        "vix": "< 18",
        "credit_spread": "narrowing",
        "spy_tlt_corr": "<= 0.2",
    },
    "risk_off": {
        "spy_ma": "< 200d",
        "vix": "> 22",
        "credit_spread": "widening",
        "spy_tlt_corr": "<= -0.3",
    },
    "inflation": {
        "spy_tlt_corr": "> 0",
        "tip_momentum": "rising yields",
        "commodities": "positive trend",
    },
}

# Regime-specific portfolio guardrails (min/max cash etc.)
REGIME_CONSTRAINTS: Dict[str, Dict[str, float]] = {
    "risk_on": {
        "cash_min": 0.05,
        "cash_max": 0.25,
        "inverse_cap": 0.15,
        "leverage_cap": 1.1,
    },
    "risk_off": {
        "cash_min": 0.2,
        "cash_max": 0.6,
        "inverse_cap": 0.25,
        "leverage_cap": 1.15,
    },
    "inflation": {
        "cash_min": 0.1,
        "cash_max": 0.35,
        "inverse_cap": 0.2,
        "leverage_cap": 1.25,
    },
}

# Default risk-free assumption when cash proxy history unavailable
DEFAULT_RF = 0.02 / 252

# Columns we care about from price data
PRICE_COLUMN = "Adj Close"


def asset_index() -> Dict[str, int]:
    """Map tickers to their index position in the weight vector."""
    return {asset.ticker: idx for idx, asset in enumerate(UNIVERSE)}


TICKER_INDEX = asset_index()


def ordered_tickers() -> List[str]:
    return [asset.ticker for asset in UNIVERSE]
