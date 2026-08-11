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
# Weights redistributed to include 4 NLP-derived sentiment components
FOMO_COMPONENT_WEIGHTS: Dict[str, float] = {
    # --- Price-based components (original, slightly reduced) ---
    "breadth": 0.14,
    "mega_cap": 0.09,
    "tech_leadership": 0.08,
    "small_cap_leadership": 0.06,
    "dispersion": 0.08,
    "cash_shortage": 0.09,
    "liquidity_stress": 0.08,
    "berkshire_cash": 0.05,
    "vol_complacency": 0.06,
    "options_hedging": 0.04,
    # --- NLP sentiment components (new — from sentiment.py) ---
    "news_sentiment_momentum": 0.08,
    "social_media_intensity": 0.05,
    "fear_language_ratio": 0.06,
    "fed_hawkishness": 0.04,
}

FOMO_SCORE_THRESHOLDS = {
    "fomo": 0.75,
    "fobi": -0.75,
}

FOMO_LONG_LOOKBACK = 252
FOMO_SHORT_LOOKBACK = 63

# --- Execution model -------------------------------------------------------
# Weights decided using data through date d are executable no earlier than d+1.
# A backtest that applies them on d itself earns that day's return with
# knowledge of it. Expressed in trading days so it can be widened for slower
# rebalance cycles without touching call sites.
EXECUTION_LAG_DAYS = 1

# Round-trip transaction cost charged on turnover, in basis points of notional
# traded. 10bps is a deliberately conservative default for liquid US ETFs
# (spread + commission + slippage). It is a required input to any published
# performance number, not an optional refinement.
TRANSACTION_COST_BPS = 10.0

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

# ---------------------------------------------------------------------------
# Investment objectives + mandate constraints (the agentic operating layer).
#
# IMPORTANT: the three objectives are NOT three new optimizers. They are
# selection/ranking policies over the existing research-backed strategy
# library (see src/strategies.py registry + src/agentic/selection.py). The
# runtime never fabricates allocations outside that registry.
# ---------------------------------------------------------------------------
OBJECTIVES: Tuple[str, ...] = ("max_total_return", "max_return_to_risk", "min_risk")

OBJECTIVE_LABELS: Dict[str, str] = {
    "max_total_return": "Maximum Total Return",
    "max_return_to_risk": "Maximum Return-to-Risk",
    "min_risk": "Minimum Risk",
}

# Customer-facing explanation of what each objective column means.
OBJECTIVE_DESCRIPTIONS: Dict[str, str] = {
    "max_total_return": (
        "Ranks the return-seeking strategies (momentum, factor, regime dip-buying) by "
        "held-out regime-specific CAGR, with a drawdown guard. Picks the growth engine that "
        "has paid off most in conditions like today's."
    ),
    "max_return_to_risk": (
        "Ranks strategies by held-out regime-specific Sharpe / Sortino. This is the balanced "
        "objective; the guardrailed Sharpe optimizer competes here as one registered strategy."
    ),
    "min_risk": (
        "Ranks the defensive strategies (minimum-variance, risk-parity, max-diversification, "
        "equal-risk-contribution, volatility-targeting) by realized volatility, max drawdown "
        "and CVaR. Picks the calmest allocation, ignoring return chasing."
    ),
}

# Prominent, reused everywhere a recommendation is shown.
DEMO_DISCLAIMER = (
    "DEMO — decision support only, not investment advice. Paper trading; no real orders "
    "are ever placed. Every allocation is produced by a registered, research-backed strategy."
)

# Hard mandate constraints, applied BEFORE any strategy selection or learning.
# Mirrors the InvestmentProject manifest `constraints` block and the semantics
# already encoded in REGIME_CONSTRAINTS / Asset.upper / Asset.is_inverse.
MANDATE_CONSTRAINTS: Dict[str, float | bool] = {
    "long_only": True,
    "leverage_cap": 1.0,
    "inverse_exposure_cap": 0.15,
    "crypto_cap": 0.10,
    "minimum_cash": 0.05,
    "maximum_turnover": 0.25,
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
