import pandas as pd

from src.features import compute_returns
from src.regime import detect_regime


def _make_prices(spy_trend: float, vix_level: float, lqd_shift: float, dbc_shift: float, tip_shift: float):
    dates = pd.date_range("2020-01-01", periods=300)
    prices = pd.DataFrame(index=dates)
    prices["SPY"] = 100 + spy_trend * (pd.Series(range(300), index=dates))
    prices["TLT"] = 100 - spy_trend * 0.5 + 0.1 * (pd.Series(range(300), index=dates))
    prices["LQD"] = 100 + lqd_shift * (pd.Series(range(300), index=dates))
    prices["DBC"] = 80 + dbc_shift * (pd.Series(range(300), index=dates))
    prices["TIP"] = 105 + tip_shift * (pd.Series(range(300), index=dates))
    prices["BIL"] = 100 + 0.01 * (pd.Series(range(300), index=dates))
    prices["SH"] = 90
    prices["TBT"] = 90
    prices["GLD"] = 120
    prices["HYG"] = 85
    prices["^VIX"] = vix_level
    return prices


def test_risk_on_detected():
    prices = _make_prices(spy_trend=0.5, vix_level=15, lqd_shift=0.3, dbc_shift=0.1, tip_shift=0.0)
    returns = compute_returns(prices)
    regime = detect_regime(prices, returns)
    assert regime.name == "risk_on"


def test_risk_off_detected():
    prices = _make_prices(spy_trend=-0.5, vix_level=30, lqd_shift=-0.2, dbc_shift=-0.1, tip_shift=0.0)
    returns = compute_returns(prices)
    regime = detect_regime(prices, returns)
    assert regime.name == "risk_off"


def test_inflation_detected():
    prices = _make_prices(spy_trend=0.2, vix_level=18, lqd_shift=-0.1, dbc_shift=0.4, tip_shift=-0.3)
    returns = compute_returns(prices)
    regime = detect_regime(prices, returns)
    assert regime.name == "inflation"
