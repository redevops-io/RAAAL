"""Performance metrics, checked against series whose answer is known by hand.

`mission.performance` turns a value path into Sharpe, volatility and drawdown.
The risk of a statistics helper is that it looks plausible and is wrong by a
factor of √252 or a sign; these pin the shape against inputs constructed so the
right answer is arithmetic, not opinion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.mission.performance import performance


def _flat_flows(index):
    return pd.Series(0.0, index=index)


def test_a_short_path_is_zeros_not_nan():
    idx = pd.date_range("2020-01-01", periods=1, freq="B")
    value = pd.Series([100.0], index=idx)
    p = performance(value, _flat_flows(idx))
    assert (p.sharpe, p.annual_volatility, p.max_drawdown) == (0.0, 0.0, 0.0)


def test_constant_growth_has_near_zero_volatility_and_no_drawdown():
    """A path that compounds at a fixed daily rate never falls, so its drawdown
    is zero, its volatility is (almost) zero, and its annualised return is that
    rate compounded — no flows, so time-weighting changes nothing."""
    idx = pd.date_range("2020-01-01", periods=252 * 2, freq="B")
    daily = 0.0004
    value = pd.Series(100.0 * (1 + daily) ** np.arange(len(idx)), index=idx)
    p = performance(value, _flat_flows(idx))

    assert p.annual_volatility < 1e-6
    assert p.max_drawdown == 0.0
    expected_cagr = (1 + daily) ** 252 - 1
    assert abs(p.annual_return - expected_cagr) < 1e-3


def test_max_drawdown_is_the_deepest_peak_to_trough_fall():
    """Halve the value, then recover: the deepest fall is 50%, whatever the
    recovery does afterward."""
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    value = pd.Series([100, 100, 50, 60, 90, 120], index=idx, dtype=float)
    p = performance(value, _flat_flows(idx))
    assert abs(p.max_drawdown - (-0.5)) < 1e-9


def test_volatility_annualises_by_root_252():
    """Two returns of +/-1% around a flat mean have a daily std of 1%; the
    annual figure is that times √252."""
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    value = pd.Series([100.0, 101.0, 99.99], index=idx)
    p = performance(value, _flat_flows(idx))
    returns = value.pct_change().dropna()
    expected = float(returns.std(ddof=1) * np.sqrt(252))
    assert abs(p.annual_volatility - expected) < 1e-9


def test_a_contribution_does_not_count_as_a_gain():
    """The reason it reads time-weighted returns: a dollar paid in raises the
    value without being performance. A flat-priced holding with a contribution
    mid-way must show ~zero return and ~zero volatility, not a jump."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    # value jumps from 100 to 200 only because 100 was contributed on day 3.
    value = pd.Series([100, 100, 200, 200, 200], index=idx, dtype=float)
    flows = pd.Series([100, 0, 100, 0, 0], index=idx, dtype=float)
    p = performance(value, flows)
    assert abs(p.total_return) < 1e-9
    assert p.annual_volatility < 1e-9
    assert p.max_drawdown == 0.0
