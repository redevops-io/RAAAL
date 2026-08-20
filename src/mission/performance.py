"""Risk-adjusted performance from a value path — Sharpe, volatility, drawdown.

Computed from the *time-weighted* return series, never from the value path
directly, and the difference is the whole point: a plan's value rises because it
earned and because money was paid into it, and only the first is performance. A
plan that beat another by contributing more, later, has a higher final value and
no better risk-adjusted return — these describe the second thing.
`accounting.time_weighted_returns` strips the flows out (money that arrives today
was not at work today); everything here is a function of that series.

Conventions, stated because they move the numbers:

  * volatility is the daily standard deviation annualised by √252;
  * Sharpe is the annualised mean excess daily return over its daily standard
    deviation — the textbook form, arithmetic in the mean, with the risk-free
    rate the engine already declares (`config.DEFAULT_RF`, 2%/yr);
  * the maximum drawdown is the deepest peak-to-trough fall of the flow-free
    growth curve, so a contribution arriving mid-fall cannot mask it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .accounting import time_weighted_returns

TRADING_DAYS = 252
DEFAULT_RF_ANNUAL = 0.02


@dataclass(frozen=True)
class Performance:
    """What a value path earned per unit of risk it ran."""

    sharpe: float
    annual_volatility: float
    max_drawdown: float          # a fall, so <= 0
    annual_return: float         # CAGR of the flow-free growth
    total_return: float          # cumulative time-weighted return
    sessions: int

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def performance(value: pd.Series, flows: pd.Series, *,
                rf_annual: float = DEFAULT_RF_ANNUAL,
                periods: int = TRADING_DAYS) -> Performance:
    """Sharpe, volatility and drawdown for one value path and its flows.

    Degenerate paths — a plan too short to have two returns, or one that never
    invested — return zeros rather than NaN, so a caller can display them beside
    the others without special-casing.
    """
    returns = time_weighted_returns(value, flows)
    n = int(len(returns))
    if n < 2:
        return Performance(0.0, 0.0, 0.0, 0.0, 0.0, n)

    growth = (1.0 + returns).cumprod()
    daily_std = float(returns.std(ddof=1))
    ann_vol = daily_std * float(np.sqrt(periods))

    rf_daily = rf_annual / periods
    excess = returns - rf_daily
    sharpe = (float(excess.mean()) / daily_std * float(np.sqrt(periods))
              if daily_std > 1e-12 else 0.0)

    ending = float(growth.iloc[-1])
    years = n / periods
    total = ending - 1.0
    annual_return = (ending ** (1.0 / years) - 1.0
                     if years > 0 and ending > 0 else 0.0)

    peak = growth.cummax()
    max_drawdown = float((growth / peak - 1.0).min())

    return Performance(
        sharpe=sharpe,
        annual_volatility=ann_vol,
        max_drawdown=max_drawdown,
        annual_return=annual_return,
        total_return=total,
        sessions=n)


def from_path(path, **kwargs) -> Performance:
    """Convenience: performance from a `PortfolioPath` (value and flows)."""
    return performance(path.value, path.flows, **kwargs)
