import pandas as pd

from src.history import strategy_cumulative_returns


def test_strategy_cumulative_returns_builds_growth_series():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    tickers = ["SPY", "TLT", "BIL"]
    weights = pd.DataFrame(
        [
            {"date": dates[0], "ticker": "SPY", "strategy_rule_based_mock_weight": 0.6},
            {"date": dates[0], "ticker": "TLT", "strategy_rule_based_mock_weight": 0.3},
            {"date": dates[0], "ticker": "BIL", "strategy_rule_based_mock_weight": 0.1},
            {"date": dates[1], "ticker": "SPY", "strategy_rule_based_mock_weight": 0.5},
            {"date": dates[1], "ticker": "TLT", "strategy_rule_based_mock_weight": 0.4},
            {"date": dates[1], "ticker": "BIL", "strategy_rule_based_mock_weight": 0.1},
        ]
    )
    returns = pd.DataFrame(
        {
            "SPY": [0.01, -0.005, 0.002, 0.003, -0.001],
            "TLT": [0.002, 0.001, -0.004, 0.0, 0.002],
            "BIL": [0.0001] * 5,
        },
        index=dates,
    )
    cumulative = strategy_cumulative_returns(weights, returns, "strategy_rule_based_mock_weight")
    assert not cumulative.empty
    # The curve starts one business day after the first rebalance: weights set on
    # dates[0] are not executable until dates[1]. This assertion previously read
    # `cumulative.index.equals(dates)`, which only holds if weights earn the
    # return of the day they were decided — i.e. it encoded the look-ahead bug.
    assert cumulative.index.equals(dates[1:])
    assert abs(cumulative.iloc[0] - 1.0) < 1e-9
    assert cumulative.iloc[-1] > 0.99  # growth remains positive-ish
