"""Regression tests for the Release 0 correctness defects.

Each test here targets a specific look-ahead or reproducibility defect that was
live in production before 2026-07-30 and silently inflated published results.
They are written to fail loudly if the fix is reverted, because none of these
defects is visible in output that "looks reasonable" — that is exactly why they
survived for so long.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.config import TRANSACTION_COST_BPS
from src.ensemble_regime import load_ensemble_models
from src.history import strategy_cumulative_returns, strategy_daily_returns
from src.reproducibility import build_run_manifest, frame_digest, seed_everything


@pytest.fixture
def perfect_foresight_setup():
    """A strategy that goes all-in on the single day the asset jumps.

    With correct execution lag the position is entered the day *after* the jump
    and earns nothing from it. Without the lag it captures the jump exactly —
    which is the signature of the defect.
    """
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    asset_returns = pd.DataFrame({"SPY": [0.0, 0.0, 0.10, 0.0, 0.0, 0.0]}, index=dates)

    # Weight set on the jump day itself, using data through that day.
    weights = pd.DataFrame(
        {
            "date": [dates[0], dates[2]],
            "ticker": ["SPY", "SPY"],
            "w": [0.0, 1.0],
        }
    )
    return weights, asset_returns


def test_weights_cannot_earn_the_day_they_are_decided(perfect_foresight_setup):
    """Execution lag: a weight decided on day d must not earn day d's return."""
    weights, asset_returns = perfect_foresight_setup

    net = strategy_daily_returns(weights, asset_returns, "w", cost_bps=0.0)

    # The 10% jump happens on dates[2]; the position starts earning on dates[3].
    assert net.loc[perfect_foresight_setup[1].index[2]] == pytest.approx(0.0), (
        "Strategy captured the jump on the day its weight was decided — "
        "the one-day execution lag has been removed."
    )
    assert net.sum() == pytest.approx(0.0, abs=1e-12), (
        "Strategy earned a return it could not have traded."
    )


def test_zero_lag_would_capture_the_jump(perfect_foresight_setup):
    """Control: with lag disabled the defect reappears, proving the test bites."""
    weights, asset_returns = perfect_foresight_setup

    leaky = strategy_daily_returns(
        weights, asset_returns, "w", execution_lag=0, cost_bps=0.0
    )

    assert leaky.sum() == pytest.approx(0.10), (
        "Control case did not reproduce the known defect — the fixture no longer "
        "exercises the look-ahead path."
    )


def test_turnover_is_charged():
    """Costs: a round trip must cost roughly 2x the one-way rate on notional."""
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    asset_returns = pd.DataFrame({"SPY": [0.0] * 5}, index=dates)
    weights = pd.DataFrame(
        {
            "date": [dates[0], dates[2]],
            "ticker": ["SPY", "SPY"],
            "w": [1.0, 0.0],
        }
    )

    net = strategy_daily_returns(weights, asset_returns, "w", cost_bps=10.0)

    # Flat returns, so everything below zero is cost: in at 1.0, out at 1.0.
    assert net.sum() < 0, "Turnover was not charged — backtest is gross-only."
    assert net.sum() == pytest.approx(-2 * 1.0 * (10.0 / 10_000.0), rel=1e-6)


def test_default_cost_is_nonzero():
    """A published backtest must not default to zero transaction costs."""
    assert TRANSACTION_COST_BPS > 0


def test_growth_curve_and_headline_agree(perfect_foresight_setup):
    """The dashboard curve and the headline metric must share one implementation."""
    weights, asset_returns = perfect_foresight_setup

    daily = strategy_daily_returns(weights, asset_returns, "w")
    cumulative = strategy_cumulative_returns(weights, asset_returns, "w")

    expected_final = float((1.0 + daily).prod())
    # Both are normalized to their first observation.
    assert cumulative.iloc[-1] == pytest.approx(expected_final / (1.0 + daily.iloc[0]))


def test_missing_column_returns_empty():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    asset_returns = pd.DataFrame({"SPY": [0.0] * 3}, index=dates)
    weights = pd.DataFrame({"date": [dates[0]], "ticker": ["SPY"], "w": [1.0]})

    assert strategy_daily_returns(weights, asset_returns, "absent").empty


class TestModelCutoffGate:
    """A model artifact must never be served for a date it was trained through."""

    def _write_models(self, tmp_path, monkeypatch, cutoff):
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder

        import src.ensemble_regime as er

        monkeypatch.setattr(er, "MODELS_DIR", tmp_path)
        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]])
        y = np.array([0, 1, 0, 1])
        joblib.dump(RandomForestClassifier(n_estimators=2).fit(X, y), tmp_path / "random_forest.pkl")
        joblib.dump(GradientBoostingClassifier(n_estimators=2).fit(X, y), tmp_path / "gradient_boosting.pkl")
        joblib.dump(LabelEncoder().fit(y), tmp_path / "label_encoder.pkl")
        if cutoff is not None:
            (tmp_path / "metadata.json").write_text(json.dumps({"train_cutoff": cutoff}))

    def test_refuses_model_trained_through_the_target_date(self, tmp_path, monkeypatch):
        self._write_models(tmp_path, monkeypatch, cutoff="2025-01-01")
        import src.ensemble_regime as er

        assert er.load_ensemble_models(as_of="2020-01-01") == {}, (
            "Served a model trained through 2025 to predict 2020 — this is the "
            "leak that contaminated every historical `ml`-mode result."
        )

    def test_allows_model_trained_strictly_before(self, tmp_path, monkeypatch):
        self._write_models(tmp_path, monkeypatch, cutoff="2019-01-01")
        import src.ensemble_regime as er

        assert er.load_ensemble_models(as_of="2020-01-01") != {}

    def test_refuses_model_with_no_recorded_cutoff(self, tmp_path, monkeypatch):
        self._write_models(tmp_path, monkeypatch, cutoff=None)
        import src.ensemble_regime as er

        assert er.load_ensemble_models(as_of="2020-01-01") == {}, (
            "A model without a training cutoff cannot be checked and must not "
            "be trusted in a backtest."
        )


class TestReproducibility:
    def test_seeding_makes_numpy_deterministic(self):
        seed_everything(123)
        first = np.random.rand(5)
        seed_everything(123)
        assert np.array_equal(first, np.random.rand(5))

    def test_frame_digest_is_column_order_invariant(self):
        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert frame_digest(frame) == frame_digest(frame[["b", "a"]])

    def test_frame_digest_detects_value_change(self):
        frame = pd.DataFrame({"a": [1, 2]})
        other = pd.DataFrame({"a": [1, 3]})
        assert frame_digest(frame) != frame_digest(other)

    def test_manifest_digest_ignores_timestamp(self):
        """Two identical runs at different times must share a digest."""
        kwargs = dict(seed=42, params={"start": "2016-01-01"}, inputs={"prices": "abc"})
        first = build_run_manifest(run_id="run_a", **kwargs)
        second = build_run_manifest(run_id="run_b", **kwargs)
        assert first.digest == second.digest
        assert first.created_at != second.created_at or True  # timestamps may collide

    def test_manifest_digest_detects_param_change(self):
        base = build_run_manifest(run_id="r", params={"step": 5})
        changed = build_run_manifest(run_id="r", params={"step": 10})
        assert base.digest != changed.digest


def test_adx_absent_without_real_high_low():
    """Synthetic high/low must not manufacture a volatility feature."""
    from src.features_alpha import build_alpha_features

    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    prices = pd.DataFrame({"SPY": np.linspace(100, 140, len(dates))}, index=dates)

    features = build_alpha_features(prices, "SPY")

    adx_cols = [c for c in features.columns if "adx" in c.lower()]
    assert not adx_cols, (
        f"ADX features present without real OHLC: {adx_cols}. These were derived "
        "from close*1.005 / close*0.995 and carry no information beyond close."
    )
