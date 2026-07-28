"""End-to-end orchestration for the regime-adjusted allocation MVP."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from .config import AUX_SERIES, TICKER_INDEX, UNIVERSE
from .data_loader import download_prices
from .features import compute_returns, exponential_cov, exponential_mean
from .optimizer import optimize_weights
from .portfolio_utils import (
    build_rationales,
    portfolio_metrics,
    rf_from_sgov,
    weights_array,
)
from .regime import RegimeResult, detect_regime

logger = logging.getLogger(__name__)

STATE_PATH = Path("reports/state.json")


@dataclass
class AllocationResult:
    weights: Dict[str, float]
    expected_returns: pd.Series
    covariance: pd.DataFrame
    rf_rate: float
    regime: RegimeResult
    metrics: Dict[str, float]
    rationales: Dict[str, str]
    rebalance_triggered: bool


def _load_state() -> Dict[str, float] | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return None


def _save_state(state: Dict[str, float]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run_allocation(
    start: datetime,
    end: datetime,
    force_refresh: bool = False,
    use_forecaster: bool = False,
    forecaster_backend: str = "lightgbm",
    use_sentiment: bool = False,
) -> AllocationResult:
    """Run the full allocation pipeline.

    Parameters
    ----------
    use_forecaster : bool
        If True, use ML forecaster (LightGBM/LSTM/Transformer) for μ
        instead of naive exponential mean.
    forecaster_backend : str
        Which forecaster backend: "lightgbm", "lstm", "transformer".
    use_sentiment : bool
        If True, run the NLP sentiment engine and inject scores
        into the FOMO/FOBI indicator.
    """
    tickers = [asset.ticker for asset in UNIVERSE] + AUX_SERIES
    prices = download_prices(
        tickers, start=start, end=end,
        force_refresh=force_refresh,
    )
    returns = compute_returns(prices)

    regime = detect_regime(prices, returns)
    base_tickers = [asset.ticker for asset in UNIVERSE]
    base_returns = returns[base_tickers]
    cov = exponential_cov(base_returns)
    rf_rate = rf_from_sgov(prices)

    # --- Expected returns (μ) ---
    if use_forecaster:
        try:
            from .forecaster import ReturnForecaster

            fc = ReturnForecaster(backend=forecaster_backend)
            fallback = exponential_mean(base_returns)
            result = fc.rolling_retrain(prices, returns)
            mu = result.mu
            if result.drift_report and result.drift_report.is_drifted:
                logger.warning(
                    "Forecaster drift detected (z=%.2f) — "
                    "blending with exponential mean",
                    result.drift_report.value,
                )
                mu = 0.7 * mu + 0.3 * fallback
        except Exception as exc:
            logger.warning(
                "Forecaster unavailable (%s) — "
                "falling back to exponential mean",
                exc,
            )
            mu = exponential_mean(base_returns)
    else:
        mu = exponential_mean(base_returns)

    # --- Sentiment (optional) ---
    nlp_components = None
    if use_sentiment:
        try:
            from .sentiment import SentimentEngine

            engine = SentimentEngine(scorer="auto")
            nlp_components = engine.as_fomo_components(
                tickers=base_tickers[:5],
            )
            logger.info(
                "Sentiment engine returned %d components",
                len(nlp_components),
            )
        except Exception as exc:
            logger.warning(
                "Sentiment engine unavailable (%s)", exc,
            )

    prev_state = _load_state()
    prev_weights = None
    last_regime = None
    if prev_state:
        prev_weights = weights_array(
            prev_state.get("weights", {}),
        )
        last_regime = prev_state.get("regime")

    weights = optimize_weights(
        base_returns, regime.name,
        rf_rate=rf_rate, prev_weights=prev_weights,
    )
    metrics = portfolio_metrics(weights, mu, cov, rf_rate)
    rationales = build_rationales(
        weights, mu, regime.name, regime.diagnostics,
    )

    rebalance_triggered = (
        last_regime != regime.name if last_regime else True
    )

    _save_state(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "regime": regime.name,
            "weights": weights,
        }
    )

    return AllocationResult(
        weights=weights,
        expected_returns=mu,
        covariance=cov,
        rf_rate=rf_rate,
        regime=regime,
        metrics=metrics,
        rationales=rationales,
        rebalance_triggered=rebalance_triggered,
    )


# ===========================================================================
# Objective-compare: ONE evidence snapshot -> three objective plans via SELECTION
# over the approved strategy registry (Amendments §2/§4). No new optimizers.
# ===========================================================================
from .agentic.selection import (  # noqa: E402  (kept local to avoid import cost on legacy path)
    SelectionResult,
    Snapshot,
    build_snapshot,
    select_all_objectives,
)


@dataclass
class ObjectiveCompareResult:
    """The three objective plans + a current/no-action baseline, all tied to ONE snapshot."""

    snapshot_id: str
    regime: RegimeResult
    mu: pd.Series
    covariance: pd.DataFrame
    rf_rate: float
    plans: Dict[str, SelectionResult]           # objective -> ObjectivePlan
    current: Dict[str, object]                   # no-action baseline {weights, metrics, label}
    rebalance_triggered: bool
    generated_at: str
    as_of: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "as_of": self.as_of,
            "generated_at": self.generated_at,
            "regime": self.regime.name,
            "regime_diagnostics": self.regime.diagnostics,
            "rf_rate": self.rf_rate,
            "rebalance_triggered": self.rebalance_triggered,
            "current": self.current,
            "plans": {obj: plan.as_dict() for obj, plan in self.plans.items()},
        }


def _cash_baseline(mu: pd.Series, cov: pd.DataFrame, rf: float) -> Dict[str, object]:
    from .portfolio_utils import portfolio_metrics

    weights = {a.ticker: 0.0 for a in UNIVERSE}
    weights["BIL"] = 1.0
    return {"label": "Current / no action (all cash)", "weights": weights,
            "metrics": portfolio_metrics(weights, mu, cov, rf)}


def compare_from_snapshot(
    snap: Snapshot,
    current: Optional[Dict[str, object]] = None,
    regime: Optional[RegimeResult] = None,
    benchmark_stats: Optional[Dict] = None,
    last_regime: Optional[str] = None,
    generated_at: Optional[str] = None,
) -> ObjectiveCompareResult:
    """Build the three objective plans from a prepared snapshot (network-free; unit-testable)."""
    plans = select_all_objectives(snap, benchmark_stats)
    base_current = current or _cash_baseline(snap.mu, snap.cov, snap.rf)
    reg = regime or RegimeResult(name=snap.regime, matches={}, diagnostics={})

    # rebalance triggered if the regime turned, or any objective's target drifts past the turnover cap
    triggered = bool(last_regime is not None and last_regime != snap.regime)
    cur_w = base_current.get("weights", {})  # type: ignore[assignment]
    for plan in plans.values():
        drift = sum(abs(plan.weights.get(t, 0.0) - float(cur_w.get(t, 0.0)))
                    for t in plan.weights) / 2.0
        if drift > 0.10:
            triggered = True
    return ObjectiveCompareResult(
        snapshot_id=snap.snapshot_id,
        regime=reg,
        mu=snap.mu,
        covariance=snap.cov,
        rf_rate=snap.rf,
        plans=plans,
        current=base_current,
        rebalance_triggered=triggered,
        generated_at=generated_at or datetime.utcnow().isoformat(),
        as_of=snap.as_of,
    )


def run_objective_compare(
    start: datetime,
    end: datetime,
    force_refresh: bool = False,
    use_sentiment: bool = False,
    current: Optional[Dict[str, object]] = None,
    benchmark_stats: Optional[Dict] = None,
    generated_at: Optional[str] = None,
) -> ObjectiveCompareResult:
    """Assemble ONE market/portfolio evidence snapshot, then produce three objective plans.

    Reuses the same deterministic evidence path as `run_allocation` (download -> returns ->
    regime -> exponential mu/cov -> rf). The three plans SELECT among registered strategies.
    """
    tickers = [asset.ticker for asset in UNIVERSE] + AUX_SERIES
    prices = download_prices(tickers, start=start, end=end, force_refresh=force_refresh)
    returns = compute_returns(prices)
    base_tickers = [asset.ticker for asset in UNIVERSE]
    base_returns = returns[base_tickers]
    regime = detect_regime(prices, returns)
    mu = exponential_mean(base_returns)
    cov = exponential_cov(base_returns)
    rf_rate = rf_from_sgov(prices)

    context: Dict[str, object] = {}
    if use_sentiment:
        try:
            from .sentiment import SentimentEngine  # noqa: F401 (optional heavy import)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sentiment engine unavailable (%s)", exc)

    prev_state = _load_state()
    last_regime = prev_state.get("regime") if prev_state else None

    snap = build_snapshot(prices, base_returns.dropna(), regime.name, mu, cov, rf_rate, context)
    return compare_from_snapshot(snap, current=current, regime=regime,
                                 benchmark_stats=benchmark_stats, last_regime=last_regime,
                                 generated_at=generated_at)
