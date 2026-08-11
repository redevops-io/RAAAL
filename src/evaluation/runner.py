"""Evaluate a methodology under an evaluation protocol.

The single place where the two artifacts meet::

    methodology + evaluation protocol -> performance

Everything that shapes the number comes from one of the two, and both are hashed
into the result. Nothing is read from module constants.

The seal is enforced here rather than trusted: when a protocol declares a sealed
holdout, the price panel is truncated *before* the methodology executes, so a
sealed period is not merely unreported — it is unreachable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..features import compute_returns
from ..history import _annualize, strategy_daily_returns
from ..methodology.executor import backtest
from ..methodology.verify import unrealized_declarations
from ..calendars import CalendarRegistry, TradingCalendar
from ..methodology.spec import Methodology
from .protocol import EvaluationProtocol


class SealViolation(RuntimeError):
    """Raised when an evaluation would read a sealed holdout."""


class IncompatiblePairing(ValueError):
    """Raised when a protocol cannot properly evaluate a methodology.

    Not every (methodology, protocol) pair is valid. A protocol whose warmup is
    shorter than the methodology's estimation window forces the methodology onto
    its fallback path for the early evaluation dates — producing a result that is
    labelled as the methodology but is substantially something else.
    """


@dataclass(frozen=True)
class CompatibilityResult:
    """A pre-execution artifact: can this protocol validly evaluate this methodology?

    Stored rather than merely raised. A refused pairing is evidence that the
    configuration was *attempted*, which matters for trial accounting — a searcher
    who tries twenty pairings and reports the two that ran has still searched
    twenty times. Discarding the refusals would hide that.
    """

    methodology_version_id: str
    methodology_hash: str
    protocol_id: str
    protocol_hash: str
    compatible: bool
    blockers: list[Dict[str, Any]]

    def to_json(self) -> Dict[str, Any]:
        return {
            "methodology": self.methodology_version_id,
            "methodology_hash": self.methodology_hash,
            "protocol": self.protocol_id,
            "protocol_hash": self.protocol_hash,
            "compatible": self.compatible,
            "blockers": list(self.blockers),
        }


def assess_compatibility(
    methodology: Methodology, protocol: EvaluationProtocol
) -> CompatibilityResult:
    """Evaluate the pairing and return the verdict without raising."""
    blockers: list[Dict[str, Any]] = []

    lookback_param = methodology.params.get("lookback")
    if lookback_param is not None:
        lookback = int(lookback_param.value)
        if protocol.walk_forward.warmup < lookback:
            blockers.append(
                {
                    "code": "INSUFFICIENT_WARMUP",
                    "required": lookback,
                    "provided": protocol.walk_forward.warmup,
                    "detail": (
                        "early evaluation dates would fall back to a different "
                        "allocation rule while still being reported as this methodology"
                    ),
                }
            )

    if methodology.contract.requires_cost_model and protocol.transaction_costs.bps <= 0:
        blockers.append(
            {
                "code": "MISSING_COST_MODEL",
                "required": "> 0 bps",
                "provided": protocol.transaction_costs.bps,
                "detail": "the methodology's contract requires a cost model",
            }
        )

    if protocol.holdout.defined and protocol.holdout.sealed:
        boundary = pd.Timestamp(protocol.holdout.start)
        declared_start = pd.Timestamp(protocol.data_snapshot.start)
        if boundary <= declared_start:
            blockers.append(
                {
                    "code": "SEAL_COVERS_EVERYTHING",
                    "required": f"holdout start after {protocol.data_snapshot.start}",
                    "provided": protocol.holdout.start,
                    "detail": "no evaluable history remains outside the sealed window",
                }
            )

    return CompatibilityResult(
        methodology_version_id=methodology.version_id,
        methodology_hash=methodology.content_hash,
        protocol_id=protocol.protocol_id,
        protocol_hash=protocol.content_hash,
        compatible=not blockers,
        blockers=blockers,
    )


def check_compatibility(methodology: Methodology, protocol: EvaluationProtocol) -> None:
    """Refuse pairings that would silently mislabel the result."""
    result = assess_compatibility(methodology, protocol)
    if not result.compatible:
        detail = "; ".join(
            f"{b['code']} (required {b['required']}, provided {b['provided']})"
            for b in result.blockers
        )
        raise IncompatiblePairing(
            f"{methodology.version_id} × {protocol.protocol_id}: {detail}"
        )


#: A single asset above this mean weight makes the result a statement about that
#: asset, not about the methodology.
CONCENTRATION_FLAG = 0.50

#: Annualized volatility below this is a cash-equivalent portfolio. Ratios
#: computed on it are arithmetic artifacts, not risk-adjusted performance.
DEGENERATE_VOL = 0.01

#: Sharpe above this over a multi-asset ETF universe is implausible and almost
#: always indicates a degenerate denominator rather than skill.
IMPLAUSIBLE_SHARPE = 3.0

#: Share of rebalances allocated by a fallback rule beyond which the record is no
#: longer a record of the primary methodology.
FALLBACK_SHARE_LIMIT = 0.10


@dataclass(frozen=True)
class EvaluationResult:
    """A figure, the two artifacts that produced it, and its diagnostics."""

    methodology_version_id: str
    methodology_hash: str
    protocol_id: str
    protocol_hash: str
    annualized_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    n_rebalances: int
    n_observations: int
    period_start: str
    period_end: str
    sealed_period_excluded: bool
    weights: pd.DataFrame
    daily_returns: pd.Series
    diagnostics: Dict[str, Any] = None  # type: ignore[assignment]

    result_status: Dict[str, Any] = None  # type: ignore[assignment]
    execution_audit: Dict[str, Any] = None  # type: ignore[assignment]

    @property
    def flags(self) -> list[str]:
        return list((self.diagnostics or {}).get("flags", []))

    @property
    def publishable(self) -> bool:
        """Deprecated shorthand: economic validity only.

        Publication is a surface-specific decision made by `policy.publication`,
        which needs the statistical assessment and the policy evaluation as well.
        This property answers a narrower question — did the evaluation produce an
        economically coherent portfolio — and is kept only for the CLI's
        pre-statistics gate.
        """
        return bool((self.result_status or {}).get("economic_valid", False))

    def to_json(self) -> Dict[str, Any]:
        return {
            "methodology_version_id": self.methodology_version_id,
            "methodology_hash": self.methodology_hash,
            "protocol_id": self.protocol_id,
            "protocol_hash": self.protocol_hash,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            # Which semantics produced that number. A stored result with no
            # version cannot be told apart from one measured under `@1`, and
            # the two differ materially on any series that fell early.
            "drawdown_semantics": DRAWDOWN_SEMANTICS,
            "n_rebalances": self.n_rebalances,
            "n_observations": self.n_observations,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "sealed_period_excluded": self.sealed_period_excluded,
            "diagnostics": self.diagnostics or {},
            "result_status": self.result_status or {},
            "execution_audit": self.execution_audit or {},
        }


#: The semantics this build computes drawdown under.
#:
#: `@1` built the equity curve as `(1 + returns).cumprod()`, which starts at
#: the first *return*, so the opening level was never in it and `cummax` began
#: at the post-first-move value. Any fall from the opening level was measured
#: against a peak that had already absorbed it, and a portfolio that halved on
#: its first session reported a maximum drawdown of zero.
#:
#: `@2` includes the opening level. Versioned rather than corrected in place
#: because every stored result carrying a drawdown was produced under `@1`, and
#: recomputing them silently would present old evidence as though it had always
#: been measured this way.
DRAWDOWN_SEMANTICS = "drawdown@2"


def _max_drawdown(daily: pd.Series) -> float:
    """The largest fall below the running high-water mark, as a negative
    fraction.

    The curve starts at 1.0 — the opening level — so a fall in the first
    session is measured from where the portfolio began. `docs/Drawdown.md` is
    the definition and `tests/test_drawdown_conformance.py` checks this against
    it.
    """
    curve = pd.concat([pd.Series([1.0]), (1.0 + daily).cumprod()],
                      ignore_index=True)
    return float((curve / curve.cummax() - 1.0).min())


def _diagnose(
    weights: pd.DataFrame, daily: pd.Series, annualized: float, vol: float, sharpe: float
) -> Dict[str, Any]:
    """Flag results that are arithmetically valid but economically degenerate.

    These checks exist because the first real evaluation run produced a Sharpe of
    6.59 from a portfolio that was 99.6% cash — a number that would have looked
    like a headline result and was in fact a statement about T-bills.
    """
    flags: list[str] = []
    pivot = weights.pivot(index="date", columns="ticker", values="weight").fillna(0.0)
    mean_weights = pivot.mean()
    top_asset = mean_weights.idxmax()
    top_share = float(mean_weights.max())

    if top_share > CONCENTRATION_FLAG:
        flags.append(
            f"concentration: {top_asset} holds {top_share:.1%} of the portfolio on "
            f"average — this result describes {top_asset}, not the methodology"
        )

    if vol < DEGENERATE_VOL:
        flags.append(
            f"degenerate volatility: {vol:.4%} annualized is cash-equivalent; "
            "risk-adjusted ratios computed on it are arithmetic artifacts"
        )

    if sharpe > IMPLAUSIBLE_SHARPE and vol < DEGENERATE_VOL * 5:
        flags.append(
            f"implausible Sharpe {sharpe:.2f} paired with {vol:.4%} volatility — "
            "check the denominator before treating this as performance"
        )

    effective_n = float(1.0 / (mean_weights**2).sum()) if (mean_weights**2).sum() > 0 else 0.0
    if effective_n < 2.0:
        flags.append(
            f"effective breadth {effective_n:.2f} assets — the allocation is not "
            "diversified regardless of universe size"
        )

    return {
        "flags": flags,
        "top_asset": str(top_asset),
        "top_asset_mean_weight": top_share,
        "effective_n_assets": effective_n,
        "mean_weights": {k: float(v) for k, v in mean_weights.sort_values(ascending=False).items()},
    }


def resolve_calendar(protocol: EvaluationProtocol) -> TradingCalendar:
    """Resolve the protocol's calendar reference to the versioned artifact."""
    return CalendarRegistry().resolve(protocol.walk_forward.calendar)


def periods_per_year(protocol: EvaluationProtocol) -> int:
    """Sessions per year, from the calendar unless the protocol overrides it."""
    if protocol.walk_forward.periods_per_year is not None:
        return protocol.walk_forward.periods_per_year
    return resolve_calendar(protocol).periods_per_year


def apply_calendar(prices: pd.DataFrame, protocol: EvaluationProtocol) -> pd.DataFrame:
    """Restrict the panel to the protocol's declared trading sessions.

    A price panel joined across instruments with different trading calendars
    inherits the union. Forward-filling a five-session instrument onto a seven-day
    index manufactures zero-return observations that are not market data — they
    deflate realized volatility and inflate every ratio computed from it.

    The calendar is a referenced artifact, so the sessions used are identified
    rather than implied, and a calendar applied outside the range it declares
    raises rather than extrapolating.
    """
    return resolve_calendar(protocol).filter(prices)


def apply_seal(prices: pd.DataFrame, protocol: EvaluationProtocol) -> tuple[pd.DataFrame, bool]:
    """Truncate the panel at a sealed holdout boundary.

    Returns the usable panel and whether anything was withheld. Truncation
    happens before execution so the methodology cannot see the sealed window even
    if its pipeline asked for it.
    """
    holdout = protocol.holdout
    if not (holdout.defined and holdout.sealed):
        return prices, False

    boundary = pd.Timestamp(holdout.start)
    usable = prices.loc[prices.index < boundary]
    if usable.empty:
        raise SealViolation(
            f"{protocol.protocol_id} seals everything from {holdout.start}, "
            "leaving no evaluable history"
        )
    return usable, True


def evaluate(
    methodology: Methodology,
    protocol: EvaluationProtocol,
    prices: pd.DataFrame,
    *,
    bind_snapshot: bool = True,
) -> tuple[EvaluationResult, EvaluationProtocol]:
    """Run one methodology under one protocol.

    Returns the result and the protocol actually used — which differs from the
    one passed in when `bind_snapshot` pins the data hash, because a protocol
    bound to a different snapshot is a different protocol.
    """
    from ..reproducibility import frame_digest

    check_compatibility(methodology, protocol)

    window = prices.loc[
        pd.Timestamp(protocol.data_snapshot.start) : pd.Timestamp(protocol.data_snapshot.end)
    ]
    if window.empty:
        raise ValueError(
            f"{protocol.protocol_id} declares {protocol.data_snapshot.start}.."
            f"{protocol.data_snapshot.end}, which selects no rows from the panel"
        )

    # Apply the declared trading calendar before anything else.
    #
    # The joined price panel carries a 7-day index because BTC-USD trades at
    # weekends, so equity prices are forward-filled across Saturday and Sunday.
    # That injected ~31% zero-return observations into every ETF series,
    # deflating volatility and breaking the 252-period annualization. The
    # calendar was previously ambient; it is now declared protocol data, and
    # a protocol that wants a 7-day index must say so.
    window = apply_calendar(window, protocol)

    usable, sealed_excluded = apply_seal(window, protocol)

    effective = protocol
    if bind_snapshot:
        effective = protocol.with_snapshot_hash(frame_digest(usable))

    weights = backtest(
        methodology, usable, warmup=protocol.walk_forward.warmup
    )

    returns = compute_returns(usable)
    daily = strategy_daily_returns(
        weights,
        returns,
        "weight",
        execution_lag=protocol.transaction_costs.execution_lag_days,
        cost_bps=protocol.transaction_costs.bps,
    )
    if daily.empty:
        raise ValueError("evaluation produced no return observations")

    # Annualization comes from the protocol's declared calendar, not a constant.
    # A 252 assumption applied to a 365-observation year understates the
    # annualized figure and mis-scales volatility.
    periods = periods_per_year(protocol)
    annualized = _annualize(daily, periods_per_year=periods)
    vol = float(daily.std() * np.sqrt(periods))
    sharpe = float(annualized / vol) if vol > 0 else float("nan")
    diagnostics = _diagnose(weights, daily, annualized, vol, sharpe)
    audit = dict(weights.attrs.get("execution_audit", {}))

    # Fallback usage is a validity question, not a footnote: a record largely
    # produced by a degradation rule is not a record of this methodology.
    fallback_share = float(audit.get("fallback_share", 0.0))
    if fallback_share > FALLBACK_SHARE_LIMIT:
        diagnostics.setdefault("flags", []).append(
            f"fallback usage {fallback_share:.1%} of rebalances exceeds "
            f"{FALLBACK_SHARE_LIMIT:.0%} — this record is substantially the "
            f"fallback rule, not {methodology.version_id}"
        )

    # Deliberately no `statistical_valid` and no `publication_eligible` here.
    # Whether the evidence meets a standard is a versioned-policy decision, and
    # whether it may be shown is a surface-specific publication decision. The
    # runner knows neither, and asserting either from here would collapse three
    # questions that need separate answers.
    unrealized = [r.to_json() for r in unrealized_declarations(methodology)]

    result_status = {
        "computation_valid": True,
        "contract_valid": True,
        "declarations_realized": not unrealized,
        "unrealized_declarations": unrealized,
        "economic_valid": not bool(diagnostics.get("flags")),
        "statistical_assessment_complete": False,   # set once statistics run
        "reproducible": True,
        "economically_degenerate": bool(diagnostics.get("flags")),
        "fallback_share": fallback_share,
        "precedence_overrides": int(audit.get("precedence_override_count", 0)),
        "flags": list(diagnostics.get("flags", [])),
    }

    result = EvaluationResult(
        methodology_version_id=methodology.version_id,
        methodology_hash=methodology.content_hash,
        protocol_id=effective.protocol_id,
        protocol_hash=effective.content_hash,
        annualized_return=float(annualized),
        volatility=vol,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(daily),
        n_rebalances=int(weights["date"].nunique()),
        n_observations=int(len(daily)),
        period_start=str(daily.index[0].date()),
        period_end=str(daily.index[-1].date()),
        sealed_period_excluded=sealed_excluded,
        weights=weights,
        daily_returns=daily,
        diagnostics=diagnostics,
        result_status=result_status,
        execution_audit=audit,
    )
    return result, effective
