"""Execution semantics for Methodology Specification 0.1.

This is what makes the AST *executable* data rather than a description sitting
alongside code that happens to agree with it. Every value the computation uses
comes from the methodology — lookback, linkage, bounds, rebalance cadence,
turnover cap — so two versions with different parameters necessarily produce
different results, and a published figure is bound to its methodology by
construction instead of by assertion.

The pipeline is interpreted, not compiled: each named step in `pipeline` maps to
a handler, and an unknown step is an error rather than a silent no-op. That
matters because a typo in a YAML pipeline would otherwise quietly drop a stage
and still produce plausible weights.

Causality is inherited from the engine, not reimplemented: this module produces
*weights as of a date*, and `history.strategy_daily_returns` applies the
execution lag and transaction costs. Keeping that in one place is why the
headline metric and the dashboard curve cannot disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ..hrp import _get_quasi_diag, _get_rec_bipart
from .spec import Methodology


class ExecutionError(RuntimeError):
    """Raised when a methodology cannot be executed as specified."""


@dataclass
class ExecutionContext:
    """State threaded through the pipeline.

    Steps mutate `weights` and read `returns`; anything a later step needs from
    an earlier one is passed here rather than recomputed, so the ordering
    declared in the AST is the ordering that actually runs.
    """

    methodology: Methodology
    returns: pd.DataFrame
    as_of: pd.Timestamp
    prev_weights: Optional[Mapping[str, float]] = None
    corr: Optional[pd.DataFrame] = None
    cov: Optional[pd.DataFrame] = None
    sort_order: Optional[List[str]] = None
    signal: Optional[pd.Series] = None
    ranks: Optional[pd.Series] = None
    selected: Optional[List[str]] = None
    weights: Optional[pd.Series] = None
    notes: List[str] = None  # type: ignore[assignment]
    precedence_overrides: List[Dict[str, Any]] = None  # type: ignore[assignment]
    requested_turnover_cap: Optional[float] = None
    realized_turnover: Optional[float] = None
    fallback_used: Optional[str] = None

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []
        if self.precedence_overrides is None:
            self.precedence_overrides = []

    def param(self, name: str, default=None):
        p = self.methodology.params.get(name)
        return default if p is None else p.value


# --- pipeline steps --------------------------------------------------------


def _estimate_correlation(ctx: ExecutionContext) -> None:
    """Correlation and covariance over the methodology's own lookback window.

    The estimator is a declared parameter, not an implementation detail. Two
    faithful implementations of "HRP with a 252-day lookback" that differ only in
    whether the covariance is equal- or exponentially-weighted produce different
    allocations and different published returns — the dispersion Menkveld et al.
    (2024) call non-standard error. Requiring the choice in the AST is how the
    spec removes that degree of freedom.
    """
    lookback = int(ctx.param("lookback", 252))
    window = ctx.returns.tail(lookback)

    if len(window) < lookback:
        raise ExecutionError(
            f"{ctx.methodology.version_id} requires a {lookback}-day lookback; "
            f"only {len(window)} observations available as of {ctx.as_of.date()}"
        )

    estimator = ctx.param("covariance_estimator", "sample")
    if estimator == "sample":
        ctx.cov = window.cov()
    elif estimator == "exponential":
        span = int(ctx.param("covariance_span", 63))
        ctx.cov = window.ewm(span=span).cov().groupby(level=-1).last()
        ctx.cov = ctx.cov.reindex(index=window.columns, columns=window.columns)
    else:
        raise ExecutionError(f"unsupported covariance_estimator {estimator!r}")

    ctx.corr = window.corr()

    if ctx.corr.isnull().any().any() or np.isinf(ctx.corr.values).any():
        raise ExecutionError("correlation matrix contains NaN or inf")
    if ctx.cov.isnull().any().any():
        raise ExecutionError(f"{estimator} covariance contains NaN")


def _cluster_assets(ctx: ExecutionContext) -> None:
    """Hierarchical clustering using the declared distance metric and linkage."""
    metric = ctx.param("correlation_distance", "sqrt_half_one_minus_rho")
    if metric != "sqrt_half_one_minus_rho":
        raise ExecutionError(f"unsupported correlation_distance {metric!r}")

    method = ctx.param("linkage_method", "single")
    dist = np.sqrt((1.0 - ctx.corr) / 2.0).fillna(0.0)
    ctx._link = linkage(squareform(dist.values, checks=False), method=method)  # type: ignore[attr-defined]


def _quasi_diagonalize(ctx: ExecutionContext) -> None:
    order = _get_quasi_diag(ctx._link)  # type: ignore[attr-defined]
    ctx.sort_order = [ctx.corr.index[i] for i in order]


def _recursive_bisection(ctx: ExecutionContext) -> None:
    weights = _get_rec_bipart(ctx.cov, ctx.sort_order)
    ctx.weights = weights / weights.sum()


def _apply_bounds(ctx: ExecutionContext) -> None:
    """Clip to the contract's weight bounds, then renormalize.

    Renormalizing after clipping can push a weight back over the cap, so this
    iterates to a fixed point rather than clipping once and hoping.
    """
    bounds = ctx.methodology.contract.weight_bounds
    lo, hi = float(bounds.get("min", 0.0)), float(bounds.get("max", 1.0))

    n = len(ctx.weights)
    if hi * n < 1.0 - 1e-12:
        raise ExecutionError(
            f"contract bounds are infeasible: {n} assets capped at {hi} cannot sum to 1"
        )

    w = ctx.weights.clip(lower=lo, upper=hi)
    for _ in range(100):
        total = w.sum()
        if total <= 0:
            raise ExecutionError("weights sum to zero after applying bounds")
        normalized = w / total
        clipped = normalized.clip(lower=lo, upper=hi)
        if np.allclose(clipped, normalized, atol=1e-12):
            w = clipped
            break
        w = clipped

    # Renormalizing after the final clip would reintroduce the overshoot the clip
    # just removed, so the bound is treated as hard and any residual is left
    # unallocated. It is bounded by n * float epsilon and is not economically
    # meaningful, but it must not appear as a contract breach.
    ctx.weights = w.clip(lower=lo, upper=hi)


def _apply_turnover_cap(ctx: ExecutionContext) -> None:
    """Scale the trade toward the target so turnover stays under the cap.

    With no previous holding there is nothing to cap — the initial trade in from
    cash is unavoidable, and pretending otherwise would understate cost.

    **Constraint precedence.** Blending toward a previous holding can carry a
    weight back over the contract's ceiling, because the previous holding was
    capped under a different target. Contract bounds are *hard* — they are a
    promise to consumers and appear in the output contract — while the turnover
    cap is a *soft* preference expressed as a parameter. So bounds are re-applied
    after blending, and the realised turnover may exceed the cap when the two
    conflict. The alternative, honouring turnover and breaching the contract,
    would publish a portfolio that violates its own stated constraints.
    """
    cap = ctx.param("max_turnover")
    if cap is None:
        raise ExecutionError("apply_turnover_cap is in the pipeline but max_turnover is unset")
    if ctx.prev_weights is None:
        return

    policy = ctx.methodology.contract.constraint_policy
    prev = pd.Series(ctx.prev_weights).reindex(ctx.weights.index).fillna(0.0)
    turnover = float((ctx.weights - prev).abs().sum())
    ctx.requested_turnover_cap = float(cap)

    if turnover <= float(cap) or turnover == 0:
        ctx.realized_turnover = turnover
        return

    scale = float(cap) / turnover
    ctx.weights = prev + (ctx.weights - prev) * scale
    _apply_bounds(ctx)

    realised = float((ctx.weights - prev).abs().sum())
    ctx.realized_turnover = realised

    if realised > float(cap) + 1e-9:
        if not policy.soft_may_be_violated_to_satisfy_hard:
            raise ExecutionError(
                f"turnover cap {cap} and weight bounds conflict on {ctx.as_of.date()}, "
                "and constraint_policy forbids violating the soft constraint"
            )
        ctx.precedence_overrides.append(
            {
                "date": str(ctx.as_of.date()),
                "soft_constraint": "turnover_cap",
                "hard_constraint": "weight_bounds",
                "requested": float(cap),
                "realized": realised,
                "reason": (
                    "blending toward the previous holding to respect the turnover "
                    "cap would have breached the weight ceiling; the contract "
                    "declares weight_bounds hard and turnover_cap soft"
                ),
            }
        )


# --- cross-sectional momentum ---------------------------------------------
#
# A deliberately different family from HRP: it ranks and selects rather than
# clustering a covariance matrix. Added to test whether the artifact model
# describes investment research or only hierarchical allocation.


def _compute_momentum(ctx: ExecutionContext) -> None:
    """Total return over the formation window, skipping the most recent period.

    The skip is not an implementation detail. Jegadeesh & Titman skip the most
    recent month because short-horizon reversal contaminates the signal, and an
    implementation without it is measuring something else.
    """
    lookback = int(ctx.param("lookback", 252))
    skip = int(ctx.param("skip", 21))

    needed = lookback + skip
    if len(ctx.returns) < needed:
        raise ExecutionError(
            f"{ctx.methodology.version_id} requires {needed} sessions "
            f"({lookback} formation + {skip} skip); {len(ctx.returns)} available "
            f"as of {ctx.as_of.date()}"
        )

    window = ctx.returns.iloc[-(needed):-skip] if skip else ctx.returns.tail(lookback)
    ctx.signal = (1.0 + window).prod() - 1.0

    if ctx.signal.isnull().any():
        raise ExecutionError("momentum signal contains NaN")


def _rank_assets(ctx: ExecutionContext) -> None:
    if ctx.signal is None:
        raise ExecutionError("rank_assets requires a signal; is compute_momentum in the pipeline?")
    ctx.ranks = ctx.signal.rank(ascending=False, method="first")


def _select_top_n(ctx: ExecutionContext) -> None:
    """Keep the highest-ranked assets. Selection is the defining act here."""
    top_n = ctx.param("top_n")
    if top_n is None:
        raise ExecutionError("select_top_n is in the pipeline but top_n is unset")
    if ctx.ranks is None:
        raise ExecutionError("select_top_n requires ranks; is rank_assets in the pipeline?")

    n = int(top_n)
    if n < 1:
        raise ExecutionError("top_n must be at least 1")
    if n > len(ctx.ranks):
        raise ExecutionError(
            f"top_n={n} exceeds the {len(ctx.ranks)}-asset investable universe"
        )
    ctx.selected = list(ctx.ranks.nsmallest(n).index)


def _equal_weight_selected(ctx: ExecutionContext) -> None:
    if not ctx.selected:
        raise ExecutionError("no assets selected")
    weight = 1.0 / len(ctx.selected)
    ctx.weights = pd.Series(
        {t: (weight if t in set(ctx.selected) else 0.0) for t in ctx.returns.columns}
    )


PIPELINE_STEPS: Mapping[str, Callable[[ExecutionContext], None]] = {
    "compute_momentum": _compute_momentum,
    "rank_assets": _rank_assets,
    "select_top_n": _select_top_n,
    "equal_weight_selected": _equal_weight_selected,
    "estimate_correlation": _estimate_correlation,
    "cluster_assets": _cluster_assets,
    "quasi_diagonalize": _quasi_diagonalize,
    "recursive_bisection": _recursive_bisection,
    "apply_bounds": _apply_bounds,
    "apply_turnover_cap": _apply_turnover_cap,
}


# --- fallbacks -------------------------------------------------------------


def _inverse_volatility(returns: pd.DataFrame, lookback: int) -> pd.Series:
    vol = returns.tail(lookback).std()
    inv = 1.0 / vol.replace(0.0, np.nan)
    inv = inv.fillna(0.0)
    if inv.sum() <= 0:
        raise ExecutionError("inverse-volatility fallback produced no allocation")
    return inv / inv.sum()


def _equal_weight(returns: pd.DataFrame, lookback: int) -> pd.Series:
    n = returns.shape[1]
    return pd.Series(1.0 / n, index=returns.columns)


FALLBACKS: Mapping[str, Callable[[pd.DataFrame, int], pd.Series]] = {
    "inverse_volatility": _inverse_volatility,
    "equal_weight": _equal_weight,
}


# --- entry points ----------------------------------------------------------


@dataclass
class RebalanceRecord:
    """One rebalance, with everything that had to be decided to produce it."""

    as_of: pd.Timestamp
    weights: Dict[str, float]
    fallback_used: Optional[str] = None
    requested_turnover_cap: Optional[float] = None
    realized_turnover: Optional[float] = None
    precedence_overrides: List[Dict[str, Any]] = field(default_factory=list)


def execute_detailed(
    methodology: Methodology,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    prev_weights: Optional[Mapping[str, float]] = None,
) -> RebalanceRecord:
    """Execute one rebalance and return the weights plus the audit trail.

    `execute` wraps this for callers that only want weights. The audit trail is
    what turns a silent deviation — a fallback firing, a soft constraint yielding
    to a hard one — into a reportable decision.
    """
    ctx_holder: Dict[str, ExecutionContext] = {}
    weights = _execute(methodology, prices, as_of, prev_weights, ctx_holder)
    ctx = ctx_holder["ctx"]
    return RebalanceRecord(
        as_of=as_of,
        weights=weights,
        fallback_used=ctx.fallback_used,
        requested_turnover_cap=ctx.requested_turnover_cap,
        realized_turnover=ctx.realized_turnover,
        precedence_overrides=list(ctx.precedence_overrides),
    )


def execute(
    methodology: Methodology,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    prev_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Execute one rebalance, returning weights only."""
    return _execute(methodology, prices, as_of, prev_weights, {})


def _execute(
    methodology: Methodology,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    prev_weights: Optional[Mapping[str, float]],
    ctx_holder: Dict[str, "ExecutionContext"],
) -> Dict[str, float]:
    """Compute weights for one rebalance date.

    `prices` is sliced to `as_of` inclusive before anything else runs, so a
    methodology physically cannot see the future regardless of what its pipeline
    declares.
    """
    universe = [t for t in methodology.contract.universe if t in prices.columns]
    missing = set(methodology.contract.universe) - set(universe)
    if missing:
        raise ExecutionError(
            f"{methodology.version_id} contract names assets absent from the price "
            f"panel: {sorted(missing)}"
        )

    history = prices.loc[:as_of, universe]
    returns = np.log(history / history.shift(1)).dropna()

    excluded = set(methodology.excluded_assets)
    if excluded:
        keep = [t for t in returns.columns if t not in excluded]
        if not keep:
            raise ExecutionError("exclusions leave an empty universe")
        returns = returns[keep]

    ctx = ExecutionContext(
        methodology=methodology, returns=returns, as_of=as_of, prev_weights=prev_weights
    )
    ctx_holder["ctx"] = ctx

    unknown = [s for s in methodology.pipeline if s not in PIPELINE_STEPS]
    if unknown:
        raise ExecutionError(f"unknown pipeline steps: {unknown}")

    try:
        for step in methodology.pipeline:
            PIPELINE_STEPS[step](ctx)
    except ExecutionError:
        lookback = int(ctx.param("lookback", 252))
        for name in methodology.fallback_chain:
            handler = FALLBACKS.get(name)
            if handler is None:
                continue
            try:
                ctx.weights = handler(returns, min(lookback, len(returns)))
            except ExecutionError:
                continue
            ctx.fallback_used = name
            # Fallbacks are subject to the same contract as the primary path.
            # Returning here without bounds enforcement let inverse-volatility
            # allocate 83% to the cash proxy under a declared 25% ceiling.
            _apply_bounds(ctx)
            weights = {t: float(ctx.weights.get(t, 0.0)) for t in universe}
            _verify_contract(methodology, weights, as_of)
            return weights
        raise

    weights = {t: float(ctx.weights.get(t, 0.0)) for t in universe}
    _verify_contract(methodology, weights, as_of)
    return weights


def _verify_contract(
    methodology: Methodology, weights: Mapping[str, float], as_of: pd.Timestamp
) -> None:
    """Check the produced weights against the contract the methodology published.

    A pipeline that silently produces out-of-bounds weights makes the output
    contract a comment rather than a promise. This caught `apply_turnover_cap`
    carrying weights back over the ceiling after `apply_bounds` had enforced it.
    """
    bounds = methodology.contract.weight_bounds
    lo, hi = float(bounds.get("min", 0.0)), float(bounds.get("max", 1.0))
    tol = 1e-6

    breaches = {
        t: w for t, w in weights.items() if w < lo - tol or w > hi + tol
    }
    if breaches:
        raise ExecutionError(
            f"{methodology.version_id} produced weights outside its contract "
            f"bounds [{lo}, {hi}] on {as_of.date()}: "
            + ", ".join(f"{t}={w:.6f}" for t, w in sorted(breaches.items()))
        )

    gross = sum(abs(w) for w in weights.values())
    if gross > methodology.contract.gross_leverage_max + tol:
        raise ExecutionError(
            f"{methodology.version_id} gross leverage {gross:.6f} exceeds contract "
            f"maximum {methodology.contract.gross_leverage_max} on {as_of.date()}"
        )


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str, warmup: int) -> List[pd.Timestamp]:
    """Rebalance dates from the contract's frequency, e.g. ``5B`` or ``21B``.

    Only business-day intervals are supported in 0.1; anything else is an error
    rather than a silent fallback to a default cadence, because a cadence that
    differs from the published one invalidates the result.
    """
    if not frequency.endswith("B"):
        raise ExecutionError(
            f"unsupported rebalance_frequency {frequency!r}; spec 0.1 supports NB only"
        )
    try:
        step = int(frequency[:-1])
    except ValueError as exc:
        raise ExecutionError(f"malformed rebalance_frequency {frequency!r}") from exc
    if step < 1:
        raise ExecutionError("rebalance_frequency must be at least 1 business day")

    return list(index[warmup::step])


def backtest(
    methodology: Methodology,
    prices: pd.DataFrame,
    warmup: Optional[int] = None,
) -> pd.DataFrame:
    """Run a methodology across its own rebalance schedule.

    Returns a long-form weights frame (`date`, `ticker`, `weight`) in exactly the
    shape `history.strategy_daily_returns` consumes, so the corrected execution
    lag and cost model apply to methodology-driven results too.
    """
    lookback = int(methodology.params["lookback"].value) if "lookback" in methodology.params else 252
    warmup = lookback if warmup is None else warmup

    dates = _rebalance_dates(prices.index, methodology.contract.rebalance_frequency, warmup)
    if not dates:
        raise ExecutionError(
            f"no rebalance dates: {len(prices)} rows, {warmup} warmup, "
            f"{methodology.contract.rebalance_frequency} cadence"
        )

    rows: List[Dict[str, object]] = []
    prev: Optional[Dict[str, float]] = None
    records: List[RebalanceRecord] = []
    for date in dates:
        record = execute_detailed(methodology, prices, date, prev_weights=prev)
        records.append(record)
        for ticker, weight in record.weights.items():
            rows.append({"date": date, "ticker": ticker, "weight": weight})
        prev = record.weights

    frame = pd.DataFrame(rows)
    frame.attrs["execution_audit"] = summarize_execution(records)
    return frame


def summarize_execution(records: Sequence[RebalanceRecord]) -> Dict[str, Any]:
    """Aggregate the audit trail across a backtest.

    `fallback_share` is the number that matters most: a strategy that spent 40%
    of its evaluation period on a fallback rule must not be presented as though
    the primary methodology generated the whole record.
    """
    n = len(records)
    fallback_dates = [r for r in records if r.fallback_used]
    by_rule: Dict[str, int] = {}
    for r in fallback_dates:
        by_rule[r.fallback_used] = by_rule.get(r.fallback_used, 0) + 1

    overrides = [o for r in records for o in r.precedence_overrides]
    realized = [r.realized_turnover for r in records if r.realized_turnover is not None]
    requested = next(
        (r.requested_turnover_cap for r in records if r.requested_turnover_cap is not None),
        None,
    )

    return {
        "n_rebalances": n,
        "fallback_rebalances": len(fallback_dates),
        "fallback_share": (len(fallback_dates) / n) if n else 0.0,
        "fallback_by_rule": by_rule,
        "fallback_first_date": str(fallback_dates[0].as_of.date()) if fallback_dates else None,
        "fallback_last_date": str(fallback_dates[-1].as_of.date()) if fallback_dates else None,
        "requested_turnover_cap": requested,
        "realized_turnover_mean": (sum(realized) / len(realized)) if realized else None,
        "realized_turnover_max": max(realized) if realized else None,
        "precedence_override_count": len(overrides),
        "precedence_overrides": overrides[:50],
    }
