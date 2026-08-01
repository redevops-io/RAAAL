"""Deterministic statistical estimators.

Kept strictly separate from interpretive diagnostics: these functions compute
numbers from declared inputs and do not decide whether a strategy is any good.
That judgement belongs to the publication policy, which can then be changed
without touching the arithmetic.

Every estimator returns a structured result rather than a bare float, carrying
the assumptions it made, the trial count it used, its own version, and the runs
it drew on. A Deflated Sharpe Ratio without its ``N`` is not interpretable, and
returning one as a float invites exactly that.

References
----------
Bailey & López de Prado (2014), *The Deflated Sharpe Ratio* — SSRN 2460551.
Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest
Overfitting*, Journal of Computational Finance.
Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns*.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

ESTIMATOR_VERSION = "0.1.0"

#: Euler–Mascheroni constant, used in the expected-maximum-Sharpe benchmark.
EULER_GAMMA = 0.5772156649015329


@dataclass(frozen=True)
class StatisticsInput:
    """Everything an estimator was told. Recorded so a number can be re-derived."""

    return_frequency: str = "daily"
    observations: int = 0
    trials_observed: int = 1
    skewness: float = 0.0
    kurtosis: float = 3.0
    benchmark_sharpe: float = 0.0
    cv_scheme: Optional[str] = None
    purge: int = 0
    embargo: int = 0
    factor_model: Optional[str] = None
    cost_model: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "return_frequency": self.return_frequency,
            "observations": self.observations,
            "trials_observed": self.trials_observed,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "benchmark_sharpe": self.benchmark_sharpe,
            "cv_scheme": self.cv_scheme,
            "purge": self.purge,
            "embargo": self.embargo,
            "factor_model": self.factor_model,
            "cost_model": self.cost_model,
        }


@dataclass(frozen=True)
class StatisticResult:
    """A statistic, its eligibility, and the trail back to the runs behind it."""

    name: str
    value: float
    eligible: bool
    inputs: StatisticsInput
    estimator_version: str = ESTIMATOR_VERSION
    assumptions: Sequence[str] = ()
    warnings: Sequence[str] = ()
    run_ids: Sequence[str] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "eligible": self.eligible,
            "estimator_version": self.estimator_version,
            "inputs": self.inputs.to_json(),
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "run_ids": list(self.run_ids),
            **dict(self.extra),
        }


PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}


def _annualization(frequency: str) -> int:
    if frequency not in PERIODS_PER_YEAR:
        raise ValueError(f"unsupported return_frequency {frequency!r}")
    return PERIODS_PER_YEAR[frequency]


def sharpe_ratio(returns: pd.Series, frequency: str = "daily") -> float:
    """Annualized Sharpe on excess returns already net of the risk-free rate."""
    periods = _annualization(frequency)
    sd = float(returns.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(returns.mean() / sd * math.sqrt(periods))


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    frequency: str = "daily",
    run_ids: Sequence[str] = (),
) -> StatisticResult:
    """PSR — probability the true Sharpe exceeds a benchmark.

    Corrects the Sharpe estimate for sample length, skewness and kurtosis. A high
    Sharpe on a short, negatively skewed, fat-tailed record is much weaker
    evidence than the same Sharpe on a long, well-behaved one, and PSR is what
    makes that difference visible.

        PSR = Φ[ (SR − SR*)·√(T−1) / √(1 − γ₃·SR + ((γ₄−1)/4)·SR²) ]

    Note the estimate uses *per-period* Sharpe, not annualized, since T is a count
    of periods.
    """
    warnings: List[str] = []
    n = len(returns)
    if n < 3:
        return StatisticResult(
            name="psr", value=float("nan"), eligible=False,
            inputs=StatisticsInput(return_frequency=frequency, observations=n),
            warnings=["fewer than 3 observations"], run_ids=run_ids,
        )

    periods = _annualization(frequency)
    sd = float(returns.std(ddof=1))
    if sd == 0:
        return StatisticResult(
            name="psr", value=float("nan"), eligible=False,
            inputs=StatisticsInput(return_frequency=frequency, observations=n),
            warnings=["zero return variance — Sharpe undefined"], run_ids=run_ids,
        )

    sr_period = float(returns.mean() / sd)
    sr_benchmark_period = benchmark_sharpe / math.sqrt(periods)
    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))

    denominator = 1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * sr_period**2
    if denominator <= 0:
        return StatisticResult(
            name="psr", value=float("nan"), eligible=False,
            inputs=StatisticsInput(
                return_frequency=frequency, observations=n,
                skewness=skew, kurtosis=kurt, benchmark_sharpe=benchmark_sharpe,
            ),
            warnings=["non-positive variance term; moments are pathological"],
            run_ids=run_ids,
        )

    z = (sr_period - sr_benchmark_period) * math.sqrt(n - 1) / math.sqrt(denominator)
    value = float(stats.norm.cdf(z))

    if n < 3 * periods:
        warnings.append(
            f"{n} observations is under three years at {frequency} frequency; "
            "PSR is sensitive to sample length"
        )

    return StatisticResult(
        name="psr",
        value=value,
        eligible=True,
        inputs=StatisticsInput(
            return_frequency=frequency, observations=n, skewness=skew,
            kurtosis=kurt, benchmark_sharpe=benchmark_sharpe,
        ),
        assumptions=[
            "returns are IID under the null",
            "benchmark Sharpe is a fixed threshold, not estimated from this sample",
        ],
        warnings=warnings,
        run_ids=run_ids,
        extra={"sharpe_annualized": sr_period * math.sqrt(periods)},
    )


def expected_max_sharpe(trials: int, variance_of_sharpes: float) -> float:
    """Expected maximum Sharpe from `trials` independent noise draws.

        SR* = √V · ( (1−γ)·Z⁻¹[1 − 1/N] + γ·Z⁻¹[1 − 1/(N·e)] )

    This is the benchmark a candidate must beat to be evidence of anything: with
    enough attempts, some strategy clears any fixed Sharpe by luck alone.

    **Units.** `variance_of_sharpes` must be the variance of *annualized* Sharpe
    ratios — the units :func:`sharpe_ratio` returns — and the result is likewise
    an annualized Sharpe. Mixing per-period and annualized units here inflates
    the benchmark by √periods and makes every candidate fail.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if trials == 1:
        return 0.0
    sd = math.sqrt(max(variance_of_sharpes, 0.0))
    a = stats.norm.ppf(1.0 - 1.0 / trials)
    b = stats.norm.ppf(1.0 - 1.0 / (trials * math.e))
    return float(sd * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe_ratio(
    returns: pd.Series,
    trials_observed: int,
    variance_of_sharpes: Optional[float] = None,
    trial_sharpes: Optional[Sequence[float]] = None,
    frequency: str = "daily",
    run_ids: Sequence[str] = (),
) -> StatisticResult:
    """DSR — PSR against the expected maximum Sharpe across `trials_observed`.

    The trial count is *not* a caller convenience: it is the correction. Passing
    1 when twenty configurations were attempted produces a number that looks like
    evidence and is not. The ledger supplies it.
    """
    warnings: List[str] = []

    if trial_sharpes is not None and len(trial_sharpes) > 1:
        variance_of_sharpes = float(np.var(trial_sharpes, ddof=1))
    elif variance_of_sharpes is None:
        # Without an observed spread, assume unit variance of annualized trial
        # Sharpes. Conservative in that it demands more from the candidate, but
        # it is an assumption and is recorded as one.
        variance_of_sharpes = 1.0
        warnings.append(
            "variance of trial Sharpes not supplied; assumed 1.0 (annualized "
            "units). Supply the observed spread across the lineage for a "
            "calibrated benchmark."
        )

    if trials_observed < 1:
        raise ValueError("trials_observed must be at least 1")
    if trials_observed == 1:
        warnings.append(
            "trials_observed = 1: no deflation is applied, so DSR equals PSR "
            "against a zero benchmark. This is only honest if exactly one "
            "configuration was ever attempted."
        )

    # `expected_max_sharpe` consumes and returns annualized units, matching
    # `sharpe_ratio`. Re-annualizing here would inflate the benchmark by
    # √periods and fail every candidate regardless of merit.
    sr_star_annualized = expected_max_sharpe(trials_observed, variance_of_sharpes)

    psr = probabilistic_sharpe_ratio(
        returns, benchmark_sharpe=sr_star_annualized, frequency=frequency, run_ids=run_ids
    )

    return StatisticResult(
        name="dsr",
        value=psr.value,
        eligible=psr.eligible,
        inputs=StatisticsInput(
            return_frequency=frequency,
            observations=len(returns),
            trials_observed=trials_observed,
            skewness=psr.inputs.skewness,
            kurtosis=psr.inputs.kurtosis,
            benchmark_sharpe=sr_star_annualized,
        ),
        assumptions=[
            "trial Sharpes are drawn from a common distribution",
            "trials_observed is the platform-observed count, not self-reported",
            f"variance of trial Sharpes = {variance_of_sharpes:.6f}",
        ],
        warnings=list(psr.warnings) + warnings,
        run_ids=run_ids,
        extra={
            "expected_max_sharpe_annualized": sr_star_annualized,
            "sharpe_annualized": psr.extra.get("sharpe_annualized", float("nan")),
        },
    )


def minimum_track_record_length(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    frequency: str = "daily",
) -> StatisticResult:
    """MinTRL — observations needed to claim SR > benchmark at `confidence`.

        MinTRL = 1 + [1 − γ₃·SR + ((γ₄−1)/4)·SR²] · (Z_α / (SR − SR*))²
    """
    n = len(returns)
    periods = _annualization(frequency)
    sd = float(returns.std(ddof=1))
    if n < 3 or sd == 0:
        return StatisticResult(
            name="min_trl", value=float("nan"), eligible=False,
            inputs=StatisticsInput(return_frequency=frequency, observations=n),
            warnings=["insufficient observations or zero variance"],
        )

    sr = float(returns.mean() / sd)
    sr_star = benchmark_sharpe / math.sqrt(periods)
    if sr <= sr_star:
        return StatisticResult(
            name="min_trl", value=float("inf"), eligible=False,
            inputs=StatisticsInput(
                return_frequency=frequency, observations=n,
                benchmark_sharpe=benchmark_sharpe,
            ),
            warnings=["observed Sharpe does not exceed the benchmark at any length"],
        )

    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))
    z = stats.norm.ppf(confidence)
    value = 1.0 + (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2) * (z / (sr - sr_star)) ** 2

    return StatisticResult(
        name="min_trl",
        value=float(value),
        eligible=True,
        inputs=StatisticsInput(
            return_frequency=frequency, observations=n, skewness=skew,
            kurtosis=kurt, benchmark_sharpe=benchmark_sharpe,
        ),
        assumptions=[f"confidence level {confidence}"],
        warnings=(
            [f"track record of {n} is shorter than the required {value:.0f}"]
            if n < value else []
        ),
        extra={"observations_available": n, "sufficient": n >= value},
    )


def probability_of_backtest_overfitting(
    trial_returns: pd.DataFrame,
    n_splits: int = 8,
    frequency: str = "daily",
) -> StatisticResult:
    """PBO via Combinatorially Symmetric Cross-Validation.

    Partitions the return matrix into `n_splits` blocks, forms every balanced
    train/test division, selects the in-sample best configuration in each, and
    measures how often it lands below median out-of-sample.

    PBO near 0.5 means in-sample ranking carries no information about
    out-of-sample ranking — the selection procedure is choosing noise.
    """
    warnings: List[str] = []
    n_trials = trial_returns.shape[1]
    if n_trials < 2:
        return StatisticResult(
            name="pbo", value=float("nan"), eligible=False,
            inputs=StatisticsInput(return_frequency=frequency, trials_observed=n_trials),
            warnings=["PBO requires at least 2 configurations"],
        )
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even for a symmetric split")

    rows = len(trial_returns)
    block = rows // n_splits
    if block < 2:
        return StatisticResult(
            name="pbo", value=float("nan"), eligible=False,
            inputs=StatisticsInput(
                return_frequency=frequency, observations=rows, trials_observed=n_trials
            ),
            warnings=[f"{rows} observations cannot form {n_splits} usable blocks"],
        )

    blocks = [trial_returns.iloc[i * block : (i + 1) * block] for i in range(n_splits)]
    logits: List[float] = []

    for train_idx in itertools.combinations(range(n_splits), n_splits // 2):
        test_idx = [i for i in range(n_splits) if i not in train_idx]
        train = pd.concat([blocks[i] for i in train_idx])
        test = pd.concat([blocks[i] for i in test_idx])

        train_sr = train.apply(lambda c: sharpe_ratio(c, frequency))
        test_sr = test.apply(lambda c: sharpe_ratio(c, frequency))
        if train_sr.isna().all() or test_sr.isna().all():
            continue

        best = train_sr.idxmax()
        # Relative rank of the in-sample winner, out of sample.
        rank = float(test_sr.rank(pct=True).loc[best])
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1.0 - rank)))

    if not logits:
        return StatisticResult(
            name="pbo", value=float("nan"), eligible=False,
            inputs=StatisticsInput(
                return_frequency=frequency, observations=rows, trials_observed=n_trials
            ),
            warnings=["no usable splits"],
        )

    pbo = float(np.mean([1.0 if x < 0 else 0.0 for x in logits]))
    if pbo > 0.5:
        warnings.append(
            "PBO above 0.5: the in-sample winner lands below median out of sample "
            "more often than not, so selection is anti-informative"
        )

    return StatisticResult(
        name="pbo",
        value=pbo,
        eligible=True,
        inputs=StatisticsInput(
            return_frequency=frequency, observations=rows,
            trials_observed=n_trials, cv_scheme=f"cscv_{n_splits}",
        ),
        assumptions=[
            "configurations are comparable and evaluated on aligned periods",
            f"{len(logits)} symmetric splits of {n_splits} blocks",
        ],
        warnings=warnings,
        extra={"n_splits_evaluated": len(logits), "median_logit": float(np.median(logits))},
    )
