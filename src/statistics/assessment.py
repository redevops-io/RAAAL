"""Statistical assessment — facts and diagnostics, no verdicts.

This layer answers exactly one question: **what does the evidence say?** It does
not decide whether a result meets a standard (that is the policy layer) and it
does not decide whether anyone may see it (that is the publication gate).

Keeping the three apart matters because they change at different rates and for
different reasons. The arithmetic of a Deflated Sharpe Ratio should be stable for
years; the threshold a library requires is a product decision that will be
revised; what a private draft may show differs from what a "validated" badge may
claim. Collapsing them into one `statistical_valid` boolean makes all three
unrevisable at once, and makes "valid" mean too many things.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from .estimators import (
    ESTIMATOR_VERSION,
    StatisticResult,
    deflated_sharpe_ratio,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from .neutralize import FactorModel, NeutralizationResult, neutralize_returns

ASSESSMENT_VERSION = "0.1.0"


@dataclass(frozen=True)
class StatisticalAssessment:
    """Everything the estimators found, with no judgement attached."""

    computation_status: str                      # VALID | PARTIAL | FAILED
    psr: Optional[Dict[str, Any]] = None
    dsr: Optional[Dict[str, Any]] = None
    pbo: Optional[Dict[str, Any]] = None
    min_track_record_length: Optional[Dict[str, Any]] = None
    factor_neutralization: Optional[Dict[str, Any]] = None
    observations: int = 0
    trial_count: int = 0
    count_policy: str = "DSR_COUNTABLE_OUTCOMES"
    warnings: Sequence[str] = field(default_factory=tuple)
    assessment_version: str = ASSESSMENT_VERSION
    estimator_version: str = ESTIMATOR_VERSION
    run_ids: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "StatisticalAssessment":
        """Rebuild a recorded assessment exactly.

        `to_json` is lossless over every field, which is what makes it legitimate
        to re-judge a historical run under a current policy: the facts are the
        recorded facts, not a re-derivation. The `note` key is prose for readers
        and carries no state, so it is dropped.
        """
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in fields})

    @property
    def complete(self) -> bool:
        """Whether every estimator that was requested produced a value."""
        return self.computation_status == "VALID"

    def to_json(self) -> Dict[str, Any]:
        return {
            "computation_status": self.computation_status,
            "psr": self.psr,
            "dsr": self.dsr,
            "pbo": self.pbo,
            "min_track_record_length": self.min_track_record_length,
            "factor_neutralization": self.factor_neutralization,
            "observations": self.observations,
            "trial_count": self.trial_count,
            "count_policy": self.count_policy,
            "warnings": list(self.warnings),
            "assessment_version": self.assessment_version,
            "estimator_version": self.estimator_version,
            "run_ids": list(self.run_ids),
            "note": (
                "Facts and diagnostics only. Whether these meet a standard is a "
                "statistical-policy decision; whether they may be published is a "
                "publication-gate decision."
            ),
        }


def assess(
    returns: pd.Series,
    *,
    trial_count: int,
    trial_sharpes: Optional[Sequence[float]] = None,
    lineage_returns: Optional[pd.DataFrame] = None,
    factor_returns: Optional[pd.DataFrame] = None,
    factor_model: Optional[FactorModel] = None,
    benchmark_sharpe: float = 0.0,
    frequency: str = "daily",
    run_ids: Sequence[str] = (),
    count_policy: str = "DSR_COUNTABLE_OUTCOMES",
) -> StatisticalAssessment:
    """Compute the full statistical picture for one result.

    `trial_count` comes from the ledger, never from the caller's recollection —
    it is the correction, not a formality. `lineage_returns` enables PBO, which
    needs the whole configuration set rather than a single series.
    """
    warnings: list[str] = []
    partial = False

    psr = probabilistic_sharpe_ratio(
        returns, benchmark_sharpe=benchmark_sharpe, frequency=frequency, run_ids=run_ids
    )
    if not psr.eligible:
        partial = True
        warnings.extend(psr.warnings)

    variance = None
    if trial_sharpes is None and lineage_returns is not None:
        trial_sharpes = [sharpe_ratio(lineage_returns[c], frequency) for c in lineage_returns]

    dsr = deflated_sharpe_ratio(
        returns,
        trials_observed=max(trial_count, 1),
        trial_sharpes=trial_sharpes,
        variance_of_sharpes=variance,
        frequency=frequency,
        run_ids=run_ids,
    )
    if not dsr.eligible:
        partial = True
    warnings.extend(dsr.warnings)

    mtrl = minimum_track_record_length(
        returns, benchmark_sharpe=benchmark_sharpe, frequency=frequency
    )

    pbo_payload = None
    if lineage_returns is not None and lineage_returns.shape[1] >= 2:
        pbo = probability_of_backtest_overfitting(lineage_returns, frequency=frequency)
        pbo_payload = pbo.to_json()
        warnings.extend(pbo.warnings)
    else:
        warnings.append(
            "PBO not computed: it requires at least two comparable configurations "
            "from the same lineage"
        )
        partial = True

    neutralization_payload = None
    if factor_model is not None and factor_returns is not None:
        try:
            result: NeutralizationResult = neutralize_returns(
                returns, factor_returns, factor_model
            )
            neutralization_payload = result.to_json()
        except ValueError as exc:
            warnings.append(f"factor neutralization failed: {exc}")
            partial = True
    else:
        warnings.append("factor neutralization not requested")

    status = "VALID" if not partial else "PARTIAL"
    if not psr.eligible and not dsr.eligible:
        status = "FAILED"

    return StatisticalAssessment(
        computation_status=status,
        psr=psr.to_json(),
        dsr=dsr.to_json(),
        pbo=pbo_payload,
        min_track_record_length=mtrl.to_json(),
        factor_neutralization=neutralization_payload,
        observations=len(returns),
        trial_count=trial_count,
        count_policy=count_policy,
        warnings=tuple(dict.fromkeys(warnings)),
        run_ids=tuple(run_ids),
    )
