"""Factor neutralization against a versioned factor model.

`risk_model.factor_model` must reference an identified artifact, not a bare
string. "Factor-neutralized" otherwise hides a large set of implementation
choices — which factors, estimated over what window, rebalanced how often, with
what data policy and what neutralization method — any of which changes the
residual and therefore the published number.

**Raw and residual returns are both preserved, never substituted.** A neutralized
return answers a different question from a raw one ("is there alpha beyond these
factors?" versus "what did this earn?"), and replacing one with the other loses
the question the reader was asking.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

NEUTRALIZATION_SPEC_VERSION = "0.1"


@dataclass(frozen=True)
class FactorModel:
    """A versioned, hashable factor model."""

    name: str
    version: int
    factors: Sequence[str]
    estimation_window: int = 252
    rebalance_frequency: str = "monthly"
    data_snapshot_policy: str = "point_in_time"
    neutralization_method: str = "ols_residual"
    spec_version: str = NEUTRALIZATION_SPEC_VERSION

    @property
    def model_id(self) -> str:
        return f"factor-model/{self.name}@{self.version}"

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "factors": list(self.factors),
            "estimation_window": self.estimation_window,
            "rebalance_frequency": self.rebalance_frequency,
            "data_snapshot_policy": self.data_snapshot_policy,
            "neutralization_method": self.neutralization_method,
        }

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.canonical_form(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "model_id": self.model_id,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class NeutralizationResult:
    """Raw and residual returns side by side, with the model that separated them."""

    factor_model_id: str
    factor_model_hash: str
    raw_returns: pd.Series
    residual_returns: pd.Series
    betas: Mapping[str, float]
    r_squared: float
    observations: int
    warnings: Sequence[str] = field(default_factory=tuple)

    def to_json(self) -> Dict[str, Any]:
        return {
            "factor_model_id": self.factor_model_id,
            "factor_model_hash": self.factor_model_hash,
            "betas": dict(self.betas),
            "r_squared": self.r_squared,
            "observations": self.observations,
            "raw_annualized_mean": float(self.raw_returns.mean() * 252),
            "residual_annualized_mean": float(self.residual_returns.mean() * 252),
            "warnings": list(self.warnings),
            "note": (
                "Raw and residual returns are both retained. A residual answers "
                "'is there alpha beyond these factors?'; a raw return answers "
                "'what did this earn?'. Neither substitutes for the other."
            ),
        }


def neutralize_returns(
    returns: pd.Series,
    factor_returns: pd.DataFrame,
    model: FactorModel,
) -> NeutralizationResult:
    """Regress returns on factors and retain the residual.

    Uses OLS with an intercept: the residual plus intercept is the return that
    the factor exposures do not explain. Betas are reported so the exposure being
    removed is visible rather than implied.
    """
    warnings: list[str] = []

    aligned = pd.concat([returns.rename("y"), factor_returns], axis=1).dropna()
    if len(aligned) < len(model.factors) + 2:
        raise ValueError(
            f"{len(aligned)} aligned observations is too few to estimate "
            f"{len(model.factors)} factor loadings"
        )

    missing = [f for f in model.factors if f not in factor_returns.columns]
    if missing:
        raise ValueError(f"{model.model_id} names factors absent from the data: {missing}")

    y = aligned["y"].to_numpy()
    X = aligned[list(model.factors)].to_numpy()
    X_design = np.column_stack([np.ones(len(X)), X])

    coefficients, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    fitted = X_design @ coefficients
    residual = y - fitted

    ss_total = float(((y - y.mean()) ** 2).sum())
    ss_residual = float((residual**2).sum())
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")

    if len(aligned) < model.estimation_window:
        warnings.append(
            f"{len(aligned)} observations is shorter than the model's declared "
            f"{model.estimation_window}-observation estimation window"
        )

    # The intercept is the part of the mean return the factors do not explain;
    # adding it back makes the residual series interpretable as a return stream.
    residual_series = pd.Series(residual + coefficients[0], index=aligned.index)

    return NeutralizationResult(
        factor_model_id=model.model_id,
        factor_model_hash=model.content_hash,
        raw_returns=aligned["y"],
        residual_returns=residual_series,
        betas=dict(zip(model.factors, coefficients[1:])),
        r_squared=r_squared,
        observations=len(aligned),
        warnings=tuple(warnings),
    )
