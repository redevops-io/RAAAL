"""Trial identity — the authoritative unit of multiple testing.

A trial is **an attempted, materially distinct analytical configuration**, not a
published methodology version and not a Cartesian product of versions and
protocols. Counting versions × protocols overcounts pairings never run and
undercounts repeated variants within one pairing; either error corrupts the
Deflated Sharpe Ratio, whose entire purpose is an honest ``N``.

Identity is a hash of everything that could have been varied in search of a
preferred result::

    trial_identity = sha256(
        methodology hypothesis,
        evaluation protocol,
        declared optimization objective,
        data partition,
        materially relevant execution assumptions,
    )

Two runs with the same identity are one trial repeated — a reproducibility check,
not a new search. Two runs differing in any component are two trials, whether the
difference was a lookback, an embargo, or a cost assumption.

**Failed and blocked attempts are recorded.** A pairing rejected for
incompatibility is still evidence that the configuration was attempted, and a
searcher who tries twenty configurations and reports only the two that executed
has still searched twenty times. Whether a given failure class counts toward the
DSR denominator is a stated policy (see :data:`DSR_COUNTABLE_OUTCOMES`), not an
accident of which errors happened to raise.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class TrialOutcome(str, Enum):
    """What happened to an attempted configuration."""

    COMPLETED = "completed"
    """Ran to completion and produced a result."""

    BLOCKED_INCOMPATIBLE = "blocked_incompatible"
    """Refused before execution — the pairing was invalid (e.g. warmup shorter
    than lookback). The configuration was still attempted."""

    FAILED_EXECUTION = "failed_execution"
    """Started and raised. Includes contract breaches discovered at run time."""

    ABANDONED = "abandoned"
    """Cancelled by the operator before completion."""


#: Which outcomes enter the DSR trial count.
#:
#: Stated explicitly because the choice is contestable and must not be implicit.
#: A configuration blocked *before* execution reveals nothing about the data — the
#: searcher learned only that the pairing was malformed — so it does not inflate
#: the maximum-Sharpe distribution and is excluded. A configuration that ran and
#: failed *may* have been abandoned after a peek at partial results, so it counts.
#: Completed runs always count.
DSR_COUNTABLE_OUTCOMES = frozenset(
    {TrialOutcome.COMPLETED, TrialOutcome.FAILED_EXECUTION, TrialOutcome.ABANDONED}
)


@dataclass(frozen=True)
class TrialIdentity:
    """The five components that make a configuration materially distinct."""

    methodology_hash: str
    protocol_hash: str
    objective: str
    data_partition: str
    execution_assumptions: Mapping[str, Any] = field(default_factory=dict)

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "methodology_hash": self.methodology_hash,
            "protocol_hash": self.protocol_hash,
            "objective": self.objective,
            "data_partition": self.data_partition,
            "execution_assumptions": {
                k: self.execution_assumptions[k]
                for k in sorted(self.execution_assumptions)
            },
        }

    @property
    def trial_id(self) -> str:
        payload = json.dumps(
            self.canonical_form(), sort_keys=True, separators=(",", ":"), default=str
        )
        return "trial_" + hashlib.sha256(payload.encode()).hexdigest()[:32]

    def to_json(self) -> Dict[str, Any]:
        return {**self.canonical_form(), "trial_id": self.trial_id}


def build_trial_identity(
    *,
    methodology_hash: str,
    protocol_hash: str,
    objective: str = "annualized_return",
    data_partition: str = "full",
    execution_assumptions: Optional[Mapping[str, Any]] = None,
) -> TrialIdentity:
    """Construct a trial identity.

    `data_partition` distinguishes in-sample from holdout evaluation of the same
    methodology and protocol: those are genuinely different attempts, and
    collapsing them would let a searcher evaluate on both and report the better.
    """
    return TrialIdentity(
        methodology_hash=methodology_hash,
        protocol_hash=protocol_hash,
        objective=objective,
        data_partition=data_partition,
        execution_assumptions=dict(execution_assumptions or {}),
    )
