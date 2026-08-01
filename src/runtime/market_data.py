"""Where prices come from, and how they must be read.

`data_snapshot` was a bare string in `ISOLATION_DIMENSIONS` — the same defect
`tax_treatment` had. Two runs whose snapshot strings differ are reported as
incomparable even when nothing about the data policy changed, and two runs whose
strings match are reported as comparable even when one was point-in-time and the
other was not.

The split matters and mirrors one already made for `DataSnapshot`:

    MarketDataRuntime   how data must be sourced and interpreted.  Declared.
    Run record          what was actually served.  Realized.

A vendor restating history does not change the runtime. It changes what the run
received, and that is exactly the event worth seeing — a *revision*, not a
different policy. Collapsing the two would make every restatement look like a
methodology change and every policy change look like a data refresh.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Sequence

from .base import Exclusion, RuntimeArtifact, RuntimeAssumption, RuntimeLimitation


class AdjustmentPolicy(str, Enum):
    ADJUSTED_ONLY = "adjusted_only"
    """Prices arrive already adjusted for splits and dividends."""

    UNADJUSTED_ONLY = "unadjusted_only"
    """Raw prices; adjustments are applied downstream."""

    BOTH = "both"


class PointInTimePolicy(str, Enum):
    POINT_IN_TIME = "point_in_time"
    """Each date shows what was knowable then. Restatements are kept separately."""

    LATEST_RESTATED = "latest_restated"
    """Each date shows the vendor's current view. Cheap, and quietly
    lookahead-contaminated for anything that gets revised."""


class UniversePolicy(str, Enum):
    SURVIVORSHIP_FREE = "survivorship_free"
    CURRENT_MEMBERS_ONLY = "current_members_only"
    """Only instruments that still exist. Every backtest over this is biased
    upward by exactly the ones that did not make it."""


@dataclass(frozen=True)
class MarketDataRuntime(RuntimeArtifact):
    kind: ClassVar[str] = "market_data"

    name: str
    version: int
    provider: str
    dataset: str = ""
    adjustment_policy: AdjustmentPolicy = AdjustmentPolicy.ADJUSTED_ONLY
    point_in_time_policy: PointInTimePolicy = PointInTimePolicy.LATEST_RESTATED
    timezone: str = "America/New_York"
    session_alignment: str = ""
    """The calendar these sessions are aligned to, by name. Checked against the
    calendar runtime rather than assumed to agree."""

    corporate_action_source: str = ""
    universe_membership_policy: UniversePolicy = UniversePolicy.CURRENT_MEMBERS_ONLY
    revision_policy: str = ""
    coverage: str = ""
    title: str = ""

    def declared_form(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "name": self.name, "version": self.version,
            "provider": self.provider, "dataset": self.dataset,
            "adjustment_policy": self.adjustment_policy.value,
            "point_in_time_policy": self.point_in_time_policy.value,
            "timezone": self.timezone,
            "session_alignment": self.session_alignment,
            "corporate_action_source": self.corporate_action_source,
            "universe_membership_policy": self.universe_membership_policy.value,
            "revision_policy": self.revision_policy,
            "coverage": self.coverage, "title": self.title,
        }

    def comparable_form(self) -> Dict[str, Any]:
        """How the data must be read — not which files arrived.

        `coverage`, `revision_policy` prose and `title` are excluded: they
        describe the arrangement without changing how a price is interpreted.
        The realized snapshot is not here at all; it belongs to the run.
        """
        return {
            "provider": self.provider,
            "dataset": self.dataset,
            "adjustment_policy": self.adjustment_policy.value,
            "point_in_time_policy": self.point_in_time_policy.value,
            "timezone": self.timezone,
            "session_alignment": self.session_alignment,
            "corporate_action_source": self.corporate_action_source,
            "universe_membership_policy": self.universe_membership_policy.value,
        }

    @property
    def is_point_in_time(self) -> bool:
        return self.point_in_time_policy is PointInTimePolicy.POINT_IN_TIME

    @property
    def assumptions(self) -> Sequence[RuntimeAssumption]:
        out = [RuntimeAssumption(
            name="adjustment-policy",
            statement=(f"Prices arrive {self.adjustment_policy.value.replace('_', ' ')} "
                       f"from {self.provider}."),
            realized_by="load_prices",
        )]
        if self.is_point_in_time:
            out.append(RuntimeAssumption(
                name="point-in-time",
                statement="Each date shows what was knowable on that date; "
                          "restatements are retained separately.",
                realized_by="load_as_of",
            ))
        return tuple(out)

    @property
    def limitations(self) -> Sequence[RuntimeLimitation]:
        out = []
        if not self.is_point_in_time:
            out.append(RuntimeLimitation(
                name="restated-history",
                statement=("Prices reflect the vendor's current view rather than "
                           "what was knowable at the time. Anything the vendor "
                           "later revised is visible to a backtest before it "
                           "happened."),
                reason=Exclusion.OUT_OF_SCOPE,
            ))
        if self.universe_membership_policy is UniversePolicy.CURRENT_MEMBERS_ONLY:
            out.append(RuntimeLimitation(
                name="survivorship",
                statement=("Only instruments that still exist are included. "
                           "Results are biased upward by exactly the ones that "
                           "did not survive."),
                reason=Exclusion.OUT_OF_SCOPE,
            ))
        if not self.corporate_action_source:
            out.append(RuntimeLimitation(
                name="no-corporate-action-source",
                statement=("No corporate-action feed is declared, so splits and "
                           "spin-offs are whatever the adjusted series already "
                           "contains."),
                reason=Exclusion.UNRESOLVED,
                applicable_unless=("market_data:adjusted_only_is_sufficient",),
            ))
        return tuple(out)

    @property
    def adjusted_only_is_sufficient(self) -> bool:
        return self.adjustment_policy is AdjustmentPolicy.ADJUSTED_ONLY


@dataclass(frozen=True)
class RealizedData:
    """What the run actually received. Not part of runtime identity.

    A vendor restatement changes this and leaves the runtime untouched, which is
    what makes "same runtime, different realized snapshot" a visible revision
    event rather than a policy change nobody made.
    """

    snapshot_hash: str
    retrieved_at: str
    vendor_revision: str = ""
    partitions: Sequence[str] = ()
    coverage_received: str = ""
    missing: Sequence[str] = ()

    def to_json(self) -> Dict[str, Any]:
        return {"snapshot_hash": self.snapshot_hash,
                "retrieved_at": self.retrieved_at,
                "vendor_revision": self.vendor_revision,
                "partitions": list(self.partitions),
                "coverage_received": self.coverage_received,
                "missing": list(self.missing)}

    def is_restatement_of(self, other: "RealizedData") -> bool:
        """Same policy, different content — the event worth surfacing."""
        return self.snapshot_hash != other.snapshot_hash


YFINANCE_DAILY = MarketDataRuntime(
    name="yfinance-daily", version=1, provider="yfinance", dataset="daily-ohlcv",
    adjustment_policy=AdjustmentPolicy.ADJUSTED_ONLY,
    point_in_time_policy=PointInTimePolicy.LATEST_RESTATED,
    session_alignment="nyse",
    universe_membership_policy=UniversePolicy.CURRENT_MEMBERS_ONLY,
    title="Yahoo Finance daily bars, adjusted, restated",
)

IMPLEMENTED = ("load_prices",)
