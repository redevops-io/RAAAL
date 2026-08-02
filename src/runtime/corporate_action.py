"""What happened to a grant's identity and quantity between grant and vest.

    granted quantity + original symbol
        -> eligible actions through the vest date
        -> resolved identity
        -> adjusted gross quantity
        -> withholding
        -> delivered shares

Deliberately not a corporate-actions research engine. It answers one question,
and the question is the one a vest cannot be computed without.

**Adjustment precedes withholding.** A hundred granted shares through a
two-for-one split are two hundred at vest, and withholding applies to two
hundred. The wrong order is not always visible — a plain percentage commutes
with a split and gives the same answer — so the permanent test uses whole-share
withholding, where 101 granted becomes 157 delivered in the right order and 156
in the wrong one.

**Policy and history are separate**, as they are for market data:

    CorporateActionRuntime   how actions are interpreted
    RealizedCorporateActions which actions this run actually received

A vendor correction produces a new realized snapshot and a restatement. It never
edits the runtime and never rewrites a run that has already been pinned.

Ratios are exact. A three-for-two split held as 1.5 in binary floating point
turns a share count nobody can reproduce into one nobody can check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import (
    Exclusion,
    RuntimeArtifact,
    RuntimeAssumption,
    RuntimeLimitation,
)


class CorporateActionKind(str, Enum):
    SPLIT = "SPLIT"
    REVERSE_SPLIT = "REVERSE_SPLIT"
    SYMBOL_CHANGE = "SYMBOL_CHANGE"
    MERGER_STOCK_FOR_STOCK = "MERGER_STOCK_FOR_STOCK"
    MERGER_CASH_ONLY = "MERGER_CASH_ONLY"
    MERGER_MIXED = "MERGER_MIXED"
    GRANT_CANCELLED = "GRANT_CANCELLED"
    GRANT_REPLACED = "GRANT_REPLACED"
    SPINOFF = "SPINOFF"
    SPECIAL_DIVIDEND = "SPECIAL_DIVIDEND"
    RIGHTS_ISSUE = "RIGHTS_ISSUE"


class FractionalPolicy(str, Enum):
    RETAIN = "RETAIN"
    CASH_IN_LIEU = "CASH_IN_LIEU"
    ROUND_DOWN = "ROUND_DOWN"
    UNRESOLVED = "UNRESOLVED"
    """Not stated. A reverse split leaving a fraction is refused rather than
    rounded, because rounding in either direction is a decision nobody made."""


class UnsupportedCorporateAction(ValueError):
    """An action this runtime does not interpret. Blocks before arithmetic."""


class UnresolvedCorporateAction(ValueError):
    """An action that is supported and cannot be applied from what it carries."""


class GrantCancelled(ValueError):
    """The grant was cancelled. There is no vest to compute."""


#: Actions the first implementation resolves, each with the mechanism that does
#: it. `test_corporate_action.py` walks `CorporateActionKind` itself and
#: requires every member to appear here or in `UNSUPPORTED` — a new member fails
#: until somebody classifies it.
SUPPORTED: Mapping[CorporateActionKind, str] = {
    CorporateActionKind.SPLIT: "apply_split",
    CorporateActionKind.REVERSE_SPLIT: "apply_reverse_split",
    CorporateActionKind.SYMBOL_CHANGE: "apply_symbol_change",
    CorporateActionKind.MERGER_STOCK_FOR_STOCK: "apply_stock_conversion",
    CorporateActionKind.GRANT_CANCELLED: "apply_cancellation",
    CorporateActionKind.GRANT_REPLACED: "apply_replacement",
}

#: Blocked, with the reason. Each would need tax or cash-flow treatment this
#: system does not yet model, and approximating one silently changes what a
#: person owns.
UNSUPPORTED: Mapping[CorporateActionKind, str] = {
    CorporateActionKind.MERGER_CASH_ONLY: (
        "a cash-only merger converts the position to cash and realises a gain, "
        "and neither the proceeds treatment nor the tax consequence is modelled"),
    CorporateActionKind.MERGER_MIXED: (
        "mixed consideration splits into stock and cash on terms this runtime "
        "cannot read, and guessing the split changes what is owned"),
    CorporateActionKind.SPINOFF: (
        "a spin-off creates a second holding with its own cost basis, which is "
        "not modelled"),
    CorporateActionKind.SPECIAL_DIVIDEND: (
        "a special dividend is a cash event with its own tax treatment, not a "
        "change to the grant's quantity"),
    CorporateActionKind.RIGHTS_ISSUE: (
        "a rights issue is an optional purchase, and whether it was taken up "
        "is a decision this runtime cannot infer"),
}


@dataclass(frozen=True)
class CorporateActionEvent:
    """One action, as reported by a source at a point in time."""

    issuer_ref: str
    effective_date: str
    kind: CorporateActionKind
    source_ref: str
    observed_at: str = ""
    old_symbol: Optional[str] = None
    new_symbol: Optional[str] = None
    ratio_numerator: Optional[Decimal] = None
    ratio_denominator: Optional[Decimal] = None
    cash_component: Optional[Decimal] = None
    replacement_security: Optional[str] = None
    replacement_grant_ref: Optional[str] = None

    @property
    def ratio(self) -> Optional[Decimal]:
        """Exact. Held as a float, a three-for-two split produces a share count
        nobody can reproduce."""
        if self.ratio_numerator is None or self.ratio_denominator is None:
            return None
        if self.ratio_denominator == 0:
            return None
        return Decimal(self.ratio_numerator) / Decimal(self.ratio_denominator)

    def to_json(self) -> Dict[str, Any]:
        return {"issuer_ref": self.issuer_ref,
                "effective_date": self.effective_date, "kind": self.kind.value,
                "source_ref": self.source_ref, "observed_at": self.observed_at,
                "old_symbol": self.old_symbol, "new_symbol": self.new_symbol,
                "ratio_numerator": (str(self.ratio_numerator)
                                    if self.ratio_numerator is not None else None),
                "ratio_denominator": (str(self.ratio_denominator)
                                      if self.ratio_denominator is not None
                                      else None),
                "cash_component": (str(self.cash_component)
                                   if self.cash_component is not None else None),
                "replacement_security": self.replacement_security,
                "replacement_grant_ref": self.replacement_grant_ref}


@dataclass(frozen=True)
class RealizedCorporateActions:
    """Which actions this run received, and under which snapshot.

    Separate from the runtime for the same reason realized market data is: the
    policy says how a split is interpreted, and this says which splits were
    known. A vendor correction changes the second and not the first.
    """

    snapshot_ref: str
    events: Sequence[CorporateActionEvent] = ()
    restates: Optional[str] = None
    """The snapshot this one corrects. A restatement is a new record that names
    what it supersedes, never an edit to the earlier one."""

    def through(self, issuer_ref: str, as_of: str
                ) -> List[CorporateActionEvent]:
        """Actions effective on or before a date, in order.

        Point-in-time by construction: a run pinned to a snapshot sees what that
        snapshot held, so a later correction cannot silently move an old result.
        """
        return sorted(
            (one for one in self.events
             if one.issuer_ref == issuer_ref and one.effective_date <= as_of),
            key=lambda one: one.effective_date)

    def to_json(self) -> Dict[str, Any]:
        return {"snapshot_ref": self.snapshot_ref, "restates": self.restates,
                "events": [one.to_json() for one in self.events]}


@dataclass(frozen=True)
class ResolvedGrant:
    """The grant's identity and quantity at the vest date."""

    grant_ref: str
    symbol: str
    gross_shares: Decimal
    fractional_shares: Decimal = Decimal(0)
    cash_in_lieu: Decimal = Decimal(0)
    applied: Sequence[str] = ()
    cancelled: bool = False
    replaced_by: Optional[str] = None

    @property
    def vests(self) -> bool:
        return not self.cancelled

    def to_json(self) -> Dict[str, Any]:
        return {"grant_ref": self.grant_ref, "symbol": self.symbol,
                "gross_shares": str(self.gross_shares),
                "fractional_shares": str(self.fractional_shares),
                "cash_in_lieu": str(self.cash_in_lieu),
                "applied": list(self.applied), "cancelled": self.cancelled,
                "replaced_by": self.replaced_by, "vests": self.vests}


@dataclass(frozen=True)
class CorporateActionRuntime(RuntimeArtifact):
    """How actions are interpreted. Not which ones happened."""

    kind: ClassVar[str] = "corporate_action"

    name: str
    version: int
    source_policy: str = ""
    point_in_time_policy: str = ""
    fractional_policy: FractionalPolicy = FractionalPolicy.UNRESOLVED
    title: str = ""

    @property
    def supported_actions(self) -> Tuple[CorporateActionKind, ...]:
        return tuple(SUPPORTED)

    @property
    def unsupported_actions(self) -> Tuple[CorporateActionKind, ...]:
        return tuple(UNSUPPORTED)

    def declared_form(self) -> Dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "version": self.version,
                "source_policy": self.source_policy,
                "point_in_time_policy": self.point_in_time_policy,
                "fractional_policy": self.fractional_policy.value,
                "supported_actions": [one.value for one in self.supported_actions],
                "unsupported_actions": [one.value
                                        for one in self.unsupported_actions],
                "title": self.title}

    def comparable_form(self) -> Dict[str, Any]:
        declared = self.declared_form()
        for prose in ("title", "name", "version"):
            declared.pop(prose, None)
        return declared

    @property
    def assumptions(self) -> Sequence[RuntimeAssumption]:
        return (
            RuntimeAssumption(
                name="adjustment-before-withholding",
                statement=("Granted quantities are adjusted for splits and "
                           "conversions before withholding is applied."),
                realized_by="resolve_grant",
                risk=("Withheld first, a whole-share calculation on 101 "
                      "granted shares through a two-for-one split delivers 156 "
                      "instead of 157 — plausible, and wrong."),
            ),
            RuntimeAssumption(
                name="identity-continuity",
                statement=("A symbol change carries the grant forward. It is "
                           "not a sale, a purchase or a new grant."),
                realized_by="apply_symbol_change",
            ),
            RuntimeAssumption(
                name="point-in-time-actions",
                statement=("A run sees only actions known under its pinned "
                           "snapshot; later corrections arrive as restatements."),
                realized_by="through",
            ),
            RuntimeAssumption(
                name="cancelled-grants-do-not-vest",
                statement="A cancelled grant produces no vest and no shares.",
                realized_by="apply_cancellation",
            ),
        )

    @property
    def limitations(self) -> Sequence[RuntimeLimitation]:
        return tuple(
            RuntimeLimitation(name=f"unsupported-{one.value.lower()}",
                              statement=reason,
                              reason=Exclusion.OUT_OF_SCOPE)
            for one, reason in UNSUPPORTED.items())


# --- mechanisms ------------------------------------------------------------


def apply_split(shares: Decimal, event: CorporateActionEvent) -> Decimal:
    if event.ratio is None:
        raise UnresolvedCorporateAction(
            f"split on {event.effective_date} states no ratio, so the adjusted "
            "quantity cannot be computed")
    return shares * event.ratio


def apply_reverse_split(shares: Decimal, event: CorporateActionEvent
                        ) -> Decimal:
    if event.ratio is None:
        raise UnresolvedCorporateAction(
            f"reverse split on {event.effective_date} states no ratio")
    return shares * event.ratio


def apply_symbol_change(symbol: str, event: CorporateActionEvent) -> str:
    if not event.new_symbol:
        raise UnresolvedCorporateAction(
            f"symbol change on {event.effective_date} names no new symbol")
    return event.new_symbol


def apply_stock_conversion(shares: Decimal, symbol: str,
                           event: CorporateActionEvent) -> Tuple[Decimal, str]:
    if event.ratio is None or not event.replacement_security:
        raise UnresolvedCorporateAction(
            f"stock-for-stock merger on {event.effective_date} needs both a "
            "conversion ratio and the security received")
    if event.cash_component:
        raise UnsupportedCorporateAction(
            UNSUPPORTED[CorporateActionKind.MERGER_MIXED])
    return shares * event.ratio, event.replacement_security


def apply_cancellation(event: CorporateActionEvent) -> None:
    raise GrantCancelled(
        f"the grant was cancelled effective {event.effective_date}")


def apply_replacement(event: CorporateActionEvent) -> str:
    if not event.replacement_grant_ref:
        raise UnresolvedCorporateAction(
            f"replacement on {event.effective_date} names no replacement grant")
    return event.replacement_grant_ref


def _split_fraction(shares: Decimal, policy: FractionalPolicy
                    ) -> Tuple[Decimal, Decimal]:
    """Whole shares and the fraction left over, per the declared policy."""
    whole = shares.to_integral_value(rounding="ROUND_FLOOR")
    fraction = shares - whole
    if fraction == 0:
        return shares, Decimal(0)
    if policy is FractionalPolicy.RETAIN:
        return shares, Decimal(0)
    if policy in (FractionalPolicy.ROUND_DOWN, FractionalPolicy.CASH_IN_LIEU):
        return whole, fraction
    raise UnresolvedCorporateAction(
        f"this leaves {fraction} of a share and no fractional policy is "
        "declared. Rounding in either direction is a decision nobody made")


def resolve_grant(*, grant_ref: str, granted_shares: Decimal, symbol: str,
                  issuer_ref: str, vest_date: str,
                  realized: Optional[RealizedCorporateActions],
                  runtime: CorporateActionRuntime) -> ResolvedGrant:
    """Identity and quantity at the vest date, before any withholding.

    Refuses before arithmetic when the action history is absent or contains
    something this runtime does not interpret. A quantity computed from a
    partial history is a confident number derived from an unknown.
    """
    if realized is None:
        raise UnresolvedCorporateAction(
            f"no corporate-action history is pinned for {issuer_ref}, so the "
            "share count between grant and vest is being trusted blindly")

    shares, current = Decimal(granted_shares), symbol
    applied: List[str] = []
    fraction_total, cash = Decimal(0), Decimal(0)

    for event in realized.through(issuer_ref, vest_date):
        if event.kind in UNSUPPORTED:
            raise UnsupportedCorporateAction(
                f"{event.kind.value} on {event.effective_date}: "
                f"{UNSUPPORTED[event.kind]}")

        if event.kind is CorporateActionKind.GRANT_CANCELLED:
            return ResolvedGrant(grant_ref=grant_ref, symbol=current,
                                 gross_shares=Decimal(0), cancelled=True,
                                 applied=tuple(applied + [event.kind.value]))

        if event.kind is CorporateActionKind.GRANT_REPLACED:
            return ResolvedGrant(
                grant_ref=grant_ref, symbol=current, gross_shares=Decimal(0),
                cancelled=True, replaced_by=apply_replacement(event),
                applied=tuple(applied + [event.kind.value]))

        if event.kind in (CorporateActionKind.SPLIT,
                          CorporateActionKind.REVERSE_SPLIT):
            shares = (apply_split(shares, event)
                      if event.kind is CorporateActionKind.SPLIT
                      else apply_reverse_split(shares, event))
            shares, fraction = _split_fraction(shares,
                                               runtime.fractional_policy)
            fraction_total += fraction
            if runtime.fractional_policy is FractionalPolicy.CASH_IN_LIEU:
                cash += fraction
        elif event.kind is CorporateActionKind.SYMBOL_CHANGE:
            current = apply_symbol_change(current, event)
        elif event.kind is CorporateActionKind.MERGER_STOCK_FOR_STOCK:
            shares, current = apply_stock_conversion(shares, current, event)

        applied.append(event.kind.value)

    return ResolvedGrant(grant_ref=grant_ref, symbol=current,
                         gross_shares=shares,
                         fractional_shares=fraction_total, cash_in_lieu=cash,
                         applied=tuple(applied))


#: Mechanisms that exist here. Resolved to real callables by the test suite.
IMPLEMENTED = ("resolve_grant", "apply_split", "apply_reverse_split",
               "apply_symbol_change", "apply_stock_conversion",
               "apply_cancellation", "apply_replacement", "through")


US_CORPORATE_ACTIONS = CorporateActionRuntime(
    name="corporate-action/us-equity", version=1,
    source_policy="vendor-reported, restated by new snapshot",
    point_in_time_policy="actions known as of the pinned data snapshot",
    fractional_policy=FractionalPolicy.CASH_IN_LIEU,
    title="US equity splits, symbol changes and stock-for-stock mergers",
)
