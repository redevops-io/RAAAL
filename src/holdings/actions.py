"""Corporate actions as typed events, and resolution as a record.

A split multiplier is enough for the NVDA case and wrong as a general answer.
A merger pays cash and stock, a spinoff creates a second instrument, a reverse
split leaves a fraction that is paid out rather than held, a symbol change
renames the thing a lot points at, and a return of capital reduces basis
without being income. None of those is a number.

So resolving a lot produces a `ResolvedLotQuantity` — what it was, what it is
now, which actions were applied, under which snapshot, and what fell out as
cash. A caller that wants only the quantity can read one field; a caller
reconciling against a statement can read the rest.

**Unimplemented action kinds refuse.** Declaring `MERGER` and then treating it
as a no-op would produce a position that silently ignores what happened to the
company. A named refusal is a worse user experience and a much better answer.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional, Sequence, Tuple


class ActionKind(str, Enum):
    SPLIT = "split"
    """Forward or reverse. A reverse split is a ratio below one."""

    SYMBOL_CHANGE = "symbol_change"
    MERGER = "merger"
    SPINOFF = "spinoff"
    RETURN_OF_CAPITAL = "return_of_capital"
    CANCELLED = "cancelled"


#: What the resolver can currently apply. Everything else refuses rather than
#: being ignored — an unhandled merger is a holding in a company that no
#: longer exists, reported as though nothing happened.
SUPPORTED = frozenset({ActionKind.SPLIT})


class UnsupportedCorporateAction(NotImplementedError):
    """An action this build cannot resolve, named rather than skipped."""

    def __init__(self, action: "CorporateAction"):
        super().__init__(
            f"{action.kind.value} on {action.instrument_id} at "
            f"{action.effective_on} is not modelled. Resolving the lot would "
            f"report a quantity that ignores it.")
        self.action = action


@dataclass(frozen=True)
class CorporateAction:
    instrument_id: str
    kind: ActionKind
    effective_on: dt.date
    #: For a split: new shares per old share. 10 for a 10-for-1, and
    #: Decimal("0.3333333333") for a 1-for-3 reverse.
    ratio: Optional[Decimal] = None
    detail: str = ""


@dataclass(frozen=True)
class ResolvedLotQuantity:
    """What a recorded quantity means today, and how it got there."""

    lot_id: str
    instrument_id: str
    as_traded_quantity: Decimal
    current_quantity: Decimal
    actions_applied: Tuple[CorporateAction, ...]
    snapshot_id: str
    #: Fractional shares a reverse split could not deliver. Held separately
    #: because they are not shares — they were paid out.
    residual_fraction: Decimal = Decimal(0)
    cash_in_lieu: Decimal = Decimal(0)
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def unchanged(self) -> bool:
        return not self.actions_applied


@dataclass(frozen=True)
class ActionSnapshot:
    """A pinned set of corporate actions.

    Pinned because two runs over the same transactions under different action
    sets are not comparable, and the difference is invisible in the result.
    """

    snapshot_id: str
    actions: Tuple[CorporateAction, ...]

    def after(self, instrument_id: str, when: dt.date) -> Tuple[CorporateAction, ...]:
        """Actions strictly after a date, in order.

        Strictly: a price on the effective date is already expressed in the
        new units, and a lot acquired that day is too. Including it applies
        the ratio twice.
        """
        return tuple(sorted(
            (a for a in self.actions
             if a.instrument_id == instrument_id and a.effective_on > when),
            key=lambda a: a.effective_on))


def resolve(lot, snapshot: ActionSnapshot, *,
            whole_shares_only: bool = False,
            price_on_action: Optional[Decimal] = None) -> ResolvedLotQuantity:
    """Carry a recorded quantity forward through every later action.

    `lot` is anything with `lot_id`, `instrument_id`, `acquired_at` and
    `as_traded_quantity` — the ledger's `AcquisitionLot`, but the resolver does
    not need to know that.

    `whole_shares_only` models the reverse-split case where a fraction cannot
    be held and is paid out. It is off by default because most brokers hold
    fractional ETF shares quite happily, and assuming otherwise would invent a
    cash payment nobody received.
    """
    applied = snapshot.after(lot.instrument_id, lot.acquired_at)

    unsupported = [a for a in applied if a.kind not in SUPPORTED]
    if unsupported:
        raise UnsupportedCorporateAction(unsupported[0])

    quantity = Decimal(lot.as_traded_quantity)
    for action in applied:
        if action.ratio is None:
            raise UnsupportedCorporateAction(action)
        quantity = quantity * Decimal(action.ratio)

    residual = Decimal(0)
    cash = Decimal(0)
    notes: list[str] = []
    if whole_shares_only and quantity != quantity.to_integral_value():
        whole = quantity.to_integral_value(rounding="ROUND_FLOOR")
        residual = quantity - whole
        quantity = whole
        if price_on_action is not None:
            cash = (residual * Decimal(price_on_action))
        notes.append(
            "a reverse split left a fraction that cannot be held; it is "
            "recorded as cash in lieu rather than rounded into the position")

    return ResolvedLotQuantity(
        lot_id=lot.lot_id,
        instrument_id=lot.instrument_id,
        as_traded_quantity=Decimal(lot.as_traded_quantity),
        current_quantity=quantity,
        actions_applied=applied,
        snapshot_id=snapshot.snapshot_id,
        residual_fraction=residual,
        cash_in_lieu=cash,
        notes=tuple(notes),
    )


def from_split_table(instrument_id: str, splits, snapshot_id: str) -> ActionSnapshot:
    """Build a snapshot from the vendor's split series.

    Bridges `market_data.ingest`, which captures what the vendor reported, to
    the typed events the resolver consumes. The vendor gives ratios; the type
    is what makes an unhandled merger refuse rather than pass through.
    """
    actions = []
    # `splits or {}` is wrong for a pandas Series: truthiness raises rather
    # than being falsy, and an empty Series is not None.
    entries = [] if splits is None or len(splits) == 0 else list(splits.items())
    for stamp, ratio in entries:
        effective = stamp.date() if hasattr(stamp, "date") else stamp
        actions.append(CorporateAction(
            instrument_id=instrument_id,
            kind=ActionKind.SPLIT,
            effective_on=effective,
            ratio=Decimal(str(ratio)),
        ))
    return ActionSnapshot(snapshot_id=snapshot_id, actions=tuple(actions))
