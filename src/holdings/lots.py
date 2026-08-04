"""Immutable acquisition lots, and the projections over them.

**A position is not an object.** It is a time- and policy-dependent projection
over immutable lots, corporate actions, prices and dispositions. Storing a
share count as authoritative destroys the only record of what a person
actually did, to save a multiplication — and a single quantity cannot express
multiple acquisition dates, multiple cost bases, a split between two
purchases, a partial sale, a holding period, or which lot was sold.

Four projections, deliberately separate so that none is reused because it
happens to contain a number called `price`:

    statement    what was bought, in the units of that day
    current      how many present-day shares that lot represents
    valuation    current units x market price
    performance  total-return series, or explicit distributions

The disposition mechanics here are lot identity and allocation only. FIFO,
LIFO, highest-cost, specific identification, holding-period classification and
wash sales are policies *over* this, and building them before the mechanics
would bake one policy into the ledger.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

from .actions import ActionSnapshot, ResolvedLotQuantity, resolve
from .prices import PricePurpose, PriceSeries


class AcquisitionKind(str, Enum):
    PURCHASE = "purchase"
    DIVIDEND_REINVESTMENT = "dividend_reinvestment"
    RSU_VEST = "rsu_vest"
    TRANSFER_IN = "transfer_in"
    SPINOFF = "spinoff"
    GIFT = "gift"
    INHERITANCE = "inheritance"


@dataclass(frozen=True)
class AcquisitionLot:
    """What a statement said, preserved exactly as it said it.

    `as_traded_quantity` and `as_traded_unit_price` are in the share units of
    `acquired_at`. They are never rewritten: 10 NVDA at $1,209 stays 10 at
    $1,209 after the 10-for-1, and the 100 shares held today are derived.

    There is deliberately no `current_quantity` field. A stored one would be a
    second authority that disagrees with the corporate-action snapshot the
    moment either changes, and the disagreement would be silent.
    """

    owner: str
    account_id: str
    lot_id: str
    instrument_id: str

    acquired_at: dt.date
    as_traded_quantity: Decimal
    as_traded_unit_price: Decimal
    acquisition_cost: Decimal
    fees: Decimal = Decimal(0)

    acquisition_kind: AcquisitionKind = AcquisitionKind.PURCHASE
    source_ref: str = ""
    corporate_action_snapshot: str = ""

    def statement_view(self) -> Mapping[str, object]:
        """What the user's own records say. The number they remember."""
        return {
            "acquired_at": self.acquired_at,
            "quantity": self.as_traded_quantity,
            "unit_price": self.as_traded_unit_price,
            "cost": self.acquisition_cost,
            "fees": self.fees,
            "kind": self.acquisition_kind.value,
        }

    def current(self, snapshot: ActionSnapshot, **kwargs) -> ResolvedLotQuantity:
        """How many of today's shares this lot represents, and why."""
        return resolve(self, snapshot, **kwargs)


class LotLedgerError(RuntimeError):
    pass


class OverAllocated(LotLedgerError):
    """A disposition consumed more of a lot than it holds."""


@dataclass(frozen=True)
class DispositionLotAllocation:
    disposition_id: str
    lot_id: str
    #: In current units, matching the resolved quantity rather than the
    #: as-traded one. A sale happens today, in today's shares.
    quantity_consumed: Decimal
    basis_used: Decimal
    proceeds: Decimal
    fees: Decimal = Decimal(0)

    @property
    def realized(self) -> Decimal:
        return self.proceeds - self.basis_used - self.fees


@dataclass(frozen=True)
class Disposition:
    disposition_id: str
    owner: str
    account_id: str
    instrument_id: str
    disposed_at: dt.date
    allocations: Tuple[DispositionLotAllocation, ...] = field(default_factory=tuple)

    @property
    def quantity(self) -> Decimal:
        return sum((a.quantity_consumed for a in self.allocations), Decimal(0))

    @property
    def realized(self) -> Decimal:
        return sum((a.realized for a in self.allocations), Decimal(0))


@dataclass(frozen=True)
class PositionView:
    """A projection, valid only for the pins it names.

    Two runs over identical transactions under different corporate-action
    snapshots, price snapshots or distribution treatments are not comparable,
    and nothing in a bare number would say so.
    """

    owner: str
    account_id: str
    instrument_id: str
    as_of: dt.date
    current_quantity: Decimal
    market_value: Optional[Decimal]
    cost_basis: Decimal
    #: Every pin a comparison depends on.
    price_snapshot: str
    action_snapshot: str
    price_purpose: str
    lot_selection_policy: str
    distribution_treatment: str
    lots: Tuple[str, ...] = field(default_factory=tuple)
    cash_in_lieu: Decimal = Decimal(0)

    @property
    def unrealized(self) -> Optional[Decimal]:
        if self.market_value is None:
            return None
        return self.market_value - self.cost_basis


def remaining_quantity(lot: AcquisitionLot, snapshot: ActionSnapshot,
                       allocations: Sequence[DispositionLotAllocation]) -> Decimal:
    """Current units of a lot that have not been sold."""
    resolved = lot.current(snapshot)
    consumed = sum((a.quantity_consumed for a in allocations
                    if a.lot_id == lot.lot_id), Decimal(0))
    if consumed > resolved.current_quantity:
        raise OverAllocated(
            f"{lot.lot_id}: {consumed} consumed of {resolved.current_quantity} "
            f"current units")
    return resolved.current_quantity - consumed


def value_position(lots: Sequence[AcquisitionLot], *, snapshot: ActionSnapshot,
                   prices: PriceSeries, as_of: dt.date,
                   allocations: Sequence[DispositionLotAllocation] = (),
                   lot_selection_policy: str = "none",
                   distribution_treatment: str = "excluded") -> PositionView:
    """Current units times market price.

    Refuses a total-return series by name. That is the whole reason the series
    are typed: passing the wrong one produces a value that is wrong by every
    distribution the instrument has ever paid, and looks entirely ordinary.
    """
    prices.require(PricePurpose.MARKET, "value_position")

    if not lots:
        raise LotLedgerError("no lots to value")
    owners = {l.owner for l in lots}
    accounts = {l.account_id for l in lots}
    instruments = {l.instrument_id for l in lots}
    if len(owners) > 1 or len(accounts) > 1 or len(instruments) > 1:
        raise LotLedgerError(
            "a position is one owner, one account and one instrument; "
            f"got {sorted(owners)}, {sorted(accounts)}, {sorted(instruments)}")

    quantity = Decimal(0)
    basis = Decimal(0)
    cash = Decimal(0)
    for lot in lots:
        if lot.acquired_at > as_of:
            continue
        held = remaining_quantity(lot, snapshot, allocations)
        resolved = lot.current(snapshot)
        quantity += held
        cash += resolved.cash_in_lieu
        if resolved.current_quantity:
            # Basis follows the units still held, not the units bought.
            basis += (lot.acquisition_cost + lot.fees) * (
                held / resolved.current_quantity)

    instrument = next(iter(instruments))
    unit = prices.price(instrument, as_of)
    value = (quantity * unit) if unit is not None else None

    return PositionView(
        owner=next(iter(owners)),
        account_id=next(iter(accounts)),
        instrument_id=instrument,
        as_of=as_of,
        current_quantity=quantity,
        market_value=value,
        cost_basis=basis,
        price_snapshot=prices.snapshot_id,
        action_snapshot=snapshot.snapshot_id,
        price_purpose=prices.purpose.value,
        lot_selection_policy=lot_selection_policy,
        distribution_treatment=distribution_treatment,
        lots=tuple(l.lot_id for l in lots),
        cash_in_lieu=cash,
    )
