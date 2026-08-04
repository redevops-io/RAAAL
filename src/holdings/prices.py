"""Three price series, and no way to hand one to the wrong caller.

    market        split-adjusted, dividends excluded   -> what a holding is worth
    total return  splits and dividends                 -> what a strategy returned
    as traded     reconstructed trading units          -> what the statement said

Both wrong answers are plausible and neither announces itself. Valuing a
position with total-return prices credits reinvested dividends into the share
price and then credits them again as cash. Measuring a strategy with market
prices drops the dividends entirely. On a decade of VTI the gap is 105.92
against 87.57 for the same session.

So the series are distinct types rather than a column name or a boolean, and
the functions that consume them refuse the wrong one by name. A caller cannot
pass the total-return frame to a valuation by getting an argument order wrong,
and cannot silence the refusal by renaming a column.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Mapping, Optional


class PricePurpose(str, Enum):
    """What a series is *for*. Not how it was computed — what it may answer."""

    MARKET = "market"
    TOTAL_RETURN = "total_return"
    AS_TRADED = "as_traded"


class WrongPriceSeries(TypeError):
    """A series was passed to something it cannot correctly answer.

    A named error rather than a silently plausible number. The two mistakes
    this prevents are each a few tenths of a percent per year, compounding,
    with nothing on the page to suggest anything happened.
    """

    def __init__(self, wanted: PricePurpose, given: PricePurpose, where: str):
        super().__init__(
            f"{where} requires {wanted.value} prices and was given "
            f"{given.value}. "
            + _WHY.get((wanted, given), "These series answer different questions.")
        )
        self.wanted = wanted
        self.given = given


_WHY = {
    (PricePurpose.MARKET, PricePurpose.TOTAL_RETURN):
        "Total-return prices embed reinvested dividends in the share price; "
        "valuing a holding with them counts each distribution twice.",
    (PricePurpose.TOTAL_RETURN, PricePurpose.MARKET):
        "Market prices exclude distributions; measuring a strategy with them "
        "silently drops every dividend it earned.",
    (PricePurpose.MARKET, PricePurpose.AS_TRADED):
        "As-traded prices are in the share units of their own epoch and are "
        "not comparable across a split.",
}


@dataclass(frozen=True)
class PriceSeries:
    """A price series that knows what it may be used for."""

    purpose: PricePurpose
    snapshot_id: str
    #: instrument -> date -> price. Decimal, never float: these values are
    #: multiplied by quantities to produce money.
    by_instrument: Mapping[str, Mapping[dt.date, Decimal]]

    def price(self, instrument: str, on: dt.date) -> Optional[Decimal]:
        series = self.by_instrument.get(instrument)
        if not series:
            return None
        if on in series:
            return series[on]
        # The last settled session at or before the date. A holding is valued
        # on a weekend at Friday's close, not at nothing.
        earlier = [d for d in series if d <= on]
        return series[max(earlier)] if earlier else None

    def require(self, wanted: PricePurpose, where: str) -> "PriceSeries":
        if self.purpose is not wanted:
            raise WrongPriceSeries(wanted, self.purpose, where)
        return self


def market(snapshot_id: str, by_instrument) -> PriceSeries:
    return PriceSeries(PricePurpose.MARKET, snapshot_id, by_instrument)


def total_return(snapshot_id: str, by_instrument) -> PriceSeries:
    return PriceSeries(PricePurpose.TOTAL_RETURN, snapshot_id, by_instrument)


def as_traded(snapshot_id: str, by_instrument) -> PriceSeries:
    return PriceSeries(PricePurpose.AS_TRADED, snapshot_id, by_instrument)
