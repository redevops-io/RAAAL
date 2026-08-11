"""RSU vesting — the first template, chosen because it is the demanding one.

It exercises nearly everything the generic model claims to support: dated grants
that are not purchases, statutory withholding, blackout windows that defer a
sale rather than cancel it, proceeds that have to go somewhere, and a comparison
between holding company stock and diversifying that only means anything if both
sides receive the identical vest schedule.

Two things are modelled carefully because getting them wrong is expensive:

**Withholding is a share reduction, not tax modelling.** The employer withholds
shares at vest to cover supplemental wage withholding; the withheld shares never
reach the account. That is mechanical and this template does it. What happens to
*subsequent* gains on the shares you keep is capital-gains treatment, which
depends on a jurisdiction, account type and lot method nobody has stated — so it
is declared as not modelled rather than assumed.

**The 22% default under-withholds for most people it applies to.** The statutory
supplemental rate is a withholding rate, not a tax rate, and someone whose
marginal rate is 32% or 35% will owe the difference at filing. Recorded as a risk
on the assumption rather than left for the user to discover in April.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ..accounting import CashFlow, Grant, Order
from .base import (
    InputKind,
    MissionTemplate,
    TemplateAssumption,
    TemplateCitation,
    TemplateInput,
    TemplateLimitation,
)

#: Statutory US supplemental-wage withholding. A default, cited, and confirmable
#: — not a constant buried in a function.
SUPPLEMENTAL_RATE = 0.22
SUPPLEMENTAL_RATE_HIGH = 0.37
SUPPLEMENTAL_THRESHOLD = 1_000_000.0

RSU_TEMPLATE = MissionTemplate(
    name="rsu-vesting",
    version=1,
    title="RSU vesting",
    question="What would have happened to my vested equity under this plan?",
    inputs=(
        TemplateInput("ticker", InputKind.TICKER, "Company ticker",
                      why_it_matters="Everything is priced from this series."),
        TemplateInput("vest_dates", InputKind.DATE_LIST, "Vest dates",
                      why_it_matters="Each is a separate delivery at that day's "
                                     "price, so the schedule drives the result."),
        TemplateInput("shares_per_vest", InputKind.SHARES, "Shares per vest",
                      unit="shares"),
        TemplateInput("withholding_rate", InputKind.RATE,
                      "Share withholding rate at vest", unit="fraction of 1",
                      default=SUPPLEMENTAL_RATE,
                      why_it_matters="Withheld shares never reach the account. "
                                     "The statutory 22% is a withholding rate, "
                                     "not your tax rate."),
        TemplateInput("blackout_windows", InputKind.WINDOW_LIST,
                      "Blackout windows (start, end)", required=False, default=[],
                      why_it_matters="A vest landing inside one cannot be sold "
                                     "until the window closes, which moves the "
                                     "sale price."),
        TemplateInput("disposition", InputKind.CHOICE, "What happens to net shares",
                      choices=("hold", "sell_all_and_diversify",
                               "sell_half_and_diversify"),
                      default="hold"),
        TemplateInput("diversify_into", InputKind.TICKER,
                      "Where sale proceeds go", required=False, default="SPY",
                      why_it_matters="Proceeds have to go somewhere; leaving "
                                     "them in cash is also a choice."),
    ),
    assumptions=(
        TemplateAssumption(
            name="withholding-is-share-reduction",
            statement="Shares withheld at vest never enter the account. The "
                      "delivery is net of withholding.",
            realized_by="net_shares",
            citation="irs-pub-15-supplemental",
        ),
        TemplateAssumption(
            name="statutory-supplemental-rate",
            statement=f"Withholding defaults to {SUPPLEMENTAL_RATE:.0%} of the "
                      f"vesting value, the US flat supplemental rate up to "
                      f"${SUPPLEMENTAL_THRESHOLD:,.0f} of supplemental wages in a "
                      f"calendar year ({SUPPLEMENTAL_RATE_HIGH:.0%} above it).",
            realized_by="withholding_for",
            risk="This is a withholding rate, not a tax rate. Anyone whose "
                 "marginal rate exceeds it under-withholds and owes the "
                 "difference at filing — a common and expensive surprise that "
                 "this simulation does not show as a cost.",
            citation="irs-pub-15-supplemental",
        ),
        TemplateAssumption(
            name="vest-is-not-a-purchase",
            statement="Vested shares arrive in kind at the vest price. No cash "
                      "is spent and no order is placed.",
            realized_by="grants_for",
        ),
        TemplateAssumption(
            name="blackout-defers-not-cancels",
            statement="A sale that would fall inside a blackout window is "
                      "deferred to the first session after it closes, not "
                      "abandoned.",
            realized_by="next_open_session",
        ),
        TemplateAssumption(
            name="proceeds-are-reinvested-same-session",
            statement="Sale proceeds are invested into the diversification "
                      "target at the next execution opportunity, not held.",
            realized_by="disposition_program",
        ),
    ),
    citations=(
        TemplateCitation(
            identifier="irs-pub-15-supplemental",
            title="IRS Publication 15 (Circular E), Employer's Tax Guide — "
                  "supplemental wages",
            supports="The flat supplemental withholding rate applied at vest, and "
                     "the mandatory higher rate above the annual threshold.",
            url="https://www.irs.gov/publications/p15",
        ),
    ),
    limitations=(
        TemplateLimitation(
            name="no-capital-gains",
            statement="Gains on shares you keep are not taxed in this simulation. "
                      "Capital-gains treatment depends on a jurisdiction, account "
                      "type, holding period and lot method that have not been "
                      "stated, and assuming one would be worse than reporting "
                      "pre-tax and saying so.",
        ),
        TemplateLimitation(
            name="no-payroll-or-state-tax",
            statement="Social Security, Medicare, and state or local withholding "
                      "are not modelled. Actual net shares are usually fewer than "
                      "shown here.",
        ),
        TemplateLimitation(
            name="no-price-impact",
            statement="Sales execute at the session price with a flat cost. A "
                      "large position sold into a thin market would not.",
        ),
        TemplateLimitation(
            name="no-10b5-1-plan",
            statement="Scheduled trading plans that permit sales during a "
                      "blackout are not modelled; every sale here waits for an "
                      "open window.",
        ),
    ),
)


def withholding_for(vest_value: float, *, rate: float = SUPPLEMENTAL_RATE,
                    cumulative_supplemental: float = 0.0) -> float:
    """Withholding on one vest, applying the higher rate above the threshold.

    The threshold is cumulative across the calendar year, so a vest is split when
    it straddles it. Applying one rate to the whole vest is wrong in the
    direction that flatters the result.
    """
    below = max(0.0, min(vest_value, SUPPLEMENTAL_THRESHOLD - cumulative_supplemental))
    above = vest_value - below
    return below * rate + above * SUPPLEMENTAL_RATE_HIGH


def net_shares(shares: float, price: float, *, rate: float = SUPPLEMENTAL_RATE,
               cumulative_supplemental: float = 0.0) -> float:
    """Shares that actually reach the account."""
    value = shares * price
    withheld_value = withholding_for(value, rate=rate,
                                     cumulative_supplemental=cumulative_supplemental)
    return shares * (1.0 - withheld_value / value) if value > 0 else 0.0


def next_open_session(session: pd.Timestamp, sessions: pd.DatetimeIndex,
                      blackouts: Sequence[tuple]) -> Optional[pd.Timestamp]:
    """First session on or after `session` that is not inside a blackout window.

    Deferral, not cancellation: an unsold vest is still owned, and dropping the
    sale would silently convert a diversification plan into a hold.
    """
    candidates = sessions[sessions >= session]
    for candidate in candidates:
        inside = any(pd.Timestamp(start) <= candidate <= pd.Timestamp(end)
                     for start, end in blackouts)
        if not inside:
            return candidate
    return None


def grants_for(values: Mapping[str, Any], prices: pd.DataFrame) -> List[Grant]:
    """Net share deliveries, one per vest date."""
    ticker = values["ticker"]
    rate = float(values.get("withholding_rate") or SUPPLEMENTAL_RATE)
    per_vest = float(values["shares_per_vest"])

    grants: List[Grant] = []
    cumulative: Dict[int, float] = {}
    for raw in values["vest_dates"]:
        date = pd.Timestamp(raw)
        later = prices.index[prices.index >= date]
        if not len(later) or ticker not in prices.columns:
            continue
        session = later[0]
        price = float(prices.at[session, ticker])
        value = per_vest * price
        year = session.year
        net = net_shares(per_vest, price, rate=rate,
                         cumulative_supplemental=cumulative.get(year, 0.0))
        cumulative[year] = cumulative.get(year, 0.0) + value
        grants.append(Grant(date=session, ticker=ticker, shares=net,
                            reason=f"vest, net of {rate:.0%} withholding"))
    return grants


def disposition_program(values: Mapping[str, Any], sessions: pd.DatetimeIndex):
    """The event program: what happens to net shares once they arrive.

    `hold` is a real strategy and the default, because it is what happens when
    nobody decides. Naming it makes the alternative a choice rather than an
    intervention.
    """
    disposition = values.get("disposition", "hold")
    ticker = values["ticker"]
    target = values.get("diversify_into") or "SPY"
    blackouts = [tuple(w) for w in (values.get("blackout_windows") or [])]
    fraction = {"hold": 0.0, "sell_all_and_diversify": 1.0,
                "sell_half_and_diversify": 0.5}[disposition]

    sold: Dict[str, bool] = {}

    def program(session, visible, holdings, cash):
        orders: List[Order] = []

        if cash > 1e-9 and target in visible.columns:
            orders.append(Order(session, target, cash, reason="reinvest proceeds"))

        held = holdings.get(ticker, 0.0)
        if fraction > 0 and held > 1e-9:
            key = str(session.date())
            if not sold.get(key):
                when = next_open_session(session, sessions, blackouts)
                if when == session:
                    price = float(visible.iloc[-1].get(ticker, float("nan")))
                    if price == price and price > 0:
                        orders.append(Order(
                            session, ticker, -held * fraction * price,
                            reason=f"{disposition.replace('_', ' ')}"))
                        sold[key] = True
        return orders

    return program


#: Behaviours this module actually implements, for the declaration verifier.
IMPLEMENTED = ("net_shares", "withholding_for", "grants_for",
               "next_open_session", "disposition_program")
