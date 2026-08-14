"""Well-formed statements to pick from and edit.

The harvest found that real strategy statements routinely omit what to buy —
"I contribute $750/month to my 401k" is a complete thought to the person writing
it, and the runtime cannot run it. Asking people to guess the vocabulary that
gets them a plan is the interaction defect underneath most of the follow-up
burden. A list of statements they can pick and then edit removes the guessing
without removing the typing: the text box stays the primary input, and a picked
sentence lands in it as ordinary editable text.

Two rules hold this together, and both exist because a catalogue is the easiest
place in a product to make a claim nobody checks.

**Everything is offered; what cannot run is refused by name.** The catalogue
used to carry only what executed, on the reasoning that a dropdown entry is the
product claiming "this works". That rule made the product look smaller than it
is and, worse, made its gaps invisible: a strategy nobody can select is a
strategy nobody asks for, and one nobody asks for never gets built.

So all twenty families are here, and the engine answers each one honestly. A
refusal names the dimension and gives the reason — this build only buys, it
computes no tax, it holds one pool rather than buckets — which turns the
catalogue into a map of what is supported and what is not, and turns each
selection into a recorded request for the thing that is missing.

`tests/test_strategy_library.py` no longer requires every entry to run. It
requires every entry to *resolve*: execute, or refuse by name with a reason a
person can act on. Silence is the only failure.

**Refusal happens at the point of the request.** Selecting an entry the
engine cannot run puts the sentence in the box like any other; the refusal
arrives when it is read, naming the dimension and the reason. The catalogue
does not pre-emptively grey anything out — what this build supports is a
property of the engine on the day you ask, not a list maintained beside it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Entry:
    """One statement somebody can pick.

    `text` is what lands in the box, and it is deliberately a whole sentence in
    ordinary words rather than a fill-in-the-blanks form. The runtime reads
    prose; a template with slots would be a second grammar to maintain, and the
    thing the pilot is measuring is whether prose works.
    """

    key: str
    title: str
    text: str
    #: The strategy family this belongs to, as the corpus names it. Carried so
    #: a refusal can be traced back to the family it came from and counted.
    family: str = ""
    #: Where the phrasing came from. Every entry is a strategy people actually
    #: describe, taken from a cited definition rather than invented here — the
    #: same rule the harvested corpus follows, for the same reason.
    source: str = ""


@dataclass(frozen=True)
class Group:
    key: str
    title: str
    #: Why these belong together, in the user's terms rather than the schema's.
    note: str
    entries: Tuple[Entry, ...]


#: The offered catalogue.
#:
#: Grouped by the distinction the engine actually makes — what puts money in,
#: and what it buys — because that is what changes the answer. Grouping by the
#: names the industry uses would be grouping by a distinction this build cannot
#: act on, and every group but one would be empty.
LIBRARY: Tuple[Group, ...] = (
    Group(
        key='money-in',
        title='Putting money in',
        note='Contributions, on a calendar or on a market condition.',
        entries=(
            Entry('scheduled-funding', 'Contribute a fixed amount on a schedule',
                  'I invest $500 into VTI every month, on the same day each '
                  'month.',
                  family='scheduled_funding', source='https://www.investor.gov/introduction-investing/investing-basics/glossary/dollar-cost-averaging'),
            Entry('event-triggered-funding', 'Buy when the market hits a condition',
                  'I buy VOO when SPY falls below its 200-day moving average.',
                  family='event_triggered_funding', source='https://www.investopedia.com/terms/m/movingaverage.asp'),
        ),
    ),
    Group(
        key='allocation',
        title='Choosing and holding the mix',
        note='What the money buys, in what proportions, and whether that is restored over time.',
        entries=(
            Entry('stated-weights', 'Hold a stated split, such as 60/40',
                  'I hold a 60/40 portfolio: 60% stocks and 40% bonds.',
                  family='stated_weights', source='https://www.investor.gov/introduction-investing/investing-basics/glossary/asset-allocation'),
            Entry('rebalancing', 'Rebalance back to the target weights',
                  'I hold 60/40 and rebalance back to those weights once a year.',
                  family='rebalancing', source='https://www.investor.gov/introduction-investing/investing-basics/glossary/rebalancing'),
            Entry('risk-based-allocation', 'Allocate by risk rather than by dollars',
                  'I allocate across my holdings by inverse volatility rather '
                  'than by dollars.',
                  family='risk_based_allocation', source='https://www.investopedia.com/terms/r/risk-parity.asp'),
            Entry('factor-tilt', 'Tilt toward a factor',
                  'I tilt 20% of my portfolio toward small cap value.',
                  family='factor_tilt', source='https://www.investopedia.com/terms/s/smallcap.asp'),
            Entry('glidepath', 'Shift from stocks to bonds as you age',
                  'I shift 1% from stocks to bonds every year as I get older.',
                  family='glidepath', source='https://benchmarkfg.com/wp-content/uploads/2025/05/Reducing-Retirement-Risk-with-a-Rising-Equity-Glide-Path-2.pdf'),
        ),
    ),
    Group(
        key='money-out',
        title='Taking money out',
        note='Withdrawals, income and the order accounts are drawn down in.',
        entries=(
            Entry('safe-withdrawal-rate', 'Withdraw a fixed percentage each year',
                  'I withdraw 4% of the portfolio each year, adjusted for '
                  'inflation.',
                  family='safe_withdrawal_rate', source='https://www.nysdcp.com/rsc-preauth/learn-about-retirement/close-to-or-living-in-retirement/articles/withdrawal-strategies-to-consider-for-retirement'),
            Entry('withdrawal-ordering', 'Draw accounts down in a chosen order',
                  'I spend the taxable account first, then the IRA, then the '
                  'Roth.',
                  family='withdrawal_ordering', source='https://www.nysdcp.com/rsc-preauth/learn-about-retirement/close-to-or-living-in-retirement/articles/withdrawal-strategies-to-consider-for-retirement'),
            Entry('required-minimum-distribution', 'Take required minimum distributions',
                  'I take the required minimum distribution starting at 73.',
                  family='required_minimum_distribution', source='https://www.irs.gov/retirement-plans/retirement-plan-and-ira-required-minimum-distributions-faqs'),
            Entry('annuitisation', 'Annuitise part of the portfolio',
                  'I annuitize a third of the portfolio at 70.',
                  family='annuitisation', source='https://gainbridge.com/post/decumulation-strategy'),
            Entry('dividend-income', 'Live off the dividends',
                  'I live off the dividends and never touch the principal.',
                  family='dividend_income', source='https://www.investopedia.com/terms/d/dividend.asp'),
        ),
    ),
    Group(
        key='accounts',
        title='Accounts and tax',
        note='Where holdings sit, and moves whose whole effect is a tax one.',
        entries=(
            Entry('asset-location', 'Put particular holdings in particular accounts',
                  'I hold the bonds in the IRA and the stocks in the taxable '
                  'account.',
                  family='asset_location', source='https://www.tencap.com/blog/6-asset-location-strategies-place-investments/'),
            Entry('roth-conversion', 'Convert between account types',
                  'I convert $30,000 from the traditional IRA to the Roth each '
                  'year.',
                  family='roth_conversion', source='https://www.themoneypocket.com/articles/roth-conversion-ladder-strategy-retirement-tax-planning'),
            Entry('tax-loss-harvesting', 'Harvest losses',
                  'I harvest losses whenever a position falls 10% below its cost '
                  'basis.',
                  family='tax_loss_harvesting', source='https://www.financialplanningassociation.org/learning/publications/journal/OCT22-direct-indexing-tax-loss-harvesting-OPEN'),
        ),
    ),
    Group(
        key='other',
        title='Cash, leverage and everything else',
        note='Positions this engine values differently, or does not value at all.',
        entries=(
            Entry('cash-reserve', 'Keep a cash reserve before investing',
                  'I keep six months of expenses in cash before investing '
                  'anything.',
                  family='cash_reserve', source='https://www.investor.gov/introduction-investing/getting-started/emergency-fund'),
            Entry('bucket-strategy', 'Split into near-term and long-term buckets',
                  'I keep three years of expenses in cash and the rest in stocks.',
                  family='bucket_strategy', source='https://blincoe.uk/the-blincoe-blog/retirement-income-bucketing-strategy'),
            Entry('leverage', 'Use leverage on part of the portfolio',
                  'I hold 2x leverage on the equity sleeve of my portfolio.',
                  family='leverage', source='https://www.investopedia.com/terms/l/leverage.asp'),
            Entry('option-income', 'Sell options for income',
                  'I sell covered calls one strike out of the money each month.',
                  family='option_income', source='https://www.investopedia.com/terms/c/coveredcall.asp'),
            Entry('non-market-alternative', 'Compare against paying down a debt',
                  'I pay off the mortgage instead of investing.',
                  family='non_market_alternative', source='https://www.investopedia.com/articles/pf/07/mortgage_investment.asp'),
        ),
    ),
)


def entry(key: str) -> Optional[Entry]:
    for group in LIBRARY:
        for candidate in group.entries:
            if candidate.key == key:
                return candidate
    return None


def offered() -> Sequence[Entry]:
    return [e for group in LIBRARY for e in group.entries]


#: How a sentence got into the box. Recorded on every attempt, because without
#: it the cohort measures the catalogue rather than the runtime: a high success
#: rate over sentences we wrote, read by a reader we wrote, is a closed loop.
#: `TYPED` is the only origin that carries evidence about people's own words,
#: and `EDITED` is the interesting middle — it says the catalogue got them
#: close and something was missing.
PICKED, EDITED, TYPED = "PICKED", "EDITED", "TYPED"


def origin_of(text: str, picked_key: str) -> str:
    """Derived from the text and the pick, never taken from the client.

    A hidden field saying "PICKED" is a claim the browser makes, and the one
    thing it cannot be trusted about is whether the user then changed the
    sentence — which is exactly the distinction this is for. Comparing against
    the catalogue is cheap and cannot be wrong about it.
    """
    if not picked_key:
        return TYPED
    chosen = entry(picked_key)
    if chosen is None:
        return TYPED
    return PICKED if " ".join(text.split()) == " ".join(
        chosen.text.split()) else EDITED
