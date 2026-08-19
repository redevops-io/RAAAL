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
    #: The licence the source carries, where it carries one.
    #:
    #: Entries promoted from the harvested corpus come from Stack Exchange
    #: posts under CC-BY-SA 4.0. The statement here is authored rather than
    #: quoted — forum prose is full of context a menu entry cannot carry —
    #: but the strategy is somebody else's and the attribution travels with
    #: it. Empty for entries written from a definition that states no licence.
    licence: str = ""


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
            Entry('percentage-of-income', 'Contribute a percentage of income',
                  'I put 10% of my after-tax salary into a traditional IRA every '
                  'year.',
                  family='percentage_funding',
                  source='https://money.stackexchange.com/questions/20835/investments-other-than-cds',
                  licence='CC-BY-SA 4.0'),
            Entry('percentage-of-paycheck', 'Contribute a percentage of each paycheck',
                  'I put 8% of every paycheck into my 401k.',
                  family='percentage_funding',
                  source='https://money.stackexchange.com/questions/55829/should-i-buy-2200-of-a-hot-stock-or-invest-elsewhere',
                  licence='CC-BY-SA 4.0'),
            Entry('weekly-funding', 'Contribute every week',
                  'I invest $50 into VTI every week.',
                  family='scheduled_funding',
                  source='https://money.stackexchange.com/questions/136064/focus-on-mortgage-or-keep-investing',
                  licence='CC-BY-SA 4.0'),
            Entry('annual-lump', 'Add a lump sum once a year',
                  'I contribute $10,000 to the portfolio once a year.',
                  family='scheduled_funding',
                  source='https://money.stackexchange.com/questions/16057/what-else-besides-fees-should-i-consider-in-rebalancing-my-fund-portfolios-asse',
                  licence='CC-BY-SA 4.0'),
            Entry('max-the-limit', 'Contribute up to the annual limit',
                  'I contribute 22% of my salary to my 401k, up to the $18,000 '
                  'annual limit.',
                  family='contribution_limit',
                  source='https://money.stackexchange.com/questions/45275/am-i-investing-properly-for-my-future',
                  licence='CC-BY-SA 4.0'),
            Entry('max-several-accounts', 'Max out several accounts each year',
                  'I fully fund my Roth IRA each year and max out my 403(b) as '
                  'well.',
                  family='contribution_limit',
                  source='https://money.stackexchange.com/questions/134835/excess-income-after-fully-funding-all-retirement-accounts-now-what',
                  licence='CC-BY-SA 4.0'),
            Entry('employer-match', 'Capture the employer match',
                  'My employer contributes $100 to my HSA and matches the next '
                  '$400 I put in.',
                  family='employer_match',
                  source='https://money.stackexchange.com/questions/115105/choosing-between-tradtional-ppo-and-hdhp-with-high-prescription-cost',
                  licence='CC-BY-SA 4.0'),
            Entry('split-across-accounts', 'Split each contribution across accounts',
                  'I contribute $50 into each of two accounts every month.',
                  family='scheduled_funding',
                  source='https://money.stackexchange.com/questions/76425/should-i-switch-to-a-new-ira-custodian',
                  licence='CC-BY-SA 4.0'),
            Entry('escalating-contribution', 'Increase the contribution over time',
                  'I invest 100 EUR a month now and raise it to 150 EUR a month '
                  'next year.',
                  family='escalating_funding',
                  source='https://money.stackexchange.com/questions/75058/how-aggressive-should-my-personal-portfolio-be',
                  licence='CC-BY-SA 4.0'),
        ),
    ),
    Group(
        key='allocation',
        title='Choosing and holding the mix',
        note='What the money buys, in what proportions, and whether that is restored over time.',
        entries=(
            Entry('stated-weights', 'Hold a stated split, such as 60/40',
                  'I invest $500 a month, 60% in VTI and 40% in BND.',
                  family='stated_weights', source='https://www.investor.gov/introduction-investing/investing-basics/glossary/asset-allocation'),
            Entry('rebalancing', 'Rebalance back to the target weights',
                  'I invest $500 a month, 60% in VTI and 40% in BND, and '
                  'rebalance back to that split once a year.',
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
            Entry('single-fund', 'Hold a single fund',
                  'I invest $500 a month, all of it into VOO.',
                  family='stated_weights',
                  source='https://money.stackexchange.com/questions/39679/how-or-is-it-necessary-to-rebalance-a-401k-with-only-one-index-fund',
                  licence='CC-BY-SA 4.0'),
            Entry('three-way-split', 'Hold a three-way split',
                  'I invest $600 a month, 60% in VTI, 30% in VXUS and 10% in BND.',
                  family='stated_weights',
                  source='https://money.stackexchange.com/questions/55630/first-401k-portfolio-with-high-expense-ratios-which-funds-to-pick-24yo',
                  licence='CC-BY-SA 4.0'),
            Entry('mean-variance', 'Optimise the mix by mean-variance',
                  'I set my weights by mean-variance optimisation across 30 '
                  'holdings.',
                  family='mean_variance',
                  source='https://money.stackexchange.com/questions/145454/is-there-a-quantitative-answer-to-how-frequently-i-should-optimize-my-portfolio',
                  licence='CC-BY-SA 4.0'),
            Entry('volatility-target', 'Target a level of volatility',
                  'I size the portfolio to target 10% annualised volatility.',
                  family='volatility_target',
                  source='https://quant.stackexchange.com/questions/71489/when-how-do-i-vol-scale-portfolio-weights-when-optimizing-the-portfolio',
                  licence='CC-BY-SA 4.0'),
            Entry('fund-and-rebalance', 'Add money and rebalance together',
                  'Every three months I invest $600, 70% in VTI and 30% in BND, '
                  'and rebalance back to that split.',
                  family='rebalancing',
                  source='https://money.stackexchange.com/questions/142270/do-bond-funds-have-an-inherent-advantage-over-individual-bonds-within-a-portfoli',
                  licence='CC-BY-SA 4.0'),
            Entry('holding-period', 'Hold each position for a fixed period',
                  'I hold each position for 21 days and then close it.',
                  family='holding_period',
                  source='https://quant.stackexchange.com/questions/77807/intuition-behind-portfolio-weights-with-lower-rmse-but-higher-variance',
                  licence='CC-BY-SA 4.0'),
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
            Entry('fixed-dollar-withdrawal', 'Withdraw a fixed amount each year',
                  'I withdraw $20,000 from the portfolio each year.',
                  family='fixed_withdrawal',
                  source='https://money.stackexchange.com/questions/95821/how-to-place-value-on-small-business-that-has-0-earnings',
                  licence='CC-BY-SA 4.0'),
            Entry('withdraw-to-clear-debt', 'Withdraw once to clear a debt',
                  'I withdraw $18,000 from the portfolio to pay off my debt.',
                  family='debt_clearing_withdrawal',
                  source='https://money.stackexchange.com/questions/102957/withdraw-ira-funds-after-divorce',
                  licence='CC-BY-SA 4.0'),
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
            Entry('roth-deferral', 'Make contributions Roth rather than pre-tax',
                  'I make 100% of my 401k contributions Roth rather than pre-tax.',
                  family='deferral_choice',
                  source='https://money.stackexchange.com/questions/169554/rule-of-55-with-a-401k-plan-with-mixed-roth-and-401k-contributions',
                  licence='CC-BY-SA 4.0'),
            Entry('pretax-deferral', 'Make contributions pre-tax rather than Roth',
                  'I make 100% of my 401k contributions pre-tax rather than Roth.',
                  family='deferral_choice',
                  source='https://money.stackexchange.com/questions/169554/rule-of-55-with-a-401k-plan-with-mixed-roth-and-401k-contributions',
                  licence='CC-BY-SA 4.0'),
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
            Entry('months-of-expenses', 'Hold a set number of months of expenses',
                  'I keep 12 months of expenses in cash before investing '
                  'anything.',
                  family='cash_reserve',
                  source='https://money.stackexchange.com/questions/126216/home-mortgage-w-very-low-interest-rates-more-or-less-down',
                  licence='CC-BY-SA 4.0'),
            Entry('split-a-lump-sum', 'Split a lump sum across purposes',
                  'I put $10,000 into an emergency fund and invest the remaining '
                  '$20,000.',
                  family='lump_allocation',
                  source='https://money.stackexchange.com/questions/46415/building-financial-independence',
                  licence='CC-BY-SA 4.0'),
            Entry('payoff-versus-invest', 'Compare paying down a debt with investing',
                  'I have $10,000 and I compare paying down a 5% mortgage with '
                  'investing it.',
                  family='non_market_alternative',
                  source='https://money.stackexchange.com/questions/7246/do-i-end-up-richer-paying-down-a-mortgage-or-contributing-to-an-ira',
                  licence='CC-BY-SA 4.0'),
            Entry('margin-repayment', 'Pay down a margin loan',
                  'I use $5,000 of cash to pay down the margin loan on my stock '
                  'position.',
                  family='leverage',
                  source='https://money.stackexchange.com/questions/157737/calculating-foreign-exchange-capital-loss-on-margin-loan-repayment',
                  licence='CC-BY-SA 4.0'),
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
