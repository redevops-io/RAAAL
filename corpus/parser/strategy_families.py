"""The wealth-management strategy families, and what each one must produce.

    python corpus/parser/strategy_families.py        # writes strategy_families.json

**Why this tier exists.** The phrasing pack was built almost entirely from
"how do I put money in" sentences, because that is what the sources it could
lawfully read happen to contain. A web sweep across planning literature found
roughly twenty families people actually ask about, and most of them are about
taking money *out*, moving it between account types, or changing the shape of
the portfolio over time — none of which the pack contained a single example of.

**What each case asserts, and it is not "the parser understands this".** For a
family this build cannot run, understanding is the wrong goal. The manifest
already refuses `sell_action` by name — *selling, withdrawing and harvesting are
not modelled* — and `tax_treatment` as `NOT_MODELLED`. What matters is whether
the *recognition* step fires those refusals, because a refusal that never fires
is not a boundary. So each case carries the dimension that should carry it, and
the outcome required is one of:

    RECOGNISED          this build runs it; the dimension must be read
    REFUSED_BY_NAME     this build cannot; the dimension must still be read,
                        so Mission can refuse it by the name of the thing
                        the person actually asked for

**The failure this measures.** A sentence whose refusable dimension goes
unrecognised does not produce a refusal. It produces a plan built from whatever
fragment *was* recognised — and those fragments are accumulation-shaped, so
"convert $30,000 from the traditional IRA to the Roth each year" becomes
indistinguishable from "contribute $30,000 to a Roth each year". That is not an
approximation flagged as one. It is a different strategy wearing the same
numbers, and the coverage gate cannot catch it because coverage compares the
declaration against the execution, never against what the person said.

**Provenance.** These sentences are authored, and marked as such. They are not
attested phrasings and must never be counted as any — `real_phrasings.json` is
the attested pack and this is not it. What is *not* authored is the family list
and the definitions behind it, which come from the planning sources cited on
each case. An authored sentence is admissible here because the property under
test is whether a named capability is recognised at all, which does not depend
on the sentence being one somebody really typed. It would be inadmissible as
evidence about phrasing coverage, and this file is not used for that.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "strategy_families.json"

RECOGNISED = "RECOGNISED"
REFUSED_BY_NAME = "REFUSED_BY_NAME"

#: No dimension in this schema can express the request, so no carrier exists to
#: read. Distinct from REFUSED_BY_NAME, where the concept has a name and the
#: only question is whether recognition fires it.
#:
#: Asset location is the case that forced this state. `account_type` *is* read
#: from "hold the bonds in the IRA and the stocks in the taxable account" — it
#: returns TAXABLE — so the family scored as understood while the thing being
#: asked for, a mapping of holdings onto accounts, was gone. A single-valued
#: dimension cannot carry a mapping, and calling that a recognition success
#: would have hidden a schema gap behind a passing check.
NO_DIMENSION = "NO_DIMENSION"

#: family -> (source, dimensions any ONE of which proves the request was read,
#:            outcome, sentences)
#:
#: A *set* of carriers, not one. The first version named a single dimension per
#: family and scored two families wrong: a moving-average trigger is proved
#: read by `moving_average_window` just as well as by `trigger_semantics`, and
#: insisting on one name reported a working recognition as a defect. A
#: measurement that miscounts working behaviour cannot be trusted about the
#: broken kind either.
FAMILIES = [
    # ---- what this build actually runs ---------------------------------
    ("scheduled_funding",
     "https://www.investor.gov/introduction-investing/investing-basics/glossary/dollar-cost-averaging",
     ("cadence",), RECOGNISED, [
         "invest $500 monthly into VTI",
         "put $250 into the index fund every two weeks",
         "contribute $1,000 a quarter",
     ]),
    ("event_triggered_funding",
     "https://www.investopedia.com/terms/m/movingaverage.asp",
     ("trigger_semantics", "moving_average_window"), RECOGNISED, [
         "buy VOO when SPY falls below its 200-day moving average",
         "add to BND while TLT is under its 200-day",
     ]),

    # ---- refused, and the manifest says so by name ----------------------
    ("safe_withdrawal_rate",
     "https://www.nysdcp.com/rsc-preauth/learn-about-retirement/close-to-or-living-in-retirement/articles/withdrawal-strategies-to-consider-for-retirement",
     ("sell_action",), REFUSED_BY_NAME, [
         "withdraw 4% of the portfolio each year, adjusted for inflation",
         "draw down 3% a year in retirement",
         "take $40,000 a year out of the portfolio",
     ]),
    ("withdrawal_ordering",
     "https://www.nysdcp.com/rsc-preauth/learn-about-retirement/close-to-or-living-in-retirement/articles/withdrawal-strategies-to-consider-for-retirement",
     ("sell_action",), REFUSED_BY_NAME, [
         "spend the taxable account first, then the IRA, then the Roth",
         "take from bonds in a down year and from stocks otherwise",
     ]),
    ("required_minimum_distribution",
     "https://www.irs.gov/retirement-plans/retirement-plan-and-ira-required-minimum-distributions-faqs",
     ("sell_action",), REFUSED_BY_NAME, [
         "take the required minimum distribution starting at 73",
     ]),
    ("tax_loss_harvesting",
     "https://www.financialplanningassociation.org/learning/publications/journal/OCT22-direct-indexing-tax-loss-harvesting-OPEN",
     ("sell_action",), REFUSED_BY_NAME, [
         "harvest losses whenever a position falls 10% below its cost basis",
         "sell the loser and buy a similar fund to avoid a wash sale",
         # The sharpest case found. Reads as `assets: VTI, BND` — the sell
         # becomes a purchase, so the plan holds the thing the person said
         # they were getting rid of.
         "sell VTI and buy BND",
     ]),
    ("roth_conversion",
     "https://www.themoneypocket.com/articles/roth-conversion-ladder-strategy-retirement-tax-planning",
     ("sell_action",), REFUSED_BY_NAME, [
         "convert $30,000 from the traditional IRA to the Roth each year",
         "convert up to the top of the 22% bracket each year",
     ]),
    ("asset_location",
     "https://www.tencap.com/blog/6-asset-location-strategies-place-investments/",
     ("asset_location",), REFUSED_BY_NAME, [
         "hold the bonds in the IRA and the stocks in the taxable account",
         "keep the REITs in the Roth",
     ]),
    ("rebalancing",
     "https://www.investor.gov/introduction-investing/investing-basics/glossary/rebalancing",
     ("periodic_rebalancing",), REFUSED_BY_NAME, [
         "rebalance back to 60/40 every year",
         "rebalance whenever an allocation drifts more than 5 points",
     ]),
    ("stated_weights",
     "https://www.investor.gov/introduction-investing/investing-basics/glossary/asset-allocation",
     ("stated_weights",), REFUSED_BY_NAME, [
         "a 60/40 portfolio",
         "70% stocks, 20% bonds, 10% cash",
     ]),
    ("glidepath",
     "https://benchmarkfg.com/wp-content/uploads/2025/05/Reducing-Retirement-Risk-with-a-Rising-Equity-Glide-Path-2.pdf",
     ("periodic_rebalancing",), REFUSED_BY_NAME, [
         "shift 1% from stocks to bonds every year as I get older",
         "hold my age in bonds",
     ]),
    ("bucket_strategy",
     "https://blincoe.uk/the-blincoe-blog/retirement-income-bucketing-strategy",
     ("sell_action",), REFUSED_BY_NAME, [
         "keep three years of expenses in cash and the rest in stocks",
         "refill the cash bucket from stocks after a good year",
     ]),
    ("risk_based_allocation",
     "https://www.investopedia.com/terms/r/risk-parity.asp",
     ("allocation_method",), REFUSED_BY_NAME, [
         "allocate by inverse volatility",
         "run it at a 10% volatility target",
     ]),
    ("factor_tilt",
     "https://www.investopedia.com/terms/s/smallcap.asp",
     ("stated_weights",), REFUSED_BY_NAME, [
         "tilt 20% toward small cap value",
     ]),
    ("option_income",
     "https://www.investopedia.com/terms/c/coveredcall.asp",
     ("sell_action",), REFUSED_BY_NAME, [
         "sell covered calls one strike out of the money each month",
     ]),
    ("annuitisation",
     "https://gainbridge.com/post/decumulation-strategy",
     ("sell_action",), REFUSED_BY_NAME, [
         "annuitize a third of the portfolio at 70",
     ]),
    ("dividend_income",
     "https://www.investopedia.com/terms/d/dividend.asp",
     ("dividend_policy",), REFUSED_BY_NAME, [
         "live off the dividends and never touch the principal",
         "do not reinvest the dividends",
     ]),
    ("leverage",
     "https://www.investopedia.com/terms/l/leverage.asp",
     ("allocation_method",), REFUSED_BY_NAME, [
         "hold 2x leverage on the equity sleeve",
     ]),
    ("cash_reserve",
     "https://www.investor.gov/introduction-investing/getting-started/emergency-fund",
     ("sell_action",), REFUSED_BY_NAME, [
         "keep six months of expenses in cash before investing anything",
     ]),
    ("non_market_alternative",
     "https://www.investopedia.com/articles/pf/07/mortgage_investment.asp",
     ("objective",), REFUSED_BY_NAME, [
         "pay off the mortgage instead of investing",
     ]),
]


def build() -> dict:
    cases, by_family, by_outcome = [], {}, {}
    for family, source, carriers, outcome, sentences in FAMILIES:
        for index, text in enumerate(sentences, start=1):
            cases.append({
                "id": f"{family}-{index:02d}",
                "family": family,
                "text": text,
                "carriers": list(carriers),
                "must_be": outcome,
                "provenance": "authored_from_cited_definition",
                "source": source})
        by_family[family] = len(sentences)
        by_outcome[outcome] = by_outcome.get(outcome, 0) + len(sentences)

    return {
        "schema": "quantify-strategy-families@1",
        "count": len(cases),
        "families": len(FAMILIES),
        "by_family": by_family,
        "by_outcome": by_outcome,
        "provenance_note": (
            "The sentences are authored and marked as such. They are NOT "
            "attested phrasings and must never be counted as any — "
            "real_phrasings.json is the attested pack. The family list and the "
            "definitions behind it come from the sources cited on each case. "
            "An authored sentence is admissible here because the property "
            "under test is whether a named capability is recognised at all, "
            "which does not depend on anybody having really typed it."),
        "property_note": (
            "REFUSED_BY_NAME does not mean the sentence should fail to parse. "
            "It means the dimension that carries the request must be READ, so "
            "that Mission can refuse the thing the person actually asked for "
            "rather than silently running the accumulation-shaped fragment "
            "that survived recognition."),
        "cases": cases}


if __name__ == "__main__":
    document = build()
    OUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"{document['count']} cases across {document['families']} families "
          f"-> {OUT.name}")
    for outcome, total in sorted(document["by_outcome"].items()):
        print(f"  {outcome:18} {total}")
