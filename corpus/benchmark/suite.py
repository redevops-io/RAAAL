"""The Quantify Strategy Evaluation Benchmark — the corpus.

    python corpus/benchmark/suite.py        # writes suite.json

**Not a coverage measurement.** "What percentage of strategies does Quantify
support?" is the wrong question and was already answered badly once: most of
what a planner is asked about, this build refuses by design, and a percentage
would make correct refusals look like failures. What this hunts is
counterexamples — places where the system is misleading, unstable,
mathematically wrong, or refuses something it should handle.

Three structures, and none of them needs a known portfolio value:

**Equivalence classes.** One strategy, expressed many ways. Every phrasing in a
class must reach the same disposition, and where it executes, the same compiled
plan. No oracle is required — the class is its own oracle, and a phrasing that
diverges from its siblings is the finding.

**Contrast pairs.** Two prompts a word apart that must *not* agree. `contribute
monthly` against `rebalance monthly`; `buy when below` against `buy when
crossing below`. These are worth more than a thousand random prompts, because a
one-word change with a known semantic consequence is a test with a known
answer.

**Metamorphic relations.** A transformation and what it must do to the result.
Doubling the contribution must double what is contributed; reordering the
holdings of an equal-weight plan must change nothing; adding irrelevant
autobiography must change nothing.

**Every phrasing declares its expected disposition**, because an unsupported
strategy refused by name is a pass, not a failure, and a benchmark that scored
executions would reward exactly the silent reduction this project spent months
removing.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "suite.json"

# Dispositions a class may expect.
EXECUTES = "EXECUTES"
REFUSES = "REFUSES"
CLARIFIES = "CLARIFIES"


def _class(cid, family, disposition, phrasings, *, refuses=None, note="",
           states=None):
    """`refuses` — capabilities whose refusal is a correct answer here. Any one
    of them satisfies the class, because a sentence can state several
    unsupported things and naming one of them is a real refusal.

    `states` — every unsupported thing the sentence actually asserts. All must
    be named, or the person is told about one and left believing the others
    were fine. "rebalance back to 60/40 every year" is refused for
    `stated_weights` and never mentions the rebalancing.
    """
    return {"id": cid, "family": family, "disposition": disposition,
            "refuses": refuses or [], "states": states or [], "note": note,
            "phrasings": list(phrasings)}


#: One entry per strategy meaning. The phrasings inside must agree with each
#: other; what they must agree *on* is the disposition, and the compiled plan
#: where one exists.
CLASSES = [
    # ---- accumulation, which this build executes ----------------------
    _class("dca-monthly-fixed", "accumulation", EXECUTES, [
        "invest $500 monthly into VTI",
        "put $500 into VTI every month",
        "buy $500 of VTI each month",
        "contribute $500 a month to VTI",
        "monthly $500 purchase of VTI",
        "I want to add $500 to VTI every month",
        "set aside $500 for VTI monthly",
        "$500/month into VTI",
    ]),
    _class("dca-weekly", "accumulation", EXECUTES, [
        "invest $100 weekly into VTI",
        "put $100 a week into VTI",
        "buy $100 of VTI every week",
        "weekly $100 contribution to VTI",
        "$100 into VTI each week",
    ]),
    _class("lump-sum", "accumulation", EXECUTES, [
        "invest $10,000 in VTI once",
        "a single $10,000 purchase of VTI",
        "put $10,000 into VTI as a lump sum",
        "one-off $10,000 investment in VTI",
        "buy $10,000 of VTI, just once",
    ]),

    # ---- conditional buying -------------------------------------------
    _class("ma-cross-below", "conditional", EXECUTES, [
        "buy $1,000 of VTI when it crosses below its 200-day moving average",
        "whenever VTI crosses under the 200 DMA, invest $1,000",
        "put a grand into VTI on the day it falls through its 200-day average",
        "if VTI moves from above its 200-day MA to below it, buy $1,000",
        "invest $1,000 in VTI the day it drops beneath the 200-day average",
        "$1,000 into VTI each time it breaks below its 200 day moving average",
    ], note="a crossing, not a state"),
    _class("ma-persistent-below", "conditional", EXECUTES, [
        "buy $1,000 of VTI whenever it is below its 200-day moving average",
        "invest $1,000 in VTI on any day it sits under the 200 DMA",
        "while VTI is beneath its 200-day average, put in $1,000",
        "$1,000 into VTI on every day it stays below the 200-day average",
    ], note="a state, not a crossing"),

    # ---- allocation ----------------------------------------------------
    _class("equal-weight", "allocation", EXECUTES, [
        "invest $1,000 monthly split equally between VTI and BND",
        "$1,000 a month, equal weight across VTI and BND",
        "put $1,000 monthly into VTI and BND, evenly",
        "monthly $1,000 divided equally between VTI and BND",
    ]),
    _class("stated-weights", "allocation", REFUSES, [
        "invest $1,000 monthly, 60% VTI and 40% BND",
        "a 60/40 split of $1,000 a month between VTI and BND",
        "$1,000 monthly, sixty percent VTI, forty percent BND",
        "monthly $1,000 weighted 60/40 across VTI and BND",
    ], refuses=["stated_weights"],
       note="this build divides each purchase equally"),
    _class("inverse-volatility", "allocation", REFUSES, [
        "allocate $1,000 monthly by inverse volatility across VTI and BND",
        "$1,000 a month weighted by inverse volatility",
        "invest $1,000 monthly, volatility-weighted across VTI and BND",
    ], refuses=["allocation_method"]),

    # ---- rebalancing ---------------------------------------------------
    _class("rebalance-annual", "rebalancing", REFUSES, [
        "rebalance back to 60/40 every year",
        "annually rebalance to a 60/40 target",
        "each year, bring the portfolio back to 60/40",
        "yearly rebalancing to 60% stocks and 40% bonds",
    ], refuses=["periodic_rebalancing", "stated_weights", "allocation_method"],
       states=["periodic_rebalancing", "stated_weights"],
       note="two unsupported things in one sentence"),
    _class("rebalance-threshold", "rebalancing", REFUSES, [
        "rebalance whenever an allocation drifts more than 5 points",
        "rebalance if any holding moves 5% away from its target",
        "bring it back to target when drift exceeds five percentage points",
    ], refuses=["periodic_rebalancing"]),

    # ---- withdrawal ----------------------------------------------------
    _class("swr-percentage", "withdrawal", REFUSES, [
        "withdraw 4% of the portfolio each year",
        "take out 4% a year",
        "draw down 4% annually",
        "an annual withdrawal of four percent",
        "spend 4% of the balance every year",
    ], refuses=["sell_action"]),
    _class("swr-fixed-amount", "withdrawal", REFUSES, [
        "withdraw $40,000 a year from the portfolio",
        "take $40,000 out each year",
        "an annual $40,000 withdrawal",
        "draw $40,000 per year",
    ], refuses=["sell_action"]),

    # ---- tax -----------------------------------------------------------
    _class("tax-loss-harvest", "tax", REFUSES, [
        "harvest losses whenever a position falls 10% below its cost basis",
        "sell losers that are down 10% from what I paid",
        "tax-loss harvest positions 10% underwater",
    ], refuses=["sell_action"]),
    _class("roth-conversion", "tax", REFUSES, [
        "convert $30,000 from the traditional IRA to the Roth each year",
        "move $30,000 a year from my traditional IRA into a Roth",
        "annually convert $30,000 of traditional IRA to Roth",
    ], refuses=["sell_action", "objective"]),
    _class("asset-location", "tax", REFUSES, [
        "hold the bonds in the IRA and the stocks in the taxable account",
        "keep bonds in my IRA, equities in taxable",
        "put the fixed income in the IRA and the shares in the brokerage",
    ], refuses=["asset_location"]),

    # ---- retirement structures -----------------------------------------
    _class("bucket-strategy", "retirement", REFUSES, [
        "keep three years of expenses in cash and the rest in stocks",
        "hold three years of spending in cash, invest the remainder",
        "a cash bucket covering three years, equities for the rest",
    ], refuses=["reserve_policy", "bucket_policy"]),
    _class("glidepath", "retirement", REFUSES, [
        "shift 1% from stocks to bonds every year as I get older",
        "move one percent a year out of equities into fixed income",
        "reduce the equity share by 1% annually",
    ], refuses=["periodic_rebalancing", "sell_action"],
       note="moving money out of equities is selling, and refusing on that is "
            "as correct as refusing on the rebalancing"),

    # ---- risk and leverage ---------------------------------------------
    _class("volatility-target", "risk", REFUSES, [
        "run it at a 10% volatility target",
        "target 10% annualised volatility",
        "size positions for 10% vol",
    ], refuses=["allocation_method"]),
    _class("sleeve-leverage", "leverage", REFUSES, [
        "hold 2x leverage on the equity sleeve",
        "run the equity portion at two times leverage",
        "double leverage on the stock sleeve",
    ], refuses=["portfolio_sleeves"]),

    # ---- factor and rotation --------------------------------------------
    _class("factor-tilt", "factor", REFUSES, [
        "tilt 20% toward small cap value",
        "a 20% allocation tilted to small-cap value",
        "overweight small cap value by 20%",
    ], refuses=["portfolio_sleeves", "stated_weights"]),
    _class("momentum-rotation", "rotation", REFUSES, [
        "each month hold whichever of VTI and BND performed best",
        "rotate monthly into the stronger of VTI and BND",
        "monthly, buy the better performer of VTI or BND",
    ], refuses=["selection_rule", "sell_action"],
       note="the selection is the strategy; `selection_rule` was added after "
            "this family executed as a plain purchase of every candidate"),

    # ---- comparison questions -------------------------------------------
    _class("mortgage-versus-investing", "comparison", REFUSES, [
        "pay off the mortgage instead of investing",
        "should I overpay the mortgage rather than invest",
        "put the money into the mortgage instead of the market",
    ], refuses=["objective"]),
]


#: Prompts a word apart whose meanings differ. Each pair names *what* must
#: differ, so a pair that agrees is a specific finding rather than a surprise.
CONTRASTS = [
    ("contribute monthly versus rebalance monthly",
     "invest $500 monthly into VTI",
     "rebalance VTI and BND monthly",
     "one funds, one rearranges; the second is refused"),
    ("crossing versus state",
     "buy $1,000 of VTI when it crosses below its 200-day moving average",
     "buy $1,000 of VTI whenever it is below its 200-day moving average",
     "a transition fires once per episode, a state fires every session"),
    ("sell-and-buy versus buy-both",
     "sell VTI and buy BND",
     "buy VTI and BND",
     "the first disposes of something and must be refused"),
    ("withdraw versus contribute",
     "withdraw $1,000 monthly",
     "contribute $1,000 monthly",
     "opposite directions of money"),
    ("conversion direction",
     "convert $30,000 from the traditional IRA to the Roth",
     "convert $30,000 from the Roth to the traditional IRA",
     "direction is the meaning of a transition"),
    ("moving average versus a holding period",
     "buy VTI below its 200-day moving average",
     "hold VTI for 200 days",
     "one is a threshold, the other a duration"),
    ("weights reversed",
     "invest $1,000 monthly, 60% VTI and 40% BND",
     "invest $1,000 monthly, 40% VTI and 60% BND",
     "different portfolios; both refused, and the stated weights differ"),
    ("equal weight versus stated weight",
     "invest $1,000 monthly split equally between VTI and BND",
     "invest $1,000 monthly, 60% VTI and 40% BND",
     "the first executes, the second is refused"),
]


#: A transformation of a prompt, and what it must do to the compiled plan.
#: `SAME` means the plan must be identical; `DIFFER` means it must not be.
METAMORPHIC = [
    ("irrelevant autobiography", "SAME",
     "invest $500 monthly into VTI",
     "I'm 42 and have been thinking about this for a while. "
     "invest $500 monthly into VTI"),
    ("thousands shorthand", "SAME",
     "invest $1,000 monthly into VTI",
     "invest $1k monthly into VTI"),
    ("moving-average abbreviation", "SAME",
     "buy $1,000 of VTI when it crosses below its 200-day moving average",
     "buy $1,000 of VTI when it crosses below its 200 DMA"),
    ("holding order, equal weight", "SAME",
     "invest $1,000 monthly split equally between VTI and BND",
     "invest $1,000 monthly split equally between BND and VTI"),
    ("trailing punctuation", "SAME",
     "invest $500 monthly into VTI",
     "Invest $500 monthly into VTI."),
    ("doubled contribution", "DIFFER",
     "invest $500 monthly into VTI",
     "invest $1,000 monthly into VTI"),
    ("changed cadence", "DIFFER",
     "invest $500 monthly into VTI",
     "invest $500 weekly into VTI"),
    ("changed holding", "DIFFER",
     "invest $500 monthly into VTI",
     "invest $500 monthly into BND"),
    ("reversed weights", "DIFFER",
     "invest $1,000 monthly, 60% VTI and 40% BND",
     "invest $1,000 monthly, 40% VTI and 60% BND"),
]


def build() -> dict:
    prompts = {p for c in CLASSES for p in c["phrasings"]}
    prompts |= {p for _n, a, b, _w in CONTRASTS for p in (a, b)}
    prompts |= {p for _n, _r, a, b in METAMORPHIC for p in (a, b)}

    by_family: dict = {}
    for entry in CLASSES:
        by_family.setdefault(entry["family"], 0)
        by_family[entry["family"]] += len(entry["phrasings"])

    return {
        "schema": "quantify-strategy-benchmark@1",
        "classes": CLASSES,
        "contrasts": [{"name": n, "left": a, "right": b, "why": w}
                      for n, a, b, w in CONTRASTS],
        "metamorphic": [{"name": n, "relation": r, "from": a, "to": b}
                        for n, r, a, b in METAMORPHIC],
        "counts": {
            "classes": len(CLASSES),
            "phrasings": sum(len(c["phrasings"]) for c in CLASSES),
            "contrast_pairs": len(CONTRASTS),
            "metamorphic_relations": len(METAMORPHIC),
            "distinct_prompts": len(prompts),
            "by_family": by_family,
        },
        "provenance_note": (
            "Authored, and marked as such. Not attested phrasings — "
            "`real_phrasings.json` is the attested pack. Authored sentences "
            "are admissible here because the properties under test are "
            "agreement between phrasings, disagreement between contrasts, and "
            "invariance under transformation, none of which depends on "
            "somebody having really typed the sentence."),
        "scoring_note": (
            "There is no pass rate. An unsupported strategy refused by name is "
            "a pass; a benchmark that scored executions would reward the "
            "silent reduction this project spent months removing."),
        "prompts": sorted(prompts),
    }


if __name__ == "__main__":
    document = build()
    OUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    counts = document["counts"]
    print(f"{counts['classes']} classes, {counts['phrasings']} phrasings")
    print(f"{counts['contrast_pairs']} contrast pairs, "
          f"{counts['metamorphic_relations']} metamorphic relations")
    print(f"{counts['distinct_prompts']} distinct prompts -> {OUT.name}")
    for family, total in sorted(counts["by_family"].items()):
        print(f"   {family:14} {total}")
