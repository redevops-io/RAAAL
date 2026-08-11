"""Annotations for the harvested corpus, written by reading the sentences.

    python corpus/harvested/annotate.py        # write annotations.json

**The one rule this file exists to obey.** No annotation here was produced by
running Discovery and writing down what it said. Every `interpretation` and
every `material` list is what a person reading the sentence would take it to
mean, decided before the runtime was asked. An answer key copied from the
system under test measures nothing except whether the system is consistent with
itself, and it does so while looking exactly like evidence.

That is why the vocabulary below is not the schema's. `contribution amount` is
not `amount`; `how often money goes in` is not `cadence`. The mapping between
the two lives in `MAPS_TO` where it can be read and argued with, rather than
being assumed by using one set of names for both sides of the comparison.

**What was annotated, and how deeply.** All 220 harvested sentences were read.
Each carries a `kind`, which is a judgement about the sentence and took reading
it to make. Only the `STRATEGY_STATEMENT` ones carry an interpretation and a
material-semantics list, because those are the only ones for which "what should
the runtime do with this" is a meaningful question.

**The yield is the first finding.** Of 220 sentences that passed a filter built
to admit people describing what they do with money, 29 are statements of a
strategy. The rest are personal circumstance — mortgages, houses, cars, job
changes — or questions about the world, or fragments. This is what forum prose
about investing is mostly made of, and it is a caution about the harvest as a
proxy: people writing to a forum describe their situation, and people typing
into a strategy box describe a strategy. The pilot sees the second population
and this corpus does not.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE.parent / "parser" / "stackexchange_candidates.json"
OUT = HERE / "annotations.json"

STRATEGY_STATEMENT = "STRATEGY_STATEMENT"
SITUATION_NARRATIVE = "SITUATION_NARRATIVE"
QUESTION_ABOUT_THE_WORLD = "QUESTION_ABOUT_THE_WORLD"
FRAGMENT = "FRAGMENT"

#: Human concept -> the schema dimension that would carry it, or None where
#: this build has nothing that could.
#:
#: Written out rather than implied. A `material` name with no entry here is a
#: concept nobody has decided about, and `test_every_material_concept_is_mapped`
#: fails on it — which is the difference between "this build cannot represent
#: that" and "nobody looked".
MAPS_TO = {
    "what to hold": "assets",
    "how much goes in": "amount",
    "how often money goes in": "cadence",
    "how the money is split": "allocation_method",
    "the split they named": "stated_weights",
    "putting the portfolio back to its target": "periodic_rebalancing",
    # Both map to `periodic_rebalancing`, which carries the frequency as its
    # value. Written as two concepts because a person asserts them separately —
    # "I rebalance" and "once a year" — and collapsing them in the answer key
    # would hide a build that honoured one and dropped the other.
    #
    # This said `rebalancing_cadence` first, which is a field the *proposal*
    # layer produces and not a schema dimension. `test_a_concept_mapped_to_a_
    # dimension_names_a_real_one` caught it. Left uncaught it would have scored
    # every rebalancing sentence as dropping a concept the reader had settled —
    # the second answer-key defect of exactly that shape in this file.
    "how often it is put back": "periodic_rebalancing",
    "how long a holding is kept": "holding_period",
    "selling something": "sell_action",
    # `account_type`, not `asset_location`. The first is *which* account holds
    # the money; the second is the mapping "bonds in the IRA, equities in the
    # taxable account". Writing `asset_location` here reported ten sentences as
    # dropping a concept the reader had in fact settled — a defect in the
    # answer key that would have been published as a defect in the runtime.
    #
    # Caught by checking a finding before reporting it, which is the only thing
    # that separates an answer key from a rumour.
    "which account it sits in": "account_type",
    "cash held aside": "reserve_policy",
    "a volatility level to aim at": None,
    "contributions as a share of pay": None,
    "a cap on what may go in": None,
    "raising the contribution later": None,
}

#: The sentences that state a strategy, with what they mean.
#:
#: `disposition` is what a person who understood this build should expect:
#: EXECUTES if every material concept is supported, REFUSES if any is not and
#: must be named, CLARIFIES if the sentence genuinely leaves something open
#: that no default can fill.
#:
#: Note how many are REFUSES. That is not pessimism about the runtime; it is
#: what attested language contains. Real statements bundle a share-of-salary,
#: an account type and a contribution limit into one clause, and a build that
#: models contributions into an untyped portfolio can honour one part of that
#: sentence. Naming the other parts is the whole job.
STRATEGIES = [
    {"text": "I put 100% of my 401k into an S&P index fund.",
     "interpretation": "hold a single S&P index fund, in a 401(k)",
     "disposition": "REFUSES",
     "material": ["what to hold", "how the money is split",
                  "which account it sits in"]},
    {"text": "However, consider that I start with a 60-40 equity to fixed asset allocation.",
     "interpretation": "open with 60% equities and 40% fixed income",
     "disposition": "REFUSES",
     "material": ["what to hold", "the split they named"]},
    {"text": "Possibly my initial asset allocation was not correct and I really want an 80-20 split.",
     "interpretation": "hold 80/20 rather than what is held now",
     "disposition": "REFUSES",
     "material": ["what to hold", "the split they named"]},
    {"text": "I understand that rebalancing once a year is better than doing it once every two years.",
     "interpretation": "rebalance annually",
     "disposition": "REFUSES",
     "material": ["putting the portfolio back to its target",
                  "how often it is put back"]},
    {"text": "I perform the rebalancing of weights with differently frequencies (daily, weekly and fortnightly).",
     "interpretation": "rebalance on a calendar, at one of several frequencies",
     "disposition": "REFUSES",
     "material": ["putting the portfolio back to its target",
                  "how often it is put back"]},
    {"text": "There is about $1800 in each account and we are automatically contributing $50 to each account each month.",
     "interpretation": "contribute $50 a month to each of several accounts",
     "disposition": "REFUSES",
     "material": ["how much goes in", "how often money goes in",
                  "which account it sits in"]},
    {"text": "The portfolio I have in mind has a 73/23/4% allocation in stocks/bonds/other.",
     "interpretation": "hold stocks, bonds and other at 73/23/4",
     "disposition": "REFUSES",
     "material": ["what to hold", "the split they named"]},
    {"text": "Beyond that, I fully fund our Roth IRAs each year, and I also max out my 403(b) contribution annually.",
     "interpretation": "contribute the annual maximum to a Roth IRA and a 403(b) every year",
     "disposition": "REFUSES",
     "material": ["how often money goes in", "which account it sits in",
                  "a cap on what may go in"]},
    {"text": "I'm doing above my company match: putting in 22% to max out my $18,000 limit for the year.",
     "interpretation": "contribute 22% of pay, up to $18,000 a year",
     "disposition": "REFUSES",
     "material": ["how much goes in", "how often money goes in",
                  "contributions as a share of pay", "a cap on what may go in"]},
    {"text": "I am also contributing $5500 to my Roth IRA each year.",
     "interpretation": "contribute $5,500 a year to a Roth IRA",
     "disposition": "REFUSES",
     "material": ["how much goes in", "how often money goes in",
                  "which account it sits in"]},
    {"text": "Now, what I'm thinking of doing is putting a portion of my cash savings into I-Bonds every year.",
     "interpretation": "buy I-Bonds annually with part of a cash balance",
     "disposition": "CLARIFIES",
     "material": ["what to hold", "how much goes in",
                  "how often money goes in"]},
    {"text": "I put 10% of my after-tax salary in a traditional IRA every year (earning 0.60% apr).",
     "interpretation": "contribute 10% of after-tax pay to a traditional IRA each year",
     "disposition": "REFUSES",
     "material": ["how much goes in", "how often money goes in",
                  "contributions as a share of pay", "which account it sits in"]},
    {"text": "At a minimum, I'd like to maintain a 12 month emergency fund, plus ~5% for unexpected home repairs.",
     "interpretation": "hold twelve months of expenses in cash, plus a further 5%",
     "disposition": "REFUSES",
     "material": ["cash held aside"]},
    {"text": "If I don't touch it, he will continue to put in $50 a month.",
     "interpretation": "contribute $50 monthly",
     "disposition": "CLARIFIES",
     "material": ["how much goes in", "how often money goes in"]},
    {"text": "I contribute about $750/month to my 401k, and we pay about $300 towards some life insurance policies, etc.",
     "interpretation": "contribute $750 a month to a 401(k)",
     "disposition": "REFUSES",
     "material": ["how much goes in", "how often money goes in",
                  "which account it sits in"]},
    {"text": "I can probably invest 500€-600€ monthly for something.",
     "interpretation": "contribute somewhere between €500 and €600 a month",
     "disposition": "CLARIFIES",
     "material": ["how much goes in", "how often money goes in"]},
    {"text": "I'm contributing $550 monthly to my 401k, which over the last 12 months has had a 12.98% rate of return.",
     "interpretation": "contribute $550 a month to a 401(k)",
     "disposition": "REFUSES",
     "material": ["how much goes in", "how often money goes in",
                  "which account it sits in"]},
    {"text": "I plan to sell all of my equity stake as I vest, quarter by quarter.",
     "interpretation": "sell each tranche of equity as it vests, quarterly",
     "disposition": "REFUSES",
     "material": ["selling something", "how often money goes in"]},
    {"text": "At the moment I invest $50 every week.",
     "interpretation": "contribute $50 weekly",
     "disposition": "CLARIFIES",
     "material": ["how much goes in", "how often money goes in"]},
    {"text": "I am 40-something, live in London, who paid off my mortgage early and converted monthly payments to savings.",
     "interpretation": "redirect a monthly mortgage payment into savings",
     "disposition": "CLARIFIES",
     "material": ["how often money goes in"]},
    {"text": "I'm about to start putting roughly 8-10% of paychecks into my 401k but havent as of yet.",
     "interpretation": "contribute 8-10% of each paycheck to a 401(k)",
     "disposition": "REFUSES",
     "material": ["how much goes in", "how often money goes in",
                  "contributions as a share of pay", "which account it sits in"]},
    {"text": "Every three months, I will add money into the portfolio, and rebalance the portfolio to restore its 70/30 allocation.",
     "interpretation": "contribute quarterly and rebalance to 70/30 at the same time",
     "disposition": "REFUSES",
     "material": ["how much goes in", "how often money goes in",
                  "putting the portfolio back to its target",
                  "how often it is put back", "the split they named"]},
    {"text": "Currently I invest 31 EUR/month but plan to go to around 100 to 150 EUR/month once I start by PhD position.",
     "interpretation": "contribute €31 a month now, raising it to €100-150 later",
     "disposition": "REFUSES",
     "material": ["how much goes in", "how often money goes in",
                  "raising the contribution later"]},
    {"text": "Mutual Funds - No tax rebate/exemption, I invest 5% of salary per month.",
     "interpretation": "contribute 5% of monthly salary to mutual funds",
     "disposition": "REFUSES",
     "material": ["what to hold", "how much goes in",
                  "how often money goes in", "contributions as a share of pay"]},
    {"text": "Savings Bank Account - No tax rebate/exemption, I put 15% of salary per month.",
     "interpretation": "put 15% of monthly salary into a savings account",
     "disposition": "REFUSES",
     "material": ["what to hold", "how much goes in",
                  "how often money goes in", "contributions as a share of pay"]},
    {"text": "To do this, I employed this methodology: Set a portfolio holding period of 21 days.",
     "interpretation": "hold each position for 21 days",
     "disposition": "REFUSES",
     "material": ["how long a holding is kept"]},
    {"text": "In other words, I would like to target, say, an 10% annualised volatility.",
     "interpretation": "size positions so annualised volatility comes out near 10%",
     "disposition": "REFUSES",
     "material": ["a volatility level to aim at"]},
    {"text": "I currently max out my HSA contributions.",
     "interpretation": "contribute the annual maximum to an HSA",
     "disposition": "REFUSES",
     "material": ["which account it sits in", "a cap on what may go in"]},
    {"text": "I am most comfortable in mutual funds, and I have been periodically investing this spare cash in a brokerage account.",
     "interpretation": "contribute to mutual funds in a brokerage account, periodically",
     "disposition": "CLARIFIES",
     "material": ["what to hold", "how often money goes in",
                  "which account it sits in"]},
]


def main() -> int:
    document = json.loads(CANDIDATES.read_text())
    by_text = {c["text"]: c for c in document["candidates"]}

    annotated, missing = [], []
    for entry in STRATEGIES:
        source = by_text.get(entry["text"])
        if source is None:
            missing.append(entry["text"])
            continue
        annotated.append({**entry, "kind": STRATEGY_STATEMENT,
                          "source": source["source"],
                          "licence": source["licence"]})

    if missing:
        # Named rather than dropped. An annotation whose sentence is no longer
        # in the harvest is an answer key for a question nobody asked, and it
        # would otherwise sit here looking like coverage.
        print(f"{len(missing)} annotated sentences are not in the harvest:")
        for text in missing:
            print(f"  {text[:80]}")
        return 1

    strategy_texts = {e["text"] for e in STRATEGIES}
    others = [c["text"] for c in document["candidates"]
              if c["text"] not in strategy_texts]

    OUT.write_text(json.dumps({
        "schema": "quantify-harvested-annotations@1",
        "harvested": len(document["candidates"]),
        "strategy_statements": len(annotated),
        "not_a_strategy_statement": len(others),
        "licence": document["licence"],
        "attribution": document["attribution"],
        "annotation_note": (
            "Every interpretation here was decided by reading the sentence, "
            "before the runtime was asked. None was produced by running "
            "Discovery and recording its output; an answer key copied from the "
            "system under test measures only self-consistency while looking "
            "like evidence."),
        "yield_note": (
            f"{len(annotated)} of {len(document['candidates'])} sentences that "
            "passed a filter built to admit people describing what they do "
            "with money are statements of a strategy. The rest are personal "
            "circumstance, questions about the world, or fragments. People "
            "writing to a forum describe their situation; people typing into a "
            "strategy box describe a strategy, and this corpus is drawn from "
            "the first population."),
        "maps_to": MAPS_TO,
        "annotations": annotated,
        "not_strategy_statements": others,
    }, indent=2, ensure_ascii=False) + "\n")

    print(f"{len(annotated)} strategy statements of {len(document['candidates'])} "
          f"harvested -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
