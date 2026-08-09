"""Generates `cases.json` — the parser regression corpus.

**One case asserts one property.** That is the rule the whole corpus is built
on, and it is why there are two hundred short sentences rather than forty rich
ones. A sentence asserting four things fails as a unit: the reader learns that
*something* about "invest $500 monthly into a 60/40 split for ten years" broke,
which is the same information a stack trace already gave them.

Three tiers, in dependency order. Each is a different kind of claim and fails
for different reasons:

    normalization   characters -> a value        no parser needed
    dependency      which token governs which    needs a parse
    semantics       which field the value fills  needs a parse and a rule

A tier-2 failure with tier 1 green means the parser attached something wrongly.
A tier-3 failure with tiers 1 and 2 green means the scoring rule is wrong, not
the parser. Collapsing them would lose that.

**What this corpus is not.** It is not a sample of how people write, and it must
not be read as one — the sentences are constructed to isolate properties, and
the distribution is chosen by what breaks rather than by what is common. It is
also self-authored: I wrote both the sentences and the expected values, which is
the weakest form of evidence this project recognises. Two things limit the
damage. Every case asserts one property, so a wrong expectation is visible
rather than buried in a conjunction. And the discriminating cases come from
observed failures and from the plan's own falsification list, not from
imagination — those are marked `origin: falsification` and are the ones worth
trusting.

Run: `python corpus/parser/build_cases.py`
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "quantify-parser-corpus@1"
OUT = Path(__file__).resolve().parent / "cases.json"

cases: list = []


def case(tier: str, prop: str, text: str, asserts: dict, *,
         origin: str = "constructed", language: str = "en",
         note: str = "") -> None:
    number = sum(1 for c in cases if c["property"] == prop) + 1
    cases.append({
        "id": f"{tier[:4]}-{prop.replace('.', '-')}-{number:03d}",
        "tier": tier, "property": prop, "text": text, "language": language,
        "asserts": asserts, "origin": origin, "note": note})


def norm(prop, text, asserts, **kw):
    case("normalization", prop, text, asserts, **kw)


def dep(prop, text, asserts, **kw):
    case("dependency", prop, text, asserts, **kw)


def sem(prop, text, asserts, **kw):
    case("semantics", prop, text, asserts, **kw)


# ── Tier 1: normalization ────────────────────────────────────────────────────
# Characters to values. Assertable today against `syntax.normalize`.

# money — written forms, multipliers, currencies
for text, amount, unit in [
    ("$500", "500", "USD"), ("$1,500", "1500", "USD"),
    ("$1k", "1000", "USD"), ("$1.5k", "1500", "USD"),
    ("$2K", "2000", "USD"), ("$3m", "3000000", "USD"),
    ("$2.5m", "2500000", "USD"), ("$1bn", "1000000000", "USD"),
    ("£500", "500", "GBP"), ("£1,250", "1250", "GBP"),
    ("£2k", "2000", "GBP"), ("€500", "500", "EUR"),
    ("€1.2k", "1200", "EUR"), ("€10m", "10000000", "EUR"),
    ("500 dollars", "500", "USD"), ("1,000 dollars", "1000", "USD"),
    ("250 USD", "250", "USD"), ("400 usd", "400", "USD"),
    ("300 pounds", "300", "GBP"), ("750 GBP", "750", "GBP"),
    ("600 euros", "600", "EUR"), ("900 EUR", "900", "EUR"),
    ("$0.50", "0.50", "USD"), ("$12,345.67", "12345.67", "USD"),
]:
    norm("money.amount", f"contribute {text} each month",
         {"kind": "money", "canonical": amount, "unit": unit})

norm("money.absent", "hold 90 shares of VTI", {"absent": "money"},
     origin="falsification",
     note="a bare number is not an amount; if it were, every window and every "
          "duration would be one too")
norm("money.absent", "buy 200 units", {"absent": "money"},
     origin="falsification")
norm("money.currency_is_not_assumed", "invest £500 a month",
     {"kind": "money", "unit": "GBP"}, origin="falsification",
     note="dropping the symbol makes every non-dollar prompt silently wrong "
          "rather than unsupported")

# percentages
for text, fraction in [
    ("60%", "0.60"), ("40%", "0.40"), ("4.5%", "0.045"),
    ("0.5%", "0.005"), ("100%", "1.00"), ("7 %", "0.07"),
    ("12.75%", "0.1275"), ("3%", "0.03"),
]:
    norm("percentage.fraction", f"allocate {text} to equities",
         {"kind": "percentage", "canonical": fraction})

# ratios — a split only when the parts sum to 100
for text, parts in [
    ("60/40", [60, 40]), ("70/30", [70, 30]), ("80/20", [80, 20]),
    ("50/50", [50, 50]), ("90/10", [90, 10]), ("45/55", [45, 55]),
    ("70/20/10", [70, 20, 10]), ("60/30/10", [60, 30, 10]),
    ("40/40/20", [40, 40, 20]), ("25/25/25/25", [25, 25, 25, 25]),
]:
    norm("ratio.split", f"a {text} portfolio",
         {"kind": "ratio", "canonical": parts})

# Hyphenated splits. Every invented case used a slash; four of the first
# twenty-nine sentences harvested from Stack Exchange wrote a hyphen, and the
# normaliser read none of them.
for text, parts in [
    ("I start with a 60-40 equity to fixed asset allocation", [60, 40]),
    ("my allocation becomes 80-20", [80, 20]),
    ("rebalance down to 60-40", [60, 40]),
    ("I really want an 80-20 split", [80, 20]),
    ("a 50-50 split between the two", [50, 50]),
]:
    norm("ratio.split_written_with_a_hyphen", text,
         {"kind": "ratio", "canonical": parts}, origin="observed",
         note="attested on money.stackexchange.com; the slash-only rule read "
              "nothing at all here")

norm("ratio.not_a_split", "upgraded from 2012 to 2015", {"absent": "ratio"},
     origin="observed",
     note="the sums-to-100 test does the same work for hyphens as for slashes")

# Ranges. Both of these were read *wrongly* rather than not at all, which is
# the worse failure — a plausible number for a request nobody made.
norm("range.percentage_is_refused", "10-20% of my allocation",
     {"absent": "percentage"}, origin="observed",
     note="attested; came back as 20%, silently collapsing to the upper bound")
norm("range.money_is_refused", "currently make ~$200-$220k per year",
     {"absent": "money"}, origin="observed",
     note="attested; came back as 200 and 220000, so the low end was out by a "
          "factor of a thousand — the multiplier at the far end governs both, "
          "and a reader taking the first match cannot know that")
norm("range.money_is_refused", "somewhere between $500 and $800 a month",
     {"absent": "money"}, origin="falsification",
     note="the worded form of the same thing")

norm("duration.an_age_is_not_a_horizon", "Me - 32 years old, currently",
     {"absent": "duration"}, origin="observed",
     note="attested; came back as an 11,680-day duration. A biography is not "
          "a backtest length")
norm("duration.an_age_is_not_a_horizon", "I am 45 years old and retiring soon",
     {"absent": "duration"}, origin="falsification")
norm("duration.days", "an investment horizon of 50 years",
     {"kind": "duration", "canonical": 18250, "unit": "days"},
     origin="observed",
     note="the discriminating opposite: the age rule must not eat real "
          "horizons, which are written the same way minus one word")

norm("ratio.order_is_content", "a 70/30 portfolio",
     {"kind": "ratio", "canonical": [70, 30]}, origin="falsification",
     note="70/30 is not 30/70; flattening a split to a set loses the sentence")
for text in ["due 12/25", "3/4 of the way", "section 9/11", "a 2/3 majority"]:
    norm("ratio.not_a_split", text, {"absent": "ratio"},
         origin="falsification",
         note="parts not summing to 100 are a date, a fraction or an "
              "identifier; guessing which would be a substitution")

# durations
for text, days in [
    ("90 days", 90), ("30 days", 30), ("1 day", 1), ("14 days", 14),
    ("2 weeks", 14), ("6 weeks", 42), ("1 week", 7),
    ("3 months", 90), ("6 months", 180), ("18 months", 540),
    ("12 months", 360), ("1 month", 30),
    ("5 years", 1825), ("10 years", 3650), ("1 year", 365),
    ("20 years", 7300), ("2 yrs", 730),
    ("18-month", 540), ("90-day", 90), ("5-year", 1825),
]:
    norm("duration.days", f"hold the position for {text}",
         {"kind": "duration", "canonical": days, "unit": "days"})

# moving-average windows
for text, window, unit in [
    ("200-day moving average", 200, "day"),
    ("50-day moving average", 50, "day"),
    ("90 day moving average", 90, "day"),
    ("20-day SMA", 20, "day"), ("100-day sma", 100, "day"),
    ("12-month moving average", 12, "month"),
    ("10-week moving average", 10, "week"),
    ("50 day MA", 50, "day"), ("200-day ma", 200, "day"),
    ("30-day EMA", 30, "day"),
]:
    norm("window.moving_average", f"buy when it falls below the {text}",
         {"kind": "moving_average_window", "canonical": window, "unit": unit})

norm("window.not_a_duration", "buy below the 90-day moving average",
     {"absent": "duration"}, origin="falsification",
     note="the plan's own case: reading 90 as a holding period must fail. One "
          "span carries one reading")
norm("window.not_a_duration", "sell under the 200-day moving average",
     {"absent": "duration"}, origin="falsification")
norm("duration.not_a_window", "hold the annual bonus for 90 days",
     {"absent": "moving_average_window"}, origin="falsification",
     note="the mirror case, and the one a single-pass regex gets wrong")
norm("duration.not_a_window", "keep the cash for 200 days",
     {"absent": "moving_average_window"}, origin="falsification")

norm("span.both_survive_one_sentence",
     "hold the bonus for 90 days, then buy below the 200-day moving average",
     {"kinds": ["duration", "moving_average_window"]}, origin="falsification",
     note="two values, two spans, no overlap — the case that motivated "
          "claiming spans rather than emitting every match")
norm("span.order_is_reading_order",
     "put $500 monthly into a 60/40 split for 10 years",
     {"kinds": ["money", "ratio", "duration"]})

# ── Tier 2: dependency ───────────────────────────────────────────────────────
# Which token governs which. Needs a parse; asserts one edge.

for text, dependent, head in [
    ("contribute $500 monthly", "monthly", "contribute"),
    ("invest $500 monthly", "monthly", "invest"),
    ("deposit $200 weekly", "weekly", "deposit"),
    ("add $1,000 quarterly", "quarterly", "add"),
    ("save $300 every month", "month", "save"),
    ("buy $500 of VTI monthly", "monthly", "buy"),
]:
    dep("cadence.attaches_to_contribution", text,
        {"dependent": dependent, "head_lemma": head})

for text, dependent, head in [
    ("rebalance the portfolio annually", "annually", "rebalance"),
    ("rebalanced monthly", "monthly", "rebalance"),
    ("harvest losses daily", "daily", "harvest"),
    ("adjust the weights quarterly", "quarterly", "adjust"),
    ("review the allocation yearly", "yearly", "review"),
    ("withdraw $500 monthly", "monthly", "withdraw"),
]:
    dep("cadence.attaches_to_other_verb", text,
        {"dependent": dependent, "head_lemma": head}, origin="falsification",
        note="attaching this cadence to a contribution verb must fail")

dep("cadence.two_in_one_sentence",
    "invest $500 monthly and rebalance annually",
    {"dependent": "monthly", "head_lemma": "invest"}, origin="falsification",
    note="the sentence the whole layer exists for: a reader that collects "
         "cadences finds two and picks one")
dep("cadence.two_in_one_sentence",
    "invest $500 monthly and rebalance annually",
    {"dependent": "annually", "head_lemma": "rebalance"},
    origin="falsification")
dep("cadence.two_in_one_sentence",
    "contribute weekly, rebalanced quarterly",
    {"dependent": "weekly", "head_lemma": "contribute"},
    origin="falsification")

dep("cadence.attaches_to_an_unscored_verb", "gift $500 monthly",
    {"dependent": "monthly", "head_lemma": "gift"}, origin="falsification",
    note="`gift` is in neither the supporting nor the opposing table, and the "
         "scorer must return zero rather than a guess — the case that keeps "
         "`against` from becoming the complement of `supports`")

for text, dependent, head in [
    ("buy VOO when SPY crosses below its average", "SPY", "cross"),
    ("purchase VTI if QQQ drops", "QQQ", "drop"),
    ("buy BND while TLT stays below trend", "TLT", "stay"),
]:
    dep("condition.subject_of_the_trigger", text,
        {"dependent": dependent, "head_lemma": head}, origin="falsification",
        note="the observed asset is the subject of the crossing, not the "
             "thing being bought")

for text, dependent, head in [
    ("buy VOO when SPY crosses below its average", "VOO", "buy"),
    ("purchase VTI if QQQ drops", "VTI", "purchase"),
    ("sell BND after the signal", "BND", "sell"),
]:
    dep("condition.object_of_the_purchase", text,
        {"dependent": dependent, "head_lemma": head}, origin="falsification",
        note="held and observed are different roles; swapping them must fail")

for text, dependent, head in [
    ("buy at the next open", "open", "buy"),
    ("execute at the close", "close", "execute"),
    ("invest on the first session of the month", "session", "invest"),
]:
    dep("timing.attaches_to_execution", text,
        {"dependent": dependent, "head_lemma": head})

for text, dependent, head in [
    ("hold the bonus for 90 days", "days", "hold"),
    ("keep the cash for six months", "months", "keep"),
    ("evaluate over five years", "years", "evaluate"),
    ("measure across ten years", "years", "measure"),
]:
    dep("window.attaches_to_its_verb", text,
        {"dependent": dependent, "head_lemma": head})

dep("timing.case_marker", "since Jan 2020",
    {"dependent": "since", "head_lemma": "jan", "relation": "case"},
    origin="observed",
    note="UD again: the month is the head and `since` attaches to it. Written "
         "the other way round first, and the recorded parse disagreed")

for text, negated in [
    ("rather than through an ETF", "ETF"),
    ("not through a mutual fund", "fund"),
    ("instead of buying bonds", "bonds"),
    ("without using leverage", "leverage"),
]:
    dep("negation.is_carried", text, {"negated": negated},
        origin="falsification",
        note="dropping the negation reverses the sentence")

dep("negation.absent_when_absent", "through an ETF", {"negated": None},
    origin="falsification",
    note="the discriminating opposite: if everything looked negated, the "
         "signal would carry no information")

# ── Tier 3: semantics ────────────────────────────────────────────────────────
# Which field a value fills, once syntax and the model have both spoken.

for text, field, value in [
    ("invest $500 monthly", "cadence", "monthly"),
    ("contribute $500 monthly, rebalanced annually", "cadence", "monthly"),
    ("add $250 every two weeks", "cadence", "biweekly"),
    ("put in $1,000 each quarter", "cadence", "quarterly"),
    ("deposit $100 a week", "cadence", "weekly"),
    ("a one-off $10,000 investment", "cadence", "once"),
]:
    sem("funding.cadence", text, {"field": field, "value": value})

for text, value in [
    ("invest $500 monthly", "500"),
    ("contribute £1k a month", "1000"),
    ("add $2.5k quarterly", "2500"),
]:
    sem("funding.amount", text, {"field": "amount", "value": value})

for text, value in [
    ("a 60/40 portfolio", "60/40"),
    ("split it 70/30 between stocks and bonds", "70/30"),
    ("equal weight across the four", "equal_weight_at_purchase"),
    ("weight by inverse volatility", "inverse_volatility"),
    ("market-cap weighted", "market_cap"),
]:
    sem("weighting.method", text, {"field": "allocation_method", "value": value})

for text, value in [
    ("when SPY crosses below its 200-day average", "crossing_event"),
    ("whenever SPY drops under the 200-day", "crossing_event"),
    ("while SPY stays below its average", "persistent_condition"),
    ("as long as SPY is under trend", "persistent_condition"),
    ("any time SPY is below the 200-day", "persistent_condition"),
]:
    sem("trigger.semantics", text,
        {"field": "trigger_semantics", "value": value},
        origin="falsification",
        note="crosses-below is an event; stays-below is a state. Collapsing "
             "them changes how often the strategy fires")

for text, held, observed in [
    ("buy VOO when SPY crosses below its average", "VOO", "SPY"),
    ("purchase VTI whenever QQQ drops 10%", "VTI", "QQQ"),
    ("add to BND while TLT is under its 200-day", "BND", "TLT"),
]:
    sem("trigger.asset_roles", text,
        {"held": held, "observed": observed}, origin="falsification",
        note="swapping held and observed must fail")

for text, value in [
    ("below the 200-day moving average", "200"),
    ("under its 50-day average", "50"),
    ("beneath the 12-month moving average", "12"),
]:
    sem("window.moving_average", text,
        {"field": "moving_average_window", "value": value})

for text, value in [
    ("hold the bonus for 90 days", "90"),
    ("keep it for six months", "180"),
]:
    sem("window.holding_period", text,
        {"field": "holding_period_days", "value": value},
        origin="falsification",
        note="a holding period is not a moving-average window, even when both "
             "are 90")

for text, value in [
    ("through an ETF", "etf"),
    ("rather than through an ETF", "not_etf"),
    ("directly, not through a fund", "not_fund"),
]:
    sem("negation.changes_the_value", text,
        {"field": "instrument", "value": value}, origin="falsification")


# ── coordination, which is where clause boundaries actually break ────────────

for text, dependent, head in [
    ("buy VTI and BND monthly", "monthly", "buy"),
    ("hold VTI, BND and GLD, rebalanced yearly", "yearly", "rebalance"),
    ("invest monthly and withdraw quarterly", "monthly", "invest"),
    ("invest monthly and withdraw quarterly", "quarterly", "withdraw"),
    ("contribute $500 and $200 to the second sleeve", "$500", "contribute"),
]:
    dep("coordination.scope", text, {"dependent": dependent, "head_lemma": head},
        origin="falsification",
        note="a conjunction is where a reader most often carries a modifier "
             "across a boundary it does not cross")

for text, dependent, head in [
    ("invest $500 monthly for the past five years", "years", "invest"),
    ("every month for the past five years", "years", "month"),
    ("hold the bonus for 90 days each year", "days", "hold"),
]:
    dep("temporal.window_versus_cadence", text,
        {"dependent": dependent, "head_lemma": head}, origin="falsification",
        note="'every month for five years' has a cadence and an evaluation "
             "period; reading the period as the cadence is the common miss")

# Direction is carried by the *case marker*, not by the noun hanging off a
# preposition. Universal Dependencies makes the content word the head and
# attaches `from`/`to` to it with `case` — the first version of these cases
# asserted the reverse and the recorded parses said so immediately, which is
# the corpus doing its job on its own first run.
for text, marker, account in [
    ("move the cash from the ISA to the SIPP", "from", "isa"),
    ("move the cash from the ISA to the SIPP", "to", "sipp"),
    ("transfer from the 401k into the IRA", "from", "401k"),
    ("transfer from the 401k into the IRA", "into", "ira"),
]:
    dep("relation.direction", text,
        {"dependent": marker, "head_lemma": account, "relation": "case"},
        origin="falsification",
        note="from and to reversed is the opposite transaction; the contract "
             "makes member order part of identity for exactly this")

# Which weight belongs to which sleeve is a *shared head*: `60` is `nummod` of
# a `%`, and `VTI` is `nmod` of the same `%`. Asserting a head lemma would not
# distinguish the two `%` tokens in one sentence, which is the case that
# matters.
for text, weight, asset in [
    ("60% to VTI and 40% to BND", "60", "VTI"),
    ("60% to VTI and 40% to BND", "40", "BND"),
    ("put 70% in equities, 30% in bonds", "70", "equities"),
    ("put 70% in equities, 30% in bonds", "30", "bonds"),
]:
    dep("relation.member_qualifier", text,
        {"dependent": weight, "shares_head_with": asset},
        origin="falsification",
        note="a set of weights beside a set of assets is not the same fact as "
             "knowing which belongs to which")

for text, dependent, head in [
    ("invest what is left after expenses", "expenses", "leave"),
    ("contribute whatever remains each month", "month", "remain"),
    ("buy on the first Monday of the month", "Monday", "buy"),
    ("execute at the following session's open", "open", "execute"),
    ("start in January 2020", "January", "start"),
]:
    dep("timing.attaches_to_execution", text,
        {"dependent": dependent, "head_lemma": head})

# ── more semantics ───────────────────────────────────────────────────────────

for text, value in [
    ("invest whatever is left over each month", "residual"),
    ("contribute a fixed $500", "fixed"),
    ("invest 10% of my salary monthly", "proportional"),
    ("put in half of any bonus", "proportional"),
]:
    sem("funding.amount_kind", text, {"field": "amount_kind", "value": value})

for text, value in [
    ("buy at the next open", "next_session_open"),
    ("execute at the close", "same_session_close"),
    ("trade at the following day's open", "next_session_open"),
    ("fill on the signal bar", "same_session_close"),
]:
    sem("timing.execution", text,
        {"field": "execution_timing", "value": value})

for text, value in [
    ("on the first trading day of the month", "first_session_of_period"),
    ("on the last session of each quarter", "last_session_of_period"),
    ("mid-month", "mid_period"),
]:
    sem("timing.day_rule", text, {"field": "day_rule", "value": value})

for text, value in [
    ("evaluate over the past five years", "1825"),
    ("measured across ten years", "3650"),
    ("since January 2020", "2020-01"),
]:
    sem("window.evaluation_period", text,
        {"field": "evaluation_period", "value": value})

for text, held in [
    ("buy a core index fund monthly", "a core index fund"),
    ("invest in the S&P 500 tracker", "the S&P 500 tracker"),
    ("put it into an SPX ETF", "an SPX ETF"),
]:
    sem("assets.stay_as_written", text, {"field": "assets", "value": held},
        origin="falsification",
        note="never resolved to a ticker here. Choosing VTI or SPY on the "
             "user's behalf is the substitution the whole boundary prevents")

for text, value in [
    ("reinvest the dividends", "reinvested"),
    ("take the dividends as cash", "cash"),
    ("do not reinvest", "cash"),
]:
    sem("dividends.policy", text,
        {"field": "dividend_policy", "value": value},
        note="Mission refuses this dimension by name — it is here because the "
             "schema is what can be *meant*, not what can be run")

# ── a first multilingual slice ───────────────────────────────────────────────
#
# Not coverage. Five sentences per language would be a claim this corpus cannot
# support. These exist so the schema's `language` field is exercised by
# something real, and so the first Trankit comparison has somewhere to start.

for language, text, amount, unit in [
    ("es", "invertir 500 € cada mes", "500", "EUR"),
    ("de", "monatlich 500 € investieren", "500", "EUR"),
    ("fr", "investir 500 € par mois", "500", "EUR"),
    ("es", "aportar 1.000 € al mes", "1000", "EUR"),
]:
    norm("money.amount", text, {"kind": "money", "canonical": amount,
                                "unit": unit}, language=language,
         origin="observed",
         note="postfix currency, which the first normaliser missed entirely — "
              "it read only `€500` and this corpus caught it on the first run")

for language, text, amount in [
    ("de", "monatlich 500,50 € investieren", "500.50"),
    ("fr", "investir 1.234,56 € par mois", "1234.56"),
    ("es", "aportar 2.500,75 € al mes", "2500.75"),
]:
    norm("money.grouping_follows_the_language", text,
         {"kind": "money", "canonical": amount}, language=language,
         origin="observed",
         note="under a declared convention this is not ambiguous at all. The "
              "first normaliser had no language and therefore silently assumed "
              "en, which is not neutrality")

for language, text in [
    ("en", "invest 500,50 EUR monthly"),
    ("en", "contribute 1.234,56 EUR each month"),
]:
    norm("money.malformed_for_the_language_is_refused", text,
         {"absent": "money"}, language=language, origin="falsification",
         note="`500,50` is not a well-formed en number — comma before two "
              "digits is no thousands group. Refused rather than read under "
              "the other convention, because a plausible wrong amount is worse "
              "than a question")

norm("money.grouping_follows_the_language", "invest $1.000 monthly",
     {"kind": "money", "canonical": "1.000"}, language="en",
     origin="falsification",
     note="one dollar, not a thousand. Correct under the declared convention "
          "and surprising to a European reader, which is exactly why the "
          "convention is declared rather than inferred")

for language, text, dependent, head in [
    ("es", "invertir 500 € cada mes, reequilibrado anualmente",
     "anualmente", "reequilibrar"),
    ("de", "monatlich investieren, jährlich neu gewichten",
     "jährlich", "gewichten"),
    ("fr", "investir chaque mois, rééquilibré chaque année",
     "année", "rééquilibrer"),
    ("ru", "инвестировать ежемесячно, ребалансировать ежегодно",
     "ежегодно", "ребалансировать"),
]:
    dep("cadence.attaches_to_other_verb", text,
        {"dependent": dependent, "head_lemma": head}, language=language,
        origin="falsification",
        note="the same two-cadence sentence as the English case. If the "
             "layer generalises at all, it generalises here")


if __name__ == "__main__":
    tiers = {}
    for one in cases:
        tiers[one["tier"]] = tiers.get(one["tier"], 0) + 1
    document = {"schema": SCHEMA, "count": len(cases), "by_tier": tiers,
                "cases": cases}
    OUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases -> {OUT}")
    for tier, count in sorted(tiers.items()):
        print(f"  {tier:15} {count}")
    falsification = sum(1 for c in cases if c["origin"] == "falsification")
    print(f"  {'falsification':15} {falsification} of {len(cases)}")
