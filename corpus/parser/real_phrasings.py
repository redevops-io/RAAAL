"""Attested phrasings — how people actually write it, not how we imagined it.

`cases.json` is self-authored: I wrote both the sentences and the expectations,
which is the weakest evidence this project recognises. This pack exists to put
real language beside it, grouped by the parser property each example stresses
and **not** labelled with an expected plan. Labelling them would re-introduce
the same defect one layer up.

**Provenance is recorded per entry and is not uniform.** Bogleheads returns HTTP
402 to this environment's fetcher and reddit.com blocks it; those are stated
preferences about automated reading, not obstacles to route around, so neither
was scraped. Stack Exchange publishes an API for exactly this and licenses its
content CC-BY-SA, so that is where the verbatim entries come from. Each entry
says where it actually came from:

    user_reported   quoted by the user from a cited Bogleheads thread
    search_summary  appeared in a search-result summary I read
    stackexchange   verbatim from a Stack Exchange post under CC-BY-SA 4.0,
                    pulled through the public API, with its question URL
    variant         a minimal variation I wrote on an attested form, to
                    isolate one property — marked so it is never mistaken for
                    attested language

Anything marked `variant` is mine and carries the same discount as
`cases.json`. It is still not a sample of how people write — the searches were
chosen to surface descriptions of actions, and that is a selection — but the
attested share is now real text with a URL rather than recollection. See
`docs/parser-corpus.md` for what the harvest found.

Run: `python corpus/parser/real_phrasings.py`
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "quantify-real-phrasings@1"
OUT = Path(__file__).resolve().parent / "real_phrasings.json"

#: The threads the attested entries came from, as cited by the user.
SOURCES = {
    "bh-447768": "https://www.bogleheads.org/forum/viewtopic.php?t=447768",
    "bh-343589": "https://www.bogleheads.org/forum/viewtopic.php?t=343589",
    "bh-lazy": "https://www.bogleheads.org/wiki/How_to_build_a_lazy_portfolio",
    "bh-446034": "https://www.bogleheads.org/forum/viewtopic.php?t=446034",
    "bh-459742": "https://www.bogleheads.org/forum/viewtopic.php?t=459742",
    "bh-457074": "https://www.bogleheads.org/forum/viewtopic.php?t=457074",
}

entries: list = []


def phrase(group: str, text: str, stresses: str, *, provenance: str,
           source: str = "", note: str = "") -> None:
    entries.append({
        "id": f"real-{group}-{sum(1 for e in entries if e['group'] == group) + 1:03d}",
        "group": group, "text": text, "stresses": stresses,
        "provenance": provenance, "source": SOURCES.get(source, source),
        "note": note})


# ── allocation ───────────────────────────────────────────────────────────────
# Ratios with the noun left out entirely, which the constructed corpus never
# did — every invented case wrote "a 60/40 portfolio".

phrase("allocation", "60/40", "ratio with no head noun at all",
       provenance="user_reported", source="bh-447768")
phrase("allocation", "maintain the asset allocation of 60/40",
       "ratio as the complement of a nominalisation",
       provenance="user_reported", source="bh-447768")
phrase("allocation", "I'm 60/40", "ratio as a predicate of the speaker",
       provenance="variant",
       note="the compressed form the user named; a reader looking for a noun "
            "to attach the ratio to finds a pronoun")
phrase("allocation", "move to 60/40", "ratio as a goal of a change verb",
       provenance="variant")
phrase("allocation", "maintain 60/40", "ratio as the object of a hold verb",
       provenance="variant")
phrase("allocation", "70/30 vs 60/40", "two ratios, neither one the target",
       provenance="user_reported", source="bh-446034",
       note="thread title. A consumer taking the first ratio gets a "
            "comparison, not a portfolio")
phrase("allocation", "my 60/40 is acting like 70/30",
       "target ratio and observed ratio in one sentence",
       provenance="user_reported", source="bh-457074",
       note="paraphrased from the thread title 'Asset Allocation - 70/30 "
            "acting like 60/40'; the ratios here are deliberately in the "
            "order that makes taking-the-first wrong")
phrase("allocation", "401k (50/50), Roth IRA (85/15), taxable brokerage (70/30)",
       "three ratios, each bound to a different account",
       provenance="user_reported", source="bh-lazy",
       note="the case a scalar cannot hold. Which ratio belongs to which "
            "account is the whole content")

# ── rebalancing ──────────────────────────────────────────────────────────────

phrase("rebalancing", "rebalance at year end", "worded period, no digits",
       provenance="user_reported", source="bh-447768")
phrase("rebalancing", "buy-hold-and rebalance", "hyphenated verb compound",
       provenance="user_reported", source="bh-lazy")
phrase("rebalancing", "maintain 60/40 by contributions and rebalance at year end",
       "allocation, funding method and rebalancing cadence in one clause chain",
       provenance="user_reported", source="bh-447768",
       note="the sentence the whole layer is for")
phrase("rebalancing", "rebalance with new contributions",
       "rebalancing whose mechanism is the contribution",
       provenance="search_summary",
       note="'rebalance with dividends and new contributions' appeared in a "
            "search summary of Bogleheads material")
phrase("rebalancing", "I rebalance in my birthday month",
       "cadence expressed as a named recurring point",
       provenance="search_summary",
       note="'rebalance ... during their review period, such as their "
            "birthday month' appeared in a search summary")
phrase("rebalancing", "rebalance when it drifts 5% from target",
       "threshold band expressed as a percentage",
       provenance="variant")
phrase("rebalancing", "I use a 5/25 band", "ratio-shaped notation that is not a split",
       provenance="search_summary", source="bh-459742",
       note="the 5%/25% rebalancing band. Shaped exactly like an allocation "
            "and meaning something else entirely")
phrase("rebalancing", "rebalance annually or when a band is breached",
       "calendar cadence and threshold as alternatives",
       provenance="variant")
phrase("rebalancing", "don't know how to rebalance/reallocate",
       "the two senses the user cannot separate either",
       provenance="user_reported", source="bh-459742",
       note="thread title. Evidence that the ambiguity is in the language, "
            "not only in the reader — the case for asking rather than scoring")

# ── cadence ──────────────────────────────────────────────────────────────────

phrase("cadence", "I contribute monthly and rebalance at year end",
       "two cadences, compressed, no amounts",
       provenance="variant",
       note="the minimal form of the attested sentence above")
phrase("cadence", "invest as soon as it is available",
       "cadence expressed as a condition, not a period",
       provenance="search_summary",
       note="'the best time to invest money is as soon as it is available' "
            "appeared in a search summary")
phrase("cadence", "max the 401k every year and the Roth in January",
       "one cadence per account in one sentence",
       provenance="variant")

# ── funding ──────────────────────────────────────────────────────────────────

phrase("funding", "$500 a month", "amount and cadence as one noun phrase",
       provenance="variant")
phrase("funding", "I put in whatever is left at month end",
       "residual amount with a timing phrase",
       provenance="variant")
phrase("funding", "rebalance with dividends and new contributions",
       "two funding sources coordinated",
       provenance="search_summary")

# ── trigger ──────────────────────────────────────────────────────────────────

phrase("trigger", "when the allocation exceeds a predefined band",
       "trigger on a derived quantity, not a price",
       provenance="search_summary",
       note="'rebalancing when asset allocation exceeds a predefined band' "
            "appeared in a search summary")
phrase("trigger", "buy when it falls below its 200-day",
       "elided head noun after the window",
       provenance="variant",
       note="real writing drops 'moving average' constantly")
phrase("trigger", "sell when it drops under its 50-day average",
       "window written as `N-day average`, without `moving`",
       provenance="variant",
       note="OBSERVED GAP: `_WINDOW` requires `moving average|ma|sma|ema`, so "
            "this reads as a 50-day *duration* and fills holding_period_days. "
            "Recorded rather than patched — `X-day average` is a window in this "
            "domain and not in others, and widening tier 1 on one case is how a "
            "normaliser starts guessing")

# ── window ───────────────────────────────────────────────────────────────────

phrase("window", "over the last 10 years", "evaluation period, worded",
       provenance="variant")
phrase("window", "for retirement in 10 years", "horizon that is not a backtest window",
       provenance="user_reported", source="bh-343589",
       note="from the thread title. A reader that treats every duration as an "
            "evaluation period reads a retirement date as a backtest length")

# ── timing ───────────────────────────────────────────────────────────────────

phrase("timing", "at year end", "period boundary with no digits",
       provenance="user_reported", source="bh-447768")
phrase("timing", "on my birthday", "recurring point that is not a calendar period",
       provenance="search_summary")
phrase("timing", "as soon as it is available", "timing as a condition",
       provenance="search_summary")


# ── harvested from Stack Exchange, reviewed ──────────────────────────────────
#
# Verbatim, under CC-BY-SA 4.0, each with the question it came from. Pulled by
# `harvest_stackexchange.py` through the public API — Bogleheads and reddit
# block automated fetchers, which is a stated preference rather than an
# obstacle to route around, so neither was scraped.
#
# These are the ones I read and kept. The full candidate list is in
# `stackexchange_candidates.json`; nothing merges automatically, because an
# unreviewed sentence is a sentence nobody has read.

SE = "https://money.stackexchange.com"

for text, group, stresses_it, source in [
    ("I start with a 60-40 equity to fixed asset allocation",
     "allocation", "split written with a hyphen, not a slash",
     f"{SE}/questions/1849"),
    ("After year 1 my allocation becomes 80-20",
     "allocation", "hyphenated split as the result of drift",
     f"{SE}/questions/1849"),
    ("Why do I want to rebalance down to 60-40?",
     "rebalancing", "target ratio as the goal of a rebalance verb",
     f"{SE}/questions/1849"),
    ("Possibly my initial asset allocation was not correct and I really want an 80-20 split",
     "allocation", "target change rather than a rebalance",
     f"{SE}/questions/1849"),
    ("I understand that rebalancing once a year is better than doing it once every two years",
     "rebalancing", "two worded cadences compared in one sentence",
     f"{SE}/questions/1849"),
    ("I also have a 401k (contributing about $13k/year) and an emergency fund",
     "funding", "amount and cadence fused by a slash, inside a parenthetical",
     f"{SE}/questions/2789"),
    ("There is about $1800 in each account and we are automatically contributing $50 to each account each month",
     "funding", "balance and contribution in one sentence, both bare amounts",
     f"{SE}/questions/2789"),
    ("The portfolio I have in mind has a 73/23/4% allocation in stocks/bonds/other",
     "allocation", "three-way split with a trailing percent and a slashed noun list",
     f"{SE}/questions/2789"),
    ("I have recently began to question the wisdom of a 100% stock allocation",
     "allocation", "allocation as a single percentage, no ratio",
     f"{SE}/questions/2789"),
    ("I have an investment horizon of 50 years",
     "window", "horizon stated as a duration",
     f"{SE}/questions/2789"),
    ("My employer will contribute $100 to the HSA, and will match another 400 of my contributions",
     "funding", "a second amount with no currency symbol at all",
     f"{SE}/questions/2789"),
    ("To do this, I would have to divert a portion of my current monthly contributions",
     "funding", "proportional amount with no number",
     f"{SE}/questions/2789"),
    ("I'm using Markowitz Mean-Variance Optimization to rebalance my portfolio of 30 holdings",
     "rebalancing", "a count that is not an amount, beside a rebalance verb",
     f"{SE}/questions/2789"),
]:
    phrase(group, text, stresses_it, provenance="stackexchange", source=source,
           note="verbatim, CC-BY-SA 4.0")


if __name__ == "__main__":
    groups: dict = {}
    provenance: dict = {}
    for one in entries:
        groups[one["group"]] = groups.get(one["group"], 0) + 1
        provenance[one["provenance"]] = provenance.get(one["provenance"], 0) + 1

    OUT.write_text(json.dumps(
        {"schema": SCHEMA, "count": len(entries),
         "by_group": groups, "by_provenance": provenance,
         "collection_note": (
             "Bogleheads returns HTTP 402 to this environment's fetcher and "
             "reddit.com blocks it; those are stated preferences about "
             "automated reading, not obstacles to route around, so neither was "
             "scraped. The stackexchange entries are verbatim under CC-BY-SA "
             "4.0 through the public API, each with its question URL. The rest "
             "were quoted by the user from cited threads, read in "
             "search-result summaries, or are `variant` sentences of mine."),
         "licence_note": (
             "Entries with provenance 'stackexchange' are quoted from Stack "
             "Exchange under CC-BY-SA 4.0; the source URL is the attribution "
             "the licence requires."),
         "entries": entries}, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(entries)} phrasings -> {OUT}")
    for group, count in sorted(groups.items()):
        print(f"  {group:14} {count}")
    print("  --")
    for kind, count in sorted(provenance.items()):
        print(f"  {kind:14} {count}")
