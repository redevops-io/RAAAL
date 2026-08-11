"""Harvests attested phrasings from Stack Exchange, with attribution.

    python corpus/parser/harvest_stackexchange.py           # write candidates
    python corpus/parser/harvest_stackexchange.py --dry-run # print only

**Why this source and not the obvious ones.** Bogleheads returns HTTP 402 to
automated fetchers and reddit.com blocks them; those are deliberate signals
about how the sites wish to be read, and swapping a user agent to get round
them would be evading a stated preference rather than solving a technical
problem. Stack Exchange publishes an API for exactly this use, and its content
is CC-BY-SA — so every candidate carries its question URL and the licence,
which is both the attribution requirement and the provenance this pack needs.

**What it does not do.** It does not label meaning. Each candidate records the
verbatim sentence, where it came from, and which *pattern* matched it — the
parser property it stresses. Deciding what the sentence means is the job of the
layer under test, and a harvester that pre-decided would be writing its own
answer key.

Candidates land in `stackexchange_candidates.json` for review. They are not
merged into `real_phrasings.json` automatically: an unreviewed sentence is a
sentence nobody has read, and this pack's whole value is that its provenance is
known rather than assumed.
"""
from __future__ import annotations

import gzip
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "stackexchange_candidates.json"
API = "https://api.stackexchange.com/2.3/search/advanced"
LICENCE = "CC-BY-SA 4.0"

#: (site, query) — chosen to surface *descriptions of what someone does*, not
#: discussions about markets. The parser properties live in the first kind.
SEARCHES = [
    ("money", "rebalance asset allocation"),
    ("money", "rebalancing frequency portfolio"),
    ("money", "dollar cost averaging monthly contribution"),
    ("money", "401k Roth IRA allocation percentage"),
    ("money", "index fund portfolio allocation bonds stocks"),
    ("money", "automatic investment contribution schedule"),
    ("money", "moving average buy signal"),
    ("money", "target allocation drift threshold"),
    ("quant", "rebalancing frequency portfolio weights"),
    ("quant", "moving average crossover strategy"),
    ("quant", "backtest monthly contribution"),

    # Wealth management beyond accumulation. The pack was built almost entirely
    # from "how do I put money in" phrasings, which is one corner of what
    # people actually ask a planner. These searches cover the families that
    # corner was hiding: taking money out, moving it between account types,
    # and changing the shape of the portfolio over time.
    #
    # Most of what comes back will name capabilities Quantify does not have.
    # That is the point of collecting it: a refusal by name is only possible
    # for a dimension the reader can *recognise*, and an unrecognised strategy
    # is refused as an unparseable sentence instead — which tells the person
    # nothing about why.
    ("money", "safe withdrawal rate retirement 4% rule"),
    ("money", "retirement withdrawal strategy which account first"),
    ("money", "bucket strategy retirement cash years expenses"),
    ("money", "glide path equity allocation age bonds"),
    ("money", "target date fund allocation age"),
    ("money", "tax loss harvesting wash sale"),
    ("money", "Roth conversion ladder traditional IRA"),
    ("money", "asset location taxable account bonds IRA"),
    ("money", "required minimum distribution RMD withdraw"),
    ("money", "dividend income living off dividends"),
    ("money", "annuity guaranteed income retirement"),
    ("money", "529 plan HSA contribution strategy"),
    ("money", "emergency fund months expenses cash"),
    ("money", "pay off mortgage or invest"),
    ("quant", "risk parity leverage volatility target"),
    ("quant", "small cap value factor tilt portfolio"),
    ("quant", "momentum strategy monthly ranking"),
    ("quant", "covered call income strategy"),
    ("quant", "value averaging versus dollar cost averaging"),
    ("quant", "sequence of returns risk withdrawal"),

    # Widened to reach the volume the evaluation corpus needs. The first two
    # rounds produced 97 sentences, which is not enough for a recurrence
    # ranking to mean anything — a defect seen twice in 97 and a defect seen
    # twice in 400 are different claims. These searches add breadth rather than
    # depth: new families and new vocabulary for families already present, so
    # the pack does not simply get more of what it already had.
    ("money", "how should I invest my savings monthly"),
    ("money", "portfolio allocation stocks bonds international"),
    ("money", "lump sum versus dollar cost averaging"),
    ("money", "when should I sell my shares"),
    ("money", "stop loss limit order investing"),
    ("money", "buy and hold versus market timing"),
    ("money", "increase contributions with salary raise"),
    ("money", "three fund portfolio simple"),
    ("money", "bond ladder treasury duration"),
    ("money", "sell losers keep winners portfolio"),
    ("money", "inheritance windfall invest strategy"),
    ("money", "college savings withdraw tuition"),
    ("money", "match employer contribution percent salary"),
    ("money", "currency hedged international allocation"),
    ("money", "rental property versus index fund"),
    ("money", "gold commodities inflation hedge allocation"),
    ("money", "cash sitting on sidelines waiting"),
    ("money", "reduce risk before retirement"),
    ("money", "withdraw from taxable or retirement account first"),
    ("money", "social security claiming age delay"),
    ("quant", "mean reversion pairs trading rules"),
    ("quant", "equal weight versus cap weight index"),
    ("quant", "minimum variance portfolio optimization"),
    ("quant", "transaction costs turnover rebalancing"),
    ("quant", "trend following moving average rules"),
    ("quant", "stop loss drawdown control strategy"),
    ("quant", "portfolio rebalancing bands tolerance"),
    ("quant", "monte carlo retirement simulation withdrawal"),

    # Aimed at the thin groups. The first widening tripled `cadence` and left
    # `trigger`, `factor`, `timing` and `leverage` in single figures, which is
    # itself worth knowing: the authored corpus tested triggers heavily and the
    # attested language barely contains them. Searching *for* those families is
    # how to tell "people do not say this" from "these searches do not surface
    # it", and the two have opposite consequences for what to build.
    ("quant", "buy when price crosses above 200 day moving average"),
    ("quant", "signal threshold entry exit rules backtest"),
    ("quant", "rebalance when weight drifts more than 5 percent"),
    ("quant", "execute at open or close slippage"),
    ("quant", "momentum ranking top n assets monthly"),
    ("quant", "volatility targeting position sizing"),
    ("quant", "leveraged etf daily rebalancing decay"),
    ("quant", "factor exposure value momentum quality portfolio"),
    ("money", "invest when market drops percent buy the dip"),
    ("money", "sell when stock doubles rule"),
    ("money", "rebalance threshold percentage bands"),
    ("money", "tilt portfolio towards small cap value"),
    ("money", "margin loan borrow to invest"),
    ("money", "trade at market open or end of day"),
]

#: Which parser property a sentence stresses, by the shape of the sentence and
#: nothing else. Deliberately shallow: the moment this file starts inferring
#: intent it becomes an answer key.
PATTERNS = [
    ("allocation", re.compile(r"\b\d{1,3}\s?/\s?\d{1,3}\b|\ballocation\b", re.I)),
    ("rebalancing", re.compile(r"\brebalanc\w*|\breallocat\w*", re.I)),
    ("cadence", re.compile(r"\b(month|quarter|annual|year|week|dai)\w*\b", re.I)),
    ("funding", re.compile(r"\bcontribut\w*|\bdeposit\w*|\binvest\s+\$?\d", re.I)),
    ("trigger", re.compile(r"\b(when|whenever|if|once)\b.{0,40}"
                           r"\b(below|above|cross\w*|drift\w*|exceed\w*)\b", re.I)),
    ("window", re.compile(r"\b\d+[\s-]?(day|week|month|year)s?\b", re.I)),
    ("timing", re.compile(r"\b(year[- ]end|open|close|first|last)\b.{0,20}"
                          r"\b(day|session|month|quarter|year)\b", re.I)),

    # The families the pack was missing. Still shallow — these match the shape
    # of a sentence, never its meaning. A group here is a claim about which
    # parser property the sentence stresses, not about what it asks for.
    ("withdrawal", re.compile(r"\bwithdraw\w*|\bdraw(ing)?\s+down\b|"
                              r"\bdecumulat\w*|\bspend(ing)?\s+rate\b|"
                              r"\bsafe\s+withdrawal\b|\bSWR\b", re.I)),
    ("tax", re.compile(r"\btax[- ]loss\b|\bharvest\w*|\bwash\s+sale\b|"
                       r"\bcapital\s+gains?\b|\btax[- ]?(deferred|free|able)\b",
                       re.I)),
    ("conversion", re.compile(r"\bRoth\s+conver\w*|\bconvert\w*\s+"
                              r"(to|into)\s+(a\s+)?Roth\b|\bbackdoor\b", re.I)),
    ("account", re.compile(r"\b401\(?k\)?|\b403\(?b\)?|\bIRA\b|\bHSA\b|"
                           r"\b529\b|\bbrokerage\b|\btaxable\s+account\b|"
                           r"\bISA\b|\bSIPP\b", re.I)),
    ("glidepath", re.compile(r"\bglide\s?path\b|\btarget[- ]date\b|"
                             r"\bde[- ]?risk\w*|\bage\s+in\s+bonds\b|"
                             r"\bas\s+I\s+(get|grow)\s+older\b", re.I)),
    ("factor", re.compile(r"\bsmall[- ]cap\b|\bvalue\s+tilt\b|\bfactor\b|"
                          r"\bmomentum\b|\bquality\b|\btilt\w*", re.I)),
    ("income", re.compile(r"\bdividend\w*|\byield\b|\bcovered\s+call\b|"
                          r"\bannuit\w*|\bincome\s+(from|stream)\b", re.I)),
    ("leverage", re.compile(r"\bleverag\w*|\bmargin\b|\brisk\s+parity\b|"
                            r"\b[23]x\b|\bvol(atility)?\s+target\w*", re.I)),
    ("reserve", re.compile(r"\bemergency\s+fund\b|\bcash\s+(buffer|reserve)\b|"
                           r"\b(months?|years?)\s+of\s+expenses\b", re.I)),
]

#: A candidate must look like somebody describing an action, not asking an
#: open question about the world.
ACTIONISH = re.compile(
    r"\b(I|we|my|our)\b.{0,60}\b(invest|contribut|rebalanc|hold|buy|sell|"
    r"allocat|put|add|keep|move|maintain|target|"
    # Decumulation and tax verbs. Without these the filter admitted only
    # sentences about putting money in, which is how a corpus of 45 authentic
    # phrasings ended up describing one third of what people actually do.
    r"withdraw|draw|convert|harvest|tilt|glide|spend|shift|ladder|"
    r"defer|delay|annuitiz|reinvest|liquidat)\w*", re.I)

SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


#: How many pages of results to take per search.
#:
#: Volume here is bought by reading further down the same searches, not by
#: loosening the filters. That distinction matters: relaxing the sentence
#: window or the action filter would raise the count by admitting sentences
#: that are not strategy statements, and a corpus inflated that way reports a
#: survival rate for a population it does not have.
PAGES = 3


def fetch(site: str, query: str, pagesize: int = 60, page: int = 1) -> list:
    url = API + "?" + urllib.parse.urlencode({
        "order": "desc", "sort": "votes", "q": query, "site": site,
        "page": page, "pagesize": pagesize, "filter": "withbody"})
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    payload = json.loads(raw)
    if payload.get("quota_remaining", 1) < 20:
        print(f"  quota low: {payload.get('quota_remaining')} left", file=sys.stderr)
    return payload.get("items", []), payload.get("has_more", False)


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def stresses(sentence: str) -> list:
    return [name for name, pattern in PATTERNS if pattern.search(sentence)]


def candidates_from(item: dict) -> list:
    found = []
    body = clean(item.get("body", ""))
    for sentence in SENTENCE.split(body):
        sentence = sentence.strip(" -–—•*")
        # Short, because a paragraph asserts many things and this pack is built
        # on one property at a time. Long enough to carry a clause.
        if not (20 <= len(sentence) <= 120):
            continue
        if not ACTIONISH.search(sentence):
            continue
        groups = stresses(sentence)
        if len(groups) < 1:
            continue
        found.append({
            "text": sentence,
            "stresses": groups,
            "provenance": "stackexchange",
            "source": item["link"],
            "site": item.get("site", ""),
            "licence": LICENCE,
            "reviewed": False})
    return found


def main(dry_run: bool = False) -> int:
    seen, candidates = set(), []
    for site, query in SEARCHES:
        for page in range(1, PAGES + 1):
            try:
                items, has_more = fetch(site, query, page=page)
            except Exception as failure:                  # noqa: BLE001
                print(f"  {site}/{query!r} p{page}: FAILED "
                      f"{type(failure).__name__}: {str(failure)[:100]}",
                      file=sys.stderr)
                break
            for item in items:
                item["site"] = site
                for candidate in candidates_from(item):
                    if candidate["text"].lower() in seen:
                        continue
                    seen.add(candidate["text"].lower())
                    candidates.append(candidate)
            time.sleep(1.0)   # polite; the API is free and shared
            if not has_more:
                break
        print(f"  {site}: {query!r} -> {len(candidates)} running total")

    groups: dict = {}
    for candidate in candidates:
        for group in candidate["stresses"]:
            groups[group] = groups.get(group, 0) + 1

    document = {
        "schema": "quantify-stackexchange-candidates@1",
        "count": len(candidates), "by_group": groups,
        "licence": LICENCE,
        "attribution": (
            "Sentences are quoted verbatim from Stack Exchange posts under "
            "CC-BY-SA 4.0. Each carries the URL of the question it came from, "
            "which is the attribution the licence requires."),
        "collection_note": (
            "Harvested through the public Stack Exchange API. Bogleheads and "
            "reddit block automated fetchers; that is a stated preference, not "
            "an obstacle to route around, so neither was scraped."),
        "candidates": candidates}

    if dry_run:
        for candidate in candidates[:40]:
            print(f"  [{','.join(candidate['stresses'])}] {candidate['text']}")
        print(f"\n{len(candidates)} candidates (dry run, nothing written)")
        return 0

    OUT.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(candidates)} candidates -> {OUT}")
    for group, count in sorted(groups.items()):
        print(f"  {group:14} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--dry-run" in sys.argv))
