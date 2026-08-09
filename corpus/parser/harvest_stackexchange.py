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
]

#: A candidate must look like somebody describing an action, not asking an
#: open question about the world.
ACTIONISH = re.compile(
    r"\b(I|we|my|our)\b.{0,60}\b(invest|contribut|rebalanc|hold|buy|sell|"
    r"allocat|put|add|keep|move|maintain|target)\w*", re.I)

SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def fetch(site: str, query: str, pagesize: int = 30) -> list:
    url = API + "?" + urllib.parse.urlencode({
        "order": "desc", "sort": "votes", "q": query, "site": site,
        "pagesize": pagesize, "filter": "withbody"})
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
    payload = json.loads(raw)
    if payload.get("quota_remaining", 1) < 20:
        print(f"  quota low: {payload.get('quota_remaining')} left", file=sys.stderr)
    return payload.get("items", [])


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
        try:
            items = fetch(site, query)
        except Exception as failure:                      # noqa: BLE001
            print(f"  {site}/{query!r}: FAILED {type(failure).__name__}: "
                  f"{str(failure)[:100]}", file=sys.stderr)
            continue
        for item in items:
            item["site"] = site
            for candidate in candidates_from(item):
                if candidate["text"].lower() in seen:
                    continue
                seen.add(candidate["text"].lower())
                candidates.append(candidate)
        print(f"  {site}: {query!r} -> {len(candidates)} running total")
        time.sleep(1.0)   # polite; the API is free and shared

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
