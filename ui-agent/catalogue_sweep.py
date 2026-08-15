#!/usr/bin/env python3
"""Walk every strategy in the menu, in a browser, and vary the details.

`tests/test_catalogue_sweep.py` asks the same questions of every catalogue
entry without a browser, in eight seconds, on recorded readings. It found five
strategies whose pages neither ran, asked, nor refused — reachable straight
from the menu. Run that one first; it is free and it catches most of this.

What it cannot see is the page. A reading can be coherent while the form that
renders it is wired wrong: a button gated on the wrong list, an input whose
name the handler does not read, a selection that fills the box and never
submits. Those need a browser and a deployment, which is what this is.

It also does what the fast sweep cannot: **changes the sentence**. Every
catalogue entry is submitted as written and then again with the amount, the
holding and the period substituted, because a strategy that works with VTI at
$500 a month and breaks with NVDA at 200usd is exactly the surprise a user
finds first. Those variants are not recorded and are read live, so this costs
provider calls and belongs on demand rather than in CI.

    python ui-agent/catalogue_sweep.py --url https://quantify.club \\
        --email you@example.com --password '...' [--variants 2] [--limit 5]

Exit 0 only if every page was coherent.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from regression_smoke import Report, sign_in                # noqa: E402

#: Substitutions applied to each catalogue sentence. Chosen to move one thing
#: at a time: a different amount with a unit glued on, a different holding, a
#: different period. `200usd` is here because a real submission of exactly that
#: produced a page offering to fill in a blank it did not show.
VARIANTS = (
    ("amount", (("$500", "200usd"), ("$200", "1500 dollars"),
                ("$1,000", "2k"), ("$50", "75usd"))),
    ("holding", (("VTI", "NVDA"), ("VOO", "QQQ"), ("BND", "TLT"),
                 ("SPY", "^GSPC"))),
    ("period", (("every month", "every two weeks"),
                ("each year", "every quarter"),
                ("once a year", "every 6 months"))),
)


@dataclass
class Case:
    strategy: str
    sentence: str
    variant: str = "as written"
    problems: List[str] = field(default_factory=list)


def variants_of(sentence: str, how_many: int) -> List[tuple]:
    """The sentence, and up to `how_many` altered versions of it."""
    out = [("as written", sentence)]
    for kind, pairs in VARIANTS:
        for original, replacement in pairs:
            if original in sentence and len(out) <= how_many:
                out.append((f"{kind}: {original} -> {replacement}",
                            sentence.replace(original, replacement, 1)))
                break
    return out


async def submit(page, base: str, sentence: str) -> dict:
    """Submit a sentence and read back what the page says about it."""
    import urllib.parse

    await page.goto(
        f"{base}/workspace/new?describe={urllib.parse.quote(sentence)}",
        wait_until="domcontentloaded")
    try:
        await page.wait_for_load_state("networkidle", timeout=60000)
    except Exception:                                        # noqa: BLE001
        pass

    rows = await page.locator("table tr.p-needed, table tr.p-refused, "
                              "table tr.p-settled, table tr.p-chosen").count()
    needed = await page.locator("tr.p-needed").count()
    inputs = await page.locator("input[name^='answer_']").count()
    refused = await page.locator("tr.p-refused").count()
    body = await page.locator("body").inner_text()
    return {
        "rows": rows, "needed": needed, "inputs": inputs, "refused": refused,
        "has_button": "run it" in body.lower(),
        "has_result": "Result" in body,
        "has_ledger": "What actually happened" in body,
        "url": page.url,
    }


def problems_with(seen: dict) -> List[str]:
    """The inconsistencies a person would hit, named."""
    found = []

    if seen["has_button"] and not seen["inputs"]:
        found.append(
            "the page offers to fill something in and shows no input — "
            "pressing it returns to the same page")

    if seen["needed"] and not seen["inputs"]:
        found.append("a parameter is marked as needed and has no input")

    if not seen["rows"]:
        found.append("no parameter table at all")

    if not (seen["has_result"] or seen["needed"] or seen["refused"]):
        found.append(
            "the plan neither ran, asked, nor refused — the one outcome "
            "nobody can act on and nobody can report")

    if seen["has_result"] and not seen["has_ledger"]:
        found.append("a figure with no ledger under it")

    return found


async def run(base: str, email: str, password: str, variants: int,
              limit: int, report: Report) -> None:
    from playwright.async_api import async_playwright

    from strategies import catalogue                        # noqa: E402

    base = base.rstrip("/")
    async with async_playwright() as driver:
        browser = await driver.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()
        await sign_in(page, base, email, password)

        entries = catalogue()
        if limit:
            entries = entries[:limit]

        for key, sentence in entries:
            for label, text in variants_of(sentence, variants):
                check = report.record(f"{key} [{label}]",
                                      "a coherent page for an offered strategy")
                try:
                    problems = problems_with(await submit(page, base, text))
                    check.passed = not problems
                    check.detail = ("; ".join(problems) if problems
                                    else f"{text[:58]}...")
                except Exception as error:                   # noqa: BLE001
                    check.passed = False
                    check.detail = f"{type(error).__name__}: {error}"

        await browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://quantify.club")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--variants", type=int, default=1,
                        help="altered sentences per strategy, beyond the "
                             "original. Each is a live provider call.")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many strategies")
    args = parser.parse_args()

    report = Report()
    asyncio.run(run(args.url, args.email, args.password,
                    args.variants, args.limit, report))
    print(report.render())
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
