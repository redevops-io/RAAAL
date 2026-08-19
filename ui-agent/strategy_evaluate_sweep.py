#!/usr/bin/env python3
"""Select every strategy, take the offered defaults, run it, and say what happened.

This is the agent doing what a person would to get a first result fast: pick a
strategy from the menu, accept the suggested value for each blank the plan needs
— by Tab, the same one-keystroke fill the page offers — and press run. It then
reads the outcome and reports it, strategy by strategy:

  * evaluated  — the plan ran and produced a figure (captured, for the graph
                 comparison that is the next step)
  * refused    — the build declines this strategy by name, with a reason. Not a
                 defect: the catalogue offers more than this build executes, and
                 a named refusal is the honest answer. Reported so the map of
                 what runs and what does not is visible.
  * blocked    — a required blank has no default to accept, so the agent cannot
                 fill it by Tab. This is the gap that matters after defaults
                 landed: a strategy the one-keystroke path cannot complete.
  * unresolved — everything was filled and the page still neither ran nor
                 refused. The outcome nobody can act on; a bug.
  * error      — the page threw, or a control the flow depends on was missing.

`blocked`, `unresolved` and `error` set a non-zero exit; `evaluated` and
`refused` are both clean outcomes. Every strategy is listed either way.

    python ui-agent/strategy_evaluate_sweep.py --url https://quantify.club \\
        --email pilot@quantify.club --password '...' [--limit 5] [--url-nav]

Each strategy is a live read, so this costs provider calls and runs on demand,
not in CI. `--limit` keeps a first run cheap. `--url-nav` drives the selection
by URL instead of the dropdown control (faster, skips the selector's own script,
which `tests/test_selector_in_a_browser.py` already covers).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Optional

from regression_smoke import sign_in                          # noqa: E402

EVALUATED, REFUSED, BLOCKED, UNRESOLVED, ERROR = (
    "evaluated", "refused", "blocked", "unresolved", "error")
CLEAN = {EVALUATED, REFUSED}


@dataclass
class Outcome:
    key: str
    status: str
    figure: str = ""
    reason: str = ""
    filled: List[str] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)


async def _land_on_strategy(page, base: str, key: str, sentence: str,
                            url_nav: bool) -> None:
    """Get to the plan page for one strategy — by the dropdown, or by URL.

    The dropdown is the path a person takes: choosing an option writes its
    sentence into the box and submits. `--url-nav` skips it for speed; the
    selector's own behaviour is pinned by tests/test_selector_in_a_browser.py,
    so covering it on every strategy here buys little."""
    if url_nav:
        await page.goto(
            f"{base}/workspace/new?picked={urllib.parse.quote(key)}"
            f"&describe={urllib.parse.quote(sentence)}",
            wait_until="domcontentloaded")
    else:
        await page.goto(f"{base}/workspace/new", wait_until="domcontentloaded")
        await page.wait_for_selector("#pick", timeout=20000)
        # Selecting fires the change handler, which fills #describe and submits.
        async with page.expect_navigation(wait_until="domcontentloaded",
                                           timeout=60000):
            await page.select_option("#pick", key)
    try:
        await page.wait_for_load_state("networkidle", timeout=90000)
    except Exception:                                          # noqa: BLE001
        pass


async def _tab_fill(page) -> tuple:
    """Accept the suggested default for every blank the plan needs, by Tab.

    Focus the field and press Tab: on an empty required field the page fills it
    with its top suggestion in place. A field with no `data-suggest` has no
    default to accept — Tab would just move on — so it is recorded as blocked,
    which is exactly the gap this sweep exists to surface now that defaults
    landed. A field already carrying a value (a sentence read but not usable) is
    left as read."""
    needed = page.locator("input.pfield.needs")
    count = await needed.count()
    filled, blocked = [], []
    for i in range(count):
        field_input = needed.nth(i)
        name = (await field_input.get_attribute("name") or "").removeprefix("answer_")
        try:
            await field_input.scroll_into_view_if_needed(timeout=5000)
            await field_input.focus()
        except Exception:                                      # noqa: BLE001
            blocked.append(f"{name} (unreachable)")
            continue
        if (await field_input.input_value()).strip():
            filled.append(f"{name} (as read)")
            continue
        suggest = await field_input.get_attribute("data-suggest")
        if not suggest:
            blocked.append(f"{name} (no default)")
            continue
        await field_input.press("Tab")
        if (await field_input.input_value()).strip():
            filled.append(name)
        else:
            blocked.append(f"{name} (tab did not fill)")
    return filled, blocked


async def _read_outcome(page) -> tuple:
    """Whether the page ran, refused, or did neither — and the detail."""
    figure = page.locator("section.result p.figure")
    if await figure.count():
        text = (await figure.first.inner_text()).strip()
        gain = page.locator("section.result p.quiet")
        if await gain.count():
            first = (await gain.first.inner_text()).strip()
            if first.lower().startswith("gain"):
                text = f"{text}  ({first})"
        return EVALUATED, text
    refusal = page.locator("section.refusal")
    if await refusal.count():
        # The refusal names the dimension and the reason; keep it to a line.
        raw = " ".join((await refusal.first.inner_text()).split())
        return REFUSED, raw[:200]
    return "", ""


async def evaluate_one(page, base: str, key: str, sentence: str,
                       url_nav: bool) -> Outcome:
    try:
        await _land_on_strategy(page, base, key, sentence, url_nav)
    except Exception as error:                                 # noqa: BLE001
        return Outcome(key, ERROR, reason=f"reaching the page: "
                       f"{type(error).__name__}: {error}")

    # A fully specified sentence runs on arrival; report it without touching it.
    status, detail = await _read_outcome(page)
    if status == EVALUATED:
        return Outcome(key, EVALUATED, figure=detail)

    filled, blocked = await _tab_fill(page)

    run = page.locator("button.prun")
    if await run.count():
        try:
            async with page.expect_navigation(wait_until="domcontentloaded",
                                               timeout=90000):
                await run.first.click()
            await page.wait_for_load_state("networkidle", timeout=90000)
        except Exception as error:                             # noqa: BLE001
            return Outcome(key, ERROR, filled=filled, blocked=blocked,
                           reason=f"running: {type(error).__name__}: {error}")

    status, detail = await _read_outcome(page)
    if status == EVALUATED:
        return Outcome(key, EVALUATED, figure=detail, filled=filled,
                       blocked=blocked)
    if status == REFUSED:
        return Outcome(key, REFUSED, reason=detail, filled=filled,
                       blocked=blocked)
    if blocked:
        return Outcome(key, BLOCKED, filled=filled, blocked=blocked,
                       reason="no default to Tab into: " + ", ".join(blocked))
    still_needed = await page.locator(".prow.p-needed").count()
    return Outcome(key, UNRESOLVED, filled=filled, blocked=blocked,
                   reason=(f"{still_needed} field(s) still needed after filling"
                           if still_needed else
                           "the page neither ran nor refused"))


def render(results: List[Outcome]) -> str:
    by = {s: [r for r in results if r.status == s]
          for s in (EVALUATED, REFUSED, BLOCKED, UNRESOLVED, ERROR)}
    lines = ["", "=" * 72,
             f"strategy evaluate sweep — {len(results)} strategies",
             "=" * 72]
    mark = {EVALUATED: "OK ", REFUSED: " · ", BLOCKED: "!! ",
            UNRESOLVED: "XX ", ERROR: "XX "}
    for r in results:
        head = f"{mark[r.status]}{r.key:<34} {r.status}"
        tail = r.figure if r.status == EVALUATED else r.reason
        lines.append(head + (f"  {tail}" if tail else ""))
        if r.blocked and r.status != BLOCKED:
            lines.append(f"      (also blocked: {', '.join(r.blocked)})")
    lines.append("-" * 72)
    lines.append(
        f"evaluated {len(by[EVALUATED])}  ·  refused {len(by[REFUSED])}  ·  "
        f"blocked {len(by[BLOCKED])}  ·  unresolved {len(by[UNRESOLVED])}  ·  "
        f"error {len(by[ERROR])}")
    attention = len(by[BLOCKED]) + len(by[UNRESOLVED]) + len(by[ERROR])
    lines.append("all strategies either ran or refused by name"
                 if not attention else
                 f"{attention} strategy(ies) need attention (blocked/unresolved/error)")
    return "\n".join(lines)


async def run(base: str, email: str, password: str, limit: int,
              url_nav: bool) -> List[Outcome]:
    from playwright.async_api import async_playwright

    from strategies import catalogue                           # noqa: E402

    base = base.rstrip("/")
    entries = catalogue()
    if limit:
        entries = entries[:limit]

    results: List[Outcome] = []
    async with async_playwright() as driver:
        browser = await driver.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()
        await sign_in(page, base, email, password)
        for index, (key, sentence) in enumerate(entries, 1):
            outcome = await evaluate_one(page, base, key, sentence, url_nav)
            results.append(outcome)
            tail = outcome.figure or outcome.reason
            print(f"[{index}/{len(entries)}] {key}: {outcome.status}"
                  f"{'  ' + tail if tail else ''}", file=sys.stderr)
        await browser.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://quantify.club")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many strategies (0 = all)")
    parser.add_argument("--url-nav", action="store_true",
                        help="select by URL rather than the dropdown control")
    args = parser.parse_args()

    results = asyncio.run(run(args.url, args.email, args.password,
                              args.limit, args.url_nav))
    print(render(results))
    attention = sum(1 for r in results
                    if r.status not in CLEAN)
    return 1 if attention else 0


if __name__ == "__main__":
    sys.exit(main())
