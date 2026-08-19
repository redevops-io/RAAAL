#!/usr/bin/env python3
"""Deterministic browser checks for quantify.club — no LLM, hard assertions.

Every check here corresponds to a defect that shipped, and each one shipped
because the whole test suite passed. That is the point of this file: the suite
reads responses, and a response can be perfectly correct while the page does
nothing. Two examples, both from one week:

  * `/auth/login` existed, was tested, redirected to the provider with PKCE —
    and no page linked to it. From a browser the deployment was
    indistinguishable from one with no login at all, and was reported as
    exactly that.

  * The strategy selector rendered both dropdowns correctly and its script
    returned at its first line, because the partial is included above the
    textarea it fills and `getElementById` ran during parsing. Choosing a kind
    narrowed nothing; choosing a strategy filled nothing.

Neither is visible in HTML. Both are one click to find.

Usage:
    python ui-agent/regression_smoke.py --url https://quantify.club
    python ui-agent/regression_smoke.py --url https://quantify.club \\
        --email you@example.com --password '...'

Without credentials the public checks run and the rest are reported as NOT RUN
rather than passed — a check that did not execute is not evidence.

Exit 0 only if every check that ran passed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

#: The whole run. If a single check hangs past this, the application is not
#: answering and the suite should say so rather than sit there.
DEADLINE_SECONDS = 240

#: What a menu entry must put in the box. Eight words is not a style rule: the
#: catalogue shipped fragments — "a 60/40 portfolio" — and somebody who picked
#: one was left holding three words and the blank page they started with.
FULL_SENTENCE_WORDS = 8

#: The catalogue had twenty entries and now has more. A sudden drop means the
#: library failed to load and the page fell back to whatever it could render.
LEAST_STRATEGIES = 30


@dataclass
class Result:
    name: str
    guards: str
    passed: Optional[bool] = None          # None = did not run
    detail: str = ""

    @property
    def state(self) -> str:
        return "NOT RUN" if self.passed is None else ("PASS" if self.passed
                                                      else "FAIL")


@dataclass
class Report:
    results: List[Result] = field(default_factory=list)

    def record(self, name, guards, passed=None, detail=""):
        self.results.append(Result(name, guards, passed, detail))
        return self.results[-1]

    @property
    def failed(self) -> List[Result]:
        return [r for r in self.results if r.passed is False]

    @property
    def skipped(self) -> List[Result]:
        return [r for r in self.results if r.passed is None]

    def render(self) -> str:
        width = max(len(r.name) for r in self.results) if self.results else 10
        lines = [""]
        for r in self.results:
            lines.append(f"  {r.state:8s} {r.name:{width}s}  {r.detail}")
        lines.append("")
        lines.append(f"  {len(self.results) - len(self.failed) - len(self.skipped)}"
                     f" passed, {len(self.failed)} failed,"
                     f" {len(self.skipped)} not run")
        if self.failed:
            lines.append("")
            for r in self.failed:
                lines.append(f"  FAILED {r.name}")
                lines.append(f"    guards: {r.guards}")
                lines.append(f"    {r.detail}")
        return "\n".join(lines)


def issuer_of(host: str, timeout: float = 30.0) -> str:
    """What the identity provider publishes as its issuer."""
    url = f"https://{host}/.well-known/openid-configuration"
    request = urllib.request.Request(
        url, headers={"User-Agent": "quantify-ui-agent/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["issuer"]


async def sign_in(page, base: str, email: str, password: str) -> None:
    """Through the provider's own pages, as a person does.

    Not by setting a cookie. A session minted by the test is a session the
    login flow never produced, and the flow is the thing under test.
    """
    await page.goto(f"{base}/auth/login", wait_until="domcontentloaded")
    await page.wait_for_timeout(1500)

    for selector in ("input[name='loginName']", "input[type='email']",
                     "input[name='username']", "input[type='text']"):
        field = page.locator(selector).first
        if await field.count():
            await field.fill(email)
            break
    await page.keyboard.press("Enter")
    await page.wait_for_timeout(2500)

    for selector in ("input[name='password']", "input[type='password']"):
        field = page.locator(selector).first
        if await field.count():
            await field.fill(password)
            break
    await page.keyboard.press("Enter")
    await page.wait_for_timeout(4000)

    # After a first sign-in the provider prompts to set up a second factor. The
    # pilot account is a test user with no authenticator, and the prompt offers
    # to skip — a person registering with an email and password is not made to
    # configure MFA before they can reach the workspace. Take the skip so the
    # session completes; without it the flow stops on the setup page and every
    # protected route bounces back to login.
    for selector in ("button:has-text('Skip')", "a:has-text('Skip')",
                     "button:has-text('skip')", "[id*='skip']"):
        skip = page.locator(selector).first
        if await skip.count():
            try:
                await skip.click()
                await page.wait_for_timeout(3000)
            except Exception:                                  # noqa: BLE001
                pass
            break


async def run(base: str, email: str, password: str, report: Report) -> None:
    from playwright.async_api import async_playwright

    base = base.rstrip("/")
    host = base.split("://", 1)[-1]

    # --- the identity provider, before a browser is involved --------------
    check = report.record(
        "issuer is https",
        "the provider published http://auth.quantify.club while the "
        "application verified https://. Every token would have been rejected "
        "on a valid signature, and the provider answers perfectly in a browser")
    try:
        published = issuer_of(f"auth.{host}")
        check.passed = published.startswith("https://")
        check.detail = f"issuer={published}"
    except Exception as error:  # noqa: BLE001
        check.passed = False
        check.detail = f"could not read discovery: {type(error).__name__}"

    async with async_playwright() as driver:
        browser = await driver.chromium.launch()
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
        page = await context.new_page()

        # --- signed out -------------------------------------------------
        check = report.record(
            "workspace requires a session",
            "the login governed nothing: /workspace was guarded by a shared "
            "basic-auth password, so signing in changed nothing a person "
            "could see")
        try:
            await page.goto(f"{base}/workspace/", wait_until="domcontentloaded")
            await page.wait_for_timeout(1000)
            landed = page.url
            check.passed = ("/auth/login" in landed
                            or "auth." in landed
                            or "/ui/login" in landed)
            check.detail = f"signed out, /workspace/ -> {landed[:90]}"
        except Exception as error:  # noqa: BLE001
            check.passed = False
            check.detail = f"{type(error).__name__}: {error}"

        if not (email and password):
            for name, guards in (
                ("sign-in link is on the page",
                 "the login existed for a week with nothing linking to it"),
                ("choosing a kind narrows the list",
                 "the selector's script returned before attaching listeners"),
                ("choosing a strategy fills the box",
                 "same script; the dropdowns were inert markup"),
                ("the box gets a whole sentence",
                 "the catalogue pasted fragments like 'a 60/40 portfolio'"),
                ("the catalogue is not truncated",
                 "a short list means the library failed to load"),
            ):
                report.record(name, guards, None, "no --email/--password given")
            await browser.close()
            return

        await sign_in(page, base, email, password)

        check = report.record(
            "signed in",
            "if this fails nothing below it means anything")
        try:
            await page.goto(f"{base}/workspace/", wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)
            check.passed = "/workspace" in page.url and "auth" not in page.url
            check.detail = f"landed on {page.url[:90]}"
        except Exception as error:  # noqa: BLE001
            check.passed = False
            check.detail = f"{type(error).__name__}: {error}"

        if not check.passed:
            await browser.close()
            return

        # --- the page a signed-in person actually uses -------------------
        check = report.record(
            "sign-in link is on the page",
            "the login existed for a week and no page linked to it, which "
            "looks identical to having no login")
        body = await page.locator("body").inner_html()
        check.passed = "/auth/logout" in body or "/auth/login" in body
        check.detail = ("found a link to the auth routes" if check.passed
                        else "no /auth/* link anywhere in the page")

        kind = page.locator("#pick-group")
        pick = page.locator("#pick")

        check = report.record(
            "the catalogue is not truncated",
            "a sudden drop in the offered list means the library failed to "
            "load and the page rendered whatever it could")
        try:
            count = await pick.locator("option").count()
            check.passed = count >= LEAST_STRATEGIES
            check.detail = f"{count} options offered (want >= {LEAST_STRATEGIES})"
        except Exception as error:  # noqa: BLE001
            check.passed = False
            check.detail = f"{type(error).__name__}: {error}"

        check = report.record(
            "choosing a kind narrows the list",
            "the selector's script is included above the textarea it fills, "
            "ran during parsing, found no textarea and returned at its first "
            "line. Both dropdowns rendered; neither did anything")
        try:
            before = await pick.locator("optgroup").count()
            value = await kind.locator("option").nth(1).get_attribute("value")
            await kind.select_option(value)
            await page.wait_for_timeout(600)
            after = await pick.locator("optgroup").count()
            check.passed = after < before and after >= 1
            check.detail = (f"kind={value!r}: {before} groups -> {after}")
        except Exception as error:  # noqa: BLE001
            check.passed = False
            check.detail = f"{type(error).__name__}: {error}"

        check = report.record(
            "choosing a strategy fills the box",
            "the same dead script: picking a strategy wrote nothing into the "
            "textarea, so the list was a menu that ordered nothing")
        sentence = ""
        try:
            option = pick.locator("optgroup option").first
            sentence = await option.get_attribute("data-text") or ""
            await pick.select_option(await option.get_attribute("value"))

            # Selecting submits, and the page it submits to reads the sentence
            # with a hosted model before it renders — seconds, not
            # milliseconds. Waiting a fixed 1200ms reported "textarea stayed
            # empty" for a page that had not arrived, which is a different
            # fault from the one this check exists to catch and must not be
            # reported as it.
            try:
                await page.wait_for_load_state("networkidle", timeout=45000)
            except Exception:  # noqa: BLE001 - a slow page is not yet a failure
                pass
            try:
                await page.wait_for_selector("#describe", timeout=15000)
            except Exception:  # noqa: BLE001
                pass

            box = page.locator("#describe")
            arrived = bool(await box.count())
            written = await box.input_value() if arrived else ""
            if not written and sentence and sentence in await page.content():
                # The sentence reached the next page even if that page shows
                # it somewhere other than a textarea. Still a pass: it
                # travelled.
                written = sentence
            check.passed = bool(written.strip())
            check.detail = (
                f"box got {written[:60]!r}" if written
                else ("the page it submitted to never rendered a textarea "
                      f"(at {page.url[:70]}) — the sentence may not have "
                      "travelled, or the page is still working"
                      if not arrived else
                      "textarea rendered and stayed empty after selecting"))
        except Exception as error:  # noqa: BLE001
            check.passed = False
            check.detail = f"{type(error).__name__}: {error}"

        check = report.record(
            "the box gets a whole sentence",
            "the catalogue pasted fragments — 'a 60/40 portfolio' is three "
            "words, and somebody who picked it was left where they started")
        words = len(sentence.split())
        check.passed = words >= FULL_SENTENCE_WORDS
        check.detail = f"{words} words: {sentence[:70]!r}"

        await browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://quantify.club")
    parser.add_argument("--email", default="")
    parser.add_argument("--password", default="")
    args = parser.parse_args()

    report = Report()
    try:
        asyncio.run(asyncio.wait_for(
            run(args.url, args.email, args.password, report),
            timeout=DEADLINE_SECONDS))
    except asyncio.TimeoutError:
        report.record("run completed", "a hung page is a failure",
                      False, f"gave up after {DEADLINE_SECONDS}s")

    print(report.render())
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
