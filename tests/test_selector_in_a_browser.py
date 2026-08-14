"""The strategy selector, executed rather than read.

Every other test in this suite reads a response body. The selector's two
defects were both invisible there:

  * the script is included above the textarea it fills, so it ran during
    parsing, found no `#describe`, and returned at its own guard. Both
    dropdowns rendered perfectly. Choosing a kind narrowed nothing and
    choosing a strategy filled nothing.

  * before that, the entries it pastes were fragments — "a 60/40 portfolio" —
    so even a working script left somebody holding three words.

The HTML was correct in both cases. What was wrong was what happened when a
browser ran it, and the only way to see that is to run one.

This renders the template and drives it in Chromium, with no server, no
network and no credentials, so it belongs in the ordinary suite rather than in
a deployment check. `ui-agent/regression_smoke.py` asks the same questions of
the deployed site, where a template that is right can still be included on a
page that is wrong.
"""
from __future__ import annotations

import pytest

pytest.importorskip("playwright",
                    reason="playwright drives the browser these checks need")

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "src" / "workspace" / "templates"


def rendered_page() -> str:
    """The picker inside a form with the textarea it expects.

    Deliberately in the order the real pages use — picker first, textarea
    after — because that order is what broke it. A fixture that put them the
    other way round would test a page nobody serves.
    """
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    from src.workspace.strategy_library import LIBRARY

    environment = Environment(loader=FileSystemLoader(str(TEMPLATES)),
                              undefined=StrictUndefined)
    environment.globals["strategy_library"] = lambda: LIBRARY
    picker = environment.get_template("_strategy_picker.html").render()
    return (
        "<!doctype html><meta charset=utf-8><body>"
        # Selecting a strategy submits the form — the behaviour we want, and
        # one that would navigate this fixture to a page no server is serving,
        # taking the textarea with it. `form.submit()` deliberately skips
        # onsubmit handlers, so the method itself is the only place to
        # intercept. In the document rather than an init script because an
        # init script only applies from the next navigation, which made this
        # depend on test order. Recorded rather than swallowed: submitting is
        # part of what selecting must do, and the tests assert it happened.
        "<script>HTMLFormElement.prototype.submit = function () {"
        "  this.setAttribute('data-submitted', '1'); };</script>"
        "<form action='/workspace/new' method='get' "
        "     onsubmit='return false'>"
        f"{picker}"
        "<textarea id='describe' name='describe'></textarea>"
        "<button type='submit'>Read it</button>"
        "</form></body>")


@pytest.fixture(scope="module")
def page():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as driver:
        try:
            browser = driver.chromium.launch()
        except Exception as error:  # noqa: BLE001 - no browser binary here
            pytest.skip(f"chromium is not available: {type(error).__name__}")
        context = browser.new_context()
        found = context.new_page()
        yield found
        browser.close()


@pytest.fixture(autouse=True)
def _fresh(page):
    page.set_content(rendered_page())
    page.wait_for_timeout(150)


class TestTheSelectorActuallyRuns:
    def test_choosing_a_kind_narrows_the_strategies(self, page):
        """The first symptom. The kind dropdown changed nothing at all."""
        before = page.locator("#pick optgroup").count()
        assert before > 1, "the fixture rendered one group; nothing to narrow"

        value = page.locator("#pick-group option").nth(1).get_attribute("value")
        page.select_option("#pick-group", value)
        page.wait_for_timeout(200)

        after = page.locator("#pick optgroup").count()
        assert after == 1, (
            f"choosing the {value!r} kind left {after} groups in the strategy "
            "list. The script that narrows it is included above the textarea "
            "it fills; without waiting for the document it returns at its "
            "first line and both dropdowns become inert markup")

    def test_the_narrowed_list_is_the_kind_that_was_chosen(self, page):
        value = page.locator("#pick-group option").nth(2).get_attribute("value")
        page.select_option("#pick-group", value)
        page.wait_for_timeout(200)
        remaining = page.locator("#pick optgroup").first
        assert remaining.get_attribute("data-group") == value

    def test_choosing_a_strategy_writes_its_sentence_into_the_box(self, page):
        """The second symptom, and the one that makes the list worth opening."""
        option = page.locator("#pick optgroup option").first
        sentence = option.get_attribute("data-text")
        page.select_option("#pick", option.get_attribute("value"))
        page.wait_for_timeout(300)

        assert page.locator("#describe").input_value() == sentence, (
            "selecting a strategy left the textarea empty, so the list is a "
            "menu that orders nothing")
        assert page.locator("form").get_attribute("data-submitted") == "1", (
            "the sentence was written and the form was not submitted, so the "
            "person is left looking at a filled box wondering what to press")

    def test_what_lands_in_the_box_is_a_whole_sentence(self, page):
        """Few people know what to type; that is the reason the list exists.

        A fragment in the box leaves them exactly where they started, which is
        what "a 60/40 portfolio" did.
        """
        option = page.locator("#pick optgroup option").first
        page.select_option("#pick", option.get_attribute("value"))
        page.wait_for_timeout(300)

        written = page.locator("#describe").input_value()
        assert len(written.split()) >= 8, (
            f"the box received {written!r}, which is a label rather than a "
            "statement somebody could run or edit")

    def test_a_hidden_strategy_cannot_be_reached_by_keyboard(self, page):
        """Narrowing detaches groups rather than hiding them.

        A hidden `<option>` stays selectable with the arrow keys in several
        browsers, and a filter somebody can arrow past is not a filter.
        """
        value = page.locator("#pick-group option").nth(1).get_attribute("value")
        page.select_option("#pick-group", value)
        page.wait_for_timeout(200)

        groups = page.locator("#pick optgroup").evaluate_all(
            "nodes => nodes.map(n => n.getAttribute('data-group'))")
        assert set(groups) == {value}, (
            f"the strategy list still contains {sorted(set(groups))} after "
            f"choosing {value!r}")


class TestItDoesNotDestroyWhatSomebodyWrote:
    def test_an_empty_box_is_filled_without_asking(self, page):
        option = page.locator("#pick optgroup option").first
        page.select_option("#pick", option.get_attribute("value"))
        page.wait_for_timeout(300)
        assert page.locator("#describe").input_value()

    def test_typed_text_is_not_replaced_without_a_confirmation(self, page):
        """Losing a sentence somebody wrote to a mis-click is worse than an
        extra prompt. The page asks; dismissing it must keep their words."""
        page.fill("#describe", "my own plan that I typed out myself")
        page.once("dialog", lambda dialog: dialog.dismiss())

        option = page.locator("#pick optgroup option").first
        page.select_option("#pick", option.get_attribute("value"))
        page.wait_for_timeout(300)

        assert page.locator("#describe").input_value() == \
            "my own plan that I typed out myself", (
                "the selector overwrote text somebody had written after the "
                "replacement was declined")
