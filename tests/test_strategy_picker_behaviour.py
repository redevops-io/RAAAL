"""The selector's script must survive being included above the field it drives.

Two dropdowns rendered correctly, and neither did anything. The picker is
included above the textarea it fills, so when its inline script ran at parse
time `document.getElementById("describe")` was null, the guard at the top
returned, and no listener was ever attached. Choosing a kind narrowed nothing.
Choosing a strategy filled nothing.

Every server-side test passed, because the HTML was right. What was wrong was
*when* the script ran, which no amount of reading the response body reveals.

These checks are structural — the behaviour itself needs a browser, and that is
what the UI agent is for. What they can establish is that the script does not
assume the document is finished, and that the elements it reaches for are the
ones the including pages actually provide.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "src" / "workspace" / "templates"
PICKER = TEMPLATES / "_strategy_picker.html"


@pytest.mark.skipif(not PICKER.exists(), reason="no picker here")
class TestTheScriptWaitsForTheDocument:
    def test_it_defers_until_the_document_is_parsed(self):
        """The fix, named. Without this the script runs before the textarea
        exists and returns at its own guard."""
        source = PICKER.read_text()
        assert "DOMContentLoaded" in source and "readyState" in source, (
            "the picker's script runs inline and is included above the "
            "textarea it fills. Without a readiness guard it executes before "
            "that element exists, returns, and leaves two dropdowns that look "
            "like a selector and do nothing")

    def test_it_still_gives_up_quietly_when_an_element_is_genuinely_absent(self):
        """The guard stays. A page that includes the picker without a textarea
        should do nothing rather than raise in somebody's browser."""
        source = PICKER.read_text()
        assert re.search(r"if\s*\(!\s*kind\s*\|\|\s*!\s*pick\s*\|\|\s*!\s*box\s*\)",
                         source), "the element guard has gone"


@pytest.mark.skipif(not PICKER.exists(), reason="no picker here")
class TestThePagesProvideWhatTheScriptReachesFor:
    """The ids are a contract between a partial and its including pages.

    The partial names three: `pick-group`, `pick` and `describe`. It renders
    the first two itself; the third belongs to the page. A page that includes
    the picker and calls its textarea something else gets a dead selector, and
    the server's HTML looks perfect.
    """

    def including_pages(self):
        found = [path for path in TEMPLATES.glob("*.html")
                 if "_strategy_picker.html" in path.read_text()]
        assert found, "no page includes the picker; this check is stale"
        return found

    def test_every_including_page_has_the_textarea_the_script_fills(self):
        for path in self.including_pages():
            source = path.read_text()
            assert 'id="describe"' in source, (
                f"{path.name} includes the strategy picker and has no "
                'textarea with id="describe". The script fills that element '
                "by id; without it the selector silently does nothing")

    def test_every_including_page_wraps_it_in_a_form(self):
        """Choosing a strategy submits. `pick.form` is null outside a form,
        and the choice would fill the box and stop there."""
        for path in self.including_pages():
            assert "<form" in path.read_text(), (
                f"{path.name} includes the picker outside a form, so "
                "selecting a strategy cannot submit it")


@pytest.mark.skipif(not PICKER.exists(), reason="no picker here")
class TestTheListIsWorthOpening:
    """What the person actually asked for: pick a kind, get that kind's
    strategies, and have the sentence written for you."""

    def rendered(self):
        from src.workspace.strategy_library import LIBRARY

        return LIBRARY

    def test_every_group_has_strategies_under_it(self):
        for group in self.rendered():
            assert group.entries, (
                f"the {group.title!r} kind has no strategies, so choosing it "
                "narrows the list to nothing")

    def test_every_strategy_carries_a_description_not_its_own_label(self):
        """The point of the list is that few people know what to type.

        A title in the box would leave them exactly where they started, so the
        text has to be something the engine reads and a person can edit —
        distinct from the label that led them to it.

        Length is deliberately not asserted. Several entries are terser than
        they should be ("a 60/40 portfolio" is three words, and the ask was for
        a properly formulated description); making them fuller means
        re-recording each one against the hosted reader, which is a deliberate
        step rather than a wording change. Asserting a word count here would
        either fail on shipped data or pass on a threshold chosen to fit it,
        and neither is evidence.
        """
        for group in self.rendered():
            for entry in group.entries:
                assert entry.text and entry.text.strip(), (
                    f"{entry.key} has no description, so choosing it puts "
                    "nothing in the box")
                assert entry.text != entry.title, (
                    f"{entry.key} has no description distinct from its title")

    def test_the_script_puts_that_sentence_in_the_box(self):
        source = PICKER.read_text()
        assert 'data-text' in source and 'box.value = text' in source, (
            "selecting a strategy must write its sentence into the textarea; "
            "otherwise the list is a menu that orders nothing")
