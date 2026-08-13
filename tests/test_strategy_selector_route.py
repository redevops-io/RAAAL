"""The selector over HTTP, because the catalogue tests do not touch a page.

`test_strategy_library.py` proves every offered sentence runs. That is the
safety property, and it says nothing about whether a person can reach one — a
catalogue registered as a template global and never rendered would pass all of
it. These drive real requests.

The origin assertions are the ones that matter for the cohort. If picking a
sentence and typing one are indistinguishable in the evidence, then a good
result over a catalogue we wrote, read by a reader we wrote, would be reported
as a result about people's own words.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from src.workspace.strategy_library import EDITED, PICKED, TYPED, offered


@pytest.fixture
def pilot_client(monkeypatch, tmp_path):
    from src.deploy import context as deploy_context

    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used-by-recordings")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/pilot.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    return TestClient(app)


class TestThePageOffersIt:
    def test_the_kind_selector_comes_first(self, pilot_client):
        """Category, then strategy. Twenty entries under five headings is a
        wall somebody scans rather than a choice they make; the heading was the
        useful distinction and the part you could not act on."""
        page = pilot_client.get("/workspace/").text
        assert 'id="pick-group"' in page, "no kind selector"
        assert page.index('id="pick-group"') < page.index('id="pick"'), (
            "the strategy list comes before the kind that narrows it")
        for group in __import__("src.workspace.strategy_library",
                                fromlist=["LIBRARY"]).LIBRARY:
            assert f'value="{group.key}"' in page, group.key

    def test_every_option_declares_its_kind(self, pilot_client):
        """The filter is driven by the markup, not by a second list in script.
        An option with no kind is one the narrowing cannot place, and it would
        vanish from every category including its own."""
        import re

        page = pilot_client.get("/workspace/").text
        options = re.findall(r'<option value="([a-z-]+)"[^>]*>', page)
        for key in [e.key for e in offered()]:
            assert key in options, key
            block = page[page.index(f'value="{key}"'):][:220]
            assert "data-group=" in block, f"{key} declares no kind"

    def test_the_selector_is_on_the_empty_page(self, pilot_client):
        """The empty page is where somebody with no idea what to type lands.
        A selector that only appeared after a first attempt would arrive one
        step after it was needed."""
        page = pilot_client.get("/workspace/new").text
        assert 'id="pick"' in page
        for group in __import__(
                "src.workspace.strategy_library",
                fromlist=["LIBRARY"]).LIBRARY:
            assert group.title in page

    def test_every_offered_sentence_is_reachable_from_the_page(self,
                                                                pilot_client):
        """Rendered, not merely registered. A group added to `LIBRARY` and left
        out of the template would still pass every catalogue test."""
        page = pilot_client.get("/workspace/new").text
        for case in offered():
            assert case.key in page, f"{case.key!r} is offered and unreachable"

    def test_it_is_on_the_page_people_actually_arrive_at(self, pilot_client):
        """`/workspace/`, not `/workspace/new`.

        The selector was written into `pilot.html` only, which is the page you
        reach *after* submitting something. The landing page has its own prompt
        box, so somebody arriving at the site saw a bare textarea and no list —
        which is exactly the person the list exists for. Reported by a user
        hard-refreshing and finding nothing.
        """
        page = pilot_client.get("/workspace/").text
        assert 'id="pick"' in page, (
            "the landing page offers no strategy selector, so the first thing "
            "a new visitor sees is a blank box")
        for case in offered():
            assert case.key in page

    def test_both_pages_render_one_definition(self, pilot_client):
        """A second copy of the dropdown drifts, and the copy that drifts is
        the one nobody is looking at. Both pages include the same partial, so
        an entry added to the library reaches both or neither."""
        landing = pilot_client.get("/workspace/").text
        draft = pilot_client.get("/workspace/new").text
        for case in offered():
            assert (case.key in landing) == (case.key in draft), case.key

    def test_it_survives_onto_the_result_page(self, pilot_client):
        """Rendered from a template global rather than threaded through six
        contexts, so a page that reads a sentence still offers the list. The
        result page is where somebody decides to try a different strategy."""
        page = pilot_client.get(
            "/workspace/new", params={"describe": "invest $500 a month into VTI"}).text
        assert 'id="pick"' in page


class TestTheOriginIsRecorded:
    """The edited cases below send one entry's text under another entry's key.

    An invented sentence would be more natural to read and would fail here for
    the wrong reason: the recorded reader raises on text it has never seen, and
    calling the provider from a test would hide exactly the gap that raise
    exists to expose. What is under test is the comparison between the
    submitted text and the named entry, and text that matches no entry
    exercises it whether or not a human would have typed it.

    `test_strategy_library.py` covers realistic edits — trailing clauses,
    whitespace — at the unit level, where no reader is involved.
    """

    def test_a_typed_sentence_is_recorded_as_typed(self, pilot_client,
                                                    monkeypatch):
        seen = []
        import src.workspace.pilot_routes as routes

        monkeypatch.setattr(
            routes, "record_transcript",
            lambda *a, **k: seen.append(k.get("origin")))
        pilot_client.get("/workspace/new",
                         params={"describe": "invest $500 a month into VTI"})
        assert seen == [TYPED]

    def test_an_untouched_pick_is_recorded_as_picked(self, pilot_client,
                                                      monkeypatch):
        seen = []
        import src.workspace.pilot_routes as routes

        monkeypatch.setattr(
            routes, "record_transcript",
            lambda *a, **k: seen.append(k.get("origin")))
        case = offered()[0]
        pilot_client.get("/workspace/new",
                         params={"describe": case.text, "picked": case.key})
        assert seen == [PICKED]

    def test_an_edited_pick_is_recorded_as_edited(self, pilot_client,
                                                   monkeypatch):
        """The interesting middle. `EDITED` says the catalogue got somebody
        close and something they wanted was missing, which is the signal for
        what to add to it."""
        seen = []
        import src.workspace.pilot_routes as routes

        monkeypatch.setattr(
            routes, "record_transcript",
            lambda *a, **k: seen.append(k.get("origin")))
        picked, other = offered()[0], offered()[2]
        pilot_client.get("/workspace/new",
                         params={"describe": other.text, "picked": picked.key})
        assert seen == [EDITED]

    def test_the_form_cannot_claim_an_origin_it_does_not_have(self,
                                                               pilot_client,
                                                               monkeypatch):
        """The client sends `picked`. It must not be able to launder a typed
        sentence into the picked bucket by naming an entry it did not use."""
        seen = []
        import src.workspace.pilot_routes as routes

        monkeypatch.setattr(
            routes, "record_transcript",
            lambda *a, **k: seen.append(k.get("origin")))
        pilot_client.get("/workspace/new",
                         params={"describe": offered()[3].text,
                                 "picked": offered()[0].key})
        assert seen == [EDITED]
