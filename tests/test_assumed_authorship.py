"""Assumed values are visible, and stay assumed until somebody changes them.

The page used to render an assumed value as text beside an **empty** input.
The reasoning was sound: a pre-filled box gets posted back by anyone who
presses Run without reading the row, and a posted value returned authored
`USER` — which dominates every other author and is never overwritten by a
re-read. The product would have promoted its own guess to the person's stated
word, permanently, on a click that meant "run it".

The mechanism was a proxy. It inferred authorship from *emptiness*, which is
correlated with "the person did not state this" and is not the same fact. The
cost was a form with a dozen blank boxes and the assumed value written beside
each one, asking somebody to retype what was already on screen.

So the form states authorship instead — `original_*` for what the row was
offered as, `author_*` for whose it is — and the handler compares them. These
tests are the ones that matter, because the safety property must survive the
change:

    assumed values are visibly pre-filled
    clicking Run unchanged preserves ASSUMED
    editing an assumed value creates USER
    USER values remain USER
    defaults cannot become USER without an actual edit

Every assertion reads the settled record, not the rendered HTML. What the page
shows is a rendering of authorship; the record is authorship.
"""
from __future__ import annotations

import os
import re

import pytest
from fastapi.testclient import TestClient

#: A catalogue strategy, so the page carries catalogue assumptions. Free text
#: alone settles nothing as `CATALOG_ASSUMED`, and a test for assumed-value
#: handling that produced no assumed values would pass by having nothing to
#: check — which is why every test below asserts it found some before
#: asserting anything about them.
#:
#: `employer-match` because it is the page the complaint came from: seventeen
#: rendered inputs, three of them assumed.
PICKED = "employer-match"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/a.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    return TestClient(app, follow_redirects=False)


def _rows(html):
    """Every rendered input, with what it was offered as and whose it is."""
    found = {}
    for name, value in re.findall(r'name="answer_([a-z_]+)"[^>]*value="([^"]*)"',
                                  html):
        original = re.search(
            rf'name="original_{name}"[^>]*value="([^"]*)"', html)
        author = re.search(rf'name="author_{name}"[^>]*value="([^"]*)"', html)
        found[name] = {
            "value": value,
            "original": original.group(1) if original else None,
            "author": author.group(1) if author else None,
        }
    return found


def _authorship(stored):
    """The last word on each dimension, from the record rather than the page.

    `settle` appends, so the final entry is the current authorship and the
    earlier ones are the history that says it began as a guess.
    """
    latest = {}
    for entry in stored.get("settled", ()):
        latest[entry["field"]] = entry
    return latest


def _open_page(client):
    """The page as the catalogue link opens it: a selection and its text.

    Both, because the parameter table is a rendering of a reading and there is
    no reading without something to read. `?picked=` alone returns the form
    with nothing in it.
    """
    from urllib.parse import quote

    from src.workspace.strategy_library import entry

    chosen = entry(PICKED)
    assert chosen is not None, f"{PICKED} is not in the catalogue"
    got = client.get(
        f"/workspace/new?picked={PICKED}&describe={quote(chosen.text)}")
    assert got.status_code == 200, got.status_code
    return got.text


def _submit(client, html, **overrides):
    """Post the form exactly as the browser would, with optional edits."""
    body = {"describe": "", "picked": PICKED}
    describe = re.search(r'name="describe" value="([^"]*)"', html)
    if describe:
        body["describe"] = describe.group(1)
    for name, row in _rows(html).items():
        body[f"answer_{name}"] = row["value"]
        if row["original"] is not None:
            body[f"original_{name}"] = row["original"]
        if row["author"] is not None:
            body[f"author_{name}"] = row["author"]
    body.update(overrides)
    return client.post("/pilot/answer", data=body)


def _stored(client, posted):
    from src.workspace.pilot_store import load_review

    assert posted.status_code == 303, posted.status_code
    return load_review(posted.headers["location"].rsplit("/", 1)[-1])


def test_assumed_values_are_rendered_pre_filled(client):
    """The complaint that started this: the boxes were empty."""
    rows = _rows(_open_page(client))
    assumed = {n: r for n, r in rows.items() if r["author"] == "ASSUMED"}
    assert assumed, "no assumed rows rendered; this test would prove nothing"
    blank = [n for n, r in assumed.items() if not r["value"]]
    assert not blank, (
        f"assumed rows still render empty inputs: {blank}. The value is on "
        "the page; making somebody retype it is the defect.")


def test_assumed_rows_state_their_author(client):
    """Authorship is on the form, not inferred from the box being non-empty."""
    rows = _rows(_open_page(client))
    assumed = [n for n, r in rows.items() if r["author"] == "ASSUMED"]
    assert assumed, "no row declared itself assumed"
    for name in assumed:
        assert rows[name]["original"] == rows[name]["value"], (
            f"{name} was offered a value its `original_` field disagrees "
            "with, so an unchanged submission would look like an edit")


def test_running_without_touching_anything_keeps_it_assumed(client):
    """The safety property, and the whole reason the boxes were blank.

    A click-through must not promote the product's guess to the person's word.
    This is the test that has to hold after pre-filling.
    """
    html = _open_page(client)
    before = {n for n, r in _rows(html).items() if r["author"] == "ASSUMED"}
    assert before, "no assumed rows; this test would prove nothing"

    stored = _stored(client, _submit(client, html))
    latest = _authorship(stored)

    promoted = [n for n in before
                if latest.get(n, {}).get("provenance") == "USER_ANSWERED"]
    assert not promoted, (
        f"{promoted} became USER_ANSWERED on a submission that changed "
        "nothing. Pressing Run is not a statement about a value.")


#: A dimension `employer-match` assumes, and a value that is a real
#: alternative to the assumption rather than a corrupted version of it. An
#: earlier draft appended " changed" to the assumed value, which produced
#: `"$500 changed"` — correctly refused as not an amount, and the test then
#: reported that editing does not attribute to the person when what it had
#: actually shown is that nonsense is rejected.
EDITABLE = "amount"
EDITED_TO = "$750"


def test_editing_an_assumed_value_makes_it_the_persons(client):
    """The other half: a real edit must be attributed to them."""
    html = _open_page(client)
    assumed = {n: r for n, r in _rows(html).items() if r["author"] == "ASSUMED"}
    assert EDITABLE in assumed, (
        f"{EDITABLE} is not assumed on this page; the fixture no longer tests "
        f"what it says. assumed={sorted(assumed)}")
    name = EDITABLE

    edited = _submit(client, html, **{f"answer_{name}": EDITED_TO})
    latest = _authorship(_stored(client, edited))
    assert latest.get(name, {}).get("provenance") == "USER_ANSWERED", (
        f"{name} was edited and is not recorded as the person's: "
        f"{latest.get(name)}")


def test_a_persons_value_survives_a_later_unchanged_submission(client):
    """Theirs stays theirs.

    The reading is rebuilt from the sentence on every submission, so a value
    the person supplied earlier has to be carried back or it silently reverts
    to whatever the reader says. `author_=USER` is what carries it.
    """
    html = _open_page(client)
    assumed = {n: r for n, r in _rows(html).items() if r["author"] == "ASSUMED"}
    assert EDITABLE in assumed, f"{EDITABLE} is not assumed on this page"
    name = EDITABLE

    first = _submit(client, html, **{f"answer_{name}": EDITED_TO})
    review = client.get(first.headers["location"]).text

    again = _submit(client, review)
    latest = _authorship(_stored(client, again))
    assert latest.get(name, {}).get("provenance") == "USER_ANSWERED", (
        f"{name} stopped being the person's after a submission that changed "
        f"nothing: {latest.get(name)}")


def test_emptiness_is_no_longer_what_decides(client):
    """The proxy is gone, asserted directly on the helper.

    A value identical to what was offered is not an answer however non-empty
    it is; a value that differs is, however ordinary it looks. This is the
    substitution the whole change is about, so it is checked at the function
    that makes the decision rather than only through the page.
    """
    from src.workspace.pilot_routes import _answers_in

    unchanged = _answers_in({"answer_cadence": "monthly",
                             "original_cadence": "monthly",
                             "author_cadence": "ASSUMED"})
    assert unchanged == {}, f"an untouched assumed row counted as an answer: {unchanged}"

    edited = _answers_in({"answer_cadence": "weekly",
                          "original_cadence": "monthly",
                          "author_cadence": "ASSUMED"})
    assert edited == {"cadence": "weekly"}

    theirs = _answers_in({"answer_cadence": "monthly",
                          "original_cadence": "monthly",
                          "author_cadence": "USER"})
    assert theirs == {"cadence": "monthly"}, (
        "a value already the person's was dropped because it had not changed "
        "since it was rendered, which would revert it to the reader's")

    # No `original_` at all: hand-made requests and older forms. Attributing
    # to the person over-credits them and never the reverse.
    bare = _answers_in({"answer_cadence": "weekly"})
    assert bare == {"cadence": "weekly"}
