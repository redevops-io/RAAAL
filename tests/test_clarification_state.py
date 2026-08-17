"""Clarification state is persisted whole, and a lap is named as a lap.

Two properties, from the plan.

**Transactional.** A successful answer must persist before the redirect, and
the GET must render exactly that persisted state. What has to survive is not
"the answers" but everything the next step is decided from: the intent, the
unresolved dimensions, the amendments, the authorship, the execution identity,
and the question set. Losing any one of them makes the reopened page a
different situation wearing the same URL.

**No silent loops.** A review is addressed by its content, so a submission that
settles nothing lands on the id it came from. Redirecting there without a word
puts somebody back on the page they just left, which reads as a broken button —
and is exactly what a clarification loop feels like from the outside.
"""
from __future__ import annotations

import os
import re

import pytest
from fastapi.testclient import TestClient

PICKED = "employer-match"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/c.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    return TestClient(app, follow_redirects=False)


def _page(client):
    from urllib.parse import quote

    from src.workspace.strategy_library import entry

    chosen = entry(PICKED)
    got = client.get(
        f"/workspace/new?picked={PICKED}&describe={quote(chosen.text)}")
    assert got.status_code == 200
    return got.text


def _form(html, **overrides):
    body = {"picked": PICKED, "describe": ""}
    found = re.search(r'name="describe" value="([^"]*)"', html)
    if found:
        body["describe"] = found.group(1)
    review = re.search(r'name="from_review" value="([^"]*)"', html)
    if review:
        body["from_review"] = review.group(1)
    for name, value in re.findall(
            r'name="answer_([a-z_]+)"[^>]*value="([^"]*)"', html):
        body[f"answer_{name}"] = value
        original = re.search(rf'name="original_{name}"[^>]*value="([^"]*)"', html)
        author = re.search(rf'name="author_{name}"[^>]*value="([^"]*)"', html)
        if original:
            body[f"original_{name}"] = original.group(1)
        if author:
            body[f"author_{name}"] = author.group(1)
    body.update(overrides)
    return body


def test_the_persisted_state_carries_what_the_next_step_needs(client):
    """Every part named in the plan, present in the stored artifact.

    Asserted key by key rather than "the artifact is non-empty", because the
    failure this guards against is a partial write: a review that reopens with
    the answers but without the question set renders a page that has forgotten
    what it was asking.
    """
    from src.workspace.pilot_store import load_review

    posted = client.post("/pilot/answer", data=_form(_page(client)))
    assert posted.status_code == 303
    stored = load_review(posted.headers["location"].rsplit("/", 1)[-1])

    for key in ("intent", "settled", "open_fields", "absent_fields",
                "refusals", "reader_id", "interpreter_version", "text",
                "picked"):
        assert key in stored, (
            f"{key} is not in the persisted review, so the page rebuilt from "
            "it is a different situation wearing the same URL")

    assert stored["intent"] is not None, "no pinned intent was persisted"
    assert stored["settled"], "no settled record was persisted"


def test_the_redirect_target_exists_before_the_redirect(client):
    """Persist, then redirect. Not the other way round.

    A 303 pointing at a row that was not written is worse than the defect
    being fixed: the answers would be gone *and* the URL would look like it
    held them. Checked by reading the row directly rather than by following
    the redirect, so a handler that wrote on read would not satisfy it.
    """
    from src.workspace.pilot_store import load_review

    posted = client.post("/pilot/answer", data=_form(_page(client)))
    review_id = posted.headers["location"].rsplit("/", 1)[-1].split("?")[0]
    assert load_review(review_id) is not None, (
        "the redirect names a review that is not in the store")


def test_the_rendered_page_is_the_persisted_state(client):
    """What the GET shows is what was written, not a re-derivation.

    Compares the settled values in the store against the values on the page,
    rather than checking the page merely returns 200.
    """
    from src.workspace.pilot_store import load_review

    posted = client.post("/pilot/answer", data=_form(_page(client)))
    location = posted.headers["location"]
    stored = load_review(location.rsplit("/", 1)[-1].split("?")[0])
    rendered = client.get(location).text

    latest = {}
    for entry in stored["settled"]:
        latest[entry["field"]] = str(entry["value"])

    missing = [f"{k}={v}" for k, v in latest.items()
               if v and v not in rendered]
    assert not missing, (
        f"persisted values absent from the page they were persisted for: "
        f"{missing}")


def test_a_submission_that_settles_nothing_says_so(client):
    """The cycle guard.

    Submitting a review's own form back unchanged cannot move the state — the
    values are exactly what produced it. That must land somewhere that says so
    rather than silently re-serving the same page.
    """
    first = client.post("/pilot/answer", data=_form(_page(client)))
    review = client.get(first.headers["location"]).text

    again = client.post("/pilot/answer", data=_form(review))
    assert again.status_code == 303
    location = again.headers["location"]

    assert "stalled=" in location, (
        "a submission that settled nothing redirected with no indication, so "
        f"the person lands on the page they submitted from: {location}")

    rendered = client.get(location).text
    assert "did not change anything" in rendered, (
        "the page does not tell the person their answers moved nothing")


def test_a_real_edit_is_not_reported_as_a_lap(client):
    """The guard must discriminate, or it is noise on every submission."""
    html = _page(client)
    edited = client.post("/pilot/answer",
                         data=_form(html, answer_amount="$750"))
    assert edited.status_code == 303
    assert "stalled=" not in edited.headers["location"], (
        "an edit that changed the state was reported as settling nothing")


def test_the_notice_is_not_part_of_the_stored_state(client):
    """It describes the transition, not the state.

    The same review URL without the query renders the same page. If the notice
    were stored, a review reached later by an ordinary link would claim a lap
    that never happened.
    """
    first = client.post("/pilot/answer", data=_form(_page(client)))
    review = client.get(first.headers["location"]).text
    again = client.post("/pilot/answer", data=_form(review))

    location = again.headers["location"]
    plain = location.split("?")[0]
    assert "did not change anything" not in client.get(plain).text, (
        "the stalled notice is rendered without the query that caused it, so "
        "it has become part of the state rather than the transition")
