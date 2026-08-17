"""`/pilot/answer` is a write that redirects; the review URL is a read.

The defect these cover was live on quantify.club. `POST /pilot/answer`
rendered HTML at its own URL and stored nothing, so:

    refresh / Back / Forward / paste
        -> GET against a POST-only route
        -> 405 Method Not Allowed

and Back returned to the last real GET — the empty form — discarding every
value the person had typed. The answers existed only in a request body.

The exit tests named in the plan, one test each:

    POST returns 303
    the redirected GET renders the persisted answers
    refresh produces no new Discovery call
    back/forward preserves values
    repeated GET creates no duplicate plan or answer

The third is the one that cannot be taken on trust, so it is not asserted
against latency, text equality, or plan ids. The reader is replaced with one
that records every call, and the assertion is over that record.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A deployment that declares the runtime, on its own database.

    The recorded reader, not a live one: these tests are about navigation, and
    a test that reaches a hosted model to prove a redirect works is a test that
    fails for reasons it is not about.
    """
    import os

    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/p.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    return TestClient(app, follow_redirects=False)


#: A sentence the recorded reader actually holds. Chosen from the corpus
#: rather than invented: the recorded reader refuses an unknown sentence
#: outright, which is correct — a fixture that quietly called the provider
#: would make these navigation tests depend on a hosted model.
RECORDED = "invest $500 monthly into VTI"


def _submit(client, **fields):
    body = {"describe": RECORDED, "picked": ""}
    body.update(fields)
    return client.post("/pilot/answer", data=body)


def test_post_redirects_rather_than_rendering(client):
    """303, and to a URL that is not the one just posted to.

    A 200 here is the defect itself: it means the browser is parked on a
    POST-only URL and every later navigation to it fails.
    """
    posted = _submit(client)
    assert posted.status_code == 303, (
        f"POST /pilot/answer returned {posted.status_code}, not a redirect. "
        "Rendering at the POST URL is what makes refresh and Back fail.")
    location = posted.headers.get("location", "")
    assert location.startswith("/pilot/reviews/"), location


def test_the_review_url_answers_get(client):
    """The URL the browser is left on must be re-requestable.

    This is the assertion that would have caught the production 405: not "the
    POST worked", but "the address it left in the bar responds to the method a
    browser uses when somebody presses reload".
    """
    posted = _submit(client)
    assert posted.status_code == 303, "the POST must redirect for this to mean anything"
    location = posted.headers["location"]
    got = client.get(location)
    assert got.status_code == 200, (
        f"GET {location} returned {got.status_code}. The browser sits on this "
        "URL after submitting; a refresh or a Back must not fail.")


def test_refresh_is_byte_identical(client):
    """Re-requesting renders the same page, not a new interpretation."""
    posted = _submit(client)
    assert posted.status_code == 303, "the POST must redirect for this to mean anything"
    location = posted.headers["location"]
    first = client.get(location)
    second = client.get(location)
    assert first.status_code == second.status_code == 200
    assert first.text == second.text, (
        "two GETs of one review rendered differently, so something on the "
        "read path is not a read")


def test_the_same_answers_address_the_same_review(client):
    """Submitting identical answers twice must not mint a second review.

    Content addressing, and the reason it matters here: a person who
    double-submits, or whose browser retries, must land on the state they are
    already looking at rather than accumulating near-identical rows.
    """
    first = _submit(client)
    assert first.status_code == 303, "the POST must redirect for this to mean anything"
    second = _submit(client)

    assert first.headers["location"] == second.headers["location"], (
        "identical submissions produced different review URLs")


def test_a_missing_review_is_404_not_an_empty_form(client):
    """A stale link says the state is gone.

    Redirecting to a fresh form would show a person an empty page that looks
    exactly like their submission never happened — which is the failure mode
    being fixed, wearing a 200.
    """
    got = client.get("/pilot/reviews/review-doesnotexist")
    assert got.status_code == 404


def test_the_review_get_cannot_construct_a_reader(client, monkeypatch):
    """The no-new-Discovery-call property, witnessed rather than inferred.

    Not asserted from latency, identical rendered text, or matching plan ids —
    all three are correlated with the property and none of them is it. The
    reader factory is replaced with one that raises, and the GET is required to
    succeed anyway. If anything on the read path reaches for a reader, this
    fails with that exception rather than with a judgement call.
    """
    posted = _submit(client)
    assert posted.status_code == 303
    location = posted.headers["location"]

    from src.workspace import pilot_routes

    called = []

    def refuse():
        called.append(1)
        raise AssertionError(
            "the review GET constructed a reader. Reopening persisted "
            "clarification state is a read; interpreting the sentence again "
            "makes a refresh a new Discovery Mission.")

    monkeypatch.setattr(pilot_routes, "configured_reader", refuse)
    monkeypatch.setattr(pilot_routes, "configured_syntax_reader", refuse)

    got = client.get(location)
    assert got.status_code == 200
    assert not called, "a reader was constructed on the read path"


def test_the_answer_survives_the_redirect(client):
    """Back and Forward preserve what was typed, because the server holds it.

    A browser's Back re-requests the previous URL. The values are therefore
    preserved exactly if the redirected GET renders them — which is the thing
    that was broken: the answers lived only in a POST body, so Back returned to
    an empty form.

    Asserts the submitted value appears in the rendered review, not merely that
    the page is 200.
    """
    posted = _submit(client, answer_dividend_policy="reinvested")
    assert posted.status_code == 303

    rendered = client.get(posted.headers["location"])
    assert rendered.status_code == 200
    assert "reinvested" in rendered.text, (
        "the value submitted with the answer is not on the page the redirect "
        "led to, so pressing Back would lose it")


def test_an_unsupported_answer_survives_as_a_named_refusal(client):
    """The value is kept and refused by name — not dropped, not substituted.

    `held as cash` canonicalises to `held_as_cash`, which the engine cannot
    model: it runs on a total-return series and has no path that pays
    distributions out and leaves them idle. The rule is that an unsupported
    *stated* value refuses and never silently becomes the supported neighbour,
    so this asserts three things at once — the answer persisted across the
    redirect, it was canonicalised rather than discarded, and the page says
    which semantic is unsupported instead of quietly running `reinvested` and
    reporting a better number.
    """
    posted = _submit(client, answer_dividend_policy="held as cash")
    assert posted.status_code == 303

    rendered = client.get(posted.headers["location"]).text
    assert "held_as_cash" in rendered, (
        "the unsupported answer did not survive the redirect")
    assert "p-refused" in rendered, "it was not marked as a refusal"
    assert "not modelled" in rendered, (
        "the page does not name why the semantic is unsupported")
