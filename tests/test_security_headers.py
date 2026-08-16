"""The response headers, and the fact that they belong to the application.

They were Caddy's. Moving the deployment to Kubernetes removed Caddy and took
all four with it, and nothing noticed — the site served correctly, every health
check passed, and the headers were simply gone. That is the failure this file
exists to make loud: their absence has no symptom.

So the test is not "the proxy is configured". It is "an application response
carries them", which stays true whatever is in front.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import SECURITY_HEADERS, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as ready:
        yield ready


@pytest.mark.parametrize("header,value", sorted(SECURITY_HEADERS.items()))
def test_every_security_header_is_present(client, header, value):
    """Named individually, so a failure says which one went missing."""
    response = client.get("/health/live")
    assert response.headers.get(header) == value, (
        f"{header} is absent or weakened. It used to be supplied by the "
        f"reverse proxy, and the proxy is not always there.")


def test_headers_survive_a_failing_request(client):
    """A 404 is a response too.

    Middleware that only decorates the happy path leaves every error response
    bare — and error pages are where a sniffed content type or a leaked
    referrer actually costs something.
    """
    response = client.get("/definitely-not-a-route-here")
    assert response.status_code == 404
    missing = [h for h in SECURITY_HEADERS if h not in response.headers]
    assert not missing, f"absent on a 404: {missing}"


def test_a_plain_fastapi_app_has_none_of_them():
    """The control, and the reason the assertions above mean anything.

    An identical request to an application without the middleware must come
    back bare. Without this, "the header is present" could be FastAPI, or
    Starlette, or the test client being helpful, and the tests above would pass
    just as well with the middleware deleted.
    """
    from fastapi import FastAPI

    bare = FastAPI()

    @bare.get("/health/live")
    def live():
        return {"status": "ok"}

    with TestClient(bare) as plain:
        response = plain.get("/health/live")

    present = [h for h in SECURITY_HEADERS if h in response.headers]
    assert not present, (
        f"{present} appears without any middleware, so its presence on the "
        "real application is not evidence the application sets it")
