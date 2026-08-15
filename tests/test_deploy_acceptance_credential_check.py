"""The check that asks whether the private surface is open, checked itself.

It has now been wrong in both directions on the same deployment, one after the
other, which is why it gets its own file.

It expected 401 or 403 — the proxy's shared password — so when the deployment
moved to accounts and answered `303 -> /auth/login`, it reported the workspace
as open while `/pilot` was genuinely open beside it, unprobed. Then, fixed to
accept a redirect, it still reported both paths open: `fetch` follows
redirects, so it never saw the `303` at all and read the `200` from the login
page as the workspace answering.

Both readings say "the private surface is open" about a site that is refusing
correctly. A check that cannot tell a refusal from a breach, in the direction
of crying breach, is muted after the second false alarm — and then it is not
there for the real one.

So this runs the real script against a real server that answers in each way
that matters. No mocking of `fetch`: the two defects were both *in* `fetch` and
its caller, and a test that replaced it would have passed through both.
"""
from __future__ import annotations

import http.server
import json
import socket
import threading

import pytest

from deploy.acceptance import fetch, main


class Handler(http.server.BaseHTTPRequestHandler):
    """Answers the private surface however the test asked it to."""

    behaviour = "refuse"

    def log_message(self, *args):
        pass

    def do_GET(self):                                        # noqa: N802
        if self.path.startswith("/workspace") or self.path.startswith("/pilot"):
            return self._private()
        if self.path in ("/health/ready", "/ready"):
            # `ready: true` and nothing about why. The shape the check reads,
            # copied from the deployment rather than guessed, so a fake that
            # drifted from the real contract would fail here rather than
            # quietly test a different agreement.
            return self._json({"ready": True})
        if self.path in ("/health", "/health/live"):
            return self._json({"status": "ok", "build": {"observable": True}})
        if self.path == "/info":
            return self._json({"personalization": {"enabled": False}})
        # Carries a request_id, because the leak checks below it read this
        # body: an error page with no correlation id fails a check that has
        # nothing to do with the private surface, and a red line nobody caused
        # is how a test file stops being read.
        return self._json({"detail": "not found",
                           "request_id": "req-test"}, status=404,
                          correlated=True)

    def _private(self):
        if self.behaviour == "refuse":
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"no")
        elif self.behaviour == "redirect_to_login":
            self.send_response(303)
            self.send_header("Location", "/auth/login?next=" + self.path)
            self.end_headers()
        elif self.behaviour == "redirect_elsewhere":
            # A redirect is not a refusal by itself. Bouncing a signed-out
            # request to the front page leaves the surface reachable by anyone
            # who does not follow it.
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
        else:                                                # "open"
            self._json({"plans": ["somebody else's"]})

    def _json(self, payload, status=200, correlated=False):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if correlated:
            self.send_header("x-request-id", "req-test")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    httpd = http.server.HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def acceptance_says(base, behaviour, tmp_path):
    Handler.behaviour = behaviour
    record = tmp_path / "acceptance.json"
    code = main(base, record_to=str(record))
    return code, json.loads(record.read_text())


def private_results(record):
    return [one for one in record["checks"]
            if "requires a credential" in one["name"]]


class TestItRecognisesARefusal:
    def test_a_403_passes(self, server, tmp_path):
        code, record = acceptance_says(server, "refuse", tmp_path)
        assert all(one["passed"] for one in private_results(record)), record
        assert code == 0

    def test_a_redirect_to_the_login_passes(self, server, tmp_path):
        """The defect that made this file necessary the second time.

        `fetch` follows redirects, so before `follow=False` this arrived as a
        200 from the login page and was reported as the workspace being open.
        """
        code, record = acceptance_says(server, "redirect_to_login", tmp_path)
        assert all(one["passed"] for one in private_results(record)), record
        assert code == 0


class TestItStillRecognisesAnOpenSurface:
    """Without these the fix above is indistinguishable from deleting the check."""

    def test_a_200_fails(self, server, tmp_path):
        code, record = acceptance_says(server, "open", tmp_path)
        failed = [one for one in private_results(record) if not one["passed"]]
        assert failed, "an open private surface passed"
        assert code != 0

    def test_a_redirect_that_is_not_a_login_fails(self, server, tmp_path):
        """A redirect is not a refusal by itself.

        Accepting any 3xx would have been the easy way to make the live run
        green, and it would pass a deployment that bounces signed-out users to
        the front page while still serving the workspace to anyone who ignores
        the redirect.
        """
        code, record = acceptance_says(server, "redirect_elsewhere", tmp_path)
        failed = [one for one in private_results(record) if not one["passed"]]
        assert failed, "a redirect away from any login was accepted as refusal"
        assert code != 0


class TestBothMountsAreProbed:
    def test_every_private_prefix_gets_its_own_check(self, server, tmp_path):
        """One line per mount, because one line for two paths hides which.

        The first version probed `/workspace/` alone. `/pilot` was open and
        there was no check whose failure could have said so.
        """
        from src.api import PRIVATE_PREFIXES

        _code, record = acceptance_says(server, "refuse", tmp_path)
        names = {one["name"] for one in private_results(record)}
        assert len(names) == len(PRIVATE_PREFIXES)
        for prefix in PRIVATE_PREFIXES:
            assert any(prefix in name for name in names), (
                f"{prefix} is gated by the application and never probed here")


class TestFetchItself:
    def test_following_is_the_default(self, server):
        Handler.behaviour = "redirect_to_login"
        status, _body, _headers = fetch(server, "/workspace/")
        assert status != 303, "the default stopped following redirects"

    def test_not_following_returns_the_redirect(self, server):
        Handler.behaviour = "redirect_to_login"
        status, _body, headers = fetch(server, "/workspace/", follow=False)
        assert status == 303
        location = {name.lower(): value for name, value in headers.items()}
        assert "/auth/login" in location["location"]
