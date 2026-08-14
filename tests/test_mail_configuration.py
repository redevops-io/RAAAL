"""The mail-configuration script, against the answers the provider gives.

It failed on its first real run, and on the only run that mattered: the list
endpoint returns 404 — "SMTP configuration not found" — when nothing is
configured yet, rather than an empty list. Treating every 404 as fatal meant
the script worked in exactly the case where mail was already set up and failed
in the case it existed for.

The script is a Jinja template rendered onto a host, so it is loaded here the
way the host would run it: rendered, then executed as a module. The provider is
answered in-process, so nothing reaches a network.
"""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2")

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = (ROOT / "infra" / "ansible" / "roles" / "quantify" / "templates"
            / "configure-mail.py.j2")


def script():
    """The rendered script, as a module."""
    import importlib.util
    import tempfile

    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    if not TEMPLATE.exists():
        pytest.skip("no mail template here")
    environment = Environment(loader=FileSystemLoader(str(TEMPLATE.parent)),
                              undefined=StrictUndefined, trim_blocks=True)
    rendered = environment.get_template(TEMPLATE.name).render(
        quantify_identity_domain="auth.quantify.test",
        quantify_region="us-east-1",
        quantify_secret_smtp="quantify-test/smtp-credentials")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(rendered)
        path = handle.name
    spec = importlib.util.spec_from_file_location("configure_mail", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def http_error(code: int, body: bytes = b"{}"):
    return urllib.error.HTTPError("http://x", code, "no", {}, BytesIO(body))


class TestAnAbsentConfigurationIsNotAFailure:
    """The bug, named.

    A provider with no SMTP configured answers the list with 404. That is the
    state every first deploy is in.
    """

    def test_the_list_endpoint_returning_404_reads_as_empty(self, monkeypatch):
        module = script()

        def urlopen(request, timeout=None):
            raise http_error(404, b'{"message": "SMTP configuration not found"}')

        monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
        assert module.call("/admin/v1/email/providers", "tok",
                           absent_ok=True) == {}

    def test_a_404_is_still_fatal_where_it_is_not_expected(self, monkeypatch):
        """The tolerance is per-call, not global. A 404 from the endpoint that
        creates the provider means the path is wrong, and swallowing it would
        report configured mail that does not exist."""
        module = script()

        monkeypatch.setattr(
            module.urllib.request, "urlopen",
            lambda request, timeout=None: (_ for _ in ()).throw(http_error(404)))
        with pytest.raises(SystemExit):
            module.call("/admin/v1/email/smtp", "tok", {"host": "x"})

    def test_other_failures_are_still_fatal(self, monkeypatch):
        """401 means the token is wrong. Continuing would leave the provider
        with no sender and the deploy reporting success."""
        module = script()

        monkeypatch.setattr(
            module.urllib.request, "urlopen",
            lambda request, timeout=None: (_ for _ in ()).throw(http_error(401)))
        with pytest.raises(SystemExit):
            module.call("/admin/v1/email/providers", "tok", absent_ok=True)


class TestItAsksTheProviderTheRightWay:
    def test_requests_carry_the_host_that_selects_the_identity_site(
            self, monkeypatch):
        """Through the proxy on this host, not out through Cloudflare — where
        a management call was already answered with a bot rule."""
        module = script()
        seen = {}

        class Response:
            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def urlopen(request, timeout=None):
            seen["url"] = request.full_url
            seen["host"] = request.get_header("Host")
            seen["scheme"] = request.get_header("X-forwarded-proto")
            return Response()

        monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
        module.call("/admin/v1/email/providers", "tok")

        assert seen["url"].startswith("http://127.0.0.1/")
        assert seen["host"] == "auth.quantify.test"
        assert seen["scheme"] == "https"

    def test_activation_addresses_the_provider_by_id(self, monkeypatch):
        """`/admin/v1/email/{id}/_activate`. The first version posted to a
        path that does not exist, which would have left a configured provider
        inactive — mail silently unsent, and the deploy reporting success."""
        module = script()
        called = []

        class Response:
            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        def urlopen(request, timeout=None):
            called.append(request.full_url)
            return Response()

        monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
        module.call("/admin/v1/email/abc123/_activate", "tok", {})
        assert called == ["http://127.0.0.1/admin/v1/email/abc123/_activate"]


class TestNoCredentialsIsReportedRatherThanGuessed:
    def test_it_says_what_the_consequence_is(self, monkeypatch, capsys):
        """A deployment may legitimately have no sender yet. Failing the deploy
        would not explain it any better — but neither would silence, because
        registration then completes and sign-in refuses."""
        module = script()
        monkeypatch.setattr(module, "credentials", lambda: {"host": "", "user": ""})

        assert module.main() == 0
        printed = capsys.readouterr().out
        assert "MAIL_NOT_CONFIGURED" in printed
        assert "sign-in will then refuse" in printed
