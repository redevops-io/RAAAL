"""The synthetic-data caveat appears wherever the service describes itself.

`/info` is linked from the footer of every page as "Service details" and is
what an integrator reads. It carried the demo notice and the licence and said
nothing about the numbers being invented, so the disclosure that appears on
every HTML page stopped at the one endpoint whose whole purpose is to answer
"what is this".

Found by running the pre-invite acceptance checks rather than by any test —
the HTML pages were covered and the JSON surface was not.

Read from `_data_notice` rather than restated. A second copy of a disclosure
is the copy that goes stale when the policy changes, and this one is keyed on
the deployment's actual data policy, so a licensed deployment says nothing
rather than saying something false.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import src.api as api
    from src.deploy.context import bind, resolve, unbind

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    api._bootstrap()
    try:
        yield TestClient(api.app)
    finally:
        unbind()


class TestTheServiceDescriptionCarriesIt:
    def test_info_declares_the_data_policy(self, client):
        body = client.get("/info").json()
        assert body.get("data_policy"), (
            "the endpoint that describes the service does not say the figures "
            "are synthetic")

    def test_it_says_what_the_pages_say(self, client):
        """One source. A hand-written second copy is what goes stale."""
        from src.workspace.routes import _data_notice

        assert client.get("/info").json()["data_policy"] == _data_notice()

    def test_it_names_the_consequence_not_just_the_fact(self, client):
        detail = client.get("/info").json()["data_policy"]["detail"].lower()
        assert "not based on licensed" in detail


class TestItDoesNotLeakOntoTheWrongSurface:
    def test_health_stays_a_liveness_probe(self, client):
        """The first attempt put it here, because `health` and `info` share a
        line and a blind replace took the first. `/health` answers whether the
        process is alive to a load balancer; it is not a disclosure surface."""
        assert "data_policy" not in client.get("/health").json()


class TestALicensedDeploymentSaysNothing:
    def test_the_notice_is_absent_rather_than_false(self, monkeypatch):
        """Keyed on the deployment's policy. A hardcoded string would claim
        synthetic data on a deployment serving licensed prices, which is the
        same defect pointing the other way."""
        from src.api import _service_data_policy
        from src.deploy.context import bind, resolve, unbind

        monkeypatch.setenv("PILOT_DATA_POLICY", "LICENSED")
        try:
            bind(resolve({"PILOT_DATA_POLICY": "LICENSED"}))
        except Exception:
            pytest.skip("this build does not accept a licensed policy value")
        try:
            assert _service_data_policy() is None
        finally:
            unbind()
