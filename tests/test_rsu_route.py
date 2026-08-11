"""The HTTP boundary for a capability this build cannot execute.

    describe -> parse -> template_hint -> 501 unavailable

`RSUDeclaration` is consumed by the route that builds it, the card that renders
it, and their own tests. **Nothing turns a declaration into a scenario, a run
or a worksheet.** Vest events are not cash flows the compiler understands, and
there is no `compile_rsu_declaration`.

So the confirmation card was a polished surface in front of an unimplemented
feature — a declaration with no reachable behaviour, which is the shape this
codebase exists to remove. Rendering it implied that saving was one step away.
It was not: there is no save path, and building the form would have produced a
submit button with nothing to submit to.

The pilot therefore launches with the two scenarios that complete end to end,
and RSU is post-pilot feature one. What this file now protects is the honesty
of the refusal:

    the description is still recognised as equity compensation
    the generic compiler is still never reached
    nothing plan-shaped is created
    the user is told before investing in refining a description

The component work — declaration, card, vest runtime, handoff — is unchanged
and still covered by `test_rsu_confirmation.py`, `test_rsu_handoff.py`,
`test_rsu_vesting_runtime.py` and `test_rsu_template.py`: 157 tests that all
still pass. It has not been assembled into a product path, which is a different
statement from it being absent.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app

RSU = ("100 ACME shares vest quarterly. Withhold 22% in shares. "
       "Sell as soon as I can after the blackout window. "
       "Keep company stock below 20%. "
       "Allocate proceeds 60% VTI, 30% VXUS, 10% BND.")

GENERIC = "I put $500 into SPY every month in my taxable brokerage"

RECURRING_SHARES = "I receive 100 ACME shares quarterly"


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def draft(client, description):
    return client.get("/workspace/new", params={"describe": description})


class TestVestLanguageIsStillRecognised:
    """Deferring the feature must not silently degrade recognition. A vest
    description read as an ordinary contribution would be worse than a
    refusal — it would produce a number."""

    def test_it_is_routed_as_equity_compensation(self, client):
        assert draft(client, RSU).status_code == 501

    def test_a_generic_description_is_unaffected(self, client):
        assert draft(client, GENERIC).status_code == 200

    def test_recurring_shares_without_vest_language_stay_generic(self, client):
        """A DRIP, a gift and a transfer are not vests. Routing them to the
        unavailable surface would refuse a scenario the product supports —
        the mirror of routing them to the vesting runtime, which would invent
        semantics the template prevents.

        Flattened into the parametrised case above when this file was
        rewritten, which turned a distinction the original drew deliberately
        into a bug. Restored.
        """
        assert draft(client, RECURRING_SHARES).status_code == 200


class TestTheGenericCompilerIsNeverReached:
    """The original property, and it still holds. A vest read as cash arriving
    and then a purchase is the silent misreading this dispatch exists to
    prevent, and that is true whether or not the feature is available."""

    def test_an_rsu_request_does_not_compile(self, client, monkeypatch):
        import src.workspace.routes as routes

        def refuse(*args, **kwargs):
            raise AssertionError(
                "the generic compiler was reached for a vest description")

        monkeypatch.setattr(routes, "compile_scenario", refuse)
        assert draft(client, RSU).status_code == 501

    def test_an_unregistered_hint_fails_rather_than_falling_back(self, client,
                                                                  monkeypatch):
        import src.workspace.routes as routes

        monkeypatch.setattr(routes, "UNAVAILABLE_TEMPLATES", {})
        response = draft(client, RSU)
        assert response.status_code == 501
        assert "no handler" in response.text.lower()


class TestTheRefusalIsHonest:
    def test_it_says_the_capability_is_unavailable(self, client):
        assert "not available in this pilot" in draft(client, RSU).text.lower()

    def test_it_says_what_was_recognised(self, client):
        """A user told only "unsupported" cannot tell whether the product
        misread them or declined them."""
        assert "equity" in draft(client, RSU).text.lower()

    def test_it_names_what_does_work(self, client):
        body = draft(client, RSU).text.lower()
        assert "contribution" in body or "historical" in body

    def test_it_offers_no_save(self, client):
        body = draft(client, RSU).text
        assert "Save this plan" not in body
        assert "/workspace/save" not in body

    def test_the_description_is_returned_for_editing(self, client):
        """Returned, not stored. The user keeps their words without a
        plan-shaped record existing for a plan that cannot be built."""
        assert "vest quarterly" in draft(client, RSU).text


class TestNothingPlanShapedIsCreated:
    """A draft that cannot become a plan is a record whose only purpose is to
    look like progress."""

    def test_no_plan_is_written(self, client, tmp_path, monkeypatch):
        import src.workspace.routes as routes
        from src.workspace.store import WorkspaceStore

        store = WorkspaceStore(tmp_path / "w.db")
        monkeypatch.setattr(routes, "_store", lambda: store)
        draft(client, RSU)
        assert store.list_plans("pilot") == []

    def test_no_worksheet_is_written(self, client, tmp_path, monkeypatch):
        import src.workspace.routes as routes
        from src.workspace.store import WorkspaceStore

        store = WorkspaceStore(tmp_path / "w.db")
        monkeypatch.setattr(routes, "_store", lambda: store)
        draft(client, RSU)
        with store._conn() as conn:
            assert conn.execute(
                "SELECT COUNT(*) AS n FROM worksheet").fetchone()["n"] == 0


class TestTheDeferralIsRecorded:
    """So it is a decision someone made rather than a page that quietly
    stopped working."""

    def test_the_capability_is_named_with_a_reason(self):
        from src.workspace.routes import UNAVAILABLE_TEMPLATES

        assert "rsu-vesting" in UNAVAILABLE_TEMPLATES
        assert len(UNAVAILABLE_TEMPLATES["rsu-vesting"]) > 80

    def test_no_compiler_exists_for_a_declaration(self):
        """The reason for the deferral, asserted rather than described. When
        this fails, the feature has a destination and this file should be
        rewritten around the journey rather than the refusal."""
        import src.mission.rsu_declaration as declaration

        assert not hasattr(declaration, "compile_rsu_declaration"), (
            "a declaration compiler now exists; RSU is no longer blocked on "
            "its executable model and this refusal should be reconsidered")
