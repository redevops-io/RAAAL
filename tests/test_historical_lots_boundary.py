"""A described holding is refused, not folded into starting capital.

Before this, "I already own 500 shares of AAPL that I bought in 2019 at $50"
compiled with no material blocker and no stated fields — the holding was
simply dropped. "I bought 10 shares of NVDA in May 2024" was asked *how much
are you starting with*, which turns a share count into a cash amount.

`src/holdings/` can resolve a lot across a split correctly and is wired to no
user surface. Until it is, the honest answer is a refusal that names itself.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.mission.compiler import compile_scenario
from src.workspace import historical_lots
from src.workspace.feasibility import Resolution, blockers

#: Priceable instruments deliberately. With an unpriceable ticker the
#: market-data refusal fires first and the holdings guard is never reached —
#: which is how the first version of this file passed while proving nothing.
HOLDINGS = [
    "I already own 500 shares of VOO that I bought in 2019 at $50 a share.",
    "I hold 100 shares of VTI with a cost basis of $12,000 and add $500 a month to VTI.",
    "I have $80,000 of VTI I purchased over the last five years, and I contribute $1,000 monthly.",
    "I bought 10 shares of VOO in May 2024 before the split.",
    "I have RSUs that vested last year and I want to diversify.",
]

SUPPORTED = [
    "I contribute $7,000 a year to a Roth IRA in VOO on the first trading day "
    "of January, reinvesting dividends, and I never sell.",
    "I put $500 a month into VTI starting next month and never sell.",
]


class TestTheBoundaryDiscriminates:
    @pytest.mark.parametrize("text", HOLDINGS)
    def test_a_described_holding_blocks(self, text):
        scenario = compile_scenario(text).scenario
        found = blockers(scenario, stated_text=text)
        assert found.material, f"no material blocker for: {text}"
        assert any(historical_lots.HISTORICAL_LOTS_NOT_AVAILABLE in one.field
                   for one in found.material)

    @pytest.mark.parametrize("text", SUPPORTED)
    def test_a_supported_journey_is_untouched(self, text):
        """The other half. A guard that blocks everything is not a boundary."""
        scenario = compile_scenario(text).scenario
        found = blockers(scenario, stated_text=text)
        assert not any(historical_lots.HISTORICAL_LOTS_NOT_AVAILABLE in one.field
                       for one in found.material)


class TestItIsMaterialNotDismissible:
    def test_the_user_cannot_acknowledge_it_away(self):
        """An existing holding is not extra prose beside the request. It is a
        claim about what the figure covers, and dismissing it answers a
        different question."""
        text = HOLDINGS[0]
        found = blockers(compile_scenario(text).scenario, stated_text=text)
        item = next(one for one in found.material
                    if historical_lots.HISTORICAL_LOTS_NOT_AVAILABLE in one.field)
        assert item.resolution is Resolution.MATERIAL
        assert not item.dismissible

    def test_the_refusal_explains_why_rather_than_naming_a_code(self):
        text = HOLDINGS[0]
        found = blockers(compile_scenario(text).scenario, stated_text=text)
        item = next(one for one in found.material
                    if historical_lots.HISTORICAL_LOTS_NOT_AVAILABLE in one.field)
        why = item.why_it_matters.lower()
        assert "no longer exist" in why
        assert "available to invest" in why
        # And the page must not show the operator's code as a sentence.
        assert historical_lots.HISTORICAL_LOTS_NOT_AVAILABLE.lower() \
            not in found.detail().lower()
        assert "an existing holding" in found.detail().lower()


class TestTheLiveRouteRefuses:
    """The guard reads text the scenario does not carry, so it had to be
    passed at each call site. Written against `scenario.stated_text` it would
    have been dead on exactly the path that matters."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """The isolated store the other route tests use.

        Written against the ambient database first, the control case returned
        a 500 from whatever state that database happened to be in — which
        would have made "the refusal works" a claim resting on a broken
        comparison.
        """
        import src.api as api
        import src.web.routes as web_routes
        import src.workspace.routes as workspace_routes
        from src.ledger import Ledger
        from src.workspace.store import WorkspaceStore

        ledger = Ledger(tmp_path / "public.db")
        monkeypatch.setattr(api, "_ledger", ledger)
        monkeypatch.setattr(web_routes, "Ledger", lambda *a, **k: ledger)
        store = WorkspaceStore(tmp_path / "workspace.db")
        monkeypatch.setattr(workspace_routes, "_store", lambda: store)
        api._bootstrap()
        return TestClient(api.app)

    def test_the_confirmation_screen_says_so(self, client):
        page = client.get("/workspace/new", params={"describe": HOLDINGS[0]})
        assert page.status_code == 200
        assert "already own" in page.text.lower() or "existing position" in page.text.lower()

    def test_saving_is_refused_and_says_why(self, client):
        """A 422 alone proves nothing.

        Written as `assert status == 422`, this passed with the guard removed
        from the save path entirely — every one of these descriptions has
        other open questions, and any of them returns 422. The refusal has to
        *name the holding*, or the test is about the account-type question.
        """
        from tests.conftest import submit_rendered_confirmation

        response, plan_id = submit_rendered_confirmation(
            client, HOLDINGS[0], title="Existing VOO")

        assert response.status_code == 422, response.status_code
        assert plan_id is None, "a plan naming an existing holding was saved"
        assert "existing holding" in response.text.lower(), (
            "the save was refused for some other reason; the historical-lot "
            "guard is not on this path")

    def test_the_answered_plan_is_still_refused(self, client):
        """The strongest form: answer every ordinary question and the holding
        must still block. Otherwise the guard only works while something else
        happens to be outstanding."""
        from tests.conftest import submit_rendered_confirmation

        text = ("I already own 500 shares of VOO that I bought in 2019 at $50 "
                "a share, in a taxable brokerage account, and I add $1,000 "
                "every month, reinvesting dividends.")
        response, plan_id = submit_rendered_confirmation(
            client, text, title="Existing VOO answered")

        assert response.status_code == 422
        assert plan_id is None
        assert "existing holding" in response.text.lower()

    def test_an_unpriceable_holding_names_the_holding_not_the_ticker(self, client):
        """The ordering case, and the only one that discriminates it.

        Asked to model 500 shares of AAPL — which the pilot cannot price —
        the feasibility check answers "there is no price history for AAPL".
        True, and misleading: the plan would not run with prices either. A
        user reads that as an instrument problem and tries another ticker.

        With a priceable instrument both refusals agree, so this is the only
        case where running the unconditional guard first is observable.
        """
        from tests.conftest import submit_rendered_confirmation

        response, plan_id = submit_rendered_confirmation(
            client,
            "I already own 500 shares of AAPL that I bought in 2019 at $50.",
            title="Unpriceable holding")

        assert response.status_code == 422
        assert plan_id is None
        body = response.text.lower()
        assert "existing holding" in body, (
            "the refusal blamed the instrument for a limitation that has "
            "nothing to do with which instrument it is")
        assert "no price history" not in body

    def test_a_supported_journey_still_saves(self, client):
        """The premise. If nothing saved, the refusal above would prove
        nothing about historical lots."""
        from tests.conftest import submit_rendered_confirmation

        response, plan_id = submit_rendered_confirmation(
            client, SUPPORTED[0], title="Roth")
        assert response.status_code == 303, response.text
        assert plan_id


class TestDetectionCoversHowPeopleActuallyWrite:
    @pytest.mark.parametrize("text,expected", [
        ("I own 250 shares of MSFT.", True),
        ("my cost basis is about $40 a share", True),
        ("I purchased VTI back in 2016", True),
        ("shares from my vesting last April", True),
        ("I want to buy $500 of VTI every month", False),
        ("I will hold it for thirty years", False),
        ("I never sell anything", False),
    ])
    def test_phrasing(self, text, expected):
        assert historical_lots.blocks(text) is expected
