"""One complete user journey, end to end.

Eleven steps, in order, each depending on the last. A suite of green unit tests
is compatible with a product nobody can complete a task in; this asserts that the
task completes and that every protection survives the whole path rather than
only the point where it was introduced.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from src.mission import (
    CashPolicy,
    ComparisonClass,
    Eligibility,
    ExpectedEvent,
    ObservedEvent,
    PlanObservation,
    Proposal,
    ProposalStatus,
    RunConditions,
    classify_counterfactual,
    compile_scenario,
    expire_overdue,
    reconcile,
    scan_language,
    simulate,
)
from src.mission.scenario import ScenarioSpecification
from src.mission.spec import Inference, Provenance
from src.mission.templates import (
    RSU_TEMPLATE,
    disposition_program,
    grants_for,
)
from src.workspace.store import WorkspaceStore

DESCRIPTION = ("My RSUs vest quarterly. I sell them as soon as I can and put the "
               "money into SPY.")


@pytest.fixture
def prices():
    idx = pd.bdate_range("2021-01-04", periods=520)
    return pd.DataFrame(
        {"ACME": np.linspace(50.0, 90.0, 520), "SPY": np.linspace(100.0, 140.0, 520)},
        index=idx,
    )


@pytest.fixture
def store(tmp_path):
    return WorkspaceStore(tmp_path / "journey.db")


@pytest.fixture
def client(tmp_path, monkeypatch, store):
    from fastapi.testclient import TestClient

    import src.api as api
    import src.web.routes as web_routes
    import src.workspace.routes as workspace_routes
    from src.ledger import Ledger

    ledger = Ledger(tmp_path / "public.db")
    monkeypatch.setattr(api, "_ledger", ledger)
    monkeypatch.setattr(web_routes, "Ledger", lambda *a, **k: ledger)
    monkeypatch.setattr(workspace_routes, "_store", lambda: store)
    api._bootstrap()
    return TestClient(api.app)


def text(html: str) -> str:
    body = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


def rsu_inputs():
    return {
        "ticker": "ACME",
        "vest_dates": ["2021-02-15", "2021-05-17", "2021-08-16", "2021-11-15"],
        "shares_per_vest": 100.0,
        "withholding_rate": 0.22,
        "blackout_windows": [("2021-08-01", "2021-09-10")],
        "disposition": "sell_all_and_diversify",
        "diversify_into": "SPY",
    }


def confirmed(scenario):
    p = scenario.provenance
    return ScenarioSpecification(**{
        **scenario.__dict__,
        "provenance": Provenance(
            stated=p.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in p.inferred),
            contradictions=p.contradictions, unresolved=()),
    })


class TestTheWholeJourney:
    """Each step asserts the product still holds its shape at that point."""

    def test_01_the_user_describes_an_rsu_scenario(self):
        compiled = compile_scenario(DESCRIPTION)

        assert compiled.template_hint == "rsu-vesting", (
            "the compiler must hand off to the template rather than improvise "
            "vesting rules from prose"
        )

    def test_02_the_compiler_surfaces_what_it_did_and_did_not_decide(self):
        compiled = compile_scenario(DESCRIPTION)
        confirmation = compiled.confirmation()

        assert confirmation["we_still_need"], "an RSU plan needs its specifics"
        assert any(u["controls"].startswith("template:")
                   for u in confirmation["we_still_need"])
        # An RSU description with no specifics is incomplete, not contradictory:
        # the template will supply the flows, so the blocker is the missing
        # template input rather than an imagined absence of money.
        assert compiled.scenario.self_conflicts() == []
        assert compiled.can_simulate is True
        assert compiled.can_save is False

    def test_03_the_user_confirms_and_the_template_validates(self):
        assert RSU_TEMPLATE.validate(rsu_inputs()) == []
        assert compile_scenario(DESCRIPTION).defaults_ref.startswith(
            "compiler-defaults/")

    def test_04_replay_reports_both_returns_scope_and_benchmarks(self, prices):
        values = rsu_inputs()
        result = simulate(
            prices, flows=[], grants=grants_for(values, prices),
            program=disposition_program(values, prices.index),
            cash_policy=CashPolicy.idle(),
            modelling_scope=RSU_TEMPLATE.modelling_scope())
        payload = result.to_json()

        assert payload["time_weighted_annualized"] is not None
        assert payload["money_weighted_annualized"] is not None
        assert payload["modelling_scope"]["not_modelled"]
        assert "how did I do" in payload["return_basis_note"]

    def test_05_trial_and_selection_disclosures_are_visible(self, client, store,
                                                            prices):
        self._save(store)
        page = text(client.get("/workspace/plans/my-rsu").text)

        assert "How this plan was arrived at" in page
        assert "Selection basis" in page
        assert "Trials counted" in page
        assert "Hidden selection" in page
        assert "Reads as a recommendation" in page

    def test_06_the_plan_saves_privately(self, store, client):
        self._save(store)
        record = store.get_plan("my-rsu", "pilot")

        assert record is not None
        assert store.get_plan("my-rsu", "somebody-else") is None
        assert "/workspace" not in client.get("/ui/").text

    def test_07_a_blackout_delays_a_vest_sale(self, prices):
        values = rsu_inputs()
        result = simulate(
            prices, flows=[], grants=grants_for(values, prices),
            program=disposition_program(values, prices.index),
            cash_policy=CashPolicy.idle())

        august = [f for f in result.path.fills
                  if f.shares < 0 and f.date.month == 8 and f.date.year == 2021]
        assert not august, "a sale executed inside the declared blackout"

    def test_08_reconciliation_records_missing_and_unexpected(self, store, client):
        self._save(store)
        expected = (ExpectedEvent("2021-08-16", "vest", "100 shares",
                                  source="rsu schedule"),)
        observed = (ObservedEvent("2021-09-13", "vest", "100 shares"),)
        deviations = reconcile(expected, observed)

        assert {d.kind.value for d in deviations} == {"MISSING", "UNEXPECTED"}

        store.save_observation(owner="pilot", observation=PlanObservation(
            plan_id="my-rsu", observed_at="2021-09-30T00:00:00Z",
            data_snapshot="prices@2021-09-30", expected_events=expected,
            observed_events=observed, deviations=tuple(deviations)))
        page = text(client.get("/workspace/plans/my-rsu/observations").text)

        assert "missing" in page and "unexpected" in page
        assert "does not change" in page, "the plan must be stated as immutable"

    def test_09_a_proposal_expires_without_implying_execution(self, store, client):
        self._save(store)
        proposal = Proposal(
            proposal_id="p1", plan_id="my-rsu", generated_at="2021-08-17",
            generated_from="disposition: sell_all_and_diversify",
            reason="your plan sells vested shares and buys SPY",
            event="vest 2021-08-16", ticker="ACME", notional=-5000.0,
            expires="2021-09-10",
            eligibility=Eligibility.BLOCKED_BY_WINDOW,
            detail="inside a blackout window until 2021-09-10")
        [expired] = expire_overdue([proposal], as_of="2021-09-30")
        store.save_proposal(owner="pilot", proposal=expired)

        assert expired.status is ProposalStatus.EXPIRED
        assert expired.placed is False

        page = text(client.get("/workspace/plans/my-rsu/proposals").text)
        assert "expired" in page
        assert "NONE" in page
        assert "no execution capability" in page

    def test_10_the_counterfactual_isolates_the_constraint(self, client, store):
        self._save(store)
        common = dict(
            flow_schedule_hash="rsu-vests", starting_capital=0.0,
            cash_policy_rate=0.0, tax_treatment="NONE_APPLIED", cost_bps=10.0,
            period_start="2021-01-04", period_end="2023-01-01",
            allocation_rule_hash="rule-1", data_snapshot="prices@2023-01-01")
        verdict = classify_counterfactual(
            RunConditions(**common, execution_lag=1),
            RunConditions(**common, execution_lag=0),
            constraint="the blackout window")

        assert verdict.comparison_class is ComparisonClass.CONSTRAINT_EFFECT
        assert verdict.attribution_isolated
        assert verdict.isolates == "the blackout window"

        page = text(client.get("/workspace/plans/my-rsu/counterfactual").text)
        assert "Constraint isolated" in page
        assert "Held identical" in page

    def test_11_no_private_artifact_reaches_a_public_surface(self, client, store):
        self._save(store)
        for path in ("/ui/", "/ui/findings", "/ui/claims", "/ui/investigations",
                     "/current-strategies", "/surfaces", "/info"):
            body = client.get(path).text
            for marker in ("my-rsu", "mission/", "intent/", "proposal/",
                           "observation/", "ACME"):
                assert marker not in body, f"{path} exposes {marker}"

    # ---- helper ----------------------------------------------------------

    @staticmethod
    def _save(store):
        compiled = compile_scenario(
            "I buy $2000 of SPY on the first trading day of every month, "
            "reinvest the dividends, and never sell.",
            name="my-rsu", benchmark_rule="benchmark-policy/public-default@1")
        store.save_plan(plan_id="my-rsu", owner="pilot",
                        scenario=confirmed(compiled.scenario),
                        stated_text=compiled.scenario.provenance.stated
                        and "I buy $2000 of SPY on the first trading day of every "
                            "month, reinvest the dividends, and never sell.",
                        saved_at="2021-12-31T00:00:00Z")


class TestNothingInTheJourneyRecommends:
    def test_every_workspace_page_passes_the_language_scan(self, client, store):
        TestTheWholeJourney._save(store)
        for path in ("/workspace/", "/workspace/plans/my-rsu",
                     "/workspace/plans/my-rsu/proposals",
                     "/workspace/plans/my-rsu/observations",
                     "/workspace/plans/my-rsu/counterfactual"):
            found = {k: v for k, v in
                     scan_language(text(client.get(path).text)).items() if v}
            assert not found, f"{path} contains {found}"
