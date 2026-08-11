"""The happy path: one natural sentence, from prompt to reconciled result.

Every other suite here probes an edge. This one exists because nine prompts can
all refuse correctly and prove nothing about whether the product works — a
suite of refusals reads as a working guard and an unusable product, and the two
are indistinguishable from the refusals alone.

The sentence is the one a person would type:

    I buy $1,000 of SPY whenever it crosses below its 200-day moving average,
    over the past five years.

It could not run. `SPY` was reserved away from holdings whenever it appeared
near signal language — correct for "notify me whenever SPY crosses below", and
in the sentence that both watches and buys it, the reservation consumed it
entirely. The plan held nothing, no funding policy was built, and the most
natural phrasing of the strategy was the one shape that could not execute.

Fixing that exposed a second defect only this prompt could reach. A daily
condition produces contributions on adjacent sessions, and the ledger paired
fills to "everything before the next contribution" — a window that is always
empty when cash lands on session N and fills on N+1, which is also the next
contribution's session. Nine of sixty rows bought nothing, and the
reconciliation refused the result rather than showing it.

Both premises are asserted below. Sparse fixtures cannot produce the second
defect, and every earlier test in this codebase used one.
"""
from __future__ import annotations

import pytest

from src.mission.compiler import (
    _acquired_instruments,
    _observed_in_signal,
    compile_scenario,
    parse,
)
from src.mission.spec import ScenarioAmendment

CONTROL = ("I buy $1,000 of SPY whenever it crosses below its 200-day moving "
           "average, over the past five years.")

#: The counterexamples the role split must preserve. Reserving an instrument
#: away from holdings exists for the third of these, and must keep working.
WATCHED_AND_BOUGHT = CONTROL
BOUGHT_ONE_WATCHED_ANOTHER = ("Buy VOO whenever SPY crosses below its 200-day "
                              "moving average.")
WATCHED_ONLY = ("Notify me whenever SPY crosses below its 200-day moving "
                "average.")

ANSWERS = (
    ScenarioAmendment(question_id="account_type", answer="TAXABLE",
                      recorded_at="t"),
    ScenarioAmendment(question_id="funding_source", answer="contribution",
                      recorded_at="t"),
)


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")


@pytest.fixture
def deployment():
    from src.deploy.context import bind, resolve, unbind

    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    try:
        yield
    finally:
        unbind()


@pytest.fixture
def executed(deployment):
    import src.workspace.routes as routes

    access = routes._market_data("control")
    plan = compile_scenario(
        CONTROL, name="control", version=1, amendments=ANSWERS,
        benchmark_rule="benchmark-policy/public-default@1",
        priceable=tuple(access.frame.columns))
    return plan.scenario, routes._run(plan.scenario, access,
                                      stated_text=CONTROL)


class TestTheRoleSplit:
    """An instrument may be watched and bought. Reserve it only when the
    sentence gives no action that acquires it."""

    def test_watched_and_bought_is_held(self):
        assert "SPY" in parse(WATCHED_AND_BOUGHT).assets

    def test_watched_and_bought_is_also_the_signal_subject(self):
        assert "SPY" in _observed_in_signal(WATCHED_AND_BOUGHT)
        assert "SPY" in _acquired_instruments(WATCHED_AND_BOUGHT)

    def test_a_different_instrument_may_be_watched(self):
        """The roles are separate, not merged. Buying VOO while watching SPY
        must hold VOO and watch SPY."""
        assert _acquired_instruments(BOUGHT_ONE_WATCHED_ANOTHER) == {"VOO"}
        assert _observed_in_signal(BOUGHT_ONE_WATCHED_ANOTHER) == {"SPY"}
        assert parse(BOUGHT_ONE_WATCHED_ANOTHER).assets == ("VOO",)

    def test_a_signal_only_instruction_holds_nothing(self):
        """The reason the reservation exists. Losing this would turn every
        watched index into a purchase nobody asked for."""
        assert not _acquired_instruments(WATCHED_ONLY)
        assert _observed_in_signal(WATCHED_ONLY) == {"SPY"}
        assert parse(WATCHED_ONLY).assets == ()


class TestThePremisesTheDefectsNeeded:
    """Both defects need a condition the fixture must actually produce."""

    def test_the_control_compiles_to_an_event_funded_plan(self, executed):
        scenario, _ = executed
        assert scenario.is_event_funded
        assert scenario.allocation_rule.assets == ("SPY",)
        assert scenario.funding.trigger.subject == "SPY"

    def test_nothing_is_left_open(self, executed):
        scenario, _ = executed
        assert [one.field for one in scenario.provenance.unresolved] == []

    def test_contributions_land_on_adjacent_sessions(self, executed):
        """The premise the join defect needed.

        Paired by date boundary, a contribution whose fill lands on the next
        contribution's session gets nothing. With events spaced weeks apart the
        broken pairing looks correct, and every earlier fixture here was
        spaced.
        """
        import pandas as pd

        _, run = executed
        sessions = sorted(pd.Timestamp(row.contribution_session)
                          for row in run["ledger"].rows)
        adjacent = sum(1 for a, b in zip(sessions, sessions[1:])
                       if (b - a).days <= 3)
        assert adjacent > 0, (
            "no two contributions are adjacent; this fixture cannot exercise "
            "the fill-pairing boundary")


class TestTheControlProducesAResult:
    def test_a_figure_exists(self, executed):
        _, run = executed
        assert run["unavailable"] is None
        assert run["result"] is not None

    def test_every_signal_became_a_purchase(self, executed):
        _, run = executed
        summary = run["ledger"].summary()
        assert summary["purchases_executed"] == summary["signals_detected"]
        assert summary["signals_not_executable"] == 0

    def test_every_row_bought_shares_at_a_price(self, executed):
        """Nine of sixty bought nothing under the boundary pairing."""
        _, run = executed
        empty = [row for row in run["ledger"].rows
                 if row.shares <= 0 or row.price <= 0]
        assert not empty, f"{len(empty)} rows bought nothing"

    def test_the_three_dates_are_ordered(self, executed):
        _, run = executed
        for row in run["ledger"].rows:
            assert row.signal_session < row.contribution_session
            assert row.contribution_session <= row.execution_session

    def test_the_ledger_reconciles(self, executed):
        _, run = executed
        assert run["reconciliation"].agrees, run["reconciliation"].failures()

    def test_coverage_is_complete(self, executed):
        _, run = executed
        assert run["coverage"].publishable
        assert not run["coverage"].blocking

    def test_the_total_is_the_amount_times_the_purchases(self, executed):
        from decimal import Decimal

        _, run = executed
        ledger = run["ledger"]
        assert ledger.total_contributed == Decimal("1000") * len(ledger.rows)

    def test_the_window_was_applied(self, executed):
        """Five years, not the whole snapshot. The period is part of the
        plan and a figure over ten years answers a different question."""
        import src.workspace.routes as routes

        scenario, run = executed
        access = routes._market_data("control")
        resolved = routes._resolve_window(scenario, access.frame)
        assert resolved is not None
        assert 4.9 <= (resolved.end - resolved.start).days / 365.25 <= 5.1


class TestTheWatchedSeriesIsTheOneNamed:
    """The observed role must be *consumed*, not merely derived.

    A mutation that dropped the observed role when it coincided with the
    acquired one survived every test: the role was computed, used once to
    decide a reservation, and never recorded or read again. Meanwhile a plan
    buying VOO on an SPY signal evaluated the condition on VOO — the fallback
    took the watched series to be whichever instrument the plan held.

    An ETF and the index it tracks do not cross their averages on the same
    days, so that is a different rule producing different purchases.
    """

    def compiled(self, text, deployment):
        import src.workspace.routes as routes

        access = routes._market_data("control")
        return compile_scenario(
            text, name="p", version=1, amendments=ANSWERS,
            benchmark_rule="benchmark-policy/public-default@1",
            priceable=tuple(access.frame.columns)).scenario

    def test_the_parse_records_what_is_watched(self, deployment):
        assert parse(BOUGHT_ONE_WATCHED_ANOTHER).observed == ("SPY",)

    def test_a_plan_watches_the_instrument_the_sentence_names(self, deployment):
        scenario = self.compiled(
            "Buy VOO with $1,000 whenever SPY crosses below its 200-day "
            "moving average, over the past five years.", deployment)
        assert scenario.allocation_rule.assets == ("VOO",)
        assert scenario.funding.trigger.subject == "SPY"

    def test_one_instrument_in_both_roles_still_works(self, deployment):
        scenario = self.compiled(CONTROL, deployment)
        assert scenario.allocation_rule.assets == ("SPY",)
        assert scenario.funding.trigger.subject == "SPY"


class TestTheControlJourneyThroughTheRoutes:
    """The same sentence, through the HTTP surface a person uses.

    Everything above runs the compiler and the engine directly. Two of the
    defects in this slice lived between those and the page — a candidate
    lookup that reversed a key back into a phrase, and a rebuilt parse that
    dropped the watched instrument — so a claim about the product has to be
    made where the product is.
    """

    @pytest.fixture
    def client(self, tmp_path, deployment, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api
        import src.workspace.routes as routes
        from src.workspace.store import WorkspaceStore

        class Counting:
            def __init__(self):
                self.calls = 0

            def complete(self, *, system, user):
                self.calls += 1
                raise TimeoutError("counted, not answered")

        counter = Counting()
        store = WorkspaceStore(tmp_path / "w.db")
        monkeypatch.setattr(routes, "_parser_client", lambda: counter)
        monkeypatch.setattr(routes, "_store", lambda: store)
        api._bootstrap()
        return TestClient(api.app), counter, store

    @staticmethod
    def token(body: str) -> str:
        import html as html_module
        import re

        found = re.search(r'name="parse" value="([^"]*)"', body)
        return html_module.unescape(found.group(1)) if found else ""

    def walk(self, http):
        """Submit the control and answer what it asks, as a browser would."""
        import html as html_module
        import re

        body = http.get("/workspace/new", params={"describe": CONTROL}).text
        payload = {"describe": CONTROL, "title": "control",
                   "parse": self.token(body)}
        for _ in range(4):
            selects = dict(re.findall(
                r'<select[^>]*name="(answer:[^"]+|confirm:[^"]+)"[^>]*>(.*?)</select>',
                body, re.S))
            # An inference with no alternative is confirmed by a checkbox, not
            # a select. Submitting only selects left `signal_series`
            # unconfirmed for ever, and a plan cannot save while an inference
            # is unconfirmed — which read exactly like a product that would not
            # complete.
            checkboxes = dict(re.findall(
                r'<input type="checkbox" name="(confirm:[^"]+)"\s+value="([^"]*)"',
                body))
            if not selects and not checkboxes:
                break
            for name, block in selects.items():
                options = [html_module.unescape(v) for v in
                           re.findall(r'<option value="([^"]*)"', block) if v]
                if options:
                    payload[name] = options[0]
            for name, value in checkboxes.items():
                payload[name] = html_module.unescape(value)
            response = http.post("/workspace/save", data=payload,
                                 follow_redirects=False)
            if response.status_code in (302, 303):
                return response.headers["location"].rsplit("/", 1)[-1], payload
            body = response.text
            payload["parse"] = self.token(body) or payload["parse"]
        return "", payload

    def test_the_journey_saves(self, client):
        http, _, _ = client
        plan_id, _ = self.walk(http)
        assert plan_id, (
            "the control did not reach a saved plan; unsupported prompts "
            "stopping safely says nothing about a supported one completing")

    def test_the_journey_costs_one_provider_call(self, client):
        http, counter, _ = client
        self.walk(http)
        assert counter.calls == 1, f"{counter.calls} provider calls"

    def test_no_exclusion_was_needed(self, client):
        """A plan reached by dismissing parts of itself is a different plan."""
        http, _, _ = client
        _, payload = self.walk(http)
        assert not [k for k in payload if k.startswith("exclude:")]

    def test_the_saved_plan_holds_and_watches_the_right_things(self, client):
        """Where the last two defects lived: the split has to survive being
        written down and read back, not merely computed once."""
        import src.workspace.routes as routes

        http, _, store = client
        plan_id, _ = self.walk(http)
        assert plan_id

        record = store.get_plan(plan_id, "pilot")
        body = record["scenario"]
        assert body["methodology"]["allocation_rule"]["assets"] == ["SPY"]

        funding = (body.get("flows") or {}).get("funding") or {}
        assert funding.get("kind") == "EVENT_TRIGGERED"
        assert (funding.get("trigger") or {}).get("subject") == "SPY"

    def test_reopening_makes_no_further_provider_call(self, client):
        """A saved plan is replayed, never re-read. A model call here would
        re-describe an old plan with today's configuration."""
        http, counter, _ = client
        plan_id, _ = self.walk(http)
        before = counter.calls
        assert http.get(f"/workspace/plans/{plan_id}").status_code == 200
        assert counter.calls == before

    def test_the_reopened_page_shows_the_execution_evidence(self, client):
        http, _, _ = client
        plan_id, _ = self.walk(http)
        body = http.get(f"/workspace/plans/{plan_id}").text
        assert "Signals detected" in body
        assert "Purchases executed" in body
        assert "Signal date" in body and "Execution date" in body
