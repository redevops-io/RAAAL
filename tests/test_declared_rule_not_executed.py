"""A declared rule that never ran must not produce a figure.

Found by a pilot user, not by a test. They described

    I buy $1,000 of SP500 ETF every time the S&P 500 crosses below its
    200-day moving average for the past 5 years.

and were shown $5,160 and +18.09% — identical to the buy-and-hold benchmark to
the penny, beside a disclosure reading "every dimension outside the investment
rule was held identical, so a difference between these figures is attributable
to the rule."

There was no difference and no rule. `_run` called
`simulate(..., program=buy_and_hold(tradeable))` whatever the scenario
declared, and nothing converted `event_program` into an `EventProgram`. The
mechanism existed — `simulate` takes a `program` argument and the engine can
run one — and the live path did not reach it. Fifth instance of that shape in
this codebase and the first to move money.

`test_the_original_prompt.py` covered this exact sentence and passed
throughout. It asserted parsing, asset identity and the time window: whether we
*understood* the user, never whether we *did* what we understood.
"""
from __future__ import annotations

import pytest

from src.mission.comparability import RunConditions, classify
from src.mission.spec import ScenarioAmendment

DESCRIPTION = ("I buy $1,000 of SP500 ETF every time the S&P 500 crosses "
               "below its 200-day moving average for the past 5 years.")

#: Settling the trigger is what builds the program. Without this the compiler
#: leaves `trigger_semantics` unresolved and `event_program` empty — so a test
#: using the bare description would exercise none of this and pass against
#: every mutation below. The same missing-producer trap that hid the telemetry
#: leak: the input class did not exist in the environment.
SETTLED = (ScenarioAmendment(question_id="trigger_semantics",
                             answer="crossing_event",
                             recorded_at="2026-08-05T00:00:00Z"),)


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


def compiled(amendments=SETTLED, description=DESCRIPTION):
    from src.mission.compiler import compile_scenario

    import src.workspace.routes as routes

    access = routes._market_data("test")
    return compile_scenario(
        description, name="p", version=1, amendments=amendments,
        benchmark_rule="benchmark-policy/public-default@1",
        priceable=tuple(access.frame.columns)), access


class TestThePremise:
    """Witness that the dangerous input exists before asserting about it."""

    def test_the_description_compiles_to_a_rule(self, deployment):
        plan, _ = compiled()
        assert plan.scenario.event_program, (
            "this description produced no event program, so every assertion "
            "below would hold against an engine with no guard at all")

    def test_without_the_trigger_answer_there_is_no_rule(self, deployment):
        """Why the fixture settles the trigger. Stated so that a future edit
        removing the amendment fails here rather than silently emptying the
        whole file."""
        plan, _ = compiled(amendments=())
        assert not plan.scenario.event_program


class TestNoFigureIsProduced:
    def test_the_run_carries_no_result(self, deployment):
        import src.workspace.routes as routes

        plan, access = compiled()
        run = routes._run(plan.scenario, access)
        assert run["result"] is None
        assert run["strategy_not_executed"] is True

    def test_no_benchmark_comparison_either(self, deployment):
        """The comparison is the more persuasive artifact. A table showing the
        plan beating bonds is read as a claim about the rule even with the
        headline figure removed."""
        import src.workspace.routes as routes

        plan, access = compiled()
        run = routes._run(plan.scenario, access)
        assert run["benchmarks"] == []
        assert run["comparability"] is None

    def test_the_refusal_says_the_rule_was_not_executed(self, deployment):
        import src.workspace.routes as routes

        plan, access = compiled()
        run = routes._run(plan.scenario, access)
        assert "did not execute that rule" in run["unavailable"]
        assert "remain saved" in run["unavailable"]

    def test_it_refuses_before_the_price_gap(self, deployment):
        """Ordering, not politeness.

        Reported as a data gap, the user corrects the instrument, the gap
        closes and the reward for fixing it is a figure for a rule that still
        never ran. This description resolves to `ETF`, which cannot be priced,
        so it exercises the case where both refusals apply.
        """
        import src.workspace.routes as routes

        plan, access = compiled()
        assert not [a for a in plan.scenario.allocation_rule.assets
                    if a in access.frame.columns], (
            "this description became priceable; the two-refusal case is no "
            "longer exercised and this test proves nothing about ordering")
        run = routes._run(plan.scenario, access)
        assert "No price history" not in run["unavailable"]


class TestTheScopeDeclaresIt:
    def test_the_event_program_reaches_the_modelling_scope(self, deployment):
        import src.workspace.routes as routes

        plan, _ = compiled()
        scope = routes.declare_unsimulated(plan.scenario, {})
        assert "event_program" in scope["declared_but_not_simulated"]

    def test_a_plan_without_a_rule_declares_nothing_extra(self, deployment):
        """The disclosure must discriminate. Naming every plan would make the
        NOT MODELLED column noise, and noise is the first step to ignored."""
        import src.workspace.routes as routes

        plan, _ = compiled(amendments=(),
                           description="I buy $500 of VOO every month.")
        scope = routes.declare_unsimulated(plan.scenario, {})
        assert "event_program" not in (
            scope.get("declared_but_not_simulated") or {})

    def test_every_unsimulated_entry_has_a_reason(self):
        from src.mission.scenario import UNSIMULATED

        assert all(UNSIMULATED.values())


BASE = dict(flow_schedule_hash="f", starting_capital=0.0, cash_policy_rate=0.0,
            tax_treatment="TAXABLE", cost_bps=10.0, execution_lag=1,
            period_start="2020-01-01", period_end="2025-01-01",
            allocation_rule_hash="r", account_hash="a", calendar_hash="c",
            market_data_hash="m", data_snapshot="s")


class TestTheStrategyEffectClaimIsRefused:
    """Independently of the suppression above.

    Every dimension can match, every one be checked, and the claim still be
    false — because "the difference is attributable to the rule" presumes a
    rule ran. Classifier @2 had nothing to object to: it only ever compared
    conditions to each other, never a run to its own declaration.
    """

    def test_identical_conditions_still_claim_strategy_effect(self):
        """The premise. Without this the next test passes for the wrong
        reason — against a classifier that never claims isolation at all."""
        verdict = classify(RunConditions(**BASE), RunConditions(**BASE))
        assert verdict.disclosure_key == "STRATEGY_EFFECT"
        assert verdict.attribution_isolated

    def test_an_unexecuted_rule_refuses_the_claim(self):
        verdict = classify(
            RunConditions(**BASE, declared_rule_executed=False),
            RunConditions(**BASE, declared_rule_executed=False))
        assert not verdict.attribution_isolated
        assert verdict.disclosure_key != "STRATEGY_EFFECT"

    def test_one_broken_side_is_enough(self):
        """Two identically-broken runs match on every dimension. Requiring
        both sides to be sound is what a dimension check cannot express."""
        verdict = classify(RunConditions(**BASE),
                           RunConditions(**BASE, declared_rule_executed=False))
        assert not verdict.attribution_isolated

    def test_the_reason_names_the_rule_not_a_missing_dimension(self):
        """Two different causes of lost attribution, and a reader has to be
        able to tell them apart. This one also crashed the sentence: the
        unchecked-dimension text joined an empty list."""
        verdict = classify(
            RunConditions(**BASE, declared_rule_executed=False),
            RunConditions(**BASE, declared_rule_executed=False))
        assert "not executed" in verdict.detail
        assert "never evaluated" not in verdict.detail

    def test_the_classifier_version_moved(self):
        """Changing what a verdict claims is a change to the claim, and old
        verdicts must keep meaning what they meant."""
        from src.mission.comparability import CLASSIFIER_VERSION

        assert CLASSIFIER_VERSION == "comparability/classifier@3"


class TestTheAffectedInventoryIsDerived:
    """Which runs were wrong is a property of the artifacts, not a list.

    The affected plan was found by a user opening one page. Naming it in a
    sweep would fix the instance and leave the class — and there is no way to
    know from a page how many other plans carried the same defect.
    """

    @pytest.fixture
    def store(self, tmp_path, deployment):
        from src.workspace.store import WorkspaceStore

        return WorkspaceStore(tmp_path / "w.db")

    def saved(self, store, *, plan_id, amendments, description=DESCRIPTION,
              result=None):
        """A plan with a stored run, as the defective engine produced them."""
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance

        plan, _ = compiled(amendments=amendments, description=description)
        source = plan.scenario.provenance
        scenario = ScenarioSpecification(**{
            **plan.scenario.__dict__,
            "provenance": Provenance(
                stated=source.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in source.inferred),
                contradictions=source.contradictions, unresolved=())})
        store.save_plan(plan_id=plan_id, owner="pilot", scenario=scenario,
                        stated_text=description, saved_at="2026-08-05T00:00:00Z")
        store.record_run(
            run_id=f"run-{plan_id}", plan_id=plan_id, owner="pilot",
            ran_at="2026-08-05T00:00:00Z",
            result=result or {"modelling_scope": {"excludes": []},
                              "market_data": {"status": "NOT_APPLICABLE"},
                              "final_value": 5160.0},
            comparison={})
        return plan_id

    def test_a_run_for_a_declared_rule_is_affected(self, store):
        from src.workspace.invalidate import affected

        self.saved(store, plan_id="plan-rule", amendments=SETTLED)
        assert [r["plan_id"] for r in affected(store, "pilot")] == ["plan-rule"]

    def test_an_ordinary_plan_is_not(self, store):
        """Discrimination. A sweep that invalidated everything would satisfy a
        test counting invalidations and destroy every correct result."""
        from src.workspace.invalidate import affected

        self.saved(store, plan_id="plan-plain", amendments=(),
                   description="I buy $500 of VOO every month.")
        assert affected(store, "pilot") == []

    def test_a_run_that_executed_the_rule_is_not(self, store):
        """Forward compatibility. Once the engine executes the program and
        records `rule_events`, a correct run must stop matching without this
        sweep being edited — otherwise the fix ships and the sweep starts
        invalidating good results."""
        from src.workspace.invalidate import affected

        self.saved(store, plan_id="plan-ran", amendments=SETTLED,
                   result={"modelling_scope": {"excludes": []},
                           "market_data": {"status": "NOT_APPLICABLE"},
                           "rule_events": 7})
        assert affected(store, "pilot") == []

    def test_a_zero_event_count_is_not_the_same_as_no_field(self, store):
        """A result recorded before the field existed reports nothing, not
        zero. Reading absence as zero would leave every affected run looking
        like one that legitimately found no crossings."""
        from src.workspace.invalidate import executed_rule_events

        assert executed_rule_events({"final_value": 1.0}) is None
        assert executed_rule_events({"rule_events": 0}) == 0

    def test_invalidation_is_persisted_and_read_back(self, store):
        from src.workspace.invalidate import CLASSIFICATION, ENGINE_VERSION, REASON

        plan_id = self.saved(store, plan_id="plan-rule", amendments=SETTLED)
        store.invalidate_run(run_id=f"run-{plan_id}", plan_id=plan_id,
                             owner="pilot", classification=CLASSIFICATION,
                             reason=REASON, engine_version=ENGINE_VERSION,
                             at="2026-08-05T12:00:00Z")
        runs = store.runs_for(plan_id, "pilot")
        assert runs[0]["invalidation"]["classification"] == "RULE_NOT_EXECUTED"

    def test_the_run_itself_survives(self, store):
        """The record of what was shown is the evidence that it was shown."""
        plan_id = self.saved(store, plan_id="plan-rule", amendments=SETTLED)
        store.invalidate_run(run_id=f"run-{plan_id}", plan_id=plan_id,
                             owner="pilot", classification="RULE_NOT_EXECUTED",
                             reason="r", engine_version="e",
                             at="2026-08-05T12:00:00Z")
        runs = store.runs_for(plan_id, "pilot")
        assert len(runs) == 1
        assert runs[0]["result"]["final_value"] == 5160.0

    def test_an_uninvalidated_run_reads_as_valid(self, store):
        plan_id = self.saved(store, plan_id="plan-rule", amendments=SETTLED)
        assert store.runs_for(plan_id, "pilot")[0]["invalidation"] is None

    def test_one_tenant_cannot_see_another_s_invalidations(self, store):
        plan_id = self.saved(store, plan_id="plan-rule", amendments=SETTLED)
        store.invalidate_run(run_id=f"run-{plan_id}", plan_id=plan_id,
                             owner="pilot", classification="RULE_NOT_EXECUTED",
                             reason="r", engine_version="e",
                             at="2026-08-05T12:00:00Z")
        assert store.invalidations_for(plan_id, "someone-else") == {}


class TestThePageShowsNoFigure:
    """The promise lands in the template or it does not land.

    Everything above proves `_run` returns no result. A page that rendered the
    stored run instead, or kept the comparison table under the warning, would
    satisfy all of it and still show the user $5,160.
    """

    @pytest.fixture
    def client(self, tmp_path, deployment, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api
        import src.workspace.routes as routes
        from src.workspace.store import WorkspaceStore

        store = WorkspaceStore(tmp_path / "w.db")
        monkeypatch.setattr(routes, "_store", lambda: store)
        api._bootstrap()
        return TestClient(api.app), store

    def stored(self, store, plan_id="plan-rule"):
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance

        plan, _ = compiled()
        source = plan.scenario.provenance
        scenario = ScenarioSpecification(**{
            **plan.scenario.__dict__,
            "provenance": Provenance(
                stated=source.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in source.inferred),
                contradictions=source.contradictions, unresolved=())})
        store.save_plan(plan_id=plan_id, owner="pilot", scenario=scenario,
                        stated_text=DESCRIPTION, saved_at="2026-08-05T00:00:00Z",
                        title="SPX 200DMA")
        store.record_run(
            run_id=f"run-{plan_id}", plan_id=plan_id, owner="pilot",
            ran_at="2026-08-05T00:00:00Z",
            result={"modelling_scope": {"excludes": []},
                    "market_data": {"status": "NOT_APPLICABLE"},
                    "final_value": 5160.0},
            comparison={})
        return plan_id

    def page(self, client, plan_id):
        http, _ = client
        response = http.get(f"/workspace/plans/{plan_id}")
        assert response.status_code == 200, response.text
        return response.text

    def test_the_notice_is_shown(self, client):
        _, store = client
        body = self.page(client, self.stored(store))
        assert "This result is unavailable" in body

    def test_no_result_figure_appears(self, client):
        """`$5,160` is what a user remembers. The caveat is not.

        Asserted on the result rows rather than on any dollar sign: the money
        path the user declared — "once · $1,000" — is their own statement and
        belongs on the page. A test forbidding every figure would have to be
        loosened the first time it was right.
        """
        _, store = client
        body = self.page(client, self.stored(store))
        for row in ("Final value", "Money-weighted return",
                    "Time-weighted return", "5,160"):
            assert row not in body, f"the page still shows {row!r}"

    def test_no_return_percentage_appears(self, client):
        import re

        _, store = client
        body = self.page(client, self.stored(store))
        assert not re.search(r"[+-][0-9]+\.[0-9]{2}%", body)

    def test_the_benchmark_table_is_gone(self, client):
        _, store = client
        body = self.page(client, self.stored(store))
        assert "Compared with" not in body

    def test_the_scope_names_the_unexecuted_rule(self, client):
        """Under "Not modelled", where a reader looks — not only in a JSON key.

        `declare_unsimulated` wrote `declared_but_not_simulated` and
        `_scope.html` renders `scope.not_modelled`, so this disclosure had
        never once appeared on a page. Both columns were empty on the plan
        that prompted this work.
        """
        from src.mission.scenario import UNSIMULATED

        _, store = client
        body = self.page(client, self.stored(store))
        assert "Not modelled" in body

        # A fragment only the scope panel can carry. The first version of this
        # asserted "did not execute" — which is in the unavailable notice at
        # the top of the same page, so it passed with the scope rendering
        # deleted entirely. An assertion satisfied by a different element is
        # not an assertion about this one.
        reason = UNSIMULATED["event_program"]
        assert len(reason) > 40, "the reason text is empty; this proves nothing"
        assert "hashed into the methodology" in reason
        assert "hashed into the methodology" in body, (
            "the unsimulated declaration is not rendered in the scope panel")

    def test_an_invalidated_run_is_disclosed(self, client):
        _, store = client
        plan_id = self.stored(store)
        store.invalidate_run(run_id=f"run-{plan_id}", plan_id=plan_id,
                             owner="pilot", classification="RULE_NOT_EXECUTED",
                             reason="the engine replayed buy-and-hold",
                             engine_version="engine/buy-and-hold-only@1",
                             at="2026-08-05T12:00:00Z")
        body = self.page(client, plan_id)
        assert "Withdrawn results" in body
        assert "RULE_NOT_EXECUTED" in body

    def test_a_plan_with_no_invalidation_shows_no_such_section(self, client):
        """Or the section becomes furniture, present on every page and read on
        none."""
        _, store = client
        body = self.page(client, self.stored(store))
        assert "Withdrawn results" not in body


class TestAWithdrawalIsNotRewritten:
    """The table is classified IMMUTABLE_ARTIFACT. This is what makes that
    true rather than declared.

    A second sweep must not move `invalidated_at` to today, which would erase
    when users were first told, and must not let a reason be softened on a
    later pass.
    """

    @pytest.fixture
    def store(self, tmp_path, deployment):
        from src.workspace.store import WorkspaceStore

        return WorkspaceStore(tmp_path / "w.db")

    def seed(self, store):
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance

        plan, _ = compiled()
        source = plan.scenario.provenance
        scenario = ScenarioSpecification(**{
            **plan.scenario.__dict__,
            "provenance": Provenance(
                stated=source.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in source.inferred),
                contradictions=source.contradictions, unresolved=())})
        store.save_plan(plan_id="p", owner="pilot", scenario=scenario,
                        stated_text=DESCRIPTION, saved_at="2026-08-05T00:00:00Z")
        store.record_run(run_id="r", plan_id="p", owner="pilot",
                         ran_at="2026-08-05T00:00:00Z",
                         result={"modelling_scope": {"excludes": []},
                                 "market_data": {"status": "NOT_APPLICABLE"}},
                         comparison={})

    def withdraw(self, store, *, reason, at):
        return store.invalidate_run(
            run_id="r", plan_id="p", owner="pilot",
            classification="RULE_NOT_EXECUTED", reason=reason,
            engine_version="engine/buy-and-hold-only@1", at=at)

    def test_the_first_write_reports_that_it_wrote(self, store):
        self.seed(store)
        assert self.withdraw(store, reason="first", at="2026-08-05T12:00:00Z")

    def test_a_second_sweep_writes_nothing(self, store):
        self.seed(store)
        self.withdraw(store, reason="first", at="2026-08-05T12:00:00Z")
        assert not self.withdraw(store, reason="softened",
                                 at="2026-09-01T12:00:00Z")

    def test_the_original_date_and_reason_survive(self, store):
        """The assertion that matters. A no-op return value with a rewritten
        row would satisfy the test above and lose the record."""
        self.seed(store)
        self.withdraw(store, reason="first", at="2026-08-05T12:00:00Z")
        self.withdraw(store, reason="softened", at="2026-09-01T12:00:00Z")
        stored = store.invalidations_for("p", "pilot")["r"]
        assert stored["invalidated_at"] == "2026-08-05T12:00:00Z"
        assert stored["reason"] == "first"


class TestAWithdrawnRunDoesNotEscapeThroughExport:
    """A figure must not regain authority by leaving the interface.

    `run_invalidation` reaches the export on its own because the retention
    inventory classifies it. That is not enough: a consumer reading `plan_run`
    would see $5,160 with nothing on it and would have to know to join a second
    table before believing it.
    """

    @pytest.fixture
    def store(self, tmp_path, deployment):
        from src.workspace.store import WorkspaceStore

        return WorkspaceStore(tmp_path / "w.db")

    def seed(self, store, *, withdraw):
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance

        plan, _ = compiled()
        source = plan.scenario.provenance
        scenario = ScenarioSpecification(**{
            **plan.scenario.__dict__,
            "provenance": Provenance(
                stated=source.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in source.inferred),
                contradictions=source.contradictions, unresolved=())})
        store.save_plan(plan_id="p", owner="pilot", scenario=scenario,
                        stated_text=DESCRIPTION, saved_at="2026-08-05T00:00:00Z")
        store.record_run(run_id="r", plan_id="p", owner="pilot",
                         ran_at="2026-08-05T00:00:00Z",
                         result={"modelling_scope": {"excludes": []},
                                 "market_data": {"status": "NOT_APPLICABLE"},
                                 "final_value": 5160.0},
                         comparison={})
        if withdraw:
            store.invalidate_run(
                run_id="r", plan_id="p", owner="pilot",
                classification="RULE_NOT_EXECUTED", reason="the engine "
                "replayed buy-and-hold", engine_version="engine/bh@1",
                at="2026-08-05T12:00:00Z")

    def exported(self, store):
        from src.workspace.erasure import export_workspace

        return export_workspace(store, "pilot")

    def test_the_withdrawal_is_in_the_export_at_all(self, store):
        self.seed(store, withdraw=True)
        assert self.exported(store)["tables"]["run_invalidation"]

    def test_the_run_row_itself_carries_the_withdrawal(self, store):
        """Not only the separate table. The figure and the fact that it was
        withdrawn must travel together."""
        self.seed(store, withdraw=True)
        run = self.exported(store)["tables"]["plan_run"][0]
        assert run["invalidation"]["classification"] == "RULE_NOT_EXECUTED"

    def test_the_reason_and_time_are_the_original_ones(self, store):
        self.seed(store, withdraw=True)
        run = self.exported(store)["tables"]["plan_run"][0]
        assert run["invalidation"]["invalidated_at"] == "2026-08-05T12:00:00Z"
        assert "buy-and-hold" in run["invalidation"]["reason"]

    def test_a_sound_run_beside_a_withdrawn_one_is_not_marked(self, store):
        """Discrimination. Marking every row would make the flag furniture and
        tell a reader nothing about which figure to distrust.

        Both kinds must be present. Seeded with no withdrawal at all, the
        marking function returns early and a version that marked every row
        would pass — which it did, until this case was widened.
        """
        self.seed(store, withdraw=True)
        store.record_run(run_id="r2", plan_id="p", owner="pilot",
                         ran_at="2026-08-06T00:00:00Z",
                         result={"modelling_scope": {"excludes": []},
                                 "market_data": {"status": "NOT_APPLICABLE"},
                                 "final_value": 100.0},
                         comparison={})
        rows = {row["run_id"]: row
                for row in self.exported(store)["tables"]["plan_run"]}
        assert "invalidation" in rows["r"]
        assert "invalidation" not in rows["r2"]

    def test_the_user_still_receives_the_figure(self, store):
        """Marked, not removed. The account holds the record that they were
        shown a number, and they are entitled to it."""
        import json

        self.seed(store, withdraw=True)
        run = self.exported(store)["tables"]["plan_run"][0]
        stored = run["result"]
        if isinstance(stored, str):        # exported as stored, not re-parsed
            stored = json.loads(stored)
        assert stored["final_value"] == 5160.0


class TestTheSweepSeesAnExecutedRule:
    """Forward compatibility, against where the producer actually writes it.

    `rule_events` lives in the result's modelling scope — beside the other
    statements about what the figure accounts for. A sweep reading only the top
    level would report every correct run as affected the day the producer moved
    it, and would withdraw good results.
    """

    def test_the_count_is_found_in_the_modelling_scope(self):
        from src.workspace.invalidate import executed_rule_events

        assert executed_rule_events(
            {"modelling_scope": {"rule_events": 30}}) == 30

    def test_a_top_level_count_still_works(self):
        from src.workspace.invalidate import executed_rule_events

        assert executed_rule_events({"rule_events": 7}) == 7

    def test_absence_is_still_unknown_not_zero(self):
        from src.workspace.invalidate import executed_rule_events

        assert executed_rule_events({"modelling_scope": {}}) is None
        assert executed_rule_events({"final_value": 1.0}) is None

    def test_a_real_executed_run_is_not_swept(self, tmp_path, deployment):
        """End to end: a run the new engine produced must not be withdrawn by
        the sweep written for the old one."""
        from src.mission.compiler import compile_scenario
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance
        from src.workspace.invalidate import affected
        from src.workspace.store import WorkspaceStore

        import src.workspace.routes as routes

        store = WorkspaceStore(tmp_path / "w.db")
        access = routes._market_data("test")
        description = ("I buy $1,000 of VOO every time the S&P 500 crosses "
                       "below its 200-day moving average for the past 5 years.")
        plan = compile_scenario(
            description, name="p", version=1, amendments=SETTLED,
            benchmark_rule="benchmark-policy/public-default@1",
            priceable=tuple(access.frame.columns))
        source = plan.scenario.provenance
        scenario = ScenarioSpecification(**{
            **plan.scenario.__dict__,
            "provenance": Provenance(
                stated=source.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in source.inferred),
                contradictions=source.contradictions, unresolved=())})
        run = routes._run(scenario, access)
        assert run["result"] is not None, "premise: this plan must execute"

        store.save_plan(plan_id="p", owner="pilot", scenario=scenario,
                        stated_text=description, saved_at="2026-08-06T00:00:00Z")
        store.record_run(run_id="r", plan_id="p", owner="pilot",
                         ran_at="2026-08-06T00:00:00Z",
                         result=run["result"].to_json(), comparison={})
        assert affected(store, "pilot") == []
