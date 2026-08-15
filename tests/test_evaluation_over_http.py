"""Step 6's conformance, run again through the wire.

The same six plan shapes, the same market delivery, the seven streams extracted
independently on both sides — and now with a serialization boundary in the
middle. If local and remote disagree, the extraction changed something.

**The transport must decide nothing.** That is the claim this file tests rather
than repeats: the HTTP path serializes, posts, checks a status and
deserializes. A route that substituted a default for a missing field, or read a
partial result as a whole one, would be taking an evaluation decision inside a
transport — where nobody looks for one.

**The mutation is the point of the exercise.** Two evaluators that always agree
prove nothing if the comparison could not notice a difference. So one remote
intermediate is changed — a fill quantity, a signal session, a metric status —
while the headline figure is left alone, and conformance must fail anyway.
Equal final numbers hiding different wiring is this codebase's most reliable
defect, and a boundary that only compares the number reproduces it across a
network.
"""
from __future__ import annotations

import os

import pytest
from runtime_contracts import Author, IntentField, VerifiedIntent

from src.discovery.canonical import canonicalise
from src.evaluation.contract import CONTRACT_VERSION, result_to_json
from src.evaluation.service import STREAMS
from src.evaluation.transport import (EvaluationUnreachable, HttpEvaluator,
                                      LocalEvaluator, idempotency_key)
from src.mission.evaluation_policy import declared_policy
from src.mission.from_intent import compile_intent
from src.mission.strategy_spec import from_scenario, to_scenario

ENGINE = "quantify-engine@http-conformance"
POLICY = declared_policy(data_policy="SYNTHETIC_ONLY", as_of="2026-08-15")

CASES = {
    "monthly-contributions": {"assets": "VTI", "amount": "1000",
                              "cadence": "monthly"},
    "annual-contributions": {"assets": "VTI", "amount": "1000",
                             "cadence": "annual"},
    "two-holdings": {"assets": "VTI,BND", "amount": "1000",
                     "cadence": "monthly"},
    "one-off": {"assets": "VTI", "amount": "10000", "cadence": "once"},
    "event-triggered": {"assets": "VOO", "amount": "1000",
                        "observed_assets": "SPY",
                        "trigger_semantics": "crossing_event",
                        "moving_average_window": "200"},
    "never-sells": {"assets": "VTI", "amount": "500", "cadence": "monthly",
                    "sell_action": "never sold any of it"},
}


@pytest.fixture(autouse=True)
def deployment(monkeypatch):
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)


def spec_for(stated):
    canonical = canonicalise(stated)
    intent = VerifiedIntent(
        objective="evaluate_investment_strategy", produced_by="http",
        utterance_ref="u",
        fields={name: IntentField(value=value, author=Author.MODEL)
                for name, (value, _a) in canonical.fields.items()},
        unresolved=()).seal()
    out = compile_intent(intent, benchmark_rule="a-rule")
    assert out.scenario is not None, [r.detail for r in out.refusals]
    return from_scenario(out.scenario)


def engine():
    from src.workspace.run_boundary import execute_compiled_plan, market_data_for

    return execute_compiled_plan, market_data_for


@pytest.fixture
def service():
    """The real app, over a real serialization, without a socket.

    An in-process ASGI client rather than a listening port: what is under test
    is the contract and the routing, and both are exercised in full. What it
    does not exercise is the network — named here rather than left implied,
    because a suite that quietly proved less than it looked like it proved is
    the failure this project keeps finding.
    """
    from fastapi.testclient import TestClient

    from src.evaluation.server import create_app

    run_plan, resolve_for = engine()
    app = create_app(
        resolve_access=lambda spec, _id: resolve_for(to_scenario(spec),
                                                     context="http-conformance"),
        run_plan=run_plan)
    with TestClient(app) as client:
        def post(url, json, headers):
            answer = client.post(url, json=json, headers=headers)
            try:
                return answer.status_code, answer.json()
            except ValueError:
                return answer.status_code, None
        yield client, HttpEvaluator(post=post, url="/evaluate")


def both(case, service):
    _client, remote = service
    run_plan, resolve_for = engine()
    spec = spec_for(CASES[case])
    access = resolve_for(to_scenario(spec), context="http-conformance")

    local = LocalEvaluator(run_plan=run_plan).evaluate(
        spec, "", evaluation_policy=POLICY, engine_version=ENGINE,
        access=access)
    across = remote.evaluate(spec, "", evaluation_policy=POLICY,
                             engine_version=ENGINE, access=access)
    return local, across


@pytest.mark.parametrize("case", sorted(CASES), ids=sorted(CASES))
class TestLocalAndRemoteAgree:
    def test_every_stream_survives_the_wire(self, case, service):
        local, across = both(case, service)
        differing = [name for name in STREAMS
                     if (local.streams[name].produced,
                         local.streams[name].rows)
                     != (across.streams[name].produced,
                         across.streams[name].rows)]
        assert differing == [], (
            f"{case}: {differing} differ across the transport, so the "
            "extraction changed something rather than moving it")

    def test_absent_survives_as_absent(self, case, service):
        """The field a careless reader drops.

        `produced=False` with no rows and `produced=True` with no rows are
        different runs. A transport that rebuilt both as an empty list would
        make the remote evaluator agree about a stage it never ran.
        """
        local, across = both(case, service)
        for name in STREAMS:
            assert local.streams[name].produced is across.streams[name].produced

    def test_the_identities_survive(self, case, service):
        local, across = both(case, service)
        assert local.identity == across.identity


class TestTheComparisonWouldNoticeAChange:
    """Two evaluators that always agree prove nothing unless disagreement is
    detectable. Each of these changes one intermediate and leaves the headline
    figure alone."""

    def mutated(self, case, service, change):
        """Apply the change, and check it changed something.

        The metric mutation first set `money_weighted` to `NO_ADMISSIBLE_RATE`
        — which is what it already was on this plan. The mutation was a no-op
        and the test failed, correctly: a mutation test that does not mutate is
        a green line proving nothing, and it would have sat there claiming the
        boundary catches status changes.
        """
        import copy

        from src.evaluation.contract import result_from_json

        local, _across = both(case, service)
        body = result_to_json(local)
        before = copy.deepcopy(body)
        change(body)
        assert body != before, (
            "the mutation left the result unchanged, so what follows would "
            "pass on a comparison that notices nothing")
        return local, result_from_json(body)

    def test_a_changed_fill_quantity_is_caught(self, service):
        def change(body):
            rows = body["streams"]["fills"]["rows"]
            rows[0] = {**rows[0], "units": str(float(rows[0]["units"]) + 1)}

        local, remote = self.mutated("monthly-contributions", service, change)
        assert local.streams["fills"].rows != remote.streams["fills"].rows
        assert local.streams["metrics"].rows == remote.streams["metrics"].rows, (
            "the mutation moved the headline too, so this proves nothing about "
            "comparing intermediates")

    def test_a_changed_signal_session_is_caught(self, service):
        def change(body):
            rows = body["streams"]["signals"]["rows"]
            rows[0] = {**rows[0], "at": "1999-01-01"}

        local, remote = self.mutated("event-triggered", service, change)
        assert local.streams["signals"].rows != remote.streams["signals"].rows
        assert local.streams["metrics"].rows == remote.streams["metrics"].rows

    def test_a_changed_metric_status_is_caught(self, service):
        def change(body):
            rows = body["streams"]["metrics"]["rows"]
            for row in rows:
                if row["metric"] == "money_weighted":
                    # To something it is not. Read the current value rather
                    # than naming one: this plan already reports
                    # NO_ADMISSIBLE_RATE, and asserting a change to the value
                    # it already holds is how the first version passed nothing.
                    row["status"] = ("COMPUTED" if row["status"] != "COMPUTED"
                                     else "NO_ADMISSIBLE_RATE")

        local, remote = self.mutated("monthly-contributions", service, change)
        assert local.streams["metrics"].rows != remote.streams["metrics"].rows

    def test_a_stage_turned_from_absent_to_empty_is_caught(self):
        """The subtlest one, and the reason `produced` is on the wire.

        A remote that reported "the signal stage ran and fired nothing" where
        the local one reports "this plan has no signal stage" is describing a
        different run. Both have no rows.
        """
        from src.evaluation.service import Stream

        absent = Stream("signals", False, (), "this plan is scheduled")
        empty = Stream("signals", True, ())
        assert absent.rows == empty.rows
        assert absent.digest != empty.digest, (
            "an absent stage and an empty one digest identically, so a "
            "conformance comparison could not tell them apart")


class TestTheTransportDecidesNothing:
    def test_a_refusal_arrives_as_a_refusal(self, service):
        """Not as a transport error. "This plan cannot be evaluated" is an
        answer about the plan; "the evaluator broke" is not, and a caller that
        confused them would retry a refusal forever."""
        from src.evaluation.service import EvaluationRefused

        _client, remote = service
        spec = spec_for(CASES["monthly-contributions"])
        with pytest.raises(EvaluationRefused):
            remote.evaluate(spec, "syn-not-this-one", evaluation_policy=POLICY,
                            engine_version=ENGINE)

    def test_an_unreachable_service_is_not_a_refusal(self):
        spec = spec_for(CASES["monthly-contributions"])
        remote = HttpEvaluator(post=lambda *_a: (503, None), url="/evaluate")
        with pytest.raises(EvaluationUnreachable):
            remote.evaluate(spec, "", evaluation_policy=POLICY,
                            engine_version=ENGINE)

    def test_a_contract_it_does_not_implement_is_refused(self, service):
        client, _remote = service
        answer = client.post("/evaluate", json={"contract_version": "other@9"})
        assert answer.status_code == 400, (
            "a version mismatch answered as anything else reads as a problem "
            "with the strategy")

    def test_the_idempotency_key_has_no_clock_in_it(self):
        spec = spec_for(CASES["monthly-contributions"])
        first = idempotency_key(spec, "syn-2026-08", ENGINE)
        second = idempotency_key(spec, "syn-2026-08", ENGINE)
        assert first == second, (
            "the same computation gets two keys, so a retried request is a "
            "second run")
        assert idempotency_key(spec, "syn-other", ENGINE) != first


class TestTheServiceCanSayWhatItIs:
    def test_health_answers(self, service):
        client, _remote = service
        assert client.get("/health").status_code == 200

    def test_version_names_the_contracts_and_the_vocabulary(self, service):
        client, _remote = service
        body = client.get("/version").json()
        assert body["contract_version"] == CONTRACT_VERSION
        assert body["result_schema_version"]
        assert body["conventions_version"].startswith("QuantLib"), (
            "the evaluator cannot say which QuantLib it computed under, so its "
            "convention names are unverifiable")
