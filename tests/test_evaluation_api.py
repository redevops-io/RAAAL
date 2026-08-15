"""The evaluation interface, before it is a service.

Step 4. The point of building it in-process is that the address changes later
and the contract does not — and that Step 6 has something to compare against.

Two things are worth more than the interface itself.

**A stream that does not exist is not an empty stream.** A scheduled plan
computes no signal; reporting `()` would claim the stage ran and found nothing.
A conformance comparison that could not tell those apart would agree on two
runs that did different things, which is the failure the streams exist to
prevent.

**The result says what it was computed from.** Six identities, each of which has
made a figure unreproducible here at some point: the specification, the
snapshot, the evaluator, the engine, the conventions, and the shape of the
result.
"""
from __future__ import annotations

import os

import pytest
from runtime_contracts import Author, IntentField, VerifiedIntent

from src.discovery.canonical import canonicalise
from src.evaluation.service import (EVALUATOR, RESULT_SCHEMA_VERSION, STREAMS,
                                    EvaluationRefused, evaluate)
from src.mission.from_intent import compile_intent
from src.mission.strategy_spec import from_scenario

ENGINE = "quantify-engine@test"
from src.mission.evaluation_policy import declared_policy

POLICY = declared_policy(data_policy="SYNTHETIC_ONLY",
                         as_of="2026-08-15")

SCHEDULED = {"assets": "VTI", "amount": "1000", "cadence": "monthly"}
TRIGGERED = {"assets": "VOO", "amount": "1000", "observed_assets": "SPY",
             "trigger_semantics": "crossing_event",
             "moving_average_window": "200"}


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


def spec_for(**stated):
    canonical = canonicalise(stated)
    intent = VerifiedIntent(
        objective="evaluate_investment_strategy", produced_by="test",
        utterance_ref="u",
        fields={name: IntentField(value=value, author=Author.MODEL)
                for name, (value, _a) in canonical.fields.items()},
        unresolved=()).seal()
    out = compile_intent(intent, benchmark_rule="a-rule")
    assert out.scenario is not None, [r.detail for r in out.refusals]
    return from_scenario(out.scenario)


def runner():
    """The engine and the resolver, supplied by the caller.

    `evaluate` fetches neither. An evaluator that resolved its own market data
    would download prices inside the calculation, and one that imported the
    application's runner could not be deployed apart from it — the boundary
    test caught the first version doing both.
    """
    from src.workspace.run_boundary import execute_compiled_plan, market_data_for

    return execute_compiled_plan, market_data_for


def evaluated(**stated):
    run_plan, resolve_for = runner()
    spec = spec_for(**stated)
    from src.mission.strategy_spec import to_scenario

    access = resolve_for(to_scenario(spec), context="evaluate-test")
    return evaluate(spec, "", evaluation_policy=POLICY,
                    engine_version=ENGINE, access=access, run_plan=run_plan)


class TestTheResultSaysWhatItWasComputedFrom:
    def test_every_identity_is_present_and_not_blank(self):
        identity = evaluated(**SCHEDULED).identity
        blank = [name for name, value in identity.items()
                 if name != "market_snapshot_id" and not value]
        assert blank == [], f"{blank} carry no value, so the chain breaks there"

    def test_the_strategy_hash_is_the_specifications_own(self):
        spec = spec_for(**SCHEDULED)
        run_plan, resolve_for = runner()
        from src.mission.strategy_spec import to_scenario
        out = evaluate(spec, "", evaluation_policy=POLICY,
                       engine_version=ENGINE, run_plan=run_plan,
                       access=resolve_for(to_scenario(spec), context="t"))
        assert out.strategy_hash == spec.spec_hash

    def test_the_snapshot_hash_is_the_delivered_frames_digest(self):
        out = evaluated(**SCHEDULED)
        assert out.market_snapshot_hash.startswith("mdf1:"), (
            "the result cites no frame digest, so it names a source rather "
            "than a delivery and proves nothing about what was consumed")

    def test_the_versions_are_the_ones_it_was_given(self):
        out = evaluated(**SCHEDULED)
        assert out.engine_version == ENGINE
        assert out.evaluation_policy == POLICY.to_json()
        assert out.evaluator == EVALUATOR
        assert out.result_schema_version == RESULT_SCHEMA_VERSION
        assert out.conventions_version.startswith("QuantLib")


class TestTheStreamsAreNamedAndSeparate:
    def test_every_declared_stream_is_reported(self):
        out = evaluated(**SCHEDULED)
        assert set(out.streams) == set(STREAMS)

    def test_a_scheduled_plan_buys_and_says_so(self):
        out = evaluated(**SCHEDULED)
        for name in ("eligible_sessions", "contributions", "fills",
                     "cash_flows", "orders", "metrics"):
            stream = out.streams[name]
            assert stream.produced, f"{name} was not produced"
            assert stream.rows, f"{name} produced nothing on a plan that runs"

    def test_a_scheduled_plan_has_no_signals_and_that_is_not_an_empty_list(self):
        """The distinction the whole design turns on.

        `produced=False` says this plan has no such stage. `produced=True` with
        no rows would say the stage ran and fired nothing — a different run,
        and one a conformance comparison must not confuse with this.
        """
        signals = evaluated(**SCHEDULED).streams["signals"]
        assert signals.produced is False
        assert signals.rows == ()
        assert "scheduled" in signals.absent_because

    def test_a_triggered_plan_does_have_signals(self):
        signals = evaluated(**TRIGGERED).streams["signals"]
        assert signals.produced is True

    def test_each_stream_digests_separately(self):
        """One bundled hash says two runs differ and not where.

        Equal headline numbers hiding different wiring is not a hypothesis
        here: a frame digest matched while a plan invested nothing, and two
        plans shared a hash while being priced on different price series.
        """
        monthly = evaluated(**SCHEDULED)
        annual = evaluated(**{**SCHEDULED, "cadence": "annual"})

        differing = [name for name in STREAMS
                     if monthly.streams[name].digest != annual.streams[name].digest]
        assert "contributions" in differing, (
            "a monthly plan and an annual one contribute identically")
        assert "eligible_sessions" not in differing, (
            "changing the cadence changed which sessions the plan is eligible "
            "for, which would mean the streams are not independent")


class TestItRefusesRatherThanAttributing:
    def test_a_snapshot_it_was_not_given_ends_the_call(self):
        """Evaluating against whatever came back is how a figure acquires
        provenance it does not have."""
        with pytest.raises(EvaluationRefused) as refused:
            run_plan, resolve_for = runner()
            from src.mission.strategy_spec import to_scenario
            spec = spec_for(**SCHEDULED)
            evaluate(spec, "syn-not-this-one", evaluation_policy=POLICY,
                     engine_version=ENGINE, run_plan=run_plan,
                     access=resolve_for(to_scenario(spec), context="t"))
        assert "delivered" in str(refused.value)

    def test_no_data_ends_the_call(self):
        class Nothing:
            frame = None
            event = None

        run_plan, _resolve = runner()
        with pytest.raises(EvaluationRefused):
            evaluate(spec_for(**SCHEDULED), "", evaluation_policy=POLICY,
                     engine_version=ENGINE, access=Nothing(),
                     run_plan=run_plan)


class TestItIsReproducible:
    def test_the_same_spec_and_snapshot_give_the_same_streams(self):
        """The property Step 10 will rest on, checked while it is cheap."""
        first = evaluated(**SCHEDULED)
        second = evaluated(**SCHEDULED)
        assert first.strategy_hash == second.strategy_hash
        assert first.market_snapshot_hash == second.market_snapshot_hash
        for name in STREAMS:
            assert first.streams[name].digest == second.streams[name].digest, (
                f"{name} differs between two runs of one specification")

    def test_a_different_plan_gives_different_streams(self):
        """Without this, reproducibility passes on an evaluator that ignores
        its input."""
        one = evaluated(**SCHEDULED)
        other = evaluated(**{**SCHEDULED, "amount": "2000"})
        assert one.strategy_hash != other.strategy_hash
        assert one.streams["contributions"].digest \
            != other.streams["contributions"].digest


class TestTheSpecificationIsSufficient:
    def test_a_scenario_survives_the_round_trip(self):
        """If a scenario cannot be rebuilt from its own specification, the
        specification is lossy and the service would receive less than the
        engine needs. Step 6 would find it; this finds it a year earlier."""
        from src.mission.strategy_spec import to_scenario

        for stated in (SCHEDULED, TRIGGERED):
            spec = spec_for(**stated)
            assert to_scenario(spec).canonical_form() == \
                to_scenario(from_scenario(to_scenario(spec))).canonical_form()
