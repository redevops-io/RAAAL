"""Phase 4's first property: a plan compiled from intent, never from prose.

This is the module that lets the legacy reader be deleted. While the only route
to a `ScenarioSpecification` runs through `compile_scenario(text, ...)`, the
regex compiler is load-bearing however the intent was produced, and "Discovery
is authoritative" describes the top of a pipeline whose bottom still parses
sentences.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from runtime_contracts import Author, IntentField, NotSealable, Unresolved, OpenReason, VerifiedIntent
from src.mission.from_intent import (
    COMPILER_VERSION,
    DEFAULTS,
    NotExecutable,
    compile_intent,
)

RULE = "benchmark-policy/public-default@1"


def intent(**fields) -> VerifiedIntent:
    base = {"assets": "SPY", "amount": "1000", "cadence": "monthly"}
    base.update(fields)
    return VerifiedIntent(
        objective="evaluate_investment_strategy",
        produced_by="discovery-runtime@test",
        fields={k: IntentField(value=v, author=Author.USER)
                for k, v in base.items() if v is not None}).seal()


class TestItNeverConsultsTheSentence:
    def test_the_module_does_not_import_the_compiler(self):
        """A convenience import added later would restore the dependency
        silently, and the deletion would then fail in a way nobody predicted."""
        source = Path("src/mission/from_intent.py").read_text()
        # Code only. The module docstring names `compile_scenario` while
        # explaining why this module exists, and a scan that reads its own
        # explanation as a violation is the same false positive the oracle
        # check already produced once.
        code = source.split('"""', 2)[-1]
        for forbidden in ("from .compiler", "import compiler",
                          "compile_scenario(", "parse("):
            assert forbidden not in code, f"{forbidden!r} reintroduces prose"

    def test_the_entry_point_takes_no_text(self):
        import inspect

        signature = inspect.signature(compile_intent)
        assert "text" not in signature.parameters
        assert "utterance" not in signature.parameters

    def test_a_plan_compiles_from_fields_alone(self):
        out = compile_intent(intent(), benchmark_rule=RULE)
        assert out.executable
        assert out.scenario.allocation_rule.assets == ("SPY",)


class TestIdenticalIntentGivesIdenticalPlan:
    """Phase 4's first acceptance property. A compiler that produced two plans
    from one intent would make replay meaningless before replay was reached."""

    def test_the_same_intent_compiles_the_same_way(self):
        one = compile_intent(intent(), benchmark_rule=RULE)
        other = compile_intent(intent(), benchmark_rule=RULE)
        assert one.scenario.content_hash == other.scenario.content_hash

    def test_a_different_intent_does_not(self):
        """The discriminating half."""
        a = compile_intent(intent(cadence="monthly"), benchmark_rule=RULE)
        b = compile_intent(intent(cadence="annual"), benchmark_rule=RULE)
        assert a.scenario.content_hash != b.scenario.content_hash

    def test_the_plan_names_the_intent_it_came_from(self):
        source = intent()
        out = compile_intent(source, benchmark_rule=RULE)
        assert out.derivation["compiled_from"] == source.intent_hash
        assert out.derivation["compiled_by"] == COMPILER_VERSION
        assert out.derivation["intent_produced_by"] == "discovery-runtime@test"


class TestItRefusesRatherThanAdjusting:
    def test_an_unexecutable_value_is_refused_by_name(self):
        out = compile_intent(intent(allocation_method="inverse_volatility"),
                             benchmark_rule=RULE)
        assert not out.executable and out.scenario is None
        assert "allocation_method" in {r.dimension for r in out.refusals}

    def test_no_partial_plan_accompanies_a_refusal(self):
        """A plan beside a refusal is a plan a caller renders anyway, and then
        a figure exists for a request that was refused."""
        out = compile_intent(intent(periodic_rebalancing="quarterly"),
                             benchmark_rule=RULE)
        assert out.refusals and out.scenario is None

    def test_an_intent_holding_nothing_is_refused(self):
        """The first version compiled this happily — an empty allocation and a
        trigger with no subject. Nothing downstream would have priced it, but
        the failure would have read as missing data rather than as an intent
        that never said what to buy."""
        out = compile_intent(intent(assets=None), benchmark_rule=RULE)
        assert "assets" in {r.dimension for r in out.refusals}

    def test_a_draft_is_refused_outright(self):
        draft = VerifiedIntent(objective="o", fields={
            "assets": IntentField("SPY", Author.USER)})
        with pytest.raises(NotExecutable) as raised:
            compile_intent(draft)
        assert "draft" in str(raised.value)

    def test_an_unresolved_disagreement_is_refused(self):
        blocked = replace(intent(), unresolved=(Unresolved(
            "trigger_semantics", OpenReason.UNRESOLVED_DISAGREEMENT),))
        out = compile_intent(blocked, benchmark_rule=RULE)
        assert "trigger_semantics" in {r.dimension for r in out.refusals}


class TestSilenceIsAnAppliedDefaultAndSaysSo:
    def test_defaults_are_reported_not_hidden(self):
        """A plan is only reproducible if a reader can see which values nobody
        asked for. This is the `execution_timing` defect made visible."""
        out = compile_intent(intent(), benchmark_rule=RULE)
        assert "day_rule" in out.applied_defaults
        assert out.scenario.funding.day_rule == DEFAULTS["day_rule"]

    def test_a_stated_value_is_not_reported_as_a_default(self):
        out = compile_intent(intent(day_rule="last_session_of_period"),
                             benchmark_rule=RULE)
        assert "day_rule" not in out.applied_defaults
        assert out.scenario.funding.day_rule == "last_session_of_period"

    def test_every_default_is_for_a_dimension_the_engine_executes(self):
        """Found by this file: the table also carried `dividend_policy` and
        `tax_treatment`, which the manifest refuses and does not model. The
        compiler was supplying a value nothing would act on and reporting it as
        an applied default, which is declared-but-not-executed inside the
        module written to prevent it."""
        from src.mission.capability import EXECUTED, MANIFEST

        for dimension in DEFAULTS:
            entry = MANIFEST.get(dimension)
            if entry is None:
                continue
            assert entry.support == EXECUTED, (
                f"{dimension} has a default and is {entry.support}")

    def test_engine_constants_are_not_reported_as_choices(self):
        """Nobody left them open; they are not choices."""
        from src.mission.from_intent import ENGINE_CONSTANTS

        out = compile_intent(intent(), benchmark_rule=RULE)
        assert not set(ENGINE_CONSTANTS) & set(out.applied_defaults)


class TestTheAssetsAreTheUsersWords:
    def test_a_description_is_not_resolved_to_a_ticker(self):
        """"a core index fund" stays that. The engine refusing to price it is
        the correct failure; choosing VTI is the substitution this boundary
        exists to prevent."""
        out = compile_intent(intent(assets="a core index fund"),
                             benchmark_rule=RULE)
        assert out.scenario.allocation_rule.assets == ("a core index fund",)


class TestFundingAndItsProjectionCannotDisagree:
    def test_a_scheduled_plan_projects_its_own_cadence(self):
        out = compile_intent(intent(cadence="annual"), benchmark_rule=RULE)
        assert out.scenario.funding.cadence == "annual"
        assert out.scenario.flow_schedule.cadence == "annual"

    def test_an_event_plan_declares_no_calendar(self):
        """`funding` is the authority and `flow_schedule` is its projection."""
        out = compile_intent(intent(trigger_semantics="crossing_event",
                                    cadence=None), benchmark_rule=RULE)
        assert out.scenario.flow_schedule.cadence == "event_triggered"
        assert out.scenario.flow_schedule.amount == 0.0

    def test_a_calendar_stated_beside_a_trigger_is_refused_not_dropped(self):
        """The second silent reduction, found by the general stranded-dimension
        check rather than by looking for it.

        This case used to compile: `cadence="monthly"` alongside a crossing
        trigger produced an `EventTriggered` schedule whose `cadence` read
        `"event_triggered"`, and the stated *monthly* went nowhere. Nothing in
        the result said a word of the request had been discarded, so the person
        who asked to contribute monthly *and* on a crossing was shown a plan
        that did one of those things and told it was their plan.

        The event path never consults `cadence`, which is defensible — a
        trigger and a calendar are two different authorities on when money
        moves, and this build has no representation for both at once. What is
        not defensible is deciding that silently. Refusing by name leaves the
        person able to drop one of the two and get what they asked for.
        """
        out = compile_intent(intent(trigger_semantics="crossing_event"),
                             benchmark_rule=RULE)
        assert out.scenario is None
        assert [r.dimension for r in out.refusals] == ["cadence"]
        assert out.refusals[0].kind == "UNSUPPORTED_DIMENSION"
