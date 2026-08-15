"""`StrategySpec` is what crosses the boundary, so what it carries is the point.

Step 3. Three properties, and the third is the one that keeps the other two
honest over time:

  * execution semantics only — no prose, no reader, no Discovery, no UI;
  * hashed over what executes, so the same plan described differently is one
    run and two plans the evaluator would price differently are two;
  * mechanically produced from the scenario, deciding nothing of its own.

The last matters because a default applied here would be a third place values
come from, after Discovery's canonicalisation and Mission's declared defaults —
and the plan a person reviewed would not be the plan that ran.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from runtime_contracts import Author, IntentField, VerifiedIntent

from src.discovery.canonical import canonicalise
from src.mission.from_intent import compile_intent
from src.mission.strategy_spec import SPEC_VERSION, StrategySpec, from_scenario

SRC = Path(__file__).resolve().parent.parent / "src"
BASE = {"assets": "VTI", "amount": "1000", "cadence": "monthly"}


def spec(**stated) -> StrategySpec:
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


class TestItCarriesOnlyExecutionSemantics:
    def test_the_wire_form_contains_no_sentence(self):
        """An evaluator that could see the words could act on them.

        Checked on the serialized body rather than the class, because that is
        what would actually travel.
        """
        body = json.dumps(spec(**BASE).to_json())
        for word in ("I invest", "every month", "utterance", "reader_id",
                     "produced_by", "intent_hash"):
            assert word not in body, f"{word!r} reached the specification"

    def test_it_imports_nothing_that_reads_or_renders(self):
        source = (SRC / "mission" / "strategy_spec.py").read_text()
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                imported.add(("." * node.level) + (node.module or ""))
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        forbidden = [name for name in imported
                     if "discovery" in name or "workspace" in name]
        assert forbidden == [], (
            f"{forbidden} — the specification is what the evaluation service "
            "receives, and it cannot take an interpreter or a web layer with "
            "it")

    def test_it_names_the_conventions_by_their_public_names(self):
        """A reader outside this repository can check `ModifiedFollowing`
        against a definition that is not ours. "the first session on or after
        the 15th unless that crosses a month" they cannot."""
        conventions = spec(**BASE).conventions
        assert conventions.vocabulary.startswith("QuantLib")
        assert conventions.business_day == "ModifiedFollowing"
        assert conventions.calendar == "NYSE"
        assert conventions.settlement_lag == "T+1"
        assert conventions.currency == "USD"

    def test_the_frequency_comes_from_this_plans_cadence(self):
        """Not a constant. A spec naming one frequency for every plan would let
        the evaluator infer the schedule it is supposed to be told."""
        assert spec(**BASE).conventions.schedule_frequency == "Monthly"
        assert spec(**{**BASE, "cadence": "annual"}) \
            .conventions.schedule_frequency == "Annual"

    def test_measurement_conventions_are_not_in_the_specification(self):
        """Two people running one strategy under different measurement
        conventions are running one strategy. A `strategy_hash` that moved with
        the annualisation basis would call them different plans."""
        body = json.dumps(spec(**BASE).to_json())
        for measurement in ("annualisation", "Business/252", "compounding",
                            "evaluation_date", "sessions_per_year"):
            assert measurement not in body, (
                f"{measurement!r} is in the strategy identity, so changing how "
                "a figure is measured changes which plan it is")


class TestTheHashIsOverWhatExecutes:
    def test_the_same_plan_described_differently_is_one_spec(self):
        assert spec(**{**BASE, "amount": "$1,000"}).spec_hash \
            == spec(**{**BASE, "amount": "1000 usd"}).spec_hash

    def test_a_different_cadence_is_a_different_spec(self):
        assert spec(**{**BASE, "cadence": "monthly"}).spec_hash \
            != spec(**{**BASE, "cadence": "annual"}).spec_hash

    def test_a_different_holding_is_a_different_spec(self):
        assert spec(**{**BASE, "assets": "VTI"}).spec_hash \
            != spec(**{**BASE, "assets": "VTI,BND"}).spec_hash

    def test_the_dividend_policy_is_in_the_hash(self):
        """The field that was an engine constant until it turned out to choose
        the price series. Two plans the evaluator would price differently must
        not share an identity."""
        from dataclasses import replace

        one = spec(**BASE)
        other = replace(one, dividend_policy="held_as_cash")
        assert one.spec_hash != other.spec_hash

    def test_the_version_is_in_the_hash(self):
        """A field added later changes every hash, and a comparison across that
        change must be able to say the rules differed rather than the plans."""
        from dataclasses import replace

        one = spec(**BASE)
        assert one.version == SPEC_VERSION
        assert replace(one, version="quantify-strategy-spec@2").spec_hash \
            != one.spec_hash

    def test_the_hash_is_stable_across_calls(self):
        assert spec(**BASE).spec_hash == spec(**BASE).spec_hash


class TestItDecidesNothing:
    def test_every_value_is_present_in_the_scenario(self):
        """Mechanical, not interpretive.

        Spot-checked on the values that have been silently supplied before:
        cadence, amount and the dividend policy each reached a plan without
        anybody choosing them, and this is the step where that would be
        invisible again.
        """
        canonical = canonicalise(BASE)
        intent = VerifiedIntent(
            objective="evaluate_investment_strategy", produced_by="test",
            utterance_ref="u",
            fields={n: IntentField(value=v, author=Author.MODEL)
                    for n, (v, _a) in canonical.fields.items()},
            unresolved=()).seal()
        scenario = compile_intent(intent, benchmark_rule="a-rule").scenario
        built = from_scenario(scenario)

        assert built.funding.cadence == scenario.funding.cadence
        assert built.funding.amount == str(scenario.funding.amount)
        assert built.dividend_policy == scenario.holdings_policy.dividend_policy
        assert built.sells_allowed == scenario.holdings_policy.sells_allowed
        assert built.assets == tuple(scenario.allocation_rule.assets)

    def test_a_triggered_plan_carries_its_trigger_and_no_cadence(self):
        built = spec(**{"assets": "VOO", "amount": "1000",
                        "observed_assets": "SPY",
                        "trigger_semantics": "crossing_event",
                        "moving_average_window": "200"})
        assert built.funding.kind == "event_triggered"
        assert built.funding.trigger is not None
        assert built.funding.trigger.subject == "SPY"
        assert built.funding.trigger.window == 200
        assert built.funding.cadence == "", (
            "a triggered plan carries a cadence, so an evaluator could act on "
            "a schedule nobody described")
        assert built.observed_assets == ("SPY",)

    def test_a_scheduled_plan_carries_no_trigger(self):
        built = spec(**BASE)
        assert built.funding.kind == "scheduled"
        assert built.funding.trigger is None
        assert built.observed_assets == ()
