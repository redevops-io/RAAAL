"""The posture, made checkable.

Three properties carry it, and none of them may be a promise:

* whether a response reads as a recommendation is **derived** from the payload;
* candidates the platform generated and measured are **all** trials, and ones it
  measured without showing cannot exist;
* an extracted rule is scanned for personal data at every depth, not trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mission import (
    DISCLOSURE_VERSION,
    DISCLOSURES,
    AllocationRule,
    Basis,
    BenchmarkSet,
    Candidate,
    CandidateOrigin,
    CashFlow,
    CashPolicy,
    ComparisonClass,
    FlowSchedule,
    HiddenSelection,
    HoldingsPolicy,
    Intent,
    Mission,
    Objective,
    PrivacyLeak,
    Provenance,
    RunConditions,
    ScenarioSpecification,
    SelectionBasis,
    assess_recommendation,
    buy_and_hold,
    classify,
    compare,
    comparison_payload,
    extract_rule,
    hold_cash,
    scan_for_personal_data,
    scan_language,
    simulate,
)


@pytest.fixture
def prices():
    idx = pd.bdate_range("2020-01-01", periods=200)
    return pd.DataFrame(
        {"A": np.linspace(100, 110, 200), "B": np.linspace(100, 140, 200)}, index=idx
    )


@pytest.fixture
def payload_parts(prices):
    flows = [CashFlow(prices.index[0], 1000.0)]
    specs = [
        {"name": "A", "tickers": ["A"], "program": buy_and_hold(["A"])},
        {"name": "B", "tickers": ["B"], "program": buy_and_hold(["B"])},
        {"name": "Cash", "tickers": [], "program": hold_cash()},
    ]
    mission = simulate(prices, flows=flows, program=buy_and_hold(["A"]),
                       cash_policy=CashPolicy.idle())
    benchmarks = compare(prices, flows=flows, cash_policy=CashPolicy.idle(),
                         benchmarks=specs)
    return mission, benchmarks, [s["name"] for s in specs]


class TestTheVerdictIsDerivedNotDeclared:
    def test_a_neutral_payload_is_not_a_recommendation(self, payload_parts):
        mission, benchmarks, declared = payload_parts
        payload = comparison_payload(
            mission, benchmarks, declared_order=declared,
            rendered_text="Here are the outcomes of the scenario and three "
                          "benchmarks under identical contributions.",
        )
        assert payload["is_recommendation"] is False

    def test_prescriptive_copy_makes_it_a_recommendation(self, payload_parts):
        mission, benchmarks, declared = payload_parts
        payload = comparison_payload(
            mission, benchmarks, declared_order=declared,
            rendered_text="HRP is best for your situation. You should switch to it.",
        )

        assert payload["is_recommendation"] is True
        codes = payload["recommendation_assessment"]["failures"]
        assert "NEXT_ACTION_SUGGESTION" in codes
        assert "PERSONALIZED_SUPERLATIVE_LANGUAGE" in codes

    def test_reordering_the_set_makes_it_a_ranking(self, payload_parts):
        """The check no wording can satisfy."""
        mission, benchmarks, declared = payload_parts
        by_outcome = sorted(
            benchmarks,
            key=lambda b: -(b.result.money_weighted if b.result else -9),
        )
        payload = comparison_payload(mission, by_outcome, declared_order=declared,
                                     rendered_text="neutral")

        assert payload["is_recommendation"] is True
        assert "BENCHMARK_ORDER_SEMANTICALLY_UNRANKED" in \
            payload["recommendation_assessment"]["failures"]

    def test_coincidental_ordering_is_noted_not_failed(self, payload_parts):
        """Three benchmarks sort by chance a third of the time.

        The first version of this check detected sorting and fired constantly on
        payloads that had done nothing. A check that cries wolf gets switched off,
        and a switched-off check protects nothing.
        """
        mission, benchmarks, declared = payload_parts
        payload = comparison_payload(mission, benchmarks, declared_order=declared,
                                     rendered_text="neutral")
        [check] = [c for c in payload["recommendation_assessment"]["checks"]
                   if c["code"] == "BENCHMARK_ORDER_SEMANTICALLY_UNRANKED"]

        assert check["passed"]

    def test_asymmetric_funding_is_caught_from_the_payload(self, prices):
        """Not from a promise that the benchmarks were treated the same."""
        early = compare(prices, flows=[CashFlow(prices.index[0], 1000.0)],
                        cash_policy=CashPolicy.idle(),
                        benchmarks=[{"name": "A", "tickers": ["A"],
                                     "program": buy_and_hold(["A"])}])
        rich = compare(prices, flows=[CashFlow(prices.index[0], 5000.0)],
                       cash_policy=CashPolicy.idle(),
                       benchmarks=[{"name": "B", "tickers": ["B"],
                                    "program": buy_and_hold(["B"])}])

        verdict = assess_recommendation(benchmarks=[*early, *rich],
                                        rendered_text="neutral")
        assert verdict.is_recommendation
        assert "BENCHMARK_SET_SYMMETRIC" in [c.code for c in verdict.failures]

    def test_declared_checks_are_visibly_weaker_than_derived_ones(self, payload_parts):
        mission, benchmarks, declared = payload_parts
        payload = comparison_payload(mission, benchmarks, declared_order=declared,
                                     rendered_text="neutral")
        assessment = payload["recommendation_assessment"]

        assert assessment["derivation_complete"] is False
        assert "rest on assertion" in assessment["headline"]

    def test_peer_behaviour_language_is_caught(self):
        found = scan_language("Investors like you typically switched to HRP.")
        assert found["peer_behavior_used"]

    def test_trade_instructions_are_caught(self):
        assert scan_language("Place an order for 10 shares.")[
            "execution_or_trade_instruction"]

    def test_analytical_explanation_is_permitted(self):
        """Explaining why results differ must stay available."""
        neutral = ("The HRP benchmark had lower drawdown because it allocated "
                   "less to equities.")
        assert not any(scan_language(neutral).values())


class TestThePlatformCannotSearchQuietly:
    def test_measuring_a_candidate_without_showing_it_is_refused(self):
        with pytest.raises(HiddenSelection, match="makes the platform the researcher"):
            Intent(
                name="retire", version=1, stated="invest safely",
                generation_constraints=["passive"],
                candidates=[
                    Candidate("a", "A", evaluated=True, shown_to_user=True),
                    Candidate("b", "B", evaluated=True, shown_to_user=False),
                ],
            )

    def test_platform_candidates_require_declared_generation_constraints(self):
        """A set not built from stated rule attributes was built from results."""
        with pytest.raises(ValueError, match="built from results"):
            Intent(name="r", version=1, stated="s",
                   candidates=[Candidate("a", "A", origin=CandidateOrigin.PLATFORM)])

    def test_constraints_must_be_knowable_before_running(self):
        with pytest.raises(ValueError, match="knowable before anything is run"):
            Intent(name="r", version=1, stated="s",
                   candidates=[Candidate("a", "A")],
                   generation_constraints=["best_cagr"])

    def test_user_authored_candidates_need_no_generation_constraints(self):
        intent = Intent(name="r", version=1, stated="s",
                        candidates=[Candidate("a", "A", origin=CandidateOrigin.USER)])
        assert intent.trials_incurred == 0

    def test_visible_results_contradict_a_blind_basis(self):
        with pytest.raises(ValueError, match="cannot be"):
            Intent(name="r", version=1, stated="s",
                   candidates=[Candidate("a", "A", origin=CandidateOrigin.USER)],
                   selected="a", selection_basis=SelectionBasis.BEFORE_RESULTS,
                   results_visible_before_selection=True)

    def test_rejected_candidates_are_retained(self):
        """So the set that was offered can be reconstructed, not inferred."""
        intent = Intent(
            name="r", version=1, stated="s", generation_constraints=["passive"],
            candidates=[Candidate("a", "A"), Candidate("b", "B")],
            rejected_candidates=["c", "d"],
        )
        assert intent.to_json()["rejected_candidates"] == ["c", "d"]

    def test_the_trial_count_is_exported_under_the_name_deflation_uses(self):
        intent = Intent(
            name="r", version=1, stated="s", generation_constraints=["passive"],
            candidates=[Candidate("a", "A", evaluated=True),
                        Candidate("b", "B", evaluated=True)],
            selected="a", selection_basis=SelectionBasis.AFTER_RESULTS,
            results_visible_before_selection=True,
        )
        assert intent.to_json()["dsr_countable_trials"] == 2


class TestScenarioSeparatesRuleFromMoney:
    def _scenario(self, **kw):
        defaults = dict(
            name="basket", version=1, objective=Objective.REPLAY,
            event_program=[{"observe": "sp500_close"},
                           {"condition": "below_sma_200"},
                           {"action": "allocate_contribution"}],
            flow_schedule=FlowSchedule(cadence="monthly", amount=2000.0),
            allocation_rule=AllocationRule(assets=["AMZN", "AAPL"]),
        )
        defaults.update(kw)
        return ScenarioSpecification(**defaults)

    def test_the_same_rule_under_different_salaries_shares_a_rule_hash(self):
        modest = self._scenario(flow_schedule=FlowSchedule("monthly", 500.0))
        generous = self._scenario(flow_schedule=FlowSchedule("monthly", 5000.0))

        assert modest.rule_hash == generous.rule_hash
        assert modest.content_hash != generous.content_hash

    def test_the_money_path_is_not_part_of_the_rule(self):
        assert "flows" not in self._scenario().methodology_part()
        assert "amount" not in self._scenario().methodology_part()

    def test_never_sell_and_maintained_equal_weight_conflict_structurally(self):
        """Caught from the compiled form, not only from the prose.

        A check that only reads the user's text trusts the compiler to have been
        right about it.
        """
        s = self._scenario(
            allocation_rule=AllocationRule(assets=["A", "B"],
                                           weighting="equal_weight_maintained"),
            holdings_policy=HoldingsPolicy(sells_allowed=False),
        )

        assert not s.is_runnable
        assert "requires selling what rose" in s.self_conflicts()[0]

    def test_equal_dollars_at_purchase_does_not_conflict_with_never_selling(self):
        s = self._scenario(
            allocation_rule=AllocationRule(assets=["A", "B"],
                                           weighting="equal_weight_at_purchase"),
            holdings_policy=HoldingsPolicy(sells_allowed=False),
        )
        assert s.is_runnable

    def test_no_money_at_all_is_a_conflict_not_a_zero(self):
        s = self._scenario(flow_schedule=FlowSchedule("once", 0.0))
        assert "undefined rather than zero" in s.self_conflicts()[0]

    def test_the_benchmark_set_names_the_rule_that_generated_it(self):
        s = self._scenario(benchmark_set=BenchmarkSet(
            generated_by_rule="benchmark-policy/public-default@1",
            members=["spy-dca", "sixty-forty"]))

        assert s.protocol_part()["benchmark_set"]["generated_by_rule"] == \
            "benchmark-policy/public-default@1"
        assert s.protocol_part()["benchmark_set"]["ordering"] == "unordered"


class TestTwoComparisonClasses:
    #: Pins the runtime dimensions. Under classifier @2 an absent value is
    #: NOT_EVALUATED rather than a match, so "identical conditions" has to
    #: include what the runtimes were — otherwise the comparison claims an
    #: isolation it never established.
    BASE = dict(flow_schedule_hash="h1", starting_capital=0.0,
                cash_policy_rate=0.0, tax_treatment="NONE_APPLIED",
                cost_bps=10.0, execution_lag=1,
                period_start="2020-01-01", period_end="2025-01-01", allocation_rule_hash="r1", data_snapshot="s1",
                account_hash="a1", calendar_hash="c1", market_data_hash="m1")

    def test_identical_conditions_isolate_the_rule(self):
        verdict = classify(RunConditions(**self.BASE), RunConditions(**self.BASE))

        assert verdict.comparison_class is ComparisonClass.STRATEGY_EFFECT
        assert verdict.attribution_isolated
        # Against the versioned constant, not a substring: the disclosure is an
        # artifact, and a test matching prose drifts the moment it is reworded.
        assert verdict.required_disclosure == DISCLOSURES["STRATEGY_EFFECT"]
        assert verdict.to_json()["disclosure_version"] == DISCLOSURE_VERSION

    def test_monthly_versus_bonus_is_comparable_but_not_attributable(self):
        """The comparison a user most wants, and the sentence they must not read."""
        other = {**self.BASE, "flow_schedule_hash": "h2"}
        verdict = classify(RunConditions(**self.BASE), RunConditions(**other))

        assert verdict.comparison_class is ComparisonClass.PERSONAL_OUTCOME
        assert verdict.comparable
        assert not verdict.attribution_isolated
        assert verdict.required_disclosure == DISCLOSURES["PERSONAL_OUTCOME"]
        assert "does not isolate which strategy is better" in \
            verdict.what_a_difference_means()
        assert "flow_schedule" in verdict.what_a_difference_means(), (
            "the disclosure must name what actually differed"
        )

    def test_different_periods_defeat_comparison_entirely(self):
        other = {**self.BASE, "period_end": "2023-01-01"}
        verdict = classify(RunConditions(**self.BASE), RunConditions(**other))

        assert not verdict.comparable
        assert "different markets rather than to different rules" in verdict.detail

    def test_every_isolation_dimension_is_checked(self):
        from src.mission import ISOLATION_DIMENSIONS

        for dimension, field, changed in (
            ("starting_capital", "starting_capital", 1000.0),
            ("cash_policy", "cash_policy_rate", 0.04),
            ("tax_treatment", "tax_treatment", "LONG_TERM_CG"),
            ("fees", "cost_bps", 50.0),
            ("execution_timing", "execution_lag", 0),
        ):
            verdict = classify(RunConditions(**self.BASE),
                               RunConditions(**{**self.BASE, field: changed}))
            assert dimension in verdict.differing_dimensions
        assert set(ISOLATION_DIMENSIONS) >= {"fees", "tax_treatment"}


class TestExtractionIsVerifiedNotTrusted:
    def _mission(self, events):
        return Mission(name="p", version=1, title="P", objective=Objective.REPLAY,
                       flows=FlowSchedule("monthly", 2000.0), events=events,
                       provenance=Provenance())

    def test_personal_keys_are_found_at_any_depth(self):
        """A field-level strip removes personal data from where it is supposed
        to live. This finds it where it is not supposed to be."""
        leaks = scan_for_personal_data(
            {"events": [{"trigger": "vest", "then": {"employer": "ACME"}}]}
        )
        assert leaks and "employer" in leaks[0]

    def test_private_references_are_found_by_value_not_by_key(self):
        leaks = scan_for_personal_data({"constraints": ["mission/my-plan@1"]})
        assert "private artifact" in leaks[0]

    def test_a_leaking_rule_is_not_proposable(self):
        extraction = extract_rule(
            self._mission([{"trigger": "vest", "vesting_schedule": "4y1c"}])
        )

        assert not extraction.proposable
        assert extraction.leaks
        with pytest.raises(PrivacyLeak, match="still carries personal data"):
            extraction.verify()

    def test_a_clean_rule_passes_the_verifier(self):
        extraction = extract_rule(
            self._mission([{"trigger": "spy_below_200dma", "action": "buy_basket"}])
        )

        assert extraction.proposable
        extraction.verify()

    def test_the_strip_list_is_enumerated(self):
        """"We strip personal data" is a claim; this is a list."""
        from src.mission.boundary import PERSONAL_FIELDS

        for field in ("flows", "starting_capital", "tax_treatment", "intent_ref"):
            assert field in PERSONAL_FIELDS
