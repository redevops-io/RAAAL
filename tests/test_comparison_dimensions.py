"""The last hand-maintained causal table, replaced by declarations.

`ISOLATION_DIMENSIONS` was a tuple of strings that had to be remembered whenever
a new runtime or scenario field appeared. It was wrong twice — `allocation_rule`
and `data_snapshot` were both missing, and both silently let a comparison claim
attribution it did not have. A list that must be remembered is a list that will
be forgotten.
"""
from __future__ import annotations

import pytest

from src.comparison.dimensions import (
    ISOLATION_DIMENSIONS,
    REGISTRY,
    ComparisonDimension,
    Requirement,
    SourceKind,
    UndeclaredDimension,
    dimension,
    dimensions_defeating,
    dimensions_requiring_equality,
    isolation_of,
    unextractable,
)

STRATEGY, PERSONAL, CONSTRAINT = (
    "STRATEGY_EFFECT", "PERSONAL_OUTCOME", "CONSTRAINT_EFFECT")


class TestEveryDimensionDeclaresItsParticipation:
    def test_an_unregistered_field_cannot_be_used(self):
        with pytest.raises(UndeclaredDimension, match="must declare which"):
            dimension("something_new")

    def test_every_registered_dimension_names_all_three_classes(self):
        for spec in REGISTRY.values():
            for cls in (STRATEGY, PERSONAL, CONSTRAINT):
                assert cls in spec.supports, (
                    f"{spec.id} does not say what {cls} demands of it"
                )

    def test_every_dimension_has_a_plain_language_label(self):
        """"flow_schedule" is a field name; "cash-flow schedule" is a sentence."""
        for spec in REGISTRY.values():
            assert spec.causal_label
            assert "_" not in spec.causal_label

    def test_every_dimension_can_actually_be_read(self):
        """A declared dimension with no extractor would never be noticed."""
        assert unextractable() == []

    def test_both_artifact_classes_are_represented(self):
        kinds = {spec.source_kind for spec in REGISTRY.values()}
        assert kinds == {SourceKind.RUNTIME, SourceKind.SCENARIO_ARTIFACT}

    def test_a_scenario_artifact_is_not_forced_to_be_a_runtime(self):
        """Personal declarations and reusable policy are different things."""
        assert dimension("flow_schedule").source_kind is SourceKind.SCENARIO_ARTIFACT
        assert dimension("tax_treatment").source_kind is SourceKind.RUNTIME

    def test_registering_the_same_id_twice_is_refused(self):
        with pytest.raises(ValueError, match="already registered"):
            from src.comparison.dimensions import register

            register(ComparisonDimension(
                id="tax_treatment", source_kind=SourceKind.RUNTIME,
                causal_label="duplicate", supports={}))


class TestTheTableIsDerived:
    def test_isolation_dimensions_comes_from_the_registry(self):
        from src.mission.comparability import ISOLATION_DIMENSIONS as LIVE

        assert set(LIVE) == set(REGISTRY)

    def test_the_two_previously_missing_dimensions_are_present(self):
        assert "allocation_rule" in ISOLATION_DIMENSIONS
        assert "data_snapshot" in ISOLATION_DIMENSIONS

    def test_adding_a_dimension_changes_the_table_without_editing_it(self):
        from src.comparison.dimensions import register

        register(ComparisonDimension(
            id="_probe", source_kind=SourceKind.RUNTIME, causal_label="a probe",
            extractor=lambda s, e: None,
            supports={STRATEGY: Requirement.MUST_EQUAL,
                      PERSONAL: Requirement.MAY_DIFFER,
                      CONSTRAINT: Requirement.MUST_EQUAL}))
        try:
            assert "_probe" in dimensions_requiring_equality(STRATEGY)
        finally:
            REGISTRY.pop("_probe")


class TestClassesDemandDifferentThings:
    def test_a_flow_schedule_may_differ_only_for_personal_outcomes(self):
        """Exactly the case a single flat list could not express."""
        spec = dimension("flow_schedule")

        assert spec.requirement_for(STRATEGY) is Requirement.MUST_EQUAL
        assert spec.requirement_for(PERSONAL) is Requirement.MAY_DIFFER
        assert spec.requirement_for(CONSTRAINT) is Requirement.MUST_EQUAL

    def test_execution_timing_is_what_a_constraint_comparison_moves(self):
        spec = dimension("execution_timing")

        assert spec.requirement_for(CONSTRAINT) is Requirement.MAY_DIFFER
        assert spec.requirement_for(STRATEGY) is Requirement.MUST_EQUAL

    def test_a_different_period_defeats_every_class(self):
        """Not a weaker comparison — no comparison."""
        for cls in (STRATEGY, PERSONAL, CONSTRAINT):
            assert "evaluation_period" in dimensions_defeating(cls)

    def test_defeating_is_distinct_from_must_equal(self):
        assert Requirement.DEFEATS_COMPARISON is not Requirement.MUST_EQUAL


class TestIsolationChecksSemanticDependencies:
    def test_one_eligible_dimension_isolates(self):
        assert isolation_of(["tax_treatment"]) == "tax_treatment"

    def test_a_dimension_whose_dependency_also_differs_does_not_isolate(self):
        """Tax means different things under different accounts.

        "Only tax differs" is a claim about causation only if account is equal.
        Without the check, an explanation gets assembled from a dimension whose
        semantics were never verified.
        """
        assert dimension("tax_treatment").depends_on == ("account",)
        assert isolation_of(["tax_treatment", "account"]) is None

    def test_a_flow_schedule_depends_on_the_calendar(self):
        assert "calendar" in dimension("flow_schedule").depends_on
        assert isolation_of(["flow_schedule", "calendar"]) is None

    def test_a_snapshot_depends_on_the_data_policy(self):
        assert "market_data" in dimension("data_snapshot").depends_on

    def test_an_ineligible_dimension_never_explains_anything(self):
        assert not dimension("evaluation_period").isolation_eligible
        assert isolation_of(["evaluation_period"]) is None

    def test_two_differences_isolate_nothing(self):
        assert isolation_of(["tax_treatment", "fees"]) is None

    def test_no_difference_isolates_nothing(self):
        assert isolation_of([]) is None


class TestTheVerdictReportsWhatItCouldNotCheck:
    #: Pins every dimension, so "a full comparison" below is actually full.
    #: Under classifier @1 this set looked complete because two absent hashes
    #: compared equal; under @2 an absent value is NOT_EVALUATED, and the test
    #: was asserting completeness it did not have.
    BASE = dict(flow_schedule_hash="h1", starting_capital=0.0,
                cash_policy_rate=0.0, tax_treatment="NONE", cost_bps=10.0,
                execution_lag=1, period_start="2021-01-01",
                period_end="2023-01-01", allocation_rule_hash="r1",
                data_snapshot="s1", account_hash="a1", calendar_hash="c1",
                market_data_hash="m1")

    def test_a_full_comparison_reports_nothing_unchecked(self):
        from src.mission import RunConditions, classify

        verdict = classify(RunConditions(**self.BASE), RunConditions(**self.BASE))
        assert verdict.unchecked_dimensions == ()

    def test_differences_are_reported_in_plain_language(self):
        from src.mission import RunConditions, classify

        verdict = classify(
            RunConditions(**self.BASE),
            RunConditions(**{**self.BASE, "tax_treatment": "ROTH"}))

        assert verdict.to_json()["differing_labels"] == ["tax treatment"]

    def test_the_new_runtime_dimensions_are_compared(self):
        from src.mission import RunConditions, classify

        verdict = classify(
            RunConditions(**{**self.BASE, "account_hash": "a"}),
            RunConditions(**{**self.BASE, "account_hash": "b"}))

        assert "account" in verdict.differing_dimensions

    def test_a_counterfactual_with_a_differing_dependency_stops_isolating(self):
        from src.mission import RunConditions, classify_counterfactual

        verdict = classify_counterfactual(
            RunConditions(**{**self.BASE, "execution_lag": 1, "calendar_hash": "a"}),
            RunConditions(**{**self.BASE, "execution_lag": 0, "calendar_hash": "b"}),
            constraint="the blackout window")

        assert not verdict.attribution_isolated


class TestDependenciesAreDerivedNotRestated:
    """Two declarations of one fact drift, and this drift would be silent — a
    comparison would stop checking something and still report isolation."""

    def test_the_registry_and_the_runtimes_agree(self):
        from src.comparison.dimensions import reconcile_dependencies

        assert reconcile_dependencies() == {}, (
            "a runtime's causal dependencies and the comparison registry's "
            "depends_on have diverged"
        )

    def test_tax_inherits_its_dependency_from_the_runtime(self):
        from src.comparison.dimensions import derived_dependencies
        from src.runtime import TaxRuntime

        assert TaxRuntime.undefined_without == ("account",)
        assert derived_dependencies(dimension("tax_treatment")) == ["account"]

    def test_a_new_runtime_precondition_would_be_caught(self):
        """The check that makes the derivation worth having."""
        from src.comparison.dimensions import reconcile_dependencies
        from src.runtime import TaxRuntime

        original = TaxRuntime.undefined_without
        TaxRuntime.undefined_without = ("account", "calendar")
        try:
            drift = reconcile_dependencies()
            assert "tax_treatment" in drift
            assert drift["tax_treatment"]["derived"] == ["account", "calendar"]
        finally:
            TaxRuntime.undefined_without = original

    def test_not_every_interpretation_relation_is_causal(self):
        """An account may refuse a flow without changing what the flow means.

        Marking every `interpreted_with` relation causal would defeat isolation
        on relations that do not bear on it.
        """
        from src.runtime import CashFlowRuntime

        assert "account" in CashFlowRuntime.interpreted_with
        assert "account" not in CashFlowRuntime.affects_causal_isolation
        assert "account" not in CashFlowRuntime.causal_dependencies()

    def test_a_causal_interpretation_would_be_included(self):
        from src.runtime import CashFlowRuntime

        original = CashFlowRuntime.affects_causal_isolation
        CashFlowRuntime.affects_causal_isolation = ("account",)
        try:
            assert "account" in CashFlowRuntime.causal_dependencies()
        finally:
            CashFlowRuntime.affects_causal_isolation = original

    def test_a_relation_not_in_interpreted_with_cannot_be_marked_causal(self):
        """The causal set is a subset, so it cannot invent a relation."""
        from src.runtime import AccountRuntime

        original = AccountRuntime.affects_causal_isolation
        AccountRuntime.affects_causal_isolation = ("nonexistent",)
        try:
            assert "nonexistent" not in AccountRuntime.causal_dependencies()
        finally:
            AccountRuntime.affects_causal_isolation = original

    def test_every_runtime_dimension_can_be_reconciled(self):
        """A dimension naming an unregistered runtime looks reconciled while its
        dependencies are still unchecked. `TradingCalendar` was the last one."""
        from src.comparison.dimensions import unreconcilable

        assert unreconcilable() == {}

    def test_a_scenario_only_dimension_stays_hand_authored(self):
        """`data_snapshot` is a realized instance, not a policy, so it names no
        runtime type and its dependency is authored deliberately."""
        from src.comparison.dimensions import derived_dependencies

        assert dimension("data_snapshot").depends_on == ("market_data",)
        assert derived_dependencies(dimension("data_snapshot")) is None


class TestClassifierV2:
    """`@2`: an absent value is NOT_EVALUATED, never a match.

    A stored verdict does not merely answer "did I find a difference?". It
    claims "these dimensions were checked and found equivalent" — and when both
    hashes are empty that claim is false.
    """

    COMMON = dict(flow_schedule_hash="f1", starting_capital=0.0,
                  cash_policy_rate=0.0, tax_treatment="ROTH", cost_bps=10.0,
                  execution_lag=1, period_start="2016-01-04",
                  period_end="2025-11-19", data_snapshot="prices@x")

    def conditions(self, **overrides):
        from src.mission.comparability import RunConditions

        return RunConditions(**{**self.COMMON, "allocation_rule_hash": "a",
                                **overrides})

    def test_two_absent_values_are_not_a_match(self):
        from src.mission.comparability import DimensionStatus, classify

        verdict = classify(self.conditions(), self.conditions())
        statuses = {r.dimension: r.status for r in verdict.dimension_results}
        assert statuses["account"] is DimensionStatus.NOT_EVALUATED
        assert statuses["calendar"] is DimensionStatus.NOT_EVALUATED
        assert statuses["market_data"] is DimensionStatus.NOT_EVALUATED

    def test_a_one_sided_absence_is_not_a_match(self):
        """A value cannot be compared against an unknown, and calling that a
        match asserts the unknown away."""
        from src.mission.comparability import DimensionStatus, classify

        verdict = classify(self.conditions(account_hash="acc"),
                           self.conditions())
        statuses = {r.dimension: r.status for r in verdict.dimension_results}
        assert statuses["account"] is DimensionStatus.NOT_EVALUATED

    def test_isolation_is_refused_while_a_dimension_is_unevaluated(self):
        """The most important consequence. The comparison may still be shown;
        it may not claim the strategy was isolated."""
        from src.mission.comparability import classify

        verdict = classify(self.conditions(), self.conditions())
        assert verdict.comparable is True
        assert verdict.attribution_isolated is False
        assert "never evaluated" in verdict.detail

    def test_isolation_is_granted_when_everything_is_pinned(self):
        from src.mission.comparability import classify

        pinned = dict(account_hash="acc", calendar_hash="cal",
                      market_data_hash="md")
        verdict = classify(self.conditions(**pinned), self.conditions(**pinned))
        assert verdict.attribution_isolated is True
        assert verdict.unchecked_dimensions == ()

    def test_a_real_difference_is_still_reported(self):
        from src.mission.comparability import DimensionStatus, classify

        pinned = dict(account_hash="acc", calendar_hash="cal",
                      market_data_hash="md")
        verdict = classify(self.conditions(**pinned),
                           self.conditions(**{**pinned, "account_hash": "other"}))
        statuses = {r.dimension: r.status for r in verdict.dimension_results}
        assert statuses["account"] is DimensionStatus.NOT_MATCHED
        assert "account" in verdict.differing_dimensions

    def test_every_dimension_explains_itself(self):
        """A boolean said only whether something differed. It could not say
        whether anyone had looked."""
        from src.mission.comparability import classify

        for result in classify(self.conditions(), self.conditions()).dimension_results:
            assert result.reason, result.dimension

    def test_the_classifier_version_travels_with_the_verdict(self):
        """`@1` verdicts must keep meaning what they meant, so a reader has to
        be able to tell which rules produced one."""
        from src.mission.comparability import CLASSIFIER_VERSION, classify

        verdict = classify(self.conditions(), self.conditions())
        assert verdict.classifier_version == CLASSIFIER_VERSION
        assert verdict.to_json()["classifier_version"] == CLASSIFIER_VERSION
        assert CLASSIFIER_VERSION.endswith("@2")


class TestWorkspaceRunsPinTheirRuntimes:
    """The deeper fix: those dimensions were unevaluated because the run never
    recorded what it used."""

    def test_a_known_account_is_pinned(self):
        from src.mission.compiler import compile_scenario
        from src.workspace.environment import pins_for

        compiled = compile_scenario(
            "I put $500 into SPY monthly in my Roth IRA and never sell.",
            name="p", version=1, benchmark_rule="benchmark-policy/public-default@1")
        pins = pins_for(compiled.scenario, snapshot="prices@2025-11-19")
        assert pins.account_hash and pins.calendar_hash and pins.market_data_hash
        assert pins.unpinned == ()

    def test_an_account_the_compiler_cannot_place_is_declared(self):
        """Pinning the nearest available kind would record a tax treatment the
        user did not describe.

        Roth 401(k) used to be this example and is now a first-class kind. What
        remains are accounts the compiler recognises as *phrases* and cannot
        place at all — an inherited IRA has its own distribution schedule, a
        donor-advised fund its own contribution and grant rules.
        """
        from src.mission.compiler import compile_scenario
        from src.workspace.environment import pins_for

        compiled = compile_scenario(
            "I put $500 into SPY monthly in my inherited IRA account and never "
            "sell.", name="p", version=1,
            benchmark_rule="benchmark-policy/public-default@1")
        pins = pins_for(compiled.scenario, snapshot="prices@2025-11-19")
        assert pins.account_hash == ""
        assert "account" in pins.unpinned
        assert pins.limitations()[0]["dimension"] == "account"

    def test_every_account_the_compiler_reads_can_now_be_pinned(self):
        """The mapping and the recogniser must not drift apart: a vocabulary
        the compiler can read and the runtime cannot represent is a plan that
        silently loses its tax treatment."""
        from src.mission.compiler import _RULES
        from src.workspace.environment import ACCOUNT_KINDS

        readable = {value for field, value, _ in _RULES
                    if field == "account_type"}
        assert readable <= set(ACCOUNT_KINDS), readable - set(ACCOUNT_KINDS)

    def test_a_missing_snapshot_is_declared_not_defaulted(self):
        """A default filled in at read time describes the current setup rather
        than the historical one."""
        from src.mission.compiler import compile_scenario
        from src.workspace.environment import pins_for

        compiled = compile_scenario(
            "I put $500 into SPY monthly in my Roth IRA and never sell.",
            name="p", version=1, benchmark_rule="benchmark-policy/public-default@1")
        pins = pins_for(compiled.scenario, snapshot="")
        assert pins.market_data_hash == ""
        assert "market_data" in pins.unpinned
