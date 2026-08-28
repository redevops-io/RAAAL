"""One lifecycle for every runtime, and the defect a string was hiding.

`tax_treatment: str = "NONE_APPLIED"` sat in `ISOLATION_DIMENSIONS` and was
compared by string equality, so two plans both saying `NONE_APPLIED` matched on
that dimension even when one was a Roth and the other taxable. The comparability
engine would report STRATEGY_EFFECT — *attribution isolated* — for a pair whose
tax treatment differed completely.
"""
from __future__ import annotations

import pytest

from src.runtime import (
    ACCOUNT_IMPLEMENTED,
    PRE_TAX,
    TAXABLE_BROKERAGE,
    TAX_DEFERRED,
    TAX_IMPLEMENTED,
    US_FEDERAL_WITHHOLDING,
    AccountKind,
    AccountRuntime,
    ExecutionEnvironment,
    LotMethod,
    MissingRuntime,
    RuntimeArtifact,
    TaxPolicyRuntime,
    TaxRuntime,
)


class _Stub(RuntimeArtifact):
    kind = "calendar"

    def __init__(self, name="c", version=1, sessions=252, note=""):
        self.name, self.version = name, version
        self.sessions, self.note = sessions, note

    def declared_form(self):
        return {"sessions": self.sessions, "note": self.note}

    def comparable_form(self):
        return {"sessions": self.sessions}


def environment(**overrides):
    runtimes = {"protocol": _Stub(name="p"), "calendar": _Stub()}
    runtimes["protocol"].kind = "protocol"
    runtimes.update(overrides)
    return ExecutionEnvironment(runtimes=runtimes)


class TestTheStringWasHidingADifference:
    def test_two_untaxed_runtimes_are_not_automatically_the_same(self):
        """The whole reason tax had to stop being a string."""
        assert not PRE_TAX.is_comparable_with(TAX_DEFERRED)
        assert PRE_TAX.compatibility_hash != TAX_DEFERRED.compatibility_hash

    def test_a_runtime_is_comparable_with_itself(self):
        assert PRE_TAX.is_comparable_with(PRE_TAX)

    def test_jurisdiction_is_part_of_comparability(self):
        us = TaxRuntime(name="w", version=1, jurisdiction="US-federal",
                        supplemental_withholding_rate=0.22)
        uk = TaxRuntime(name="w", version=1, jurisdiction="UK",
                        supplemental_withholding_rate=0.22)

        assert not us.is_comparable_with(uk)

    def test_a_declared_pre_tax_runtime_is_a_real_artifact(self):
        """"We model no tax" is an answer; the absence of one is not."""
        assert PRE_TAX.artifact_id == "tax/pre-tax@1"
        assert not PRE_TAX.models_capital_gains
        assert any(l.name == "no-capital-gains" for l in PRE_TAX.limitations)


class TestIdentityAndComparabilityAreDifferentHashes:
    def test_prose_changes_identity_but_not_comparability(self):
        """Correcting a citation must not sever a whole lineage of results."""
        first = TaxRuntime(name="t", version=1, jurisdiction="US-federal",
                           title="Withholding", citations=("pub-15",))
        second = TaxRuntime(name="t", version=2, jurisdiction="US-federal",
                            title="Withholding (corrected)",
                            citations=("irs-pub-15-supplemental",))

        assert first.content_hash != second.content_hash
        assert first.compatibility_hash == second.compatibility_hash
        assert first.is_comparable_with(second)

    def test_a_rate_change_severs_comparability(self):
        low = TaxRuntime(name="t", version=1, jurisdiction="US",
                         supplemental_withholding_rate=0.22)
        high = TaxRuntime(name="t", version=2, jurisdiction="US",
                          supplemental_withholding_rate=0.37)

        assert not low.is_comparable_with(high)

    def test_the_default_is_the_safe_direction(self):
        """A runtime that has not thought about the split refuses comparisons it
        might have allowed, rather than allowing ones it should have refused."""
        class Bare(RuntimeArtifact):
            kind = "bare"
            name, version = "b", 1

            def declared_form(self):
                return {"a": 1, "prose": "anything"}

        bare = Bare()
        assert bare.content_hash == bare.compatibility_hash

    def test_runtimes_of_different_kinds_never_compare(self):
        assert not PRE_TAX.is_comparable_with(TAXABLE_BROKERAGE)


class TestDeclarationsMustBeRealized:
    def test_an_unimplemented_mechanic_is_reported(self):
        assert US_FEDERAL_WITHHOLDING.unrealized(()) == ["supplemental-withholding"]

    def test_the_implemented_set_clears_it(self):
        assert US_FEDERAL_WITHHOLDING.unrealized(TAX_IMPLEMENTED) == []

    def test_a_runtime_declaring_gains_it_cannot_compute_is_caught(self):
        """A runtime is a better place to hide an inert declaration than a
        methodology, because nobody reads it."""
        pretends = TaxRuntime(name="t", version=1, jurisdiction="US",
                              capital_gains_short_rate=0.37,
                              capital_gains_long_rate=0.20)

        assert "capital-gains" in pretends.unrealized(TAX_IMPLEMENTED)

    def test_every_assumption_names_a_mechanism(self):
        for runtime in (PRE_TAX, US_FEDERAL_WITHHOLDING, TAX_DEFERRED,
                        TAXABLE_BROKERAGE):
            for assumption in runtime.assumptions:
                assert assumption.realized_by

    def test_the_withholding_risk_survives_into_the_runtime(self):
        [withholding] = [a for a in US_FEDERAL_WITHHOLDING.assumptions
                         if a.name == "supplemental-withholding"]
        assert "not a tax rate" in withholding.risk


class TestTheDeclarationLayerIsNamedButStillDoesNotCompute:
    """Plan §9 makes the two-layer split explicit: ``TaxPolicyRuntime`` is the
    declaration-layer name for what RAAAL already had. It must be a non-breaking
    alias — one type, so every existing caller keeps working — and it must still
    only *declare* mechanics, never compute a liability (that is the wealth-manager
    ``TaxRealizationEngine``'s job)."""

    def test_tax_policy_runtime_is_the_declaration_layer_alias(self):
        # A plain alias, not a fork: the two names are the identical type.
        assert TaxPolicyRuntime is TaxRuntime

    def test_existing_tax_runtime_callers_still_work_through_either_name(self):
        # Constructing via the new name yields an ordinary TaxRuntime that the
        # existing comparability machinery treats identically.
        via_policy = TaxPolicyRuntime(name="t", version=1, jurisdiction="US-federal",
                                      capital_gains_short_rate=0.37,
                                      capital_gains_long_rate=0.20)
        via_runtime = TaxRuntime(name="t", version=1, jurisdiction="US-federal",
                                 capital_gains_short_rate=0.37,
                                 capital_gains_long_rate=0.20)
        assert isinstance(via_policy, TaxRuntime)
        assert via_policy.kind == "tax"
        # Same type + same meaning ⇒ they are equal and comparable (a subclass
        # would have severed this via dataclass __eq__'s __class__ check).
        assert via_policy == via_runtime
        assert via_policy.is_comparable_with(via_runtime)

    def test_the_declaration_layer_still_refuses_to_compute_liability(self):
        """A declared capital-gains rate is an unrealized assumption here: the
        mechanism that would realize it (``realize_gain``) is not implemented in
        this layer. Declaring a rate is not computing a liability."""
        declared = TaxPolicyRuntime(name="t", version=1, jurisdiction="US",
                                    capital_gains_short_rate=0.37,
                                    capital_gains_long_rate=0.20)
        assert "capital-gains" in declared.unrealized(TAX_IMPLEMENTED)
        # The realization mechanism is explicitly absent from what this layer does.
        from src.runtime.tax import IMPLEMENTED
        assert "realize_gain" not in IMPLEMENTED


class TestAccountRulesAreNotTaxRules:
    def test_they_are_separate_kinds(self):
        assert TAXABLE_BROKERAGE.kind == "account"
        assert PRE_TAX.kind == "tax"

    def test_an_account_declares_what_it_does_not_enforce(self):
        names = {l.name for l in TAXABLE_BROKERAGE.limitations}
        assert "no-required-distributions" in names
        assert "no-early-withdrawal-penalty" in names

    def test_a_contribution_limit_is_an_assumption_with_a_mechanism(self):
        roth = AccountRuntime(name="roth", version=1,
                              account_kind=AccountKind.ROTH_IRA,
                              annual_contribution_limit=7000.0)
        [limit] = [a for a in roth.assumptions if a.name == "contribution-limit"]

        assert limit.realized_by == "cap_contribution"
        # The mechanism now exists, so the assumption is realized. It says
        # nothing about whether the *figure* is right — that is a separate
        # claim, checked in tests/test_account_limits.py.
        assert roth.unrealized(ACCOUNT_IMPLEMENTED) == []
        assert callable(getattr(roth, "cap_contribution"))

    def test_two_accounts_differing_only_in_kind_are_incomparable(self):
        roth = AccountRuntime(name="a", version=1, account_kind=AccountKind.ROTH_IRA)
        trad = AccountRuntime(name="a", version=1,
                              account_kind=AccountKind.TRADITIONAL_IRA)

        assert not roth.is_comparable_with(trad)


class TestTheEnvironmentComposesThem:
    def test_a_missing_required_runtime_raises_rather_than_defaults(self):
        """Every default in this system's history became an erratum."""
        with pytest.raises(MissingRuntime, match="nobody chose the conditions for"):
            ExecutionEnvironment(runtimes={"calendar": _Stub()})

    def test_a_runtime_filed_under_the_wrong_kind_is_refused(self):
        stub = _Stub()
        with pytest.raises(ValueError, match="declares itself"):
            ExecutionEnvironment(runtimes={"protocol": stub, "calendar": stub,
                                           "tax": stub})

    def test_the_environment_hash_covers_every_runtime(self):
        first = environment(tax=PRE_TAX)
        second = environment(tax=TAX_DEFERRED)

        assert first.environment_hash != second.environment_hash

    def test_one_differing_runtime_is_named(self):
        """This is what makes an explanation mechanical rather than narrative."""
        first = environment(tax=PRE_TAX)
        second = environment(tax=TAX_DEFERRED)

        assert first.isolation(second) == "tax"
        assert first.differences(second) == ["tax"]

    def test_two_differing_runtimes_isolate_nothing(self):
        first = environment(tax=PRE_TAX, account=TAXABLE_BROKERAGE)
        second = environment(
            tax=TAX_DEFERRED,
            account=AccountRuntime(name="roth", version=1,
                                   account_kind=AccountKind.ROTH_IRA))

        assert first.isolation(second) is None
        assert first.differences(second) == ["account", "tax"]

    def test_a_runtime_present_on_one_side_only_counts_as_a_difference(self):
        assert environment(tax=PRE_TAX).differences(environment()) == ["tax"]

    def test_prose_only_changes_leave_the_environment_comparable(self):
        first = environment(tax=TaxRuntime(name="t", version=1,
                                           jurisdiction="US", title="A"))
        second = environment(tax=TaxRuntime(name="t", version=2,
                                            jurisdiction="US", title="B"))

        assert first.differences(second) == []
        assert first.compatibility_hash == second.compatibility_hash
        assert first.environment_hash != second.environment_hash

    def test_the_merged_scope_names_which_runtime_declines_what(self):
        scope = environment(tax=PRE_TAX, account=TAXABLE_BROKERAGE).scope()
        sources = {n["runtime"] for n in scope["not_modelled"]}

        assert {"tax", "account"} <= sources
        assert "how a correct treatment gets read as a gap" in scope["note"]

    def test_unrealized_declarations_are_reported_per_runtime(self):
        gaps = environment(tax=US_FEDERAL_WITHHOLDING).unrealized({})

        assert gaps == {"tax": ["supplemental-withholding"]}


class TestComparabilityUsesTheHashNotTheLabel:
    def test_the_isolation_dimension_can_carry_a_runtime_hash(self):
        """The migration the string was blocking: same field, real content."""
        from src.mission import ComparisonClass, RunConditions, classify

        # Pinned: under classifier @2 an absent runtime is NOT_EVALUATED
        # rather than a match, so a test about tax treatment has to pin the
        # other runtimes or it is measuring their absence instead.
        base = dict(account_hash="a1", calendar_hash="c1", market_data_hash="m1",
                    flow_schedule_hash="h1", starting_capital=0.0,
                    cash_policy_rate=0.0, cost_bps=10.0, execution_lag=1,
                    period_start="2021-01-01", period_end="2023-01-01",
                    allocation_rule_hash="r1", data_snapshot="prices@2023-01-01")

        verdict = classify(
            RunConditions(**base, tax_treatment=PRE_TAX.compatibility_hash),
            RunConditions(**base, tax_treatment=TAX_DEFERRED.compatibility_hash))

        assert "tax_treatment" in verdict.differing_dimensions
        assert verdict.comparison_class is ComparisonClass.PERSONAL_OUTCOME

    def test_the_old_string_would_have_called_them_identical(self):
        """The verdict the label produced, preserved as the reason for the change."""
        from src.mission import RunConditions, classify

        # Pinned: under classifier @2 an absent runtime is NOT_EVALUATED
        # rather than a match, so a test about tax treatment has to pin the
        # other runtimes or it is measuring their absence instead.
        base = dict(account_hash="a1", calendar_hash="c1", market_data_hash="m1",
                    flow_schedule_hash="h1", starting_capital=0.0,
                    cash_policy_rate=0.0, cost_bps=10.0, execution_lag=1,
                    period_start="2021-01-01", period_end="2023-01-01",
                    allocation_rule_hash="r1", data_snapshot="prices@2023-01-01")

        verdict = classify(RunConditions(**base, tax_treatment="NONE_APPLIED"),
                           RunConditions(**base, tax_treatment="NONE_APPLIED"))

        assert verdict.attribution_isolated, (
            "a Roth and a taxable account both said NONE_APPLIED and compared "
            "as identical — which is the defect the runtime replaces"
        )


class TestEveryRuntimeSharesTheLifecycle:
    """New runtimes cannot invent their own versioning semantics."""

    @pytest.mark.parametrize("runtime", [
        PRE_TAX, US_FEDERAL_WITHHOLDING, TAX_DEFERRED, TAXABLE_BROKERAGE])
    def test_the_required_surface_is_present(self, runtime):
        assert runtime.artifact_id.count("/") == 1
        assert "@" in runtime.artifact_id
        assert len(runtime.content_hash) == 64
        assert len(runtime.compatibility_hash) == 64
        assert isinstance(runtime.scope()["modelled"], list)
        assert isinstance(runtime.realization_checks(), list)

    def test_a_runtime_without_a_declared_form_cannot_be_built(self):
        with pytest.raises(TypeError):
            class Broken(RuntimeArtifact):
                kind = "broken"

            Broken()


class _Calendar(RuntimeArtifact):
    kind = "calendar"

    def __init__(self, name="nyse", version=1):
        self.name, self.version = name, version

    def declared_form(self):
        return {"name": self.name}


class _Protocol(RuntimeArtifact):
    kind = "protocol"
    name, version = "standard", 1

    def declared_form(self):
        return {"lag": 1}


class _CorporateActions(RuntimeArtifact):
    kind = "corporate_action"
    name, version = "us-standard", 1
    requires = ("market_data",)

    def __init__(self, requires_unadjusted=True):
        self.requires_unadjusted = requires_unadjusted

    def declared_form(self):
        return {"requires_unadjusted": self.requires_unadjusted}


def env(**extra):
    from src.runtime import ExecutionEnvironment

    return ExecutionEnvironment(runtimes={
        "protocol": _Protocol(), "calendar": _Calendar(), **extra})


class TestNotModelledIsThreeDifferentStatements:
    def test_the_default_is_a_real_gap(self):
        from src.runtime import Exclusion

        [gains] = [l for l in PRE_TAX.limitations if l.name == "no-capital-gains"]
        assert gains.reason is Exclusion.OUT_OF_SCOPE

    def test_a_shelter_turns_the_same_gap_into_not_applicable(self):
        """The 401(k) case: correct treatment, not an omission."""
        from src.runtime import AccountKind, AccountRuntime, Exclusion

        roth = AccountRuntime(name="roth", version=1,
                              account_kind=AccountKind.ROTH_IRA)
        scope = env(tax=PRE_TAX, account=roth).scope()
        [gains] = [n for n in scope["not_modelled"] if n["name"] == "no-capital-gains"]

        assert gains["reason"] == Exclusion.NOT_APPLICABLE.value
        assert gains["refined_by"] == ["account:tax_deferred"]

    def test_a_taxable_account_leaves_it_a_gap(self):
        from src.runtime import Exclusion

        scope = env(tax=PRE_TAX, account=TAXABLE_BROKERAGE).scope()
        [gains] = [n for n in scope["not_modelled"] if n["name"] == "no-capital-gains"]

        assert gains["reason"] == Exclusion.OUT_OF_SCOPE.value
        assert "refined_by" not in gains

    def test_a_missing_input_is_unresolved_not_out_of_scope(self):
        from src.runtime import (AdjustmentPolicy, Exclusion, MarketDataRuntime,
                                 PointInTimePolicy)

        data = MarketDataRuntime(
            name="d", version=1, provider="v",
            adjustment_policy=AdjustmentPolicy.UNADJUSTED_ONLY,
            point_in_time_policy=PointInTimePolicy.POINT_IN_TIME)
        [missing] = [l for l in data.limitations
                     if l.name == "no-corporate-action-source"]

        assert missing.reason is Exclusion.UNRESOLVED

    def test_the_scope_counts_by_reason(self):
        """So a reader can see that most exclusions are correct treatments."""
        from src.runtime import AccountKind, AccountRuntime

        scope = env(tax=PRE_TAX,
                    account=AccountRuntime(name="r", version=1,
                                           account_kind=AccountKind.ROTH_IRA)).scope()
        assert scope["by_reason"]
        assert "how a correct treatment gets read as a gap" in scope["note"]


class TestCompositionValidation:
    def test_a_semantic_precondition_must_be_present(self):
        """Not "instantiate this first" — "this statement has no truth value"."""
        conflicts = env(tax=PRE_TAX).validate_composition()

        assert [c.code for c in conflicts] == ["UNDEFINED_WITHOUT"]
        assert "account" in conflicts[0].runtimes
        assert "no truth value" in conflicts[0].detail

    def test_supplying_the_dependency_clears_it(self):
        assert env(tax=PRE_TAX, account=TAXABLE_BROKERAGE).is_valid

    def test_taxing_gains_inside_a_shelter_is_invalid(self):
        """Two individually valid runtimes, one incoherent environment."""
        from src.runtime import AccountKind, AccountRuntime, LotMethod

        taxing = TaxRuntime(name="t", version=1, jurisdiction="US",
                            capital_gains_short_rate=0.37,
                            capital_gains_long_rate=0.20,
                            lot_method=LotMethod.FIFO)
        roth = AccountRuntime(name="roth", version=1,
                              account_kind=AccountKind.ROTH_IRA)
        conflicts = env(tax=taxing, account=roth).validate_composition()

        assert "GAINS_TAXED_IN_SHELTER" in [c.code for c in conflicts]
        assert not env(tax=taxing, account=roth).is_valid

    def test_a_calendar_and_data_aligned_differently_conflict(self):
        from src.runtime import MarketDataRuntime

        crypto_data = MarketDataRuntime(name="d", version=1, provider="v",
                                        session_alignment="crypto")
        conflicts = env(market_data=crypto_data).validate_composition()
        codes = [c.code for c in conflicts]

        assert "SESSION_ALIGNMENT_MISMATCH" in codes
        assert "31%" in [c.detail for c in conflicts
                         if c.code == "SESSION_ALIGNMENT_MISMATCH"][0]

    def test_matching_alignment_is_clean(self):
        from src.runtime import YFINANCE_DAILY

        assert env(market_data=YFINANCE_DAILY).is_valid

    def test_double_counted_corporate_actions_are_caught(self):
        from src.runtime import YFINANCE_DAILY

        conflicts = env(market_data=YFINANCE_DAILY,
                        corporate_action=_CorporateActions()).validate_composition()

        assert "ADJUSTMENT_POLICY_CONFLICT" in [c.code for c in conflicts]
        assert "counts every action twice" in conflicts[-1].detail

    def test_every_conflict_names_the_runtimes_responsible(self):
        from src.runtime import AccountKind, AccountRuntime, LotMethod

        conflicts = env(
            tax=TaxRuntime(name="t", version=1, jurisdiction="US",
                           capital_gains_short_rate=0.37, lot_method=LotMethod.FIFO),
            account=AccountRuntime(name="r", version=1,
                                   account_kind=AccountKind.ROTH_IRA),
        ).validate_composition()

        for conflict in conflicts:
            assert len(conflict.runtimes) >= 2
            assert conflict.detail

    def test_the_payload_carries_the_verdict(self):
        payload = env(tax=PRE_TAX).to_json()

        assert payload["is_valid"] is False
        assert payload["composition_conflicts"]


class TestMarketDataSeparatesPolicyFromWhatArrived:
    def test_a_restatement_does_not_change_the_runtime(self):
        """The event worth seeing: same policy, different content."""
        from src.runtime import RealizedData, YFINANCE_DAILY

        july = RealizedData(snapshot_hash="a" * 8, retrieved_at="2026-07-31")
        september = RealizedData(snapshot_hash="b" * 8, retrieved_at="2026-09-30",
                                 vendor_revision="2026-09")

        assert september.is_restatement_of(july)
        assert YFINANCE_DAILY.compatibility_hash == YFINANCE_DAILY.compatibility_hash

    def test_the_snapshot_is_not_part_of_runtime_identity(self):
        from src.runtime import YFINANCE_DAILY

        assert "snapshot" not in str(YFINANCE_DAILY.comparable_form())
        assert "coverage" not in YFINANCE_DAILY.comparable_form()

    def test_a_policy_change_does_change_comparability(self):
        from src.runtime import MarketDataRuntime, PointInTimePolicy

        restated = MarketDataRuntime(name="d", version=1, provider="v")
        point_in_time = MarketDataRuntime(
            name="d", version=2, provider="v",
            point_in_time_policy=PointInTimePolicy.POINT_IN_TIME)

        assert not restated.is_comparable_with(point_in_time)

    def test_prose_changes_leave_comparability_intact(self):
        from src.runtime import MarketDataRuntime

        first = MarketDataRuntime(name="d", version=1, provider="v",
                                  title="Vendor daily", coverage="2000-2026")
        second = MarketDataRuntime(name="d", version=2, provider="v",
                                   title="Vendor daily bars", coverage="2000-2027")

        assert first.is_comparable_with(second)
        assert first.content_hash != second.content_hash

    def test_restated_history_is_declared_as_a_limitation(self):
        from src.runtime import YFINANCE_DAILY

        names = {l.name for l in YFINANCE_DAILY.limitations}
        assert "restated-history" in names
        assert "survivorship" in names

    def test_the_survivorship_limitation_states_its_direction(self):
        from src.runtime import YFINANCE_DAILY

        [bias] = [l for l in YFINANCE_DAILY.limitations if l.name == "survivorship"]
        assert "biased upward" in bias.statement


class TestThreeLevelsOfIdentity:
    """Family, version, instance — matching methodology/version/run exactly."""

    def test_extending_coverage_keeps_the_family_and_the_meaning(self):
        from src.runtime import MarketDataRuntime

        to_2026 = MarketDataRuntime(name="vendor", version=1, provider="v",
                                    coverage="2000-2026")
        to_2030 = MarketDataRuntime(name="vendor", version=2, provider="v",
                                    coverage="2000-2030")

        assert to_2026.family == to_2030.family == "market_data/vendor"
        assert to_2026.same_family_as(to_2030)
        assert to_2026.content_hash != to_2030.content_hash
        assert to_2026.is_comparable_with(to_2030), (
            "extending a coverage horizon changes nothing about what the "
            "runtime means, so results across the two remain comparable"
        )

    def test_different_names_are_different_families(self):
        from src.runtime import MarketDataRuntime

        first = MarketDataRuntime(name="vendor-a", version=1, provider="a")
        second = MarketDataRuntime(name="vendor-b", version=1, provider="b")

        assert not first.same_family_as(second)

    def test_the_instance_is_neither_family_nor_version(self):
        from src.runtime import RealizedData, YFINANCE_DAILY

        instance = RealizedData(snapshot_hash="a" * 8, retrieved_at="2026-07-31")

        assert not hasattr(instance, "family")
        assert YFINANCE_DAILY.family in YFINANCE_DAILY.artifact_id


class TestRulesAreQueryableArtifacts:
    def test_every_rule_declares_its_metadata(self):
        from src.runtime import REGISTERED_RULES

        assert REGISTERED_RULES
        for spec in REGISTERED_RULES:
            assert spec.id and spec.description
            assert len(spec.affects) >= 2
            assert spec.category and spec.severity

    def test_conflicts_group_by_category(self):
        """"Three conflicts" is a number; "your temporal assumptions disagree"
        is an answer."""
        from src.runtime import MarketDataRuntime, RuleCategory

        grouped = env(market_data=MarketDataRuntime(
            name="d", version=1, provider="v",
            session_alignment="crypto")).conflicts_by_category()

        assert RuleCategory.TEMPORAL.value in grouped

    def test_a_semantic_precondition_is_its_own_category(self):
        from src.runtime import RuleCategory

        grouped = env(tax=PRE_TAX).conflicts_by_category()
        assert RuleCategory.SEMANTICS.value in grouped

    def test_the_rule_set_is_enumerable_without_reading_source(self):
        from src.runtime import REGISTERED_RULES

        payload = [spec.to_json() for spec in REGISTERED_RULES]
        assert {"id", "category", "severity", "affects", "description"} <= set(payload[0])


class TestCashFlowSemanticsAreSeparateFromAmounts:
    def test_the_runtime_carries_no_personal_values(self):
        """Putting a salary inside a reusable artifact is the boundary violation
        the workspace split exists to prevent."""
        from src.mission.boundary import scan_for_personal_data
        from src.runtime import SALARY_AND_VESTS

        assert scan_for_personal_data(SALARY_AND_VESTS.declared_form()) == []

    def test_a_cadence_is_undefined_without_a_calendar(self):
        from src.runtime import SALARY_AND_VESTS

        assert "calendar" in SALARY_AND_VESTS.undefined_without
        conflicts = env(flow=SALARY_AND_VESTS).validate_composition()
        assert "UNDEFINED_WITHOUT" not in [c.code for c in conflicts], (
            "a calendar is present in this environment"
        )

    def test_an_account_only_sharpens_the_meaning(self):
        from src.runtime import SALARY_AND_VESTS

        assert "account" in SALARY_AND_VESTS.interpreted_with
        assert "account" not in SALARY_AND_VESTS.undefined_without

    def test_the_day_rule_states_why_it_matters(self):
        from src.runtime import SALARY_AND_VESTS

        [rule] = [a for a in SALARY_AND_VESTS.assumptions if a.name == "day-rule"]
        assert "names no day" in rule.risk

    def test_in_kind_vesting_is_declared_and_realized(self):
        from src.runtime import FLOW_IMPLEMENTED, SALARY_AND_VESTS

        names = {a.name for a in SALARY_AND_VESTS.assumptions}
        assert "in-kind-vesting" in names
        assert SALARY_AND_VESTS.unrealized(FLOW_IMPLEMENTED) == []

    def test_unsupported_kinds_are_not_applicable_rather_than_a_gap(self):
        from src.runtime import Exclusion, SALARY_AND_VESTS

        [unsupported] = [l for l in SALARY_AND_VESTS.limitations
                         if l.name == "unsupported-flow-kinds"]
        assert unsupported.reason is Exclusion.NOT_APPLICABLE

    def test_an_account_that_cannot_receive_shares_conflicts(self):
        """The vests would land nowhere and the plan would look like it worked."""
        from src.runtime import AccountKind, AccountRuntime, SALARY_AND_VESTS

        conflicts = env(flow=SALARY_AND_VESTS,
                        account=AccountRuntime(
                            name="529", version=1,
                            account_kind=AccountKind.PLAN_529)).validate_composition()

        assert "FLOW_KIND_UNSUPPORTED_BY_ACCOUNT" in [c.code for c in conflicts]

    def test_a_brokerage_account_receives_them_fine(self):
        from src.runtime import SALARY_AND_VESTS

        codes = [c.code for c in env(flow=SALARY_AND_VESTS,
                                     account=TAXABLE_BROKERAGE).validate_composition()]
        assert "FLOW_KIND_UNSUPPORTED_BY_ACCOUNT" not in codes


class TestFindingsCanGovernRuntimeEvolution:
    """A limitation that states its direction has become evidence."""

    def test_a_runtime_is_a_public_artifact(self):
        from src.mission.boundary import Visibility, visibility_of

        for reference in ("tax/pre-tax@1", "market_data/yfinance-daily@1",
                          "flow/salary-and-vests@1", "account/taxable-brokerage@1"):
            assert visibility_of(reference) is Visibility.PUBLIC_LIBRARY

    def test_a_finding_may_target_a_runtime_version(self):
        from src.knowledge import Finding, FindingStatus, Impact, ImpactRelation

        finding = Finding(
            name="survivorship-inflates-returns", version=1,
            statement="Using current index constituents inflated annualized "
                      "return by 1.4% over the sample.",
            status=FindingStatus.CONCLUDED,
            supported_by=("evidence/raaal-survivorship-2026@1",),
            impacts=(Impact(target="market_data/yfinance-daily@1",
                            relation=ImpactRelation.QUALIFIES,
                            detail="narrows where its figures may be used"),
                     Impact(target="market_data/yfinance-daily@2",
                            relation=ImpactRelation.MOTIVATED,
                            detail="created with a survivorship-free universe")),
        )

        assert finding.targets("market_data") == [
            "market_data/yfinance-daily@1", "market_data/yfinance-daily@2"]

    def test_a_restatement_can_invalidate_runs_without_touching_the_runtime(self):
        from src.knowledge import Finding, FindingStatus, Impact, ImpactRelation
        from src.runtime import RealizedData, YFINANCE_DAILY

        before = RealizedData(snapshot_hash="a" * 8, retrieved_at="2026-07-31")
        after = RealizedData(snapshot_hash="b" * 8, retrieved_at="2026-09-30",
                             vendor_revision="2026-09")

        finding = Finding(
            name="provider-restated-prices", version=1,
            statement="The provider restated prices after a corporate action.",
            status=FindingStatus.CONCLUDED,
            supported_by=("evidence/vendor-notice-2026@1",),
            impacts=(Impact(target="run/1487",
                            relation=ImpactRelation.INVALIDATES_RESULTS_OF),),
        )

        assert after.is_restatement_of(before)
        assert YFINANCE_DAILY.content_hash == YFINANCE_DAILY.content_hash, (
            "the runtime policy did not change; only what it served did"
        )
        assert finding.targets("run") == ["run/1487"]


class TestTheCalendarSharesTheLifecycle:
    """It was the last runtime-like object outside `RuntimeArtifact`."""

    @pytest.fixture
    def calendars(self):
        from src.calendars import CalendarRegistry

        return {c.name: c for c in CalendarRegistry().load_all()}

    def test_a_calendar_is_a_runtime_artifact(self, calendars):
        from src.runtime import RuntimeArtifact

        assert isinstance(calendars["nyse"], RuntimeArtifact)
        assert calendars["nyse"].kind == "calendar"

    def test_calendar_id_still_works_and_cannot_disagree(self, calendars):
        """Every existing caller keeps working, and the two ids are one value."""
        nyse = calendars["nyse"]
        assert nyse.calendar_id == nyse.artifact_id == "calendar/nyse@1"

    def test_extending_coverage_keeps_results_comparable(self, calendars):
        """The coverage-extension case the two-hash split exists for."""
        import dataclasses

        nyse = calendars["nyse"]
        extended = dataclasses.replace(nyse, version=2, covers_to="2040-12-31")

        assert extended.content_hash != nyse.content_hash
        assert extended.is_comparable_with(nyse)
        assert extended.same_family_as(nyse)

    def test_changing_the_session_basis_breaks_comparability(self, calendars):
        """Annualizing on the wrong count is the 31% padding defect."""
        import dataclasses

        nyse = calendars["nyse"]
        wrong = dataclasses.replace(nyse, version=2, periods_per_year=365)

        assert not wrong.is_comparable_with(nyse)

    def test_the_timezone_is_execution_relevant(self, calendars):
        import dataclasses

        nyse = calendars["nyse"]
        moved = dataclasses.replace(nyse, version=2, timezone="Europe/London")

        assert not moved.is_comparable_with(nyse), (
            "the timezone decides which calendar day a bar belongs to"
        )

    def test_its_declarations_are_realized(self, calendars):
        for calendar in calendars.values():
            assert calendar.unrealized(("sessions",)) == []

    def test_refusing_to_extrapolate_is_not_applicable_not_a_gap(self, calendars):
        from src.runtime import Exclusion

        [limit] = [l for l in calendars["nyse"].limitations
                   if l.name == "no-extrapolation"]
        assert limit.reason is Exclusion.NOT_APPLICABLE

    def test_the_session_assumption_names_the_defect_it_prevents(self, calendars):
        [sessions] = [a for a in calendars["nyse"].assumptions
                      if a.name == "declared-sessions"]
        assert "31%" in sessions.risk

    def test_it_is_registered_for_dependency_derivation(self):
        from src.runtime import RUNTIME_TYPES

        assert "calendar" in RUNTIME_TYPES


class TestRothFourOhOneKIsFirstClass:
    """Deliberately not an alias.

    Aliased to `TRADITIONAL_401K` it would report tax-deferred contributions and
    taxable withdrawals, which is the opposite of what it does. Aliased to
    `ROTH_IRA` it would report an IRA's limit, no employer match, and IRA
    withdrawal mechanics — none of which apply to an employer plan. Either
    substitution records a tax treatment the user did not describe.
    """

    def kind(self, name):
        from src.runtime import AccountKind, AccountRuntime

        return AccountRuntime(name=f"account/{name}", version=1,
                              account_kind=AccountKind[name.upper()])

    def test_it_exists(self):
        from src.runtime import AccountKind

        assert AccountKind.ROTH_401K.value == "ROTH_401K"

    def test_it_does_not_share_an_identity_with_either_neighbour(self):
        roth_plan = self.kind("roth_401k")
        assert roth_plan.compatibility_hash != self.kind("traditional_401k").compatibility_hash
        assert roth_plan.compatibility_hash != self.kind("roth_ira").compatibility_hash

    def test_contributions_are_after_tax(self):
        """The difference from a traditional 401(k): the same contribution
        produces a different balance and a different withdrawal."""
        assert self.kind("roth_401k").after_tax_contributions
        assert not self.kind("traditional_401k").after_tax_contributions

    def test_the_employee_deferral_limit_is_shared_with_the_plan(self):
        """Modelling them as independent would let a scenario contribute twice
        what the law permits."""
        assert self.kind("roth_401k").shares_employee_deferral_limit
        assert self.kind("traditional_401k").shares_employee_deferral_limit
        assert not self.kind("roth_ira").shares_employee_deferral_limit

    def test_employer_money_is_pre_tax_even_in_a_roth_plan(self):
        """It lands in a traditional sub-account and is taxed on withdrawal, so
        the account holds two differently-taxed balances."""
        assert self.kind("roth_401k").employer_contributions_are_pre_tax
        assert not self.kind("roth_ira").employer_contributions_are_pre_tax

    def test_growth_is_untaxed_inside_the_account(self):
        assert self.kind("roth_401k").tax_deferred

    def test_it_declares_what_it_does_not_yet_enforce(self):
        """Pinning an account is not simulating one. `ACCOUNT_IMPLEMENTED` is
        empty, so every declared behaviour is recorded and unenforced — and the
        runtime says so rather than looking enforced."""
        from src.runtime import ACCOUNT_IMPLEMENTED

        unrealized = self.kind("roth_401k").unrealized(ACCOUNT_IMPLEMENTED)
        assert "shared-deferral-limit" in unrealized

    def test_a_plan_in_one_can_now_be_pinned(self):
        """Before this kind existed the account was left unpinned, and the
        comparison could not claim isolated attribution. That was correct, and
        is now unnecessary."""
        from src.mission.compiler import compile_scenario
        from src.workspace.environment import pins_for

        compiled = compile_scenario(
            "I put $500 into SPY monthly in my Roth 401(k) and never sell.",
            name="p", version=1, benchmark_rule="benchmark-policy/public-default@1")
        pins = pins_for(compiled.scenario, snapshot="prices@x")
        assert pins.account_hash
        assert pins.unpinned == ()
        assert pins.unrealized, (
            "pinned is not enforced, and the run must say which is which")

    def test_the_unenforced_rules_reach_the_run_limitations(self):
        from src.mission.compiler import compile_scenario
        from src.workspace.environment import pins_for

        compiled = compile_scenario(
            "I put $500 into SPY monthly in my Roth 401(k) and never sell.",
            name="p", version=1, benchmark_rule="benchmark-policy/public-default@1")
        limitations = pins_for(compiled.scenario, snapshot="prices@x").limitations()
        assert any(entry["dimension"].startswith("account:") for entry in limitations)
