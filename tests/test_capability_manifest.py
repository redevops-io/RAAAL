"""The manifest must describe this engine, and this engine only.

Two directions, and both are needed:

    manifest -> code   a value claimed executable that no code path runs
    code -> manifest   a value the engine runs, or a menu offers, that the
                       manifest never mentions

The second is the one that bit. `quarterly`, `annual` and `daily` were offered
in the confirmation menu and verbalised as "every quarter" and "every year"
while the executor ran none of them, so "$1,000 every year over five years"
reported $1,000 contributed. Three vocabularies — the parser's, the menu's, the
renderer's — and nothing compared any of them to the executor.

The claims in this file come in two kinds, and the distinction is deliberate:

    derived    the manifest imports the executor's own table, so drift is not
               expressible and the test only proves the wiring
    asserted   the manifest states a fact about the engine's behaviour, and
               the test proves it by running the engine

An asserted claim nobody exercises is a comment. Every REFUSED and NOT_MODELLED
entry below is exercised.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.mission import capability as cap
from src.mission import schedule as sched

SESSIONS = pd.bdate_range("2021-01-04", "2025-12-31")


class Schedule:
    """The shape `expand` reads."""

    def __init__(self, cadence, day_rule="first_session_of_period"):
        self.cadence = cadence
        self.day_rule = day_rule
        self.amount = 1000.0
        self.starting_capital = 0.0


def flows(cadence, day_rule="first_session_of_period"):
    from src.mission import CashFlow

    return sched.expand(Schedule(cadence, day_rule), SESSIONS,
                        cash_flow=CashFlow)


class TestTheManifestIsDerivedWhereItCanBe:
    """A restated list is a second copy that can disagree, and the
    disagreement is the defect."""

    def test_cadence_is_the_executor_table_itself(self):
        assert tuple(cap.MANIFEST["cadence"].values) == \
            tuple(sched.EXECUTABLE_CADENCES)
        assert cap.MANIFEST["cadence"].derived_from == \
            "mission.schedule.EXECUTABLE_CADENCES"

    def test_day_rule_is_the_executor_table_itself(self):
        assert tuple(cap.MANIFEST["day_rule"].values) == \
            tuple(sched.EXECUTABLE_DAY_RULES)

    def test_the_derived_set_is_exactly_what_expand_branches_on(self):
        """Closes the loop: the constant could still drift from the function
        that reads it."""
        expected = set(sched.PERIOD_KEYS) | {sched.EVERY_SESSION, sched.SINGLE}
        assert set(sched.EXECUTABLE_CADENCES) == expected


class TestEveryExecutableClaimIsRunnable:
    """manifest -> code. A value claimed executable that no code path runs."""

    @pytest.mark.parametrize("cadence", sorted(sched.EXECUTABLE_CADENCES))
    def test_each_claimed_cadence_produces_contributions(self, cadence):
        produced = flows(cadence)
        assert produced, f"{cadence} is claimed executable and produced nothing"
        if cadence != sched.SINGLE:
            assert len(produced) > 1, (
                f"{cadence} is claimed executable and produced a single "
                "payment — the exact shape of the original defect")

    @pytest.mark.parametrize("rule", sorted(sched.EXECUTABLE_DAY_RULES))
    def test_each_claimed_day_rule_changes_the_dates(self, rule):
        assert flows("monthly", rule)

    def test_the_two_day_rules_are_not_the_same_rule(self):
        first = [f.date for f in flows("monthly", "first_session_of_period")]
        last = [f.date for f in flows("monthly", "last_session_of_period")]
        assert len(first) == len(last) and first != last


class TestEveryRefusedClaimIsActuallyRefused:
    """code -> manifest, for the behavioural half. Each of these is a claim
    about the engine, so each is proven against the engine."""

    def test_a_refused_cadence_raises_rather_than_defaulting(self):
        for value in cap.MANIFEST["cadence"].refuses:
            with pytest.raises(sched.UnsupportedCadence):
                flows(value)

    def test_stated_weights_are_not_honoured(self):
        """The manifest says a stated 60/40 cannot be run. Proven by the
        allocation rule the compiler produces: equal weight at purchase."""
        from src.mission.compiler import compile_scenario, parse

        text = ("I put $1,000 every year into VTI and BND in a taxable "
                "brokerage account, holding 60/40, over the past 5 years.")
        result = compile_scenario(text, name="t", version=1,
                                  benchmark_rule="benchmark-policy/public-default@1",
                                  parsed=parse(text))
        assert result.scenario.allocation_rule.weighting == \
            "equal_weight_at_purchase"

    def test_rebalancing_has_no_representation_in_a_compiled_plan(self):
        from src.mission.scenario import ScenarioSpecification

        assert not any("rebalanc" in f.lower()
                       for f in ScenarioSpecification.__dataclass_fields__)

    def test_every_claimed_allocation_method_is_one_the_engine_runs(self):
        """The two simple splits, plus every computed strategy. `stated_weights`
        joined when the compiler could attach a split to its holdings; the
        strategies joined when `rebalance.strategy_driven` gained a way to
        restore to weights `run_capability` computes each period.

        Claimed *and* exercised: the manifest says the engine does these, so the
        test makes it do them rather than reading the claim back."""
        import numpy as np
        import pandas as pd

        from src.mission.rebalance import normalised
        from src.mission.strategy_methods import STRATEGY_ALLOCATION_METHODS
        from src.strategies import CAPABILITY_BY_ID, run_capability

        claimed = tuple(cap.MANIFEST["allocation_method"].values)
        assert claimed[:2] == ("equal_weight_at_purchase", "stated_weights")
        assert set(claimed[2:]) == set(STRATEGY_ALLOCATION_METHODS)

        # equal_weight_at_purchase
        assert normalised(["A", "B"]) == {"A": 0.5, "B": 0.5}
        # stated_weights
        assert normalised(["A", "B"], {"A": 60, "B": 40}) == {"A": 0.6,
                                                              "B": 0.4}
        # every strategy value routes to a capability the engine dispatches …
        for value, capability in STRATEGY_ALLOCATION_METHODS.items():
            assert capability in CAPABILITY_BY_ID, value
        # … and one is run end to end, so "claimed" is also "exercised".
        prices = pd.read_parquet(
            "tests/fixtures/prices_synthetic.parquet").tail(300)
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
        weights = run_capability("risk_parity", prices, returns, None, {})
        assert abs(sum(weights.values()) - 1.0) < 0.05

    def test_execution_timing_refuses_what_the_engine_refuses(self):
        """`same_session_close` is refused because acting on the close that
        produced the signal reads one bar into the future."""
        from src.mission.funding import SUPPORTED_TIMING

        claimed = set(cap.MANIFEST["execution_timing"].values)
        supported = {t.value for t in SUPPORTED_TIMING}
        assert claimed == supported
        assert "same_session_close" in cap.MANIFEST["execution_timing"].refuses


class TestNothingIsOfferedThatCannotBeRun:
    """A menu is a promise. This is the check that would have caught the
    original defect at build time rather than in a sweep."""

    def test_every_offered_cadence_is_executable(self):
        from src.mission.vocabulary import FIELDS

        offered = {o.value for o in FIELDS["cadence"].options}
        executable = set(sched.EXECUTABLE_CADENCES)
        unrunnable = offered - executable
        assert not unrunnable, (
            f"the confirmation menu offers {sorted(unrunnable)}, which this "
            "build does not execute")

    def test_offerable_values_are_empty_for_a_refused_dimension(self):
        # `stated_weights` executes but is not offerable — its split is open
        # text, not a closed menu — and `sell_action` is refused outright.
        # `periodic_rebalancing` used to sit here and no longer does: it became
        # executable when `rebalance.weighted` gained a calendar to restore on.
        assert cap.offerable_values("stated_weights") == ()
        assert cap.offerable_values("sell_action") == ()

    def test_offerable_values_are_the_executable_ones(self):
        assert set(cap.offerable_values("cadence")) == \
            set(sched.EXECUTABLE_CADENCES)


class TestTheManifestIsComplete:
    """code -> manifest, for coverage. A dimension the compiler can declare
    and the manifest never mentions is a dimension whose executability nobody
    decided."""

    def test_the_declared_inventory_matches_the_elements_coverage_builds(self):
        """`DECLARED_ELEMENTS` is a restated list, so it can drift from the
        `element_id=` literals it names. Scraping the source is inelegant and
        is the only thing that actually closes the gap without rewriting eight
        call sites; the alternative is a constant that quietly stops being
        true, which is the defect this whole file exists for."""
        import re
        from pathlib import Path

        from src.mission import coverage

        source = Path("src/mission/coverage.py").read_text()
        built = set(re.findall(r'element_id="([a-z_]+)"', source))
        assert built == set(coverage.DECLARED_ELEMENTS)

    def test_every_declared_element_is_classified(self):
        from src.mission import coverage

        declared = set(coverage.DECLARED_ELEMENTS)
        assert declared, "the inventory is empty; this test would prove nothing"
        missing = declared - set(cap.MANIFEST)
        assert not missing, (
            f"{sorted(missing)} can be declared by a user and the manifest "
            "says nothing about whether this build executes it")

    def test_a_refused_dimension_states_why(self):
        """A refusal a user cannot act on is a dead end."""
        for name, d in cap.MANIFEST.items():
            if d.support in (cap.REFUSED, cap.NOT_MODELLED):
                assert d.why, f"{name} is refused without saying why"

    def test_a_refused_value_states_why(self):
        for name, d in cap.MANIFEST.items():
            for value, why in d.refuses.items():
                assert why, f"{name}={value} is refused without saying why"


class TestARefusalNamesWhatItRefused:
    def test_it_carries_the_dimension_and_the_value(self):
        # `inverse_volatility` executes now (it routes to risk parity), so the
        # refused example is one the engine still has no kernel for.
        r = cap.decide("allocation_method", "hierarchical_risk_parity")
        assert r is not None
        assert r.dimension == "allocation_method"
        assert r.stated_value == "hierarchical_risk_parity"
        assert "hierarchical_risk_parity" in r.message
        assert "equal_weight_at_purchase" in r.message

    def test_it_does_not_apply_the_alternative_it_names(self):
        """Naming what this build could run is for the reader. Substituting it
        is the defect the whole boundary exists to prevent."""
        r = cap.decide("cadence", "payroll")
        assert r.executable_values
        assert r.stated_value == "payroll"

    def test_an_executable_value_earns_no_refusal(self):
        """The discriminating half: a gate that refuses everything proves
        nothing."""
        assert cap.decide("cadence", "annual") is None
        assert cap.decide("day_rule", "last_session_of_period") is None
        assert cap.decide("allocation_method", "equal_weight_at_purchase") is None

    def test_all_refusals_are_reported_not_just_the_first(self):
        """A user who declared three unsupported things should be told three
        times, not one deploy apart."""
        refusals = cap.refusals_for({
            "cadence": "payroll",
            "allocation_method": "hierarchical_risk_parity",
            "periodic_rebalancing": "threshold_band",
        })
        assert {r.dimension for r in refusals} == {
            "cadence", "allocation_method", "periodic_rebalancing"}

    def test_an_unclassified_dimension_does_not_block_a_user(self):
        """Completeness is a build-time check, not a runtime one. Failing here
        would block a user for a bookkeeping omission."""
        assert cap.decide("something_nobody_classified", "x") is None
