"""A stated allocation method and a stated rebalancing schedule must not vanish.

    Allocate $100,000 across VTI, BND and GLD by inverse volatility,
    rebalanced monthly, past 5 years.

After the contribution-cadence defect was fixed, this prompt reported
`$100,000` contributed — the right number — for a strategy that was still not
the one described. `by inverse volatility` had become
`equal_weight_at_purchase` and was recorded as an **inference**, so a value the
user stated was replaced by a default and labelled as the system's own choice.
`rebalanced monthly` reached nothing at all: `holdings_policy` carries
`rebalancing_allowed` as a boolean and no frequency, so there is nowhere for
"monthly" to go.

Coverage said **1/1, publishable**, because the only declared element it could
see was the evaluation period.

That is the shape the roadmap already predicts:

    user declares a dimension
    the compiler cannot represent it
    the declaration disappears
    the denominator never sees it
    the figure is allowed

Two elements, not one, because they fail independently: a plan may state a
weighting scheme without a schedule, or a schedule without a scheme.

**Why fix this before the general abstraction.** A compiler-derived inventory
is the right answer and is on the roadmap with a trigger. It is not a reason to
show a user a figure for a strategy that is not theirs in the meantime — and
the bounded behaviour is one sentence: this version does not model that, so no
result is shown.
"""
from __future__ import annotations

import pytest

BOTH = ("Allocate $100,000 across VTI, BND and GLD by inverse volatility, "
        "rebalanced monthly, past 5 years.")
WEIGHTING_ONLY = ("Allocate $100,000 across VTI and BND by risk parity, "
                  "past 5 years.")
REBALANCING_ONLY = ("Allocate $100,000 equally across VTI and BND, "
                    "rebalanced quarterly, past 5 years.")

#: Plans that state neither. If these blocked, the gate would be refusing the
#: product's ordinary output rather than its unsupported edges.
CONTROLS = (
    "I buy $1,000 of SPY whenever it crosses below its 200-day moving "
    "average, over the past five years.",
    "I put $500 into VTI every month for the past five years.",
)


@pytest.fixture
def deployment(monkeypatch):
    from src.deploy.context import bind, resolve, unbind

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    try:
        yield
    finally:
        unbind()


def run(text):
    import src.workspace.routes as routes
    from src.workspace.draft import compile_draft

    scenario = compile_draft(text, name="p", context="alloc test").scenario
    return scenario, routes._run(scenario, routes._market_data("alloc test"),
                                 stated_text=text)


def coverage(text):
    import src.mission.coverage as coverage_module
    import src.workspace.routes as routes

    scenario, executed = run(text)
    access = routes._market_data("alloc test")
    return coverage_module.assess(
        scenario, stated_text=text,
        resolved_window=routes._resolve_window(scenario, access.frame),
        frame_sessions=len(access.frame), ledger=executed.get("ledger"))


class TestThePremise:
    def test_the_compiler_cannot_express_either(self, deployment):
        """If it could, these would be execution tests rather than coverage
        tests, and blocking would be the wrong answer."""
        scenario, _ = run(BOTH)
        assert scenario.allocation_rule.weighting == "equal_weight_at_purchase"
        assert not hasattr(scenario.holdings_policy, "rebalancing_frequency")

    def test_the_contribution_total_is_already_correct(self, deployment):
        """The earlier defect stays fixed. This file is about what is still
        wrong once the number is right — a correct total for the wrong
        strategy is the more dangerous of the two."""
        scenario, _ = run(BOTH)
        assert float(scenario.flow_schedule.amount) == 100000.0
        assert scenario.flow_schedule.cadence == "once"


class TestEachIsCountedAndBlocks:
    @pytest.mark.parametrize("text,element", [
        (BOTH, "allocation_method"),
        (BOTH, "periodic_rebalancing"),
        (WEIGHTING_ONLY, "allocation_method"),
        (REBALANCING_ONLY, "periodic_rebalancing"),
    ])
    def test_it_is_declared(self, deployment, text, element):
        found = {one.element_id: one for one in coverage(text).elements}
        assert found[element].declared, (
            f"{element} was stated and coverage does not know it was declared")

    @pytest.mark.parametrize("text", [BOTH, WEIGHTING_ONLY, REBALANCING_ONLY])
    def test_no_figure_is_published(self, deployment, text):
        _scenario, executed = run(text)
        assert executed["result"] is None, (
            "a figure was published for a strategy this build cannot model")

    @pytest.mark.parametrize("text,phrase", [
        (WEIGHTING_ONLY, "divides each purchase equally"),
        (REBALANCING_ONLY, "without rebalancing"),
    ])
    def test_the_refusal_says_what_was_substituted(self, deployment, text,
                                                   phrase):
        """Naming the element is not enough; a user needs to know what the
        engine would have done instead, because that is the difference between
        "rephrase this" and "this product cannot answer my question"."""
        assert phrase in coverage(text).refusal()

    def test_they_are_independent(self, deployment):
        """A plan stating one and not the other must block on that one alone —
        otherwise a single combined element would pass every test above."""
        weighting = {one.element_id: one
                     for one in coverage(WEIGHTING_ONLY).elements}
        assert weighting["allocation_method"].declared
        assert not weighting["periodic_rebalancing"].declared

        rebalancing = {one.element_id: one
                       for one in coverage(REBALANCING_ONLY).elements}
        assert rebalancing["periodic_rebalancing"].declared
        assert not rebalancing["allocation_method"].declared


class TestTheUserMayProceedWithout:
    @pytest.mark.parametrize("element", ["allocation_method",
                                         "periodic_rebalancing"])
    def test_an_authorised_exclusion_unblocks(self, deployment, element):
        import src.mission.coverage as coverage_module
        import src.workspace.routes as routes

        scenario, executed = run(BOTH)
        access = routes._market_data("alloc test")
        record = coverage_module.assess(
            scenario, stated_text=BOTH,
            resolved_window=routes._resolve_window(scenario, access.frame),
            frame_sessions=len(access.frame), ledger=executed.get("ledger"),
            excluded_items=("allocation_method", "periodic_rebalancing"))
        found = {one.element_id: one for one in record.elements}
        assert found[element].state.value == "EXCLUDED_BY_USER"
        assert not found[element].executed


class TestTheOrdinaryPlansAreUntouched:
    @pytest.mark.parametrize("text", CONTROLS)
    def test_a_plan_stating_neither_still_publishes(self, deployment, text):
        _scenario, executed = run(text)
        assert executed["result"] is not None, executed.get("unavailable")

    @pytest.mark.parametrize("text", CONTROLS)
    def test_neither_element_is_declared(self, deployment, text):
        found = {one.element_id: one for one in coverage(text).elements}
        assert not found["allocation_method"].declared
        assert not found["periodic_rebalancing"].declared

    def test_a_supported_equal_weighting_is_not_caught(self, deployment):
        """"Equally" is a weighting this build does express. Catching it would
        refuse a plan the engine executes correctly."""
        text = "Allocate $100,000 equally across VTI and BND, past 5 years."
        found = {one.element_id: one for one in coverage(text).elements}
        assert not found["allocation_method"].declared
        _scenario, executed = run(text)
        assert executed["result"] is not None


class TestStatedWeights:
    """Per-asset percentages the user wrote.

        I hold 60% VTI and 40% BND, past 5 years, starting with $50,000.

    executed 50/50 and published a figure. `AllocationRule` holds `assets` and
    a `weighting` enum and no per-asset weights at all, so the numbers had
    nowhere to go. Over five years the difference between a 60/40 and a 50/50
    equity/bond split is not a rounding error, and this is a far more ordinary
    way to describe a portfolio than "inverse volatility".

    **Equal percentages are not an unsupported allocation.** "50% VTI and 50%
    BND" is the supported weighting written as numbers. Blocking it because
    percentages were used would refuse a plan the engine executes correctly —
    the over-reach this area has walked back twice. It compiles to
    `equal_weight_at_purchase` and is recorded as *stated*, because the user
    said it; calling it the compiler's own inference is the authority
    inversion found elsewhere in this branch.

    Percentages are read as an allocation only when they sum to about 100. A
    percentage in a sentence is usually a threshold — "sell if it drops 10%" —
    and thresholds do not add up to a portfolio.
    """

    UNEQUAL = "I hold 60% VTI and 40% BND, past 5 years, starting with $50,000."
    EQUAL = "I hold 50% VTI and 50% BND, past 5 years, starting with $50,000."
    WORDED = "Allocate $100,000 equally across VTI and BND, past 5 years."
    THRESHOLDS = ("I invest $1,000 a month into VTI, and sell if it drops 10% "
                  "and buy if it rises 15%, past 5 years.")

    def test_unequal_weights_block_by_name(self, deployment):
        _scenario, executed = run(self.UNEQUAL)
        assert executed["result"] is None
        refusal = coverage(self.UNEQUAL).refusal()
        assert "60%" in refusal and "40%" in refusal, refusal
        assert "divides each purchase equally" in refusal

    def test_equal_weights_execute(self, deployment):
        """The case that must not be caught."""
        _scenario, executed = run(self.EQUAL)
        assert executed["result"] is not None, executed.get("unavailable")

    def test_equal_weights_are_stated_not_inferred(self, deployment):
        """The equivalence has to be recorded as the user's, not ours."""
        scenario, _ = run(self.EQUAL)
        assert scenario.allocation_rule.weighting == "equal_weight_at_purchase"
        assert "weighting" not in [one.field
                                   for one in scenario.provenance.inferred]

    def test_the_worded_form_still_executes(self, deployment):
        _scenario, executed = run(self.WORDED)
        assert executed["result"] is not None, executed.get("unavailable")

    def test_thresholds_are_not_read_as_weights(self, deployment):
        """Two percentages in a sentence are usually not a portfolio. This
        plan blocks for its conditional purchase, which is correct, and must
        not also be accused of stating an allocation."""
        from src.mission.compiler import stated_weights

        assert stated_weights(self.THRESHOLDS) == ()
        found = {one.element_id: one
                 for one in coverage(self.THRESHOLDS).elements}
        assert not found["stated_weights"].declared

    @pytest.mark.parametrize("text,expected", [
        ("I hold 60% VTI and 40% BND.", (60.0, 40.0)),
        ("I hold 50% VTI and 50% BND.", (50.0, 50.0)),
        ("30% VTI, 30% BND and 40% GLD.", (30.0, 30.0, 40.0)),
        ("sell if it drops 10% and buy if it rises 15%", ()),
        ("after the market falls more than 20%", ()),
        ("Allocate equally across VTI and BND.", ()),
    ])
    def test_what_counts_as_an_allocation(self, text, expected):
        from src.mission.compiler import stated_weights

        assert stated_weights(text) == expected, text

    def test_a_three_way_uneven_split_blocks(self, deployment):
        text = "I hold 30% VTI, 30% BND and 40% GLD, past 5 years, starting with $50,000."
        _scenario, executed = run(text)
        assert executed["result"] is None
