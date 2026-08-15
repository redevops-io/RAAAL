"""Step 1's exit gate, as five properties rather than five opinions.

    zero interpretation branches in `from_intent`
    zero silent execution-changing defaults
    zero hardcoded values that change a figure or select data
    a stated value this build cannot execute refuses, never substitutes
    same meaning -> same spec, different execution meaning -> different spec

The last pair is the one that proves the boundary is about meaning rather than
wording, and it needs both halves. Equivalence alone passes trivially if the
compiler is lossy in the same direction for both utterances — two sentences
that both compile to nothing are byte-identical. The converse is what makes it
discriminating, so the negatives are not extras here; they are the evidence.
"""
from __future__ import annotations

import json
from hashlib import sha256

import pytest
from runtime_contracts import Author, IntentField, VerifiedIntent

from src.discovery.canonical import canonicalise
from src.mission.from_intent import (DEFAULTS, ENGINE_CONSTANTS,
                                     compile_intent)

RULE = "the-benchmark-rule"


def compiled(**stated):
    """One utterance's worth of stated values, through the real path.

    Canonicalised then sealed then compiled, which is the order production
    uses. Building an intent by hand would skip the layer under test and prove
    something about a shape nobody produces.
    """
    canonical = canonicalise(stated)
    intent = VerifiedIntent(
        objective="evaluate_investment_strategy", produced_by="test",
        utterance_ref="u",
        fields={name: IntentField(value=value, author=Author.MODEL)
                for name, (value, _author) in canonical.fields.items()},
        unresolved=()).seal()
    return compile_intent(intent, benchmark_rule=RULE), canonical.refusals


def spec_digest(**stated) -> str:
    """The execution identity, byte-for-byte."""
    out, unreadable = compiled(**stated)
    assert not unreadable, f"not canonicalisable: {unreadable}"
    assert out.scenario is not None, [r.detail for r in out.refusals]
    return sha256(json.dumps(out.scenario.canonical_form(), sort_keys=True,
                             default=str).encode()).hexdigest()


BASE = {"assets": "VTI", "amount": "1000", "cadence": "monthly"}


class TestSameMeaningSameSpec:
    """Different wording, one execution identity."""

    @pytest.mark.parametrize("said,also", [
        ("$1,000", "1000 usd"),
        ("$1,000", "1000"),
        ("1000 dollars", "1,000"),
    ])
    def test_a_figure_spelled_differently(self, said, also):
        assert spec_digest(**{**BASE, "amount": said}) \
            == spec_digest(**{**BASE, "amount": also})

    @pytest.mark.parametrize("said,also", [
        ("monthly", "every month"),
        ("annual", "each year"),
        ("biweekly", "every two weeks"),
    ])
    def test_a_cadence_spelled_differently(self, said, also):
        assert spec_digest(**{**BASE, "cadence": said}) \
            == spec_digest(**{**BASE, "cadence": also})

    @pytest.mark.parametrize("said,also", [
        ("VTI and BND", "VTI, BND"),
        ("VTI and BND", "BND and VTI"),
        ("VTI; BND", "VTI,BND"),
    ])
    def test_holdings_named_differently(self, said, also):
        assert spec_digest(**{**BASE, "assets": said}) \
            == spec_digest(**{**BASE, "assets": also})


class TestDifferentExecutionMeaningDifferentSpec:
    """The half that makes the half above mean anything.

    Without these, a compiler that dropped cadence entirely would pass every
    equivalence test in this file.
    """

    def test_monthly_is_not_annual(self):
        assert spec_digest(**{**BASE, "cadence": "monthly"}) \
            != spec_digest(**{**BASE, "cadence": "annual"})

    def test_a_different_amount_is_a_different_plan(self):
        assert spec_digest(**{**BASE, "amount": "1000"}) \
            != spec_digest(**{**BASE, "amount": "2000"})

    def test_different_holdings_are_a_different_plan(self):
        assert spec_digest(**{**BASE, "assets": "VTI"}) \
            != spec_digest(**{**BASE, "assets": "VTI,BND"})

    def test_selling_and_never_selling_are_different_outcomes(self):
        """Polarity reaches execution and decides whether there is a plan.

        Not two specs: this build only buys, so a stated disposal is refused by
        name and a denied one compiles. That *is* the difference, and it is the
        one the negation work exists for — the same sentence used to be refused
        for describing the engine's own behaviour exactly.

        The tempting pair, never-sell against a rebalancing plan, cannot be
        written: this build does not execute stated rebalancing either, so both
        sides would refuse and the test would pass on a compiler that ignored
        polarity completely.
        """
        sells, _ = compiled(**{**BASE, "sell_action": "sell half of it in May"})
        never, _ = compiled(**{**BASE, "sell_action": "never sold any of it"})

        assert not sells.executable, "this build sold something"
        assert any(r.dimension == "sell_action" for r in sells.refusals)
        assert never.executable, [r.detail for r in never.refusals]
        assert never.scenario.holdings_policy.sells_allowed is False

    def test_a_denied_disposal_matches_the_engine_rather_than_changing_it(self):
        """And the honest converse: it does *not* change the spec.

        A build that never sells executes "never sell" and silence identically,
        so the specs are equal and must be — a StrategySpec carries execution
        semantics, and this difference is provenance. Asserted rather than left
        implicit, because the natural next move is to make the two differ, and
        that would put a statement about the user into the execution identity.
        """
        assert spec_digest(**{**BASE, "sell_action": "never sold any of it"}) \
            == spec_digest(**BASE)

    def test_reinvesting_is_not_the_same_as_holding_dividends_as_cash(self):
        """The finding that made `dividend_policy` part of identity.

        It was an ENGINE_CONSTANT described as changing no figure. It selects
        which price series is resolved — total-return against price-return —
        so the two are materially different strategies over a long horizon, and
        the choice now travels in the delivery record.

        This build executes only one of them, so the difference shows as a
        refusal by name rather than as a second figure. That is the honest
        outcome and still a difference: what must never happen is both
        producing the same plan.
        """
        reinvested, _ = compiled(**{**BASE, "dividend_policy": "reinvested"})
        as_cash, _ = compiled(**{**BASE, "dividend_policy": "held as cash"})

        assert reinvested.executable
        assert not as_cash.executable, (
            "a plan that pays distributions out compiled to the same thing as "
            "one that reinvests them, which is a better-performing strategy "
            "than the one described")
        assert any(r.dimension == "dividend_policy" for r in as_cash.refusals)

    def test_the_policy_is_in_the_execution_identity(self):
        out, _ = compiled(**BASE)
        body = json.dumps(out.scenario.canonical_form(), sort_keys=True,
                          default=str)
        assert "reinvested" in body, (
            "the dividend policy is absent from the canonical form, so two "
            "plans the evaluator would price differently share a hash")


class TestNoSilentExecutionChangingDefault:
    """Every default a plan runs on is named where a person can see it."""

    @pytest.mark.parametrize("dimension", sorted(DEFAULTS))
    def test_each_default_is_reported_when_it_applies(self, dimension):
        out, _ = compiled(**{"assets": "VTI"})
        if dimension in ("moving_average_window", "execution_timing"):
            # Trigger-only. A plan with no trigger never consults them, and
            # reporting a default nothing used would be the opposite defect.
            pytest.skip("consulted only by an event-triggered plan")
        assert dimension in out.applied_defaults, (
            f"{dimension} was supplied by the engine and does not appear in "
            "applied_defaults, so the plan runs on a value nobody chose and "
            "nobody can find")

    def test_the_two_that_were_silent_are_now_named(self):
        """`cadence` and `dividend_policy`, the pair this gate was written for.

        `cadence` fell out of `_funding` as `or "once"` and `dividend_policy`
        was an ENGINE_CONSTANT. A plan stating only its holdings ran once, on a
        total-return series, and reported neither.
        """
        out, _ = compiled(**{"assets": "VTI"})
        assert "cadence" in out.applied_defaults
        assert "dividend_policy" in out.applied_defaults
        assert out.scenario.funding.cadence == "once"


class TestNothingHardcodedThatMovesAFigure:
    def test_engine_constants_hold_nothing_that_selects_data(self):
        """`tax_treatment` alone, and it is not a figure.

        The engine computes no tax, so this changes nothing and is recorded
        only so two strategies can be told apart. `dividend_policy` used to sit
        beside it under the same claim, while choosing the price series.
        """
        assert set(ENGINE_CONSTANTS) == {"tax_treatment"}

    def test_the_resolution_request_follows_the_plan_and_not_a_constant(self):
        from src.workspace.run_boundary import _reinvests

        out, _ = compiled(**BASE)
        assert _reinvests(out.scenario) is True

        # The same code path, asked about a plan that says otherwise. Built
        # directly because this build refuses to compile one — the point is
        # that the market-data request reads the plan rather than a constant.
        from dataclasses import replace as _replace

        held = _replace(out.scenario, holdings_policy=_replace(
            out.scenario.holdings_policy, dividend_policy="held_as_cash"))
        assert _reinvests(held) is False, (
            "the market-data request ignores the plan's dividend policy, so "
            "every run resolves a total-return frame whatever it says")


class TestAStatedValueThisBuildCannotExecuteRefuses:
    """Never substitutes. Each of these silently became something else."""

    def test_an_unreadable_amount(self):
        _out, unreadable = compiled(**{**BASE, "amount": "a portion"})
        assert any(name == "amount" for name, _why in unreadable)

    def test_an_unreadable_cadence(self):
        _out, unreadable = compiled(**{**BASE, "cadence": "when I feel like it"})
        assert any(name == "cadence" for name, _why in unreadable)

    def test_an_execution_timing_this_build_has_no_path_for(self):
        """`_timing` answered anything unrecognised with `next_session_open`.

        A substitution on a *stated* value, which is worse than one on an
        absent field: somebody asked for an execution this build cannot perform
        and was given a different one, silently.
        """
        out, _ = compiled(**{**BASE, "trigger_semantics": "crossing_event",
                             "observed_assets": "SPY",
                             "moving_average_window": "200",
                             "execution_timing": "at the close of the third Friday"})
        assert not out.executable
        assert any(r.dimension == "execution_timing" for r in out.refusals)

    def test_more_than_one_watched_asset(self):
        """`_funding` took the first of the list. Watching SPY when somebody
        named SPY and QQQ is a different rule under the same name."""
        out, _ = compiled(**{**BASE, "trigger_semantics": "crossing_event",
                             "observed_assets": "SPY and QQQ",
                             "moving_average_window": "200"})
        assert not out.executable
        assert any(r.dimension == "observed_assets" for r in out.refusals)

    def test_never_selling_while_asking_to_rebalance(self):
        out, _ = compiled(**{**BASE, "sell_action": "never sold any of it",
                             "periodic_rebalancing": "annually"})
        assert not out.executable
        assert any(r.dimension == "sells_allowed" for r in out.refusals)


class TestTheNegatedDisposalStillCompiles:
    """The sentence the whole negation path exists for.

    It broke while this was being written, and the suite did not notice:
    sealing `sells_allowed` put a field in the intent that no builder consumed,
    so the stranded-dimension check refused it. Then the fix broke on a
    short-circuit — `bool(rebalances) and value("sells_allowed")` never calls
    `value` on a plan that does not rebalance, which is every ordinary plan.
    """

    def test_it_runs(self):
        out, _ = compiled(**{**BASE, "sell_action": "never sold any of it"})
        assert out.executable, [r.detail for r in out.refusals]
        assert out.scenario.holdings_policy.sells_allowed is False
