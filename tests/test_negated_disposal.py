"""Saying you never sell is not selling.

The launch journey caught this on a host, at the first deployment that served
the pilot interpreter. "I put $500 a month into VTI starting in January 2015 and
never sold any of it" came back refused for `sell_action` — on a build whose
entire behaviour is buying and never selling. The reader had done its job: the
span it extracted was "never sold any of it". `decide()` refuses any value of a
refused dimension, so the polarity never reached it.

The person described this build exactly and was told it could not run their
plan. "Buy and hold, never sell" is also the most common way anybody describes
what this engine is, so the refusal would have met a large share of real
sentences.

`UNNECESSARY_REFUSAL` in the benchmark taxonomy: safe, in that no wrong figure
is produced, and corrosive, because a refusal reads as authoritative and this
one is false.
"""
from __future__ import annotations

import pytest

from src.mission.from_intent import _is_negated


class TestNegationIsReadByWord:
    @pytest.mark.parametrize("span", [
        "never sold any of it", "no selling", "don't sell", "I do not sell",
        "never sell", "without selling", "I never sold anything",
    ])
    def test_a_denial_is_negated(self, span):
        assert _is_negated(span)

    @pytest.mark.parametrize("span", [
        "sell when it drops 10%", "sell half in retirement", "I will sell",
        "harvest losses each December",
    ])
    def test_an_ordinary_disposal_is_not(self, span):
        assert not _is_negated(span)

    @pytest.mark.parametrize("span", [
        "another sale", "nonetheless sell", "nothing prevents a sale",
        "a note about selling", "unnoticed sale",
    ])
    def test_a_word_containing_a_negation_is_not_one(self, span):
        """The check matches whole words. A substring test makes "another"
        negate and reads an ordinary sale as a refusal to sell — this defect
        running backwards, which is worse: it would silently execute a
        withdrawal plan as buy-and-hold."""
        assert not _is_negated(span)

    def test_nothing_stated_is_not_a_denial(self):
        assert not _is_negated(None)
        assert not _is_negated("")


class TestOnlyDisposalCountsAsAgreement:
    def test_the_set_is_narrow(self):
        """A negated cadence is not agreement with anything — "I don't
        contribute monthly" leaves the question open. Treating every negation
        as assent would turn refusals off wholesale, which is the failure mode
        of a fix like this one."""
        from src.mission.from_intent import NEGATABLE_DISPOSALS

        assert NEGATABLE_DISPOSALS == {"sell_action"}


class TestTheCompilerStopsRefusingIt:
    """Through `refusals_for`, which is what the compiler actually calls."""

    def refusals(self, declared) -> set:
        from src.mission.capability import refusals_for
        from src.mission.from_intent import NEGATABLE_DISPOSALS, _is_negated

        negated = {n for n, v in declared.items()
                   if n in NEGATABLE_DISPOSALS and _is_negated(v)}
        return {r.dimension for r in refusals_for(
            {n: v for n, v in declared.items() if n not in negated})}

    def test_a_never_sell_plan_is_not_refused_for_selling(self):
        assert "sell_action" not in self.refusals(
            {"assets": "VTI", "amount": "500", "cadence": "monthly",
             "sell_action": "never sold any of it"})

    def test_a_real_sale_is_still_refused_by_name(self):
        """The property that must survive the fix. A build that only buys has
        to keep refusing a plan that sells, or the fix has turned a refusal
        into a silent reduction."""
        assert "sell_action" in self.refusals(
            {"assets": "VTI", "sell_action": "sell 4% each year"})

    def test_other_refusals_are_untouched(self):
        """`evaluation_period` in the same sentence stays refused. The journey
        that found this states a start date the engine cannot honour, and that
        refusal is correct — fixing the false one must not take the true one
        with it."""
        found = self.refusals(
            {"assets": "VTI", "sell_action": "never sold any of it",
             "evaluation_period": "since:2015-01"})
        assert found == {"evaluation_period"}
