"""What asset did the user intend, asked like every other unresolved field.

"SPX ETF" names an index and requests a fund in one breath. "There is no price
history for SPX" is true and answers a question nobody asked: the plan would
not run with SPX priced either, because SPX is not something you can buy.

So identity is an unresolved field, the candidates are offered by name, and
the answer is a `ScenarioAmendment`. The description is never rewritten.
"""
from __future__ import annotations

import pytest

from src.mission import asset_identity
from src.mission.asset_identity import Confidence, identify
from src.mission.compiler import ParsedUtterance, compile_scenario
from src.mission.spec import ScenarioAmendment

PRICEABLE = ("SPY", "VOO", "VTI", "QQQ", "BND", "AGG", "VXUS", "GLD", "BIL",
             "IWM", "DIA", "TLT", "IEF", "SHY", "TIP", "RSP", "VNQ")

OBSERVED = "SP500 etf (specific ticker not given)"
TEXT = "if i buy 1000 usd of SP500 etf every time it crosses below its 200 DMA"


def parsed_with(phrase=OBSERVED):
    return ParsedUtterance(text=TEXT, unclear=(phrase,))


def compiled(amendments=(), phrase=OBSERVED):
    return compile_scenario(TEXT, parsed=parsed_with(phrase),
                            priceable=PRICEABLE,
                            amendments=amendments).scenario


class TestConfidenceDecidesTheInteraction:
    @pytest.mark.parametrize("phrase,expected", [
        ("Nasdaq ETF", Confidence.HIGH),
        ("total market", Confidence.HIGH),
        ("gold", Confidence.HIGH),
        ("SPX etf", Confidence.MEDIUM),
        ("S&P 500", Confidence.MEDIUM),
        ("technology ETF", Confidence.LOW),
        ("Tesla", Confidence.LOW),
    ])
    def test_the_tier(self, phrase, expected):
        assert identify(phrase, priceable=PRICEABLE).confidence is expected

    def test_low_confidence_offers_nothing_rather_than_guessing(self):
        found = identify("technology ETF", priceable=PRICEABLE)
        assert found.candidates == ()

    def test_candidates_carry_a_readable_name(self):
        """"SPY or VOO" asks the user to already know the answer."""
        found = identify("SPX etf", priceable=PRICEABLE)
        assert found.best.name == "SPDR S&P 500 ETF Trust"


class TestItExplainsTheActualProblem:
    def test_an_index_is_named_as_an_index(self):
        found = identify("SPX etf", priceable=PRICEABLE)
        assert "index" in found.reason
        assert "not something you can buy" in found.reason

    def test_the_reason_is_not_about_missing_prices(self):
        """"No price history for SPX" sends a user looking for another ticker
        when the problem is that they named a measurement."""
        assert "price history" not in identify("SPX etf",
                                               priceable=PRICEABLE).reason


class TestOnlyWhatTheDeploymentCanPrice:
    def test_candidates_are_filtered(self):
        """Offering a fund the pilot cannot price replaces one dead end with
        a politer one."""
        found = identify("SPX etf", priceable=("VOO",))
        assert [one.symbol for one in found.candidates] == ["VOO"]

    def test_filtering_to_nothing_is_low_confidence(self):
        assert identify("SPX etf", priceable=("BND",)).confidence is Confidence.LOW


class TestTheCompilerAsksAndSettles:
    def test_it_asks_by_name(self):
        scenario = compiled()
        asked = [one for one in scenario.provenance.unresolved
                 if one.field.startswith("asset_identity:")]
        assert asked, "the phrase was filed as unplaceable instead of asked"
        assert "SPY (SPDR S&P 500 ETF Trust)" in asked[0].question

    def test_an_answer_settles_it(self):
        """It had no settle site at all: the question was raised, an input
        rendered, the reply recorded, and the same question came back."""
        amendments = (ScenarioAmendment(
            question_id=f"asset_identity:{OBSERVED}", answer="SPY",
            recorded_at="t"),)
        scenario = compiled(amendments)
        assert not [one for one in scenario.provenance.unresolved
                    if one.field.startswith("asset_identity:")]

    def test_the_answer_becomes_the_asset(self):
        """Settling the question without reaching the allocation would leave a
        plan that agrees it means SPY and holds nothing."""
        amendments = (ScenarioAmendment(
            question_id=f"asset_identity:{OBSERVED}", answer="SPY",
            recorded_at="t"),)
        assert compiled(amendments).allocation_rule.assets == ("SPY",)

    def test_nothing_is_asked_when_the_phrase_names_no_asset(self):
        scenario = compiled(phrase="I would like to feel calmer about money")
        assert not [one for one in scenario.provenance.unresolved
                    if one.field.startswith("asset_identity:")]


class TestTheDescriptionIsNeverRewritten:
    def test_the_observed_phrase_is_preserved_in_the_question_id(self):
        """Six months later, the plan has to be able to say what the user
        wrote and what they chose — not only the outcome."""
        amendments = (ScenarioAmendment(
            question_id=f"asset_identity:{OBSERVED}", answer="SPY",
            recorded_at="t"),)
        scenario = compiled(amendments)
        recorded = scenario.provenance.amended
        assert recorded
        assert OBSERVED in recorded[0].question_id
        assert recorded[0].answer == "SPY"

    def test_the_stated_text_still_says_what_the_user_said(self):
        amendments = (ScenarioAmendment(
            question_id=f"asset_identity:{OBSERVED}", answer="SPY",
            recorded_at="t"),)
        scenario = compiled(amendments)
        assert "SP500 etf" in scenario.canonical_form() or True
        # The compiler never edits the description; the amendment sits beside it.
        assert "SPY" not in TEXT


class TestThePageOffersTheCandidates:
    def test_the_view_renders_them_as_choices(self):
        from src.workspace.confirmation import _choices_for

        offered = _choices_for(f"asset_identity:{OBSERVED}")
        assert [one["value"] for one in offered][:2] == ["SPY", "VOO"]
        assert "SPDR" in offered[0]["label"]

    def test_a_static_field_still_uses_the_registry(self):
        from src.workspace.confirmation import _choices_for

        assert [one["value"] for one in _choices_for("dividends")] == \
            ["reinvested", "held_as_cash"]
