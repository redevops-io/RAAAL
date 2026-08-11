"""The sentence this whole line of work started from, end to end.

    "I buy $1,000 of SP500 ETF every time the S&P 500 crosses below its
     200-day moving average for the past 5 years. What would the total amount
     and return be now?"

Every part of it was a dead end at some point: the instrument reported a
missing price for something nobody can buy, the amount had no control, the
period was unplaceable prose, and answering anything at all discarded the rest.
This is the fixture that says it no longer is.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src.mission.compiler import ParsedUtterance, compile_scenario
from src.mission.spec import ScenarioAmendment
from src.mission.time_window import WindowKind, detect, resolve

PROMPT = ("I buy $1,000 of SP500 ETF every time the S&P 500 crosses below its "
          "200-day moving average for the past 5 years. What would the total "
          "amount and return be now?")

#: What stage 1 hands back for this sentence — the phrases it cannot place.
UNCLEAR = ("SP500 ETF (asset referenced by name, not a literal ticker)",
           "for the past 5 years (time horizon)")

PRICEABLE = ("SPY", "VOO", "IVV", "VTI", "QQQ", "BND", "AGG", "BIL")


def compiled(amendments=()):
    parsed = ParsedUtterance(text=PROMPT, unclear=UNCLEAR)
    return compile_scenario(PROMPT, parsed=parsed, priceable=PRICEABLE,
                            amendments=amendments).scenario


class TestWhatTheCompilerUnderstandsUnaided:
    def test_the_trigger_is_recognised(self):
        inferred = {one.field for one in compiled().provenance.inferred}
        assert "moving_average_kind" in inferred

    def test_the_time_window_is_recognised(self):
        window = compiled().provenance.time_window
        assert window.kind is WindowKind.TRAILING
        assert window.years == 5

    def test_the_period_is_no_longer_unplaceable_prose(self):
        """It arrived as an "unclear" phrase whose only control was "continue
        without modelling it" — which meant discarding the period the question
        was about."""
        fields = {one.field for one in compiled().provenance.unresolved}
        assert not any(one.startswith("unclear:for the past") for one in fields)

    def test_the_asset_is_asked_rather_than_reported_missing(self):
        """"No price history for SPX" was true and answered a question nobody
        asked: SPX is an index, and the plan would not run with it priced."""
        fields = {one.field for one in compiled().provenance.unresolved}
        assert any(one.startswith("asset_identity:") for one in fields)


class TestTheQuestionsAreAnswerable:
    def test_the_asset_question_offers_priceable_funds(self):
        record = compiled().provenance.asset_resolutions[0]
        assert record.candidates_shown[0] == "SPY"
        assert set(record.candidates_shown) <= set(PRICEABLE)

    def test_it_explains_the_index_mismatch(self):
        from src.mission.asset_identity import identify

        found = identify(UNCLEAR[0], priceable=PRICEABLE)
        assert "is an index" in found.reason

    def test_answering_settles_the_asset(self):
        chosen = (ScenarioAmendment(
            question_id=f"asset_identity:{UNCLEAR[0]}", answer="SPY",
            recorded_at="t"),)
        scenario = compiled(chosen)
        assert scenario.allocation_rule.assets == ("SPY",)
        assert not any(one.field.startswith("asset_identity:")
                       for one in scenario.provenance.unresolved)


class TestTheDescriptionIsNeverRewritten:
    def test_the_prompt_still_says_sp500_etf(self):
        chosen = (ScenarioAmendment(
            question_id=f"asset_identity:{UNCLEAR[0]}", answer="SPY",
            recorded_at="t"),)
        scenario = compiled(chosen)
        record = scenario.provenance.asset_resolutions[0]

        assert "SP500 ETF" in record.observed_phrase
        assert record.chosen_instrument_id == "SPY"
        assert "SPY" not in PROMPT

    def test_the_resolution_is_pinned_to_a_registry(self):
        assert compiled().provenance.asset_resolutions[0] \
            .registry_digest.startswith("reg1:")


class TestTheWindowResolvesAgainstTheSnapshot:
    def sessions(self):
        start = dt.date(2015, 1, 2)
        return sorted({start + dt.timedelta(days=i) for i in range(4200)
                       if (start + dt.timedelta(days=i)).weekday() < 5})

    def test_now_is_the_latest_session_not_the_clock(self):
        available = self.sessions()
        found = resolve(compiled().provenance.time_window, available)
        assert found.end == available[-1]

    def test_the_window_spans_five_calendar_years(self):
        found = resolve(compiled().provenance.time_window, self.sessions())
        assert 4.98 <= (found.end - found.start).days / 365.25 <= 5.02

    def test_the_moving_average_warm_up_precedes_the_window(self):
        """A 200-day average needs 200 sessions before the first session it
        can judge. Taking them out of the five years would report on four
        years and three months while saying five."""
        available = self.sessions()
        found = resolve(compiled().provenance.time_window, available,
                        warmup_sessions=200)

        assert found.warmup_start < found.start
        assert available.index(found.start) \
            - available.index(found.warmup_start) == 200

    def test_the_reported_period_excludes_the_warm_up(self):
        available = self.sessions()
        plain = resolve(compiled().provenance.time_window, available)
        warmed = resolve(compiled().provenance.time_window, available,
                         warmup_sessions=200)
        assert (warmed.start, warmed.end) == (plain.start, plain.end)


class TestTheWholeJourneyConverges:
    def test_answering_everything_leaves_no_asset_or_window_question(self):
        """The property the whole line of work was for: ordinary prose,
        typed ambiguities, user clarification, and a plan that no longer asks
        about the things that were settled."""
        amendments = (
            ScenarioAmendment(question_id=f"asset_identity:{UNCLEAR[0]}",
                              answer="SPY", recorded_at="t"),
            ScenarioAmendment(question_id="account_type", answer="TAXABLE",
                              recorded_at="t"),
            ScenarioAmendment(question_id="amount", answer="1000",
                              recorded_at="t"),
            ScenarioAmendment(question_id="starting_capital", answer="0",
                              recorded_at="t"),
            ScenarioAmendment(question_id="trigger_semantics",
                              answer="crossing_event", recorded_at="t"),
        )
        scenario = compiled(amendments)
        remaining = {one.field for one in scenario.provenance.unresolved}

        for settled in ("account_type", "amount", "starting_capital",
                        "trigger_semantics"):
            assert settled not in remaining, f"{settled} was asked again"
        assert not any(one.startswith("asset_identity:") for one in remaining)
        assert not any(one.startswith("time_window:") for one in remaining)
        assert scenario.allocation_rule.assets == ("SPY",)
        assert scenario.provenance.time_window.years == 5
