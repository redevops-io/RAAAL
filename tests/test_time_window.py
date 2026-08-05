"""A period is an instruction, not a pair of dates.

"the past 5 years" and "2021-08-05 through 2026-08-05" resolve to the same
sessions today and are not the same thing: re-run next month and one moves.
So the instruction is stored and the dates are derived from it.

Four semantics that are easy to get wrong by accident and hard to notice
afterwards: the anchor, calendar versus trading years, boundary inclusion, and
whether indicator warm-up eats into the analysis period.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src.mission.compiler import ParsedUtterance, compile_scenario
from src.mission.time_window import WindowKind, detect, resolve


def sessions(start=dt.date(2015, 1, 2), days=4200):
    return sorted({start + dt.timedelta(days=i) for i in range(days)
                   if (start + dt.timedelta(days=i)).weekday() < 5})


class TestTheInstructionIsTyped:
    @pytest.mark.parametrize("phrase,kind,years", [
        ("for the past 5 years", WindowKind.TRAILING, 5),
        ("over the past five years", WindowKind.TRAILING, 5),
        ("last 3 years", WindowKind.TRAILING, 3),
        ("5-year lookback period for backtest", WindowKind.TRAILING, 5),
    ])
    def test_a_trailing_period(self, phrase, kind, years):
        window = detect(phrase)
        assert window.kind is kind and window.years == years
        assert window.supported

    @pytest.mark.parametrize("phrase,kind", [
        ("since 2021", WindowKind.SINCE),
        ("through 2024", WindowKind.UNTIL),
        ("from January 2020 to December 2024", WindowKind.EXPLICIT_RANGE),
        ("since the 2022 drawdown", WindowKind.EVENT_RELATIVE),
        ("every month over the past five years", WindowKind.ROLLING),
    ])
    def test_other_forms_are_recognised_and_refused(self, phrase, kind):
        """Typed, not coerced. Reading "since 2021" as a trailing window would
        answer a different question with a number that looks right."""
        window = detect(phrase)
        assert window.kind is kind
        assert not window.supported

    def test_a_rolling_phrase_is_not_read_as_the_window_inside_it(self):
        """"every month over the past five years" contains "the past five
        years" and is not one window."""
        assert detect("every month over the past five years").kind \
            is WindowKind.ROLLING

    def test_prose_with_no_period_yields_nothing(self):
        assert detect("I never sell anything") is None

    def test_the_observed_phrase_is_kept(self):
        assert "past 5 years" in detect("for the past 5 years").observed


class TestTheAnchorIsTheSnapshot:
    def test_it_is_the_latest_session_not_today(self):
        """Anchoring to the clock makes one plan give different figures on
        different days from the same data."""
        available = sessions(days=800)
        found = resolve(detect("the past 1 year"), available)
        assert found.end == available[-1]
        assert found.end != dt.date.today()
        assert found.anchor_source == "snapshot latest session"

    def test_two_resolutions_of_one_snapshot_agree(self):
        available = sessions()
        first = resolve(detect("the past 5 years"), available)
        second = resolve(detect("the past 5 years"), available)
        assert (first.start, first.end) == (second.start, second.end)


class TestFiveYearsMeansFiveYears:
    def test_the_span_is_calendar_not_trading_days(self):
        """5 x 252 sessions drifts about a fortnight a year against the
        calendar, and nobody asked for that."""
        found = resolve(detect("the past 5 years"), sessions())
        span = (found.end - found.start).days / 365.25
        assert 4.98 <= span <= 5.02, span

    def test_the_start_is_a_real_session(self):
        available = sessions()
        found = resolve(detect("the past 5 years"), available)
        assert found.start in available

    def test_a_leap_day_boundary_clamps_rather_than_raising(self):
        """29 February has no counterpart in a non-leap year.

        Asserted on the calendar arithmetic directly. Through `resolve` the
        boundary is then aligned forward to a real session, so 2020-02-29 minus
        five years lands on 2015-02-28 and the first session on or after it is
        2015-03-02 — which made an assertion about the start month wrong about
        the code rather than about the calendar.
        """
        from src.mission.time_window import _back

        assert _back(dt.date(2020, 2, 29), years=1) == dt.date(2019, 2, 28)
        assert _back(dt.date(2020, 2, 29), years=4) == dt.date(2016, 2, 29)

    def test_a_leap_day_anchor_resolves(self):
        available = sorted({dt.date(2020, 2, 28) - dt.timedelta(days=i)
                            for i in range(2600)
                            if (dt.date(2020, 2, 28)
                                - dt.timedelta(days=i)).weekday() < 5})
        found = resolve(detect("the past 5 years"), available)
        span = (found.end - found.start).days / 365.25
        assert 4.98 <= span <= 5.02, span

    def test_a_window_longer_than_the_snapshot_says_so(self):
        found = resolve(detect("the past 20 years"), sessions(days=800))
        assert found.short


class TestWarmUpIsNotAnalysis:
    def test_it_extends_before_the_window(self):
        """Taking warm-up out of the analysis period would silently answer a
        question about four years and three months."""
        found = resolve(detect("the past 5 years"), sessions(),
                        warmup_sessions=200)
        assert found.warmup_start < found.start

    def test_the_analysis_window_is_unchanged_by_it(self):
        available = sessions()
        without = resolve(detect("the past 5 years"), available)
        with_warmup = resolve(detect("the past 5 years"), available,
                              warmup_sessions=200)
        assert (with_warmup.start, with_warmup.end) == (without.start, without.end)

    def test_it_is_exactly_the_sessions_asked_for(self):
        available = sessions()
        found = resolve(detect("the past 5 years"), available,
                        warmup_sessions=200)
        assert available.index(found.start) - available.index(found.warmup_start) \
            == 200

    def test_no_warm_up_is_requested_by_default(self):
        assert resolve(detect("the past 5 years"), sessions()).warmup_start is None


class TestBoundariesAreDeclared:
    def test_inclusive_at_both_ends(self):
        """A trigger firing on the first or last session is where two
        implementations quietly disagree."""
        assert resolve(detect("the past 5 years"), sessions()).inclusive


class TestTheCompilerRecordsIt:
    TEXT = ("if i buy 1000 usd of SP500 etf every time it crosses below its "
            "200 DMA for the past 5 years - what is the total return by today")

    def compiled(self, text=None, unclear=()):
        text = text or self.TEXT
        parsed = ParsedUtterance(text=text, unclear=tuple(unclear))
        return compile_scenario(text, parsed=parsed,
                                priceable=("SPY", "VOO", "IVV")).scenario

    def test_the_window_reaches_the_scenario(self):
        window = self.compiled().provenance.time_window
        assert window.kind is WindowKind.TRAILING and window.years == 5

    def test_a_phrase_describing_it_is_no_longer_unplaceable(self):
        """It used to arrive as "5-year lookback period for backtest" with
        only "continue without modelling it" beneath it — which meant
        discarding the period the whole question was about."""
        scenario = self.compiled(
            unclear=("5-year lookback period for backtest",))
        fields = {one.field for one in scenario.provenance.unresolved}
        assert not any(one.startswith("unclear:5-year") for one in fields)

    def test_an_unsupported_form_becomes_a_question(self):
        scenario = self.compiled("I buy VOO monthly since 2021.")
        fields = {one.field for one in scenario.provenance.unresolved}
        assert "time_window:since" in fields

    def test_the_question_does_not_pretend_it_was_understood(self):
        scenario = self.compiled("I buy VOO monthly since 2021.")
        asked = next(one for one in scenario.provenance.unresolved
                     if one.field.startswith("time_window:"))
        assert "trailing" in asked.question
        assert "different question" in asked.why_it_matters
