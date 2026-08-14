"""A contribution on a named day of the month.

Somebody wrote "I invest $200 into NVDA every month, on the same day each
month - the 15th for the past 5 years" and the page recorded
`day_rule: calendar_first_rolled_forward` — the *first* of the period — then
refused them for asking for a rule this build does not run. They had not asked
for it. The vocabulary had no way to say "the 15th", so the reader answered
with the nearest thing it could say, and a wrong reading arrived wearing a
refusal's clothes.

Two halves, and the tests keep them apart. The reader must read the day out of
the sentence without confusing it with the bare numbers around it, and the
engine must land money on it — rolling forward off a closed market, because
the 15th is a weekend about two months in seven.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.discovery.derived_readers import DAY_READER_ID, day_of_month
from src.mission.capability import dimension
from src.mission.schedule import CALENDAR_DAY, expand
from src.mission.schedule import day_of_month as day_named


class TestReadingTheDay:
    def test_the_reported_sentence_is_read_as_the_fifteenth(self):
        """The sentence that produced the wrong reading."""
        found = day_of_month([], text=(
            "I invest $200 into NVDA every month, on the same day each "
            "month - the 15th ofr the past 5 years."))
        assert found is not None, "the day was dropped again"
        assert found.value == "calendar_day:15"
        assert found.dimension == "day_rule"
        assert found.reader_id == DAY_READER_ID

    @pytest.mark.parametrize("text,expected", [
        ("I contribute $300 on the 3rd of every month", "calendar_day:3"),
        ("buy on the 28th each month", "calendar_day:28"),
        ("the 1st of the month", "calendar_day:1"),
    ])
    def test_other_days_are_read(self, text, expected):
        found = day_of_month([], text=text)
        assert found is not None and found.value == expected


class TestTheNeighboursAreNotMistakenForDays:
    """Every other number in these sentences is a different dimension.

    Reading one of them as a day would place money on a date nobody named,
    which is the wrong-executable-meaning class this project spends most of
    its effort refusing to produce.
    """

    @pytest.mark.parametrize("text,why", [
        ("I invest $500 into VTI every month, on the same day each month.",
         "no day is named at all — 'the same day' says only that it is fixed"),
        ("I buy VOO when SPY falls below its 200-day moving average.",
         "200-day is a window"),
        ("I withdraw 4% of the portfolio each year, adjusted for inflation.",
         "4% is a rate"),
        ("I invest $200 into NVDA every month for the past 5 years.",
         "5 years is an evaluation period"),
        ("I hold a 60/40 portfolio: 60% stocks and 40% bonds.",
         "60/40 is a weight"),
        ("buy on the 1st trading day of the month",
         "the first trading day is the session rule this build already runs"),
        ("invest on the 1st and the 15th of each month",
         "two days is a schedule, and choosing one would be a coin toss"),
    ])
    def test_it_stays_silent(self, text, why):
        assert day_of_month([], text=text) is None, (
            f"read a day of the month out of a sentence where {why}")


class TestTheManifestKnowsWhatItCanRun:
    def test_a_named_day_executes(self):
        assert dimension("day_rule").executes("calendar_day:15")

    def test_a_day_outside_the_month_does_not(self):
        """The family passing a prefix test is not the same as the day being
        one the engine can honour."""
        assert not dimension("day_rule").executes("calendar_day:99")
        assert not dimension("day_rule").executes("calendar_day:0")
        assert not dimension("day_rule").executes("calendar_day:fifteen")

    def test_the_rule_the_reader_used_to_invent_is_still_refused(self):
        assert not dimension("day_rule").executes("calendar_first_rolled_forward")

    def test_the_session_rules_still_execute(self):
        assert dimension("day_rule").executes("first_session_of_period")
        assert dimension("day_rule").executes("last_session_of_period")

    @pytest.mark.parametrize("rule,day", [
        ("calendar_day:15", 15), ("calendar_day:1", 1),
        ("first_session_of_period", None), ("calendar_day:99", None),
        ("calendar_day:", None), ("", None),
    ])
    def test_the_engine_reads_the_day_back(self, rule, day):
        assert day_named(rule) == day


class Schedule:
    """The little the expander needs."""

    def __init__(self, day_rule, cadence="monthly", amount=100.0):
        self.day_rule = day_rule
        self.cadence = cadence
        self.amount = amount
        self.starting_capital = 0.0


class Flow:
    def __init__(self, date, amount, why):
        self.date, self.amount, self.why = date, amount, why


def weekdays(start="2024-01-01", end="2024-12-31"):
    """Sessions, as a market has them: weekdays only."""
    return pd.bdate_range(start, end)


class TestLandingTheMoney:
    def test_it_lands_on_the_named_day(self):
        flows = expand(Schedule("calendar_day:15"), weekdays(), cash_flow=Flow)
        assert len(flows) == 12
        for flow in flows:
            assert flow.date.day >= 15, (
                f"a contribution landed on the {flow.date.day}th, before the "
                "day that was named")

    def test_a_closed_market_rolls_forward_rather_than_skipping(self):
        """15 June 2024 is a Saturday. The money still goes in that month."""
        flows = expand(Schedule("calendar_day:15"), weekdays(), cash_flow=Flow)
        june = [f.date for f in flows if f.date.month == 6]
        assert len(june) == 1
        assert june[0].day == 17, (
            f"landed on the {june[0].day}th; the 15th was a Saturday and the "
            "next session is Monday the 17th")

    def test_a_day_the_month_does_not_reach_takes_its_last_session(self):
        """February has no 31st. Rolling into March would put two
        contributions in one month and none in another; monthly means once a
        month, and landing late within it keeps that true."""
        flows = expand(Schedule("calendar_day:31"), weekdays(), cash_flow=Flow)
        by_month = {}
        for flow in flows:
            by_month.setdefault(flow.date.month, []).append(flow.date)
        assert all(len(dates) == 1 for dates in by_month.values()), (
            "some month received two contributions or none")
        assert by_month[2][0].day == 29, (
            f"February landed on the {by_month[2][0].day}th; 2024 is a leap "
            "year and its last session is the 29th")

    def test_it_differs_from_the_rule_it_used_to_be_read_as(self):
        """The whole point. If a named day produced the same dates as the
        first session, the misreading would have been harmless and this would
        be ceremony."""
        named = [f.date for f in expand(Schedule("calendar_day:15"),
                                        weekdays(), cash_flow=Flow)]
        first = [f.date for f in expand(Schedule("first_session_of_period"),
                                        weekdays(), cash_flow=Flow)]
        assert named != first
        assert all(n != f for n, f in zip(named, first))

    def test_the_session_rules_are_unchanged(self):
        first = [f.date for f in expand(Schedule("first_session_of_period"),
                                        weekdays(), cash_flow=Flow)]
        last = [f.date for f in expand(Schedule("last_session_of_period"),
                                       weekdays(), cash_flow=Flow)]
        assert [d.month for d in first] == list(range(1, 13))
        assert first[0].day <= 3 and last[0].day >= 29

    def test_an_unreadable_day_does_not_silently_choose_one(self):
        """`calendar_day:99` is refused by the manifest. If it ever reached
        the expander it must not land money on an invented date."""
        flows = expand(Schedule("calendar_day:99"), weekdays(), cash_flow=Flow)
        first = [f.date for f in expand(Schedule("first_session_of_period"),
                                        weekdays(), cash_flow=Flow)]
        assert [f.date for f in flows] == first, (
            "an unreadable day rule invented a schedule instead of falling "
            "back to the rule the manifest checks")


class TestTheReaderIsRegistered:
    def test_it_is_in_the_pipeline(self):
        """A reader nobody runs is a reader that does nothing — the same shape
        as a login route nothing links to."""
        from src.discovery.derived_readers import DERIVED_READERS

        assert DAY_READER_ID in dict(DERIVED_READERS)


class TestTheDerivedReadersRunWhereProductionRunsThem:
    """A reader that only runs beside a parser production does not install.

    `pilot.read` fuses through `pipeline.read` when a syntax reader is present
    and through a shorter path when it is not. Only the first ran the derived
    readers, and no deployment this project serves declares a syntax witness —
    so trigger semantics, weight binding and day-of-month had never run for a
    single user. `weight_binding` had even been rewritten to read from the
    sentence rather than the parse *because* production has no Stanza, and
    then sat behind the branch that requires one.

    The same shape as a login route nothing links to, and the third of its
    kind found by somebody using the site rather than by this suite.
    """

    def reading_without_a_parser(self, text):
        from dataclasses import dataclass
        from typing import Any, Sequence

        from src.discovery.schema import QUANTIFY_SCHEMA
        from src.discovery.witnesses import MODEL_ONLY
        from src.workspace.pilot import read

        @dataclass
        class One:
            dimension: str
            value: Any
            source_span: Any = None

        @dataclass
        class Answer:
            readings: Sequence[One]
            reader_id: str = "stub@1"
            ok: bool = True
            failed: str = ""
            unread: Sequence[str] = ()
            relations: Sequence[Any] = ()

        class Reader:
            id = "stub@1"

            def read(self, _text, _schema):
                # The hosted reader answers everything except the day, which
                # is the case the derived reader exists for.
                return Answer(readings=[
                    One("objective", "evaluate_investment_strategy"),
                    One("amount", "200"),
                    One("assets", "NVDA"),
                    One("cadence", "monthly"),
                ])

        return read(text, Reader(), schema=QUANTIFY_SCHEMA,
                    profile=MODEL_ONLY, syntax_reader=None)

    def test_the_day_is_read_with_no_syntax_reader(self):
        found = self.reading_without_a_parser(
            "I invest $200 into NVDA every month, on the 15th.")
        settled = {f.field: f.value for f in found.settled}
        assert settled.get("day_rule") == "calendar_day:15", (
            "the day was not read on the path every real deployment takes. "
            f"settled: {settled}")

    def test_the_reader_that_answered_is_named(self):
        """Provenance survives the shorter path, or a derived reading would
        look like the hosted reader's own answer."""
        found = self.reading_without_a_parser(
            "I invest $200 into NVDA every month, on the 15th.")
        for settled in found.settled:
            if settled.field == "day_rule":
                named = f"{settled.reader_id} {settled.provenance} {settled.detail}"
                assert DAY_READER_ID in named, (
                    "the day was settled without saying which reader derived "
                    f"it: {named!r}")
                break
        else:
            pytest.fail("day_rule was not settled at all")
