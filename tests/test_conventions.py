"""This build's time rules, checked against the definitions they claim to be.

A rule reasoned out in a comment is a rule only this repository knows. The
contribution rule was written from scratch, tested from scratch, and described
in three separate comments before anybody compared it with the convention it
turned out to be — `ModifiedFollowing`, exactly, on every month of 2024.

So the value of this file is not that QuantLib is faster. It is that a claim
here can be checked against a definition outside here, and that a plan saying
"ModifiedFollowing" says the same thing to somebody who has never read this
code.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.mission import conventions

ql = pytest.importorskip("QuantLib",
                         reason="the convention vocabulary is what is under test")


class TestTheVocabularyReachesProduction:
    def test_it_is_installed(self):
        assert conventions.AVAILABLE, (
            "QuantLib is not importable, so every convention below falls back "
            "to unnamed arithmetic")

    def test_it_is_declared_where_the_image_installs_from(self):
        """The Stanza mistake, not repeated.

        The deterministic reader has been absent from every deployment for
        months because it sits in `requirements.txt` while the image installs
        `requirements-core.txt` — a fact the pilot page prints about itself. A
        convention library in the wrong file would be that again.
        """
        from pathlib import Path

        core = (Path(__file__).resolve().parent.parent
                / "requirements-core.txt").read_text()
        assert "QuantLib" in core, (
            "QuantLib is not in requirements-core.txt, so it will not be in "
            "the image and every convention here silently falls back")


class TestTheContributionRuleIsTheConventionItClaims:
    """`ModifiedFollowing`, checked rather than asserted."""

    def test_every_month_of_a_year_agrees(self):
        calendar = conventions.calendar()
        for year in (2023, 2024, 2025):
            for month in range(1, 13):
                for day in (1, 15, 28, 30, 31):
                    ours = conventions.adjust(day, month, year)
                    last = ql.Date.endOfMonth(
                        ql.Date(1, month, year)).dayOfMonth()
                    theirs = calendar.adjust(
                        ql.Date(min(day, last), month, year),
                        ql.ModifiedFollowing)
                    assert (ours.day, ours.month, ours.year) == (
                        theirs.dayOfMonth(), theirs.month(), theirs.year()), (
                            f"{year}-{month:02d} day {day}")

    def test_a_named_convention_stays_inside_its_month(self):
        """The property the name buys. `Following` would move a contribution
        into the next month, giving one month two and another none."""
        for month in range(1, 13):
            settled = conventions.adjust(31, month, 2024)
            assert settled.month == month, (
                f"a contribution for month {month} landed in {settled.month}")

    def test_the_engine_and_the_convention_agree(self):
        """The engine works off sessions and the convention off a calendar.
        Where the data has a session for every open day, they must match."""
        from src.mission.schedule import expand

        calendar = conventions.calendar()
        sessions = pd.DatetimeIndex([
            pd.Timestamp(d.year(), d.month(), d.dayOfMonth())
            for d in ql.MakeSchedule(ql.Date(1, 1, 2024), ql.Date(31, 12, 2024),
                                     ql.Period('1d'))
            if calendar.isBusinessDay(d)])

        class Schedule:
            day_rule, cadence, amount = "calendar_day:15", "monthly", 100.0
            starting_capital = 0.0

        class Flow:
            def __init__(self, date, amount, why):
                self.date = date

        landed = [f.date.date() for f in
                  expand(Schedule(), sessions, cash_flow=Flow)]
        named = [conventions.adjust(15, month, 2024) for month in range(1, 13)]
        assert landed == named


class TestTheDayCountsAreReal:
    @pytest.mark.parametrize("count", conventions.DAY_COUNTS)
    def test_each_names_a_quantlib_day_counter(self, count):
        """The declared name must be QuantLib's own.

        Compared by containment rather than equality because `Business/252`
        reports the calendar it counts against —
        `Business/252(New York stock exchange)` — and that suffix is a fact
        about this build's exchange, not a different convention.
        """
        theirs = count.quantlib().name()
        assert count.name in theirs, (
            f"{count.name!r} is not QuantLib's {theirs!r}, so the label is "
            "decoration rather than a convention anybody can look up")

    def test_what_this_build_annualises_on_is_declared(self):
        """A run that does not say which year it divided by cannot be compared
        with one that does."""
        declared = conventions.declared()
        assert declared["annualisation"] == "Business/252"
        assert declared["sessions_per_year"] == 252.0
        assert declared["exchange"] == "NYSE"
        assert declared["contribution_convention"] == "ModifiedFollowing"

    def test_the_number_the_simulator_uses_is_the_declared_one(self):
        """The point of naming it. If the simulator's default and the declared
        convention drift apart, figures are computed on one and reported under
        the other."""
        import inspect

        # `simulate` is re-exported as the function, not the module.
        from src.mission import simulate

        default = inspect.signature(simulate).parameters[
            "periods_per_year"].default
        assert default == conventions.ANNUALISATION.per_year, (
            f"the simulator annualises on {default} and this build declares "
            f"{conventions.ANNUALISATION.name}")


class TestPeriodsAreSayable:
    @pytest.mark.parametrize("text,expected", [
        ("1d", "1D"), ("2w", "2W"), ("1m", "1M"), ("1y", "1Y"),
    ])
    def test_the_shorthand_a_person_writes_parses(self, text, expected):
        """`+1d`, `+2w`, `+1m` — the vocabulary a cadence is heading towards.
        `monthly` is this build's word for `1m`."""
        assert str(conventions.period(text)) == expected
