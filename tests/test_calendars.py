"""Trading calendars as versioned artifacts.

`calendar: business_days` was a string enum that silently meant "Monday to
Friday, holidays included as flat days" — not any real exchange. These tests hold
the replacement to being a citable artifact whose sessions are checkable against
a published schedule.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.calendars import CalendarRegistry, HolidayRule, TradingCalendar, easter_sunday


class TestComputus:
    @pytest.mark.parametrize(
        "year,expected",
        [
            (2016, date(2016, 3, 27)),
            (2019, date(2019, 4, 21)),
            (2021, date(2021, 4, 4)),
            (2024, date(2024, 3, 31)),
            (2025, date(2025, 4, 20)),
        ],
    )
    def test_known_easter_dates(self, year, expected):
        """Good Friday is Easter-relative, so the computus must be right."""
        assert easter_sunday(year) == expected


class TestHolidayRules:
    def test_nth_weekday_forward(self):
        rule = HolidayRule("MLK", "nth_weekday", month=1, weekday=0, nth=3)
        assert rule.resolve(2025) == date(2025, 1, 20)

    def test_nth_weekday_from_end(self):
        rule = HolidayRule("Memorial", "nth_weekday", month=5, weekday=0, nth=-1)
        assert rule.resolve(2025) == date(2025, 5, 26)

    def test_saturday_holiday_observed_friday(self):
        rule = HolidayRule("Independence", "fixed", month=7, day=4)
        assert date(2020, 7, 4).weekday() == 5
        assert rule.resolve(2020) == date(2020, 7, 3)

    def test_sunday_holiday_observed_monday(self):
        rule = HolidayRule("Christmas", "fixed", month=12, day=25)
        assert date(2022, 12, 25).weekday() == 6
        assert rule.resolve(2022) == date(2022, 12, 26)

    def test_good_friday(self):
        rule = HolidayRule("Good Friday", "easter_offset", offset=-2)
        assert rule.resolve(2025) == date(2025, 4, 18)

    def test_rule_respects_first_year(self):
        """Juneteenth became a market holiday in 2022; earlier years must not
        retroactively close the exchange."""
        rule = HolidayRule("Juneteenth", "fixed", month=6, day=19, first_year=2022)
        assert rule.resolve(2021) is None
        assert rule.resolve(2023) == date(2023, 6, 19)


class TestNyseCalendar:
    @pytest.fixture
    def nyse(self):
        return CalendarRegistry().get("nyse", 1)

    def test_identity(self, nyse):
        assert nyse.calendar_id == "calendar/nyse@1"
        assert nyse.content_hash

    def test_known_closures(self, nyse):
        holidays = nyse.holiday_dates(2025, 2025)
        for expected in (
            date(2025, 1, 1),    # New Year's Day
            date(2025, 1, 20),   # MLK
            date(2025, 2, 17),   # Washington's Birthday
            date(2025, 4, 18),   # Good Friday
            date(2025, 5, 26),   # Memorial Day
            date(2025, 6, 19),   # Juneteenth
            date(2025, 7, 4),    # Independence Day
            date(2025, 9, 1),    # Labor Day
            date(2025, 11, 27),  # Thanksgiving
            date(2025, 12, 25),  # Christmas
        ):
            assert expected in holidays, f"{expected} should be an NYSE closure"

    def test_roughly_ten_closures_a_year(self, nyse):
        for year in (2018, 2021, 2024):
            count = len(nyse.holiday_dates(year, year))
            assert 8 <= count <= 11, f"{year} produced {count} closures"

    def test_juneteenth_only_from_2022(self, nyse):
        assert date(2021, 6, 18) not in nyse.holiday_dates(2021, 2021)
        assert date(2022, 6, 20) in nyse.holiday_dates(2022, 2022)

    def test_filtering_removes_weekends_and_holidays(self, nyse):
        index = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        frame = pd.DataFrame({"x": range(len(index))}, index=index)

        sessions = nyse.filter(frame)

        assert all(ts.dayofweek < 5 for ts in sessions.index)
        assert pd.Timestamp("2025-07-04") not in sessions.index
        assert pd.Timestamp("2025-12-25") not in sessions.index
        # 2025 has 261 weekdays; ten closures fall on weekdays.
        assert 249 <= len(sessions) <= 253

    def test_refuses_data_outside_declared_coverage(self, nyse):
        index = pd.date_range("2040-01-01", periods=10, freq="D")
        frame = pd.DataFrame({"x": range(10)}, index=index)
        with pytest.raises(ValueError, match="covers"):
            nyse.filter(frame)


class TestCryptoCalendar:
    def test_trades_every_day(self):
        crypto = CalendarRegistry().get("crypto", 1)
        index = pd.date_range("2025-01-01", periods=30, freq="D")
        frame = pd.DataFrame({"x": range(30)}, index=index)

        assert len(crypto.filter(frame)) == 30
        assert crypto.periods_per_year == 365

    def test_differs_from_nyse(self):
        """The mismatch that caused Erratum 02 is now expressible, and therefore
        checkable, rather than implicit in a joined index."""
        registry = CalendarRegistry()
        assert registry.get("crypto", 1).content_hash != registry.get("nyse", 1).content_hash


class TestVersioning:
    def test_changing_a_rule_changes_identity(self):
        base = TradingCalendar(name="x", version=1, title="x", holidays=())
        with_holiday = TradingCalendar(
            name="x", version=1, title="x",
            holidays=(HolidayRule("New Year", "fixed", month=1, day=1),),
        )
        assert base.content_hash != with_holiday.content_hash

    def test_title_does_not_change_identity(self):
        a = TradingCalendar(name="x", version=1, title="A")
        b = TradingCalendar(name="x", version=1, title="B")
        assert a.content_hash == b.content_hash

    def test_registry_resolves_references(self):
        registry = CalendarRegistry()
        assert registry.resolve("calendar/nyse@1").calendar_id == "calendar/nyse@1"
        assert registry.resolve("nyse@1").version == 1
        assert registry.resolve("nyse").version == max(registry.names()["nyse"])


class TestProtocolIntegration:
    def test_protocols_reference_a_calendar_artifact(self):
        from src.evaluation import ProtocolRegistry

        for protocol in ProtocolRegistry().load_all():
            reference = protocol.walk_forward.calendar
            assert "@" in reference, (
                f"{protocol.protocol_id} names calendar {reference!r} — a protocol "
                "must reference a versioned artifact, not a bare enum"
            )
            CalendarRegistry().resolve(reference)

    def test_periods_per_year_comes_from_the_calendar(self):
        from src.evaluation import ProtocolRegistry
        from src.evaluation.runner import periods_per_year

        protocol = ProtocolRegistry().get("standard", 1)
        assert protocol.walk_forward.periods_per_year is None, (
            "duplicating the session count on the protocol invites disagreement "
            "with the calendar"
        )
        assert periods_per_year(protocol) == 252

    def test_calendar_reference_is_part_of_protocol_identity(self):
        from src.evaluation import ProtocolRegistry
        from dataclasses import replace

        protocol = ProtocolRegistry().get("standard", 1)
        switched = replace(
            protocol,
            walk_forward=replace(protocol.walk_forward, calendar="crypto@1"),
        )
        assert protocol.content_hash != switched.content_hash
