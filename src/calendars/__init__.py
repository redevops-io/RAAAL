"""Trading calendars as versioned, citable artifacts.

    calendar: nyse@1

A protocol references a calendar rather than naming a rule inline, so the
sessions a result was measured over are identified data with their own hash and
their own version history.
"""
from .calendar import (
    CALENDAR_SPEC_VERSION,
    HolidayRule,
    TradingCalendar,
    easter_sunday,
)
from .registry import CalendarRegistry

__all__ = [
    "CALENDAR_SPEC_VERSION",
    "CalendarRegistry",
    "HolidayRule",
    "TradingCalendar",
    "easter_sunday",
]
