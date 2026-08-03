"""The two temporal grammars, and the refusals that keep them apart.

Timestamps remain strings through this migration, so the database compares
bytes and nothing else. That makes the canonical form load-bearing: two
spellings of one instant would be two different values to PostgreSQL, and one
spelling covering two meanings would be worse.

A date and an instant are different kinds of fact in this system. A vest settles
on a day; a report arrives at a moment. The reconciler's central distinction —
a vest reported in July that settled in June is on time — only survives if the
two never quietly become each other.
"""
from __future__ import annotations

import datetime as dt

import pytest

from src.db.temporal import (
    NotCanonicalTemporal,
    canonical_date,
    canonical_timestamp,
    same_instant,
)
from src.mission.rsu_reconcile import ObservedEvent, PlannedEvent


class TestDates:
    @pytest.mark.parametrize("value", ["2026-06-15", "2026-01-01",
                                       "2024-02-29"])
    def test_canonical_dates_are_accepted_unchanged(self, value):
        assert canonical_date(value) == value

    def test_a_date_object_is_accepted(self):
        assert canonical_date(dt.date(2026, 6, 15)) == "2026-06-15"

    def test_none_stays_none(self):
        assert canonical_date(None) is None

    @pytest.mark.parametrize("value", [
        "2026-06-15T00:00:00Z",   # a timestamp where a date belongs
        "2026-6-15",              # not zero-padded
        "2026-06-15 ",            # trailing whitespace
        " 2026-06-15",
        "15/06/2026",             # a different format entirely
        "2026-06-15Z",
        "20260615",
        "2026-13-01",             # not a real month
        "2026-02-30",             # not a real day
        "2023-02-29",             # not a leap year
    ])
    def test_refused(self, value):
        with pytest.raises(NotCanonicalTemporal):
            canonical_date(value)

    def test_a_datetime_is_refused_with_the_reason(self):
        """Coercing it would let an effective date sort against a report
        time, which is how an on-time vest starts looking late."""
        with pytest.raises(NotCanonicalTemporal, match="datetime"):
            canonical_date(dt.datetime(2026, 6, 15, tzinfo=dt.timezone.utc))


class TestTimestamps:
    def test_canonical_utc_is_accepted_unchanged(self):
        assert canonical_timestamp("2026-08-03T00:00:00Z") == \
            "2026-08-03T00:00:00Z"

    def test_an_offset_is_normalized_to_utc(self):
        """Two spellings of one instant would be two values to a byte-comparing
        database."""
        assert canonical_timestamp("2026-08-02T20:00:00-04:00") == \
            "2026-08-03T00:00:00Z"

    def test_equivalent_instants_normalize_identically(self):
        assert same_instant("2026-08-02T20:00:00-04:00",
                            "2026-08-03T00:00:00Z")

    def test_an_aware_datetime_is_normalized(self):
        moment = dt.datetime(2026, 8, 2, 20, 0, 0,
                             tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        assert canonical_timestamp(moment) == "2026-08-03T00:00:00Z"

    def test_a_fraction_is_kept_when_there_is_one(self):
        assert canonical_timestamp("2026-08-03T00:00:00.500000Z") == \
            "2026-08-03T00:00:00.500000Z"

    def test_no_fraction_is_added_when_there_is_none(self):
        """Emitting `.000000` always would make an equal instant compare
        unequal to every value written before the fraction existed."""
        assert canonical_timestamp("2026-08-03T00:00:00Z") == \
            "2026-08-03T00:00:00Z"

    def test_none_stays_none(self):
        assert canonical_timestamp(None) is None

    @pytest.mark.parametrize("value", [
        "2026-08-03",                   # bare date
        "2026-08-03T00:00:00",          # no timezone
        "2026-08-03T00:00:00z",         # lowercase, a second spelling
        "2026-08-03T00:00:00 Z",
        "2026-08-03T00:00:00Z ",        # trailing whitespace
        " 2026-08-03T00:00:00Z",
        "2026-08-03T00:00:00+0400",     # offset without a colon
        "2026-08-03T00:00:00UTC",
        "2026-08-03 00:00:00Z",         # space instead of T
        "2026-08-03T24:00:00Z",         # not a real hour
        "2026-08-03T00:60:00Z",
        "2026-08-03T23:59:60Z",         # leap second, not supported
    ])
    def test_refused(self, value):
        with pytest.raises(NotCanonicalTemporal):
            canonical_timestamp(value)

    def test_a_naive_datetime_is_refused(self):
        """Assuming UTC would turn a local wall-clock reading into a specific
        wrong instant, silently and by hours."""
        with pytest.raises(NotCanonicalTemporal, match="timezone"):
            canonical_timestamp(dt.datetime(2026, 8, 3, 0, 0, 0))

    def test_a_date_object_is_refused(self):
        with pytest.raises(NotCanonicalTemporal, match="date"):
            canonical_timestamp(dt.date(2026, 8, 3))


class TestTheTypesRefuseEachOther:
    """Neither grammar accepts the other's values."""

    def test_a_planned_event_refuses_a_timestamp_for_its_date(self):
        with pytest.raises(NotCanonicalTemporal):
            PlannedEvent(event_id="pe-1", grant_ref="grant/g1",
                         expected_date="2026-06-15T00:00:00Z")

    def test_a_planned_event_refuses_an_unreal_date(self):
        with pytest.raises(NotCanonicalTemporal):
            PlannedEvent(event_id="pe-1", grant_ref="grant/g1",
                         expected_date="2026-02-30")

    def test_an_observed_event_validates_both_of_its_dates(self):
        with pytest.raises(NotCanonicalTemporal):
            ObservedEvent(observation_id="oe-1", observed_date="2026-06-16",
                          effective_date="2026-06-15T00:00:00Z")
        with pytest.raises(NotCanonicalTemporal):
            ObservedEvent(observation_id="oe-1",
                          observed_date="2026-06-16T00:00:00Z",
                          effective_date="2026-06-15")

    def test_the_two_dates_stay_independent(self):
        """Validation must not make them interchangeable — a vest reported in
        July may have settled in June."""
        event = ObservedEvent(observation_id="oe-1",
                              observed_date="2026-07-02",
                              effective_date="2026-06-15")
        assert event.observed_date != event.effective_date
        assert event.to_json()["observed_date"] == "2026-07-02"
        assert event.to_json()["effective_date"] == "2026-06-15"


class TestTheDatabaseOnlyEverCompares:
    """PostgreSQL is never asked what these strings mean."""

    def test_canonical_strings_sort_chronologically_as_text(self):
        """Lexicographic order equals chronological order for this grammar,
        which is what lets an ordinary text index serve a date range."""
        moments = ["2026-08-03T00:00:00Z", "2026-01-01T00:00:00Z",
                   "2026-08-02T23:59:59Z", "2025-12-31T23:59:59Z"]
        canonical = [canonical_timestamp(one) for one in moments]
        as_instants = sorted(
            dt.datetime.fromisoformat(one.replace("Z", "+00:00"))
            for one in canonical)
        assert sorted(canonical) == [
            one.strftime("%Y-%m-%dT%H:%M:%S") + "Z" for one in as_instants]

    def test_dates_sort_chronologically_as_text(self):
        dates = ["2026-08-03", "2026-01-01", "2025-12-31", "2026-08-02"]
        assert sorted(dates) == sorted(
            dates, key=lambda one: dt.date.fromisoformat(one))

    def test_one_instant_has_exactly_one_stored_spelling(self):
        """The property the whole grammar exists for: a byte comparison is a
        correct instant comparison."""
        spellings = ["2026-08-03T00:00:00Z", "2026-08-02T20:00:00-04:00",
                     "2026-08-03T02:00:00+02:00",
                     dt.datetime(2026, 8, 3, tzinfo=dt.timezone.utc)]
        assert len({canonical_timestamp(one) for one in spellings}) == 1
