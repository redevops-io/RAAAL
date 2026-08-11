"""Phase 2 of the parser plan: literals, canonicalised, with their spans.

Every case in the plan's own falsification list is about *which number was
read*, so this file is written from that list rather than from the code. The
ones that need a parse tree live elsewhere; these need only the characters.

The rule the whole module turns on: one stretch of text carries one reading.
"90-day moving average" is a window, and the fact that it contains "90-day" is
not permission to also call it a holding period.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from src.discovery.syntax import normalize


def kinds(text: str) -> list:
    return [(v.kind, v.canonical, v.unit) for v in normalize(text)]


class TestMoney:
    @pytest.mark.parametrize("text,expected", [
        ("$1k", Decimal(1000)),
        ("$1,500", Decimal(1500)),
        ("$1.5k", Decimal(1500)),
        ("€2m", Decimal(2_000_000)),
        ("500 dollars", Decimal(500)),
        ("2000 USD", Decimal(2000)),
    ])
    def test_amounts(self, text, expected):
        (value,) = normalize(text)
        assert value.kind == "money" and value.canonical == expected

    def test_the_currency_is_kept_and_not_assumed(self):
        """`£500` is not 500 USD, and a normaliser that dropped the symbol
        would make every non-dollar prompt silently wrong rather than
        unsupported."""
        assert normalize("£500")[0].unit == "GBP"
        assert normalize("€500")[0].unit == "EUR"
        assert normalize("$500")[0].unit == "USD"

    def test_a_bare_number_is_not_money(self):
        """The discriminating opposite. If any digit became an amount, every
        window and every duration would also be one."""
        assert not [v for v in normalize("hold 90 shares") if v.kind == "money"]


class TestSplitsAndPercentages:
    def test_a_percentage_is_a_fraction(self):
        (value,) = normalize("60%")
        assert value.kind == "percentage" and value.canonical == Decimal("0.6")

    def test_a_split_is_ordered_weights_not_two_numbers(self):
        """Which sleeve gets which is the entire content of "60/40". Flattening
        it to a set loses the sentence."""
        (value,) = normalize("a 60/40 portfolio")
        assert value.kind == "ratio" and value.canonical == (60, 40)

    def test_three_way_splits_survive(self):
        (value,) = normalize("70/20/10 across the sleeves")
        assert value.canonical == (70, 20, 10)

    def test_a_ratio_that_is_not_a_split_is_left_alone(self):
        """`3/4` is not a portfolio, and `12/25` is probably a date. Guessing
        which would be the substitution this project refuses — so the rule is
        arithmetic: parts summing to 100 are a split, and nothing else is."""
        assert not [v for v in normalize("due 12/25") if v.kind == "ratio"]
        assert not [v for v in normalize("3/4 of the way") if v.kind == "ratio"]


class TestDurationsAgainstWindows:
    def test_a_holding_period_is_a_duration(self):
        (value,) = normalize("hold the bonus for 90 days")
        assert value.kind == "duration" and value.canonical == 90

    def test_a_moving_average_window_is_not_a_duration(self):
        """The plan's own falsification case: *"hold annual bonus for 90 days"
        read as an MA window must fail*, and its mirror. One span, one
        reading."""
        (value,) = normalize("buy below the 90-day moving average")
        assert value.kind == "moving_average_window"
        assert value.canonical == 90

    def test_both_in_one_sentence_stay_apart(self):
        """The case that a single-pass regex gets wrong, and the reason the
        window pass runs first and claims its span."""
        found = kinds("hold the bonus for 90 days, then buy below the "
                      "200-day moving average")
        assert ("duration", 90, "days") in found
        assert ("moving_average_window", 200, "day") in found
        assert len(found) == 2, f"one span, one reading: {found}"

    def test_years_and_months_become_days(self):
        assert normalize("over 5 years")[0].canonical == 365 * 5
        assert normalize("for 18 months")[0].canonical == 30 * 18


class TestEverySpanPointsAtItsOwnInput:
    @pytest.mark.parametrize("text", [
        "$1,500 monthly", "a 60/40 portfolio", "the 200-day moving average",
        "for 5 years", "60%",
    ])
    def test_the_span_is_the_text_it_was_read_from(self, text):
        """A normalisation that cannot point at its own input cannot be
        checked, and every falsification case in the plan is about which
        characters produced which value."""
        for value in normalize(text):
            assert text[value.start_char:value.end_char] == value.source_span
            assert value.source_span.strip()

    def test_values_come_back_in_reading_order(self):
        """Four kinds now: `cadence` joined when worded periods became
        normalisable. Reading order is what makes a span checkable against the
        text, so it is asserted rather than assumed."""
        found = normalize("put $500 monthly into a 60/40 split for 10 years")
        assert [v.kind for v in found] == ["money", "cadence", "ratio",
                                           "duration"]
        assert [v.start_char for v in found] == sorted(
            v.start_char for v in found)
