"""Written numbers that carry their magnitude in a letter.

`2.5k` is two and a half thousand. The numeric comparison used to strip every
non-digit and trust what was left, which turned it into `2.5` — not a failure to
compare but a wrong number, a thousand times too small.

It surfaced as a *disagreement*, which is the part worth keeping in mind: the
deterministic path read 2500, the model returned `2.5k`, they differed and the
pipeline asked a question. The safe outcome arrived by luck. Two witnesses that
both wrote `2.5k` would have compared 2.5 against 2.5, agreed, and settled a
contribution three orders of magnitude too small with no question and no
refusal — a silent reduction of exactly the kind this project exists to make
impossible.

It went unnoticed because the reader in use wrote `$2,500`, and the currency
symbol sends the string down the normaliser's path instead. Changing the reader
to one that writes the amount bare is what exposed it.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from src.discovery.adapter import NORMALIZERS


def same_value_for_mode(one, other, mode, dimension=""):
    """The runtime's comparison, with Quantify's normalisers and the mode a
    dimension declares. `dimension` selects the mode where one is given, since
    `12m` is twelve million for an amount and twelve periods for a window."""
    from discovery_runtime import same_value

    from src.discovery.adapter import compare_as

    if dimension:
        mode = compare_as(dimension)
    return same_value(one, other, mode, normalizers=NORMALIZERS)


class TestAMagnitudeSuffixIsNotDiscarded:
    @pytest.mark.parametrize("written,value", [
        ("2.5k", 2_500), ("2.5K", 2_500), ("1k", 1_000), ("750k", 750_000),
        ("1.2m", 1_200_000), ("3M", 3_000_000), ("1b", 1_000_000_000),
        ("2bn", 2_000_000_000), ("1,5k", 15_000),
    ])
    def test_it_scales(self, written, value):
        assert same_value_for_mode(written, Decimal(value), "NUMBER"), (
            f"{written!r} did not compare equal to {value}")

    def test_the_old_behaviour_would_have_failed_this(self):
        """The specific regression, stated as the number it produced. Stripping
        the letter gave `2.5`, so this pair compared equal and a plan would have
        contributed 2.5 rather than 2500."""
        assert not same_value_for_mode("2.5k", Decimal("2.5"), "NUMBER")

    def test_a_currency_symbol_still_works(self):
        """The path that always worked, kept under test so the new branch is
        not the only one exercised."""
        assert same_value_for_mode("£2.5k", Decimal(2_500), "NUMBER")
        assert same_value_for_mode("$500", "500", "NUMBER")

    def test_different_magnitudes_are_not_equal(self):
        """Nothing here may make two different amounts equal, which is the rule
        the whole comparison is written under."""
        assert not same_value_for_mode("2.5k", Decimal(2_500_000), "NUMBER")
        assert not same_value_for_mode("1k", "1m", "NUMBER")


class TestTheAmbiguousLetterIsReadByDimension:
    """`m` is the one magnitude letter that means two things, and the fix for
    `2.5k` broke a case that had been answered correctly for months.

    A reader wrote `12m` for a moving-average window — twelve months — and the
    new suffix rule scaled it to twelve million, which disagreed with syntax's
    12 and dropped the case out of the answerable set. The lesson is the one
    this project keeps relearning: a new literal class collides with its
    neighbours, so the neighbours get tested.
    """

    def test_a_window_reads_it_as_periods(self):
        assert same_value_for_mode("12m", "12", "NUMBER", "moving_average_window")

    def test_an_amount_reads_it_as_millions(self):
        assert same_value_for_mode("12m", "12000000", "NUMBER", "amount")
        assert not same_value_for_mode("12m", "12", "NUMBER", "amount")

    def test_the_unambiguous_letters_scale_everywhere(self):
        """`k`, `b` and `bn` mean one thing wherever they appear, so the
        dimension does not change them. If it did, a window would be the place
        `2.5k` silently became 2.5 again."""
        for dimension in ("amount", "moving_average_window", ""):
            assert same_value_for_mode("2.5k", "2500", "NUMBER", dimension), dimension
            assert not same_value_for_mode("2.5k", "2.5", "NUMBER", dimension), dimension

    @pytest.mark.parametrize("written", ["12m", "12mo", "6w", "3y"])
    def test_an_abbreviated_period_keeps_its_count(self, written):
        digits = "".join(c for c in written if c.isdigit())
        assert same_value_for_mode(written, digits, "NUMBER", "moving_average_window")

    def test_a_spelled_out_unit_is_still_the_open_schema_gap(self):
        """`12 months` does *not* compare equal to `12`, and this test asserts
        that rather than hiding it.

        The normaliser answers first and reads it as a duration of 360 days,
        which for a window dimension with no declared unit is one defensible
        reading of two. Making it equal here would be choosing the unit in the
        comparison layer, which is the wrong place and the wrong authority —
        `moving_average_window has no unit` is in docs/Benchmark-Queue.md, and
        it stays a schema question.
        """
        assert not same_value_for_mode("12 months", "12", "NUMBER",
                              "moving_average_window")


class TestWhatIsAndIsNotGuaranteed:
    """The scope of the fix, stated so nobody reads more into it.

    Recognised magnitude suffixes are honoured. An unrecognised unit still
    falls through to its digits, exactly as it always did — `500 shares`
    compares equal to `500`. That is a separate defect from this one, it is not
    fixed here, and an attempt to fix it in the same place took out 131 tests:
    refusing every value containing a letter refuses most of the corpus.
    """

    def test_a_plain_number_is_unaffected(self):
        assert same_value_for_mode("2500", Decimal(2_500), "NUMBER")
        assert same_value_for_mode("2,500", "2500", "NUMBER")

    def test_an_unrecognised_unit_still_falls_through_to_its_digits(self):
        """Recorded rather than asserted as desirable. If this ever changes,
        it should change deliberately and this test should say so."""
        assert same_value_for_mode("500 shares", "500", "NUMBER")
