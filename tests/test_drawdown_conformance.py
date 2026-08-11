"""Whether `_max_drawdown` computes the drawdown Lean defines.

    Quantify/Returns/Drawdown.lean   the definition
    evaluation/runner._max_drawdown  the implementation on a real result path

Formalised in that order and only because the implementation already exists.
Volatility was the intended slice and was dropped: nothing in the engine
computes one, so a proof of it would be a definition with nothing to conform
to, and adding the metric in order to have something to verify would reverse
the dependency.

**Sign, converted explicitly.** Lean states drawdown as a non-negative
magnitude — a quarter below the peak is `1/4`. The implementation returns
`min(curve/cummax - 1)`, which is `-1/4` for the same series. The conversion
happens here, once, rather than the two conventions meeting by accident.
"""
from __future__ import annotations

import pytest


def _returns_from(values):
    """A value series as the period returns the implementation expects."""
    import pandas as pd

    return pd.Series([values[i] / values[i - 1] - 1.0
                      for i in range(1, len(values))])


def _implementation(values) -> float:
    from src.evaluation.runner import _max_drawdown

    return abs(float(_max_drawdown(_returns_from(values))))


class TestTheImplementationAgreesWhereItCan:
    def test_the_recovering_series(self):
        """`100 → 120 → 90 → 110 → 130`. Peak 120, trough 90, so a quarter —
        on a series that finishes at a new high.

        The case that separates a correct implementation from three plausible
        wrong ones: loss from the start, terminal drawdown and current
        drawdown all give zero here.
        """
        assert _implementation([100, 120, 90, 110, 130]) == pytest.approx(0.25)

    def test_a_rising_series_has_none(self):
        assert _implementation([100, 110, 120, 130]) == pytest.approx(0.0)

    def test_and_the_reported_slice_of_the_crash_series(self):
        assert _implementation([100, 101, 102]) == pytest.approx(0.0)


class TestTheOpeningLevelIsMissingFromTheCurve:
    """The defect, found by running the implementation against the definition.

        curve = (1 + returns).cumprod()

    The curve begins at the *first return*, so the starting value is never in
    it. `cummax` therefore starts at the post-first-move level, and any fall
    from the opening level is measured against a peak that already reflects the
    fall — or is invisible entirely.

    A portfolio that halves on its first day and never recovers reports a
    maximum drawdown of zero.
    """

    def test_a_first_day_halving_reports_nothing(self):
        import pandas as pd

        from src.evaluation.runner import _max_drawdown

        assert _max_drawdown(pd.Series([-0.5])) == 0.0

    def test_a_fall_from_the_opening_level_is_understated(self):
        """`100 → 75 → 50` is a half below its peak. The implementation sees a
        third, because it takes 75 as the peak."""
        assert _implementation([100, 75, 50]) == pytest.approx(1 / 3)
        assert _implementation([100, 75, 50]) != pytest.approx(0.5)

    def test_and_a_crash_before_a_calm_stretch_vanishes(self):
        """`100 → 50 → 100 → 101 → 102` fell by half. The implementation
        reports nothing, because the curve starts at the bottom."""
        assert _implementation([100, 50, 100, 101, 102]) == pytest.approx(0.0)

    def test_the_definition_says_otherwise(self):
        """What Lean states for the same series, so the disagreement is
        recorded as a disagreement rather than as a tolerance."""
        from pathlib import Path

        lean = (Path(__file__).resolve().parent.parent / "formal" / "Quantify"
                / "Returns" / "Drawdown.lean")
        if not lean.exists():
            pytest.skip("Drawdown.lean is absent")
        text = lean.read_text()
        assert "maxDrawdown falls = 1 / 2" in text
        assert "maxDrawdown crashThenCalm = 1 / 2" in text

    def test_the_gap_is_recorded_where_somebody_will_look(self):
        from pathlib import Path

        doc = Path(__file__).resolve().parent.parent / "docs" / "Drawdown.md"
        if not doc.exists():
            pytest.skip("docs/Drawdown.md is absent")
        assert "_max_drawdown" in doc.read_text()


class TestTheFixWouldBeOneLine:
    """Stated as a test rather than applied, for the reason the MWR slice
    followed: a lane that finds a defect should not also choose the remedy.

    Prepending the opening level to the curve gives the definition's answer on
    every fixture above.
    """

    @staticmethod
    def _with_opening_level(values):
        import pandas as pd

        returns = _returns_from(values)
        curve = pd.concat([pd.Series([1.0]), (1.0 + returns).cumprod()],
                          ignore_index=True)
        return abs(float((curve / curve.cummax() - 1.0).min()))

    @pytest.mark.parametrize("values,expected", [
        ([100, 120, 90, 110, 130], 0.25),
        ([100, 110, 120, 130], 0.0),
        ([100, 75, 50], 0.5),
        ([100, 50, 100, 101, 102], 0.5),
        ([100, 101, 102], 0.0),
    ])
    def test_it_would_agree_with_the_definition(self, values, expected):
        assert self._with_opening_level(values) == pytest.approx(expected)
