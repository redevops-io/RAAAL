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


class TestTheOpeningLevelIsInTheCurve:
    """The defect this lane found, and the cases that prove it closed.

        curve = (1 + returns).cumprod()

    began at the first *return*, so the opening level was never in it and
    `cummax` started at the post-first-move value. A portfolio that halved on
    its first session reported a maximum drawdown of zero.

    Note which fixture caught it. `100 → 120 → 90 → 110 → 130` — the series
    chosen to separate correct implementations from three plausible wrong ones
    — conformed under the defect, because it rises first and the missing
    opening level never becomes the peak. It took a series that falls
    immediately.
    """

    def test_a_first_session_crash_is_reported(self):
        import pandas as pd

        from src.evaluation.runner import _max_drawdown

        assert abs(_max_drawdown(pd.Series([-0.5]))) == pytest.approx(0.5)

    @pytest.mark.parametrize("values,expected", [
        ([100, 50], 0.5),
        ([100, 75, 50], 0.5),
        ([100, 50, 100, 101, 102], 0.5),
        ([100, 120, 90, 110, 130], 0.25),
        ([100, 110, 120, 130], 0.0),
    ])
    def test_every_case_lean_exposed(self, values, expected):
        assert _implementation(values) == pytest.approx(expected)

    def test_removing_the_opening_level_breaks_conformance(self):
        """The mutation, kept. Without it these fixtures would pass for an
        implementation that had quietly reverted."""
        import pandas as pd

        def without_opening(values):
            returns = _returns_from(values)
            curve = (1.0 + returns).cumprod()
            return abs(float((curve / curve.cummax() - 1.0).min()))

        assert without_opening([100, 75, 50]) == pytest.approx(1 / 3)
        assert without_opening([100, 50, 100, 101, 102]) == pytest.approx(0.0)
        assert without_opening([100, 120, 90, 110, 130]) == pytest.approx(0.25)


class TestTheSemanticsAreVersioned:
    """`@1` and `@2` differ materially on any series that fell early.

    A stored result with no version cannot be told apart from one measured
    under `@1`, so recomputing history silently would present old evidence as
    though it had always been measured this way.
    """

    def test_the_build_declares_which_semantics_it_computes(self):
        from src.evaluation.runner import DRAWDOWN_SEMANTICS

        assert DRAWDOWN_SEMANTICS == "drawdown@2"

    def test_the_version_travels_with_the_number(self):
        import inspect

        from src.evaluation.runner import EvaluationResult

        source = inspect.getsource(EvaluationResult.to_json)
        assert '"drawdown_semantics"' in source
        assert '"max_drawdown"' in source

    def test_the_document_records_both_versions(self):
        from pathlib import Path

        doc = Path(__file__).resolve().parent.parent / "docs" / "Measures.md"
        if not doc.exists():
            pytest.skip("docs/Measures.md is absent")
        text = doc.read_text()
        assert "drawdown@1" in text and "drawdown@2" in text
