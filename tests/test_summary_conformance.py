"""Whether the headline money numbers trace to one authority.

    contributed   the positive flows
    withdrawn     the negative flows, as a magnitude
    final_value   the last portfolio value
    gain          final_value − (contributed − withdrawn)

Every one is a view of `path.flows` and `path.value`. That is the property:
a summary recomputed from somewhere else can disagree with the ledger it claims
to summarise, and the disagreement surfaces as a headline number nobody can
reconcile.

The Lean statement is `Quantify/Summary.lean`. This checks the engine agrees,
and — because a fixture of contributions only would hold in the
accumulation-only world this project spent a long time escaping — every case
here moves money in both directions.
"""
from __future__ import annotations

import pytest


def _path(flows, values):
    """A `PortfolioPath` carrying just what the summary reads."""
    import pandas as pd

    from src.mission.accounting import PortfolioPath

    index = range(len(values))
    return PortfolioPath(
        value=pd.Series([float(v) for v in values], index=index),
        cash=pd.Series([0.0] * len(values), index=index),
        holdings=pd.DataFrame(index=index),
        flows=pd.Series([float(f) for f in flows], index=index))


class TestTheTwoHalvesRecombineIntoTheTotal:
    """`contributed − withdrawn` must be the flow series itself.

    The engine computes the halves with two separate filters. Nothing else
    checks that they partition the series, so a filter on `>= 0` or a dropped
    zero would leave both numbers looking plausible.
    """

    @pytest.mark.parametrize("flows", [
        [1000.0, -300.0, 500.0],
        [1000.0, 0.0, -400.0],
        [500.0, -800.0],
        [0.0, 0.0],
        [-250.0, 250.0],
    ])
    def test_the_halves_recombine(self, flows):
        path = _path(flows, [0.0] * len(flows))
        assert path.contributed - path.withdrawn == pytest.approx(sum(flows))

    def test_a_zero_flow_is_in_neither_half(self):
        """The case that separates a correct split from a filter on `>= 0`."""
        path = _path([1000.0, 0.0, -400.0], [0.0, 0.0, 0.0])
        assert path.contributed == pytest.approx(1000.0)
        assert path.withdrawn == pytest.approx(400.0)

    def test_withdrawn_is_a_magnitude(self):
        """Reported positive. "withdrawn: −500" reads as money arriving, and
        the sign convention meets the flow series in exactly one place."""
        path = _path([-500.0], [0.0])
        assert path.withdrawn == pytest.approx(500.0)
        assert path.contributed == pytest.approx(0.0)


class TestGainIsValueLessTheFlowTotal:
    def test_with_money_moving_both_ways(self):
        """1000 in, 300 out, 500 in — net 1200 — ending at 1500."""
        from src.mission.simulate import MissionResult

        path = _path([1000.0, -300.0, 500.0], [1000.0, 700.0, 1500.0])
        result = MissionResult.__new__(MissionResult)
        object.__setattr__(result, "path", path)

        assert result.final_value == pytest.approx(1500.0)
        assert result.gain == pytest.approx(300.0)
        assert result.gain == pytest.approx(
            result.final_value - sum([1000.0, -300.0, 500.0]))

    def test_taking_out_more_than_was_put_in(self):
        """Net contributed is negative and the gain still reconciles — the
        case an accumulation-only fixture never reaches."""
        from src.mission.simulate import MissionResult

        path = _path([500.0, -800.0], [500.0, 0.0])
        result = MissionResult.__new__(MissionResult)
        object.__setattr__(result, "path", path)

        assert path.contributed - path.withdrawn == pytest.approx(-300.0)
        assert result.gain == pytest.approx(300.0)

    def test_final_value_is_the_last_value_not_a_recomputation(self):
        from src.mission.simulate import MissionResult

        path = _path([100.0, 100.0], [100.0, 250.0])
        result = MissionResult.__new__(MissionResult)
        object.__setattr__(result, "path", path)
        assert result.final_value == pytest.approx(250.0)


class TestAllFourReadOneAuthority:
    """A proof about an identity is worth nothing if a later change routes one
    of the four through a second source."""

    def test_the_summary_properties_read_the_flow_and_value_series(self):
        import inspect

        from src.mission.accounting import PortfolioPath

        for name, source in (("contributed", "flows"), ("withdrawn", "flows"),
                             ("terminal_value", "value")):
            body = inspect.getsource(getattr(PortfolioPath, name).fget)
            assert f"self.{source}" in body, f"{name} no longer reads {source}"

    def test_gain_is_derived_and_not_stored(self):
        import inspect

        from src.mission.simulate import MissionResult

        body = inspect.getsource(MissionResult.gain.fget)
        assert "self.final_value" in body
        assert "self.path.contributed" in body
        assert "self.path.withdrawn" in body

    def test_the_lean_file_states_the_same_identity(self):
        from pathlib import Path

        lean = (Path(__file__).resolve().parent.parent / "formal" / "Quantify"
                / "Summary.lean")
        if not lean.exists():
            pytest.skip("Summary.lean is absent")
        text = lean.read_text()
        assert "finalValue - netContributed flows" in text
        assert "netContributed flows = total flows" in text
        assert "def bothDirections : List Int := [100000, -30000, 50000]" in text

    def test_the_lean_fixtures_use_the_same_numbers(self):
        """Scaled: Lean works in minor units, so 1000.00 is 100000."""
        path = _path([1000.0, -300.0, 500.0], [0.0, 0.0, 1500.0])
        assert round(path.contributed * 100) == 150000
        assert round(path.withdrawn * 100) == 30000
        assert round((path.contributed - path.withdrawn) * 100) == 120000
