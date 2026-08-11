"""Whether `time_weighted_returns` computes the TWR Lean defines.

    Quantify/Returns/TimeWeighted.lean   the contract, proven
    accounting.time_weighted_returns     the implementation
    MissionResult.time_weighted_annualized   what a user is shown

Verified because it travels a governed result path, which is the rule the last
two slices established: Lean certifies semantics that already reach a Quantify
result, and does not create product semantics because a metric would be useful.

**The convention is part of the contract.** A flow lands at the start of its
session and is uninvested for that session, so

    r_t = (V_t − F_t) / V_{t−1} − 1

and the growth factor `(V_t − F_t) / V_{t−1}` is exactly Lean's
`finish / start` with `start = V_{t−1}` and `finish = V_t − F_t`. Getting that
boundary wrong is one of the mutations below, because it produces a plausible
number rather than an error.

**Intermediates, not just the headline.** Two different implementations can
agree on an annualized figure and disagree about every subperiod inside it, so
the per-session growth factors are compared as well.
"""
from __future__ import annotations

import pytest


def _returns(values, flows=None):
    import pandas as pd

    from src.mission.accounting import time_weighted_returns

    flows = flows if flows is not None else [0.0] * len(values)
    return time_weighted_returns(pd.Series([float(v) for v in values]),
                                 pd.Series([float(f) for f in flows]))


def _chained(values, flows=None) -> float:
    return float((1.0 + _returns(values, flows)).prod()) - 1.0


class TestTheSubperiodBoundaries:
    """Where the flow sits relative to the session decides the number."""

    VALUES = [100.0, 110.0, 171.0, 188.1]
    FLOWS = [0.0, 0.0, 50.0, 0.0]

    def test_every_session_earned_the_same_ten_percent(self):
        """The intermediates. A contribution of 50 arrives at session 2 and is
        uninvested that session, so the invested base stays 110 and the growth
        is 1.1 — the same as the sessions either side of it."""
        assert list(_returns(self.VALUES, self.FLOWS)) == \
            pytest.approx([0.1, 0.1, 0.1])

    def test_the_chain_is_the_product_not_the_sum(self):
        assert _chained(self.VALUES, self.FLOWS) == pytest.approx(1.1 ** 3 - 1)
        assert _chained(self.VALUES, self.FLOWS) != pytest.approx(0.3)

    def test_the_contribution_is_not_performance(self):
        """The property TWR exists for, and the one the Lean contract proves in
        general: a flow at a subperiod boundary cannot change the return."""
        naive = (188.1 - 100.0 - 50.0) / 100.0
        assert naive == pytest.approx(0.381)
        assert _chained(self.VALUES, self.FLOWS) == pytest.approx(0.331)
        assert naive > _chained(self.VALUES, self.FLOWS)

    def test_the_flow_size_does_not_move_it(self):
        """Sampled across sizes. Each series is built so the market return is
        10% a session whatever arrives, so a figure that moved with the flow
        would be counting the depositor's money."""
        for flow in (0.0, 50.0, 500.0, 10_000.0):
            v1 = 110.0
            v2 = (v1 * 1.1) + flow
            v3 = v2 * 1.1
            assert _chained([100.0, v1, v2, v3], [0.0, 0.0, flow, 0.0]) == \
                pytest.approx(1.1 ** 3 - 1)


class TestFlowsInBothDirections:
    def test_a_withdrawal_is_not_a_loss(self):
        """Money leaving is the depositor's decision, not the strategy's
        performance. A withdrawal is a negative flow and the same boundary
        rule applies."""
        v1 = 110.0
        v2 = v1 * 1.1 - 30.0
        v3 = v2 * 1.1
        assert _chained([100.0, v1, v2, v3], [0.0, 0.0, -30.0, 0.0]) == \
            pytest.approx(1.1 ** 3 - 1)

    def test_several_flows_across_several_subperiods(self):
        values, flows, v = [100.0], [0.0], 100.0
        for flow in (25.0, -10.0, 40.0):
            v = v * 1.05 + flow
            values.append(v)
            flows.append(flow)
        assert _chained(values, flows) == pytest.approx(1.05 ** 3 - 1)


class TestTheControls:
    """Without these, an implementation returning zero would pass everything
    above that only checks a flow cannot move the answer."""

    def test_a_flat_series_returns_nothing(self):
        assert _chained([100.0, 100.0, 100.0]) == pytest.approx(0.0)

    def test_a_loss_is_negative(self):
        assert _chained([100.0, 90.0]) == pytest.approx(-0.1)

    def test_up_then_down_loses_one_percent(self):
        """The compounding control the Lean contract also states: +10% then
        −10% is not zero."""
        assert _chained([100.0, 110.0, 99.0]) == pytest.approx(-0.01)


class TestAnnualizationIsASeparateStep:
    """Verified apart from the chain, because the two fail differently: a
    wrong horizon scales a correct chain, and a wrong chain survives any
    horizon."""

    def _annualized(self, values, periods_per_year, flows=None):
        import pandas as pd

        from src.mission.simulate import MissionResult

        returns = _returns(values, flows)
        result = MissionResult.__new__(MissionResult)
        object.__setattr__(result, "time_weighted", returns)
        object.__setattr__(result, "periods_per_year", periods_per_year)
        return result.time_weighted_annualized

    def test_one_year_of_sessions_annualizes_to_the_chain(self):
        """Four sessions at 10% with `periods_per_year=4` is one year, so the
        annualized figure is the chained one."""
        values = [100.0 * 1.1 ** i for i in range(5)]
        assert self._annualized(values, 4) == pytest.approx(1.1 ** 4 - 1)

    def test_half_a_year_is_scaled_up(self):
        values = [100.0 * 1.1 ** i for i in range(5)]
        assert self._annualized(values, 8) == pytest.approx(1.1 ** 8 - 1)

    def test_an_empty_series_is_undefined_not_zero(self):
        """`None`, and it must stay `None` all the way to the surface. The
        money-weighted column rendered its missing value as +0.00% for months,
        which reads as "broke even" for a number that does not exist."""
        assert self._annualized([100.0], 252) is None

    def test_a_portfolio_that_reached_zero_is_undefined_not_minus_one(self):
        assert self._annualized([100.0, 0.0], 252) is None

    def test_the_surface_does_not_render_undefined_as_zero(self):
        """Checked in the templates, because that is where the substitution
        happened before."""
        from pathlib import Path

        templates = Path(__file__).resolve().parent.parent / "src" \
            / "workspace" / "templates"
        for name in ("plan.html", "new.html", "_comparison.html"):
            text = (templates / name).read_text()
            assert "time_weighted_annualized or 0" not in text, name
            assert "time_weighted_annualized is not none" in text, name


class TestTheResultPathUsesTheVerifiedImplementation:
    """A proof about one helper is worth nothing if the result path quietly
    switches to another."""

    def test_the_field_is_produced_from_the_verified_function(self):
        import inspect

        from src.mission import simulate

        source = inspect.getsource(simulate)
        assert "time_weighted=time_weighted_returns(" in source, (
            "MissionResult.time_weighted is no longer built by "
            "accounting.time_weighted_returns; the conformance lane is "
            "certifying a function the result path does not use")

    def test_the_annualized_field_is_derived_from_that_series(self):
        import inspect

        from src.mission.simulate import MissionResult

        source = inspect.getsource(MissionResult.time_weighted_annualized.fget)
        assert "self.time_weighted" in source

    def test_and_it_reaches_the_serialized_result(self):
        import inspect

        from src.mission.simulate import MissionResult

        assert '"time_weighted_annualized"' in \
            inspect.getsource(MissionResult.to_json)
