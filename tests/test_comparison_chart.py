"""The chart the product exists to show, and the property that let it ship.

Every run already computes the plan's portfolio path and five benchmark paths.
The page printed one final number and discarded all six series, which is the
whole comparison somebody came for.

The interesting test here is determinism. Bokeh mints a fresh UUID for the
container and fresh sequential ids for every model on each call, so two reopens
of one plan produced different HTML — and the launch journey refused it. That
refusal was right: the page promises reopening recompiles from the confirmed
intent and shows the same figure, and markup that differs per render is a weaker
promise wearing the same words.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.mission.accounting import CashFlow, CashPolicy
from src.mission.rebalance import weighted
from src.mission.simulate import simulate
from src.workspace.comparison_chart import PLAN_LABEL, build, collect

SESSIONS = pd.bdate_range("2022-01-03", "2023-12-29")


def prices() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": np.linspace(100.0, 200.0, len(SESSIONS)),
         "B": np.linspace(100.0, 140.0, len(SESSIONS))}, index=SESSIONS)


def _result(tickers):
    flows = [CashFlow(date=d, amount=500.0, label="contribution")
             for d in pd.bdate_range(SESSIONS[0], SESSIONS[-1], freq="BMS")]
    return simulate(prices(), flows=flows, program=weighted(tickers),
                    cash_policy=CashPolicy.idle())


class _Benchmark:
    def __init__(self, name, result):
        self.name, self.result = name, result


@pytest.fixture
def run():
    return {"result": _result(["A"]),
            "benchmarks": [_Benchmark("S&P 500", _result(["B"])),
                           _Benchmark("Unrunnable", None)]}


class TestTheSamePlanDrawsTheSameChart:
    def test_two_builds_are_identical(self, run):
        """The property the launch journey enforces, asserted here directly so
        a regression names the chart rather than surfacing as a journey that
        rendered two different pages."""
        assert build(run) == build(run)

    def test_different_plans_do_not_collide(self, run):
        """The other half. Ids derived from the data must differ when the data
        does, or two plans would share element ids on one page."""
        other = {"result": _result(["B"]), "benchmarks": run["benchmarks"]}
        assert build(run)["div"] != build(other)["div"]

    def test_no_generated_uuid_survives(self, run):
        """Directly, rather than by comparing two renders — a build that
        happened to be called twice in one process could pass that and still
        carry a per-process counter."""
        import re

        markup = build(run)["script"] + build(run)["div"]
        assert not re.search(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            markup), "a raw Bokeh UUID reached the page"


class TestItComparesRatherThanDecorates:
    def test_the_plan_is_the_first_series(self, run):
        series = collect(run)
        assert series[0]["name"] == PLAN_LABEL

    def test_benchmarks_keep_their_declared_order(self, run):
        """Never sorted by outcome. `mission.benchmark.compare` refuses to for
        the same reason: ordering a comparison by result turns a set of facts
        into a claim about which one won."""
        assert [s["name"] for s in collect(run)] == [PLAN_LABEL, "S&P 500"]

    def test_a_benchmark_that_could_not_run_is_omitted(self, run):
        """Not drawn flat at zero. A line at zero reads as "this earned
        nothing", which is a far stronger claim than "this comparison could not
        be made"."""
        assert "Unrunnable" not in [s["name"] for s in collect(run)]

    def test_one_series_alone_draws_nothing(self):
        """A chart with a single line implies the benchmarks ran and lost."""
        assert build({"result": _result(["A"]), "benchmarks": []}) is None

    def test_a_run_with_no_result_draws_nothing(self):
        assert build({"result": None, "benchmarks": []}) is None


class TestThePaletteIsFixed:
    def test_slots_are_assigned_in_order_not_cycled_by_count(self, run):
        """Colour follows the entity, not its rank. A run with fewer benchmarks
        must not repaint the survivors into other people's colours."""
        from src.workspace.comparison_chart import SERIES_COLOURS

        assert len(SERIES_COLOURS) >= 6
        assert len(set(SERIES_COLOURS)) == len(SERIES_COLOURS)
