"""The comparison chart, checked against the paths it claims to draw.

The graph on the result page is the product: the plan's own value beside the
same contributions bought and held in five benchmarks. "The engine is right" and
"the chart is right" are different claims — a faithful number drawn as the wrong
curve, or a benchmark computed wrong, is a lie the evaluation tests cannot see.

Three things have to hold, and each is checked against something the chart does
not define:

  * **the drawn curves are the engine's paths** — parsed back out of Bokeh's
    embedded document and matched to the series that went in, so serialisation
    cannot quietly drop or distort a line;
  * **each benchmark is the buy-and-hold it says it is** — the single-instrument
    ones reconstructed from scratch over the raw snapshot, the basket and all of
    them reconstructed from their own trades, so a benchmark computed wrong is
    caught rather than drawn confidently;
  * **the embed will render** — the document names the Bokeh version the page
    must load, and a mismatch there is the blank-chart bug this guards against.
"""
from __future__ import annotations

import base64
import json
import re
import struct

import numpy as np
import pandas as pd
import pytest

from runtime_contracts import Author, IntentField, VerifiedIntent

from src.evaluation.core import evaluate_plan
from src.mission.accounting import CashPolicy
from src.mission.benchmark import compare
from src.mission.from_intent import compile_intent
from src.workspace.comparison_chart import build, collect
from src.workspace.run_boundary import _benchmark_specs

FIXTURE = "tests/fixtures/prices_synthetic.parquet"
COST_RATE = 10.0 / 10_000.0

PLANS = {
    "risk_parity": {"amount": "500", "cadence": "monthly",
                    "allocation_method": "risk_parity",
                    "periodic_rebalancing": "quarterly"},
    "momentum": {"amount": "500", "cadence": "monthly",
                 "allocation_method": "time_series_momentum",
                 "periodic_rebalancing": "quarterly"},
    "single_fund": {"amount": "500", "cadence": "monthly", "assets": "SPY"},
}


@pytest.fixture(scope="module")
def prices():
    return pd.read_parquet(FIXTURE)


def _run(fields, prices):
    """The `run` the result page draws — built here from the fixture exactly as
    `run_boundary.execute_compiled_plan` builds it, so no market-data
    deployment config is needed to exercise the chart."""
    built = {k: IntentField(value=v, author=Author.READER)
             for k, v in fields.items()}
    scenario = compile_intent(VerifiedIntent(
        objective="evaluate_investment_strategy", produced_by="graph",
        utterance_ref="graph", fields=built, unresolved=()).seal()).scenario
    evaluated = evaluate_plan(scenario, prices)
    specs = _benchmark_specs(prices, list(evaluated.tradeable))
    benchmarks = compare(prices, flows=list(evaluated.flows),
                         cash_policy=CashPolicy.idle(), benchmarks=specs)
    return {"result": evaluated.result, "benchmarks": benchmarks}


# --- reading the numbers back out of the drawn document --------------------

def _decode(arr):
    """A Bokeh column: a plain list, or a serialised ndarray of base64 bytes."""
    if isinstance(arr, list):
        return [float(v) for v in arr if isinstance(v, (int, float))]
    if isinstance(arr, dict):
        holder = arr.get("array") or arr.get("data")
        raw = None
        if isinstance(holder, dict) and "data" in holder:
            raw = base64.b64decode(holder["data"])
        elif isinstance(holder, str):
            raw = base64.b64decode(holder)
        if raw is None:
            return []
        fmt = {"float64": "d", "float32": "f", "int64": "q",
               "int32": "i"}.get(arr.get("dtype", "float64"), "d")
        size = struct.calcsize(fmt)
        count = len(raw) // size
        return list(struct.unpack("<%d%s" % (count, fmt), raw[:count * size]))
    return []


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _drawn_value_curves(script: str):
    """Every plotted value-curve in the embed, dates excluded.

    Bokeh 3.x inlines the document as `const docs_json = '{...}'` and tags each
    source `{"name": "ColumnDataSource"}` with `data` as a map of entries. The
    date axis arrives as millisecond integers (far above any dollar figure), so
    a column whose values stay below 1e10 is a portfolio curve, not the x-axis.
    """
    match = re.search(r"docs_json = '(.*?)';", script, re.S)
    assert match, "the embed does not carry an inline Bokeh document"
    document = json.loads(match.group(1))
    curves = []
    for node in _walk(document):
        if not (isinstance(node, dict)
                and node.get("name") == "ColumnDataSource"):
            continue
        data = (node.get("attributes") or {}).get("data") or {}
        entries = data.get("entries") if isinstance(data, dict) else None
        pairs = entries if entries else (
            list(data.items()) if isinstance(data, dict) else [])
        for key, arr in pairs:
            values = _decode(arr)
            if len(values) > 2 and max(abs(v) for v in values) < 1e10:
                curves.append(values)
    return curves


# --- an independent buy-and-hold, from the snapshot alone -------------------

def _contribution_days(prices):
    idx = prices.index
    return set(prices.groupby([idx.year, idx.month]).apply(
        lambda g: g.index.min()).values)


def _buy_and_hold(prices, ticker, amount=500.0):
    """One instrument, every contribution invested and never sold — the path a
    single-instrument benchmark must produce, computed from scratch."""
    days = _contribution_days(prices)
    cash = shares = 0.0
    pending = False
    path = pd.Series(index=prices.index, dtype=float)
    for day in prices.index:
        if pending:
            price = float(prices.at[day, ticker])
            notional = cash / (1.0 + COST_RATE)
            shares += notional / price
            cash -= notional * (1.0 + COST_RATE)
            pending = False
        if day in days:
            cash += amount
        if cash > 1e-9:
            pending = True
        path.loc[day] = shares * float(prices.at[day, ticker]) + cash
    return path


def _reconstruct_nav(path, prices):
    """NAV from flows and fills alone — the valuation check, per benchmark."""
    sessions = path.value.index
    price_at = prices.reindex(sessions)
    fills_by_day = {}
    for fill in path.fills:
        fills_by_day.setdefault(pd.Timestamp(fill.date), []).append(fill)
    cash = 0.0
    holdings = {}
    nav = pd.Series(index=sessions, dtype=float)
    for session in sessions:
        if session in path.flows.index:
            cash += float(path.flows.loc[session])
        for fill in fills_by_day.get(session, ()):
            holdings[fill.ticker] = holdings.get(fill.ticker, 0.0) + fill.shares
            cash -= fill.notional + fill.cost
        invested = sum(
            sh * float(price_at.at[session, tk])
            for tk, sh in holdings.items()
            if sh != 0.0 and tk in price_at.columns
            and np.isfinite(price_at.at[session, tk]))
        nav.loc[session] = cash + invested
    return nav


# --- the tests --------------------------------------------------------------

@pytest.mark.parametrize("name", list(PLANS))
def test_the_drawn_curves_are_the_engine_paths(name, prices):
    """Parse the numbers back out of the document and match them to the series
    that went in. Serialisation may not drop or distort a line."""
    run = _run(PLANS[name], prices)
    series = collect(run)
    chart = build(run)
    assert chart is not None, f"{name}: no chart built"

    drawn = _drawn_value_curves(chart["script"])
    assert len(drawn) == len(series), (
        f"{name}: drew {len(drawn)} curves for {len(series)} series")

    drawn_finals = sorted(round(c[-1], 2) for c in drawn)
    engine_finals = sorted(round(s["values"][-1], 2) for s in series)
    assert drawn_finals == engine_finals, (
        f"{name}: drawn finals {drawn_finals} != engine {engine_finals}")

    # And the whole curve, not only its end: every engine series appears drawn.
    drawn_by_final = {round(c[-1], 2): c for c in drawn}
    for one in series:
        curve = drawn_by_final[round(one["values"][-1], 2)]
        assert len(curve) == len(one["values"])
        worst = max(abs(a - b) for a, b in zip(curve, one["values"]))
        assert worst < 1e-6, f"{name}: '{one['name']}' curve distorted by {worst}"


@pytest.mark.parametrize("ticker,label", [("SPY", "S&P 500"),
                                          ("QQQ", "Nasdaq 100"),
                                          ("AGG", "Aggregate bonds")])
def test_single_instrument_benchmarks_are_correct_buy_and_hold(ticker, label,
                                                               prices):
    """Each single-instrument benchmark, reconstructed from scratch over the raw
    snapshot and matched to the path the engine drew for it."""
    run = _run(PLANS["risk_parity"], prices)
    drawn = {getattr(b, "name"): b.result.path.value for b in run["benchmarks"]}
    engine = drawn[f"Contribute to {label}"]
    reference = _buy_and_hold(prices, ticker)

    diff = (reference - engine).abs()
    scale = engine.abs().clip(lower=1.0)
    assert float((diff / scale).max()) < 1e-9, (
        f"{label}: the drawn benchmark is not a faithful buy-and-hold of "
        f"{ticker} (max ${float(diff.max()):.4f})")


def test_the_cash_benchmark_is_exactly_the_contributions(prices):
    run = _run(PLANS["risk_parity"], prices)
    cash = {getattr(b, "name"): b.result for b in run["benchmarks"]}["Hold cash"]
    contributed = float(cash.path.flows.sum())
    assert abs(float(cash.path.value.iloc[-1]) - contributed) < 1e-6


@pytest.mark.parametrize("name", ["risk_parity", "momentum"])
def test_every_benchmark_valuation_reconstructs(name, prices):
    """The basket included: each benchmark's value rebuilt from its own trades
    and the raw prices, matched to what it drew."""
    run = _run(PLANS[name], prices)
    for benchmark in run["benchmarks"]:
        path = benchmark.result.path
        if not path.fills:               # hold-cash has no trades; covered above
            continue
        nav = _reconstruct_nav(path, prices)
        diff = (nav - path.value).abs()
        scale = path.value.abs().clip(lower=1.0)
        assert float((diff / scale).max()) < 1e-9, (
            f"{name}: '{getattr(benchmark, 'name', '?')}' valuation does not "
            "match its trades")


def test_the_basket_splits_each_contribution_across_the_universe(prices):
    """The basket benchmark buys the whole universe, not a subset — its trades
    on a contribution day name every tradeable instrument, roughly equally."""
    run = _run(PLANS["risk_parity"], prices)
    basket = {getattr(b, "name"): b for b in run["benchmarks"]}[
        "Your basket, bought and held"]
    universe = set(run["result"].path.holdings.columns)
    traded = {f.ticker for f in basket.result.path.fills}
    assert traded == universe, (
        f"basket traded {traded}, universe is {universe}")


@pytest.mark.parametrize("name", list(PLANS))
def test_the_embed_targets_the_installed_bokeh_version(name, prices):
    """The document names the version the page must load. A mismatch is the
    blank-chart bug: BokehJS refuses a document minted by another version."""
    import bokeh

    run = _run(PLANS[name], prices)
    chart = build(run)
    assert chart["version"] == bokeh.__version__, (
        f"embed built with {chart['version']}, page would load "
        f"{bokeh.__version__}")
    assert f'"version":"{bokeh.__version__}"' in chart["script"]


@pytest.mark.parametrize("name", list(PLANS))
def test_the_plan_is_the_first_curve(name, prices):
    """The subject is drawn first and labelled as the plan, not a benchmark."""
    run = _run(PLANS[name], prices)
    series = collect(run)
    assert series[0]["name"] == "Your strategy"
    assert len(series) >= 2, "a comparison needs the plan and one benchmark"
