#!/usr/bin/env python3
"""For every strategy that evaluates, look at the graph it draws.

`strategy_evaluate_sweep` establishes which strategies produce a figure. This
goes one step further, to the thing the product exists to show: the comparison
chart — the plan's own path against the same contributions bought and held in a
handful of benchmarks (`workspace/comparison_chart`). A figure with no
comparison behind it is a number with nothing to judge it by.

Per evaluated strategy it reports:

  * chart      — the Bokeh comparison rendered at all (a real graph, not a table)
  * series     — how many lines it drew: 1 is the plan alone (no benchmark ran,
                 so there is nothing to compare), >=2 is an actual comparison
  * vs bench   — of the benchmarks that drew, how many the plan's final value
                 beat, read from the chart's own series rather than recomputed

A strategy that evaluates but draws no comparison (benchmarks all fell out for
want of price history over its window) is the finding this pass exists to
surface: the figure is real, the graph capability is not exercised.

    python ui-agent/graph_capability_sweep.py --url https://quantify.club \\
        --email pilot@quantify.club --password '...' [--limit 5] [--only-evaluated]

Live reads; on demand, not CI.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

from regression_smoke import sign_in                          # noqa: E402
from strategy_evaluate_sweep import evaluate_one, EVALUATED   # noqa: E402


@dataclass
class Graph:
    key: str
    figure: str = ""
    status: str = ""
    chart: bool = False
    rendered: bool = False
    series: int = 0
    finals: List[float] = field(default_factory=list)
    plan_final: Optional[float] = None
    beat: int = 0
    of: int = 0
    note: str = ""


def _num(text: str) -> Optional[float]:
    cleaned = re.sub(r"[^0-9.]", "", (text or "").split("(")[0])
    try:
        return float(cleaned)
    except ValueError:
        return None


def _docs_json(html: str) -> str:
    """Bokeh 3.x inlines the document as `const docs_json = '{...}';` in the
    render script — a single-quoted JSON string, not a `<script
    type="application/json">` tag. Return the JSON between the quotes."""
    m = re.search(r"docs_json = '(.*?)';", html, re.S)
    return m.group(1) if m else ""


def _decode(arr) -> List[float]:
    """A Bokeh column: a plain list, or a serialised ndarray carrying its bytes
    base64-encoded. Both reduce to the numbers they hold."""
    import base64
    import struct
    if isinstance(arr, list):
        return [v for v in arr if isinstance(v, (int, float))]
    if isinstance(arr, dict):
        holder = arr.get("array") or arr.get("data")
        raw = None
        if isinstance(holder, dict) and "data" in holder:
            raw = base64.b64decode(holder["data"])
        elif isinstance(holder, str):
            raw = base64.b64decode(holder)
        if raw is None:
            return []
        fmt = {"float64": "d", "float32": "f",
               "int64": "q", "int32": "i"}.get(arr.get("dtype", "float64"), "d")
        size = struct.calcsize(fmt)
        count = len(raw) // size
        return list(struct.unpack("<%d%s" % (count, fmt), raw[:count * size]))
    return []


def _series_finals(html: str) -> List[float]:
    """The last value of every plotted line, from Bokeh's embedded document.

    Bokeh 3.x tags a source `{"name": "ColumnDataSource"}` (type is "object")
    and serialises its `data` as `{"type": "map", "entries": [[key, array], …]}`.
    The portfolio path is the numeric column that is not the date axis (dates
    arrive as millisecond integers, far larger than a dollar figure)."""
    blob = _docs_json(html)
    if not blob:
        return []
    try:
        docs = json.loads(blob)
    except Exception:                                          # noqa: BLE001
        return []
    finals: List[float] = []
    for node in _walk(docs):
        if not (isinstance(node, dict) and node.get("name") == "ColumnDataSource"):
            continue
        data = (node.get("attributes") or {}).get("data") or {}
        entries = data.get("entries") if isinstance(data, dict) else None
        pairs = entries if entries else (
            list(data.items()) if isinstance(data, dict) else [])
        for key, arr in pairs:
            if str(key).lower() in ("x", "date", "dates", "index"):
                continue
            vals = _decode(arr)
            if len(vals) > 2 and abs(vals[-1]) < 1e10:
                finals.append(float(vals[-1]))
    return finals


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


async def inspect_graph(page, key: str, figure: str) -> Graph:
    g = Graph(key=key, figure=figure, status=EVALUATED)
    g.plan_final = _num(figure)
    # The server builds the comparison as a data-root-id div plus a Bokeh render
    # script (3.x inlines the document — there is no application/json tag).
    chart_div = await page.locator("div[data-root-id]").count()
    cdn = await page.locator("script[src*='bokeh']").count()
    g.chart = bool(chart_div) and bool(cdn)
    if not g.chart:
        g.note = "figure but no comparison chart built"
        return g
    # Built is not drawn: BokehJS renders a canvas once it runs, and a
    # version mismatch leaves the div empty with a console error.
    await page.wait_for_timeout(2500)
    g.rendered = await page.locator("div[data-root-id] canvas").count() > 0
    finals = _series_finals(await page.content())
    g.series = len(finals)
    g.finals = sorted(finals, reverse=True)
    if g.plan_final is not None and finals:
        # The plan is the series whose final matches the reported figure.
        benches = list(finals)
        nearest = min(benches, key=lambda v: abs(v - g.plan_final))
        if abs(nearest - g.plan_final) <= max(1.0, g.plan_final * 0.02):
            benches.remove(nearest)
        g.of = len(benches)
        g.beat = sum(1 for v in benches if g.plan_final >= v)
    if not g.rendered:
        g.note = "chart built but did not draw (BokehJS version mismatch?)"
    elif g.series < 2:
        g.note = "chart drew only the plan — no benchmark to compare against"
    return g


async def run(base: str, email: str, password: str, limit: int,
              only_evaluated: bool) -> List[Graph]:
    from playwright.async_api import async_playwright

    from strategies import catalogue                           # noqa: E402

    base = base.rstrip("/")
    entries = catalogue()
    if limit:
        entries = entries[:limit]

    out: List[Graph] = []
    async with async_playwright() as driver:
        browser = await driver.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()
        await sign_in(page, base, email, password)
        for index, (key, sentence) in enumerate(entries, 1):
            result = await evaluate_one(page, base, key, sentence, url_nav=False)
            if result.status != EVALUATED:
                if not only_evaluated:
                    out.append(Graph(key=key, status=result.status,
                                     note=result.reason[:80]))
                print(f"[{index}/{len(entries)}] {key}: {result.status}",
                      file=sys.stderr)
                continue
            g = await inspect_graph(page, key, result.figure)
            out.append(g)
            print(f"[{index}/{len(entries)}] {key}: rendered={g.rendered} "
                  f"series={g.series}", file=sys.stderr)
        await browser.close()
    return out


def render(graphs: List[Graph]) -> str:
    evaluated = [g for g in graphs if g.status == EVALUATED]
    working = [g for g in evaluated if g.rendered and g.series >= 2]
    lines = ["", "=" * 74, f"graph capability — {len(evaluated)} evaluated strategies",
             "=" * 74]
    for g in evaluated:
        if g.rendered and g.series >= 2:
            head = (f"OK  {g.key:<22} drew {g.series} lines · "
                    f"plan beat {g.beat}/{g.of} benchmarks · {g.figure}")
        elif g.chart and g.series >= 2:
            head = (f"~~  {g.key:<22} built {g.series} lines but did not draw"
                    f" · {g.figure}")
        else:
            head = f"!!  {g.key:<22} {g.note or 'no comparison'} · {g.figure}"
        lines.append(head)
    lines.append("-" * 74)
    lines.append(f"{len(working)}/{len(evaluated)} evaluated strategies draw a working "
                 f"comparison graph (plan + >=1 benchmark, rendered)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="https://quantify.club")
    ap.add_argument("--email", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-evaluated", action="store_true",
                    help="skip strategies that do not evaluate")
    args = ap.parse_args()
    graphs = asyncio.run(run(args.url, args.email, args.password,
                             args.limit, args.only_evaluated))
    print(render(graphs))
    evaluated = [g for g in graphs if g.status == EVALUATED]
    missing = [g for g in evaluated if not (g.rendered and g.series >= 2)]
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
