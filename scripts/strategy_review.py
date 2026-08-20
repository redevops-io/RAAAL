"""A reviewable report of every computed strategy, run with its defaults.

For each strategy the pilot offers, this compiles the catalogue entry exactly as
picking it from the dropdown would (its own STATES defaults — $500/month,
rebalanced quarterly, over the ten-fund research universe), evaluates it over the
synthetic snapshot, and writes:

  * `strategy-review/index.html` — one row per strategy: what it does, its final
    value, gain, how many benchmarks it beat, and a link to its graph;
  * `strategy-review/graphs/<key>.html` — the plan's own path against the five
    benchmarks, the same comparison chart the result page draws.

Self-contained: the run is assembled from the fixture as `run_boundary` does, so
no market-data deployment config is needed.

    docker run -v "$PWD":/app -w /app quantify-test:local \\
        python3 scripts/strategy_review.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.evaluation.core import evaluate_plan                     # noqa: E402
from src.mission.accounting import CashPolicy                     # noqa: E402
from src.mission.benchmark import compare                         # noqa: E402
from src.mission.from_intent import compile_intent                # noqa: E402
from src.mission.performance import from_path                      # noqa: E402
from src.mission.strategy_methods import strategy_capability      # noqa: E402
from src.strategies import CAPABILITY_BY_ID                       # noqa: E402
from src.workspace.catalog_intent import intent_for              # noqa: E402
from src.workspace.comparison_chart import build, collect         # noqa: E402
from src.workspace.run_boundary import _benchmark_specs           # noqa: E402
from src.workspace.strategy_library import LIBRARY                # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "prices_synthetic.parquet"
OUT = REPO / "strategy-review"
GRAPHS = OUT / "graphs"


def _run(entry_key: str, prices: pd.DataFrame):
    """The `run` the result page would draw for this picked entry."""
    intent, _ = intent_for(entry_key)
    scenario = compile_intent(intent).scenario
    evaluated = evaluate_plan(scenario, prices)
    specs = _benchmark_specs(prices, list(evaluated.tradeable))
    benchmarks = compare(prices, flows=list(evaluated.flows),
                         cash_policy=CashPolicy.idle(), benchmarks=specs)
    return scenario, evaluated, {"result": evaluated.result,
                                 "benchmarks": benchmarks}


def _describe(capability_id: str) -> dict:
    spec = CAPABILITY_BY_ID.get(capability_id)
    return {
        "what": str(getattr(spec, "expected_behavior", "") or "").strip(),
        "family": str(getattr(spec, "family", "") or "").replace("_", " "),
        "risk": str(getattr(spec, "risk_profile", "") or ""),
        "min_history": int(getattr(spec, "min_history", 0) or 0),
    }


def _graph_page(title: str, subtitle: str, meta: str, chart: dict) -> str:
    v = chart["version"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — comparison</title>
<script src="https://cdn.bokeh.org/bokeh/release/bokeh-{v}.min.js"></script>
<style>
  body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; color:#1a2028;
         max-width: 980px; margin: 2rem auto; padding: 0 1rem; }}
  a {{ color:#2563eb; }} h1 {{ margin:0 0 .2rem; font-size:1.5rem; }}
  .sub {{ color:#5b6673; margin:0 0 .3rem; }}
  .meta {{ color:#5b6673; font-size:.9rem; margin:0 0 1.2rem; }}
</style></head><body>
<p><a href="../index.html">&larr; all strategies</a></p>
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(subtitle)}</p>
<p class="meta">{meta}</p>
{chart['div']}
{chart['script']}
</body></html>
"""


def main() -> int:
    prices = pd.read_parquet(FIXTURE)
    GRAPHS.mkdir(parents=True, exist_ok=True)

    group = next(g for g in LIBRARY if g.key == "computed-strategies")
    rows = []
    for entry in group.entries:
        capability = strategy_capability(
            # the method the entry seals to
            intent_for(entry.key)[0].fields["allocation_method"].value)
        scenario, evaluated, run = _run(entry.key, prices)
        series = collect(run)
        plan_final = series[0]["values"][-1]
        contributed = float(evaluated.result.path.flows.sum())
        gain = plan_final - contributed
        gain_pct = 100.0 * gain / contributed if contributed else 0.0
        benches = [(s["name"], s["values"][-1]) for s in series[1:]]
        beat = sum(1 for _n, v in benches if plan_final >= v)
        perf = from_path(evaluated.result.path)

        desc = _describe(capability)
        cadence = getattr(scenario.holdings_policy, "rebalancing_cadence", "")
        meta = (f"capability <code>{capability}</code> &middot; "
                f"family {desc['family']} &middot; {desc['risk']} risk &middot; "
                f"rebalanced {cadence} &middot; warm-up {desc['min_history']} "
                f"sessions &middot; $500/month over the ten-fund universe"
                f"<br>Sharpe {perf.sharpe:+.2f} &middot; "
                f"volatility {perf.annual_volatility * 100:.1f}%/yr &middot; "
                f"max drawdown {perf.max_drawdown * 100:.1f}% &middot; "
                f"return {perf.annual_return * 100:+.1f}%/yr "
                f"(time-weighted, risk-free 2%)")

        chart = build(run)
        graph_name = f"{entry.key}.html"
        subtitle = desc["what"] or entry.title
        (GRAPHS / graph_name).write_text(
            _graph_page(entry.title, subtitle, meta, chart))

        rows.append({
            "title": entry.title, "capability": capability,
            "what": desc["what"], "family": desc["family"], "risk": desc["risk"],
            "final": plan_final, "gain": gain, "gain_pct": gain_pct,
            "beat": beat, "of": len(benches), "graph": f"graphs/{graph_name}",
            "sharpe": perf.sharpe, "vol": perf.annual_volatility,
            "maxdd": perf.max_drawdown, "cagr": perf.annual_return,
        })
        print(f"  {entry.title:26} ${plan_final:>10,.0f}  "
              f"beat {beat}/{len(benches)}  -> {graph_name}")

    rows.sort(key=lambda r: r["final"], reverse=True)
    OUT.joinpath("index.html").write_text(_index_page(rows))
    print(f"\n{len(rows)} strategies -> {OUT/'index.html'}")
    return 0


def _index_page(rows) -> str:
    def cell(r):
        risk_col = {"aggressive": "#b91c1c", "moderate": "#b45309",
                    "defensive": "#047857"}.get(r["risk"], "#5b6673")
        dd_col = "#b91c1c" if r["maxdd"] <= -0.30 else "#5b6673"
        sh_col = "#047857" if r["sharpe"] >= 0.5 else (
            "#b91c1c" if r["sharpe"] < 0 else "#5b6673")
        return f"""<tr>
      <td><a href="{r['graph']}"><strong>{html.escape(r['title'])}</strong></a>
          <div class="cap">{html.escape(r['capability'])}</div></td>
      <td>{html.escape(r['what'])}</td>
      <td class="tag">{html.escape(r['family'])}<br>
          <span style="color:{risk_col}">{html.escape(r['risk'])}</span></td>
      <td class="num">${r['final']:,.0f}</td>
      <td class="num">{r['gain_pct']:+.1f}%</td>
      <td class="num" style="color:{sh_col}">{r['sharpe']:+.2f}</td>
      <td class="num">{r['vol'] * 100:.1f}%</td>
      <td class="num" style="color:{dd_col}">{r['maxdd'] * 100:.1f}%</td>
      <td class="num">{r['beat']}/{r['of']}</td>
      <td><a href="{r['graph']}">View graph &rarr;</a></td>
    </tr>"""

    body = "\n".join(cell(r) for r in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy review</title>
<style>
  body {{ font: 15px/1.55 -apple-system, system-ui, sans-serif; color:#1a2028;
         max-width: 1120px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 .3rem; }}
  p.lead {{ color:#5b6673; margin: 0 0 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: .55rem .7rem;
            border-bottom: 1px solid #e6e9ee; vertical-align: top; }}
  th {{ color:#5b6673; font-weight:600; font-size:12px; text-transform:uppercase;
        letter-spacing:.04em; border-bottom:2px solid #d5dae1; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums;
          white-space: nowrap; }}
  .cap {{ color:#8a94a1; font-size:12px; font-family: ui-monospace, monospace; }}
  .tag {{ color:#5b6673; font-size:12px; }}
  a {{ color:#2563eb; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
  tr:hover td {{ background:#f7f9fb; }}
</style></head><body>
<h1>Computed strategies — review</h1>
<p class="lead">Each strategy run with its dropdown defaults — $500 a month,
rebalanced quarterly, over the ten-fund research universe, on the synthetic
2016&ndash;2025 snapshot. Final value, gain, and how many of the five benchmarks
the plan beat. Click a strategy to see its comparison graph. Sorted by final
value.</p>
<table>
  <thead><tr>
    <th>Strategy</th><th>What it does</th><th>Family / risk</th>
    <th class="num">Final value</th><th class="num">Gain</th>
    <th class="num">Beat</th><th>Graph</th>
  </tr></thead>
  <tbody>
{body}
  </tbody>
</table>
<p class="lead" style="margin-top:1.5rem">Contributions total $59,500 across the
period; a plan that beats "Hold cash" grew the money, and the benchmark set is
the same contributions bought and held in the basket, the S&amp;P 500, the
Nasdaq 100, aggregate bonds, and cash.</p>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
