"""Generate interactive Bokeh dashboard for regimes and allocations."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    DataTable,
    Div,
    HoverTool,
    LinearAxis,
    NumberFormatter,
    Range1d,
    Slider,
    Span,
    TableColumn,
    TapTool,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure

from ..config import UNIVERSE
from ..history import GLD_SHARE_TO_OUNCE, HISTORY_DIR, PRICES_PATH, TIMELINE_PATH, WEIGHTS_PATH

REGIME_COLORS = {
    "risk_on": "#2ca02c",
    "risk_off": "#d62728",
    "inflation": "#ff7f0e",
}

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_sources(
    timeline: pd.DataFrame,
    weights: pd.DataFrame,
) -> tuple[ColumnDataSource, ColumnDataSource, pd.DataFrame, pd.DataFrame]:
    timeline = timeline.reset_index().sort_values("date").reset_index(drop=True)
    timeline["date_ms"] = timeline["date"].astype("int64") / 10**6

    ordered = [asset.ticker for asset in UNIVERSE]
    weights_wide = (
        weights.pivot(index="date", columns="ticker", values="weight")
        .reindex(timeline["date"], method="nearest")
        .fillna(0.0)
    )
    weights_wide = weights_wide.reindex(columns=ordered, fill_value=0.0)
    weights_reset = weights_wide.reset_index(drop=False)

    def _pivot(column: str, prefix: str) -> pd.DataFrame:
        if column not in weights.columns:
            return pd.DataFrame(
                0.0,
                index=weights_reset.index,
                columns=[f"{prefix}{ticker}" for ticker in ordered],
            )
        wide = (
            weights.pivot(index="date", columns="ticker", values=column)
            .reindex(timeline["date"], method="nearest")
            .fillna(0.0)
        )
        wide = wide.reindex(columns=ordered, fill_value=0.0)
        wide = wide.reset_index(drop=True)
        return wide.rename(columns={ticker: f"{prefix}{ticker}" for ticker in ordered})

    unrestricted_cols = _pivot("unrestricted_weight", "unres_")
    standard_cols = _pivot("standard_weight", "std_")
    standard_unres_cols = _pivot("standard_unrestricted_weight", "std_unres_")

    alloc_frame = pd.concat([weights_reset, unrestricted_cols, standard_cols, standard_unres_cols], axis=1)

    price_source = ColumnDataSource(timeline)
    alloc_source = ColumnDataSource(alloc_frame)
    return price_source, alloc_source, timeline, weights_reset


def _build_regime_segments(timeline: pd.DataFrame, weights_wide: pd.DataFrame) -> pd.DataFrame:
    ordered = [asset.ticker for asset in UNIVERSE]
    merged = timeline[["date", "regime"]].copy()
    merged = merged.merge(weights_wide, on="date", how="left")
    merged[ordered] = merged[ordered].ffill().fillna(0.0)

    segments = []
    start = 0
    regimes = merged["regime"].tolist()
    dates = merged["date"].tolist()
    for idx in range(1, len(merged)):
        if regimes[idx] != regimes[start]:
            segments.append((start, idx - 1))
            start = idx
    if merged.shape[0]:
        segments.append((start, merged.shape[0] - 1))

    rows = []
    price_min = float(timeline["spy_price"].min())
    price_max = float(timeline["spy_price"].max())
    pad = max((price_max - price_min) * 0.05, 5.0)
    for start_idx, end_idx in segments:
        row = {
            "left": dates[start_idx],
            "right": dates[end_idx],
            "regime": regimes[start_idx],
            "start_idx": start_idx,
            "end_idx": end_idx,
            "color": REGIME_COLORS.get(regimes[start_idx], "#cccccc"),
            "bottom": price_min - pad,
            "top": price_max + pad,
            "exp_return": float(timeline.at[end_idx, "exp_return"]),
            "exp_vol": float(timeline.at[end_idx, "exp_vol"]),
            "sharpe": float(timeline.at[end_idx, "sharpe"]),
            "beta": float(timeline.at[end_idx, "beta_proxy"]),
            "cash_weight": float(merged.at[end_idx, "BIL"] if "BIL" in merged.columns else 0.0),
        }
        for asset in UNIVERSE:
            row[f"start_{asset.ticker}"] = float(merged.at[start_idx, asset.ticker])
            row[f"end_{asset.ticker}"] = float(merged.at[end_idx, asset.ticker])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["left", "right"])
    return pd.DataFrame(rows)



def build_dashboard(
    timeline_path: Path = TIMELINE_PATH,
    weights_path: Path = WEIGHTS_PATH,
    output_path: Path | None = None,
) -> Path:
    if output_path is None:
        output_path = REPORTS_DIR / "regime_dashboard.html"

    timeline = pd.read_parquet(timeline_path)
    if "sharpe_unrestricted" not in timeline.columns:
        timeline["sharpe_unrestricted"] = float("nan")
    if "gold_price_oz" not in timeline.columns:
        try:
            prices = pd.read_parquet(PRICES_PATH)
            gld_series = prices.get("GLD")
            if gld_series is not None:
                timeline = timeline.join((gld_series * GLD_SHARE_TO_OUNCE).rename("gold_price_oz"), how="left")
            else:
                timeline["gold_price_oz"] = float("nan")
        except FileNotFoundError:
            timeline["gold_price_oz"] = float("nan")
        except Exception:  # noqa: BLE001 - best effort to hydrate GLD history
            timeline["gold_price_oz"] = float("nan")
    weights = pd.read_parquet(weights_path)

    price_source, alloc_source, timeline_sorted, weights_wide = _prepare_sources(timeline, weights)
    regime_segments = _build_regime_segments(timeline_sorted, weights_wide)
    regime_source = ColumnDataSource(regime_segments)

    spy_series = timeline_sorted["spy_price"]
    spy_min = float(spy_series.min())
    spy_max = float(spy_series.max())
    spy_pad = max((spy_max - spy_min) * 0.05, 5.0)
    price_range = Range1d(start=spy_min - spy_pad, end=spy_max + spy_pad)

    price_fig = figure(
        title="SPY Price with Regime Bands",
        x_axis_type="datetime",
        sizing_mode="stretch_width",
        height=350,
        tools="xpan,xwheel_zoom,reset,save",
    )
    price_fig.y_range = price_range
    price_line = price_fig.line("date", "spy_price", source=price_source, line_width=2, color="#1f77b4", legend_label="SPY")

    gld_series = timeline_sorted["gold_price_oz"].dropna()
    if gld_series.empty:
        gld_min, gld_max = 0.0, 1.0
    else:
        gld_min = float(gld_series.min())
        gld_max = float(gld_series.max())
    gld_pad = max((gld_max - gld_min) * 0.05, 1.0)
    gld_range = Range1d(start=gld_min - gld_pad, end=gld_max + gld_pad)
    price_fig.extra_y_ranges = {"gld": gld_range}
    price_fig.add_layout(LinearAxis(y_range_name="gld", axis_label="Gold (USD/oz)"), "right")
    gld_line = price_fig.line(
        "date",
        "gold_price_oz",
        source=price_source,
        line_width=2,
        color="#ffbf00",
        line_dash="dashed",
        legend_label="Gold (oz)",
        y_range_name="gld",
    )
    price_fig.yaxis.axis_label = "SPY Price"
    quad_renderer = price_fig.quad(
        left="left",
        right="right",
        bottom="bottom",
        top="top",
        fill_color="color",
        fill_alpha=0.08,
        line_alpha=0.0,
        source=regime_source,
    )
    price_fig.legend.location = "top_left"
    price_fig.add_tools(
        HoverTool(
            tooltips=[
                ("Date", "@date{%F}"),
                ("Regime", "@regime"),
                ("SPY", "@spy_price{0.2f}"),
                ("Gold", "@gold_price_oz{0.2f}"),
                ("Sharpe", "@sharpe{0.2f}"),
            ],
            formatters={"@date": "datetime"},
            mode="vline",
            renderers=[price_line],
        )
    )

    tap_tool = TapTool(renderers=[quad_renderer])
    price_fig.add_tools(tap_tool)

    palette = Category10[max(3, min(10, len(UNIVERSE)))]
    alloc_fig = figure(
        title="Optimal Allocation Weights",
        x_axis_type="datetime",
        sizing_mode="stretch_width",
        height=350,
        tools="xpan,xwheel_zoom,reset,save",
    )
    stackers = [asset.ticker for asset in UNIVERSE]
    alloc_fig.varea_stack(
        stackers=stackers,
        x="date",
        color=palette[: len(stackers)],
        legend_label=stackers,
        source=alloc_source,
    )
    alloc_fig.legend.location = "top_left"
    alloc_fig.yaxis.axis_label = "Weight"
    alloc_fig.y_range.start = 0
    alloc_fig.y_range.end = 1.25

    vix_fig = figure(
        title="VIX (daily)",
        x_axis_type="datetime",
        x_range=price_fig.x_range,
        sizing_mode="stretch_width",
        height=150,
        tools="xpan,xwheel_zoom,reset,save",
    )
    vix_fig.vbar(
        x="date",
        top="vix",
        width=1000 * 60 * 60 * 24 * 0.8,
        color="#9467bd",
        source=price_source,
    )
    vix_fig.yaxis.axis_label = "VIX"
    vix_fig.add_tools(
        HoverTool(
            tooltips=[("Date", "@date{%F}"), ("VIX", "@vix{0.2f}")],
            formatters={"@date": "datetime"},
            mode="vline",
        )
    )

    span_price = Span(location=price_source.data["date_ms"][0], dimension="height", line_color="black", line_width=2)
    span_alloc = Span(location=price_source.data["date_ms"][0], dimension="height", line_color="black", line_width=2)
    span_vix = Span(location=price_source.data["date_ms"][0], dimension="height", line_color="black", line_width=2)
    price_fig.add_layout(span_price)
    alloc_fig.add_layout(span_alloc)
    vix_fig.add_layout(span_vix)
    if len(regime_source.data.get("left", [])):
        regime_data = dict(regime_source.data)
        regime_data["bottom"] = [price_range.start] * len(regime_data["left"])
        regime_data["top"] = [price_range.end] * len(regime_data["left"])
        regime_source.data = regime_data

    initial_regime = price_source.data["regime"][0]
    initial_sharpe = price_source.data.get("sharpe", [float("nan")])[0]
    initial_unres = price_source.data.get("sharpe_unrestricted", [float("nan")])[0]

    def _format_sharpe(value: float) -> str:
        return "—" if pd.isna(value) else f"{value:.2f}"

    def _format_pct(value: float) -> str:
        return "—" if pd.isna(value) else f"{value * 100:.1f}%"

    status = Div(
        text="Regime: {} | Sharpe {} | Unrestricted Sharpe {}".format(
            initial_regime,
            _format_sharpe(initial_sharpe),
            _format_sharpe(initial_unres),
        )
    )
    regime_info = Div(text="Click a regime band to inspect allocation shifts.")
    regime_download = Div(text="")

    ticker_list = [asset.ticker for asset in UNIVERSE]
    name_list = [asset.label for asset in UNIVERSE]
    initial_regime = [alloc_source.data[t][0] for t in ticker_list]
    initial_regime_unres = [alloc_source.data.get(f"unres_{t}", [0.0])[0] for t in ticker_list]
    initial_standard = [alloc_source.data.get(f"std_{t}", [0.0])[0] for t in ticker_list]
    initial_standard_unres = [alloc_source.data.get(f"std_unres_{t}", [0.0])[0] for t in ticker_list]
    table_source = ColumnDataSource(
        data={
            "ticker": ticker_list,
            "name": name_list,
            "regime_restricted": initial_regime,
            "regime_unrestricted": initial_regime_unres,
            "standard_restricted": initial_standard,
            "standard_unrestricted": initial_standard_unres,
        }
    )
    table = DataTable(
        source=table_source,
        columns=[
            TableColumn(field="ticker", title="Ticker"),
            TableColumn(field="name", title="Name"),
            TableColumn(field="regime_restricted", title="Regime (restricted)", formatter=NumberFormatter(format="0.0%")),
            TableColumn(field="regime_unrestricted", title="Regime (unrestricted)", formatter=NumberFormatter(format="0.0%")),
            TableColumn(field="standard_restricted", title="Standard (restricted)", formatter=NumberFormatter(format="0.0%")),
            TableColumn(field="standard_unrestricted", title="Standard (unrestricted)", formatter=NumberFormatter(format="0.0%")),
        ],
        height=200,
        width=600,
        index_position=None,
    )

    constraints_text = Div(
        text=(
            "<b>Weight methodologies</b><br>"
            "<b>Standard (restricted):</b> Single risk-on guardrail set (cash floors, inverse caps, turnover) regardless of regime.<br>"
            "<b>Standard (unrestricted):</b> Same baseline data but long-only 0–100% bounds and budget constraint only.<br>"
            "<b>Regime (restricted):</b> Guardrails adapt to detected regime before solving the constrained Sharpe optimizer.<br>"
            "<b>Regime (unrestricted):</b> Regime-aware inputs with long-only bounds only (no guardrails)."
        ),
        width=350,
    )

    perf_std_res = timeline_sorted.get("total_return_standard_restricted")
    perf_std_unres = timeline_sorted.get("total_return_standard_unrestricted")
    perf_reg_res = timeline_sorted.get("total_return_regime_restricted")
    perf_reg_unres = timeline_sorted.get("total_return_regime_unrestricted")

    def _latest(series: pd.Series | None) -> float:
        if series is None or series.empty:
            return float("nan")
        return float(series.iloc[-1])

    perf_since = timeline_sorted["date"].iloc[0]
    perf_label = pd.Timestamp(perf_since).strftime("%Y-%m-%d") if pd.notna(perf_since) else "N/A"
    performance_text = Div(
        text=(
            f"<b>Simulated performance (since {perf_label})</b><br>"
            f"Standard (restricted): {_format_pct(_latest(perf_std_res))}<br>"
            f"Standard (unrestricted): {_format_pct(_latest(perf_std_unres))}<br>"
            f"Regime (restricted): {_format_pct(_latest(perf_reg_res))}<br>"
            f"Regime (unrestricted): {_format_pct(_latest(perf_reg_unres))}"
        ),
        width=350,
    )

    info_column = column(constraints_text, performance_text, sizing_mode="stretch_width")
    table_row = row(table, info_column, sizing_mode="stretch_width")

    slider = Slider(start=0, end=len(price_source.data["date"]) - 1, value=len(price_source.data["date"]) - 1, step=1, title="Timeline index", visible=False)

    callback = CustomJS(
        args=dict(
            slider=slider,
            price_source=price_source,
            span_price=span_price,
            span_alloc=span_alloc,
            span_vix=span_vix,
            status=status,
            table_source=table_source,
            alloc_source=alloc_source,
            tickers=ticker_list,
            names=name_list,
        ),
        code="""
        const idx = slider.value;
        const date_ms = price_source.data['date_ms'][idx];
        span_price.location = date_ms;
        span_alloc.location = date_ms;
        span_vix.location = date_ms;
        const regime = price_source.data['regime'][idx];
        const sharpeSeries = price_source.data['sharpe'] || [];
        const sharpe = sharpeSeries[idx];
        const unresSeries = price_source.data['sharpe_unrestricted'] || [];
        const unresSharpe = unresSeries[idx];
        const sharpeText = Number.isFinite(sharpe) ? sharpe.toFixed(2) : '—';
        const unresText = Number.isFinite(unresSharpe) ? unresSharpe.toFixed(2) : '—';
        status.text = `Regime: ${regime} | Sharpe ${sharpeText} | Unrestricted Sharpe ${unresText}`;
        span_price.change.emit();
        span_alloc.change.emit();
        const tableData = {
            ticker: [],
            name: [],
            regime_restricted: [],
            regime_unrestricted: [],
            standard_restricted: [],
            standard_unrestricted: []
        };
        for (let i = 0; i < tickers.length; i++) {
            const ticker = tickers[i];
            tableData.ticker.push(ticker);
            tableData.name.push(names[i]);
            const wSeries = alloc_source.data[ticker] || [];
            tableData.regime_restricted.push(wSeries[idx] ?? 0);
            const unresSeries = alloc_source.data[`unres_${ticker}`] || [];
            tableData.regime_unrestricted.push(unresSeries[idx] ?? 0);
            const stdSeries = alloc_source.data[`std_${ticker}`] || [];
            tableData.standard_restricted.push(stdSeries[idx] ?? 0);
            const stdUnresSeries = alloc_source.data[`std_unres_${ticker}`] || [];
            tableData.standard_unrestricted.push(stdUnresSeries[idx] ?? 0);
        }
        table_source.data = tableData;
        table_source.change.emit();
    """,
    )
    slider.js_on_change("value", callback)

    regime_source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(
                regime_source=regime_source,
                slider=slider,
                info=regime_info,
                download=regime_download,
                tickers=ticker_list,
            ),
            code="""
            const idx = regime_source.selected.indices[0];
            if (idx === undefined) { return; }
            const startDate = new Date(regime_source.data['left'][idx]).toISOString().slice(0, 10);
            const endDate = new Date(regime_source.data['right'][idx]).toISOString().slice(0, 10);
            const regime = regime_source.data['regime'][idx];
            slider.value = regime_source.data['start_idx'][idx];
            const lines = [];
            for (let i = 0; i < tickers.length; i++) {
                const ticker = tickers[i];
                const startVal = regime_source.data[`start_${ticker}`][idx] ?? 0;
                const endVal = regime_source.data[`end_${ticker}`][idx] ?? 0;
                if (Math.abs(startVal - endVal) < 0.001) { continue; }
                lines.push(`${ticker}: ${(startVal * 100).toFixed(1)}% → ${(endVal * 100).toFixed(1)}%`);
            }
            const body = lines.length ? lines.join('<br>') : 'Allocations stable';
            const ret = regime_source.data['exp_return'][idx];
            const vol = regime_source.data['exp_vol'][idx];
            const sharpe = regime_source.data['sharpe'][idx];
            const beta = regime_source.data['beta'][idx];
            const cash = regime_source.data['cash_weight'][idx] ?? 0;
            info.text = `<b>${regime}</b><br>Start: ${startDate}<br>End: ${endDate}<br>` +
                `Return ${ (ret*100).toFixed(2)}% | Vol ${(vol*100).toFixed(2)}% | Sharpe ${sharpe.toFixed(2)} | Beta ${beta.toFixed(2)} | Cash ${(cash*100).toFixed(1)}%<br>${body}`;

            const rows = [];
            for (let i = 0; i < tickers.length; i++) {
                const ticker = tickers[i];
                const startVal = regime_source.data[`start_${ticker}`][idx] ?? 0;
                const endVal = regime_source.data[`end_${ticker}`][idx] ?? 0;
                rows.push({
                    ticker,
                    start: startVal,
                    end: endVal,
                    delta: endVal - startVal,
                });
            }
            const csvLines = ['ticker,start,end,delta'];
            rows.forEach(r => {
                csvLines.push(`${r.ticker},${r.start},${r.end},${r.delta}`);
            });
            const csv = csvLines.join('\n');
            const blob = new Blob([csv], {type: 'text/csv'});
            const url = URL.createObjectURL(blob);
            download.text = `<a href="${url}" download="regime_${regime}_${startDate}_to_${endDate}.csv">Download allocation changes</a>`;
        """,
        ),
    )

    price_fig.js_on_event(
        "tap",
        CustomJS(
            args=dict(slider=slider, price_source=price_source),
            code="""
            const x = cb_obj.x;
            if (x === undefined || x === null) { return; }
            const dates = price_source.data['date'];
            let best = 0;
            let minDiff = Infinity;
            for (let i = 0; i < dates.length; i++) {
                const diff = Math.abs(dates[i] - x);
                if (diff < minDiff) {
                    minDiff = diff;
                    best = i;
                }
            }
            slider.value = best;
        """,
        ),
    )

    layout = column(
        price_fig,
        vix_fig,
        alloc_fig,
        status,
        table_row,
        regime_info,
        regime_download,
        sizing_mode="stretch_width",
    )

    # Keep slider out of layout but ensure callbacks wire up by triggering initial update
    slider.value = len(price_source.data["date"]) - 1

    output_file(output_path, title="Regime Dashboard")
    save(layout)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bokeh dashboard from historical analysis")
    parser.add_argument("--timeline", type=Path, default=TIMELINE_PATH, help="Path to timeline parquet")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH, help="Path to weights parquet")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "regime_dashboard.html", help="Output HTML path")
    args = parser.parse_args()

    path = build_dashboard(args.timeline, args.weights, args.output)
    print(f"Dashboard saved to {path}")


if __name__ == "__main__":
    main()
