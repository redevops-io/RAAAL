"""Generate interactive Bokeh dashboard for regimes and allocations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
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
    MultiSelect,
    NumberFormatter,
    Range1d,
    Slider,
    Span,
    TableColumn,
    TabPanel,
    Tabs,
    TapTool,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure

from ..config import UNIVERSE
from ..features import compute_returns
from ..history import (
    GLD_SHARE_TO_OUNCE,
    HISTORY_DIR,
    PRICES_PATH,
    TIMELINE_PATH,
    WEIGHTS_PATH,
    strategy_cumulative_returns,
)

REGIME_COLORS = {
    "risk_on": "#2ca02c",
    "risk_off": "#d62728",
    "inflation": "#ff7f0e",
}

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODE_LABELS = {
    "rule_based": "Rule-Based",
    "ml": "ML Ensemble",
    "none": "No Regime",
}


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



def build_main_dashboard_panel(
    timeline: pd.DataFrame,
    weights: pd.DataFrame,
) -> TabPanel:
    """Build the main dashboard panel (original view)."""
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

    return TabPanel(title="The Strategy", child=layout)


def _discover_strategy_columns(weights: pd.DataFrame) -> Dict[str, str]:
    columns = [col for col in weights.columns if col.startswith("strategy_") and col.endswith("_weight")]
    mapping: Dict[str, str] = {}
    for col in columns:
        label = col[len("strategy_") : -len("_weight")]
        if "_" not in label:
            continue
        mapping[label] = col
    return mapping


def _format_strategy_label(label: str) -> tuple[str, str, str]:
    mode, strategy = label.split("_", 1)
    mode_display = MODE_LABELS.get(mode, mode.title())
    strategy_display = strategy.replace("_", " ").title()
    legend_label = f"{strategy_display} ({mode_display})"
    return mode_display, strategy_display, legend_label


def build_strategy_comparison_panel(
    timeline: pd.DataFrame,
    weights: pd.DataFrame,
    prices: pd.DataFrame | None,
) -> TabPanel:
    if prices is None or prices.empty:
        return TabPanel(title="Strategy Lab", child=Div(text="Price history unavailable. Run `src.history` first."))

    strategy_columns = _discover_strategy_columns(weights)
    if not strategy_columns:
        return TabPanel(title="Strategy Lab", child=Div(text="No strategy weights found. Regenerate history after enabling StrategySuite."))

    tickers = [asset.ticker for asset in UNIVERSE]
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing:
        msg = f"Missing price series for: {', '.join(missing)}"
        return TabPanel(title="Strategy Lab", child=Div(text=msg))

    asset_returns = compute_returns(prices[tickers])
    if asset_returns.empty:
        return TabPanel(title="Strategy Lab", child=Div(text="Not enough price history to compute returns."))

    cum_df = pd.DataFrame(index=asset_returns.index)
    summary_rows = []
    for label, weight_col in sorted(strategy_columns.items()):
        series = strategy_cumulative_returns(weights, asset_returns, weight_col)
        if series.empty:
            continue
        cum_df[label] = series
        mode_display, strategy_display, legend_label = _format_strategy_label(label)
        daily_returns = series.pct_change().fillna(0.0)
        days = max(len(series), 1)
        total_return = float(series.iloc[-1] - 1.0)
        annual_return = float(series.iloc[-1] ** (252.0 / days) - 1.0)
        annual_vol = float(daily_returns.std() * np.sqrt(252))
        sharpe = float(annual_return / annual_vol) if annual_vol else float("nan")
        summary_rows.append(
            {
                "strategy_key": label,
                "mode_display": mode_display,
                "strategy_display": strategy_display,
                "legend": legend_label,
                "total_return": total_return,
                "annual_return": annual_return,
                "sharpe": sharpe,
            }
        )

    if cum_df.empty or not summary_rows:
        return TabPanel(title="Strategy Lab", child=Div(text="Strategy curves unavailable."))

    perf_source = ColumnDataSource(cum_df.reset_index().rename(columns={cum_df.index.name or "index": "date"}))
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return TabPanel(title="Strategy Lab", child=Div(text="No strategy statistics were computed."))
    summary_df = summary_df.sort_values("sharpe", ascending=False, na_position="last").reset_index(drop=True)
    summary_source = ColumnDataSource(summary_df)
    key_to_label = {row["strategy_key"]: row["legend"] for _, row in summary_df.iterrows()}

    def _top_keys(column: str, limit: int = 5) -> list[str]:
        sorted_df = summary_df.sort_values(column, ascending=False, na_position="last")
        return sorted_df["strategy_key"].head(limit).tolist()

    top_sharpe_keys = _top_keys("sharpe")
    top_return_keys = _top_keys("total_return")
    focus_keys = list(dict.fromkeys(top_sharpe_keys + top_return_keys))
    if not focus_keys:
        focus_keys = summary_df["strategy_key"].head(5).tolist()

    default_visible_keys = focus_keys if focus_keys else summary_df["strategy_key"].head(10).tolist()

    palette = Category10[max(3, min(10, len(summary_rows)))]
    perf_fig = figure(
        title="Strategy Growth (normalized to 1)",
        x_axis_type="datetime",
        sizing_mode="stretch_width",
        height=350,
        tools="xpan,xwheel_zoom,reset,save",
    )
    default_visible_set = set(default_visible_keys)
    line_renderers = []
    for idx, entry in enumerate(summary_rows):
        color = palette[idx % len(palette)]
        renderer = perf_fig.line(
            "date",
            entry["strategy_key"],
            source=perf_source,
            line_width=2,
            color=color,
            legend_label=entry["legend"],
            name=entry["strategy_key"],
        )
        renderer.visible = not default_visible_set or entry["strategy_key"] in default_visible_set
        line_renderers.append(renderer)
    perf_fig.yaxis.axis_label = "Growth"
    perf_fig.legend.location = "top_left"
    perf_fig.legend.click_policy = "hide"
    hover_tool = HoverTool(
        tooltips=[("Strategy", "$name"), ("Date", "@date{%F}"), ("Growth", "$y{0.000}")],
        formatters={"@date": "datetime"},
        mode="mouse",
        renderers=line_renderers,
    )
    perf_fig.add_tools(hover_tool)

    summary_table = DataTable(
        source=summary_source,
        columns=[
            TableColumn(field="strategy_display", title="Strategy"),
            TableColumn(field="mode_display", title="Regime Signal"),
            TableColumn(field="total_return", title="Total Return", formatter=NumberFormatter(format="0.0%")),
            TableColumn(field="annual_return", title="CAGR", formatter=NumberFormatter(format="0.0%")),
            TableColumn(field="sharpe", title="Sharpe", formatter=NumberFormatter(format="0.00")),
        ],
        height=280,
        index_position=None,
        reorderable=False,
    )

    strategy_options = [(row["strategy_key"], row["legend"]) for _, row in summary_df.iterrows()]
    strategy_select = MultiSelect(
        title="Visible strategies (Ctrl/Cmd + click)",
        value=default_visible_keys,
        options=strategy_options,
        size=min(max(len(strategy_options), 4), 12),
        width=260,
    )
    selection_hint = Div(text="Use Ctrl/Cmd + click to highlight specific strategies across charts.")
    selector_column = column(selection_hint, strategy_select, width=280)

    # Top strategy diagnostics (weights/timing)
    def _weight_history(column_name: str) -> pd.DataFrame:
        history = (
            weights.pivot(index="date", columns="ticker", values=column_name)
            .sort_index()
            .reindex(asset_returns.index)
            .ffill()
            .fillna(0.0)
        )
        return history

    def _format_holdings(snapshot: pd.Series | None) -> str:
        if snapshot is None or snapshot.empty:
            return "—"
        ordered = snapshot.sort_values(ascending=False).head(3)
        parts = []
        for ticker, weight in ordered.items():
            if weight <= 0:
                continue
            parts.append(f"{ticker} ({weight:.0%})")
        return ", ".join(parts) if parts else "—"

    def _snapshot(history_frame: pd.DataFrame, target_date: pd.Timestamp | None) -> pd.Series | None:
        if target_date is None or pd.isna(target_date):
            return None
        try:
            row = history_frame.loc[target_date]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):  # duplicate dates
            row = row.iloc[-1]
        return row

    def _to_timestamp(value: pd.Timestamp | np.datetime64 | None) -> pd.Timestamp:
        if value is None or pd.isna(value):
            return pd.NaT
        if isinstance(value, pd.Timestamp):
            return value
        try:
            return pd.Timestamp(value)
        except Exception:
            return pd.NaT

    focus_order = {key: idx for idx, key in enumerate(focus_keys)}
    top_strategy_keys = set(focus_keys)
    cash_ticker = "BIL" if "BIL" in tickers else tickers[-1]
    details_rows = []
    signal_rows = []
    history_cache: dict[str, pd.DataFrame] = {}
    latest_snapshot_cache: dict[str, pd.Series] = {}
    last_rebalance_cache: dict[str, pd.Timestamp] = {}
    threshold = 0.01
    net_threshold = 0.002
    for _, record in summary_df.iterrows():
        label = record["strategy_key"]
        weight_col = strategy_columns.get(label)
        if not weight_col:
            continue
        history = _weight_history(weight_col)
        if history.empty:
            continue
        history_cache[label] = history
        if cash_ticker in history.columns:
            cash_series = history[cash_ticker]
        else:
            cash_series = pd.Series(0.0, index=history.index)
        avg_cash = float(cash_series.mean())
        deltas = history.diff().fillna(0.0)
        turnover = deltas.abs().sum(axis=1)
        avg_turnover = float(turnover.mean())
        rebalance_events = turnover[turnover > threshold]
        rebalance_index = rebalance_events.index.sort_values()
        if len(rebalance_index):
            last_rebalance = _to_timestamp(rebalance_index[-1])
            prev_rebalance = _to_timestamp(rebalance_index[-2]) if len(rebalance_index) >= 2 else pd.NaT
        else:
            last_rebalance = _to_timestamp(history.index.max())
            prev_rebalance = pd.NaT
        last_rebalance_cache[label] = last_rebalance
        if label in top_strategy_keys:
            latest_snapshot = _snapshot(history, last_rebalance)
            prev_snapshot = _snapshot(history, prev_rebalance)
            latest_holdings = _format_holdings(latest_snapshot)
            prev_holdings = _format_holdings(prev_snapshot)
            details_rows.append(
                {
                    "strategy_key": label,
                    "strategy": record["strategy_display"],
                    "regime": record["mode_display"],
                    "last_rebalance": last_rebalance.strftime("%Y-%m-%d") if pd.notna(last_rebalance) else "—",
                    "prev_rebalance": prev_rebalance.strftime("%Y-%m-%d") if pd.notna(prev_rebalance) else "—",
                    "avg_turnover": avg_turnover,
                    "avg_cash": avg_cash,
                    "latest_holdings": latest_holdings,
                    "prev_holdings": prev_holdings,
                }
            )
        latest_snapshot_cache[label] = _snapshot(history, last_rebalance)

        gross_buy = deltas.clip(lower=0).sum(axis=1)
        gross_sell = (-deltas.clip(upper=0)).sum(axis=1)
        cash_diff = cash_series.diff().fillna(0.0)

        for date in rebalance_events.index:
            net_value = float(-(cash_diff.loc[date]) if date in cash_diff.index else 0.0)
            if abs(net_value) <= net_threshold:
                if date in gross_buy.index and date in gross_sell.index:
                    net_value = float(gross_buy.loc[date] - gross_sell.loc[date])
                else:
                    net_value = 0.0
            magnitude = abs(net_value)
            if magnitude <= net_threshold:
                continue
            direction = "Buy" if net_value >= 0 else "Sell"
            signal_rows.append(
                {
                    "date": date,
                    "strategy": record["strategy_display"],
                    "strategy_label": record["legend"],
                    "strategy_key": label,
                    "signal": direction,
                    "magnitude": magnitude,
                    "gross_buy": float(gross_buy.loc[date]) if date in gross_buy.index else 0.0,
                    "gross_sell": float(gross_sell.loc[date]) if date in gross_sell.index else 0.0,
                    "turnover": float(turnover.loc[date]) if date in turnover.index else 0.0,
                    "cash": float(cash_series.loc[date]) if date in cash_series.index else 0.0,
                }
            )

    if details_rows:
        details_frame = pd.DataFrame(details_rows)
    else:
        details_frame = pd.DataFrame(
            columns=[
                "strategy_key",
                "strategy",
                "regime",
                "last_rebalance",
                "prev_rebalance",
                "avg_turnover",
                "avg_cash",
                "latest_holdings",
                "prev_holdings",
            ]
        )
    if focus_order and not details_frame.empty:
        details_frame["_order"] = details_frame["strategy_key"].map(focus_order).fillna(len(focus_order))
        details_frame = details_frame.sort_values("_order")
        details_frame = details_frame.drop(columns=["strategy_key", "_order"], errors="ignore")
    else:
        details_frame = details_frame.drop(columns=["strategy_key"], errors="ignore")
    details_source = ColumnDataSource(details_frame)
    detail_columns = [
        TableColumn(field="strategy", title="Strategy", width=230),
        TableColumn(field="regime", title="Regime Signal", width=150),
        TableColumn(field="last_rebalance", title="Last Rebalance", width=140),
        TableColumn(field="prev_rebalance", title="Previous Rebalance", width=155),
        TableColumn(field="avg_turnover", title="Avg Turnover", formatter=NumberFormatter(format="0.0%"), width=120),
        TableColumn(field="avg_cash", title="Avg Cash", formatter=NumberFormatter(format="0.0%"), width=110),
        TableColumn(field="latest_holdings", title="Latest Top 3", width=320),
        TableColumn(field="prev_holdings", title="Previous Top 3", width=320),
    ]
    details_table = DataTable(
        source=details_source,
        columns=detail_columns,
        height=280,
        width=1200,
        autosize_mode="none",
        scroll_to_selection=True,
        index_position=None,
    )

    if signal_rows:
        signals_df = pd.DataFrame(signal_rows)
    else:
        signals_df = pd.DataFrame(
            columns=[
                "date",
                "strategy",
                "strategy_label",
                "strategy_key",
                "signal",
                "magnitude",
                "gross_buy",
                "gross_sell",
                "turnover",
                "cash",
            ]
        )

    def _filter_signals(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        if df.empty or not keys:
            return df.copy()
        return df[df["strategy_key"].isin(keys)].copy()

    if not signals_df.empty:
        min_height = 0.25
        max_height = 0.9
        scale = 12.0
        bar_width = 1000 * 60 * 60 * 24 * 2
        enriched_rows = []
        for _, signal in signals_df.iterrows():
            date = signal["date"]
            label = signal["strategy_key"]
            history = history_cache.get(label)
            snapshot = history.loc[date] if history is not None and date in history.index else None
            holdings = _format_holdings(snapshot)
            magnitude = signal["magnitude"]
            height = float(min(max_height, max(min_height, min_height + magnitude * scale)))
            enriched_rows.append(
                {
                    **signal,
                    "holdings": holdings,
                    "color": "#2ca02c" if signal["signal"] == "Buy" else "#d62728",
                    "height": height,
                    "width": bar_width,
                }
            )
        signals_df = pd.DataFrame(enriched_rows)
    else:
        signals_df["holdings"] = []
        signals_df["color"] = []
        signals_df["height"] = []
        signals_df["width"] = []

    full_signal_source = ColumnDataSource(signals_df)
    initial_filtered = _filter_signals(signals_df, default_visible_keys)
    filtered_signal_source = ColumnDataSource(initial_filtered)

    composite_defs: list[tuple[str, list[str]]] = []
    if top_sharpe_keys:
        composite_defs.append(("Top Sharpe 5", top_sharpe_keys))
    if top_return_keys:
        composite_defs.append(("Top Total Return 5", top_return_keys))

    def _composite_holdings(keys: list[str]) -> pd.Series:
        snapshots: list[pd.Series] = []
        for key in keys:
            snapshot = latest_snapshot_cache.get(key)
            if snapshot is None:
                history = history_cache.get(key)
                if history is None or history.empty:
                    continue
                snapshot = history.iloc[-1]
            snapshots.append(snapshot.fillna(0.0))
        if not snapshots:
            return pd.Series(dtype=float)
        combined = sum(snapshots) / len(snapshots)
        total = combined.sum()
        if total == 0 or pd.isna(total):
            return combined
        return combined / total

    composite_signal_rows: list[dict[str, object]] = []
    composite_holdings_rows: list[dict[str, str]] = []
    comp_min_height = 0.25
    comp_max_height = 0.9
    comp_scale = 18.0
    comp_bar_width = 1000 * 60 * 60 * 24 * 3
    for comp_name, keys in composite_defs:
        holdings = _composite_holdings(keys)
        if holdings.empty:
            holdings_text = "—"
        else:
            holdings_sorted = holdings.sort_values(ascending=False).head(5)
            holdings_text = ", ".join(f"{asset} ({weight * 100:.1f}%)" for asset, weight in holdings_sorted.items())
        composite_holdings_rows.append({"composite": comp_name, "top_holdings": holdings_text})

        if signals_df.empty:
            continue
        subset = signals_df[signals_df["strategy_key"].isin(keys)]
        if subset.empty:
            continue
        denom = max(len(keys), 1)
        grouped = subset.groupby("date")
        for date, block in grouped:
            buy_sum = float(block.loc[block["signal"] == "Buy", "magnitude"].sum())
            sell_sum = float(block.loc[block["signal"] == "Sell", "magnitude"].sum())
            net = (buy_sum - sell_sum) / denom
            if net == 0 and buy_sum == 0 and sell_sum == 0:
                continue
            height = float(min(comp_max_height, max(comp_min_height, comp_min_height + abs(net) * comp_scale)))
            composite_signal_rows.append(
                {
                    "date": date,
                    "composite": comp_name,
                    "net": net,
                    "direction": "Buy" if net >= 0 else "Sell",
                    "color": "#2ca02c" if net >= 0 else "#d62728",
                    "height": height,
                    "width": comp_bar_width,
                    "magnitude": abs(net),
                    "avg_buy": buy_sum / denom,
                    "avg_sell": sell_sum / denom,
                }
            )

    composite_signal_df = pd.DataFrame(composite_signal_rows)
    if composite_signal_df.empty:
        composite_signal_df = pd.DataFrame(
            {
                "date": [],
                "composite": [],
                "net": [],
                "direction": [],
                "color": [],
                "height": [],
                "width": [],
                "magnitude": [],
                "avg_buy": [],
                "avg_sell": [],
            }
        )
    composite_signal_source = ColumnDataSource(composite_signal_df)
    composite_holdings_df = pd.DataFrame(composite_holdings_rows)
    signal_y_labels = [opt[1] for opt in strategy_options]
    bar_width = 1000 * 60 * 60 * 24 * 2
    initial_signal_labels = [key_to_label.get(key, key) for key in default_visible_keys if key_to_label.get(key, key)]
    if not initial_signal_labels:
        initial_signal_labels = signal_y_labels
    signal_bar_fig = figure(
        title="Strategy Signal Timeline",
        x_axis_type="datetime",
        x_range=perf_fig.x_range,
        y_range=initial_signal_labels,
        height=260,
        sizing_mode="stretch_width",
        tools="xpan,xwheel_zoom,reset,save,tap",
    )
    rect_renderer = signal_bar_fig.rect(
        x="date",
        y="strategy_label",
        width="width",
        height="height",
        fill_color={"field": "color"},
        line_color={"field": "color"},
        alpha=0.75,
        source=filtered_signal_source,
    )
    signal_bar_fig.yaxis.axis_label = "Strategy"
    signal_bar_fig.xaxis.axis_label = "Date"
    signal_bar_fig.add_tools(
        HoverTool(
            tooltips=[
                ("Strategy", "@strategy_label"),
                ("Date", "@date{%F}"),
                ("Signal", "@signal"),
                ("Magnitude", "@magnitude{0.000}"),
            ],
            formatters={"@date": "datetime"},
            mode="mouse",
            renderers=[rect_renderer],
        )
    )
    signal_detail = Div(text="<b>Signal detail:</b> Click a bar to inspect that rebalance.")

    if not composite_signal_df.empty:
        composite_labels = list(composite_signal_df["composite"].unique())
    else:
        composite_labels = [name for name, _ in composite_defs]
    composite_labels = [label for label in composite_labels if label]

    if composite_labels:
        composite_fig = figure(
            title="Composite Recommendations (equal-weight top performers)",
            x_axis_type="datetime",
            x_range=perf_fig.x_range,
            y_range=composite_labels,
            height=220,
            sizing_mode="stretch_width",
            tools="xpan,xwheel_zoom,reset,save",
        )
        composite_renderer = composite_fig.rect(
            x="date",
            y="composite",
            width="width",
            height="height",
            fill_color={"field": "color"},
            line_color={"field": "color"},
            alpha=0.75,
            source=composite_signal_source,
        )
        composite_fig.yaxis.axis_label = "Composite"
        composite_fig.add_tools(
            HoverTool(
                tooltips=[
                    ("Composite", "@composite"),
                    ("Date", "@date{%F}"),
                    ("Direction", "@direction"),
                    ("Net Δ", "@net{0.000}"),
                    ("Avg Buy", "@avg_buy{0.000}"),
                    ("Avg Sell", "@avg_sell{0.000}"),
                ],
                formatters={"@date": "datetime"},
                mode="mouse",
                renderers=[composite_renderer],
            )
        )
    else:
        composite_fig = Div(text="Composite recommendations unavailable (insufficient strategy coverage).")

    if composite_holdings_df.empty:
        composite_holdings_table = Div(text="Composite holdings unavailable.")
    else:
        composite_holdings_source = ColumnDataSource(composite_holdings_df)
        composite_holdings_table = DataTable(
            source=composite_holdings_source,
            columns=[
                TableColumn(field="composite", title="Composite", width=160),
                TableColumn(field="top_holdings", title="Normalized Top Holdings", width=640),
            ],
            height=150,
            width=820,
            index_position=None,
        )

    selection_callback = CustomJS(
        args=dict(
            selector=strategy_select,
            line_renderers=line_renderers,
            full_signal_source=full_signal_source,
            filtered_signal_source=filtered_signal_source,
            signal_fig=signal_bar_fig,
            key_to_label=key_to_label,
            all_labels=signal_y_labels,
        ),
        code="""
        const selectedValues = selector.value;
        const filterAll = selectedValues.length === 0;
        const selectedSet = new Set(selectedValues);
        line_renderers.forEach((renderer) => {
            if (!renderer || !renderer.name) {
                return;
            }
            renderer.visible = filterAll ? true : selectedSet.has(renderer.name);
        });

        const raw = full_signal_source.data;
        const columnNames = Object.keys(raw);
        const empty = {};
        columnNames.forEach((name) => {
            empty[name] = [];
        });
        const hasData = columnNames.length > 0 && raw[columnNames[0]].length > 0;
        if (!hasData) {
            filtered_signal_source.data = empty;
            signal_fig.y_range.factors = [];
            return;
        }

        const includeKey = (key) => (filterAll ? true : selectedSet.has(key));
        const newData = {};
        columnNames.forEach((name) => {
            newData[name] = [];
        });
        const seen = new Set();
        const yLabels = [];
        for (let i = 0; i < raw.date.length; i++) {
            if (!includeKey(raw.strategy_key[i])) {
                continue;
            }
            columnNames.forEach((name) => {
                newData[name].push(raw[name][i]);
            });
            const label = raw.strategy_label ? raw.strategy_label[i] : raw.strategy[i];
            if (!seen.has(label)) {
                seen.add(label);
                yLabels.push(label);
            }
        }

        if (newData.date.length === 0 && !filterAll) {
            const fallbackLabels = [];
            selectedSet.forEach((key) => {
                const label = key_to_label[key];
                if (label) {
                    fallbackLabels.push(label);
                }
            });
            signal_fig.y_range.factors = fallbackLabels;
            filtered_signal_source.data = empty;
            filtered_signal_source.selected.indices = [];
            return;
        }

        filtered_signal_source.data = newData;
        signal_fig.y_range.factors = yLabels.length ? yLabels : (filterAll ? all_labels : []);
        filtered_signal_source.selected.indices = [];
        """
    )
    strategy_select.js_on_change("value", selection_callback)

    detail_callback = CustomJS(
        args=dict(source=filtered_signal_source, detail=signal_detail),
        code="""
        const indices = source.selected.indices;
        if (!indices || indices.length === 0) {
            detail.text = '<b>Signal detail:</b> Click a bar to inspect that rebalance.';
            return;
        }
        const idx = indices[indices.length - 1];
        const data = source.data;
        const toPct = (value) => isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
        const date = new Date(data.date[idx]).toISOString().slice(0, 10);
        const strategy = data.strategy_label ? data.strategy_label[idx] : data.strategy[idx];
        const signal = data.signal[idx];
        const magnitude = toPct(data.magnitude[idx]);
        const turnover = toPct(data.turnover ? data.turnover[idx] : NaN);
        const cash = toPct(data.cash ? data.cash[idx] : NaN);
        const grossBuy = toPct(data.gross_buy ? data.gross_buy[idx] : NaN);
        const grossSell = toPct(data.gross_sell ? data.gross_sell[idx] : NaN);
        const holdings = data.holdings ? data.holdings[idx] : '—';
        detail.text = `<b>${strategy}</b> ${signal} on <b>${date}</b> — Net Δ: ${magnitude}, Turnover: ${turnover}, Cash: ${cash}<br>` +
            `Gross buy: ${grossBuy} | Gross sell: ${grossSell}<br>` +
            `Top weights: ${holdings}`;
        """
    )
    filtered_signal_source.selected.js_on_change("indices", detail_callback)

    note = Div(
        text=(
            "<b>Strategy Lab</b><br>Comparisons cover momentum, relative-value, risk-based, and factor suites "
            "evaluated under rule-based, ML, or neutral regimes. Use the selector (Ctrl/Cmd + click) to focus the "
            "growth curves, see each strategy's buy/sell bars on its own line, and click any bar to read the "
            "rebalance details in the ticker below."
        )
    )

    diagnostics_layout = row(details_table, sizing_mode="stretch_width")
    comparison_row = row(perf_fig, selector_column, sizing_mode="stretch_width")
    layout_children = [note, comparison_row, signal_bar_fig, signal_detail]
    if signals_df.empty:
        layout_children.append(Div(text="No buy/sell signals exceeded the 2% threshold for the selected strategies."))
    layout_children.append(composite_fig)
    layout_children.append(composite_holdings_table)
    layout_children.extend([summary_table, diagnostics_layout])
    layout = column(*layout_children, sizing_mode="stretch_width")
    return TabPanel(title="Strategy Lab", child=layout)


def build_dashboard(
    timeline_path: Path = TIMELINE_PATH,
    weights_path: Path = WEIGHTS_PATH,
    prices_path: Path = PRICES_PATH,
    output_path: Path | None = None,
) -> Path:
    """Build complete dashboard with multiple tabs."""
    if output_path is None:
        output_path = REPORTS_DIR / "regime_dashboard.html"

    timeline = pd.read_parquet(timeline_path)
    weights = pd.read_parquet(weights_path)
    try:
        prices = pd.read_parquet(prices_path)
    except FileNotFoundError:
        prices = None

    # Build main dashboard panel (Tab 1: Regime Rules)
    main_panel = build_main_dashboard_panel(timeline, weights)
    
    # Build advanced analysis panel (Tab 2: vs ML + HRP)
    all_tabs = [main_panel]
    try:
        from .advanced_analysis import create_advanced_analysis_tab
        advanced_panel = create_advanced_analysis_tab()
        all_tabs.append(advanced_panel)
    except Exception as e:
        print(f"Warning: Could not build advanced analysis panel: {e}")
    
    # Build BRK.B comparison panel (Tab 3: vs BRK.B)
    try:
        from .brk_comparison import create_brk_comparison_tab
        brk_panel = create_brk_comparison_tab()
        all_tabs.append(brk_panel)
    except Exception as e:
        print(f"Warning: Could not build BRK.B comparison panel: {e}")
    
    try:
        strategy_panel = build_strategy_comparison_panel(timeline, weights, prices)
        all_tabs.append(strategy_panel)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Warning: Could not build strategy comparison panel: {e}")

    tabs = Tabs(tabs=all_tabs)

    output_file(output_path, title="RAAAL Dashboard")
    save(tabs)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bokeh dashboard from historical analysis")
    parser.add_argument("--timeline", type=Path, default=TIMELINE_PATH, help="Path to timeline parquet")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH, help="Path to weights parquet")
    parser.add_argument("--prices", type=Path, default=PRICES_PATH, help="Path to prices parquet")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "regime_dashboard.html", help="Output HTML path")
    args = parser.parse_args()

    path = build_dashboard(args.timeline, args.weights, args.prices, args.output)
    print(f"Dashboard saved to {path}")


if __name__ == "__main__":
    main()
