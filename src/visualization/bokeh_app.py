"""Generate interactive Bokeh dashboard for regimes and allocations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from bokeh.events import SelectionGeometry
from bokeh.io import output_file, save
from bokeh.layouts import column, row
from bokeh.models import (
    BoxAnnotation,
    BoxSelectTool,
    ColumnDataSource,
    CustomJS,
    DataTable,
    Div,
    HoverTool,
    LabelSet,
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
from bokeh.transform import dodge

from ..config import FOMO_COMPONENT_WEIGHTS, FOMO_SCORE_THRESHOLDS, UNIVERSE
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


def _beta_from_returns(
    asset_returns: pd.Series,
    market_returns: pd.Series,
) -> float:
    if len(asset_returns) < 2 or len(market_returns) < 2:
        return float("nan")
    cov = np.cov(asset_returns, market_returns)
    var = np.var(market_returns)
    if var == 0:
        return 0.0
    return float(cov[0, 1] / var)


def _salience_expectation(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    theta: float = 0.1,
    delta: float = 0.7,
) -> tuple[float, float]:
    if asset_returns.size == 0 or market_returns.size == 0:
        return float("nan"), float("nan")
    denom = np.abs(asset_returns) + np.abs(market_returns) + theta
    sigma = np.abs(asset_returns - market_returns) / denom
    order = np.argsort(-sigma)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    weights = np.power(delta, ranks.astype(float))
    if weights.sum() == 0:
        return float("nan"), float("nan")
    salience_probs = weights / weights.sum()
    salience_mean = np.sum(salience_probs * asset_returns)
    st_value = salience_mean - np.mean(asset_returns)
    return float(salience_mean), float(st_value)


def _salience_score(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    theta: float = 0.1,
    delta: float = 0.7,
) -> float:
    _, st_value = _salience_expectation(
        asset_returns,
        market_returns,
        theta=theta,
        delta=delta,
    )
    return st_value


def _build_salience_samples(
    prices: pd.DataFrame,
    tickers: list[str],
    rebalance_dates: list[pd.Timestamp] | None = None,
    lookback_days: int = 21,
    beta_window: int = 63,
    forward_window: int = 21,
    theta: float = 0.1,
    delta: float = 0.7,
) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    price_subset = prices[tickers].dropna(how="all")
    returns = price_subset.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()

    if rebalance_dates:
        anchor_dates = [pd.to_datetime(date) for date in rebalance_dates]
    else:
        month_groups = returns.groupby(pd.Grouper(freq="M"))
        anchor_dates = [
            group.index[-1] for _, group in month_groups if not group.empty
        ]
    market_returns = returns[tickers].mean(axis=1)

    rows = []
    for raw_date in anchor_dates:
        if returns.index.empty:
            continue
        available = returns.index[returns.index <= raw_date]
        if available.empty:
            continue
        date = available[-1]
        lookback_slice = returns.loc[:date].tail(lookback_days)
        beta_slice = returns.loc[:date].tail(beta_window)
        forward_slice = returns.loc[date:].iloc[1: forward_window + 1]
        if (
            len(lookback_slice) < lookback_days
            or len(beta_slice) < beta_window
            or len(forward_slice) < forward_window
        ):
            continue
        market_lookback = market_returns.loc[lookback_slice.index].to_numpy()
        market_beta = market_returns.loc[beta_slice.index]

        for ticker in tickers:
            asset_lookback = lookback_slice[ticker].to_numpy()
            predicted_return, st_value = _salience_expectation(
                asset_lookback,
                market_lookback,
                theta=theta,
                delta=delta,
            )
            beta_value = _beta_from_returns(beta_slice[ticker], market_beta)
            forward_return = (1 + forward_slice[ticker]).prod() - 1
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "st": float(st_value),
                    "beta": float(beta_value),
                    "predicted_return": float(predicted_return),
                    "forward_return": float(forward_return),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).dropna()


def _salience_quintile_summary(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if samples.empty:
        return pd.DataFrame(), pd.DataFrame()
    samples = samples.copy()
    samples["salience_group"] = np.where(
        samples["st"] >= 0,
        "Salient Upside",
        "Salient Downside",
    )

    rows = []
    for (date, group), sub in samples.groupby(["date", "salience_group"]):
        if len(sub) < 5 or sub["beta"].nunique() < 5:
            continue
        sub = sub.copy()
        try:
            sub["quintile"] = (
                pd.qcut(sub["beta"], 5, labels=False, duplicates="drop") + 1
            )
        except ValueError:
            continue
        if sub["quintile"].nunique() < 2:
            continue
        grouped = sub.groupby(
            "quintile", as_index=False
        )["forward_return"].mean()
        grouped["date"] = date
        grouped["salience_group"] = group
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    monthly = pd.concat(rows, ignore_index=True)
    summary = monthly.groupby(
        ["salience_group", "quintile"], as_index=False
    )["forward_return"].mean()
    summary = summary.sort_values(["salience_group", "quintile"])
    return monthly, summary


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

    nowcast_cols = [col for col in timeline.columns if col.startswith("nowcast_")]
    nowcast_labels = {
        col: col.replace("nowcast_", "").replace("_", " ").title()
        for col in nowcast_cols
    }
    nowcast_frame = timeline[nowcast_cols].copy().sort_index() if nowcast_cols else pd.DataFrame(index=timeline.index)

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

    def _nowcast_snapshot(target_date: pd.Timestamp | None) -> dict[str, float]:
        if (
            not nowcast_cols
            or nowcast_frame.empty
            or target_date is None
            or pd.isna(target_date)
        ):
            return {}
        index = nowcast_frame.index
        match_idx = None
        if target_date in index:
            match_idx = target_date
        else:
            try:
                loc = index.get_indexer([target_date], method="ffill")
            except Exception:
                loc = np.array([-1])
            if loc.size and loc[0] != -1:
                match_idx = index[loc[0]]
        if match_idx is None:
            return {}
        row = nowcast_frame.loc[match_idx]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return {col: float(row.get(col, float("nan"))) for col in nowcast_cols}

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
            signal_entry = {
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
            nowcast_values = _nowcast_snapshot(date)
            for col in nowcast_cols:
                signal_entry[col] = nowcast_values.get(col, float("nan"))
            signal_rows.append(signal_entry)

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

    for col in nowcast_cols:
        if col not in signals_df.columns:
            signals_df[col] = float("nan")

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
        args=dict(
            source=filtered_signal_source,
            detail=signal_detail,
            macro_columns=nowcast_cols,
            macro_labels=nowcast_labels,
        ),
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
        if (macro_columns && macro_columns.length) {
            const macroParts = [];
            macro_columns.forEach((col) => {
                const column = data[col];
                if (!column || column.length <= idx) {
                    return;
                }
                const value = column[idx];
                if (!isFinite(value)) {
                    return;
                }
                const label = macro_labels && macro_labels[col] ? macro_labels[col] : col;
                const formatted = `${value >= 0 ? '+' : ''}${value.toFixed(2)}`;
                macroParts.push(`${label}: ${formatted}`);
            });
            if (macroParts.length) {
                detail.text += '<br><i>Macro tilts:</i> ' + macroParts.join(' · ');
            }
        }
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


def build_salience_panel(
    prices: pd.DataFrame | None,
    timeline: pd.DataFrame | None = None,
) -> TabPanel:
    """Visualize salience theory metrics using project price data."""
    if prices is None or prices.empty:
        return TabPanel(
            title="Salience",
            child=Div(text="Price history unavailable."),
        )

    lookback_days = 21
    beta_window = 63
    forward_window = 21

    rebalance_dates = None
    if timeline is not None and not timeline.empty:
        if "date" in timeline.columns:
            rebalance_dates = timeline["date"].dropna().tolist()
        else:
            rebalance_dates = timeline.index.tolist()

    tickers = [
        asset.ticker
        for asset in UNIVERSE
        if asset.ticker in prices.columns
    ]
    if len(tickers) < 3:
        return TabPanel(
            title="Salience",
            child=Div(text="Not enough assets to compute salience metrics."),
        )

    samples = _build_salience_samples(
        prices,
        tickers,
        rebalance_dates=rebalance_dates,
        lookback_days=lookback_days,
        beta_window=beta_window,
        forward_window=forward_window,
    )
    if samples.empty:
        return TabPanel(
            title="Salience",
            child=Div(text="Salience samples could not be computed."),
        )

    samples["salience_group"] = np.where(
        samples["st"] >= 0,
        "Salient Upside",
        "Salient Downside",
    )

    monthly, summary = _salience_quintile_summary(samples)
    if summary.empty:
        return TabPanel(
            title="Salience",
            child=Div(text="Salience beta/return summary unavailable."),
        )

    quintiles = pd.DataFrame({"quintile": [1, 2, 3, 4, 5]})

    def _group_series(group: str) -> ColumnDataSource:
        subset = summary[summary["salience_group"] == group][
            ["quintile", "forward_return"]
        ]
        merged = quintiles.merge(subset, on="quintile", how="left")
        return ColumnDataSource(
            {
                "quintile": merged["quintile"].tolist(),
                "forward_return": merged["forward_return"].tolist(),
            }
        )

    upside_source = _group_series("Salient Upside")
    downside_source = _group_series("Salient Downside")

    line_fig = figure(
        title="Forward return by beta quintile",
        x_axis_label="Beta quintile",
        y_axis_label=(
            "Avg post-rebalancing return "
            f"(next {forward_window} trading days)"
        ),
        height=300,
        sizing_mode="stretch_width",
        tools="xpan,xwheel_zoom,reset,save",
    )
    line_fig.line(
        "quintile",
        "forward_return",
        source=upside_source,
        color="#1f77b4",
        line_width=2,
        legend_label="Salient Upside",
    )
    line_fig.circle(
        "quintile",
        "forward_return",
        source=upside_source,
        color="#1f77b4",
        size=7,
    )
    line_fig.line(
        "quintile",
        "forward_return",
        source=downside_source,
        color="#d62728",
        line_width=2,
        legend_label="Salient Downside",
    )
    line_fig.circle(
        "quintile",
        "forward_return",
        source=downside_source,
        color="#d62728",
        size=7,
    )
    line_fig.legend.location = "top_left"
    line_fig.add_tools(
        HoverTool(
            tooltips=[
                ("Quintile", "@quintile"),
                ("Post-rebalancing Return", "@forward_return{0.00%}"),
            ]
        )
    )

    def _regression(subset: pd.DataFrame) -> tuple[float, float] | None:
        if subset.empty or subset["beta"].nunique() < 2:
            return None
        x = subset["beta"].to_numpy()
        y = subset["forward_return"].to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)

    reg_fig = figure(
        title="Beta vs forward return (regression)",
        x_axis_label="Beta",
        y_axis_label=(
            "Post-rebalancing return "
            f"(next {forward_window} trading days)"
        ),
        height=320,
        sizing_mode="stretch_width",
        tools="xpan,xwheel_zoom,reset,save",
    )

    upside_samples = samples[samples["salience_group"] == "Salient Upside"]
    downside_samples = samples[samples["salience_group"] == "Salient Downside"]
    upside_source = ColumnDataSource(upside_samples)
    downside_source = ColumnDataSource(downside_samples)
    reg_fig.circle(
        "beta",
        "forward_return",
        source=upside_source,
        color="#1f77b4",
        alpha=0.6,
        size=6,
        legend_label="Salient Upside",
    )
    reg_fig.circle(
        "beta",
        "forward_return",
        source=downside_source,
        color="#d62728",
        alpha=0.6,
        size=6,
        legend_label="Salient Downside",
    )

    slope_lines = []
    slope_text = []
    for label, subset, color in (
        ("Salient Upside", upside_samples, "#1f77b4"),
        ("Salient Downside", downside_samples, "#d62728"),
    ):
        params = _regression(subset)
        if not params:
            continue
        slope, intercept = params
        x_min = float(subset["beta"].min())
        x_max = float(subset["beta"].max())
        x_line = np.linspace(x_min, x_max, 50)
        y_line = slope * x_line + intercept
        slope_lines.append(
            reg_fig.line(
                x_line,
                y_line,
                line_width=2,
                color=color,
                line_dash="dashed",
            )
        )
        slope_text.append(f"{label} slope: {slope:+.4f}")

    reg_fig.legend.location = "top_left"
    reg_fig.add_tools(
        HoverTool(
            tooltips=[
                ("Ticker", "@ticker"),
                ("Beta", "@beta{0.00}"),
                ("Post-rebalancing Return", "@forward_return{0.00%}"),
                ("Salience", "@salience_group"),
            ]
        )
    )

    slope_summary = Div(
        text=(
            "<b>Regression slopes</b><br>" + "<br>".join(slope_text)
            if slope_text
            else "<b>Regression slopes</b><br>Not enough data"
        )
    )

    latest_date = samples["date"].max()
    latest_samples = samples[samples["date"] == latest_date].copy()
    latest_samples["color"] = np.where(
        latest_samples["st"] >= 0,
        "#2ca02c",
        "#d62728",
    )
    latest_samples = latest_samples.sort_values("st", ascending=False)
    latest_source = ColumnDataSource(latest_samples)
    bar_fig = figure(
        title=(
            "Latest salience vs post-rebalancing return "
            f"({latest_date:%Y-%m-%d})"
        ),
        x_range=latest_samples["ticker"].tolist(),
        height=320,
        sizing_mode="stretch_width",
        tools="xpan,xwheel_zoom,reset,save",
    )
    bar_fig.vbar(
        x=dodge("ticker", -0.18, range=bar_fig.x_range),
        top="predicted_return",
        width=0.35,
        color="#1f77b4",
        source=latest_source,
        legend_label="Predicted (salience-weighted)",
    )
    bar_fig.vbar(
        x=dodge("ticker", 0.18, range=bar_fig.x_range),
        top="forward_return",
        width=0.35,
        color="#ff7f0e",
        source=latest_source,
        legend_label="Post-rebalancing (realized)",
    )
    bar_fig.xgrid.grid_line_color = None
    bar_fig.yaxis.axis_label = "Return"
    bar_fig.legend.location = "top_left"
    bar_fig.add_tools(
        HoverTool(
            tooltips=[
                ("Ticker", "@ticker"),
                ("ST", "@st{0.000}"),
                ("Beta", "@beta{0.00}"),
                ("Predicted", "@predicted_return{0.00%}"),
                ("Post-rebalancing", "@forward_return{0.00%}"),
            ]
        )
    )

    table = DataTable(
        source=latest_source,
        columns=[
            TableColumn(field="ticker", title="Ticker"),
            TableColumn(
                field="st",
                title="ST",
                formatter=NumberFormatter(format="0.000"),
            ),
            TableColumn(
                field="beta",
                title="Beta",
                formatter=NumberFormatter(format="0.00"),
            ),
            TableColumn(
                field="predicted_return",
                title="Predicted Return",
                formatter=NumberFormatter(format="0.00%"),
            ),
            TableColumn(
                field="forward_return",
                title="Post-rebalancing Return",
                formatter=NumberFormatter(format="0.00%"),
            ),
        ],
        height=260,
        width=420,
        index_position=None,
    )

    def _salience_portfolio_performance(
        sample_frame: pd.DataFrame,
        smooth_window: int = 3,
    ) -> pd.DataFrame:
        rows = []
        for date, sub in sample_frame.groupby("date"):
            predicted = sub["predicted_return"].copy()
            if predicted.isna().all():
                continue
            positive = predicted.clip(lower=0)
            if positive.sum() > 0:
                weights = positive / positive.sum()
            else:
                weights = pd.Series(1 / len(sub), index=sub.index)
            predicted_port = float((weights * predicted).sum())
            actual_port = float((weights * sub["forward_return"]).sum())
            rows.append(
                {
                    "date": date,
                    "predicted": predicted_port,
                    "actual": actual_port,
                }
            )
        if not rows:
            return pd.DataFrame()
        perf = pd.DataFrame(rows).sort_values("date")
        perf["predicted_nav"] = (1 + perf["predicted"]).cumprod()
        perf["actual_nav"] = (1 + perf["actual"]).cumprod()
        perf["predicted_smooth"] = (
            perf["predicted"].rolling(smooth_window).mean()
        )
        perf["actual_smooth"] = perf["actual"].rolling(smooth_window).mean()
        return perf

    perf = _salience_portfolio_performance(samples)
    if perf.empty:
        perf_fig = Div(text="Salience allocation performance unavailable.")
        perf_period_fig = Div(text="Salience period returns unavailable.")
    else:
        perf_source = ColumnDataSource(perf)
        perf_fig = figure(
            title="Salience-weighted allocation performance",
            x_axis_type="datetime",
            height=300,
            sizing_mode="stretch_width",
            tools="xpan,xwheel_zoom,reset,save",
        )
        perf_fig.line(
            "date",
            "predicted_nav",
            source=perf_source,
            color="#1f77b4",
            line_width=2,
            legend_label="Predicted (salience-weighted)",
        )
        perf_fig.line(
            "date",
            "actual_nav",
            source=perf_source,
            color="#ff7f0e",
            line_width=2,
            legend_label="Post-rebalancing (realized)",
        )
        perf_fig.yaxis.axis_label = "Cumulative growth"
        perf_fig.legend.location = "top_left"
        perf_fig.add_tools(
            HoverTool(
                tooltips=[
                    ("Date", "@date{%F}"),
                    ("Predicted", "@predicted_nav{0.000}"),
                    ("Actual", "@actual_nav{0.000}"),
                ],
                formatters={"@date": "datetime"},
            )
        )

        perf_period_fig = figure(
            title="Salience-weighted period returns",
            x_axis_type="datetime",
            height=260,
            sizing_mode="stretch_width",
            tools="xpan,xwheel_zoom,reset,save",
        )
        perf_period_fig.vbar(
            x="date",
            top="predicted",
            width=1000 * 60 * 60 * 24 * 3,
            color="#1f77b4",
            alpha=0.6,
            source=perf_source,
            legend_label="Predicted",
        )
        perf_period_fig.vbar(
            x="date",
            top="actual",
            width=1000 * 60 * 60 * 24 * 3,
            color="#ff7f0e",
            alpha=0.6,
            source=perf_source,
            legend_label="Post-rebalancing",
        )
        perf_period_fig.line(
            "date",
            "predicted_smooth",
            source=perf_source,
            color="#1f77b4",
            line_width=2,
            line_dash="dashed",
            legend_label="Predicted (3-period avg)",
        )
        perf_period_fig.line(
            "date",
            "actual_smooth",
            source=perf_source,
            color="#ff7f0e",
            line_width=2,
            line_dash="dashed",
            legend_label="Post-rebalancing (3-period avg)",
        )
        perf_period_fig.yaxis.axis_label = "Return"
        perf_period_fig.legend.location = "top_left"
        perf_period_fig.add_tools(
            HoverTool(
                tooltips=[
                    ("Date", "@date{%F}"),
                    ("Predicted", "@predicted{0.00%}"),
                    ("Post-rebalancing", "@actual{0.00%}"),
                ],
                formatters={"@date": "datetime"},
            )
        )

    description = Div(
        text=(
            "<b>Salience methodology (BGS 2012)</b><br>"
            "(Cosemans & Frehen 2021 implementation)<br>"
            f"Daily returns over the last {lookback_days} trading days "
            "approximate the state space.<br>"
            "For each asset, we compute the salience function versus the "
            "equal-weighted market, rank states by salience, and apply "
            "salience "
            "weights ($\\theta=0.1$, $\\delta=0.7$).<br>"
            "The salience distortion ST is the difference between salience-"
            "weighted and equal-weighted expected returns.<br>"
            "Positive ST denotes salient upside; negative ST denotes salient "
            "downside. Samples align to rebalance dates.<br>"
            "Predicted returns are salience-weighted expectations; realized "
            "post-rebalancing returns are observed outcomes."
        )
    )

    layout = column(
        description,
        line_fig,
        row(reg_fig, slope_summary, sizing_mode="stretch_width"),
        perf_fig,
        perf_period_fig,
        row(bar_fig, table, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )
    return TabPanel(title="Salience", child=layout)


def build_fomo_fobi_panel(timeline: pd.DataFrame) -> TabPanel:
    """Visualize the FOMO vs FOBI composite indicator in its own tab."""

    if "fomo_fobi_score" not in timeline.columns:
        raise ValueError("Timeline missing FOMO/FOBI columns")

    indicator = timeline.reset_index().rename(columns={"index": "date"})
    indicator = indicator.dropna(subset=["fomo_fobi_score"])
    if indicator.empty:
        return TabPanel(title="FOMO vs FOBI", child=Div(text="Indicator not available for the selected period."))

    indicator["date"] = pd.to_datetime(indicator["date"])
    indicator["date_ms"] = indicator["date"].astype("int64") / 10**6
    source = ColumnDataSource(indicator)

    score_fig = figure(
        title="Composite FOMO/FOBI Score",
        x_axis_type="datetime",
        sizing_mode="stretch_width",
        height=300,
        tools="xpan,xwheel_zoom,reset,save,tap",
    )
    score_renderer = score_fig.line(
        "date",
        "fomo_fobi_score",
        source=source,
        color="#d62728",
        line_width=3,
        legend_label="Score",
    )
    hi = FOMO_SCORE_THRESHOLDS["fomo"]
    lo = FOMO_SCORE_THRESHOLDS["fobi"]
    score_fig.line(x=indicator["date"], y=[hi] * len(indicator), color="#ff7f0e", line_dash="dashed", legend_label="FOMO threshold")
    score_fig.line(x=indicator["date"], y=[lo] * len(indicator), color="#1f77b4", line_dash="dotted", legend_label="FOBI threshold")
    score_fig.yaxis.axis_label = "Z-score"
    score_fig.legend.location = "top_left"
    score_fig.extra_y_ranges = {"prob": Range1d(start=0, end=1)}
    score_fig.add_layout(LinearAxis(y_range_name="prob", axis_label="Probability"), "right")
    prob_renderer = score_fig.line(
        "date",
        "fomo_probability",
        source=source,
        color="#2ca02c",
        line_width=2,
        y_range_name="prob",
        legend_label="FOMO probability (be risk-off)",
    )
    hover = HoverTool(
        tooltips=[
            ("Date", "@date{%F}"),
            ("Score", "@fomo_fobi_score{0.00}"),
            ("State", "@fomo_fobi_state"),
            ("Probability", "@fomo_probability{0.00}"),
        ],
        formatters={"@date": "datetime"},
        mode="vline",
        renderers=[score_renderer, prob_renderer],
    )
    score_fig.add_tools(hover)

    latest = indicator.iloc[-1]
    component_names = []
    component_scores = []
    component_columns = []
    for comp in FOMO_COMPONENT_WEIGHTS:
        component_col = f"fomo_component_{comp}_z"
        if component_col not in indicator.columns:
            continue
        component_names.append(comp.replace("_", " ").title())
        component_scores.append(float(latest.get(component_col, float("nan"))))
        component_columns.append(component_col)

    latest_component_label = latest["date"].strftime("%Y-%m-%d")
    COMPONENT_LABEL_OFFSET = 0.15
    comp_source: ColumnDataSource | None = None
    if component_names:
        palette = Category10[max(3, min(10, len(component_names)))]
        colors = [palette[i % len(palette)] for i in range(len(component_names))]
        labels = ["{:.2f}".format(value) if pd.notna(value) else "—" for value in component_scores]
        label_positions = []
        for value in component_scores:
            if not np.isfinite(value):
                label_positions.append(0.0)
            elif abs(value) < COMPONENT_LABEL_OFFSET:
                label_positions.append(value + (COMPONENT_LABEL_OFFSET if value >= 0 else -COMPONENT_LABEL_OFFSET))
            else:
                label_positions.append(value / 2)
        comp_source = ColumnDataSource(
            {
                "component": component_names,
                "score": component_scores,
                "color": colors,
                "score_text": labels,
                "label_y": label_positions,
            }
        )
        component_fig = figure(
            title=f"Component z-scores on {latest_component_label}",
            x_range=component_names,
            height=280,
            sizing_mode="stretch_width",
            tools="reset,save",
        )
        component_fig.vbar(x="component", top="score", width=0.7, color="color", source=comp_source)
        component_fig.yaxis.axis_label = "Z-score"
        component_fig.xaxis.major_label_orientation = 0.9
        labels = LabelSet(
            x="component",
            y="label_y",
            text="score_text",
            text_color="black",
            text_font_style="bold",
            level="glyph",
            source=comp_source,
            text_align="center",
        )
        component_fig.add_layout(labels)
    else:
        component_fig = Div(text="Component breakdown unavailable.")

    price_fig = figure(
        title="SPY vs Gold",
        x_axis_type="datetime",
        sizing_mode="stretch_width",
        height=250,
        tools="xpan,xwheel_zoom,reset,save,tap",
    )
    price_fig.x_range = score_fig.x_range
    price_fig.yaxis.axis_label = "SPY"
    price_renderer = price_fig.line("date", "spy_price", source=source, color="#1f77b4", line_width=2, legend_label="SPY")
    spy_series = indicator.get("spy_price")
    if spy_series is not None and pd.notna(spy_series).any():
        spy_values = spy_series[pd.notna(spy_series)]
        spy_min = float(spy_values.min())
        spy_max = float(spy_values.max())
    else:
        spy_min, spy_max = 0.0, 1.0
    spy_pad = max((spy_max - spy_min) * 0.08, 1.0)
    price_fig.y_range = Range1d(start=spy_min - spy_pad, end=spy_max + spy_pad)
    gold_series = indicator.get("gold_price_oz")
    if gold_series is not None:
        gold_min = float(gold_series.min()) if pd.notna(gold_series).any() else 0.0
        gold_max = float(gold_series.max()) if pd.notna(gold_series).any() else 1.0
    else:
        gold_min, gold_max = 0.0, 1.0
    gold_pad = max((gold_max - gold_min) * 0.05, 1.0)
    gold_range = Range1d(start=gold_min - gold_pad, end=gold_max + gold_pad)
    price_fig.extra_y_ranges = {"gold": gold_range}
    price_fig.add_layout(LinearAxis(y_range_name="gold", axis_label="Gold (oz)"), "right")
    gold_renderer = price_fig.line(
        "date",
        "gold_price_oz",
        source=source,
        color="#ffbf00",
        line_dash="dashed",
        line_width=2,
        y_range_name="gold",
        legend_label="Gold",
    )
    price_fig.legend.location = "top_left"
    price_fig.add_tools(
        HoverTool(
            tooltips=[
                ("Date", "@date{%F}"),
                ("SPY", "@spy_price{0.0}")
            ],
            formatters={"@date": "datetime"},
            mode="vline",
            renderers=[price_renderer],
        )
    )
    price_fig.add_tools(
        HoverTool(
            tooltips=[
                ("Date", "@date{%F}"),
                ("Gold", "@gold_price_oz{0.0}")
            ],
            formatters={"@date": "datetime"},
            mode="vline",
            renderers=[gold_renderer],
        )
    )
    slider = Slider(
        start=0,
        end=len(source.data["date"]) - 1,
        value=len(source.data["date"]) - 1,
        step=1,
        title="Timeline index",
        visible=False,
    )
    box_select = BoxSelectTool(dimensions="width")
    score_fig.add_tools(box_select)
    score_fig.toolbar.active_drag = box_select
    selection_box_score = BoxAnnotation(left=None, right=None, fill_alpha=0.1, fill_color="#c5d5f5", line_color=None)
    selection_box_price = BoxAnnotation(left=None, right=None, fill_alpha=0.1, fill_color="#c5d5f5", line_color=None)
    score_fig.add_layout(selection_box_score)
    price_fig.add_layout(selection_box_price)
    initial_date_ms = float(source.data["date_ms"][0]) if len(source.data["date_ms"]) else 0.0
    span_score_end = Span(location=initial_date_ms, dimension="height", line_color="black", line_width=2)
    span_price_end = Span(location=initial_date_ms, dimension="height", line_color="black", line_width=2)
    span_score_start = Span(location=initial_date_ms, dimension="height", line_color="#555555", line_width=2, line_dash="dashed")
    span_price_start = Span(location=initial_date_ms, dimension="height", line_color="#555555", line_width=2, line_dash="dashed")
    score_fig.add_layout(span_score_start)
    price_fig.add_layout(span_price_start)
    score_fig.add_layout(span_score_end)
    price_fig.add_layout(span_price_end)
    selection_state = ColumnDataSource(data={"active": [0]})

    slider_callback = CustomJS(
        args=dict(
            slider=slider,
            source=source,
            span_score_end=span_score_end,
            span_price_end=span_price_end,
            span_score_start=span_score_start,
            span_price_start=span_price_start,
            selection_state=selection_state,
        ),
        code="""
        const idx = slider.value;
        const dateMs = source.data['date_ms'][idx];
        if (dateMs === undefined) { return; }
        span_score_end.location = dateMs;
        span_price_end.location = dateMs;
        span_score_end.change.emit();
        span_price_end.change.emit();
        const isWindowActive = selection_state.data['active'] && selection_state.data['active'][0] === 1;
        if (!isWindowActive) {
            span_score_start.location = dateMs;
            span_price_start.location = dateMs;
            span_score_start.change.emit();
            span_price_start.change.emit();
        }
        """,
    )
    slider.js_on_change("value", slider_callback)

    const_update = """
        const x = cb_obj.x;
        if (x === undefined || x === null) { return; }
        const dates = source.data['date'];
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
    """;
    tap_callback = CustomJS(args=dict(slider=slider, source=source), code=const_update)
    score_fig.js_on_event("tap", tap_callback)
    price_fig.js_on_event("tap", tap_callback)

    selection_summary = Div(text="Drag across the score chart to measure SPY vs Gold performance across that window.")
    selection_callback = CustomJS(
        args=dict(
            source=source,
            summary=selection_summary,
            box_score=selection_box_score,
            box_price=selection_box_price,
            slider=slider,
            default_text="Drag across the score chart to measure SPY vs Gold performance across that window.",
            component_source=comp_source if component_names else None,
            component_columns=component_columns,
            component_fig=component_fig if component_names else None,
            component_label_offset=COMPONENT_LABEL_OFFSET,
            latest_component_index=len(source.data["date"]) - 1,
            latest_component_label=latest_component_label,
            span_score_start=span_score_start,
            span_score_end=span_score_end,
            span_price_start=span_price_start,
            span_price_end=span_price_end,
            selection_state=selection_state,
        ),
        code="""
        if (!cb_obj.final) { return; }
        const geometry = cb_obj.geometry || {};
        const x0raw = geometry.x0;
        const x1raw = geometry.x1;
        const dates = source.data['date'];
        const setComponents = (idx, labelText) => {
            if (!component_source || !component_columns || component_columns.length === 0) {
                return;
            }
            const scores = [];
            const scoreText = [];
            const labelY = [];
            for (let i = 0; i < component_columns.length; i++) {
                const column = component_columns[i];
                const series = source.data[column] || [];
                const value = series[idx];
                scores.push(value);
                if (Number.isFinite(value)) {
                    scoreText.push(value.toFixed(2));
                    if (Math.abs(value) < component_label_offset) {
                        labelY.push(value >= 0 ? value + component_label_offset : value - component_label_offset);
                    } else {
                        labelY.push(value / 2);
                    }
                } else {
                    scoreText.push('—');
                    labelY.push(0);
                }
            }
            component_source.data.score = scores;
            component_source.data.score_text = scoreText;
            component_source.data.label_y = labelY;
            component_source.change.emit();
            if (component_fig && component_fig.title) {
                component_fig.title.text = `Component z-scores on ${labelText}`;
            }
        };
        const resetState = () => {
            summary.text = default_text;
            box_score.left = null;
            box_score.right = null;
            box_price.left = null;
            box_price.right = null;
            box_score.change.emit();
            box_price.change.emit();
            if (dates && dates.length) {
                const idx = dates.length - 1;
                slider.value = idx;
                const dateMs = source.data['date_ms'];
                if (dateMs && dateMs.length) {
                    const ms = dateMs[idx];
                    span_score_start.location = ms;
                    span_price_start.location = ms;
                    span_score_end.location = ms;
                    span_price_end.location = ms;
                    span_score_start.change.emit();
                    span_price_start.change.emit();
                    span_score_end.change.emit();
                    span_price_end.change.emit();
                }
            }
            setComponents(latest_component_index, latest_component_label);
            if (selection_state && selection_state.data && selection_state.data['active']) {
                selection_state.data['active'][0] = 0;
                selection_state.change.emit();
            }
        };
        if (!Number.isFinite(x0raw) || !Number.isFinite(x1raw) || !dates || !dates.length) {
            resetState();
            return;
        }
        let x0 = x0raw;
        let x1 = x1raw;
        if (x0 > x1) {
            const tmp = x0;
            x0 = x1;
            x1 = tmp;
        }
        if (Math.abs(x1 - x0) < 1) {
            // treat as click — only sync slider and revert multi-chart context
            let best = 0;
            let minDiff = Infinity;
            for (let i = 0; i < dates.length; i++) {
                const diff = Math.abs(dates[i] - x0);
                if (diff < minDiff) {
                    minDiff = diff;
                    best = i;
                }
            }
            slider.value = best;
            resetState();
            return;
        }
        const nearestIndex = (target) => {
            let best = 0;
            let minDiff = Infinity;
            for (let i = 0; i < dates.length; i++) {
                const diff = Math.abs(dates[i] - target);
                if (diff < minDiff) {
                    minDiff = diff;
                    best = i;
                }
            }
            return best;
        };
        const startIdx = nearestIndex(x0);
        const endIdx = nearestIndex(x1);
        const spy = source.data['spy_price'] || [];
        const gold = source.data['gold_price_oz'] || [];
        const pctChange = (start, end) => (Number.isFinite(start) && Number.isFinite(end) && start !== 0)
            ? (end - start) / start
            : NaN;
        const fmt = (value) => Number.isFinite(value)
            ? `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`
            : '—';
        const startDate = new Date(dates[startIdx]).toISOString().slice(0, 10);
        const endDate = new Date(dates[endIdx]).toISOString().slice(0, 10);
        const daySpan = Math.max(0, Math.round((dates[endIdx] - dates[startIdx]) / (24 * 60 * 60 * 1000)));
        const spyDelta = fmt(pctChange(spy[startIdx], spy[endIdx]));
        const goldDelta = fmt(pctChange(gold[startIdx], gold[endIdx]));
        summary.text = `<b>${startDate} → ${endDate}</b> (${daySpan} days)<br>SPY: ${spyDelta} | Gold: ${goldDelta}`;
        const startDateValue = dates[startIdx];
        const endDateValue = dates[endIdx];
        box_score.left = startDateValue;
        box_score.right = endDateValue;
        box_price.left = startDateValue;
        box_price.right = endDateValue;
        box_score.change.emit();
        box_price.change.emit();
        slider.value = endIdx;
        const dateMs = source.data['date_ms'] || [];
        const startMs = dateMs[startIdx];
        const endMs = dateMs[endIdx];
        if (Number.isFinite(startMs) && Number.isFinite(endMs)) {
            span_score_start.location = startMs;
            span_price_start.location = startMs;
            span_score_end.location = endMs;
            span_price_end.location = endMs;
            span_score_start.change.emit();
            span_price_start.change.emit();
            span_score_end.change.emit();
            span_price_end.change.emit();
        }
        if (selection_state && selection_state.data && selection_state.data['active']) {
            selection_state.data['active'][0] = 1;
            selection_state.change.emit();
        }
        setComponents(startIdx, startDate);
        """,
    )
    score_fig.js_on_event(SelectionGeometry, selection_callback)
    summary = Div(
        text=(
            "<b>How to read</b><br>"
            "• Score above {:.2f} → institutions in FOMO (reduce risk)<br>"
            "• Score below {:.2f} → FOBI capitulation (deploy risk)<br>"
            "Components blend breadth, mega-cap concentration, cash positioning proxies, Berkshire cash posture, and volatility complacency."
        ).format(hi, lo)
    )

    slider.value = len(source.data["date"]) - 1

    layout = column(score_fig, price_fig, selection_summary, component_fig, summary, sizing_mode="stretch_width")
    return TabPanel(title="FOMO vs FOBI", child=layout)


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
    fomo_panel = None
    try:
        from .advanced_analysis import create_advanced_analysis_tab
        advanced_panel = create_advanced_analysis_tab()
        all_tabs.append(advanced_panel)
    except Exception as e:
        print(f"Warning: Could not build advanced analysis panel: {e}")

    try:
        salience_panel = build_salience_panel(prices, timeline)
        all_tabs.append(salience_panel)
    except Exception as e:
        print(f"Warning: Could not build salience panel: {e}")
    
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

    if fomo_panel is None:
        try:
            fomo_panel = build_fomo_fobi_panel(timeline)
        except Exception as e:
            print(f"Warning: Could not build FOMO/FOBI panel: {e}")
    if fomo_panel is not None:
        all_tabs.append(fomo_panel)

    # Sections replace tabs: one anchored block per panel, stacked so the whole
    # dashboard is a single scroll and the nav is a set of links rather than a
    # tab strip. Each `TabPanel` still carries the figures a builder produced —
    # we take its `.child` layout and embed it under an HTML heading and a short
    # blurb, with the panel's own longer text left where the builder placed it,
    # below the graphs.
    sections = [(_slug(str(p.title)), str(p.title), p.child) for p in all_tabs]
    _write_landing(output_path, sections)
    return output_path


#: A one-line description per section, shown under its heading. Lifted from the
#: old "how to read these tabs" legend so the copy stays in one voice; the long
#: methodology text stays inside each panel, below its graphs.
SECTION_BLURBS = {
    "The Strategy": "The market regime we detect right now, and the allocation "
                    "it drives.",
    "vs Academia": "Our rule-based regime detection against an ML ensemble, "
                   "with factor and network analysis.",
    "Salience": "A behavioural-finance view — forward returns sorted by beta, "
                "read through salience theory.",
    "vs Buffett": "The strategy benchmarked against Berkshire Hathaway over the "
                  "same period.",
    "Strategy Lab": "The ~20 research strategies the planner chooses among — "
                    "growth curves, signals and the composite.",
    "FOMO vs FOBI": "A composite sentiment indicator for risk-on / risk-off "
                    "positioning.",
}


def _slug(title: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section"


def _write_landing(output_path: Path, sections) -> None:
    """Assemble the scrolling landing: hero, section links, graphs, prompt.

    The graphs come from Bokeh's `components` (one script, one div per section),
    embedded into a hand-authored shell so the page can carry an anchor nav and
    an HTML form Bokeh could not. The full set of BokehJS bundles is loaded from
    the CDN by version, because the panels use tables and sliders, not only
    figures.
    """
    import html as _html

    from bokeh.embed import components
    from bokeh.resources import CDN

    models = {key: child for key, _title, child in sections}
    script, divs = components(models)
    js = "\n".join(f'<script src="{u}"></script>' for u in CDN.js_files)

    nav_links = "".join(
        f'<a href="#{key}">{_html.escape(title)}</a>'
        for key, title, _child in sections)
    nav_links += '<a href="#try" class="try">Try your own &rarr;</a>'

    blocks = []
    for key, title, _child in sections:
        blurb = SECTION_BLURBS.get(title, "")
        blocks.append(f"""
      <section id="{key}" class="sec">
        <h2>{_html.escape(title)}</h2>
        <p class="blurb">{_html.escape(blurb)}</p>
        <div class="graph">{divs.get(key, "")}</div>
      </section>""")

    body = "\n".join(blocks)
    page = _LANDING_SHELL.format(js=js, script=script, nav=nav_links, body=body)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)


_LANDING_SHELL = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAAAL — Research Dashboard (DEMO, not investment advice)</title>
{js}
<style>
  :root {{ --ink:#16202b; --muted:#5b6673; --line:#e6e9ee; --accent:#2563eb;
           --bg:#fbfcfd; }}
  * {{ box-sizing:border-box; }}
  body {{ font:15px/1.55 -apple-system,system-ui,sans-serif; color:var(--ink);
          background:var(--bg); margin:0; }}
  .wrap {{ max-width:1120px; margin:0 auto; padding:0 20px; }}
  header.hero {{ padding:34px 0 10px; }}
  header.hero h1 {{ margin:0 0 6px; font-size:1.85rem; letter-spacing:-.01em; }}
  header.hero p {{ margin:0; color:var(--muted); max-width:70ch; }}
  .demo {{ background:#fff8e1; border-left:4px solid #ffc107; padding:11px 15px;
           border-radius:6px; margin:16px 0 0; font-size:13.5px; color:#5b4b12; }}
  nav.sections {{ position:sticky; top:0; z-index:5; background:rgba(251,252,253,.92);
    backdrop-filter:blur(6px); border-bottom:1px solid var(--line);
    display:flex; flex-wrap:wrap; gap:4px 18px; padding:12px 0; margin-top:16px; }}
  nav.sections a {{ color:var(--muted); text-decoration:none; font-size:13.5px;
    font-weight:500; }}
  nav.sections a:hover {{ color:var(--ink); }}
  nav.sections a.try {{ color:var(--accent); margin-left:auto; }}
  .sec {{ padding:30px 0; border-bottom:1px solid var(--line); scroll-margin-top:64px; }}
  .sec h2 {{ margin:0 0 4px; font-size:1.3rem; }}
  .sec .blurb {{ margin:0 0 16px; color:var(--muted); max-width:75ch; }}
  .graph {{ overflow-x:auto; }}
  #try {{ padding:34px 0 60px; border-bottom:none; }}
  #try form {{ display:flex; flex-direction:column; gap:12px; max-width:760px; }}
  #try textarea {{ width:100%; min-height:92px; padding:12px 14px; font:inherit;
    border:1px solid var(--line); border-radius:8px; background:#fff; resize:vertical; }}
  #try button {{ align-self:flex-start; background:var(--accent); color:#fff;
    border:0; border-radius:8px; padding:11px 20px; font:inherit; font-weight:600;
    cursor:pointer; }}
  #try button:hover {{ background:#1d4ed8; }}
  #try .note {{ color:var(--muted); font-size:13px; margin:0; max-width:70ch; }}
  footer {{ color:#8a94a1; font-size:12px; padding:22px 0 40px; }}
</style></head>
<body>
<div class="wrap">
  <header class="hero">
    <h1>RAAAL — Research Dashboard</h1>
    <p>The analytics surface behind the Agentic Investment Operating System:
       regime detection, the research-backed strategy library, and behavioural
       signals. Each section below is refreshed from the day's market history.</p>
    <div class="demo"><b>DEMO — decision support only, not investment advice.</b>
      Paper trading; no real orders are placed. Every allocation is produced by a
      registered, research-backed strategy, and any live rebalance requires human
      approval. Past simulated performance does not guarantee future results.</div>
  </header>
</div>
<nav class="sections"><div class="wrap" style="display:flex;flex-wrap:wrap;gap:4px 18px;width:100%">{nav}</div></nav>
<div class="wrap">
  {body}
  <section id="try">
    <h2>Try your own strategy</h2>
    <p class="blurb">Describe how you invest, or a rule you are considering. We
      compile it, run it over the market snapshot, and show the same comparison
      — your plan against the same contributions bought and held elsewhere.</p>
    <form action="/workspace/new" method="get">
      <textarea name="describe" placeholder="I invest $500 a month using a risk parity strategy, rebalanced quarterly."></textarea>
      <button type="submit">Evaluate my strategy</button>
      <p class="note">You will be asked to sign in when you submit — your plan is
        private to you. What you typed is carried through the sign-in and
        evaluated on the other side.</p>
    </form>
  </section>
  <footer>RAAAL Agentic Investment OS — DEMO, not investment advice. Paper
    trading only. The governed operating console (discovery → three objective
    plans → human-approved paper trades) is the product; this is its research
    surface.</footer>
</div>
{script}
</body></html>
"""


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
