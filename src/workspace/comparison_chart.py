"""Your strategy against its benchmarks, drawn rather than tabulated.

The point of the product is that somebody describes a strategy and sees how it
did against the alternatives. Every run has already computed that: the plan's
own portfolio path and five benchmark paths — the same basket bought and held,
the S&P 500, the Nasdaq 100, aggregate bonds, and cash. The page printed one
final number and discarded six time series.

Design notes, because the palette is not a matter of taste:

**One axis.** Every series is a portfolio value in the same currency under the
same contribution schedule, which is what makes them comparable at all. A second
scale would let two lines cross without meaning anything.

**Fixed hue order, never cycled.** The plan is always slot 1; each benchmark
keeps its slot whatever else is present, so a run with fewer benchmarks does not
repaint the survivors into other people's colors.

**One palette for both page themes.** The workspace switches on
`prefers-color-scheme`, and a Bokeh document cannot: its colors are baked in
when the figure is built. These are the dark-surface steps, which validate at
≥3:1 against the dark surface and stay legible on the light one — checked with
the palette validator in both modes rather than judged by eye. The plot
background is transparent so the page's own surface shows through and the chart
does not sit in a pale rectangle on a dark page.

**A legend and direct labels.** Six series means identity is never carried by
colour alone, and the light-mode contrast check requires visible labels as
relief. The final value is written at the end of each line, which is also the
comparison a reader actually wants.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Slot order is fixed. The plan first, then the benchmarks in the order the
#: benchmark rule declares them — never sorted by outcome, for the same reason
#: `mission.benchmark.compare` refuses to: ordering a comparison by result turns
#: a set of facts into a claim about which one won.
SERIES_COLOURS: Sequence[str] = (
    "#3987e5",  # 1 blue — the plan
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
)

PLAN_LABEL = "Your strategy"


def _series(result) -> Optional[Tuple[list, list]]:
    """Dates and values from a `MissionResult`, or None if it has no path."""
    path = getattr(result, "path", None)
    value = getattr(path, "value", None)
    if value is None or len(value) == 0:
        return None
    return list(value.index), [float(v) for v in value.to_list()]


def collect(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The plan and every benchmark that produced a path, in declared order.

    A benchmark that could not run — no price history for its instrument over
    this period — is left out rather than drawn flat at zero. A line at zero
    reads as "this strategy earned nothing", which is a much stronger claim than
    "this comparison could not be made".
    """
    series: List[Dict[str, Any]] = []

    plan = _series(run.get("result"))
    if plan is None:
        return []
    series.append({"name": PLAN_LABEL, "dates": plan[0], "values": plan[1]})

    for benchmark in run.get("benchmarks") or ():
        found = _series(getattr(benchmark, "result", None))
        if found is None:
            continue
        series.append({"name": getattr(benchmark, "name", "benchmark"),
                       "dates": found[0], "values": found[1]})
    return series


def build(run: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """`{"script": ..., "div": ...}` for embedding, or None when there is
    nothing to draw.

    Returns None rather than an empty figure. A chart with no lines is a chart
    that says the run produced nothing, and the page has better words for that
    than an empty pair of axes.
    """
    series = collect(run)
    if len(series) < 2:
        # One line is not a comparison, and the figure beside it would imply
        # the benchmarks had been run and lost.
        return None

    from bokeh.embed import components
    from bokeh.models import HoverTool, Label, NumeralTickFormatter
    from bokeh.plotting import figure

    plot = figure(
        height=380, sizing_mode="stretch_width", x_axis_type="datetime",
        toolbar_location=None,
        title="Your strategy against the same contributions elsewhere",
    )

    for slot, line in enumerate(series):
        colour = SERIES_COLOURS[slot % len(SERIES_COLOURS)]
        renderer = plot.line(
            line["dates"], line["values"], line_width=2, color=colour,
            legend_label=line["name"],
            # The plan is the subject; the benchmarks are context.
            line_alpha=1.0 if slot == 0 else 0.85,
        )
        if slot == 0:
            plot.add_tools(HoverTool(
                renderers=[renderer], mode="vline",
                tooltips=[("", "@y{$0,0}"), ("on", "@x{%F}")],
                formatters={"@x": "datetime"}))

        # Direct label at the end of the line. Required relief for the
        # light-surface contrast check, and the number a reader came for.
        plot.add_layout(Label(
            x=line["dates"][-1], y=line["values"][-1], text=f" {line['values'][-1]:,.0f}",
            text_font_size="10px", text_color=colour, x_units="data",
            y_units="data"))

    plot.yaxis.formatter = NumeralTickFormatter(format="$0,0")
    plot.yaxis.axis_label = "Portfolio value"

    # Recessive frame; the page's surface shows through.
    plot.background_fill_alpha = 0
    plot.border_fill_alpha = 0
    plot.outline_line_color = None
    plot.xgrid.grid_line_alpha = 0.15
    plot.ygrid.grid_line_alpha = 0.15
    plot.legend.location = "top_left"
    plot.legend.background_fill_alpha = 0.0
    plot.legend.border_line_alpha = 0.0
    plot.legend.label_text_font_size = "11px"

    return _stable(*components(plot), series)


def _stable(script: str, div: str, series) -> Dict[str, str]:
    """Rewrite Bokeh's generated ids so the same plan renders the same markup.

    Bokeh mints a UUID for the container and sequential `pNNNN` ids for every
    model, both fresh on each call. Two reopens of one plan therefore produced
    different HTML, and `test_describe_clarify_save_figure_reopen` refused it —
    the page promises that reopening recompiles from the confirmed intent and
    shows the same figure, and markup that differs per render is a weaker
    promise wearing the same words.

    The ids are opaque tokens: nothing outside this pair of strings refers to
    them, so renaming every occurrence consistently preserves the document and
    removes the only thing in it that was not a function of the data. The new
    names are derived from the series themselves, so two runs of the same plan
    agree and two different plans do not collide.
    """
    import re
    from hashlib import sha256

    seed = sha256(repr([(s["name"], s["values"][:4], s["values"][-4:])
                        for s in series]).encode()).hexdigest()[:12]

    combined = script + div
    # First appearance order, so the mapping is itself deterministic.
    seen: List[str] = []
    for token in re.findall(r"\bp\d{4,}\b|\b[0-9a-f]{8}-[0-9a-f]{4}-"
                            r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                            combined):
        if token not in seen:
            seen.append(token)

    for index, token in enumerate(seen):
        replacement = f"q{seed}{index:04d}"
        script = script.replace(token, replacement)
        div = div.replace(token, replacement)
    return {"script": script, "div": div}
