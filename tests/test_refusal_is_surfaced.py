"""A plan that cannot run must say why; a stated trailing period must run.

Two boundaries, one flow:

* **Part A — no silent refusal.** A sentence carrying a dimension the pilot
  cannot honour compiles to a non-executable reading, and `execute()` returns
  no run for one (a figure beside an unfinished request is worse than none). So
  the result page had `figure=None`, `strategy_not_executed=False` and
  `unavailable=None`: every banner branch empty. The page rendered the strategy
  and then *nothing* — no figure and no reason — while the table still said
  "nothing is blocking a run". These assert the reason (the manifest's own
  words) is surfaced: a "Not evaluated" banner, the false all-clear gone, the
  "Won't be used" detail open. The still-refused case is an *open-ended* window
  ("since 2021") — a kind the engine cannot resolve.

* **Part B — a trailing period runs.** "over the past 5 years" / "for 5 years"
  is a window the engine restricts the replay to (`time_window.resolve`), so it
  is executable, carries the window on the compiled scenario, grows no refusal
  banner, and — once a figure exists — is captioned with the period it covers.
  This is the case Part A's example used to be, before the wire that made it run.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jinja2")

RUNS = "invest $50 into SPY every month over the past 5 years"
RUNS_BARE = "invest $50 into SPY every month for 5 years"
REFUSED = "invest $50 into SPY every month since 2021"
PLAIN = "invest $50 into SPY every month"


def _read(text):
    from src.workspace import pilot
    from src.discovery.readers_quantify import CompilerReader

    return pilot.read(text, CompilerReader())


def _render(text):
    """The pilot result page for `text`, read by the deterministic compiler
    reader (no model, no recording needed) and executed exactly as the route
    does — so `run` is `{}` for a non-executable reading, as in production."""
    from src.workspace import pilot_routes
    from src.workspace.routes import TEMPLATES

    reading = _read(text)
    run = pilot_routes.execute(reading)
    context = pilot_routes.page(reading, text=text, run=run)
    html = TEMPLATES.env.get_template("pilot.html").render(request=None, **context)
    return reading, run, html


def _window(reading):
    scenario = getattr(reading.compiled, "scenario", None)
    provenance = getattr(scenario, "provenance", None)
    return getattr(provenance, "time_window", None)


class TestANonExecutablePlanSaysWhy:
    def test_an_unresolvable_window_is_non_executable_with_no_run(self):
        """The precondition. An open-ended window ("since 2021") is a kind this
        build cannot resolve, so it refuses — non-executable, no run."""
        reading, run, _ = _render(REFUSED)
        assert reading.executable is False
        assert run == {}, "a non-executable reading must not produce a run"
        assert any(r.dimension == "evaluation_period" for r in reading.refusals)

    def test_the_reason_is_stated_at_the_top(self):
        """The banner, not the folded section. The manifest's own words for why
        the window is not honoured appear in a `Not evaluated` outcome banner."""
        _, _, html = _render(REFUSED)
        assert "outcome-top refusal-plan" in html
        assert "Not evaluated" in html
        assert "cannot yet restrict a run to" in html
        banner = html[html.index("outcome-top refusal-plan"):]
        banner = banner[: banner.index("</div>")]
        assert "cannot yet restrict a run to" in banner

    def test_the_false_all_clear_is_gone(self):
        """The parameter table must not claim nothing blocks a run when a
        refused row does."""
        _, _, html = _render(REFUSED)
        assert "nothing is blocking a run" not in html
        assert "won't evaluate as written" in html.lower()

    def test_the_wont_be_used_detail_is_open(self):
        """Folded-by-default is right for a note beside a figure that ran; wrong
        when it is the reason nothing ran."""
        _, _, html = _render(REFUSED)
        block = html[html.index('id="wont-run"'):]
        summary_open = block[: block.index(">") + 1]
        assert " open" in summary_open


class TestATrailingWindowRuns:
    @pytest.mark.parametrize("text", [RUNS, RUNS_BARE])
    def test_a_trailing_period_is_executable_and_carries_the_window(self, text):
        """The wire. "the past 5 years" and the bare "for 5 years" both compile
        to an executable plan whose scenario carries the trailing window the
        engine restricts the replay to — where the same sentence used to be
        refused as a stranded, unusable dimension."""
        reading = _read(text)
        assert reading.executable is True
        assert not any(r.dimension == "evaluation_period" for r in reading.refusals)
        window = _window(reading)
        assert window is not None and window.supported
        assert window.years == 5

    def test_a_runnable_window_grows_no_refusal_banner(self):
        """The reciprocal of Part A: a plan whose window runs is not refused, so
        it grows no `Not evaluated` banner and never claims a value blocks it."""
        _, _, html = _render(RUNS)
        assert "outcome-top refusal-plan" not in html
        assert "won't evaluate as written" not in html.lower()

    def test_the_period_caption_is_rendered_when_a_windowed_figure_exists(self):
        """Presentation. `_period` turns the resolved window into the caption the
        template shows under the figure; asserted at the unit that builds it so
        the check does not need a priced snapshot to reach a figure."""
        from src.workspace.pilot_routes import _period

        class _W:
            label = "the past 5 years"

        class _Resolved:
            window = _W()
            short = False
            start = "2021-09-01"
            end = "2026-09-01"

        got = _period({"resolved_window": _Resolved()})
        assert got is not None and got["label"] == "the past 5 years"
        assert got["short"] is False
        assert _period({}) is None
        assert _period({"resolved_window": None}) is None


class TestTheEngineSlicesToTheWindow:
    def test_the_replay_is_restricted_to_the_trailing_window(self):
        """The point of the whole change: the engine evaluates over the stated
        period, not the whole history. A ~6.7-year synthetic frame, a plan that
        names "the past 5 years", and the resolved window spans five years back
        from the last session — not the frame's start. A plain plan over the
        same frame declares no window at all."""
        import numpy as np
        import pandas as pd

        from src.evaluation.core import evaluate_plan

        dates = pd.date_range("2015-01-01", periods=1700, freq="B")
        rng = np.random.default_rng(3)
        prices = pd.DataFrame(
            {"SPY": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, len(dates))))},
            index=dates)

        windowed = evaluate_plan(_read(RUNS).compiled.scenario, prices)
        resolved = windowed.resolved_window
        assert resolved is not None, "a trailing window must resolve"
        span_years = (resolved.end - resolved.start).days / 365.25
        assert 4.7 <= span_years <= 5.2, span_years
        assert resolved.start > dates[0].date(), (
            "the window must start after the frame's first session, or nothing "
            "was restricted")

        plain = evaluate_plan(_read(PLAIN).compiled.scenario, prices)
        assert plain.resolved_window is None, (
            "a plan naming no period declares no window")


class TestAPlainPlanIsUnchanged:
    def test_no_window_no_banner_no_caption(self):
        """A plan naming no period is executable, carries no window, grows no
        refusal banner, and keeps the (now-truthful) all-clear."""
        reading, _, html = _render(PLAIN)
        assert reading.executable is True
        assert _window(reading) is None
        assert "outcome-top refusal-plan" not in html
        assert "nothing is blocking a run" in html
