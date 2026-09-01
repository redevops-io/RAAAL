"""A plan that cannot run must say why, on the page, above the fold.

The defect this guards: a sentence carrying a dimension the pilot cannot honour
— the clearest case is a stated evaluation window, "over the past 5 years",
which `capability` refuses with a reason — compiled to a non-executable reading.
`execute()` returns no run for a non-executable reading (a figure beside an
unfinished request is worse than none), so the result page had `figure=None`,
`strategy_not_executed=False` and `unavailable=None`: every banner branch was
empty. The page rendered the interpreted strategy and then *nothing* — no
figure and, worse, no reason — while the parameter table still announced
"nothing is blocking a run". A user clicked Evaluate and got the same screen
back, silently. That is the exact shape of a button that looks broken.

The reason already existed on the refused row (the capability manifest's own
words). These tests assert it is now *surfaced*: a "Not evaluated" banner states
it at the top, the false "nothing is blocking a run" line is gone, and the
"Won't be used" detail is open rather than folded away — the same "refuse by
name, with a reason a person can act on" boundary `test_no_silent_refusal`
guards for the value level, here at the whole-plan level.
"""
from __future__ import annotations

import pytest

pytest.importorskip("jinja2")

WINDOWED = "invest $50 into SPY every month over the past 5 years"
PLAIN = "invest $50 into SPY every month"


def _render(text):
    """The pilot result page for `text`, read by the deterministic compiler
    reader (no model, no recording needed) and executed exactly as the route
    does — so `run` is `{}` for a non-executable reading, as in production."""
    from src.workspace import pilot, pilot_routes
    from src.workspace.routes import TEMPLATES
    from src.discovery.readers_quantify import CompilerReader

    reading = pilot.read(text, CompilerReader())
    run = pilot_routes.execute(reading)
    context = pilot_routes.page(reading, text=text, run=run)
    html = TEMPLATES.env.get_template("pilot.html").render(request=None, **context)
    return reading, run, html


class TestANonExecutablePlanSaysWhy:
    def test_the_window_makes_the_reading_non_executable_with_no_run(self):
        """The precondition. If either of these stops being true the black hole
        is gone for a different reason and this test should be revisited, not
        deleted: it is the setup the assertions below depend on."""
        reading, run, _ = _render(WINDOWED)
        assert reading.executable is False
        assert run == {}, "a non-executable reading must not produce a run"
        assert any(r.dimension == "evaluation_period" for r in reading.refusals)

    def test_the_reason_is_stated_at_the_top(self):
        """The banner, not the folded section. The manifest's own words for
        why the window is not honoured appear in a `Not evaluated` outcome
        banner, so "why nothing happened" is the first thing read."""
        _, _, html = _render(WINDOWED)
        assert "outcome-top refusal-plan" in html
        assert "Not evaluated" in html
        assert "cannot yet restrict a run to a stated window" in html
        banner = html[html.index("outcome-top refusal-plan"):]
        banner = banner[: banner.index("</div>")]
        assert "cannot yet restrict a run to a stated window" in banner

    def test_the_false_all_clear_is_gone(self):
        """The parameter table must not claim nothing blocks a run when a
        refused row does. It is the line that made the silent failure read as
        success."""
        _, _, html = _render(WINDOWED)
        assert "nothing is blocking a run" not in html
        assert "won't evaluate as written" in html.lower()

    def test_the_wont_be_used_detail_is_open(self):
        """Folded-by-default is right for a note beside a figure that ran; wrong
        when it is the reason nothing ran. The block carries the anchor the
        banner links to and is open in the non-executable case."""
        _, _, html = _render(WINDOWED)
        block = html[html.index('id="wont-run"'):]
        summary_open = block[: block.index(">") + 1]
        assert " open" in summary_open


class TestARunnablePlanIsUnchanged:
    def test_no_refusal_banner_and_the_all_clear_returns(self):
        """The reciprocal: a plan with nothing refused is executable, grows no
        `Not evaluated` banner, and the "nothing is blocking a run" line — true
        now — is back. A surfacing fix that fired on runnable plans would be a
        new defect."""
        reading, _, html = _render(PLAIN)
        assert reading.executable is True
        assert not any(r.dimension == "evaluation_period" for r in reading.refusals)
        assert "outcome-top refusal-plan" not in html
        assert "nothing is blocking a run" in html
