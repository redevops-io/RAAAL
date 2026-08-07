"""Cadence and evaluation window must not consume each other's text.

    I put $500 into VTI every month for the past five years.

read as a single ROLLING window whose observed phrase was "every month for the
past" — a kind this build refuses — so an entirely ordinary sentence
dead-ended with "this version cannot replay 'every month for the past'". The
same plan written "monthly for the past 5 years" worked, because that phrasing
has no "every" for the rolling pattern to seize.

`_ROLLING` was tested before `_TRAILING` deliberately, on the reasoning that
"every month over the past five years" contains "the past five years" and is
not one window. That is true of a request to *measure* many windows and false
of a plan that *contributes* on a cadence. One pattern was doing the work of
two dimensions.

**Precedence alone would not fix it.** A trailing phrase appears inside
genuine rolling requests too, so preferring trailing whenever one is present
deletes the rolling capability rather than disambiguating it. What separates
them is what repeats: a contribution, or the measurement. The last test here
exists so the fix cannot be "remove ROLLING".

Both facts are asserted on every case. A window that parses correctly while
the cadence is lost would be the same defect facing the other way.
"""
from __future__ import annotations

import pytest

from src.mission import time_window
from src.mission.compiler import parse

#: The exact sentences, not fragments. The collision happens between two
#: phrases, so a test on either one alone cannot see it.
BOTH_DIMENSIONS = (
    ("I put $500 into VTI every month for the past five years.",
     "monthly", 5),
    ("I put $500 into VTI monthly for the past 5 years.",
     "monthly", 5),
    ("I put $500 into VTI every week over the past two years.",
     "weekly", 2),
    ("I add $250 to VOO every quarter for the past three years.",
     "quarterly", 3),
)


class TestBothSurvive:
    @pytest.mark.parametrize("sentence,cadence,years", BOTH_DIMENSIONS)
    def test_the_window_is_trailing_and_supported(self, sentence, cadence,
                                                  years):
        window = time_window.detect(sentence)
        assert window is not None, sentence
        assert window.kind is time_window.WindowKind.TRAILING, window.observed
        assert window.supported, (
            f"{window.observed!r} was typed {window.kind.value} and refused; "
            f"this is the dead end")
        assert window.years == years

    @pytest.mark.parametrize("sentence,cadence,years", BOTH_DIMENSIONS)
    def test_the_cadence_is_still_read(self, sentence, cadence, years):
        """The other half. Rescuing the window by swallowing the cadence would
        be the same defect pointing the other way."""
        found = parse(sentence).value_of("cadence")
        assert found is not None and found.value == cadence, sentence

    @pytest.mark.parametrize("sentence,cadence,years", BOTH_DIMENSIONS)
    def test_the_window_phrase_excludes_the_cadence(self, sentence, cadence,
                                                    years):
        """The observed phrase is quoted back to the user as what was
        recognised. "every month for the past" is not a period anyone wrote."""
        observed = time_window.detect(sentence).observed.lower()
        assert "every" not in observed and "each" not in observed, observed


class TestASingleContributionStillReadsItsWindow:
    def test_once_for_the_past_five_years(self):
        sentence = "I put $5,000 into VTI once for the past five years."
        window = time_window.detect(sentence)
        assert window.kind is time_window.WindowKind.TRAILING
        assert window.supported and window.years == 5


class TestRollingIsNotDeleted:
    """The capability the ordering existed to protect. A fix that made every
    sentence trailing would satisfy everything above."""

    ROLLING = "Show me the return of every month over the past five years."

    def test_a_measurement_request_is_still_rolling(self):
        window = time_window.detect(self.ROLLING)
        assert window.kind is time_window.WindowKind.ROLLING

    def test_and_is_still_refused_rather_than_coerced(self):
        """Recognised, typed and declined — reading it as a trailing window
        would answer a different question with a plausible number."""
        assert not time_window.detect(self.ROLLING).supported

    def test_the_difference_is_the_money_not_the_words(self):
        """Same temporal phrasing, opposite readings, and the only difference
        is whether an amount is being contributed."""
        measured = time_window.detect(
            "Show me the return of every month over the past five years.")
        contributed = time_window.detect(
            "I invest $500 every month over the past five years.")
        assert measured.kind is time_window.WindowKind.ROLLING
        assert contributed.kind is time_window.WindowKind.TRAILING


class TestItReachesThePlan:
    """Detection is not the claim; the plan running over the stated period is."""

    @pytest.fixture
    def deployment(self, monkeypatch):
        from src.deploy.context import bind, resolve, unbind

        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
        try:
            yield
        finally:
            unbind()

    def test_the_ordinary_monthly_plan_produces_a_figure(self, deployment):
        """The dead end, at the level a user meets it."""
        import src.workspace.routes as routes
        from src.workspace.draft import compile_draft

        sentence = "I put $500 into VTI every month for the past five years."
        scenario = compile_draft(sentence, name="p",
                                 context="cadence test").scenario
        run = routes._run(scenario, routes._market_data("cadence test"),
                          stated_text=sentence)
        assert run["result"] is not None, run.get("unavailable")

    def test_it_runs_the_stated_period_rather_than_the_snapshot(self,
                                                               deployment):
        import src.workspace.routes as routes
        from src.workspace.draft import compile_draft

        stated = "I put $500 into VTI every month for the past five years."
        whole = "I put $500 into VTI every month."
        runs = []
        for sentence in (stated, whole):
            scenario = compile_draft(sentence, name="p",
                                     context="cadence test").scenario
            runs.append(routes._run(scenario,
                                    routes._market_data("cadence test"),
                                    stated_text=sentence))
        assert all(one["result"] is not None for one in runs)
        assert len(runs[0]["result"].time_weighted) < \
            len(runs[1]["result"].time_weighted), (
            "the stated five years did not narrow anything")
