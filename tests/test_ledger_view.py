"""The ledger every run produced, and never showed.

The engine has been a historical portfolio accounting engine for a long time:
`accounting.Fill` records what actually happened at the price that was actually
available, `PortfolioPath` carries positions and cash per session, and
`mission.ledger` reconciles the lines against the reported figure — refusing to
show a figure at all when they disagree.

None of it reached a page. A person read a number whose provenance stopped at
"the engine says so", while the evidence for it sat one frame away.

The gap this closes is narrower than it looks. The event ledger is built only
`if events is not None` — a rule firing on an observation — so a plain monthly
schedule had no ledger at all, though its purchases were recorded all along.
Showing one kind of plan its workings and not the other would teach people that
scheduled plans are less accountable, which is untrue.
"""
from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from src.workspace.ledger_view import lines, unfilled


class Fill:
    def __init__(self, date, ticker, shares, price, notional, reason=""):
        self.date = pd.Timestamp(date)
        self.ticker, self.shares, self.price = ticker, shares, price
        self.notional, self.reason = notional, reason


class Order:
    def __init__(self, date, ticker, notional, reason=""):
        self.date = pd.Timestamp(date)
        self.ticker, self.notional, self.reason = ticker, notional, reason


class Path:
    def __init__(self, fills=(), unfilled=()):
        self.fills, self.unfilled = fills, unfilled


class Result:
    def __init__(self, path):
        self.path = path


class Row:
    def __init__(self, signal, contributed, executed, subject, contribution,
                 shares, price, reason):
        self.signal_session = pd.Timestamp(signal) if signal else None
        self.contribution_session = pd.Timestamp(contributed)
        self.execution_session = pd.Timestamp(executed)
        self.subject, self.reason = subject, reason
        self.contribution = Decimal(contribution)
        self.shares, self.price = Decimal(shares), Decimal(price)


class Ledger:
    def __init__(self, rows):
        self.rows = rows


class TestAScheduledPlanShowsItsWorkings:
    """The case that had no ledger at all."""

    def run(self):
        return {"result": Result(Path(fills=(
            Fill("2016-01-05", "VTI", 4.923127, 101.46, 499.50, "stated split"),
            Fill("2016-02-02", "VTI", 5.016576, 99.57, 499.50, "stated split"),
        )))}

    def test_every_purchase_is_a_line(self):
        found = lines(self.run())
        assert len(found) == 2

    def test_a_line_carries_what_was_bought_and_at_what(self):
        first = lines(self.run())[0]
        assert first.subject == "VTI"
        assert first.executed == "2016-01-05"
        assert first.shares == "4.923127"
        assert first.price == "101.46"
        assert first.amount == "499.50"

    def test_no_signal_is_shown_as_absent_rather_than_invented(self):
        """Nothing was watched; the date came round. An empty signal column is
        the honest reading, and a fabricated one would suggest a rule."""
        assert lines(self.run())[0].signal == ""


class TestATriggeredPlanKeepsItsObservation:
    def run(self):
        return {"ledger": Ledger(rows=(
            Row("2020-03-12", "2020-03-13", "2020-03-16", "VOO",
                "1000.00", "3.221000", "310.45", "signal"),))}

    def test_the_ledger_is_preferred_over_the_fills(self):
        line = lines(self.run())[0]
        assert line.signal == "2020-03-12"

    def test_the_three_dates_stay_distinct(self):
        """With one date, a policy acting on the very session that produced its
        signal is indistinguishable from one that waited — and the look-ahead
        check passes either way."""
        line = lines(self.run())[0]
        assert line.signal != line.contributed != line.executed


class TestOrdersThatDidNotExecute:
    def test_they_are_shown_rather_than_dropped(self):
        """An order that silently vanished is the difference between what the
        plan declared and what it did, and it is invisible in a total."""
        run = {"result": Result(Path(unfilled=(
            Order("2020-03-16", "VOO", 1000.0, "insufficient cash"),)))}
        found = unfilled(run)
        assert len(found) == 1
        assert found[0]["subject"] == "VOO"
        assert "insufficient cash" in found[0]["reason"]


class TestNothingToShow:
    @pytest.mark.parametrize("run", [None, {}, {"result": None}])
    def test_a_run_with_no_result_has_no_lines(self, run):
        assert lines(run) == ()
        assert unfilled(run) == ()


class TestThePageRendersIt:
    def test_the_template_shows_the_ledger_and_says_what_it_is_for(self):
        from pathlib import Path as FilePath

        template = (FilePath(__file__).resolve().parent.parent / "src" /
                    "workspace" / "templates" / "pilot.html").read_text()
        assert "What actually happened" in template
        assert "{% for row in ledger %}" in template
        assert "derived from these lines" in template, (
            "the ledger is shown without saying that the figure comes from it")
