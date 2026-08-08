"""A cadence the product offers must be one the engine runs.

`_flows_from` matched `monthly`, `weekly` and `biweekly` and let everything
else fall through to a single one-off contribution. The vocabulary offers eight
values and the renderer has words for all eight, so three of them —
`quarterly`, `annual` and `daily` — were presented to the user in a
confirmation menu, written back as "every quarter" and "every year", and then
executed as one payment.

    "$1,000 every year, over the past five years"  ->  $1,000 contributed

No refusal, no caveat, no coverage flag: a figure computed over a plan nobody
described, reachable by choosing an option this product had shown the user.

Two mechanisms that should have caught it did not. Coverage tracks *declared
elements*, and `cadence` was declared and — as far as anything could tell —
executed. And ~3,900 tests covered no path through this function at all.

The tests below are written against the offered vocabulary rather than against
a list typed here, so a value added to the menu and not to the executor fails
this file instead of shipping.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.workspace.routes import UnsupportedCadence, _flows_from

#: Five years of weekday sessions. The real calendar is not needed: what is
#: under test is how many periods a cadence produces over a span, not which
#: holidays a venue keeps.
SESSIONS = pd.bdate_range("2021-01-04", "2025-12-31")


class Schedule:
    def __init__(self, cadence, day_rule="first_session_of_period"):
        self.cadence = cadence
        self.day_rule = day_rule
        self.amount = 1000.0
        self.starting_capital = 0.0


def offered_cadences():
    from src.mission.vocabulary import FIELDS

    return [option.value for option in FIELDS["cadence"].options]


#: What five whole years should produce. `payroll` is absent on purpose: a pay
#: cycle is not a calendar period, and it is asserted to refuse below.
EXPECTED = {
    "annual": 5, "quarterly": 20, "monthly": 60,
    "weekly": (260, 262),
    # ~26 a year. The upper bound is loose because the grouping is
    # `(isoyear, isoweek // 2)`, and the pairing restarts at each year
    # boundary, so a year whose last ISO week is odd contributes a singleton
    # group and an extra purchase. Five years of that is 136 rather than 130.
    # A real imprecision, pre-existing and separate from the defect this file
    # was written for; recorded here rather than blessed as correct.
    "biweekly": (128, 140),
    "daily": (1250, 1350), "once": 1,
}


class TestTheMenuAndTheEngineAgree:
    def test_every_offered_cadence_is_accounted_for(self):
        """The guard that makes the rest of this file self-maintaining."""
        unaccounted = set(offered_cadences()) - set(EXPECTED) - {"payroll"}
        assert not unaccounted, (
            f"{unaccounted} is offered to users but this file says nothing "
            "about what it should do")

    @pytest.mark.parametrize("cadence", [c for c in EXPECTED])
    def test_it_produces_the_number_of_periods_it_names(self, cadence):
        flows = _flows_from(Schedule(cadence), SESSIONS)
        expected = EXPECTED[cadence]
        low, high = expected if isinstance(expected, tuple) else (expected, expected)
        assert low <= len(flows) <= high, (
            f"{cadence} produced {len(flows)} contributions over five years")

    @pytest.mark.parametrize("cadence", ["annual", "quarterly", "daily"])
    def test_it_is_not_silently_a_lump_sum(self, cadence):
        """The specific defect. Each of these returned exactly one flow."""
        assert len(_flows_from(Schedule(cadence), SESSIONS)) > 1


class TestAnUnrunnableCadenceRefuses:
    def test_payroll_refuses_rather_than_inventing_a_pay_cycle(self):
        """A pay cycle may be weekly, biweekly, semi-monthly or monthly.
        Choosing one invents the user's employer; a lump sum invented
        something further from what they said than any of the four."""
        with pytest.raises(UnsupportedCadence) as raised:
            _flows_from(Schedule("payroll"), SESSIONS)
        assert "payroll" in str(raised.value)

    def test_an_unknown_cadence_refuses_rather_than_defaulting(self):
        with pytest.raises(UnsupportedCadence):
            _flows_from(Schedule("fortnightly-ish"), SESSIONS)

    def test_the_refusal_says_what_would_work(self):
        with pytest.raises(UnsupportedCadence) as raised:
            _flows_from(Schedule("payroll"), SESSIONS)
        assert "monthly" in str(raised.value)


class TestTheDayRuleMovesTheMoney:
    """`last_session_of_period` was honoured here and unreachable from prose,
    so two different descriptions produced identical figures."""

    def test_first_and_last_land_on_different_sessions(self):
        first = _flows_from(Schedule("monthly"), SESSIONS)
        last = _flows_from(Schedule("monthly", "last_session_of_period"), SESSIONS)
        assert len(first) == len(last)
        assert [f.date for f in first] != [f.date for f in last]

    def test_the_words_reach_the_rule(self):
        """The half that was missing: the executor already worked."""
        from src.mission.compiler import parse

        def day_rule_of(text):
            return {r.field: r.value for r in parse(text).recognitions}.get(
                "contribution_day_rule")

        assert day_rule_of(
            "I invest $2000 last session every month into VT") == \
            "last_session_of_period"
        assert day_rule_of(
            "I invest $2000 first session every month into VT") == \
            "first_session_of_period"

    def test_a_rebalancing_clause_is_not_read_as_a_contribution_day(self):
        """"month end" and "quarter end" are deliberately not recognised.
        Reading a rebalancing clause as a contribution setting is the defect
        that turned one $100,000 allocation into $6,100,000 of contributions,
        and this table has no context guard to prevent it."""
        from src.mission.compiler import parse

        fields = {r.field: r.value for r in parse(
            "I hold a 60/40 stock and bond split and rebalance quarter end, "
            "over the past 5 years.").recognitions}
        assert "contribution_day_rule" not in fields
