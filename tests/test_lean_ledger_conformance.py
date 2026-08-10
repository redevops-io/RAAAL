"""The Python engine and the Lean model must agree about one ledger.

Everything in `Ledger.lean` is true by definition of `endingCash`, and that is
precisely why it is not enough on its own. A term omitted from that definition —
a fee the engine charges, a dividend it credits — leaves both conservation
theorems standing while the model describes a different ledger from the one
Quantify runs. A beautiful proof about the wrong semantics is worse than none,
because it is quotable.

So this file computes the same fixtures in Python, from the same inputs, and
asserts the same closing balances the Lean `#guard`s assert. It does not run
Lean: the toolchain is a separate CI lane, and a unit suite that needed one
would be a suite nobody runs. What it does is make the two sides fail together
if either moves.

**Exact arithmetic on both sides.** Minor units and micro-shares, integers
throughout, so a disagreement is a disagreement about semantics rather than
about float rounding — which is the one thing this comparison must not be
sensitive to.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

LEAN = (Path(__file__).resolve().parent.parent / "formal" / "Quantify"
        / "Fixtures.lean")

SHARE = 1_000_000     # micro-units per share, mirrors `Quantify.sharesScale`


def ending_cash(*, opening, contributions, withdrawals, purchases, sales,
                fees):
    """The Python statement of the same rule Lean states.

    Written out rather than imported from the engine on purpose: importing the
    engine's own function would compare it against itself, and the question is
    whether two independent statements of the ledger agree.
    """
    return (opening + sum(contributions) - sum(withdrawals)
            - sum(purchases) + sum(sales) - fees)


def ending_shares(*, opening, bought, sold):
    return opening + sum(bought) - sum(sold)


class TestTheTwoSidesAgree:
    def test_one_buy(self):
        """$100 in, five shares at $20, nothing left over."""
        assert ending_cash(opening=0, contributions=[10_000], withdrawals=[],
                           purchases=[10_000], sales=[], fees=0) == 0
        assert ending_shares(opening=0, bought=[5 * SHARE], sold=[]) == 5 * SHARE

    def test_five_annual_contributions(self):
        """The shape of the defect that started the capability manifest.

        The product reported "$1,000 every year over five years" as $1,000
        contributed — one payment, no refusal, no coverage flag. A ledger that
        dropped four payments fails here on the total rather than on a figure
        nobody checked.
        """
        assert ending_cash(opening=0, contributions=[100_000] * 5,
                           withdrawals=[], purchases=[], sales=[],
                           fees=0) == 500_000

    def test_round_trip_with_a_fee(self):
        assert ending_cash(opening=50_000, contributions=[10_000],
                           withdrawals=[2_000], purchases=[6_000],
                           sales=[2_500], fees=100) == 54_400
        assert ending_shares(opening=0, bought=[3 * SHARE],
                             sold=[1 * SHARE]) == 2 * SHARE

    def test_overselling_is_representable(self):
        """The model states what a ledger means, not what Quantify permits.

        A representation that made this impossible would also make it
        impossible to state the theorem that it never happens.
        """
        assert ending_shares(opening=0, bought=[1 * SHARE],
                             sold=[3 * SHARE]) == -2 * SHARE


@pytest.fixture(scope="module")
def guards():
    """Every `#guard` in the Lean fixtures, as (fixture, field, asset, value).

    Module-scoped and free-standing. A class-scoped fixture written as an
    instance method is deprecated — pytest builds a new instance per test while
    running the fixture once, so anything it set on `self` would be invisible.
    """
    if not LEAN.exists():
        pytest.skip("formal/Quantify/Fixtures.lean is absent")
    return re.findall(r"#guard\s+(\S+)\.(\w+)(?:\s+\"(\w+)\")?\s*==\s*"
                      r"(-?\d+)", LEAN.read_text())


class TestThePythonSideMatchesTheLeanFile:
    """Read from the Lean source, so the two cannot drift apart silently.

    The failure this catches: somebody edits a fixture on one side. Both suites
    still pass, and the conformance claim quietly becomes two unrelated sets of
    numbers that happen to be green.
    """

    def test_the_lean_file_states_the_same_numbers(self, guards):
        expected = {
            ("oneBuy", "endingCash", None): 0,
            ("oneBuy", "endingShares", "VTI"): 5 * SHARE,
            ("oneBuy", "endingShares", "BND"): 0,
            ("fiveAnnual", "contributed", None): 500_000,
            ("fiveAnnual", "endingCash", None): 500_000,
            ("roundTrip", "endingCash", None): 54_400,
            ("roundTrip", "endingShares", "VTI"): 2 * SHARE,
            ("oversold", "endingShares", "VTI"): -2 * SHARE,
            # The cadence fixture's own shape. Included rather than filtered
            # out: a scan narrowed to the ledger names would stop noticing
            # anything added beside them, which is the drift this test exists
            # to catch.
            ("fiveYearsOfMonths", "length", None): 60,
        }
        found = {(fixture, field, asset or None): int(value)
                 for fixture, field, asset, value in guards}
        assert found == expected, (
            "the Lean fixtures and this file disagree; one of them was edited "
            "alone, and the conformance claim is only worth something while "
            "they are the same numbers")

    def test_every_lean_guard_is_covered_here(self, guards):
        """A guard added on the Lean side and not here would be proved and
        never conformance-checked."""
        assert len(guards) == 9


class TestTheScaleIsShared:
    def test_the_lean_file_declares_the_same_share_scale(self):
        types = LEAN.parent / "Types.lean"
        if not types.exists():
            pytest.skip("formal/Quantify/Types.lean is absent")
        found = re.search(r"def sharesScale : Int := (\d+)", types.read_text())
        assert found, "Types.lean no longer declares sharesScale"
        assert int(found.group(1)) == SHARE, (
            "the scales differ, so every quantity in this comparison means a "
            "different number on each side")


class TestCadenceAgrees:
    """The historical defect, computed on both sides.

    The shipped build reported "$1,000 every year over five years" as $1,000
    contributed. The Lean side proves `N x A` for every schedule and guards
    that five calendar years of month-ends give N = 5; this computes the same
    N from the same sessions, independently, and asserts the same total.

    Independent means independent: the buckets are recomputed here rather than
    imported, so a change to Quantify's own bucketing does not move both sides
    together.
    """

    SESSIONS = [(2020 + y, m + 1, 28) for y in range(5) for m in range(12)]

    @staticmethod
    def _count(sessions, key):
        seen, kept = set(), 0
        for session in sessions:
            k = key(session)
            if k not in seen:
                seen.add(k)
                kept += 1
        return kept

    def test_there_are_sixty_sessions(self):
        assert len(self.SESSIONS) == 60

    def test_annual_is_five_contributions_not_one(self):
        """The defect exactly. One contribution here was the shipped
        behaviour, and no test covered the path that produced it."""
        count = self._count(self.SESSIONS, lambda s: s[0])
        assert count == 5
        assert count * 100_000 == 500_000

    def test_monthly_is_sixty(self):
        count = self._count(self.SESSIONS, lambda s: s[0] * 12 + s[1])
        assert count == 60
        assert count * 100_000 == 6_000_000

    def test_once_is_one(self):
        count = self._count(self.SESSIONS, lambda _: 0)
        assert count == 1
        assert count * 100_000 == 100_000

    def test_the_lean_file_states_the_same_cadence_numbers(self):
        import re as _re

        if not LEAN.exists():
            pytest.skip("formal/Quantify/Fixtures.lean is absent")
        text = LEAN.read_text()
        for cadence, count, total in (("annual", 5, 500_000),
                                      ("monthly", 60, 6_000_000),
                                      ("once", 1, 100_000)):
            assert _re.search(
                rf"#guard contributionCount Cadence\.{cadence} "
                rf"fiveYearsOfMonths == {count}\b", text), cadence
            assert _re.search(
                rf"#guard totalContributed Cadence\.{cadence} "
                rf"fiveYearsOfMonths 100000 == {total}\b", text), cadence


class TestTriggerSemanticsAgree:
    """Crossing against persistent, computed independently on this side.

    The second money-moving defect: a condition written as a crossing executed
    as a persistent state, so a portfolio that should have bought once bought
    on every session the condition held.

    The point is not that the two predicates differ. It is the ratio — one
    crossing under three persistent sessions is the factor by which the defect
    overspent, and a test asserting only inequality would pass for a build that
    was wrong by ten times.
    """

    @staticmethod
    def _counts(series):
        """(crossings, persistent) for a list of (value, threshold)."""
        below = [v < t for v, t in series]
        persistent = sum(below)
        crossings = sum(1 for i in range(1, len(below))
                        if below[i] and not below[i - 1])
        return crossings, persistent

    def test_one_dip_is_one_crossing_and_three_sessions(self):
        series = [(110, 100), (95, 100), (90, 100), (92, 100), (105, 100)]
        assert self._counts(series) == (1, 3)

    def test_a_longer_dip_is_still_one_crossing(self):
        """Duration does not multiply signals. The count belongs to the
        transition, not to how long the state lasts."""
        series = [(110, 100), (95, 100), (90, 100), (92, 100), (91, 100),
                  (93, 100)]
        assert self._counts(series) == (1, 5)

    def test_re_entry_signals_again(self):
        """The converse guard: a definition that only ever fired once would
        satisfy every other case here."""
        series = [(110, 100), (95, 100), (105, 100), (90, 100), (92, 100)]
        assert self._counts(series) == (2, 3)

    def test_opening_below_is_not_a_crossing(self):
        """A crossing is a change, and the first session has nothing to have
        changed from."""
        series = [(90, 100), (92, 100)]
        assert self._counts(series) == (0, 2)

    def test_the_lean_file_states_the_same_series_and_counts(self):
        import re as _re

        path = LEAN.parent / "Triggers.lean"
        if not path.exists():
            pytest.skip("formal/Quantify/Triggers.lean is absent")
        text = path.read_text()

        for name, crossings, persistent in (("oneDip", 1, 3),
                                            ("longDip", 1, 5),
                                            ("twoDips", 2, 3),
                                            ("opensBelow", 0, 2)):
            assert _re.search(
                rf"crossingCount {name} = {crossings}\b", text), name
            assert _re.search(
                rf"persistentCount {name} = {persistent}\b", text), name

        # The series themselves, so the counts are not agreeing about
        # different numbers.
        assert "⟨110, 100⟩, ⟨95, 100⟩, ⟨90, 100⟩, ⟨92, 100⟩, ⟨105, 100⟩" in text
        assert "⟨90, 100⟩, ⟨92, 100⟩" in text


class TestOrderingAndAssociationAgree:
    """When an event may execute, and which fill belongs to it.

    The third claim is the one a date inequality cannot make. A schedule can
    satisfy every ordering rule and still hand A's fill to B — the dense-event
    defect, where the totals reconcile because the same fills were used, once
    each, in the wrong places.
    """

    # (id, signal, funding, execution)
    ADJACENT = [(1, 1, 2, 3), (2, 2, 3, 4)]
    DELAYED = [(1, 1, 2, 5), (2, 2, 3, 4)]

    @staticmethod
    def _causal(e):
        _, signal, funding, execution = e
        return signal <= funding <= execution

    @staticmethod
    def _no_look_ahead(e):
        """Close-derived signal under next-open policy: strictly later.

        Strict, not `<=`. A signal derived from a session's close is not
        knowable until that close is established, so filling at the same close
        needs a price the decision helped determine.
        """
        _, signal, _, execution = e
        return signal < execution

    def test_adjacent_events_are_sound(self):
        assert all(self._causal(e) for e in self.ADJACENT)
        assert all(self._no_look_ahead(e) for e in self.ADJACENT)

    def test_delayed_events_are_also_sound(self):
        """Both hold, which is why ordering alone settles nothing about
        association."""
        assert all(self._causal(e) for e in self.DELAYED)
        assert all(self._no_look_ahead(e) for e in self.DELAYED)

    def test_identity_pairing_is_right_on_the_delayed_case(self):
        fills = {1: 5, 2: 4}            # event id -> fill session
        for event_id, _, _, execution in self.DELAYED:
            assert fills[event_id] == execution

    def test_position_pairing_steals_the_other_events_fill(self):
        """The defect, reproduced. Fills in session order, events in list
        order, zipped — each event takes the other's fill."""
        by_session = sorted([(1, 5), (2, 4)], key=lambda f: f[1])
        paired = {e[0]: f for e, f in zip(self.DELAYED, by_session)}
        assert paired[1] == (2, 4), "A should have been handed B's fill"
        assert paired[2] == (1, 5), "B should have been handed A's fill"
        assert paired[1][0] != 1 and paired[2][0] != 2

    def test_the_lean_file_states_the_same_events_and_pairings(self):
        import re as _re

        path = LEAN.parent / "Ordering.lean"
        if not path.exists():
            pytest.skip("formal/Quantify/Ordering.lean is absent")
        text = path.read_text()

        assert "def delayedA : Event := ⟨1, 1, 2, 5⟩" in text
        assert "def delayedB : Event := ⟨2, 2, 3, 4⟩" in text
        assert "def adjacentA : Event := ⟨1, 1, 2, 3⟩" in text
        assert "def adjacentB : Event := ⟨2, 2, 3, 4⟩" in text
        # Identity right, position wrong, on the same schedule.
        assert _re.search(r"fillFor delayedFills delayedA = some ⟨1, 5⟩", text)
        assert _re.search(
            r"fillByPosition delayedFills delayedEvents delayedA = some ⟨2, 4⟩",
            text)


class TestEvaluationWindowsAgree:
    """Data may exist before the evaluation period. Economic events may not.

    The "three months returned ten years" defect stated precisely. It was never
    that the engine loaded too much history — it has to, or the indicator has
    nothing to average. It was that money moving during the warm-up was counted
    in a result the person asked to be about three months.
    """

    WARM_UP, FIRST, LAST = 3, 3, 5
    EVERY_SESSION = [0, 1, 2, 3, 4, 5]

    def _in_frame(self, s):
        return (self.FIRST - self.WARM_UP) <= s <= self.LAST

    def _in_reported(self, s):
        return self.FIRST <= s <= self.LAST

    def test_the_frame_is_wider_than_the_report(self):
        assert [s for s in self.EVERY_SESSION if self._in_frame(s)] == \
            self.EVERY_SESSION
        assert [s for s in self.EVERY_SESSION if self._in_reported(s)] == \
            [3, 4, 5]

    def test_warm_up_is_loaded_and_silent(self):
        """Both halves. Either alone permits the defect: a warm-up outside the
        frame cannot feed an indicator, and one inside the report is money
        counted twice."""
        for session in (0, 1, 2):
            assert self._in_frame(session)
            assert not self._in_reported(session)

    def test_only_the_requested_period_is_reportable(self):
        reportable = [s for s in self.EVERY_SESSION if self._in_reported(s)]
        assert len(reportable) == 3, (
            "six sessions of activity, three reportable — the defect reported "
            "all six and called it three months")

    def test_the_warm_up_actually_changes_the_indicator(self):
        """Without this the whole fixture is vacuous: a window whose early
        sessions did not matter would satisfy every boundary check while
        proving nothing, because nothing was there to leak."""
        frame = [100, 100, 100, 400, 400, 400]
        reported = [400, 400, 400]
        assert sum(frame) // len(frame) == 250
        assert sum(reported) // len(reported) == 400

    def test_the_lean_file_states_the_same_window_and_counts(self):
        path = LEAN.parent / "Window.lean"
        if not path.exists():
            pytest.skip("formal/Quantify/Window.lean is absent")
        text = path.read_text()

        assert "def threeMonths : Window := ⟨3, 3, 5⟩" in text
        assert "threeMonths.reportable everySession == [3, 4, 5]" in text
        assert "meanOver frameValues = 250" in text
        assert "meanOver reportedValues = 400" in text


class TestNoProofIsAdmitted:
    """`sorry` compiles. It emits a warning and `lake build` still exits zero.

    That is the formal-layer version of the defect this project keeps finding:
    a check that passes without checking. A theorem closed with `sorry` reads
    exactly like a proven one in the source, in the build output, and in any
    summary written from either.

    The moving-average slice was drafted with two of them, so this is not a
    hypothetical guard against a habit nobody has.
    """

    FORMAL = LEAN.parent

    def test_nothing_in_the_formal_tree_is_admitted(self):
        if not self.FORMAL.exists():
            pytest.skip("formal/Quantify is absent")

        offenders = []
        for path in sorted(self.FORMAL.glob("*.lean")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("--") or stripped.startswith("/-"):
                    continue
                if re.search(r"\bsorry\b", line):
                    offenders.append(f"{path.name}:{number}")
        assert not offenders, (
            f"{offenders} close a proof with `sorry`, which compiles, warns, "
            "and exits zero — a theorem that is not proven and reads as though "
            "it were")

    def test_and_no_axiom_was_added_to_get_there(self):
        """The other way to admit something. An `axiom` is a proof obligation
        moved rather than discharged, and this project's rule is that a claim
        nobody exercises is a comment."""
        if not self.FORMAL.exists():
            pytest.skip("formal/Quantify is absent")

        offenders = []
        for path in sorted(self.FORMAL.glob("*.lean")):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if re.match(r"\s*axiom\s", line):
                    offenders.append(f"{path.name}:{number}")
        assert not offenders, f"{offenders} declare an axiom"


class TestMovingAverageAgrees:
    """The threshold series, computed independently on this side.

    Kept apart from trigger semantics on both sides: this says the number is
    right, `TestTriggerSemanticsAgree` says what crossing it means.
    """

    SERIES = [100, 100, 100, 100, 100, 100, 100, 400, 400, 400]

    @classmethod
    def _ma(cls, n, t, series=None):
        series = cls.SERIES if series is None else series
        if n == 0 or t + 1 < n:
            return None
        window = series[t + 1 - n:t + 1]
        return sum(window) // n if len(window) == n else None

    def test_the_window_ends_at_the_session_asked_for(self):
        assert self._ma(3, 9) == 400
        assert self._ma(5, 9) == 280

    def test_nothing_before_the_warm_up_completes(self):
        assert self._ma(5, 3) is None
        assert self._ma(5, 4) == 100

    def test_window_length_changes_the_threshold(self):
        assert self._ma(3, 9) != self._ma(5, 9)

    def test_off_by_one_is_a_different_statistic_where_it_shows(self):
        assert self._ma(3, 8) == 300
        assert self._ma(2, 8) == 400
        assert self._ma(4, 8) == 250

    def test_but_an_off_by_one_hides_on_a_flat_stretch(self):
        """Why a 200-day average over 199 observations can run for months
        looking right: on quiet stretches it agrees with itself, and diverges
        only where the boundary observation differs — which is exactly where a
        crossing happens."""
        assert self._ma(3, 9) == self._ma(2, 9)

    def test_a_price_outside_the_window_cannot_move_it(self):
        outside = [999] + self.SERIES[1:]
        assert self._ma(3, 9, outside) == self._ma(3, 9)

    def test_and_a_price_inside_it_does(self):
        inside = self.SERIES[:9] + [700]
        assert self._ma(3, 9, inside) != self._ma(3, 9)

    def test_a_flat_series_averages_to_the_constant(self):
        assert self._ma(4, 9, [250] * 10) == 250

    def test_the_lean_file_states_the_same_series_and_values(self):
        path = LEAN.parent / "MovingAverage.lean"
        if not path.exists():
            pytest.skip("formal/Quantify/MovingAverage.lean is absent")
        text = path.read_text()

        assert "[100, 100, 100, 100, 100, 100, 100, 400, 400, 400]" in text
        assert "movingAverage 3 series 9 == some 400" in text
        assert "movingAverage 5 series 9 == some 280" in text
        assert "movingAverage 3 series 8 = some 300" in text
        assert "movingAverage 5 series 3 == none" in text


class TestTheOperatorChainAgrees:
    """prices → threshold → crossing → contribution → next-open fill → ledger.

    The wiring, not the operators. Each is proven separately on both sides;
    what this asserts is that they connect in the same order — that the
    threshold feeding the trigger is the one the average produced, that the
    signal feeding the contribution is a transition and not a state, that the
    fill is matched by identity, and that the reporting boundary is applied to
    events rather than to data.

    Python computes the chain from the prices with its own arithmetic. Lean
    consumes the same prices and computes its own. Agreement on the
    intermediates as well as the final ledger is what makes this evidence about
    wiring rather than about one headline number.
    """

    PRICES = [100, 100, 100, 90, 100, 100, 100, 100, 85, 88, 100]
    WINDOW = 3
    REPORTED_FIRST, REPORTED_LAST = 5, 10

    def _ma(self, t):
        if t + 1 < self.WINDOW:
            return None
        return sum(self.PRICES[t + 1 - self.WINDOW:t + 1]) // self.WINDOW

    def _below(self, t):
        threshold = self._ma(t)
        return threshold is not None and self.PRICES[t] < threshold

    def _first_usable(self):
        return self.WINDOW - 1

    def _crossings(self):
        first = self._first_usable()
        return [t for t in range(first + 1, len(self.PRICES))
                if self._below(t) and not self._below(t - 1)]

    def _persistent(self):
        return [t for t in range(self._first_usable(), len(self.PRICES))
                if self._below(t)]

    def _reportable(self, sessions):
        return [s for s in sessions
                if self.REPORTED_FIRST <= s <= self.REPORTED_LAST]

    def test_the_first_usable_average_is_where_expected(self):
        assert self._ma(1) is None
        assert self._ma(self._first_usable()) == 100

    def test_the_thresholds_are_the_ones_the_series_produces(self):
        assert self._ma(3) == 96
        assert self._ma(8) == 95
        assert self._ma(9) == 91

    def test_exactly_one_crossing_is_reportable(self):
        """Two crossings, one of them in the warm-up. The early one fed the
        average and moved no money."""
        assert self._crossings() == [3, 8]
        assert self._reportable(self._crossings()) == [8]

    def test_the_condition_also_holds_at_nine(self):
        """Which is what makes the persistent wiring visible: state and
        transition differ on this series."""
        assert self._persistent() == [3, 8, 9]

    def test_execution_is_strictly_later_than_the_signal(self):
        signal, funding, execution = 8, 8, 9
        assert signal <= funding <= execution
        assert signal < execution

    def test_the_fill_is_matched_by_identity(self):
        fills = [{"eventId": 1, "session": 9}]
        contribution = {"id": 1, "execution": 9}
        matched = next(f for f in fills if f["eventId"] == contribution["id"])
        assert matched["session"] == contribution["execution"]

    def test_the_ledger_reconciles(self):
        assert ending_cash(opening=0, contributions=[8_800], withdrawals=[],
                           purchases=[8_800], sales=[], fees=0) == 0
        assert ending_shares(opening=0, bought=[100 * SHARE], sold=[]) == \
            100 * SHARE

    def test_persistent_wiring_would_pay_twice(self):
        """The wiring mutation, on this side. Every operator still correct,
        and the reported total doubles."""
        assert len(self._reportable(self._persistent())) == 2

    def test_skipping_the_window_filter_would_also_pay_twice(self):
        assert len(self._crossings()) == 2

    def test_the_lean_file_states_the_same_chain(self):
        path = LEAN.parent / "Composition.lean"
        if not path.exists():
            pytest.skip("formal/Quantify/Composition.lean is absent")
        text = path.read_text()

        assert "[100, 100, 100, 90, 100, 100, 100, 100, 85, 88, 100]" in text
        assert "(crossing observations).map sessionOf == [3, 8]" in text
        assert "(persistent observations).map sessionOf == [3, 8, 9]" in text
        assert "#guard reportableSignals == [8]" in text
        assert "def contribution : Event := ⟨1, 8, 8, 9⟩" in text
        assert "def reported : Window := ⟨5, 5, 10⟩" in text


class TestMathlibStaysAtTheReturnsBoundary:
    """The dependency is contained by module, and checked here rather than
    trusted.

    Return metrics are where the mathematics changes kind — a product of
    ratios, not a count — so exact rationals are the right representation and
    Mathlib is the way to get them. Everything before that is discrete and must
    stay buildable from a bare toolchain: an `import Mathlib` that crept into
    `Ledger.lean` would make every conservation proof inherit a dependency it
    has no use for.
    """

    FORMAL = LEAN.parent

    def test_no_core_module_imports_mathlib(self):
        if not self.FORMAL.exists():
            pytest.skip("formal/Quantify is absent")

        offenders = []
        for path in sorted(self.FORMAL.glob("*.lean")):
            if re.search(r"^import Mathlib", path.read_text(), re.M):
                offenders.append(path.name)
        assert not offenders, (
            f"{offenders} import Mathlib; the lightweight core must build "
            "from a bare toolchain")

    def test_the_returns_modules_do(self):
        """The other half. A containment test that passed because nothing used
        Mathlib at all would be checking an empty room."""
        returns = self.FORMAL / "Returns"
        if not returns.exists():
            pytest.skip("formal/Quantify/Returns is absent")

        using = [p.name for p in sorted(returns.glob("*.lean"))
                 if re.search(r"^import Mathlib", p.read_text(), re.M)]
        assert using, "no Returns module imports Mathlib"

    def test_both_libraries_are_default_targets(self):
        """The build-scope trap, one target later.

        A second `lean_lib` without `@[default_target]` would leave a bare
        `lake build` compiling the core and skipping every return proof — the
        same green-badge-over-unbuilt-proofs defect this repository already hit
        once, when the library had no default target at all and two mutations
        passed.
        """
        lakefile = (self.FORMAL.parent / "lakefile.lean").read_text()
        # Declarations only. The first version counted every occurrence in the
        # file and matched the phrase `@[default_target]` inside a doc comment
        # explaining why it is there — a scan that reads its own prose, which
        # this project has now done often enough to have a name for.
        libs = re.findall(r"^lean_lib (\w+)", lakefile, re.M)
        defaults = len(re.findall(r"^@\[default_target\]", lakefile, re.M))
        assert len(libs) >= 2
        assert defaults == len(libs), (
            f"{len(libs)} libraries and {defaults} default targets; a bare "
            "`lake build` would skip one")

    def test_every_declared_root_has_a_file(self):
        """A root naming a file that does not exist fails the build rather
        than silently shrinking the proof set — verified when
        `Quantify.Returns.TimeWeighted` was declared before it was written and
        the build refused."""
        lakefile = (self.FORMAL.parent / "lakefile.lean").read_text()
        # Read out of the `roots := #[...]` arrays, not from anywhere a
        # backtick appears: the same scan matched `Quantify.Returns` inside the
        # sentence saying Mathlib is required only for it.
        declared = []
        for block in re.findall(r"roots := #\[(.*?)\]", lakefile, re.S):
            declared += re.findall(r"`([\w.]+)", block)
        assert declared, "no roots found; the scan is looking in the wrong place"
        for root in declared:
            path = self.FORMAL.parent / (root.replace(".", "/") + ".lean")
            assert path.exists(), f"{root} is a declared root with no file"


class TestReturnsAgreeWithinTolerance:
    """Lean states the exact rational; Python computes in floating point.

    Kept explicitly separate. The engine's number is an approximation of the
    specification, not the other way round, and comparing them needs a stated
    tolerance rather than an equality that happens to hold for these inputs.

    Formalising Python's `float` instead would mean verifying IEEE-754
    rounding, which is a harder target than the finance and answers a different
    question.
    """

    TOLERANCE = 1e-12

    def test_simple_return(self):
        assert abs((110 - 100) / 100 - 0.1) < self.TOLERANCE
        assert abs((90 - 100) / 100 - -0.1) < self.TOLERANCE

    def test_up_then_down_loses_one_percent(self):
        """The compounding case a naive sum gets wrong: +10% then -10% is not
        zero."""
        twr = (110 / 100) * (99 / 110) - 1
        assert abs(twr - -0.01) < self.TOLERANCE

    def test_a_boundary_contribution_does_not_manufacture_performance(self):
        """100 grows to 110, 50 arrives, 160 grows to 176.

        True time-weighted return 21%. The naive figure counts the depositor's
        money as performance and reports 26%.
        """
        twr = (110 / 100) * (176 / 160) - 1
        assert abs(twr - 0.21) < self.TOLERANCE

        naive = (176 - 100 - 50) / 100
        assert abs(naive - 0.26) < self.TOLERANCE
        assert naive > twr, "the naive figure must overstate, or the fixture "\
                            "is not showing what TWR is for"

    def test_the_flow_size_does_not_matter(self):
        """The theorem's content, sampled. Any contribution at the boundary
        leaves the reported return alone."""
        for flow in (0, 50, 1_000, 10**6):
            start_two = 110 + flow
            twr = (110 / 100) * ((start_two * 1.1) / start_two) - 1
            assert abs(twr - 0.21) < 1e-9

    def test_the_lean_file_states_the_same_numbers(self):
        path = LEAN.parent / "Returns" / "TimeWeighted.lean"
        if not path.exists():
            pytest.skip("TimeWeighted.lean is absent")
        text = path.read_text()
        assert "= -1 / 100" in text
        assert "= 21 / 100" in text
        assert "= 26 / 100" in text
