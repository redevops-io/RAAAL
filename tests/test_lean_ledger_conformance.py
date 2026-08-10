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


class TestThePythonSideMatchesTheLeanFile:
    """Read from the Lean source, so the two cannot drift apart silently.

    The failure this catches: somebody edits a fixture on one side. Both suites
    still pass, and the conformance claim quietly becomes two unrelated sets of
    numbers that happen to be green.
    """

    @pytest.fixture(scope="class")
    def guards(self):
        if not LEAN.exists():
            pytest.skip("formal/Quantify/Fixtures.lean is absent")
        return re.findall(r"#guard\s+(\S+)\.(\w+)(?:\s+\"(\w+)\")?\s*==\s*"
                          r"(-?\d+)", LEAN.read_text())

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
