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
        assert len(guards) == 8


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
