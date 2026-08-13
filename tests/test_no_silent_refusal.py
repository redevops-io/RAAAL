"""A refusal that names a dimension and gives no reason.

The pilot page rendered

    What this will not do
      • day_rule:
      • dividend_policy: the engine runs on price series only, so dividends…
      • evaluation_period: this build evaluates over the whole price history…

The first entry is the defect. `day_rule` is an EXECUTED dimension with a closed
set, gpt-5.4 answered it with `calendar_first_rolled_forward` — a perfectly
sensible English description of a rule this build does not have — and the detail
came from `refuses.get(value, "")`, which carries only the values somebody
anticipated. A reader can always return one nobody did.

Being told something is wrong and not what is worse than not being told: the
person cannot act on it, cannot tell whether it matters, and cannot tell whether
the rest of the page is trustworthy. Refusing *by name* is the whole boundary,
and a name with no reason is half of it.
"""
from __future__ import annotations

import pytest

from src.mission.capability import EXECUTED, MANIFEST, decide


def closed_executed():
    return [(name, d) for name, d in MANIFEST.items()
            if d.support == EXECUTED and d.closed and d.values]


class TestEveryRefusalSaysWhy:
    @pytest.mark.parametrize("name", [n for n, _ in closed_executed()])
    def test_an_unanticipated_value_still_gets_a_reason(self, name):
        """The case that produced an empty line. Not a value from `refuses` —
        one nobody listed, which is what a reader actually returns."""
        refusal = decide(name, "something-no-one-wrote-down")
        assert refusal is not None, f"{name} accepted a value outside its set"
        assert refusal.detail.strip(), (
            f"{name} refuses this value and says nothing; the page renders "
            f"'{name}:' followed by blank space")

    @pytest.mark.parametrize("name", [n for n, _ in closed_executed()])
    def test_the_reason_names_what_would_work(self, name):
        """A refusal a person can act on. The executable set is the most
        useful next sentence, and the dimension already carries it."""
        refusal = decide(name, "something-no-one-wrote-down")
        assert any(str(v) in refusal.detail for v in MANIFEST[name].values), (
            f"{name} refuses without naming a value that would run")

    @pytest.mark.parametrize("name,value", [
        ("cadence", "payroll"),
        ("objective", "assess_withdrawal"),
        ("allocation_method", "risk_parity"),
    ])
    def test_a_written_reason_is_not_replaced(self, name, value):
        """The generated sentence is a floor, not a ceiling. Where somebody
        wrote a specific reason — a pay cycle is not a calendar period — that
        is what the user reads, because it explains the boundary rather than
        restating the vocabulary."""
        refusal = decide(name, value)
        assert refusal is not None
        assert refusal.detail == MANIFEST[name].refuses[value]

    def test_a_refused_dimension_still_uses_its_own_words(self):
        """Dimension-level refusals were never silent and must stay that way;
        this fix touches only the value-level branch."""
        refusal = decide("sell_action", "sell 4% a year")
        assert refusal is not None
        assert refusal.detail == MANIFEST["sell_action"].why
        assert refusal.detail.strip()


class TestNothingIsRefusedWithoutBeingAsked:
    def test_an_executable_value_is_not_refused(self):
        """The reciprocal. A reason generator is only safe if it fires on
        values that genuinely do not run."""
        assert decide("cadence", "monthly") is None
        assert decide("day_rule", "first_session_of_period") is None

    def test_an_unknown_dimension_is_not_refused(self):
        """Unchanged behaviour, restated because this file is about refusals
        appearing where they should not. A dimension nobody has classified is
        not thereby forbidden."""
        assert decide("a_dimension_nobody_has_classified", "anything") is None
