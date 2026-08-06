"""What each field may hold, in one place.

Three things were separate and disagreed with each other: the compiler decided
which questions to ask, `workspace/confirmation.CHOICES` decided which options
to show, and nothing at all decided which answers were acceptable. The third
gap is the one that matters — `cadence=banana` removed the question and
recorded "cadence: banana (answered)" as a *stated fact*, so a saved plan
could carry a cadence no part of the engine understands.

So the enumeration is the vocabulary and the validation at once. A field's
options are what the page offers, what the compiler accepts, and what a stored
plan may contain; there is no fourth list to fall out of step.

**Values are the engine's, not the page's.** `_CADENCE_WORDS` in `render.py`
already fixes the cadence vocabulary — "monthly", "annual", "once" — and a
dropdown offering "annually" would settle a question with a value nothing
renders. The labels are for people, the values are for the machine, and only
the labels are free.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple


class FieldKind(str, Enum):
    CHOICE = "choice"
    """A closed set. The page offers exactly these and refuses anything else."""

    AMOUNT = "amount"
    """Money. Any number the user likes, parsed rather than enumerated."""

    FREE = "free"
    """Open text with no vocabulary to check against."""


@dataclass(frozen=True)
class Option:
    value: str
    label: str


@dataclass(frozen=True)
class Field:
    name: str
    kind: FieldKind
    prompt: str
    options: Tuple[Option, ...] = ()

    def accepts(self, answer: str) -> bool:
        if self.kind is not FieldKind.CHOICE:
            return True
        return any(one.value == answer for one in self.options)


def _choice(name, prompt, pairs) -> Field:
    return Field(name, FieldKind.CHOICE, prompt,
                 tuple(Option(v, l) for v, l in pairs))


#: Every field a user can be asked to settle, and what it will take.
FIELDS: Mapping[str, Field] = {
    "account_type": _choice("account_type", "Which account is this in?", (
        ("TAXABLE", "Taxable brokerage account"),
        ("TRADITIONAL_IRA", "Traditional IRA"),
        ("ROTH", "Roth IRA"),
        ("TRADITIONAL_401K", "401(k)"),
        ("ROTH_401K", "Roth 401(k)"),
    )),
    # Values from `render._CADENCE_WORDS`. "annually" reads naturally and is
    # not one of them; offering it would settle the question with a value the
    # renderer cannot put into a sentence.
    # Only offered when the description named a moving average without a
    # period. Any window is accepted *in the description* — the compiler parses
    # "63-day" as readily as "200-day" — so this closed set narrows the
    # fallback question rather than the product. A free text box here would
    # accept "long" and "the usual", and a 200-session and a 50-session
    # average are different rules producing different purchases.
    "moving_average_window": _choice(
        "moving_average_window", "How many sessions does the average cover?", (
            ("20", "20 sessions (about a month)"),
            ("50", "50 sessions (about a quarter)"),
            ("100", "100 sessions"),
            ("200", "200 sessions (about a year)"),
        )),
    "cadence": _choice("cadence", "How often?", (
        ("weekly", "Every week"),
        ("biweekly", "Every other week"),
        ("monthly", "Every month"),
        ("quarterly", "Every quarter"),
        ("annual", "Every year"),
        ("payroll", "Every payday"),
        ("daily", "Every trading day"),
        ("once", "One lump sum"),
    )),
    "trigger_semantics": _choice(
        "trigger_semantics", "When the condition is true, buy...", (
            ("persistent_condition", "Every day it stays true"),
            ("crossing_event", "Only on the day it first becomes true"),
        )),
    "funding_source": _choice("funding_source", "Where does the money come from?", (
        ("contribution", "Out of the regular contribution"),
        ("additional_cash", "Additional money on top"),
    )),
    "weighting": _choice("weighting", "How are the positions sized?", (
        ("equal_weight_at_purchase", "Equal dollars at each purchase"),
        ("equal_weight_maintained", "Kept equal over time"),
    )),
    "dividends": _choice("dividends", "What happens to dividends?", (
        ("reinvested", "Reinvest them"),
        ("held_as_cash", "Hold them as cash"),
    )),
    "contribution_day_rule": _choice(
        "contribution_day_rule", "Which day does the contribution land?", (
            ("first_session_of_period", "First trading day of the period"),
            ("calendar_first_rolled_forward", "First calendar day of the month"),
        )),
    "moving_average_kind": _choice(
        "moving_average_kind", "Which kind of moving average?", (
            ("simple", "Simple"),
            ("exponential", "Exponential"),
        )),
    # Values from `render._DAY_RULE_WORDS`' sibling in the execution vocabulary.
    "execution_timing": _choice("execution_timing", "When does the order fill?", (
        ("next_session_open", "Next session's open"),
        ("same_session_close", "The same session's close"),
    )),
    "amount": Field("amount", FieldKind.AMOUNT, "How much, each time?"),
    "starting_capital": Field("starting_capital", FieldKind.AMOUNT,
                              "How much are you starting with?"),
}


#: Fields a *deployment* settles, never a user.
#:
#: `benchmark_set` is supplied by `BENCHMARK_RULE` at every entry point the
#: pilot serves; it becomes unresolved only where no policy was configured,
#: which is a deployment fault rather than a question. Listing it explicitly
#: rather than letting it fall through is the point: the guard admits exactly
#: two outcomes for a compiler field — a registry entry or a named exemption —
#: and there is no third that quietly becomes a text box.
POLICY_SETTLED: Tuple[str, ...] = ("benchmark_set",)


def field_for(name: str) -> Optional[Field]:
    """The field behind a question id.

    Prefixed ids — `unclear:x`, `asset_identity:SPX` — are generated from the
    parse and have no vocabulary; they return None rather than a fabricated
    one.
    """
    if not name or ":" in name:
        return None
    return FIELDS.get(name)


def accepts(name: str, answer: str) -> bool:
    """Whether this answer is one the field may hold.

    Unknown fields accept anything: refusing them would make every new
    question unanswerable until somebody remembered to add it here, and
    failing closed on *unrecognised* is a different decision from failing
    closed on *wrong*.
    """
    field = field_for(name)
    if field is None:
        return True
    return field.accepts(answer)


def options_for(name: str) -> Sequence[Option]:
    field = field_for(name)
    return field.options if field else ()
