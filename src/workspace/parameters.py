"""One table: the parameters this plan actually has, and what is missing.

The page used to carry four lists — what was understood, what needs an answer,
what the engine chose, what was not mentioned — and a person had to assemble
the plan from them. That is the same information arranged as work.

**Only what this strategy needs.** The manifest declares nineteen dimensions
and no sentence uses nineteen. A contribution schedule has an amount, a
cadence, a holding and a day; it has no withdrawal ordering and asking about
one would be noise. So the rows come from the reading — settled, asked,
refused, defaulted — and never from the manifest's full list.

Which means a picked strategy shows fewer blanks than a typed one, and that is
the point rather than an accident: the catalogue sentences were written to
state their parameters, so the work of filling them in is already done.

**A blank carries an example, not a prompt.** "day_rule" tells somebody
nothing; `ModifiedFollowing` tells them the vocabulary, and `the 15th` tells
them what to type. Both come from declarations that already exist — the
schema's examples and the capability manifest's executable values — so a
dimension cannot gain a value here that the engine will then refuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

#: What a row is doing on the page.
SETTLED = "SETTLED"
"""Read from the sentence. Carries a value and where it came from."""

NEEDED = "NEEDED"
"""Blank, and the plan cannot run without it. The only rows with an input."""

REFUSED = "REFUSED"
"""Stated, understood, and not something this build executes."""

CHOSEN = "CHOSEN"
"""Not stated, and the engine applied a declared default rather than asking."""


@dataclass(frozen=True)
class Parameter:
    name: str
    state: str
    value: str = ""
    #: How it came to be settled — the fusion outcome and its witnesses. Shown
    #: rather than summarised: MODEL_ONLY_ACCEPTED is not AGREE, and a pilot
    #: that printed "agreed" while running one reader would be claiming
    #: corroboration it never had.
    provenance: str = ""
    #: What the dimension is, in the schema's words.
    describes: str = ""
    #: What somebody could type. Ordinary language first, because that is what
    #: the box takes.
    examples: Sequence[str] = ()
    #: What the engine will actually run, where the set is closed. A blank
    #: whose example is not executable would be an invitation to a refusal.
    executable: Sequence[str] = ()
    detail: str = ""

    @property
    def needs_an_answer(self) -> bool:
        return self.state == NEEDED

    def to_json(self) -> Mapping[str, Any]:
        return {"name": self.name, "state": self.state, "value": self.value,
                "provenance": self.provenance, "describes": self.describes,
                "examples": list(self.examples),
                "executable": list(self.executable), "detail": self.detail}


#: Examples in the vocabulary rather than in prose, for the dimensions where a
#: convention exists. The rest fall back to the schema's own examples.
#:
#: These are the names QuantLib uses, which are the market's. A person reading
#: `ModifiedFollowing` beside a blank can look it up; a person reading "which
#: session within the period" cannot.
CONVENTION_EXAMPLES: Mapping[str, Sequence[str]] = {
    "day_rule": ("the 15th", "the first trading day", "the last trading day"),
    # The engine's own word first, then the shorthand. "every month" reads
    # better and `monthly` is what this build executes; an example that reads
    # well and earns a refusal is worse than one that reads plainly.
    "cadence": ("monthly (1m)", "biweekly (2w)", "weekly (1w)"),
    "evaluation_period": ("the last 5 years", "since 2015"),
    "stated_weights": ("60% VTI and 40% BND",),
    "rebalancing": ("back to target once a year",),
    # The six dimensions the schema describes and gives no example for. A
    # blank with no example is a prompt to guess, and guessing at `assets` is
    # how somebody writes "an index fund" and gets a refusal about tickers.
    "assets": ("VTI", "VOO and BND", "a US total market fund"),
    "observed_assets": ("SPY", "the S&P 500"),
    "amount": ("$500", "$2,000"),
    "account_type": ("a taxable account", "a Roth IRA", "a 401k"),
    "dividend_policy": ("reinvested", "held as cash"),
    "moving_average_window": ("the 200-day average", "the 50-day average"),
}


def _schema_dimension(name: str):
    from ..discovery.schema import QUANTIFY_SCHEMA

    for dimension in QUANTIFY_SCHEMA.dimensions:
        if dimension.name == name:
            return dimension
    return None


def _guidance(name: str) -> Mapping[str, Any]:
    """What to show beside a blank: what it is, and what to type."""
    from ..mission.capability import dimension as manifest_dimension

    described = _schema_dimension(name)
    declared = manifest_dimension(name)
    examples = CONVENTION_EXAMPLES.get(name) or (
        tuple(described.examples) if described is not None else ())
    return {
        "describes": described.describes if described is not None else "",
        "examples": examples,
        # Only where the manifest closes the set. An open dimension listing
        # "executable values" would be claiming a boundary it does not have.
        "executable": (tuple(declared.values)
                       if declared is not None and declared.closed else ()),
    }


def rows(reading) -> Sequence[Parameter]:
    """Every parameter this reading touched, in one list.

    Deliberately not every dimension the manifest declares. A sentence about
    monthly contributions has nothing to say about withdrawal ordering, and a
    table that asked about it would be asking a person to disregard most of it
    — which is how a form teaches people to skim.
    """
    found = []
    seen = set()

    # What was read, indexed but not yet emitted. A field can be *both* read
    # and unusable — "200usd" is what the person wrote and not a number the
    # engine can contribute — and the reading says so by listing it in
    # `questions` while also settling a value.
    read_values = {settled.field: settled
                   for settled in (getattr(reading, "settled", ()) or ())}

    # Questions first, and they win. Emitting the settled row instead dropped
    # the question from the table while the button below still offered to
    # answer it: the page asked somebody to fill something in, showed nothing
    # to fill, and looped when they pressed it.
    for name in getattr(reading, "questions", ()) or ():
        if name in seen:
            continue
        seen.add(name)
        already = read_values.get(name)
        found.append(Parameter(
            name=name, state=NEEDED,
            # Prefilled with what was read, so correcting "200usd" to "200" is
            # an edit rather than a retype. A blank here would throw away a
            # reading that was almost right.
            value=str(already.value) if already is not None else "",
            provenance=(f"read as {already.value!r}, and not usable as stated"
                        if already is not None else ""),
            **_guidance(name)))

    for settled in getattr(reading, "settled", ()) or ():
        name = settled.field
        if name in seen:
            continue
        seen.add(name)
        witnesses = ", ".join(settled.witnesses or ())
        found.append(Parameter(
            name=name, state=SETTLED, value=str(settled.value),
            provenance=(f"{settled.provenance}"
                        + (f" ({witnesses})" if witnesses else "")),
            **_guidance(name)))

    # `questions`, not `open_fields`. The reading distinguishes a dimension
    # fusion could not settle from one the engine needs supplied, and
    # `questions` is the union — the same property the page's own button is
    # driven by. Reading only the first half built a table that omitted
    # `assets` while the button below it offered to run.
    for name in getattr(reading, "questions", ()) or ():
        if name in seen:
            continue
        seen.add(name)
        found.append(Parameter(name=name, state=NEEDED, **_guidance(name)))

    for refusal in getattr(reading, "refusals", ()) or ():
        name = getattr(refusal, "dimension", "")
        if not name or name in seen:
            continue
        seen.add(name)
        found.append(Parameter(
            name=name, state=REFUSED,
            value=str(getattr(refusal, "stated_value", "") or ""),
            detail=getattr(refusal, "detail", ""), **_guidance(name)))

    for name in getattr(reading, "applied_defaults", ()) or ():
        if name in seen:
            continue
        seen.add(name)
        found.append(Parameter(name=name, state=CHOSEN,
                               provenance="the engine's declared default",
                               **_guidance(name)))

    # Needed first: it is the only part that is work. Then what was read, then
    # what will not run, then what was chosen for them.
    order = {NEEDED: 0, SETTLED: 1, REFUSED: 2, CHOSEN: 3}
    return sorted(found, key=lambda p: (order.get(p.state, 9), p.name))


def unanswered(reading) -> Sequence[Parameter]:
    return [p for p in rows(reading) if p.needs_an_answer]
