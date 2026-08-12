"""Well-formed statements to pick from, and the ones this build will not take.

The harvest found that real strategy statements routinely omit what to buy —
"I contribute $750/month to my 401k" is a complete thought to the person writing
it, and the runtime cannot run it. Asking people to guess the vocabulary that
gets them a plan is the interaction defect underneath most of the follow-up
burden. A list of statements they can pick and then edit removes the guessing
without removing the typing: the text box stays the primary input, and a picked
sentence lands in it as ordinary editable text.

Two rules hold this together, and both exist because a catalogue is the easiest
place in a product to make a claim nobody checks.

**Everything offered must execute.** An entry in a dropdown is the product
saying "this works". Offering a strategy that compiles to a refusal is a false
claim of support, and worse than the user typing the same sentence unprompted,
because we suggested it. `tests/test_strategy_library.py` runs every offered
entry through the whole pipeline and fails if one does not produce a plan.

**Everything refused is shown, with the engine's reason.** The obvious
catalogue for this domain is momentum, value, relative value — and this build
models none of them: ranking holdings is `selection_rule`, which is not
modelled, and a short leg is `sell_action`, which is refused. Leaving them out
would send somebody looking for momentum away to type it themselves and meet
the refusal three screens later, having invested a paragraph in it. They are
listed here, unselectable, carrying the manifest's own words — and the list is
derived from the manifest, not written out beside it, so it cannot drift into
describing a boundary the engine has moved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Entry:
    """One statement somebody can pick.

    `text` is what lands in the box, and it is deliberately a whole sentence in
    ordinary words rather than a fill-in-the-blanks form. The runtime reads
    prose; a template with slots would be a second grammar to maintain, and the
    thing the pilot is measuring is whether prose works.
    """

    key: str
    title: str
    text: str
    #: What this entry exists to demonstrate. Shown to nobody — it is the
    #: reason the entry is in the catalogue, for whoever edits it next.
    demonstrates: str


@dataclass(frozen=True)
class Group:
    key: str
    title: str
    #: Why these belong together, in the user's terms rather than the schema's.
    note: str
    entries: Tuple[Entry, ...]


#: The offered catalogue.
#:
#: Grouped by the distinction the engine actually makes — what puts money in,
#: and what it buys — because that is what changes the answer. Grouping by the
#: names the industry uses would be grouping by a distinction this build cannot
#: act on, and every group but one would be empty.
LIBRARY: Tuple[Group, ...] = (
    Group(
        key="regular",
        title="Money in on a schedule",
        note="A fixed amount on a calendar. The most common shape by far, and "
             "the one the engine models most directly.",
        entries=(
            Entry("monthly-single", "Monthly into one fund",
                  "invest $500 a month into VTI",
                  "the baseline: one asset, one cadence, one amount"),
            Entry("monthly-pair", "Monthly, split between two funds",
                  "invest $500 a month split equally between VTI and BND",
                  "a SET-valued holding; equal split is what the engine does"),
            Entry("weekly-single", "Weekly into one fund",
                  "put $100 into VOO every week",
                  "a non-monthly cadence, phrased as a person would"),
            Entry("quarterly-single", "Quarterly into one fund",
                  "invest $3,000 into QQQ every quarter",
                  "a cadence with a thousands separator in the amount"),
            Entry("first-of-month", "Monthly, on the first trading day",
                  "invest $750 into VTI on the first trading day of each month",
                  "day_rule stated explicitly rather than defaulted"),
        ),
    ),
    Group(
        key="triggered",
        title="Money in when the market does something",
        note="A condition rather than a date. The engine fills on the next "
             "session's open — never the close that produced the signal.",
        entries=(
            Entry("below-ma", "Buy when it falls below its moving average",
                  "buy $1,000 of SPY every time it closes below its 200 day "
                  "moving average, on the next trading day",
                  "crossing_event, and the sentence the first pilot user "
                  "actually typed"),
            Entry("stays-below", "Buy while it stays below its average",
                  "buy $1,000 of SPY on any day it is trading below its 200 "
                  "day moving average",
                  "persistent_condition — the other reading of the same shape, "
                  "and a different strategy"),
            # A drawdown entry — "invest $2,000 into VTI whenever it drops 10%
            # below its highest close of the last year" — was written here and
            # removed. It runs under claude-sonnet-5 and is refused under
            # gpt-4.1-2025-04-14, which reads the fixed $2,000 as a
            # `conditional_amount` and refuses a strategy whose amount does not
            # in fact vary. Offering a sentence that works on one provider is
            # offering a coin toss, so it is out until the false refusal is
            # fixed; docs/Benchmark-Queue.md carries it.
        ),
    ),
    Group(
        key="one-off",
        title="A single purchase",
        note="One amount, once. Useful as a baseline to compare a "
             "contribution plan against.",
        entries=(
            Entry("lump-sum", "Invest a lump sum now",
                  "invest $10,000 into VTI as a lump sum",
                  "cadence=once; the phrase that broke the clarification loop"),
            Entry("lump-sum-pair", "A lump sum across two funds",
                  "invest $10,000 equally into VTI and BND as a one-off "
                  "purchase",
                  "once + SET holding together"),
        ),
    ),
)


#: Concepts a catalogue in this domain would obviously carry, mapped to the
#: manifest dimension that decides them. The heading is ours — somebody looking
#: for "momentum" is not looking for `selection_rule` — and the reason shown
#: beside it is the engine's, read at render time.
#:
#: This is a mapping to dimension *names*, not to sentences. A hand-written
#: explanation here would be a second account of the boundary, and the two
#: would part company the first time the engine gained a capability.
#:
#: `dimension:value` where the boundary is inside a dimension the engine
#: otherwise executes. `allocation_method` is the case that forced this: it
#: *is* executed — equally, at purchase — and refuses risk parity as one of its
#: values. Mapping the heading at the dimension would have told users the
#: engine cannot allocate, which is both wrong and the opposite of the mistake
#: this list exists to prevent.
UNSUPPORTED: Mapping[str, str] = {
    "Momentum — buying what has been rising": "selection_rule",
    "Value — buying what screens as cheap": "selection_rule",
    "Relative value — long one holding, short another": "sell_action",
    "A stated split, such as 60/40": "stated_weights",
    "Rebalancing back to target weights": "periodic_rebalancing",
    "Selling, withdrawing or taking profits": "sell_action",
    "Whether you could retire or live off this": "objective:assess_withdrawal",
    "Paying down a debt instead of investing": "objective:assess_debt_repayment",
    "Holding for a set period, then exiting": "holding_period",
    "Risk parity or volatility targeting": "allocation_method:risk_parity",
    "Contributing more when the signal is stronger": "conditional_amount",
    "Choosing which account each holding sits in": "asset_location",
    "Anything whose answer depends on tax": "tax_treatment",
    "Restricting the run to a stated window": "evaluation_period",
}


def _reason(name: str) -> Tuple[str, str]:
    """Resolve `dimension` or `dimension:value` to the engine's own words."""
    from ..mission.capability import MANIFEST

    dimension_name, _, value = name.partition(":")
    dimension = MANIFEST.get(dimension_name)
    if dimension is None:
        raise KeyError(
            f"{name!r} names {dimension_name!r}, which the capability manifest "
            "does not have; the reason shown to a user would be missing and "
            "nothing else would notice")
    if value:
        if value not in dimension.refuses:
            raise KeyError(
                f"{name!r} names a value {dimension_name!r} does not refuse; "
                "the heading would claim a boundary the engine has moved")
        return dimension_name, dimension.refuses[value]
    return dimension_name, (dimension.why
                            or next(iter(dimension.refuses.values()), ""))


def unsupported() -> Tuple[Tuple[str, str], ...]:
    """The refused catalogue, with each reason read from the manifest.

    Raises if a heading names a dimension the manifest does not have. A silent
    skip would let this list decay into headings with no reason behind them,
    which is the failure the derivation was meant to prevent — and it would
    decay invisibly, because a missing entry renders as nothing at all.
    """
    return tuple((heading, _reason(name)[1])
                 for heading, name in UNSUPPORTED.items())


def entry(key: str) -> Optional[Entry]:
    for group in LIBRARY:
        for candidate in group.entries:
            if candidate.key == key:
                return candidate
    return None


def offered() -> Sequence[Entry]:
    return [e for group in LIBRARY for e in group.entries]


#: How a sentence got into the box. Recorded on every attempt, because without
#: it the cohort measures the catalogue rather than the runtime: a high success
#: rate over sentences we wrote, read by a reader we wrote, is a closed loop.
#: `TYPED` is the only origin that carries evidence about people's own words,
#: and `EDITED` is the interesting middle — it says the catalogue got them
#: close and something was missing.
PICKED, EDITED, TYPED = "PICKED", "EDITED", "TYPED"


def origin_of(text: str, picked_key: str) -> str:
    """Derived from the text and the pick, never taken from the client.

    A hidden field saying "PICKED" is a claim the browser makes, and the one
    thing it cannot be trusted about is whether the user then changed the
    sentence — which is exactly the distinction this is for. Comparing against
    the catalogue is cheap and cannot be wrong about it.
    """
    if not picked_key:
        return TYPED
    chosen = entry(picked_key)
    if chosen is None:
        return TYPED
    return PICKED if " ".join(text.split()) == " ".join(
        chosen.text.split()) else EDITED
