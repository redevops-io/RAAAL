"""Canonical values, decided where meaning is decided.

A `VerifiedIntent` that still needs re-reading was never verified. Mission had
six places where it read language to decide what a sealed value meant — a
cadence normalised from prose, holdings split on an English "and", a negation
matched against a word list, a figure with its currency stripped — and each was
a second opinion about meaning formed after Discovery had closed the question.

Two consequences, and the second is why this exists rather than a style note.
The same sealed artifact can compile differently as that code changes, which is
what pinning an intent is supposed to prevent. And the evaluation service cannot
take `discovery.syntax` with it, so a value parsed downstream is an import error
waiting for the extraction.

**Refuse, never substitute.** A stated value this cannot canonicalise produces a
refusal naming the dimension. Substituting a default would hand somebody a plan
that looks like the one they asked for and is not — the defect that compiled
`$1k monthly` to an amount of zero, with every other field correct.

**Absence is not unreadability.** A dimension nobody mentioned is left alone
here; whether the engine has a default for it is Mission's question and Mission
reports it. What this refuses is a dimension somebody *stated* and this cannot
read, which is a different sentence and gets a different one.

**A derived field is authored, not asserted.** `sells_allowed` comes from a
negated disposal, and it is emitted as `Author.READER` — a deterministic rule
read it, not the model, and not the user. The distinction survives into the
plan, where "you said you never sell" and "the engine cannot sell anyway" are
different claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: How the members of a set-valued dimension are separated.
#:
#: This rule used to live twice — here and in Mission — because Mission may not
#: import Discovery, and only one copy knew about `and`. "split equally between
#: VTI and BND" compiled to a single holding named `"VTI and BND"`: one
#: instrument, with a name no market has, at 100%. Fusion had agreed the
#: sentence named two assets and Mission built a portfolio of one.
#:
#: One copy now, on this side of the boundary, and the canonical form it
#: produces is comma-separated so the other side needs no rule at all.
SET_SEPARATOR = re.compile(r"[,;]|\band\b")

#: Currency written beside the figure rather than in front of it. People type
#: "1000 usd" as readily as "$1,000", and a reader that takes one and refuses
#: the other turns an ordinary answer into an unanswerable question — this one
#: asked, accepted the same three characters, and asked again, forever.
_CURRENCY = re.compile(
    r"\b(usd|dollars?|eur|euros?|gbp|pounds?|cad|aud|chf|jpy|yen)\b", re.I)

#: The cadences this vocabulary has names for. A canonical cadence is one of
#: these strings and nothing else, so a consumer compares values rather than
#: spellings.
CADENCES = ("once", "weekly", "biweekly", "monthly", "quarterly", "annual")

#: Dividend handling, canonical. Whether the engine executes both is the
#: manifest's question, not this one: canonicalising a value the engine refuses
#: is what lets it be refused *by name* instead of silently replaced.
DIVIDEND_POLICIES = ("reinvested", "held_as_cash")

_HELD_AS_CASH = re.compile(
    r"\b(held?\s+as\s+cash|as\s+cash|paid\s+out|not\s+reinvest\w*|"
    r"don'?t\s+reinvest\w*|no\s+reinvest\w*|uninvested|in\s+cash)\b", re.I)
_REINVESTED = re.compile(r"\breinvest\w*\b", re.I)


@dataclass(frozen=True)
class Canonicalised:
    """What the seal should carry, and what could not be read.

    `defaulted` is empty here on purpose. This layer never supplies a value for
    a dimension nobody stated — that decision belongs to whoever knows what the
    engine does with silence, and it has to be reported where it is made.
    """

    fields: Dict[str, Tuple[Any, str]] = field(default_factory=dict)
    """dimension -> (canonical value, author name)."""

    refusals: Tuple[Tuple[str, str], ...] = ()
    """(dimension, why) for a value that was stated and cannot be read."""


def _number(raw: Any) -> Optional[str]:
    """A plain decimal string, or None.

    Returns a string rather than a `Decimal` because this value is about to be
    sealed, hashed and compared. `Decimal('1000')` and `Decimal('1000.0')` are
    equal and spell differently, and an execution identity that changed with the
    spelling of a number would report two identical plans as different.
    """
    if raw is None:
        return None
    text = _CURRENCY.sub(" ", str(raw))
    for symbol in (",", "$", "£", "€"):
        text = text.replace(symbol, "")
    try:
        # `normalize()` alone spells 500 as `5E+2`, which is canonical and
        # unreadable — it reached a page as the amount somebody had typed. The
        # `f` format keeps the plain notation while `normalize` still collapses
        # `1000.00` and `1000` onto one spelling, which is the property that
        # matters for an execution identity.
        return format(Decimal(text.strip()).normalize(), "f")
    except (InvalidOperation, ValueError):
        return None


def _cadence(raw: Any) -> Optional[str]:
    """One of `CADENCES`, or None.

    Uses the same normaliser the rest of Discovery uses. That was Mission's
    argument for reaching across the boundary — one place must decide what
    "annually" means — and it was right about the principle and wrong about
    which place.
    """
    text = str(raw).strip().lower()
    if text in CADENCES:
        return text
    from .syntax import normalize

    for value in normalize(text):
        if value.kind == "cadence":
            return str(value.canonical)
    return None


def _members(raw: Any) -> str:
    """A set-valued dimension, comma-separated and stripped."""
    parts = [part.strip() for part in SET_SEPARATOR.split(str(raw))
             if part.strip()]
    return ",".join(parts)


#: Words that deny what they name. Shared with the derived readers, plus one
#: that only ever negates a disposal: "without selling" means no sale, where
#: "buy without waiting for a dip" does not deny the dip — so `without` is
#: consulted for disposals alone and never for triggers.
from .derived_readers import _NEGATIONS as _SHARED_NEGATIONS  # noqa: E402

_NEGATION_WORDS = frozenset(_SHARED_NEGATIONS) | {"without"}

#: Dimensions whose *absence* is what the engine does natively, so a negated
#: statement of them is agreement rather than a request.
#:
#: Only disposal. A negated cadence is not agreement with anything — "I don't
#: contribute monthly" leaves the question open — and treating every negation as
#: assent would turn refusals off wholesale.
NEGATABLE_DISPOSALS = frozenset({"sell_action"})


def _is_negated(raw: Any) -> bool:
    """Whether a stated span denies what it names.

    Word-boundary matched: "another" contains "not" and "nonetheless" contains
    "no", and either would make an ordinary sale read as a refusal to sell —
    this check running backwards.
    """
    if raw is None:
        return False
    words = re.findall(r"[a-z']+", str(raw).lower())
    return any(word in _NEGATION_WORDS or word.endswith("n't")
               for word in words)


def _dividend_policy(raw: Any) -> Optional[str]:
    """`reinvested` or `held_as_cash`, or None.

    Order matters: "not reinvested" contains "reinvest". The denial is tested
    first, so a sentence that says distributions are *not* put back does not
    canonicalise to the policy that puts them back — which would be a wrong
    executable meaning rather than a refused one, and a materially better
    performing strategy than the one described.
    """
    text = str(raw).strip().lower()
    if text in DIVIDEND_POLICIES:
        return text
    if _HELD_AS_CASH.search(text):
        return "held_as_cash"
    if _REINVESTED.search(text):
        return "reinvested"
    return None


def canonicalise(settled: Mapping[str, Any]) -> Canonicalised:
    """Every stated value, in the form a consumer can act on without reading it.

    Dimensions this has no rule for pass through unchanged. That is deliberate:
    a dimension acquires a canonical form when something downstream needs one,
    and inventing rules for the rest would be a second schema maintained by
    nobody.
    """
    out: Dict[str, Tuple[Any, str]] = {}
    refusals: list = []

    for name, raw in settled.items():
        if name in ("amount", "moving_average_window"):
            number = _number(raw)
            if number is None:
                refusals.append((name, (
                    f"{str(raw)!r} was stated for {name} and cannot be read as "
                    "a number. Substituting a default here would produce a plan "
                    "that looks like the one you asked for and is not")))
                continue
            out[name] = (number, "MODEL")

        elif name == "cadence":
            cadence = _cadence(raw)
            if cadence is None:
                refusals.append((name, (
                    f"{str(raw)!r} was stated for how often money moves and "
                    "this build cannot place it on a calendar. It runs "
                    + ", ".join(CADENCES) + "")))
                continue
            out[name] = (cadence, "MODEL")

        elif name == "periodic_rebalancing":
            cadence = _cadence(raw)
            if cadence is None:
                refusals.append((name, (
                    f"{str(raw)!r} asks for rebalancing without saying how "
                    "often. This build restores the split on a calendar — "
                    + ", ".join(c for c in CADENCES if c != "once")
                    + " — and picking one for you would invent a schedule of "
                      "sales you did not describe")))
                continue
            out[name] = (cadence, "MODEL")

        elif name in ("assets", "observed_assets"):
            members = _members(raw)
            if not members:
                refusals.append((name, (
                    f"{str(raw)!r} was stated for {name} and names nothing "
                    "this build can hold")))
                continue
            out[name] = (members, "MODEL")

        elif name == "dividend_policy":
            policy = _dividend_policy(raw)
            if policy is None:
                refusals.append((name, (
                    f"{str(raw)!r} was stated for what happens to "
                    "distributions, and this build understands only "
                    + " or ".join(DIVIDEND_POLICIES))))
                continue
            out[name] = (policy, "MODEL")

        elif name in NEGATABLE_DISPOSALS and _is_negated(raw):
            # Saying you never sell is not selling.
            #
            # "I put $500 a month into VTI and never sold any of it" was refused
            # by name for `sell_action` on a build whose entire behaviour is
            # buying and never selling. The span is extracted correctly and the
            # polarity never reached the decision, so the person described this
            # build exactly and was told it could not be run.
            #
            # The polarity is meaning and is settled here. What Mission receives
            # is the resulting policy, and enforcing it is Mission's job.
            out["sells_allowed"] = (False, "READER")

        else:
            out[name] = (raw, "MODEL")

    return Canonicalised(fields=out, refusals=tuple(refusals))
