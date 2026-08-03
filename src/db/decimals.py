"""One decimal grammar, so a stored quantity means the same thing everywhere.

PostgreSQL stores `NUMERIC` and SQLite stores canonical decimal `TEXT`. The
physical types differ deliberately: PostgreSQL is the deployed authority and
SQLite is a local and test profile, so SQLite adapts to the Python contract
rather than dictating it. The parity that matters is not the column type:

    same persisted economic value
        -> same Python Decimal
        -> same serialized canonical value
        -> same reconciliation behaviour

**Floats are refused, not converted.** `0.1` is not one tenth, and a quantity
that arrives as a float has already lost the property the rest of this pipeline
depends on. Converting it here would launder the loss into something that looks
exact — and `152.26` delivered shares, `0.20006` concentration and `3896.10` net
proceeds are all values where a last-place difference changes a status or an
ordering. The refusal is at the boundary so the loss is attributed to the caller
that caused it.

**Trailing zeros are significant and preserved.** `152.20` and `152.2` are the
same number and not the same statement about precision, and the payload these
columns mirror already records `str(Decimal)`. Canonicalization therefore
normalizes *form* — exponents expanded, leading zeros dropped, a digit before
the point — and never magnitude.

The grammar accepts a `Decimal` unconditionally (it is already exact) and a
string only in plain decimal form. Scientific notation, a trailing point and a
leading `+` are refused as input strings because each has more than one obvious
reading, and a stored value with two readings is the thing this module exists
to prevent.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Union

#: Plain decimal only. No exponent, no trailing point, no leading `+`.
CANONICAL = re.compile(r"^-?(?:\d+(?:\.\d+)?|\.\d+)$")

DECIMAL_GRAMMAR_VERSION = "decimal/plain@1"

Number = Union[Decimal, int, str]


class NotADecimal(TypeError):
    """A value that cannot be stored as an exact decimal."""


class DecimalDrift(ValueError):
    """A denormalized column and its authoritative payload disagree.

    Neither is chosen. The database would otherwise hold two answers to the same
    question — the hashed payload and the query-friendly column — and a caller
    would get whichever one it happened to read.
    """


def canonical(value: Optional[Any]) -> Optional[str]:
    """The canonical string for a value, or None.

    Raises `NotADecimal` for a float, for a string outside the grammar, and for
    NaN or Infinity — none of which is an economic quantity.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        # bool is an int subclass; a boolean quantity is a mistake, not a 1.
        raise NotADecimal(
            f"{value!r} is a boolean, not a quantity")

    if isinstance(value, float):
        raise NotADecimal(
            f"{value!r} is a float. Binary floating point cannot represent "
            "most decimal quantities exactly, and converting it here would "
            "make an already-lost value look exact. Pass a Decimal or a "
            "canonical decimal string.")

    if isinstance(value, Decimal):
        number = value
    elif isinstance(value, int):
        number = Decimal(value)
    elif isinstance(value, str):
        text = value.strip()
        if not CANONICAL.match(text):
            raise NotADecimal(
                f"{value!r} is not a canonical decimal. Expected plain decimal "
                "digits with an optional sign and fractional part — no "
                "exponent, no trailing point, no leading '+'.")
        try:
            number = Decimal(text)
        except InvalidOperation as exc:  # pragma: no cover - grammar covers it
            raise NotADecimal(f"{value!r} is not a decimal") from exc
    else:
        raise NotADecimal(
            f"{type(value).__name__} is not a quantity: {value!r}")

    if not number.is_finite():
        raise NotADecimal(f"{value!r} is not a finite quantity")

    # `format(..., 'f')` expands any exponent and keeps trailing zeros, so
    # Decimal('1E+3') becomes '1000' and Decimal('1.00') stays '1.00'.
    text = format(number, "f")
    if text.startswith("-") and not any(ch in "123456789" for ch in text):
        # Negative zero is zero. Two spellings of it would break equality
        # between a stored value and the payload it mirrors.
        text = text[1:]
    return text


def to_decimal(value: Optional[Any]) -> Optional[Decimal]:
    """What every caller receives, from either dialect.

    PostgreSQL hands back a Decimal and SQLite hands back canonical text; both
    arrive here and leave as the same Decimal.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    text = canonical(value)
    return None if text is None else Decimal(text)


def same_value(left: Optional[Any], right: Optional[Any]) -> bool:
    """Whether two representations denote the same quantity.

    Compares canonical strings rather than Decimals so that a difference in
    recorded precision is a difference. `152.2` and `152.20` are equal numbers
    and different statements, and a denormalized copy that quietly rounded is
    exactly the drift this detects.
    """
    return canonical(left) == canonical(right)


class Money:
    """A value bound for a decimal column.

    The same mechanism as `Json`: statements execute as raw SQL with positional
    parameters, so the connection cannot know which column a parameter binds to.
    The value says what it is, and the adapter renders it per dialect.
    """

    __slots__ = ("text",)

    def __init__(self, value: Optional[Any]) -> None:
        self.text = canonical(value)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"Money({self.text!r})"

    def as_decimal(self) -> Optional[Decimal]:
        return None if self.text is None else Decimal(self.text)
