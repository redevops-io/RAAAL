"""Canonical serialization and hashing.

Every other contract in this package depends on exactly one property:

    same handles + same preview plan + same version pins + same contract version
        = same ContextView hash

That has to hold across implementations, across process restarts, and across
languages. It does not hold by accident — it holds because every ambiguity in
serialization is decided here once, in a form another language can reproduce.

The decisions, and why each is the way it is:

**Map keys are sorted.** Insertion order is a property of the process that built
the object, not of what the object means.

**Absent and null are the same thing.** A field set to null is dropped. An
implementation that populates an optional field with null must not hash
differently from one that omits it, or every adapter has to guess which style
the reference implementation used.

**Floats are refused in canonical form.** Cross-language float formatting is the
classic source of hashes that agree on one runtime and disagree on another —
Python, Go and JavaScript do not print the same digits for the same double.
Anything fractional is carried as a decimal *string* and compared as one.

**Strings are NFC-normalized.** Two byte sequences that display identically must
hash identically; an artifact id typed on macOS should not differ from the same
id typed on Linux.

**Volatile timestamps do not participate.** When a handle was last accessed says
nothing about what it refers to.

**Authorization and omission decisions do participate.** They change what the
model was permitted to see, and two views that showed different subsets of the
same graph are not the same view — even when the difference is a redaction.

**Materialized content participates through its pinned hash, never its bytes.**
The canonical form stays small and stays comparable without carrying payloads.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

#: **Runtime Canonicalization Version.** The rules in this module, not the shape
#: of any contract. A digest reading `rcv1:` was computed under these rules.
#:
#: Deliberately independent of schema version: a contract can move to
#: `context-view/0.2` and stay on `rcv1`, and a change to *how things are
#: hashed* becomes `rcv2` without renumbering every schema. Collapsing the two
#: would make every schema bump look like a hashing change and vice versa.
CANONICALIZATION_VERSION = "rcv1"
HASH_PREFIX = CANONICALIZATION_VERSION

#: Retained for callers that used it before the split.
CONTRACT_VERSION = "0.1"


class CanonicalizationError(ValueError):
    """A value cannot be represented canonically.

    Raised rather than coerced. A silent coercion is a hash that differs between
    implementations for a reason nobody logged.
    """


def _text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


#: A decimal string, in the only form this contract accepts.
#:
#: Every rule below removes a way two implementations can write the same number
#: differently. Left undecided, the first fractional cost estimate would produce
#: hashes that differ between a Python producer and a Go one for no reason
#: anybody could see.
_DECIMAL = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$")


def decimal_string(value: "str | int | Decimal") -> str:
    """Normalize a fractional value into the canonical decimal form.

    ===========  ==========  ================================================
    Input        Output      Rule
    ===========  ==========  ================================================
    ``"1.0"``    ``"1"``     trailing zeros removed — one value, one spelling
    ``"1.500"``  ``"1.5"``   same
    ``"-0"``     ``"0"``     negative zero is zero; two spellings, one number
    ``"-0.0"``   ``"0"``     same
    ``"1e-3"``   ``"0.001"`` exponent notation expanded; ``1E-3`` would differ
    ``"+1"``     ``"1"``     leading plus removed
    ``"007"``    ``"7"``     leading zeros removed
    ``"1."``     *error*     an empty fraction is a typo, not a number
    ===========  ==========  ================================================

    Raises rather than guessing on anything else.
    """
    raw = str(value).strip()
    if raw.endswith(".") or raw.startswith("."):
        raise CanonicalizationError(
            f"{value!r} has an empty integer or fraction part. Decimal accepts "
            "it, which is exactly why it is refused here: a typo that parses is "
            "a typo that reaches production."
        )

    try:
        number = Decimal(raw)
    except (InvalidOperation, ValueError) as exc:
        raise CanonicalizationError(
            f"{value!r} is not a decimal. Fractional contract values are decimal "
            "strings; floats and free text are both refused."
        ) from exc

    if not number.is_finite():
        raise CanonicalizationError(
            f"{value!r} is not finite. NaN and infinity have no canonical form "
            "and no agreed cross-language spelling."
        )

    if number == 0:
        return "0"          # collapses -0, -0.0, 0.000

    text = format(number.normalize(), "f")
    if not _DECIMAL.match(text):  # pragma: no cover - defensive
        raise CanonicalizationError(
            f"{value!r} normalized to {text!r}, which is not canonical decimal form"
        )
    return text


def canonicalize(value: Any) -> Any:
    """Reduce a value to its canonical form.

    Recursive, total, and refusing anything it cannot represent unambiguously.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise CanonicalizationError(
            f"float {value!r} cannot appear in canonical form. Python, Go and "
            "JavaScript do not agree on how to print a double, so a float here "
            "produces hashes that match on one runtime and differ on another. "
            "Carry fractional values as decimal strings."
        )
    if isinstance(value, Decimal):
        return decimal_string(value)
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"map key {key!r} is not a string; key ordering would depend "
                    "on a comparison other languages define differently"
                )
            reduced = canonicalize(item)
            if reduced is None:
                continue  # absent and null are the same thing
            out[_text(key)] = reduced
        return {k: out[k] for k in sorted(out)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, (set, frozenset)):
        raise CanonicalizationError(
            "sets have no defined order; sort into a list at the point the "
            "ordering rule is known rather than leaving it to the hasher"
        )
    if hasattr(value, "canonical_form"):
        return canonicalize(value.canonical_form())
    raise CanonicalizationError(
        f"{type(value).__name__} has no canonical representation. Add one "
        "deliberately rather than letting the hasher guess."
    )


def canonical_json(value: Any) -> str:
    """The exact bytes another implementation must produce."""
    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    """Prefixed SHA-256 over the canonical bytes."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}:{digest}"


def sorted_unique(items: Sequence[Any], *, key) -> list:
    """Deduplicate and order, in that order.

    Duplicates are **normalized, not rejected**. Two independent parts of a plan
    legitimately reference the same artifact, and refusing that would push
    deduplication into every caller — where some would forget. Two handles that
    differ in any canonical field are different handles and both survive.
    """
    seen = {}
    for item in items:
        seen.setdefault(key(item), item)
    return [seen[k] for k in sorted(seen)]
