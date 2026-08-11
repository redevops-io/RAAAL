"""Stage-1 recognition for vesting language. Reading only, never arithmetic.

Kept separate from the general compiler rules on purpose. These patterns run
only when the template hint has already fired, so a plan that never mentions
vesting cannot acquire a withholding rate because it happened to contain the
word "shares" — and the 144-strategy corpus, which is a specification rather
than a test, is not perturbed by patterns written for a different domain.

Every recognition carries the span it came from. A field whose span is not
verbatim in the text was not read from the text, and that distinction is what
lets the confirmation card separate "you said this" from "we assumed this".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: `(field, value-or-None, pattern)`. A `None` value means the pattern captures
#: it; a literal means the phrasing itself decides.
_RULES: Sequence[Tuple[str, Optional[Any], re.Pattern]] = (
    ("gross_shares", None, re.compile(
        r"\b(\d[\d,]*)\s+(?:[A-Z]{2,5}\s+)?shares?\b", re.IGNORECASE)),

    ("withholding_rate", None, re.compile(
        r"\bwithhold\w*\s+(?:about\s+)?(\d{1,2}(?:\.\d+)?)\s*%|"
        r"\b(\d{1,2}(?:\.\d+)?)\s*%\s+(?:is\s+)?withh?eld\b", re.IGNORECASE)),

    ("withholding_method", "SHARE_WITHHOLDING", re.compile(
        r"\bwithhold\w*[^.]{0,30}\bin shares\b|\bshares?\s+(?:are\s+)?withheld\b",
        re.IGNORECASE)),
    ("withholding_method", "SELL_TO_COVER", re.compile(
        r"\bsell[- ]to[- ]cover\b|\bsell\w*\s+(?:enough\s+)?to cover\b",
        re.IGNORECASE)),

    ("disposition_policy", "HOLD", re.compile(
        r"\b(?:hold|keep)\s+(?:the\s+)?(?:vested\s+)?shares?\b|"
        r"\bdo(?:n't| not)\s+sell\b|\bnever sell\b", re.IGNORECASE)),
    ("disposition_policy", "SELL_ALL_AND_DIVERSIFY", re.compile(
        r"\bsell\b[^.]{0,40}\b(?:as soon as|first eligible|immediately|"
        r"right away|straight away)\b|"
        r"\b(?:as soon as|first eligible|immediately)\b[^.]{0,40}\bsell\b",
        re.IGNORECASE)),
    ("disposition_policy", "SELL_HALF_AND_DIVERSIFY", re.compile(
        r"\bsell\s+half\b", re.IGNORECASE)),

    ("concentration_cap", None, re.compile(
        r"\b(?:keep|hold|stay|maintain)\b[^.]{0,60}?\b(?:below|under|beneath|"
        r"less than|no more than)\s+(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE)),

    ("cadence", "quarterly", re.compile(r"\bquarterly\b|\beach quarter\b|"
                                        r"\bevery quarter\b", re.IGNORECASE)),
    ("cadence", "monthly", re.compile(r"\bmonthly\b|\beach month\b|"
                                      r"\bevery month\b", re.IGNORECASE)),
    ("cadence", "annual", re.compile(r"\bannually\b|\beach year\b|"
                                     r"\bevery year\b", re.IGNORECASE)),
)

#: `60% VTI` or `VTI 60%`. Both orders, because both are written.
_WEIGHT = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*%\s+(?:in(?:to)?\s+|to\s+)?([A-Z]{2,5})\b"
    r"|([A-Z]{2,5})\s+(?:at\s+)?(\d{1,3}(?:\.\d+)?)\s*%")

#: Where the allocation clause starts. Weights before this are not allocation
#: weights — a withholding rate is a percentage too.
_ALLOCATION_CLAUSE = re.compile(
    r"\b(?:allocat\w+|invest\w*|put|split|diversif\w+)\b[^.]*", re.IGNORECASE)

_BLACKOUT = re.compile(
    r"\bblackout\b|\btrading window\b|\bafter earnings\b|\bclosed window\b",
    re.IGNORECASE)


@dataclass(frozen=True)
class RSURecognition:
    field: str
    value: Any
    span: str

    def to_json(self) -> Dict[str, Any]:
        return {"field": self.field, "value": self.value, "span": self.span}


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def recognize(text: str, *, assets: Sequence[str] = ()) -> List[RSURecognition]:
    """What the words say about a vest. No figure is computed here."""
    found: List[RSURecognition] = []
    seen: set = set()

    for field, literal, pattern in _RULES:
        match = pattern.search(text)
        if match is None or field in seen:
            continue
        seen.add(field)

        if literal is not None:
            found.append(RSURecognition(field, literal, match.group(0)))
            continue

        captured = next((g for g in match.groups() if g), None)
        if captured is None:
            continue
        value = _number(captured)
        if field == "concentration_cap":
            value = value / 100.0
        if field == "withholding_rate":
            value = value / 100.0
        found.append(RSURecognition(field, value, match.group(0)))

    allocation = _allocation_in(text)
    if allocation:
        found.append(RSURecognition("allocation_policy", allocation,
                                    _ALLOCATION_CLAUSE.search(text).group(0)))

    employer = _employer_in(text, assets, allocation)
    if employer:
        found.append(RSURecognition("employer_ticker", employer, employer))

    blackout = _BLACKOUT.search(text)
    if blackout:
        # Recognised, and deliberately not resolved to dates. "After earnings"
        # names a window whose dates the text does not give, and inventing them
        # would schedule a sale on a day nobody described.
        found.append(RSURecognition("blackout_declared", True,
                                    blackout.group(0)))
    return found


def _allocation_in(text: str) -> Optional[Dict[str, float]]:
    """Weights from the allocation clause only.

    Scanned across the whole sentence, a 22% withholding rate becomes a 22%
    allocation to whichever ticker follows it.
    """
    clause = _ALLOCATION_CLAUSE.search(text)
    if clause is None:
        return None

    weights: Dict[str, float] = {}
    for match in _WEIGHT.finditer(clause.group(0)):
        percent, ticker = ((match.group(1), match.group(2))
                           if match.group(1) else (match.group(4),
                                                   match.group(3)))
        if percent and ticker:
            weights[ticker] = _number(percent) / 100.0
    return weights or None


def _employer_in(text: str, assets: Sequence[str],
                 allocation: Optional[Mapping[str, float]]) -> Optional[str]:
    """The ticker the vest is *in*, not the ones proceeds go to.

    Returns None rather than guessing when several remain: naming the wrong
    employer would measure concentration against the wrong holding.
    """
    targets = set(allocation or {})
    candidates = [one for one in assets if one not in targets]
    return candidates[0] if len(candidates) == 1 else None
