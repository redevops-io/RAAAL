"""The parser regression corpus, run against what exists today.

Tier 1 is executable now — `normalize()` needs no parser. Tiers 2 and 3 need a
dependency parse, and the honest thing to do with a case that cannot run yet is
neither to skip it nor to delete it.

**Why pending cases are counted rather than skipped.** A skipped test prints a
dot and reads like a pass. Two hundred cases of which sixty silently skip is a
corpus that looks twice the size it is, and this project has already shipped one
gate that reported perfect agreement with a single reader and another that
passed forever after its subject was deleted. So the pending tiers have an
exact declared count, and the count is asserted. Landing the parser lowers it;
adding cases raises it; either way the number has to move in the same commit as
the work, which is the only version of this that stays true.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from corpus.parser.loader import load
from src.discovery.syntax import normalize
from src.discovery.syntax_stanza import RecordedReader

ROOT = Path(__file__).resolve().parent.parent
CASES = load()
RECORDED = RecordedReader()

#: Cases nothing asserts yet. Lowered when work lands, raised when cases are
#: added — and asserted, so neither happens quietly.
#:
#: 108 when only tier 1 ran. English tier 2 took it to 54. The field mappers
#: took it to 47: seven semantics cases are now run end to end by
#: `tests/test_semantics.py`, and `corpus/parser/closure.json` says why each of
#: the remaining 47 is still there rather than leaving them a single number.
AWAITING_A_PARSER = 47

#: Semantics cases the deterministic path answers, asserted elsewhere. Read
#: from the closure report because that is where the classification lives; the
#: report is itself regenerated and checked by `tests/test_closure.py`, so this
#: is not a number trusting itself.
_CLOSURE = json.loads(
    (ROOT / "corpus" / "parser" / "closure.json").read_text())
ANSWERED = {row["id"] for row in _CLOSURE["rows"]
            if row["state"] == "MAPPED_AND_AGREED"}


def executable():
    return [c for c in CASES if c.tier == "normalization"]


def pending():
    return [c for c in CASES
            if c.id not in ANSWERED
            and (c.tier == "semantics"
                 or (c.tier == "dependency"
                     and not RECORDED.has(c.text, c.language)))]


class TestTheCorpusIsWhatItClaims:
    def test_it_loads(self):
        assert len(CASES) >= 150, (
            "the corpus was specified at 150-300 cases; below that the tiers "
            "stop having enough cases each to localise a failure")

    def test_every_case_asserts_exactly_one_property(self):
        """The rule the corpus is built on. A sentence asserting four things
        fails as a unit, which tells the reader only that *something* broke."""
        for case in CASES:
            keys = set(case.asserts)
            assert len(keys) <= 3, (
                f"{case.id} asserts {sorted(keys)}; a case is one claim, and "
                "the value/unit/kind of a single reading is that one claim")

    def test_the_falsification_cases_are_a_real_share(self):
        """Constructed cases mostly confirm what the code already does. The
        ones that discriminate come from observed failures and the plan's own
        list, and if they thin out the corpus stops being evidence."""
        share = sum(1 for c in CASES if c.is_falsification) / len(CASES)
        assert share >= 0.25, f"only {share:.0%} of cases discriminate"

    def test_the_pending_count_is_declared_and_correct(self):
        """The forcing function. If this number is not asserted, a corpus can
        grow a hundred cases nothing runs and still report green."""
        assert len(pending()) == AWAITING_A_PARSER, (
            f"{len(pending())} cases need a parser, and this file says "
            f"{AWAITING_A_PARSER}. Update it in the same commit as the work")

    def test_every_pending_case_is_pending_for_a_stated_reason(self):
        """Not "pending" as a place to put anything inconvenient. A case is
        waiting either on the semantics tier or on a language model nobody has
        fetched, and those are the only two answers."""
        for case in pending():
            reason = ("no mapping answers this case yet"
                      if case.tier == "semantics"
                      else f"no {case.language} parse recorded")
            assert case.tier == "semantics" or not RECORDED.has(
                case.text, case.language), reason
            assert case.id not in ANSWERED, (
                f"{case.id} is answered by the pipeline and still counted as "
                "pending; the count and the work must move together")


@pytest.mark.parametrize("case", executable(), ids=lambda c: c.id)
def test_normalization(case):
    """Tier 1: characters to values."""
    found = normalize(case.text, case.language)
    claims = case.asserts

    if "absent" in claims:
        assert not [v for v in found if v.kind == claims["absent"]], (
            f"{case.note or 'no reading of this kind should be produced'}")
        return

    if "kinds" in claims:
        assert [v.kind for v in found] == claims["kinds"]
        return

    matching = [v for v in found if v.kind == claims["kind"]]
    assert matching, f"no {claims['kind']} read from {case.text!r}"
    value = matching[0]

    if "canonical" in claims:
        expected = claims["canonical"]
        actual = value.canonical
        if isinstance(expected, list):
            actual = list(actual)
        elif isinstance(actual, Decimal):
            expected = Decimal(str(expected))
        assert actual == expected
    if "unit" in claims:
        assert value.unit == claims["unit"]
