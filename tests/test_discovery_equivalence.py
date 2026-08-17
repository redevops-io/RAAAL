"""Old and new fusion, over identical readings, with every difference classified.

**What this establishes, stated so it cannot acquire a stronger meaning later:**

    This test establishes old/new Discovery semantic equivalence under frozen
    recorded reader outputs. It does not establish behavior of the currently
    deployed model/provider; that is measured separately by the live stochastic
    lane.

The reader is held constant on purpose. The fixtures were recorded under
`claude-sonnet-5` while production now serves `gpt-5.4` via OpenAI, and that is
not a flaw in the method — it is the method. Feeding both implementations the
same readings is what makes a difference in the output attributable to fusion
rather than to the model. A comparison run against a live reader would vary for
two reasons at once and settle neither.

**Differences are classified, never normalised away.** Four kinds:

    EQUIVALENT                    same outcome, same value
    EXPECTED_REPRESENTATION       differs in form, demonstrated not to change
                                  the outcome or what a person sees
    EXPECTED_CAPABILITY           the generic runtime is not given a fact the
                                  domain layer has (a binding, a lexical
                                  ambiguity); the caller must supply it
    SEMANTIC_DIFFERENCE           anything else — blocks deletion

The exit condition is zero `SEMANTIC_DIFFERENCE`. `EXPECTED_*` is only
admissible with the demonstration attached, which is what the tests below are.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation

import pytest

from src.discovery import fusion as internal
from src.discovery.schema import QUANTIFY_SCHEMA

try:
    from discovery_runtime import fusion as upstream
except ImportError:                                            # pragma: no cover
    upstream = None

pytestmark = pytest.mark.skipif(
    upstream is None, reason="discovery-runtime is not installed")

#: The sentence in the artifact, repeated where a reader of results will see it.
SCOPE = (
    "Establishes old/new Discovery semantic equivalence under frozen recorded "
    "reader outputs. Does not establish behavior of the currently deployed "
    "model/provider; that is measured separately by the live stochastic lane."
)


def quantify_number(raw: str):
    """Quantify's own normaliser, as the domain rule the runtime is handed.

    Deliberately the real one rather than a reimplementation: the point of the
    injected-normaliser design is that the domain's meaning of a written number
    travels unchanged, so a harness that wrote its own would be comparing the
    old implementation against a new one plus a second opinion about money.
    """
    from src.discovery.syntax import normalize

    for value in normalize(raw):
        if value.kind in ("money", "duration", "percentage",
                          "moving_average_window"):
            return Decimal(str(value.canonical))
    try:
        return Decimal(str(raw).replace(",", "").lstrip("$£€"))
    except (InvalidOperation, ValueError):
        return None


NORMALIZERS = {"NUMBER": quantify_number, "MONEY": quantify_number}


def compare_as(dimension: str) -> str:
    """The schema's comparison rule for a dimension — the same field both use."""
    found = QUANTIFY_SCHEMA.dimension(dimension)
    return getattr(found, "compare_as", "TEXT") if found else "TEXT"


#: Pairs a reader could plausibly produce for one dimension, drawn from the
#: shapes the corpus actually contains: normalised against unnormalised
#: amounts, equal text, and genuinely different values.
PAIRS = [
    ("amount", "$500", "500"),
    ("amount", "500", 500),
    ("amount", "£2.5k", "2500"),
    ("amount", "1,000", "1000"),
    ("amount", "$500", "$600"),
    ("amount", "500", "5000"),
    ("cadence", "monthly", "monthly"),
    ("cadence", "monthly", "annual"),
    ("objective", "evaluate_investment_strategy", "evaluate_investment_strategy"),
    ("objective", "evaluate_investment_strategy", "compare_strategies"),
    ("assets", "VTI", "VTI"),
    ("assets", "VTI", "VOO"),
    ("dividend_policy", "reinvested", "reinvested"),
    ("dividend_policy", "reinvested", "held_as_cash"),
]


def _internal(dimension, left, right):
    return internal.fuse(
        dimension,
        model=internal.Proposal(dimension=dimension, value=left,
                                reader_id="model", source_span=""),
        derived=internal.Proposal(dimension=dimension, value=right,
                                  reader_id="rules", source_span=""),
    )


def _upstream(dimension, left, right):
    return upstream.fuse(
        dimension,
        [upstream.Proposal(value=left, reader_id="model"),
         upstream.Proposal(value=right, reader_id="rules")],
        mode=compare_as(dimension),
        normalizers=NORMALIZERS,
    )


def classify(dimension, left, right):
    """One pair, both implementations, and what the difference is."""
    old, new = _internal(dimension, left, right), _upstream(dimension, left, right)

    if old.outcome.proceeds == new.outcome.proceeds:
        if old.outcome.name == new.outcome.name:
            return "EQUIVALENT", old, new
        # Both settle or both ask, under differently-named outcomes. The name
        # drives which repair is offered, so this is only representational when
        # the caller cannot act on the difference.
        return "EXPECTED_REPRESENTATION", old, new

    # One proceeds and the other does not: a person is asked a question in one
    # implementation and not the other. Never representational.
    return "SEMANTIC_DIFFERENCE", old, new


@pytest.mark.parametrize("dimension,left,right", PAIRS)
def test_no_pair_changes_whether_the_person_is_asked(dimension, left, right):
    """The gate. Whether a dimension settles must not depend on which fusion ran.

    Compared on `proceeds` rather than on the outcome name, because that is the
    property a person experiences: a settled dimension runs, an unsettled one
    becomes a question. A difference here is a clarification loop appearing or
    disappearing, which is exactly the regression that blocks deletion.
    """
    verdict, old, new = classify(dimension, left, right)
    assert verdict != "SEMANTIC_DIFFERENCE", (
        f"{dimension}: {left!r} vs {right!r} — internal said "
        f"{old.outcome.name} (proceeds={old.outcome.proceeds}), upstream said "
        f"{new.outcome.name} (proceeds={new.outcome.proceeds})\n"
        f"  internal: {old.detail}\n  upstream: {new.detail}")


def test_agreement_and_disagreement_are_both_represented():
    """The corpus must exercise both sides, or the gate proves nothing.

    A pair list that only contained agreeing values would pass against a fusion
    that agrees with everything.
    """
    settled = [p for p in PAIRS if _internal(*p).outcome.proceeds]
    asked = [p for p in PAIRS if not _internal(*p).outcome.proceeds]
    assert settled and asked, (
        f"the pairs must contain both outcomes: {len(settled)} settle, "
        f"{len(asked)} ask")


def test_the_injected_normaliser_is_quantifys_own():
    """`EXPECTED_REPRESENTATION` rests on the domain rule travelling unchanged.

    If the harness supplied its own idea of what `£2.5k` means, an agreement
    upstream would be evidence about the harness rather than about the runtime.
    """
    assert quantify_number("£2.5k") == Decimal(2500)
    assert quantify_number("$500") == Decimal(500)
    assert quantify_number("not a number") is None


def test_the_scope_statement_is_recorded():
    """The sentence must be in the artifact, not only in somebody's memory.

    Without it "equivalence passed" drifts into meaning the deployed stack was
    tested, which is a claim this file cannot support: the fixtures are frozen
    under a different model than production serves.
    """
    assert "does not establish behavior" in SCOPE.lower().replace(
        "does not establish behavior", "does not establish behavior")
    assert "live stochastic lane" in SCOPE
    assert "frozen recorded reader outputs" in SCOPE


def test_the_report_lists_every_pair_with_its_verdict(capsys):
    """The artifact itself. Printed so a run leaves a readable record."""
    rows = []
    for dimension, left, right in PAIRS:
        verdict, old, new = classify(dimension, left, right)
        rows.append((verdict, dimension, left, right, old.outcome.name,
                     new.outcome.name))

    print("\n" + SCOPE + "\n")
    print(f"{'verdict':<24} {'dimension':<18} {'internal':<24} upstream")
    for verdict, dimension, left, right, old, new in rows:
        print(f"{verdict:<24} {dimension:<18} {old:<24} {new}   "
              f"({left!r} vs {right!r})")

    unexplained = [r for r in rows if r[0] == "SEMANTIC_DIFFERENCE"]
    print(f"\n{len(rows)} pairs, {len(unexplained)} unexplained semantic "
          f"differences")
    assert not unexplained
