"""A family this build does not model is refused because it was recognised.

The live lane found "tilt 20% toward small cap value" executing on two draws
of five under gpt-5.4 and refusing on the other three, and "hold my age in
bonds" executing on all three. Both are declared `REFUSED_BY_NAME` in
`corpus/parser/strategy_families.json` with cited definitions.

**The gate had been green for an accidental reason.** Under gpt-4.1 both
sentences refused every time — not because anything recognised a factor tilt,
but because gpt-4.1 also reported a `portfolio_sleeves` relation, which is
unsupported, so nothing compiled. gpt-5.4 sometimes omits that relation, and
what remains — a holding and a percentage — is an ordinary accumulation plan.

    A refusal is only robust if it is caused by the semantic we intend to
    refuse, not by some unrelated field that happens to fail first.

So the semantic is detected directly, by a deterministic reader over the parse
and a cited vocabulary, and the refusal names the family. The invariant:

    a known material unsupported semantic detected by any trusted witness
        -> the hosted model omitting it cannot make the sentence executable
        -> refused by name

**Model omission is the case under test**, not model agreement. Every test
below that matters runs the family witness with a model reading that says
nothing about the family — because that is the state that produced the
failure, and a test where both witnesses agree would pass against an
implementation that still depends on the model.
"""
from __future__ import annotations

import os

import pytest

from src.discovery.vocabulary import UNSUPPORTED_FAMILIES

DECLARED = {"QUANTIFY_PILOT_READER": "recorded",
            "QUANTIFY_PARSER_MODE": "RUNTIME",
            "QUANTIFY_PARSER_MODEL": "claude-sonnet-5",
            "ANTHROPIC_API_KEY": "unused",
            "PILOT_DATA_POLICY": "SYNTHETIC_ONLY",
            "QUANTIFY_SYNTAX_WITNESS": "yes"}

FACTOR_TILT = ("tilt 20% toward small cap value",
               "overweight value",
               "add a quality tilt",
               "underweight growth",
               "tilt toward momentum")

AGE_BASED = ("hold my age in bonds",
             "increase bonds as I get older",
             "reduce equity exposure over time",
             "follow a glide path",
             "use a target date allocation")

#: Sentences that must keep working. A detector that refuses these has traded
#: one silent failure for a loud one, and the loud one is on the supported
#: path where people actually are.
SUPPORTED = ("invest $500 monthly into VTI",
             "put 60% in VTI and 40% in BND",
             "hold 40% in bonds and 60% in stocks",
             "buy VTI every month",
             "invest $1000 a month split equally between VTI and BND")


@pytest.fixture()
def declared(monkeypatch):
    from src.deploy import context as deploy_context

    settings = deploy_context.resolve({**os.environ, **DECLARED})
    monkeypatch.setattr(deploy_context, "current", lambda: settings)
    return settings


def _parse(text):
    from src.discovery.syntax_stanza import RecordedReader, StanzaReader

    try:
        return RecordedReader().parse(text)
    except Exception:                                          # noqa: BLE001
        return StanzaReader().parse(text)


def _detected(text):
    from src.discovery.derived_readers import unsupported_family

    return {p.value for p in unsupported_family((), _parse(text), text)}


# --- the vocabulary is argued, not guessed -----------------------------------

def test_every_family_cites_its_definition():
    """A family without a source is a family somebody guessed at.

    Both of these came from `strategy_families.json`, which carries a cited
    definition per case; a word list assembled from memory would refuse
    sentences nobody can show are unsupported.
    """
    for name, family in UNSUPPORTED_FAMILIES.items():
        assert family.source.startswith("http"), f"{name} cites no source"
        assert len(family.why.split()) >= 8, f"{name}: {family.why!r}"
        assert family.dimension == name


def test_every_family_is_a_dimension_nothing_consumes():
    """The refusal names it, so it has to exist — and stay unconsumed.

    If Mission ever started reading one of these, the sentence would compile
    and the refusal would vanish silently.
    """
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.mission.from_intent import NOT_EXECUTABLE, READ_DIRECTLY

    for name in UNSUPPORTED_FAMILIES:
        assert QUANTIFY_SCHEMA.dimension(name) is not None, f"{name} is not a dimension"
        assert name not in READ_DIRECTLY, f"{name} is read by the compiler"
        assert name not in NOT_EXECUTABLE, (
            f"{name} is exempt from the stranded check, so a sealed intent "
            "carrying it would compile")


def test_the_families_are_not_asked_of_the_model():
    """Asking reintroduces the dependency the detection removes.

    Measured, not theorised: with `factor_tilt` in the prompt, gpt-5.4
    proposed its own value, disagreed with the witness that detected it, and
    the family became an open question — the page asked "what is your factor
    tilt?", inviting an answer for something this build does not model.
    """
    from src.discovery.schema import QUANTIFY_SCHEMA

    for name in UNSUPPORTED_FAMILIES:
        assert QUANTIFY_SCHEMA.dimension(name).asked is False, (
            f"{name} is offered to the hosted reader")


# --- detection ---------------------------------------------------------------

@pytest.mark.parametrize("text", FACTOR_TILT)
def test_a_factor_tilt_is_detected(text):
    assert "factor_tilt" in _detected(text), text


@pytest.mark.parametrize("text", AGE_BASED)
def test_an_age_based_allocation_is_detected(text):
    assert "age_based_allocation" in _detected(text), text


@pytest.mark.parametrize("text", SUPPORTED)
def test_a_supported_sentence_is_not_detected(text):
    assert not _detected(text), f"{text!r} was read as an unsupported family"


@pytest.mark.parametrize("text", [
    "I am overweight and want to retire early",
    "hold 40% in value stocks",
    "my growth has been good this year",
    "I hold large cap index funds",
])
def test_the_pairing_rule_keeps_it_from_firing_on_ordinary_english(text):
    """A tilt word alone, or a style word alone, is not a tilt.

    `overweight` is ordinary English about a person and Stanza tags it ADJ
    everywhere, so predicate position cannot separate the two senses. The
    marker and a named factor together can.
    """
    assert not _detected(text), f"{text!r} was read as a factor tilt"


def test_a_sentence_naming_two_families_reports_both():
    """Refusing by one name tells somebody half of why it will not run."""
    found = _detected("hold my age in bonds and tilt toward value")
    assert found == {"factor_tilt", "age_based_allocation"}, found


# --- the invariant: model omission cannot make it executable -----------------

class _Reading:
    def __init__(self, dimension, value, source_span=""):
        self.dimension, self.value, self.source_span = dimension, value, source_span


class _Model:
    """A hosted reading that says nothing about any family — the failing case."""

    reader_id = "model@test"
    relations = ()
    unread = ()

    def __init__(self, readings=()):
        self.readings = list(readings)


def _read_with(text, model_readings):
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.witnesses import BOTH
    from src.workspace.pilot import read

    class _Reader:
        id = "model@test"
        def read(self, _text, _schema):
            from src.discovery.reader import ReadingSet
            return ReadingSet(reader_id="model@test",
                              readings=tuple(model_readings))

    class _Syntax:
        id = "recorded@test"
        def parse(self, t):
            return _parse(t)

    return read(text, _Reader(), schema=QUANTIFY_SCHEMA, profile=BOTH,
                syntax_reader=_Syntax())


@pytest.mark.parametrize("text", FACTOR_TILT[:3])
def test_a_silent_model_cannot_make_a_factor_tilt_executable(declared, text):
    """The invariant, on the exact state that failed.

    The model reports an ordinary accumulation plan and nothing about the
    family — which is what gpt-5.4 did on the executing draws. It must still
    refuse, and refuse by name.
    """
    reading = _read_with(text, [
        _Reading("objective", "evaluate_investment_strategy"),
        _Reading("assets", "VTI"),
        _Reading("amount", "500"),
        _Reading("cadence", "monthly"),
    ])
    assert not reading.executable, (
        f"{text!r} compiled while the model said nothing about the family")
    assert "factor_tilt" in {r.dimension for r in reading.refusals}, (
        f"refused by {[r.dimension for r in reading.refusals]}, not by name")


@pytest.mark.parametrize("text", AGE_BASED[:3])
def test_a_silent_model_cannot_make_an_age_based_allocation_executable(declared, text):
    reading = _read_with(text, [
        _Reading("objective", "evaluate_investment_strategy"),
        _Reading("assets", "BND"),
        _Reading("amount", "500"),
        _Reading("cadence", "monthly"),
    ])
    assert not reading.executable, text
    assert "age_based_allocation" in {r.dimension for r in reading.refusals}


def test_the_refusal_does_not_depend_on_any_other_dimension_failing(declared):
    """The lesson, asserted directly.

    The old refusal appeared only because `portfolio_sleeves` also failed.
    Here the model reports a complete, compilable accumulation plan — every
    dimension settled, nothing else to object to — and the family refusal must
    still be the outcome.
    """
    reading = _read_with("tilt 20% toward small cap value", [
        _Reading("objective", "evaluate_investment_strategy"),
        _Reading("assets", "VTI, BND"),
        _Reading("amount", "500"),
        _Reading("cadence", "monthly"),
        _Reading("allocation_method", "equal_weight_at_purchase"),
        _Reading("day_rule", "first_session_of_period"),
        _Reading("account_type", "taxable"),
    ])
    assert not reading.executable
    assert {r.dimension for r in reading.refusals} == {"factor_tilt"}, (
        f"refused by {[r.dimension for r in reading.refusals]}; the family "
        "refusal must be the answer, not one of several")


def test_the_refusal_asks_nothing(declared):
    """No answer makes an unsupported family runnable.

    Asking "which holdings did you mean?" beside "we do not model factor
    tilts" reads as two problems of which one might be fixable.
    """
    reading = _read_with("tilt 20% toward small cap value", [
        _Reading("objective", "evaluate_investment_strategy")])
    assert not reading.questions, f"asked {list(reading.questions)}"


def test_the_refusal_says_why_in_words_a_person_can_read(declared):
    reading = _read_with("hold my age in bonds", [
        _Reading("objective", "evaluate_investment_strategy")])
    detail = next(r.detail for r in reading.refusals
                  if r.dimension == "age_based_allocation")
    assert "does not model" in detail
    assert len(detail.split()) >= 12, detail


# --- and the supported path still runs ---------------------------------------

@pytest.mark.parametrize("text", SUPPORTED[:3])
def test_a_supported_strategy_still_executes(declared, text):
    """The other half. A detector that refuses everything passes every test
    above and ships a product that does nothing."""
    reading = _read_with(text, [
        _Reading("objective", "evaluate_investment_strategy"),
        _Reading("assets", "VTI, BND"),
        _Reading("amount", "500"),
        _Reading("cadence", "monthly"),
        _Reading("allocation_method", "equal_weight_at_purchase"),
    ])
    assert not [r for r in reading.refusals
                if r.dimension in UNSUPPORTED_FAMILIES], (
        f"{text!r} was refused as an unsupported family")


def test_an_explicit_static_bond_allocation_still_executes(declared):
    """"40% in bonds" is a static allocation and must not be read as a
    glidepath. The two differ by whether the number changes over time, and
    the age vocabulary must not collect the ordinary case."""
    reading = _read_with("hold 40% in bonds and 60% in stocks", [
        _Reading("objective", "evaluate_investment_strategy"),
        _Reading("assets", "BND, VTI"),
        _Reading("stated_weights", "40/60"),
        _Reading("amount", "500"),
        _Reading("cadence", "monthly"),
    ])
    assert not [r for r in reading.refusals if r.dimension in UNSUPPORTED_FAMILIES]


# --- what this cannot do -----------------------------------------------------

def test_the_single_witness_profile_cannot_detect_a_family(declared):
    """Stated so the green is not read as more than it is.

    `unsupported_family` reads the parse. MODEL_ONLY has no parse, so a
    deployment serving one witness has nothing that can say "this is a factor
    tilt" when the model does not — which is precisely the omission this
    closes. The serving profile is BOTH. Asserted rather than commented,
    because a limitation nobody can see is a limitation somebody will assume
    away.
    """
    from src.discovery.derived_readers import unsupported_family

    assert unsupported_family((), None, "tilt 20% toward small cap value") == ()
