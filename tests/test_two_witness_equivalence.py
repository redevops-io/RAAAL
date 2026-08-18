"""The model proposes; syntax argues — asserted directly, on both lanes.

The two-witness profile is asymmetric on purpose, and the asymmetry is product
semantics rather than an implementation detail:

    model + syntax agree         settle, syntax recorded as supporting evidence
    model + syntax contradict    unresolved: the words argue with the reading
    model silent, syntax speaks  unresolved: syntax alone never carries a field
    model speaks, syntax silent  settle: silence is not an argument

The runtime's default stance is that no reader is privileged, which is right
for two peers and wrong here — a deterministic candidate is evidence *about*
the model's reading, not a competing reading. Handing syntax to the runtime as
a second reader would let it carry a field alone, which is a different product.

**The four rows are asserted one by one rather than inferred from corpus
behaviour.** A corpus run exercises whichever combinations its sentences happen
to contain, and the two rows that matter most — contradiction, and syntax
speaking alone — are the rarest. Inferring the policy from aggregate agreement
is how it would be lost.
"""
from __future__ import annotations

import os

import pytest

from src.discovery import adapter

DECLARED = {"QUANTIFY_PILOT_READER": "recorded",
            "QUANTIFY_PARSER_MODE": "RUNTIME",
            "QUANTIFY_PARSER_MODEL": "claude-sonnet-5",
            "ANTHROPIC_API_KEY": "unused",
            "PILOT_DATA_POLICY": "SYNTHETIC_ONLY",
            "QUANTIFY_SYNTAX_WITNESS": "yes"}


class _Reading:
    def __init__(self, dimension, value, source_span=""):
        self.dimension = dimension
        self.value = value
        self.source_span = source_span


class _ModelReading:
    ok = True
    failed = ()
    relations = ()

    def __init__(self, *readings, reader_id="model@1"):
        self.readings = list(readings)
        self.reader_id = reader_id


class _Candidate:
    def __init__(self, dimension, proposed_value, source_span=""):
        self.dimension = dimension
        self.proposed_value = proposed_value
        self.source_span = source_span


def _fused(model, syntax):
    """One utterance through the runtime lane's two-witness wiring."""
    reading, contradicted = adapter.two_witness_readings(model, syntax)
    from discovery_runtime import Proposal, fuse

    modes = adapter.compare_modes()
    out = {}
    dimensions = set(reading.payload) | set(reading.evidence)
    for name in sorted(dimensions):
        proposals = ([Proposal(value=reading.payload[name], reader_id="model")]
                     if name in reading.payload else [])
        out[name] = fuse(name, proposals, mode=modes.get(name, "TEXT"),
                         normalizers=adapter.NORMALIZERS,
                         contradicted_by=contradicted.get(name))
    return out


def test_model_and_syntax_agreeing_settles_with_syntax_as_evidence():
    """Row one. Agreement settles, and the record says two witnesses concurred."""
    model = _ModelReading(_Reading("cadence", "monthly", "monthly"))
    syntax = {"cadence": [_Candidate("cadence", "monthly", "every month")]}

    reading, contradicted = adapter.two_witness_readings(model, syntax)
    assert not contradicted
    assert reading.payload["cadence"] == "monthly"

    witnesses = [e.reader_id for e in reading.evidence["cadence"]]
    assert "syntax" in witnesses and "model@1" in witnesses, (
        f"agreement recorded only {witnesses}; supporting evidence is the "
        "point of running a second witness at all")
    assert _fused(model, syntax)["cadence"].proceeds


def test_model_and_syntax_contradicting_does_not_settle():
    """Row two. The words argue with the reading, so nobody wins by default."""
    model = _ModelReading(_Reading("cadence", "monthly", "monthly"))
    syntax = {"cadence": [_Candidate("cadence", "annual", "every year")]}

    _, contradicted = adapter.two_witness_readings(model, syntax)
    assert "cadence" in contradicted, "a contradiction was not recorded"

    decision = _fused(model, syntax)["cadence"]
    assert not decision.proceeds, (
        "a contradicted value settled; the model would be deciding against the "
        "sentence it was reading")


def test_syntax_alone_never_carries_a_field():
    """Row three, and the one the asymmetry exists for.

    If this ever settles, syntax has become a peer proposal source and the
    product has changed: a deterministic candidate could establish a field the
    model never read.
    """
    model = _ModelReading(_Reading("cadence", "monthly", "monthly"))
    syntax = {"assets": [_Candidate("assets", "VTI", "VTI")]}

    reading, _ = adapter.two_witness_readings(model, syntax)
    assert "assets" not in reading.payload, (
        "syntax put a value in the payload; it argues, it does not propose")

    decision = _fused(model, syntax)["assets"]
    assert not decision.proceeds, (
        "syntax alone settled a field the model never read")
    assert reading.evidence.get("assets"), (
        "what syntax said was dropped entirely — it must be recorded even "
        "though it cannot settle, or the question cannot say what prompted it")


def test_syntax_silent_does_not_block_the_model():
    """Row four. Silence is not an argument."""
    model = _ModelReading(_Reading("cadence", "monthly", "monthly"))

    reading, contradicted = adapter.two_witness_readings(model, {})
    assert not contradicted
    assert reading.payload["cadence"] == "monthly"
    assert _fused(model, {})["cadence"].proceeds, (
        "a silent second witness blocked a reading, which would make the "
        "deterministic path a veto rather than a witness")


@pytest.fixture()
def two_witness(monkeypatch):
    from src.deploy import context as deploy_context

    settings = deploy_context.resolve({**os.environ, **DECLARED})
    monkeypatch.setattr(deploy_context, "current", lambda: settings)
    return settings


def test_the_internal_lane_agrees_on_all_four(two_witness):
    """The same four rows through Quantify's own fusion.

    Compared as *outcomes* rather than by calling the same function twice: the
    point is that both implementations of the policy agree, and a test that ran
    one implementation would prove only that it is self-consistent.
    """
    from src.discovery.fusion import Proposal, fuse
    from src.discovery.syntax import SyntaxEvidence

    def internal(model_value, syntax_value):
        model = (Proposal(dimension="cadence", value=model_value,
                          reader_id="model@1", source_span="span")
                 if model_value is not None else None)
        syntax = ((SyntaxEvidence(dimension="cadence",
                                  proposed_value=syntax_value, score=1,
                                  features={}, source_span="span",
                                  sentence_id="s", parser="p", model="m",
                                  scoring_version="v"),)
                  if syntax_value is not None else ())
        return fuse("cadence", model=model, syntax=syntax,
                    available=("cadence",))

    assert internal("monthly", "monthly").proceeds, "agreement did not settle"
    assert not internal("monthly", "annual").proceeds, (
        "a contradiction settled on the internal lane")
    assert not internal(None, "monthly").proceeds, (
        "syntax alone carried a field on the internal lane")
    assert internal("monthly", None).proceeds, (
        "a silent syntax witness blocked the model on the internal lane")


def test_the_two_lanes_reach_the_same_verdict_on_each_row(two_witness):
    """Old and new, row by row, on whether the dimension settles.

    Whether a dimension settles is what a person experiences: a settled one
    runs, an unsettled one becomes a question. Agreement here is the property
    the migration has to preserve.
    """
    from src.discovery.fusion import Proposal, fuse
    from src.discovery.syntax import SyntaxEvidence

    rows = [("monthly", "monthly", True),
            ("monthly", "annual", False),
            (None, "monthly", False),
            ("monthly", None, True)]

    for model_value, syntax_value, expected in rows:
        old = fuse(
            "cadence",
            model=(Proposal(dimension="cadence", value=model_value,
                            reader_id="model@1", source_span="span")
                   if model_value is not None else None),
            syntax=((SyntaxEvidence(dimension="cadence",
                                    proposed_value=syntax_value, score=1,
                                    features={}, source_span="span",
                                    sentence_id="s", parser="p", model="m",
                                    scoring_version="v"),)
                    if syntax_value is not None else ()),
            available=("cadence",))

        model = (_ModelReading(_Reading("cadence", model_value, "span"))
                 if model_value is not None else _ModelReading())
        syntax = ({"cadence": [_Candidate("cadence", syntax_value, "span")]}
                  if syntax_value is not None else {})
        new = _fused(model, syntax).get("cadence")

        assert old.proceeds is expected, (
            f"internal disagreed with the policy on "
            f"model={model_value!r} syntax={syntax_value!r}")
        assert new is not None and new.proceeds is expected, (
            f"runtime disagreed on model={model_value!r} "
            f"syntax={syntax_value!r}: {None if new is None else new.detail}")
