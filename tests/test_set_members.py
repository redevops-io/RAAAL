"""Two readings for one SET dimension are members, not witnesses.

Given "take from bonds in a down year and from stocks otherwise" the recorded
reader emits two `assets` readings — `'bonds'` and `'stocks'`, each with its own
span. Both lanes got that wrong, in opposite directions:

    internal   `{p.dimension: p for p in proposals}` kept the last, so a plan
               for a sentence naming both mentioned only stocks. Silent.
    runtime    read two readings from one reader as a disagreement and asked
               which the person meant — of a reader disagreeing with itself.

The rule: a reader emits one semantic value per SET dimension, and where it
emits several members they are unioned into the one reading it should have
sent.

**Only the membership.** The conditional meaning in that sentence — take from
bonds *in a down year* — belongs to `sell_action` and stays there. Nothing here
infers a rule from the multiplicity, and if the condition cannot be represented
it is that dimension which must clarify or refuse, never the asset set.

Both old failures are asserted directly, because a test that only checks the
result would pass against a third implementation that lost `'bonds'` some other
way.
"""
from __future__ import annotations

import os

import pytest

from src.discovery import adapter

CONDITIONAL = "take from bonds in a down year and from stocks otherwise"

DECLARED = {"QUANTIFY_PILOT_READER": "recorded",
            "QUANTIFY_PARSER_MODE": "RUNTIME",
            "QUANTIFY_PARSER_MODEL": "claude-sonnet-5",
            "ANTHROPIC_API_KEY": "unused",
            "PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}


class _Reading:
    def __init__(self, dimension, value, source_span=""):
        self.dimension = dimension
        self.value = value
        self.source_span = source_span


def test_set_members_are_unioned_into_one_reading():
    out = adapter.one_reading_per_set_dimension([
        _Reading("assets", "bonds", "bonds"),
        _Reading("assets", "stocks", "stocks"),
    ])
    assert len(out) == 1, f"expected one assets reading, got {len(out)}"
    assert set(out[0].value.split(", ")) == {"bonds", "stocks"}


def test_last_wins_would_fail_this():
    """The internal lane's old behaviour, asserted as unacceptable."""
    out = adapter.one_reading_per_set_dimension([
        _Reading("assets", "bonds"), _Reading("assets", "stocks")])
    assert "bonds" in out[0].value, (
        "the first member was dropped — last-wins loses an asset the sentence "
        "names, silently")


def test_a_scalar_dimension_is_not_unioned():
    """Two values for a scalar dimension genuinely compete.

    Unioning them would invent a value nobody said. Only SET members merge, and
    the schema decides which dimensions those are.
    """
    out = adapter.one_reading_per_set_dimension([
        _Reading("cadence", "monthly"), _Reading("cadence", "annual")])
    assert len(out) == 2, "a scalar dimension was collapsed"
    assert {r.value for r in out} == {"monthly", "annual"}


def test_one_member_is_left_alone():
    out = adapter.one_reading_per_set_dimension([_Reading("assets", "VTI")])
    assert len(out) == 1 and out[0].value == "VTI"


def test_already_complete_sets_are_not_re_split():
    """A reader that did the right thing is not punished for it."""
    out = adapter.one_reading_per_set_dimension(
        [_Reading("assets", "VTI, BND")])
    assert set(out[0].value.split(", ")) == {"VTI", "BND"}


@pytest.fixture()
def resolved(monkeypatch):
    from src.deploy import context as deploy_context

    settings = deploy_context.resolve({**os.environ, **DECLARED})
    monkeypatch.setattr(deploy_context, "current", lambda: settings)
    return settings


def test_neither_lane_loses_an_asset_the_sentence_names(resolved):
    """End to end, on the sentence that found this.

    Both lanes, because the two failed differently and a fix to one would leave
    the other wrong while this file still passed if it checked only one.
    """
    from tests.test_corpus_equivalence import (_internal_reading,
                                               _runtime_intent, from_internal,
                                               from_runtime)

    internal = from_internal(_internal_reading(CONDITIONAL))
    runtime = from_runtime(_runtime_intent(CONDITIONAL))

    for name, state in (("internal", internal), ("runtime", runtime)):
        assets = state.settled_fields.get("assets", "")
        assert "bonds" in assets and "stocks" in assets, (
            f"{name} settled assets={assets!r}; the sentence names both")


def test_the_runtime_does_not_report_a_reader_disagreeing_with_itself(resolved):
    """The other old failure. One reader is one witness."""
    from tests.test_corpus_equivalence import _runtime_intent

    intent = _runtime_intent(CONDITIONAL)
    for open_dimension in intent.unresolved:
        assert open_dimension.dimension != "assets", (
            f"assets was reported open: {open_dimension.detail!r}. Two readings "
            "from one reader for a SET dimension are members, not a "
            "disagreement.")


def test_the_conditional_meaning_is_not_inferred_from_the_membership(resolved):
    """The restriction that keeps the union honest.

    "take from bonds in a down year" carries a condition. It belongs to
    `sell_action`; the asset set must not grow a rule because two members
    appeared, and must not be refused because a rule elsewhere is hard.
    """
    from tests.test_corpus_equivalence import _internal_reading, from_internal

    state = from_internal(_internal_reading(CONDITIONAL))
    assert "sell_action" in state.settled_fields, (
        "the conditional clause is no longer carried by sell_action")
    assert "down year" not in state.settled_fields.get("assets", ""), (
        "the condition leaked into the asset set")
