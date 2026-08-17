"""Same witness, same author, same `intent_hash` — across both implementations.

`author` is inside `canonical_form` and therefore inside `intent_hash`, while
producer, span and evidence are not. So two implementations that classify the
same witness differently produce *different identities for the same request* —
and a five-boundary equivalence run would report that as a semantic mismatch
when it is really a classification gap.

The internal path already says `MODEL` for a hosted-model reading. `draft_intent`
stamps `READER` on everything, correctly, because a generic runtime knows a
reader produced the value and cannot know what kind. `adapter.classify_authors`
is the domain saying which kind, and these tests are what make that claim
checkable rather than asserted.

Run before the corpus harness, deliberately: an equivalence gate that fails for
a reason you already know how to fix teaches nothing and erodes the gate.
"""
from __future__ import annotations

import pytest

from runtime_contracts import (Author, DecisionEvidence, IntentField,
                               ReaderKind, VerifiedIntent)

from src.discovery import adapter

#: Witness kind -> the author both implementations must agree on.
EXPECTED = {
    ReaderKind.MODEL: Author.MODEL,
    ReaderKind.RULE: Author.READER,
    ReaderKind.RETRIEVAL: Author.READER,
    ReaderKind.PRIOR: Author.DEFAULT,
    ReaderKind.POLICY: Author.POLICY,
    ReaderKind.HUMAN: Author.USER,
}


def _evidence(kind, value="500"):
    return DecisionEvidence(reader_id="w", kind=kind, value=value,
                            source_ref="span")


@pytest.mark.parametrize("kind,expected", sorted(EXPECTED.items(),
                                                 key=lambda kv: kv[0].value))
def test_each_witness_kind_maps_to_one_author(kind, expected):
    """Named per kind, so a failure says which witness was reclassified."""
    assert adapter.author_for(_evidence(kind)) is expected


def test_every_contract_witness_kind_is_classified():
    """No kind falls through to the default silently.

    The default is `READER`, which is safe but wrong for a model reading — and
    a new kind arriving in the contract would take it without anybody noticing.
    """
    missing = [k.value for k in ReaderKind if k not in EXPECTED]
    assert not missing, (
        f"{missing} are witness kinds the contract has and this mapping does "
        "not classify; they would silently become READER")
    assert set(adapter.WITNESS_AUTHORS) == {k.value for k in ReaderKind}


def test_an_unknown_witness_never_becomes_user():
    """The one classification that must not be guessed.

    `USER` is final — never overwritten by a re-read — so handing it to a
    witness nobody recognised would make an unattributable value permanent.
    """
    class Odd:
        kind = type("K", (), {"value": "SOMETHING_NEW"})()

    assert adapter.author_for(Odd()) is Author.READER


def _runtime_intent(kind, value="500"):
    """What the runtime drafts, then classified by the adapter."""
    drafted = VerifiedIntent(
        objective="evaluate_investment_strategy",
        fields={"amount": IntentField(
            value=value,
            # As `draft_intent` leaves it: generic, and deliberately so.
            author=Author.READER,
            produced_by="discovery-runtime@0.1.5",
            source_span="span",
            evidence=(_evidence(kind, value),))},
        produced_by="discovery-runtime@0.1.5",
        utterance_ref="utt-runtime")
    return adapter.classify_authors(drafted)


def _internal_intent(author, value="500"):
    """What the internal path builds for the same reading."""
    return VerifiedIntent(
        objective="evaluate_investment_strategy",
        fields={"amount": IntentField(
            value=value, author=author,
            produced_by="quantify-compiler@1",
            source_span="a different span",
            evidence=(_evidence(ReaderKind.RULE, value),))},
        produced_by="quantify-compiler@1",
        utterance_ref="utt-internal")


@pytest.mark.parametrize("kind,expected", sorted(EXPECTED.items(),
                                                 key=lambda kv: kv[0].value))
def test_the_two_paths_agree_on_author_and_identity(kind, expected):
    """The paired test. Same proposal, same witness, both implementations.

    The two intents differ in `produced_by`, `source_span`, `utterance_ref` and
    the evidence they carry — every one of which is excluded from
    `canonical_form`. What remains is value and author, so agreeing on the
    author is exactly what makes the identities agree.
    """
    runtime = _runtime_intent(kind)
    internal = _internal_intent(expected)

    assert runtime.fields["amount"].author is internal.fields["amount"].author
    assert runtime.intent_hash == internal.intent_hash, (
        f"{kind.value}: the two paths classify the witness the same way and "
        "still produce different identities, so something outside author and "
        "value has entered canonical_form")


def test_the_parity_test_can_fail():
    """Without this the pairing above proves nothing.

    If `intent_hash` ignored author, every parametrisation would pass while the
    two paths disagreed about who established the value.
    """
    assert (_internal_intent(Author.MODEL).intent_hash
            != _internal_intent(Author.READER).intent_hash), (
        "author does not affect intent_hash, so the parity assertions above "
        "are not testing what they claim")


def test_an_unclassified_runtime_draft_would_mismatch():
    """The gap this slice closes, demonstrated.

    A drafted intent left with the generic `READER` disagrees with the internal
    path's `MODEL` for a hosted-model reading — which is precisely the mismatch
    the corpus harness would have reported as a semantic difference.
    """
    unclassified = VerifiedIntent(
        objective="evaluate_investment_strategy",
        fields={"amount": IntentField(
            value="500", author=Author.READER, produced_by="d",
            evidence=(_evidence(ReaderKind.MODEL),))},
        produced_by="d")
    assert unclassified.intent_hash != _internal_intent(Author.MODEL).intent_hash
    assert (adapter.classify_authors(unclassified).intent_hash
            == _internal_intent(Author.MODEL).intent_hash)


def test_a_field_without_evidence_keeps_its_author():
    """Nothing witnessed it, so nothing is claimed about it.

    Inventing an author for an unwitnessed value would be worse than leaving
    the generic one: it would assert a source that does not exist.
    """
    bare = VerifiedIntent(
        objective="o",
        fields={"amount": IntentField(value="500", author=Author.DEFAULT,
                                      produced_by="d")},
        produced_by="d")
    assert adapter.classify_authors(bare).fields["amount"].author is Author.DEFAULT
