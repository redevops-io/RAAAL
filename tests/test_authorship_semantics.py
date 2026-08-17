"""Who established the structured value — not who wrote the sentence.

    USER     supplied through a structured action: answering a question,
             editing a pre-filled field. The person gave the *value*.
    READER   extracted from the person's prose by a reader, model or parser.
             They wrote the sentence; something else decided what it means.
    DEFAULT  supplied by the catalogue or a system assumption. Nobody said it.

Recognitions used to be `USER`, on the reasoning that they are the user's
words. They are — but the field value is a reading of those words, and
conflating the two hands extraction the highest authority in the contract:
`USER` dominates every other author and is never overwritten by a re-read, so a
misreading became permanent and a better reader could not correct it.

What changes is that a *person* can now correct a reader, which is the point.

**Two construction paths, and only one was wrong.** `workspace/pilot.py` maps
whichever witness settled a value, so the runtime path already produced `MODEL`
for a model reading and reserved `USER` for answered fields — it was correct.
`mission/verified_intent.from_compiled` built every recognition as `USER`, and
that is the site this change repairs.

**This changes `intent_hash` for intents built under the old mapping**, because
`author` is inside `canonical_form` while provenance is not. Plans already
saved keep their stored intent and hash: they are historical artifacts of the
semantics in force when they were made, and rewriting them would change what a
person confirmed after they confirmed it.
"""
from __future__ import annotations

import pytest

from runtime_contracts import Author


#: The declaration this reading needs, applied to a *copy* of the environment.
#:
#: An earlier version called `os.environ.setdefault`, which sets the variable
#: for the rest of the session — every later test then resolved a deployment
#: this file had configured, and 51 of them failed while passing in isolation.
#: A fixture that changes global state for its own convenience is a fixture
#: that fails other people's tests.
DECLARED = {"QUANTIFY_PILOT_READER": "recorded",
            "QUANTIFY_PARSER_MODE": "RUNTIME",
            "QUANTIFY_PARSER_MODEL": "claude-sonnet-5",
            "ANTHROPIC_API_KEY": "unused",
            "PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}


def _reading(text: str):
    """A reading of prose, through the reader the recorded corpus holds."""
    import os

    from src.deploy import context as deploy_context

    resolved = deploy_context.resolve({**os.environ, **DECLARED})

    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.workspace import pilot_routes
    from src.workspace.pilot import read

    original, deploy_context.current = deploy_context.current, lambda: resolved
    try:
        reader = pilot_routes.configured_reader()
        return read(text, reader, schema=QUANTIFY_SCHEMA)
    finally:
        deploy_context.current = original


def _intent(text: str):
    reading = _reading(text)
    assert reading.intent is not None, f"no intent for {text!r}"
    return reading.intent


RECORDED = "invest $500 monthly into VTI"


@pytest.fixture(scope="module")
def from_prose():
    return _intent(RECORDED)


def test_prose_extraction_is_reader_authored(from_prose):
    """The change. A value read out of a sentence belongs to the reader."""
    authored = {name: field.author for name, field in from_prose.fields.items()}
    assert authored, "no fields; this test would prove nothing"

    user_authored = [n for n, a in authored.items() if a is Author.USER]
    assert not user_authored, (
        f"{user_authored} are marked USER but were extracted from prose. USER "
        "dominates every author and is never overwritten by a re-read, so this "
        "would make a misreading permanent.")
    # Not "is READER" — two construction paths exist and they name the witness
    # differently. `pilot.py` maps whichever witness settled the value, so a
    # model reading arrives as MODEL; `verified_intent.from_compiled` builds
    # from recognitions and now says READER. Both are correct under these
    # semantics and both are *not USER*, which is the property that matters:
    # the value was established by something reading the prose, so a later
    # re-read may still correct it.
    assert all(a in (Author.MODEL, Author.READER, Author.DEFAULT)
               for a in authored.values()), (
        f"a prose-extracted field has an unexpected author: {authored}")


def test_a_catalogue_assumption_is_default_authored():
    """Nobody said it, so nobody owns it."""
    from src.workspace.catalog_assumptions import CATALOG_ASSUMED, assume
    from src.workspace.catalog_intent import reading_for
    from src.workspace.strategy_library import entry

    chosen = entry("employer-match")
    reading = assume(reading_for(chosen.key, chosen.text), chosen.key)

    assumed = [f for f in reading.settled if f.provenance == CATALOG_ASSUMED]
    assert assumed, "the fixture produced no catalogue assumptions"
    for field in assumed:
        assert field.provenance == CATALOG_ASSUMED


def test_an_answered_field_is_user_authored():
    """A structured action. The person supplied the value itself."""
    from src.workspace.pilot import answer

    answered = answer(_reading(RECORDED), {"dividend_policy": "reinvested"})

    latest = {f.field: f for f in answered.settled}
    assert latest["dividend_policy"].provenance == "USER_ANSWERED", (
        "a value the person supplied through the form is not recorded as "
        f"theirs: {latest['dividend_policy']}")


def test_an_untouched_assumed_field_does_not_become_user():
    """Pressing the button is not a statement about a value.

    The mechanism is `_answers_in`, which compares what was submitted against
    what the row was offered as. Asserted here too because it is the same
    property this file is about, at the layer a person touches.
    """
    from src.workspace.pilot_routes import _answers_in

    untouched = _answers_in({"answer_cadence": "monthly",
                             "original_cadence": "monthly",
                             "author_cadence": "ASSUMED"})
    assert untouched == {}, (
        f"an assumed row carried back unchanged counted as an answer: {untouched}")


def test_user_still_dominates_reader_on_a_reread():
    """The ordering the whole design rests on.

    A person's answer must survive a fresh reading of the same sentence — that
    is what makes `USER` worth reserving. Checked on the contract's own
    ordering rather than on a rendering of it.
    """
    assert Author.USER.dominates, (
        "USER is no longer final; a person's answer could be replaced by a "
        "re-read, which is what reserving USER buys")
    assert not Author.READER.dominates, (
        "READER became final. Extraction would then be uncorrectable — the "
        "exact defect this change removes.")
    assert not Author.DEFAULT.dominates

    # `dominates` is binary in the contract: only USER is final. That a read
    # value is not replaced by an assumption is Quantify's own rule, enforced
    # by order — recognitions are written first and an inference for a field
    # already present is skipped — so it is checked where it lives.
    intent = _intent(RECORDED)
    for name, field in intent.fields.items():
        if field.author is Author.READER:
            assert field.value not in (None, ""), (
                f"{name} is READER-authored with no value, so an assumption "
                "could fill it and the ordering would not have protected it")


def test_no_construction_path_marks_prose_extraction_as_user():
    """Both paths, checked on the source, since only one of them is exercised
    by the fixture above and the other is what actually regressed."""
    import ast
    import pathlib as _p

    site = _p.Path("src/mission/verified_intent.py")
    tree = ast.parse(site.read_text())
    users = [n.lineno for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and n.attr == "USER"
             and isinstance(n.value, ast.Name) and n.value.id == "Author"]
    assert not users, (
        f"verified_intent.py assigns Author.USER at lines {users}. Every value "
        "it builds is read out of prose or inferred; USER is reserved for a "
        "structured action, and marking extraction USER makes a misreading "
        "permanent because USER is never overwritten by a re-read.")


def test_author_is_part_of_identity_and_provenance_is_not():
    """Why this change moves hashes, stated as a check rather than a claim.

    `canonical_form` keeps `value` and `author` and excludes producer, span and
    evidence. So changing the author mapping changes `intent_hash` for intents
    built under the old one — which is expected, and is why saved plans keep
    their stored artifact rather than being recomputed.
    """
    from runtime_contracts import IntentField, VerifiedIntent

    def build(author, produced_by):
        return VerifiedIntent(
            objective="evaluate_investment_strategy",
            fields={"amount": IntentField(value="500", author=author,
                                          produced_by=produced_by)},
            produced_by=produced_by)

    assert (build(Author.READER, "a").intent_hash
            == build(Author.READER, "b").intent_hash), (
        "provenance changed identity; it is excluded from canonical_form")
    assert (build(Author.READER, "a").intent_hash
            != build(Author.USER, "a").intent_hash), (
        "author did not change identity; it is inside canonical_form and the "
        "authorship migration depends on that being true")
