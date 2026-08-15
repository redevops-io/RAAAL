"""A picked strategy reaches a sealed intent without a model.

Step 2's exit gate. Picking `scheduled-funding` used to paste a sentence into a
box and send it to a language model to be told what it meant — discarding the
one fact the product was certain of, which is which entry it offered.

Two properties, and the second is what makes the first safe:

  * no reader is consulted — checked by making the reader raise, which is the
    only form of "does not call a model" a test can hold;
  * the structured path and the prose path reach the *same execution identity*
    for every offered strategy, so the table cannot drift into offering
    somebody a different strategy than the sentence they read.
"""
from __future__ import annotations

import json
from hashlib import sha256

import pytest

from src.discovery.hosted_recording import RecordedHostedReader
from src.discovery.schema import QUANTIFY_SCHEMA
from src.discovery.syntax_stanza import RecordedReader
from src.discovery.witnesses import BOTH
from src.mission.from_intent import NotExecutable, compile_intent
from src.workspace.catalog_evidence import STATES
from src.workspace.catalog_intent import CATALOG_VERSION, intent_for, reads_a_model
from src.workspace.pilot import read
from src.workspace.strategy_library import LIBRARY

ENTRIES = [(group, entry) for group in LIBRARY for entry in group.entries]
IDS = [entry.key for _group, entry in ENTRIES]


def prose_intent(text: str):
    hosted = RecordedHostedReader()
    hosted.recorded_with = dict(hosted.recorded_with,
                                reader_id="gpt-5.4-2026-03-05@1")
    return read(text, hosted, schema=QUANTIFY_SCHEMA, profile=BOTH,
                syntax_reader=RecordedReader()).intent


def execution_identity(intent) -> str:
    """What the plan executes, or why it cannot. Both are identities.

    A refusal is part of the answer: two paths that refuse the same dimension
    for the same reason agree, and one that compiles where the other refuses
    does not. Comparing only compiled scenarios would silently pass every
    entry this build cannot run — which is most of them.
    """
    if intent is None or not intent.is_verified:
        return "unsealed"
    try:
        out = compile_intent(intent)
    except NotExecutable as refused:
        return "refused:" + ",".join(sorted(r.dimension for r in refused.refusals))
    if out.scenario is None:
        return "refused:" + ",".join(sorted(r.dimension for r in out.refusals))
    return sha256(json.dumps(out.scenario.canonical_form(), sort_keys=True,
                             default=str).encode()).hexdigest()


class TestNoModelIsConsulted:
    def test_the_structured_path_does_not_touch_a_reader(self, monkeypatch):
        """The gate, stated as a trap rather than as an absence.

        `configured_hosted_reader` is the one way to a model from here. Made to
        raise, so a path that reaches for one fails loudly instead of being
        argued about.
        """
        import src.discovery.readers_quantify as readers

        def refuse(*_args, **_kwargs):
            raise AssertionError(
                "the structured catalog path asked for a language model")

        monkeypatch.setattr(readers, "configured_hosted_reader", refuse)

        intent, unreadable = intent_for("scheduled-funding")
        assert intent is not None and intent.is_verified
        assert unreadable == ()

    def test_it_imports_no_reader(self):
        """A module that never imports one cannot call one on a path a test
        happened not to walk."""
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "src" / "workspace"
                  / "catalog_intent.py").read_text()
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                imported.add(("." * node.level) + (node.module or ""))
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert not any("reader" in name or "hosted" in name
                       for name in imported), sorted(imported)


class TestTheTwoPathsMeanTheSameThing:
    @pytest.mark.parametrize("group,entry", ENTRIES, ids=IDS)
    def test_structured_and_prose_reach_one_execution_identity(self, group, entry):
        """The table is checked against the sentence rather than trusted.

        A row that drifts from the entry it describes offers somebody a
        different strategy than the one they read, and nothing else in the
        system would notice — the sentence still says what it always said.
        """
        structured, _unreadable = intent_for(entry.key)
        assert structured is not None, f"{entry.key} has no structured evidence"
        assert execution_identity(structured) == \
            execution_identity(prose_intent(entry.text)), (
            f"{entry.key}: choosing it from the menu and typing its own "
            "sentence produce different plans")


class TestTheTableCoversTheMenu:
    def test_every_offered_strategy_has_structured_evidence(self):
        """Otherwise the feature is half-shipped and looks finished.

        `intent_for` returns None for an entry it does not describe, and the
        route falls back to reading the sentence. That fallback is correct and
        must not be invisible: measured here so "the catalogue no longer calls
        a model" is a fact rather than a claim about the entries somebody
        happened to try.
        """
        still_read = sorted(entry.key for _group, entry in ENTRIES
                            if reads_a_model(entry.key))
        assert still_read == [], (
            f"{len(still_read)} offered strategies still go through a language "
            f"model when picked: {still_read}")

    def test_the_table_names_nothing_the_menu_does_not_offer(self):
        offered = {entry.key for _group, entry in ENTRIES}
        assert set(STATES) <= offered, sorted(set(STATES) - offered)


class TestWhoSaidWhatSurvives:
    def test_a_catalogue_value_is_authored_by_the_reader_not_the_model(self):
        """No model was consulted, so `MODEL` would be a false provenance."""
        from runtime_contracts import Author

        intent, _ = intent_for("scheduled-funding")
        assert intent.fields["assets"].author is Author.READER
        assert all(field.author is Author.READER
                   for field in intent.fields.values())

    def test_an_edit_is_the_users_and_wins(self):
        from runtime_contracts import Author

        intent, _ = intent_for("scheduled-funding", edits={"assets": "SPY"})
        assert intent.fields["assets"].value == "SPY"
        assert intent.fields["assets"].author is Author.USER
        assert intent.fields["cadence"].author is Author.READER, (
            "editing one value re-authored the rest as the user's")

    def test_an_edit_is_canonicalised_like_anything_else(self):
        intent, _ = intent_for("scheduled-funding", edits={"amount": "$2,000"})
        assert intent.fields["amount"].value == "2000"

    def test_an_unreadable_edit_is_reported_and_does_not_settle(self):
        intent, unreadable = intent_for("scheduled-funding",
                                        edits={"amount": "a portion"})
        assert any(name == "amount" for name, _why in unreadable)
        assert intent.fields["amount"].value == "500", (
            "an unreadable edit replaced the entry's own value")

    def test_the_record_says_it_came_from_the_catalogue(self):
        intent, _ = intent_for("scheduled-funding")
        assert intent.produced_by.startswith(CATALOG_VERSION)
        assert "scheduled-funding" in intent.produced_by
        assert intent.utterance_ref == "catalog:scheduled-funding"
