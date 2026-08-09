"""The first path a person can take that reaches the runtime.

Before `workspace/pilot.py`, `compile_intent` was called from tests and nowhere
else: every workspace route ran `compile_scenario`, so the whole Discovery →
fusion → Mission pipeline was correct, tested, and unreachable from any journey
a user could make.

These tests drive the seam directly. The route-level version — proving a real
HTTP request reaches the runtime and does not reach the legacy compiler — is in
`test_pilot_route.py`; this file is about the semantics that must hold whichever
entry point calls them.

The reader is a recording, so this suite makes no provider call and cannot move
when a model changes its mind between runs.
"""
from __future__ import annotations

import json

import pytest
from runtime_contracts import Author

from src.discovery.hosted_recording import RecordedHostedReader
from src.discovery.witnesses import BOTH, MODEL_ONLY, Witness, provenance_of
from src.workspace.pilot import (
    InterpreterUnavailable,
    answer,
    read,
    reopen,
)

READER = RecordedHostedReader()

SIMPLE = "invest $500 monthly"
TWO_CADENCES = "contribute $500 monthly, rebalanced annually"


class TestTheLoopCloses:
    """Submit → clarify → answer → save → reopen, which is the pilot journey."""

    def test_a_submission_that_needs_an_answer_says_which(self):
        reading = read(SIMPLE, READER)
        assert reading.questions == ("assets",)
        assert not reading.executable

    def test_answering_it_makes_the_plan_executable(self):
        done = answer(read(SIMPLE, READER), {"assets": "VTI"})
        assert done.executable and not done.questions

    def test_reopening_recompiles_the_same_plan(self):
        """The property the whole migration was built for, exercised by a
        person for the first time: a plan reopened is the plan that was
        confirmed, not a fresh request wearing an old name."""
        done = answer(read(SIMPLE, READER), {"assets": "VTI"})
        stored = json.loads(json.dumps(done.to_json()))

        back = reopen(stored)
        assert back.intent.intent_hash == done.intent.intent_hash
        assert back.compiled.scenario.content_hash == done.compiled.scenario.content_hash

    def test_reopening_cannot_reach_a_reader(self):
        """Structural, not a promise. `reopen` takes a dict — there is no
        reader in scope for it to call, so "no fresh interpretation" is a fact
        about the signature rather than a rule someone has to keep."""
        import inspect

        signature = inspect.signature(reopen)
        assert list(signature.parameters) == ["stored"]

    def test_a_plan_without_a_pinned_intent_refuses_to_reopen(self):
        """Plans created before the runtime was wired in have no intent, and
        reopening one would mean re-reading its sentence — a different request
        with the same name."""
        with pytest.raises(KeyError):
            reopen({"text": SIMPLE})


class TestWhoSaidWhat:
    def test_a_model_reading_is_authored_model(self):
        reading = read(SIMPLE, READER)
        assert reading.intent.fields["cadence"].author is Author.MODEL

    def test_a_human_answer_is_authored_user(self):
        """The distinction Mission's defaults depend on: "the user agreed" and
        "we stopped asking" must never be the same record."""
        done = answer(read(SIMPLE, READER), {"assets": "VTI"})
        assert done.intent.fields["assets"].author is Author.USER
        assert done.intent.fields["cadence"].author is Author.MODEL

    def test_authorship_survives_storage(self):
        done = answer(read(SIMPLE, READER), {"assets": "VTI"})
        back = reopen(json.loads(json.dumps(done.to_json())))
        assert back.intent.fields["assets"].author is Author.USER
        assert back.intent.fields["cadence"].author is Author.MODEL


class TestModelOnlyIsAProfileNotAWeakerRule:
    """`syntax unavailable` must never be recorded as `syntax agrees`."""

    def test_every_settled_field_records_one_witness(self):
        for field in read(SIMPLE, READER).settled:
            assert field.witnesses == ["model"]

    def test_and_it_is_not_recorded_as_agreement(self):
        """A pilot reporting `AGREE` while running one reader would be
        claiming corroboration it never had."""
        provenances = {f.provenance for f in read(SIMPLE, READER).settled}
        assert provenances == {"MODEL_ONLY_ACCEPTED"}
        assert "AGREE" not in provenances

    def test_the_same_decision_would_read_AGREE_with_two_witnesses(self):
        """The discriminating opposite. Without it, `MODEL_ONLY_ACCEPTED`
        could be a constant rather than a reading of what happened."""
        from src.discovery.fusion import Proposal, fuse
        from src.discovery.syntax import SyntaxEvidence

        model = Proposal("cadence", "monthly", "claude-sonnet-5@1")
        alone = fuse("cadence", model=model)
        both = fuse("cadence", model=model,
                    syntax=[SyntaxEvidence(dimension="cadence",
                                           proposed_value="monthly", score=1)])
        assert provenance_of(alone, MODEL_ONLY) == "MODEL_ONLY_ACCEPTED"
        assert provenance_of(both, BOTH) == "AGREE"

    def test_the_profile_says_why_the_other_witness_is_absent(self):
        """"Not installed in this image" is a different fact from "the model
        key is missing", and a reader of the plan can act on neither if the
        record says only that a witness was silent."""
        stored = read(SIMPLE, READER).to_json()
        assert stored["profile"]["single_witness"] is True
        assert "not installed" in stored["profile"]["reason"]


class TestOpenIsNotAbsent:
    def test_a_dimension_the_sentence_omits_is_not_a_question(self):
        """"I looked and it does not say" is a reading. Mission answers it with
        a declared default *and says so*, or refuses by name.

        The first version merged the two and turned a four-word sentence into
        eleven questions, ten of whose answers would have been "I do not mind"
        — which is what a declared default already expresses.
        """
        reading = read(SIMPLE, READER)
        assert len(reading.absent_fields) > 5
        assert not set(reading.absent_fields) & set(reading.questions)

    def test_the_split_survives_an_amendment(self):
        """Answering one question must not turn the absent dimensions into new
        ones — the same conflation a step later, where it would look like the
        page inventing questions in response to being answered."""
        done = answer(read(SIMPLE, READER), {"assets": "VTI"})
        assert not done.questions
        assert done.absent_fields

    def test_a_refusal_by_name_becomes_the_question(self):
        """The question comes from the manifest, not from the reader: "the
        intent names nothing to hold, so there is no plan to compile — this is
        a missing statement, not missing data"."""
        reading = read(SIMPLE, READER)
        assert reading.needs_input == ("assets",)
        assert {r.dimension for r in reading.refusals} >= {"assets"}


class TestRefusalRatherThanDegradation:
    def test_an_unreachable_reader_refuses(self):
        """A deployment that quietly parsed with the legacy grammar instead
        would serve two different products under one name, and the user who got
        the narrower one would never be told."""
        class Down:
            def read(self, text, schema):
                from src.discovery.reader import ReadingSet

                return ReadingSet(reader_id="claude-sonnet-5@1",
                                  failed="timeout")

        with pytest.raises(InterpreterUnavailable):
            read(SIMPLE, Down())

    def test_the_seam_does_not_import_the_legacy_compiler(self):
        """Checked as a dependency rather than as a substring — a source grep
        asserting a structural property matches its own prose eventually, and
        has done so three times in this project."""
        import ast
        from pathlib import Path

        from src.workspace import pilot

        tree = ast.parse(Path(pilot.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)

        assert not [n for n in imported
                    if "compile_scenario" in n or "parse_model" in n], (
            "the pilot seam reaches the legacy compiler; a deployment must "
            "choose one interpreter and say which")
        assert any("compile_intent" in n for n in imported)


class TestTheStoredArtifactCarriesItsProvenance:
    def test_it_names_the_reader_and_the_compiler(self):
        done = answer(read(SIMPLE, READER), {"assets": "VTI"})
        stored = done.to_json()
        assert stored["reader_id"] == "claude-sonnet-5@1"
        assert stored["derivation"]["compiled_by"] == "quantify-mission@1"

    def test_the_plan_names_the_intent_it_came_from(self):
        """`compiled_from` is the intent hash. A plan that could not name its
        intent could not be shown to have been compiled from one."""
        done = answer(read(SIMPLE, READER), {"assets": "VTI"})
        assert (done.compiled.derivation["compiled_from"]
                == done.intent.intent_hash)
