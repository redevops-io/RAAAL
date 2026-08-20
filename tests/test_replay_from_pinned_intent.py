"""Phase 4 property 1: replay is a function of the pinned intent.

    same intent hash  ->  same compiled artifact  ->  same outcome
                          with no model call and no re-read of the sentence

The failure this closes is the one the migration named early: a model is
non-deterministic, so a plan that recompiled from its sentence on reopen would
be a plan whose history could be rewritten by an upgrade. Pinning the intent
makes the plan reproducible; re-reading the prose makes it a fresh request
wearing an old name.

Every test here goes through storage. Compiling twice from an object still in
memory proves the function is deterministic and proves nothing about replay —
the interesting path is write, restore, recompile.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from runtime_contracts import (
    Author,
    CorruptIntent,
    IntentField,
    OpenReason,
    Unresolved,
    VerifiedIntent,
    intent_from_json,
)
from src.mission.from_intent import compile_intent

RULE = "benchmark-policy/public-default@1"


def pinned(**fields) -> dict:
    """An intent as it would be stored beside a plan."""
    base = {"assets": "SPY", "amount": "1000", "cadence": "monthly"}
    base.update(fields)
    intent = VerifiedIntent(
        objective="evaluate_investment_strategy",
        produced_by="discovery-runtime@0.4.2",
        utterance_ref="utt-1",
        fields={k: IntentField(value=v, author=Author.USER)
                for k, v in base.items() if v is not None}).seal()
    return json.loads(json.dumps(intent.to_json()))


class TestReplayReachesTheSamePlan:
    def test_a_stored_intent_recompiles_identically(self):
        stored = pinned()
        first = compile_intent(intent_from_json(stored), benchmark_rule=RULE)
        second = compile_intent(intent_from_json(stored), benchmark_rule=RULE)
        assert first.scenario.content_hash == second.scenario.content_hash

    def test_it_matches_what_was_compiled_before_storage(self):
        """Storage must not be a semantic step. If the round trip changed the
        plan, every saved plan would differ from the one the user confirmed."""
        stored = pinned()
        before = compile_intent(intent_from_json(stored), benchmark_rule=RULE)
        after = compile_intent(
            intent_from_json(json.loads(json.dumps(stored))),
            benchmark_rule=RULE)
        assert before.scenario.content_hash == after.scenario.content_hash

    def test_the_plan_still_names_the_intent_it_came_from(self):
        stored = pinned()
        out = compile_intent(intent_from_json(stored), benchmark_rule=RULE)
        assert out.derivation["compiled_from"] == stored["intent_hash"]

    def test_the_applied_defaults_are_the_same_on_replay(self):
        """A default applied on one run and not the next would be a silent
        difference in what the user was told they had left open."""
        stored = pinned()
        one = compile_intent(intent_from_json(stored), benchmark_rule=RULE)
        two = compile_intent(intent_from_json(stored), benchmark_rule=RULE)
        assert one.applied_defaults == two.applied_defaults


class TestReplayReachesTheSameOutcome:
    """Not only the same plan — the same *verdict*. A refusal that became an
    execution on replay would be the worse direction of the same defect."""

    def test_a_refusal_replays_as_a_refusal(self):
        # `inverse_volatility` runs now; the pinned refusal is a method the
        # engine still has no kernel for.
        stored = pinned(allocation_method="hierarchical_risk_parity")
        for _ in range(2):
            out = compile_intent(intent_from_json(stored), benchmark_rule=RULE)
            assert not out.executable
            assert "allocation_method" in {r.dimension for r in out.refusals}

    def test_the_refusal_names_the_same_dimensions_each_time(self):
        stored = pinned(allocation_method="hierarchical_risk_parity",
                        periodic_rebalancing="threshold_band")
        first = {r.dimension for r in
                 compile_intent(intent_from_json(stored)).refusals}
        second = {r.dimension for r in
                  compile_intent(intent_from_json(stored)).refusals}
        assert first == second and len(first) >= 2

    def test_an_execution_replays_as_an_execution(self):
        """The discriminating half: a test that only checked refusals would
        pass on a build that refused everything."""
        stored = pinned()
        assert compile_intent(intent_from_json(stored),
                              benchmark_rule=RULE).executable


class TestReplayConsultsNothingElse:
    def test_the_sentence_is_never_needed(self):
        """`utterance_ref` is an id and the text is not in the record at all.
        A replay that needed it could not run from what was stored."""
        stored = pinned()
        assert "utterance_ref" in stored
        blob = json.dumps(stored)
        assert "I buy" not in blob and "whenever" not in blob
        assert compile_intent(intent_from_json(stored),
                              benchmark_rule=RULE).executable

    def test_replay_does_not_import_a_reader(self, monkeypatch):
        """Proven by removing them. If either the legacy compiler or the hosted
        reader is reachable during replay, a future change can start using one
        and nothing will notice."""
        import sys

        stored = pinned()
        for name in list(sys.modules):
            if name.endswith("readers_quantify"):
                monkeypatch.setitem(sys.modules, name, None)
        out = compile_intent(intent_from_json(stored), benchmark_rule=RULE)
        assert out.executable


class TestAnEditedRecordIsNotReplayed:
    def test_a_tampered_value_is_caught_rather_than_run(self):
        """The record and its identity disagree, and there is no way to tell
        which was edited — so neither is trusted."""
        stored = pinned()
        stored["fields"]["amount"]["value"] = "999999"
        with pytest.raises(CorruptIntent):
            intent_from_json(stored)

    def test_a_reopened_plan_cannot_be_silently_unsealed(self):
        """A dimension nobody settled, added to a sealed record. The seal
        check refuses it rather than downgrading the intent to a draft."""
        stored = pinned()
        stored["unresolved"] = [{"dimension": "day_rule", "reason": "NOT_ASKED",
                                 "result_changing": True}]
        with pytest.raises(CorruptIntent) as raised:
            intent_from_json(stored)
        assert "seal" in str(raised.value)

    def test_a_structurally_broken_record_raises_the_same_type(self):
        """A caller needs one exception type to mean "this record is not
        usable". A reader forced to catch two will eventually catch one."""
        stored = pinned()
        stored["unresolved"] = [{"dimension": "cadence", "reason": "NOT_ASKED",
                                 "result_changing": True}]
        with pytest.raises(CorruptIntent) as raised:
            intent_from_json(stored)
        assert "does not hold together" in str(raised.value)
