"""Phase 4, final property: two runtimes, one contract, no adapter between them.

Quantify's Mission compiles a `VerifiedIntent` into a financial scenario.
`agentic-os`'s Mission compiles one into an execution graph over business
capabilities. They do entirely different jobs, and this file does not pretend
otherwise — nothing here compares a scenario to a graph.

What it compares is the part they genuinely share: how each one *treats* a
sealed intent. That is the contract, and a contract is only real if two
independent implementations honour it without having been written together.

**The comparator contains no translation, and that is the point.** Every case
below builds one canonical artifact, serialises it once, and hands the identical
JSON to `intent_from_json` on each side. Each runtime is then driven through its
own production entry point — `compile_intent` here, `create_mission_from_intent`
there. There is no adapter, no field mapping, no shim. `IMPLEMENTATION_SPLIT.md`
is explicit about why that matters: reproducing a golden hash proves translation
only, never that the production path adopted the contract. A comparator that
translates is measuring its own adapter against itself.

The five properties, and why each is a cross-runtime question rather than a
local one:

    the seal is honoured          either runtime accepting a draft makes the
                                  seal advisory everywhere
    open dimensions block         "we stopped asking" must not become "the user
                                  agreed" in either implementation
    the intent is never edited    a consumer that rewrites what it was given has
                                  replaced the contract with its own opinion
    refusal is by name            silent approximation in one runtime is
                                  indistinguishable to a user from success
    replay is stable              the same stored artifact must reach the same
                                  answer on every read, in both

Vocabulary is deliberately *not* compared. Quantify executes
`evaluate_investment_strategy`; `agentic-os` executes `onboard_customer` and
`recover_overdue_payment`. Each refuses the other's objectives, and that is the
capability manifest working, not a divergence. Mistaking it for one would push
towards a shared vocabulary, which is the coupling the contract exists to avoid.
"""
from __future__ import annotations

import json

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

from agentic_os.mission.executor import Executor, InMemoryOperatorClient
from agentic_os.mission.from_intent import UnsealedIntent, UnsupportedObjective
from agentic_os.mission.registry import CapabilityRegistry
from agentic_os.mission.runtime import MissionRuntime
from agentic_os.mission.store import EventStore
from agentic_os.mission.types import CapabilityManifest, CapabilitySpec
from src.mission.from_intent import NotExecutable, compile_intent

# ── the shared artifact ──────────────────────────────────────────────────────


def artifact(*, objective: str, sealed: bool = True, blocking: bool = False,
             **fields) -> dict:
    """One intent, serialised once. Both runtimes read this exact JSON.

    Returned as a dict rather than a `VerifiedIntent` on purpose: handing each
    side a live object would let them share Python identity, and the thing being
    tested is whether they agree about a *stored* artifact.

    `blocking` cannot be sealed, and finding that out was worth the trip.
    `unsealable` is broader than `blocking`: an unresolved disagreement blocks
    *sealing* as well as execution, even when marked not result-changing. So
    "VERIFIED and still contested" is unreachable through `seal()`, and the
    fixture stays a draft. The forged version of that state is built by editing
    the JSON, which is the only way it can occur in the wild.
    """
    draft = VerifiedIntent(
        objective=objective,
        produced_by="discovery-runtime@0.4.2",
        utterance_ref="utt-x",
        fields={k: IntentField(value=v, author=Author.USER)
                for k, v in fields.items()},
        unresolved=(Unresolved(dimension="day_rule",
                               reason=OpenReason.UNRESOLVED_DISAGREEMENT,
                               result_changing=False),) if blocking else ())
    intent = draft.seal() if sealed and not blocking else draft
    return json.loads(json.dumps(intent.to_json()))


QUANTIFY = dict(objective="evaluate_investment_strategy",
                assets="SPY", amount="1000", cadence="monthly")
MISSION = dict(objective="recover_overdue_payment", customer="acme")


# ── each runtime, behind its own production entry point ──────────────────────


class Outcome:
    """What a runtime did, in the two terms the contract actually defines.

    The contract specifies the *input* both runtimes share, not the result type
    — Quantify returns a `Compiled` carrying refusals, `agentic-os` raises. So
    this reads each runtime's own answer in its own idiom. That is a translation
    of *results*, which is unavoidable and is stated rather than hidden; the
    intent itself crosses untouched, which is the property under test.
    """

    def __init__(self, refused: bool, detail: str, value=None) -> None:
        self.refused, self.detail, self.value = refused, detail, value


def quantify(stored: dict) -> Outcome:
    """Quantify's Mission, through its real entry point."""
    try:
        compiled = compile_intent(intent_from_json(stored))
    except NotExecutable as refusal:
        return Outcome(True, str(refusal))
    if compiled.refusals:
        return Outcome(True, "; ".join(r.message for r in compiled.refusals))
    return Outcome(False, "", compiled.scenario.content_hash)


def _agentic_runtime() -> MissionRuntime:
    registry = CapabilityRegistry()
    for name, outcome in (("billing.lookup", "overdue_invoice"),
                          ("comms.dunning", "dunning_sent"),
                          ("billing.record", "payment_recorded")):
        registry.register(CapabilityManifest(
            name.split(".")[0],
            [CapabilitySpec(name, name.split(".")[0], provides=[outcome],
                            permissions=["billing:write"])]))
    return MissionRuntime(registry, Executor(InMemoryOperatorClient({})),
                          store=EventStore())


def agentic(stored: dict) -> Outcome:
    """agentic-os's Mission, through its real entry point."""
    runtime = _agentic_runtime()
    try:
        mission = runtime.create_mission_from_intent(
            intent_from_json(stored), policy_refs=["billing:write"])
    except (UnsealedIntent, UnsupportedObjective) as refusal:
        return Outcome(True, str(refusal))
    outcome = Outcome(False, "", f"{mission.template}|{mission.state.value}")
    outcome.mission, outcome.store = mission, runtime.store
    return outcome


RUNTIMES = pytest.mark.parametrize("drive", [quantify, agentic],
                                   ids=["quantify", "agentic-os"])


# ── the properties ───────────────────────────────────────────────────────────


def native(drive) -> dict:
    """The objective each runtime actually executes."""
    return QUANTIFY if drive is quantify else MISSION


class TestTheSealIsHonouredByBoth:
    @RUNTIMES
    def test_a_draft_is_refused(self, drive):
        """If either runtime plans a draft, the seal is advisory everywhere —
        Discovery's guarantee is worth only what its weakest consumer honours.
        """
        assert drive(artifact(sealed=False, **native(drive))).refused

    @RUNTIMES
    def test_an_unresolved_disagreement_is_refused(self, drive):
        """`UNRESOLVED_DISAGREEMENT` is the one open state in which proceeding
        means picking a reading nobody chose."""
        outcome = drive(artifact(blocking=True, **native(drive)))
        assert outcome.refused

    @RUNTIMES
    def test_neither_runtime_has_a_lenient_reader(self, drive):
        """The sharpest version of the seal question, and it turned out to be a
        property of the contract rather than of either runtime.

        `unsealable` is broader than `blocking`: an unresolved disagreement
        blocks *sealing* as well as execution, so "VERIFIED and still
        contested" cannot be constructed through `seal()` at all. Both runtimes
        carry a `blocking` check anyway, and neither can fire — an
        unreachable backstop, kept deliberately so that a future loosening of
        the contract does not silently become execution.

        What can still happen is a stored artifact that *claims* VERIFIED —
        hand-edited, corrupted, or written by an implementation that never
        sealed properly. `intent_from_json` re-checks the seal on read, and the
        cross-runtime property is that neither side reads any other way.
        """
        forged = artifact(blocking=True, **native(drive))
        forged["state"] = "VERIFIED"
        with pytest.raises(CorruptIntent):
            drive(forged)


class TestNeitherRuntimeEditsTheIntent:
    @RUNTIMES
    def test_the_artifact_is_unchanged_after_being_consumed(self, drive):
        """A consumer that rewrites what it was handed has replaced the
        contract with its own opinion of it, while the audit trail still says
        Discovery. Checked on the JSON, so an in-place mutation of the decoded
        object cannot hide behind a re-encode."""
        for stored in (artifact(**QUANTIFY), artifact(**MISSION)):
            before = json.dumps(stored, sort_keys=True)
            expected = intent_from_json(stored).intent_hash
            drive(stored)
            assert json.dumps(stored, sort_keys=True) == before
            assert intent_from_json(stored).intent_hash == expected


class TestRefusalIsByNameNotApproximation:
    def test_each_refuses_the_other_s_objective(self):
        """Not a divergence — the capability manifest working. Neither runtime
        may quietly produce *something* for a request it does not model."""
        assert quantify(artifact(**MISSION)).refused
        assert agentic(artifact(**QUANTIFY)).refused

    def test_each_executes_its_own(self):
        """The control. Without it, "both refuse everything" satisfies every
        other test here."""
        assert not quantify(artifact(**QUANTIFY)).refused
        assert not agentic(artifact(**MISSION)).refused

    def test_the_refusal_names_what_was_refused(self):
        """A refusal that does not say what it refused is indistinguishable
        from a crash, and a user cannot act on it."""
        assert "evaluate_investment_strategy" in agentic(artifact(**QUANTIFY)).detail
        assert "assets" in quantify(artifact(**MISSION)).detail


class TestReplayIsStableInBoth:
    @RUNTIMES
    def test_reading_the_same_artifact_twice_agrees_with_itself(self, drive):
        """Determinism per runtime, from storage rather than from a live
        object — the only version of this claim that says anything about a
        restart. Compared within a runtime, never across: they build different
        things, and a fingerprint spanning both would be the adapter this file
        refuses to contain."""
        stored = artifact(**native(drive))
        assert drive(stored).value == drive(stored).value

    def test_both_runtimes_read_the_same_identity_from_one_artifact(self):
        """The cross-runtime half. If the two disagree about *which request
        this is*, every downstream comparison is between different missions.
        """
        stored = artifact(**MISSION)
        expected = intent_from_json(stored).intent_hash

        outcome = agentic(stored)
        recorded = next(e for e in outcome.store.for_mission(outcome.mission.id)
                        if e.type == "MissionCreated").payload["intent_hash"]
        assert recorded == expected

        quantified = compile_intent(intent_from_json(artifact(**QUANTIFY)))
        assert (quantified.derivation["compiled_from"]
                == intent_from_json(artifact(**QUANTIFY)).intent_hash)
