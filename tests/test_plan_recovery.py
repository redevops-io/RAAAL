"""What a saved plan can be rebuilt from, and what needs its owner.

The classification turns on one distinction that is invisible in the data:

    "amended": []   in a provenance@2 body   the user answered nothing
    (key absent)    in a provenance@1 body   nobody knows

Every test here exists to keep those apart, because the code that reads
`body.get("amended") or ()` cannot.

**The `@1` fixture is not reconstructed from memory.** Its rule is taken from
the serializer as it stood at `3eaa5eb~1`, which wrote exactly four keys:

    {"stated": [...], "inferred": [...],
     "contradictions": [...], "unresolved": [...]}

no `shape`, and none of `amended`, `excluded`, `asset_resolutions`,
`time_window`. `test_the_fixture_matches_the_serializer_that_made_it` asserts
that rule against git rather than against this docstring.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from src.workspace import recovery

CONTROL = ("I buy $1,000 of SPY whenever it crosses below its 200-day moving "
           "average, over the past five years.")

#: Exactly what `Provenance.to_json` wrote before `3eaa5eb`.
LEGACY_KEYS = ("stated", "inferred", "contradictions", "unresolved")

DROPPED = ("amended", "excluded", "asset_resolutions", "time_window")


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")


@pytest.fixture
def deployment():
    from src.deploy.context import bind, resolve, unbind

    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    try:
        yield
    finally:
        unbind()


#: Answered, so `stated` carries the rendered sentences a `@1` body kept
#: instead of the structure. Without them the prose-blindness test below is
#: vacuous — a classifier that read `stated` would find nothing to read, and
#: the test would pass by absence. That was the case until a mutation which
#: reads those sentences survived it.
ANSWERS = (
    ("account_type", "TAXABLE"),
    ("funding_source", "contribution"),
)


@pytest.fixture
def modern(deployment):
    """A plan saved by today's code: `provenance@2`, funding policy present,
    and the two answers recorded as structure *and* rendered as prose."""
    from src.mission.spec import ScenarioAmendment
    from src.workspace.draft import compile_draft

    compiled = compile_draft(
        CONTROL, name="modern", context="recovery fixture",
        amendments=tuple(ScenarioAmendment(question_id=field, answer=value,
                                           recorded_at="2026-08-01T00:00:00Z")
                         for field, value in ANSWERS))
    body = compiled.scenario.to_json()
    return {"plan_id": "plan-modern", "stated_text": CONTROL,
            "scenario": body, "parse": ""}


@pytest.fixture
def legacy(modern):
    """The same plan as it would have been written before `3eaa5eb`.

    Two independent losses, which is what the production plan carries: a `@1`
    provenance body, and no funding policy (F11 compiled the save without a
    priceable set). The answers survive only as
    `"account_type: TAXABLE (answered)"` under `stated` — a sentence composed
    for a screen, which is the thing nothing may read back.
    """
    body = json.loads(json.dumps(modern["scenario"]))
    body["provenance"] = {key: body["provenance"][key] for key in LEGACY_KEYS}
    body.get("flows", {}).pop("funding", None)
    return {**modern, "plan_id": "plan-legacy", "scenario": body}


class TestThePremise:
    def test_the_fixture_matches_the_serializer_that_made_it(self):
        """Sourced from git, not from this file's belief about the past."""
        blob = subprocess.run(
            ["git", "show", "3eaa5eb~1:src/mission/spec.py"],
            capture_output=True, text=True, check=True).stdout
        body = blob.split("class Provenance")[1].split("def to_json")[1]
        body = body.split("@property")[0]
        for key in LEGACY_KEYS:
            assert f'"{key}"' in body, f"@1 did not write {key}"
        for key in DROPPED:
            assert f'"{key}"' not in body, (
                f"@1 did write {key}; the fixture below is wrong and every "
                f"conclusion drawn from it is wrong with it")

    def test_the_legacy_body_reads_as_provenance_1(self, legacy):
        from src.mission.spec import provenance_shape_of

        assert provenance_shape_of(legacy["scenario"]["provenance"]) == \
            "provenance@1"

    def test_the_modern_body_reads_as_provenance_2(self, modern):
        from src.mission.spec import provenance_shape_of

        assert provenance_shape_of(modern["scenario"]["provenance"]) == \
            "provenance@2"

    def test_the_legacy_body_carries_the_answers_only_as_prose(self, legacy):
        """The premise for `TestNothingIsReadFromRenderedProse`. Without these
        sentences that suite cannot fail, and a classifier that read them
        would pass it — which is what happened before this assertion existed.
        """
        stated = legacy["scenario"]["provenance"]["stated"]
        for field, value in ANSWERS:
            assert f"{field}: {value} (answered)" in stated, stated
        assert "amended" not in legacy["scenario"]["provenance"]

    def test_the_legacy_body_has_no_funding_policy(self, legacy):
        """The F11 shape. Without it this fixture would not be the plan the
        matrix is being built for."""
        assert not (legacy["scenario"].get("flows") or {}).get("funding")

    def test_the_modern_body_does(self, modern):
        assert (modern["scenario"]["flows"]["funding"]["kind"]
                == "EVENT_TRIGGERED")


class TestTheMatrixCoversTheSameFieldsTheGateDoes:
    def test_every_semantic_field_is_classified(self, deployment):
        """The equivalence gate compares `semantic_form`. A field it protects
        and this matrix omits would migrate unexamined."""
        from src.workspace.draft import compile_draft

        scenario = compile_draft(CONTROL, name="x", context="coverage").scenario
        assert set(scenario.semantic_form()) == \
            {one.name for one in recovery.FIELDS}


class TestAbsentIsNotEmpty:
    """The distinction the whole module exists for."""

    def test_a_modern_plan_that_answered_nothing_is_recoverable(self, modern):
        """`"amended": []` is knowledge. It must not be reported as a gap that
        needs the owner — that would send a confirmation request to every user
        who never had a question to answer."""
        assessed = recovery.assess(modern, context="recovery test")
        for field in ("amendments", "exclusions", "asset_resolutions"):
            one = _field(assessed, field)
            assert one.outcome == recovery.RECOVERABLE, (one.field, one.why)
            assert one.stored

    def test_the_same_fields_are_unknown_in_a_legacy_plan(self, legacy):
        assessed = recovery.assess(legacy, context="recovery test")
        for field in ("amendments", "exclusions", "asset_resolutions"):
            one = _field(assessed, field)
            assert one.outcome != recovery.RECOVERABLE, (
                f"{field} was never written to a @1 body, so a classifier "
                f"reporting it as recovered is reading an absence as a value")
            assert not one.stored

    def test_an_unstamped_body_is_not_trusted_when_it_claims_emptiness(
            self, legacy):
        """`@1` never wrote these keys and `@2` always stamps `shape`, so a
        body with `"amended": []` and no stamp came from somewhere else. Its
        emptiness is an assertion by an unknown writer, not a record that the
        user decided nothing — and unknown provenance is not consent.

        Written because a mutation that dropped this guard survived the
        legacy fixture above: there the keys are absent, so key-presence alone
        already answers. This is the body where it does not.
        """
        body = json.loads(json.dumps(legacy["scenario"]))
        body["provenance"].update({"excluded": [], "asset_resolutions": [],
                                   "amended": [{"question_id": "account_type",
                                                "answer": "ROTH",
                                                "recorded_at": "?"}]})
        assert "shape" not in body["provenance"]

        assessed = recovery.assess({**legacy, "scenario": body},
                                   context="recovery test")
        for field in ("amendments", "exclusions", "asset_resolutions"):
            one = _field(assessed, field)
            assert one.outcome != recovery.RECOVERABLE, (
                f"{field} was read as decided-nothing from an unstamped body")
        assert assessed.replayed_decisions == 0, (
            "a decision from a body of unknown origin was replayed into the "
            "comparison; that is the system deciding what a plan means")

    def test_what_provenance_1_did_write_is_still_trusted(self, legacy):
        """`@1` wrote `inferred`; only four keys were dropped. Treating every
        provenance field as unknown reports a value that is on disk as a gap,
        and hands an operator a reason that is false even where the outcome
        happens to be right."""
        one = _field(recovery.assess(legacy, context="recovery test"),
                     "inferred")
        assert one.stored, one.why
        assert "present in the stored structured body" in one.why

    def test_the_names_of_the_dropped_keys_match_the_serializer(self):
        assert set(recovery.DROPPED_BY_PROVENANCE_1) == set(DROPPED)

    def test_the_two_bodies_differ_only_in_what_was_persisted(self, modern,
                                                              legacy):
        """The premise for the pair of tests above: same plan, same words,
        same parse. If they differed in content the contrast would be about
        the content."""
        assert modern["stated_text"] == legacy["stated_text"]
        assert (modern["scenario"]["methodology"]
                == legacy["scenario"]["methodology"])


class TestAShapeStampIsNotProofOfStorage:
    """`_with_decisions` rebuilt the provenance from five of eight names, so a
    `provenance@2` plan can carry the stamp — the serializer had the field —
    and still have been stored without it, because a later rebuild erased it.

    Three questions that were one:

        the shape supports the field
        the field was actually persisted
        the field may have been discarded during confirmation
    """

    @pytest.fixture
    def confirmed(self, modern):
        """A `@2` plan whose owner confirmed an inference, with the three
        fields `_with_decisions` used to erase now absent — exactly what the
        four production plans look like."""
        body = json.loads(json.dumps(modern["scenario"]))
        entries = body["provenance"]["inferred"]
        assert entries, "no inference to confirm; the marker cannot appear"
        entries[0]["confirmed"] = True
        for key in ("excluded", "asset_resolutions", "time_window"):
            body["provenance"].pop(key, None)
        return {**modern, "plan_id": "plan-confirmed", "scenario": body}

    def test_the_marker_is_read_from_the_body(self, confirmed, modern):
        """Derived from a stored `confirmed`, not from a date or a build
        number — those need a table mapping builds to behaviour, and the table
        is what goes stale."""
        assert recovery.confirmation_rebuilt(confirmed["scenario"])
        assert not recovery.confirmation_rebuilt(modern["scenario"])

    @pytest.mark.parametrize("field", ["exclusions", "asset_resolutions",
                                       "time_window"])
    def test_an_erased_field_is_not_read_as_no_decision(self, confirmed,
                                                        field):
        one = _field(recovery.assess(confirmed, context="recovery test"), field)
        assert one.shape_supports, (
            "the stamp says the serializer had this field; reporting it as "
            "unsupported confuses the plan's age with what happened to it")
        assert recovery.DISCARDED in one.absence_explained_by
        assert recovery.NO_DECISION not in one.absence_explained_by, (
            "an absence a rebuild could have caused was read as the owner "
            "having decided nothing — a claim about consent, from a field "
            "that was deleted")

    def test_amended_is_not_explained_this_way(self, confirmed):
        """The old rebuild did name `amended`, so its loss is not explicable
        by confirmation, and an empty one still means the owner answered
        nothing. Getting this wrong sends a confirmation request to every user
        who never had a question to answer.

        `amended` is emptied here deliberately. The fixture carries two
        amendments, so the branch under test is never reached with them
        present — a mutation that marked every field discarded survived this
        test until the case existed.
        """
        body = json.loads(json.dumps(confirmed["scenario"]))
        body["provenance"]["amended"] = []
        one = _field(recovery.assess({**confirmed, "scenario": body},
                                     context="recovery test"), "amendments")
        assert recovery.DISCARDED not in one.absence_explained_by, (
            "an empty amendment list was blamed on the confirmation rebuild, "
            "which never touched it")
        assert recovery.NO_DECISION in one.absence_explained_by

    def test_a_surviving_field_is_still_trusted(self, modern):
        """A plan that confirmed something and still has the field was written
        by a build that preserved it. Treating its value as suspect would
        refuse a plan that is intact."""
        body = json.loads(json.dumps(modern["scenario"]))
        body["provenance"]["inferred"][0]["confirmed"] = True
        body["provenance"]["time_window"] = {"kind": "trailing",
                                             "observed": "five years",
                                             "years": 5, "months": None}
        one = _field(recovery.assess({**modern, "scenario": body},
                                     context="recovery test"), "time_window")
        assert one.stored
        assert one.outcome == recovery.RECOVERABLE
        assert not one.absence_explained_by

    def test_such_a_plan_may_not_migrate_automatically(self, confirmed):
        assert not recovery.assess(confirmed,
                                   context="recovery test").automatic


class TestTheThreeOutcomes:
    def test_a_derivation_is_recovered_without_asking(self, legacy):
        """`funding` and `time_window` are functions of the user's own words.
        Nothing was decided and nothing needs confirming — a wrong compile is
        not a lost decision."""
        assessed = recovery.assess(legacy, context="recovery test")
        for field in ("funding", "time_window"):
            one = _field(assessed, field)
            assert one.outcome == recovery.RECOVERABLE, (one.field, one.why)
            assert one.rederived, (
                f"{field} is classed recoverable but no recompile produced "
                f"it; the classification has no basis")

    def test_a_decision_is_referred_to_its_owner(self, legacy):
        assessed = recovery.assess(legacy, context="recovery test")
        one = _field(assessed, "amendments")
        assert one.outcome == recovery.NEEDS_OWNER
        assert assessed.open_questions, (
            "classed as answerable, but a recompile asks nothing — then there "
            "is no question to put to the owner and this is history")

    def test_a_derivation_blocked_on_a_question_is_not_called_historical(
            self, legacy):
        """Found by running the matrix against the real production plan.

        Its funding policy is absent and a recompile does not produce one —
        but only because the instrument is unresolved, and resolving it is a
        question the owner can answer. Reporting that as history tells an
        operator to abandon a field that four answers would restore. The
        distinction is whether the recompile is still asking.
        """
        from src.mission.spec import AssetResolution

        # An unresolved instrument: the description names a fund the compiler
        # cannot place, so no funding subject exists and no policy is built.
        unresolved = {**legacy, "stated_text":
                      "I buy $1,000 of the SP500 ETF whenever it crosses "
                      "below its 200-day moving average."}
        body = json.loads(json.dumps(legacy["scenario"]))
        body.get("flows", {}).pop("funding", None)
        assessed = recovery.assess({**unresolved, "scenario": body},
                                   context="recovery test")
        assert assessed.open_questions, (
            "the premise failed: this description resolves cleanly, so it "
            "cannot show the difference between blocked and lost")
        one = _field(assessed, "funding")
        assert not one.rederived
        assert one.outcome == recovery.NEEDS_OWNER, one.why

    def test_a_legacy_plan_may_not_migrate_automatically(self, legacy):
        assert not recovery.assess(legacy, context="recovery test").automatic

    def test_a_modern_plan_may(self, modern):
        """The control. A classifier that answered `False` for everything
        would pass every test above."""
        assessed = recovery.assess(modern, context="recovery test")
        assert assessed.automatic, (
            f"needs owner: {assessed.by_outcome(recovery.NEEDS_OWNER)}; "
            f"historical: {assessed.by_outcome(recovery.HISTORICAL)}; "
            f"disagreed: {[one.field for one in assessed.fields if one.agrees is False]}")

    def test_a_modern_plan_is_compared_against_its_own_decisions(self, modern):
        """The premise for the test above. If the owner's recorded answers
        were not replayed, the comparison would be against a reading of the
        description alone and every answered field would look like drift —
        which is exactly how this was found."""
        assessed = recovery.assess(modern, context="recovery test")
        assert assessed.replayed_decisions == len(ANSWERS)

    def test_a_legacy_plan_has_nothing_to_replay(self, legacy):
        assert recovery.assess(legacy,
                               context="recovery test").replayed_decisions == 0


class TestDisagreementIsReportedRatherThanResolved:
    def test_a_changed_reading_is_flagged(self, modern):
        """If the compiler now reads the same words differently, replaying is
        not recovery — it is a new interpretation of a plan someone already
        confirmed. It must surface, not be chosen silently."""
        body = json.loads(json.dumps(modern["scenario"]))
        body["methodology"]["allocation_rule"]["assets"] = ["VOO"]
        assessed = recovery.assess({**modern, "scenario": body},
                                   context="recovery test")
        one = _field(assessed, "held_assets")
        assert one.agrees is False
        assert not assessed.automatic, (
            "a plan whose stored reading and fresh reading disagree was "
            "cleared for automatic migration")

    def test_agreement_is_reported_too(self, modern):
        assessed = recovery.assess(modern, context="recovery test")
        assert _field(assessed, "held_assets").agrees is True


class TestNothingIsReadFromRenderedProse:
    def test_the_stated_sentences_are_not_consulted(self, legacy,
                                                    monkeypatch):
        """The rule: a rendered sentence is produced from a decision and may
        never be turned back into one. Enforced by emptying `stated` and
        requiring the classification to be identical."""
        blinded = json.loads(json.dumps(legacy["scenario"]))
        blinded["provenance"]["stated"] = []
        before = recovery.assess(legacy, context="recovery test").to_json()
        after = recovery.assess({**legacy, "scenario": blinded},
                                context="recovery test").to_json()
        before.pop("plan_id"), after.pop("plan_id")
        assert before == after, (
            "the classification changed when the rendered sentences were "
            "removed, so something is reading them")


def _field(assessed, name):
    return next(one for one in assessed.fields if one.field == name)
