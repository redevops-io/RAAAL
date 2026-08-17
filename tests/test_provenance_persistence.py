"""Every kind of provenance survives the round trip to storage.

`Provenance.to_json` serialized four of its eight fields. `amended`,
`excluded`, `asset_resolutions` and `time_window` were dropped on the way to
disk, and each has a docstring explaining why it exists — written while the
field was being discarded.

The cost was concrete. A production plan whose owner answered six questions
stored no record of having been asked. The answers survived only as rendered
sentences under `stated`:

    "account_type: TAXABLE (answered)"

That is a presentation artifact. Recompiling the plan under a newer compiler
had nothing to replay, so the migration produced a plan with every question
open again — and the only way to "recover" the answers would have been to
parse our own rendering back into structured consent, reversing the direction
of authority inside the record that exists to say what the owner agreed to.

So the plan is not migrated. It is classified, and a replacement is entered
through the builder.
"""
from __future__ import annotations

import pytest

from src.mission.spec import (
    AssetResolution,
    Contradiction,
    Inference,
    Provenance,
    ScenarioAmendment,
    ScenarioExclusion,
    OpenQuestion,
)

AT = "2026-08-06T00:00:00Z"

#: Deliberately populates every field. A fixture that left one empty would let
#: its serialization be deleted without any test noticing — which is exactly
#: how four of them came to be missing.
FULL = Provenance(
    stated=("I buy $1,000 of VOO",),
    inferred=(Inference("dividends", "reinvested", "default set"),),
    unresolved=(OpenQuestion("account_type", "Which account?", "tax differs"),),
    contradictions=(Contradiction(between=("holdings_policy", "allocation_rule"),
                                  detail="never sell while maintaining weight"),),
    amended=(ScenarioAmendment(question_id="account_type", answer="TAXABLE",
                               recorded_at=AT),),
    excluded=(ScenarioExclusion(item="forward_projection",
                                reason="not modelled", acknowledged_at=AT),),
    asset_resolutions=(AssetResolution(
        observed_phrase="SP500 ETF", registry_digest="abc123",
        chosen_instrument_id="SPY", candidates_shown=("SPY", "VOO"),
        ranking_reasons=("issuer match",)),),
)


class TestEveryFieldSurvives:
    def test_the_fixture_populates_every_field(self):
        """The premise. An empty field serializes to an empty list whether or
        not the code that would have filled it exists."""
        import dataclasses

        # `time_window` is exercised separately: it is a single object rather
        # than a sequence, and putting one here would make every other case
        # carry a window it does not need.
        empty = [f.name for f in dataclasses.fields(Provenance)
                 if f.name != "time_window" and not getattr(FULL, f.name)]
        assert not empty, f"these fields are empty in the fixture: {empty}"

    @pytest.mark.parametrize("field", [
        "stated", "inferred", "contradictions", "unresolved",
        "amended", "excluded", "asset_resolutions", "time_window",
    ])
    def test_the_field_is_serialized(self, field):
        assert field in FULL.to_json()

    def test_amendments_keep_who_answered_and_when(self):
        """Field, value, source and timestamp. An answer that lost its source
        would be indistinguishable from an inference, which misstates who
        decided — the distinction `ScenarioAmendment` exists to draw."""
        amended = FULL.to_json()["amended"][0]
        assert amended["question_id"] == "account_type"
        assert amended["answer"] == "TAXABLE"
        assert amended["recorded_at"] == AT
        assert amended["source"]

    def test_exclusions_keep_their_reason(self):
        excluded = FULL.to_json()["excluded"][0]
        assert excluded["item"] == "forward_projection"
        assert excluded["reason"]

    def test_asset_resolutions_keep_the_registry_that_read_them(self):
        """Without the digest, a resolution cannot be checked against the
        registry that produced it — which was the whole point of pinning it."""
        resolution = FULL.to_json()["asset_resolutions"][0]
        assert resolution["observed_phrase"] == "SP500 ETF"
        assert resolution["chosen_instrument_id"] == "SPY"
        assert resolution["registry_digest"] == "abc123"

    def test_the_time_window_survives_as_an_instruction(self):
        from src.mission.time_window import WindowKind, TimeWindow

        import dataclasses

        window = TimeWindow(kind=WindowKind.TRAILING, years=5,
                            observed="the past 5 years")
        carried = dataclasses.replace(FULL, time_window=window).to_json()
        assert carried["time_window"]["kind"] == WindowKind.TRAILING.value


class TestTheShapeIsStamped:
    """"No amendments" and "amendments were never recorded" look identical in
    the data and mean opposite things."""

    def test_a_current_body_names_its_shape(self):
        assert FULL.to_json()["shape"] == "provenance@2"

    def test_an_old_body_reads_as_legacy(self):
        from src.mission.spec import provenance_shape_of

        assert provenance_shape_of(
            {"stated": [], "inferred": [], "unresolved": []}) == "provenance@1"

    def test_an_empty_current_body_does_not_read_as_legacy(self):
        """A plan that genuinely had nothing to amend must not be mistaken for
        one whose amendments were lost."""
        from src.mission.spec import provenance_shape_of

        assert provenance_shape_of(Provenance().to_json()) == "provenance@2"


class TestMigrationRefusesReconstruction:
    """Migration may replay only persisted structured decisions."""

    def test_a_legacy_body_is_refused_by_name(self):
        from src.workspace.migrate_plan import migratable

        allowed, refusal = migratable(
            {"provenance": {"stated": ["account_type: TAXABLE (answered)"]}})
        assert not allowed
        assert refusal == "LEGACY_PROVENANCE_INCOMPLETE"

    def test_a_current_body_is_permitted(self):
        from src.workspace.migrate_plan import migratable

        allowed, refusal = migratable({"provenance": FULL.to_json()})
        assert allowed and not refusal

    def test_the_answers_are_read_from_structure_not_prose(self):
        """The rendered sentence carries the same information and must not be
        the source of it. Given a body with the prose and no structure, nothing
        is recovered."""
        from src.workspace.migrate_plan import stored_amendments

        prose_only = {"provenance": {
            "stated": ["account_type: TAXABLE (answered)",
                       "asset_identity:SP500 ETF: SPY (answered)"]}}
        assert stored_amendments(prose_only) == ()

    def test_structured_answers_are_recovered(self):
        from src.workspace.migrate_plan import stored_amendments

        recovered = stored_amendments({"provenance": FULL.to_json()})
        assert [one.question_id for one in recovered] == ["account_type"]
        assert recovered[0].answer == "TAXABLE"


class TestTheRoundTripThroughStorage:
    """Save, reopen, export. The serialization is only half the journey."""

    @pytest.fixture(autouse=True)
    def synthetic(self, monkeypatch):
        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")

    @pytest.fixture
    def saved(self, tmp_path):
        from src.deploy.context import bind, resolve, unbind
        from src.mission.compiler import compile_scenario
        from src.mission.scenario import ScenarioSpecification
        from src.workspace.store import WorkspaceStore

        bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
        try:
            store = WorkspaceStore(tmp_path / "w.db")
            compiled = compile_scenario(
                "I buy $500 of VOO every month.", name="p", version=1,
                benchmark_rule="benchmark-policy/public-default@1")
            source = compiled.scenario.provenance
            scenario = ScenarioSpecification(**{
                **compiled.scenario.__dict__,
                "provenance": Provenance(
                    stated=source.stated,
                    inferred=tuple(Inference(i.field, i.value, i.why,
                                             confirmed=True)
                                   for i in source.inferred),
                    contradictions=source.contradictions, unresolved=(),
                    amended=FULL.amended, excluded=FULL.excluded,
                    asset_resolutions=FULL.asset_resolutions)})
            store.save_plan(plan_id="p", owner="pilot", scenario=scenario,
                            stated_text="I buy $500 of VOO every month.",
                            saved_at=AT)
            yield store
        finally:
            unbind()

    def test_the_amendments_survive_a_reopen(self, saved):
        body = saved.get_plan("p", "pilot")["scenario"]
        assert body["provenance"]["amended"][0]["question_id"] == "account_type"

    def test_the_exclusions_survive_a_reopen(self, saved):
        body = saved.get_plan("p", "pilot")["scenario"]
        assert body["provenance"]["excluded"][0]["item"] == "forward_projection"

    def test_the_resolutions_survive_a_reopen(self, saved):
        body = saved.get_plan("p", "pilot")["scenario"]
        assert body["provenance"]["asset_resolutions"][0][
            "chosen_instrument_id"] == "SPY"

    def test_the_shape_survives_a_reopen(self, saved):
        body = saved.get_plan("p", "pilot")["scenario"]
        assert body["provenance"]["shape"] == "provenance@2"

    def test_the_amendments_survive_an_export(self, saved):
        """A user's own copy of their account must carry the decisions they
        made, not only the sentences describing them."""
        import json

        from src.workspace.erasure import export_workspace

        bundle = export_workspace(saved, "pilot")
        row = bundle["tables"]["plan"][0]
        body = row["scenario"]
        if isinstance(body, str):
            body = json.loads(body)
        assert body["provenance"]["amended"]

    def test_a_reopened_plan_is_migratable(self, saved):
        """The end of the chain: a plan saved today can be replayed under a
        later compiler, because its decisions are decisions."""
        from src.workspace.migrate_plan import migratable

        allowed, refusal = migratable(saved.get_plan("p", "pilot")["scenario"])
        assert allowed, refusal
