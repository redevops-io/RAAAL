"""Intent, Mission, and the boundary between public research and private plans.

Two invariants carry most of the weight here, and neither is about correctness of
a number:

* **Selecting a candidate after seeing its results is a search.** The platform
  generated the alternatives, so the trial count has to follow from how the
  choice was made, not from how many plans were saved.
* **Public artifacts may never reference private ones.** That single direction
  rule is what keeps the library impersonal, and it is checkable rather than
  reviewed.
"""
from __future__ import annotations

import pytest

from src.mission import (
    BoundaryViolation,
    Candidate,
    Contradiction,
    FlowSchedule,
    Inference,
    Intent,
    Mission,
    Objective,
    Provenance,
    SelectionBasis,
    UndeclaredVisibility,
    Unresolved,
    check_reference,
    extract_rule,
    visibility_of,
)


def plan(**kw) -> Mission:
    defaults = dict(
        name="my-plan", version=1, title="My Plan", objective=Objective.REPLAY,
        flows=FlowSchedule(cadence="monthly", amount=2000.0),
    )
    defaults.update(kw)
    return Mission(**defaults)


class TestChoosingOnOutcomeIsASearch:
    """The retail overfitting machine, and the thing that stops it being one."""

    def _three(self, basis, evaluated=True):
        return Intent(
            name="retire", version=1,
            stated="I want to invest safely for retirement",
            generation_constraints=["passive", "contribution_compatible"],
            candidates=[
                Candidate("dca", "Monthly DCA into a total-market fund", evaluated=evaluated),
                Candidate("glide", "Glide path from equity to bonds", evaluated=evaluated),
                Candidate("rp", "Risk parity across four asset classes", evaluated=evaluated),
            ],
            selected="glide", selection_basis=basis,
            results_visible_before_selection=(basis is SelectionBasis.AFTER_RESULTS),
        )

    def test_picking_the_best_backtest_spends_every_candidate(self):
        intent = self._three(SelectionBasis.AFTER_RESULTS)

        assert intent.trials_incurred == 3
        assert intent.is_a_search

    def test_picking_from_descriptions_spends_one(self):
        intent = self._three(SelectionBasis.BEFORE_RESULTS, evaluated=False)

        assert intent.trials_incurred == 1
        assert not intent.is_a_search

    def test_claiming_a_blind_choice_after_measuring_several_is_refused(self):
        """The order of operations felt right; the statistics do not care."""
        with pytest.raises(ValueError, match="selection on outcome"):
            self._three(SelectionBasis.BEFORE_RESULTS)

    def test_a_stated_preference_does_not_need_alternatives_measured(self):
        with pytest.raises(ValueError, match="stated up front"):
            self._three(SelectionBasis.STATED_PREFERENCE)

    def test_a_selection_with_no_basis_is_refused(self):
        with pytest.raises(ValueError, match="cannot be left blank"):
            Intent(name="r", version=1, stated="s",
                   generation_constraints=["passive"],
                   candidates=[Candidate("a", "A")], selected="a")

    def test_selecting_a_candidate_that_does_not_exist_is_refused(self):
        with pytest.raises(ValueError, match="not among the candidates"):
            Intent(name="r", version=1, stated="s",
                   generation_constraints=["passive"],
                   candidates=[Candidate("a", "A")], selected="zzz",
                   selection_basis=SelectionBasis.BEFORE_RESULTS)

    def test_the_disclosure_tells_the_user_the_penalty_applies(self):
        text = self._three(SelectionBasis.AFTER_RESULTS).disclosure()

        assert "3 candidates" in text
        assert "already account for that" in text
        assert "will always look better than it is" in text

    def test_a_clean_choice_says_no_penalty_rather_than_staying_silent(self):
        text = self._three(SelectionBasis.BEFORE_RESULTS, evaluated=False).disclosure()
        assert "no selection penalty" in text

    def test_the_basis_is_part_of_identity(self):
        """How a plan was chosen changes what its statistics mean."""
        after = self._three(SelectionBasis.AFTER_RESULTS)
        before = self._three(SelectionBasis.BEFORE_RESULTS, evaluated=False)

        assert after.content_hash != before.content_hash


class TestAMissionKnowsWhatItWasNotTold:
    def test_a_plan_with_open_questions_may_run_but_not_be_saved(self):
        m = plan(provenance=Provenance(unresolved=[
            Unresolved("starting_capital", "How much are you starting with?",
                       "Every figure scales with it."),
        ]))

        assert m.can_simulate, "showing the shape of the answer is useful"
        assert not m.can_save, "saving a placeholder commits the user to it"
        assert "starting_capital" in m.blocking_reasons()[0]

    def test_an_unconfirmed_inference_blocks_saving(self):
        """An inference the user never saw is a declaration they did not make."""
        m = plan(provenance=Provenance(inferred=[
            Inference("moving_average", "simple 200-day",
                      "You said 200DMA; simple is the common reading.", confirmed=False),
        ]))
        assert not m.can_save

    def test_confirming_the_inference_unblocks_it(self):
        m = plan(provenance=Provenance(inferred=[
            Inference("moving_average", "simple 200-day", "…", confirmed=True),
        ]))
        assert m.can_save

    def test_a_self_contradicting_description_blocks_saving(self):
        """"Never sell" and "hold them equally" cannot both hold once prices move."""
        m = plan(provenance=Provenance(contradictions=[
            Contradiction(between=("never sell", "equally"),
                          detail="equal weight cannot be maintained without selling"),
        ]))

        assert not m.can_save
        assert "conflict between" in m.blocking_reasons()[0]

    def test_a_resolved_contradiction_stops_blocking(self):
        m = plan(provenance=Provenance(contradictions=[
            Contradiction(between=("never sell", "equally"), detail="…",
                          resolution="equal dollars at purchase, never rebalanced"),
        ]))
        assert m.can_save

    def test_the_checklist_is_data_the_interface_renders(self):
        m = plan(provenance=Provenance(
            inferred=[Inference("dividends", "reinvested", "…", confirmed=True)],
            unresolved=[Unresolved("starting_capital", "How much?", "Everything scales.")],
        ))
        checklist = m.provenance.checklist()

        assert checklist["understood"][0]["value"] == "reinvested"
        assert checklist["missing"][0]["field"] == "starting_capital"
        assert checklist["ready"] is False

    def test_inferences_are_part_of_the_content_hash(self):
        """Assuming a simple average is a different program from assuming an EMA."""
        simple = plan(provenance=Provenance(
            inferred=[Inference("moving_average", "simple 200-day", "…")]))
        exponential = plan(provenance=Provenance(
            inferred=[Inference("moving_average", "exponential 200-day", "…")]))

        assert simple.content_hash != exponential.content_hash

    def test_tax_treatment_is_declared_from_the_first_version(self):
        assert plan().tax_treatment == "NONE_APPLIED"


class TestTheFlowScheduleIsSeparatelyIdentified:
    def test_identical_schedules_hash_alike(self):
        a = FlowSchedule(cadence="monthly", amount=2000.0)
        b = FlowSchedule(cadence="monthly", amount=2000.0)
        assert a.schedule_hash == b.schedule_hash

    def test_a_different_cadence_is_a_different_schedule(self):
        """Comparability turns on this and nothing else."""
        monthly = FlowSchedule(cadence="monthly", amount=2000.0)
        biweekly = FlowSchedule(cadence="biweekly", amount=2000.0)
        assert monthly.schedule_hash != biweekly.schedule_hash

    def test_the_day_rule_is_part_of_the_schedule(self):
        """"Every month" does not say which day, and the day moves the number."""
        first = FlowSchedule(cadence="monthly", amount=2000.0,
                             day_rule="first_session_of_period")
        last = FlowSchedule(cadence="monthly", amount=2000.0,
                            day_rule="last_session_of_period")
        assert first.schedule_hash != last.schedule_hash


class TestTheBoundaryRunsOneWay:
    def test_a_plan_may_cite_public_research(self):
        check_reference("mission/my-plan@1", "methodology/hrp@3")

    def test_public_research_may_not_cite_a_plan(self):
        with pytest.raises(BoundaryViolation, match="makes the library personal"):
            check_reference("finding/some-finding@1", "mission/my-plan@1")

    def test_a_finding_may_not_cite_an_intent_either(self):
        with pytest.raises(BoundaryViolation):
            check_reference("evidence/e@1", "intent/retire@1")

    def test_public_to_public_is_unaffected(self):
        check_reference("finding/f@1", "methodology/hrp@3")

    def test_private_to_private_is_unaffected(self):
        check_reference("mission/my-plan@1", "intent/retire@1")

    def test_an_undeclared_kind_is_an_error_not_a_default(self):
        """A new artifact type defaulting to public is how the boundary is lost."""
        with pytest.raises(UndeclaredVisibility, match="must state which side"):
            visibility_of("portfolio/mine@1")

    def test_every_declared_kind_resolves(self):
        from src.mission.boundary import VISIBILITY

        for kind in VISIBILITY:
            assert visibility_of(f"{kind}/x@1")


class TestContributingBackExtractsARuleNotAPlan:
    def test_extraction_strips_the_person(self):
        m = plan(events=[{"trigger": "spy_below_200dma", "action": "buy_basket"}],
                 provenance=Provenance())
        extraction = extract_rule(m)

        assert extraction.proposable
        assert "flows" in extraction.stripped
        assert "tax_treatment" in extraction.stripped
        assert "flows" not in extraction.rule

    def test_a_plan_with_no_rule_cannot_be_contributed(self):
        """A contribution schedule is not research."""
        extraction = extract_rule(plan(events=[]))

        assert not extraction.proposable
        assert "only a contribution schedule" in extraction.blockers[0]

    def test_an_unconfirmed_plan_cannot_be_contributed(self):
        m = plan(events=[{"trigger": "t"}],
                 provenance=Provenance(unresolved=[
                     Unresolved("x", "What?", "It matters.")]))
        extraction = extract_rule(m)

        assert not extraction.proposable
        assert "not yet a rule anyone stated" in extraction.blockers[0]
