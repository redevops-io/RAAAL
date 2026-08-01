"""Forward tracking: the plan is a declaration, observations are facts.

The load-bearing invariant is that the linked return is **uncomputable**, not
discouraged. A chart flowing seamlessly from backtest into live is the most
damaging artifact in this category, and a rule enforced in a style guide is a
rule enforced until someone is in a hurry.
"""
from __future__ import annotations

import pytest

from src.mission import (
    Deviation,
    DeviationKind,
    Eligibility,
    ExpectedEvent,
    LinkedSeriesRefused,
    ObservedEvent,
    PlanObservation,
    Proposal,
    ProposalStatus,
    Segment,
    SegmentedPerformance,
    reconcile,
    scan_language,
)

VEST_AUG = ExpectedEvent("2026-08-15", "vest", "100 shares", source="rsu schedule")
VEST_NOV = ExpectedEvent("2026-11-15", "vest", "100 shares", source="rsu schedule")


class TestTheLinkedSeriesIsUncomputable:
    @pytest.mark.parametrize(
        "accessor", ["linked_series", "combined", "since_inception", "full_history"])
    def test_every_way_of_asking_for_it_refuses(self, accessor):
        """Not one guarded method with three unguarded aliases beside it."""
        with pytest.raises(LinkedSeriesRefused):
            getattr(SegmentedPerformance(), accessor)()

    def test_the_refusal_explains_why(self):
        with pytest.raises(LinkedSeriesRefused, match="fitted with the whole period"):
            SegmentedPerformance().linked_series()

    def test_the_payload_keeps_the_segments_apart(self):
        payload = SegmentedPerformance().to_json()

        assert set(payload) >= {"historical", "forward", "linked"}
        assert payload["linked"] is None
        assert "Neither is a track record" in payload["note"]

    def test_the_two_segments_are_named_distinctly(self):
        assert Segment.HISTORICAL.value != Segment.FORWARD.value


class TestReconciliationMatchesOnIdentityNotOrder:
    def test_a_planned_event_that_did_not_happen_is_missing(self):
        [deviation] = reconcile([VEST_AUG], [])

        assert deviation.kind is DeviationKind.MISSING
        assert deviation.expected is VEST_AUG
        assert "everything downstream" in deviation.why_it_matters.lower()

    def test_an_unplanned_event_is_unexpected(self):
        surprise = ObservedEvent("2026-09-02", "bonus", "one-off grant")
        [deviation] = reconcile([], [surprise])

        assert deviation.kind is DeviationKind.UNEXPECTED
        assert deviation.observed is surprise

    def test_an_event_that_happened_differently_is_different(self):
        observed = ObservedEvent("2026-08-15", "vest", "60 shares")
        [deviation] = reconcile([VEST_AUG], [observed])

        assert deviation.kind is DeviationKind.DIFFERENT
        assert deviation.expected and deviation.observed

    def test_a_late_event_is_not_the_same_event(self):
        """Positional matching would call a three-week delay a match."""
        late = ObservedEvent("2026-09-05", "vest", "100 shares")
        kinds = {d.kind for d in reconcile([VEST_AUG], [late])}

        assert kinds == {DeviationKind.MISSING, DeviationKind.UNEXPECTED}

    def test_a_matching_event_produces_no_deviation(self):
        assert reconcile([VEST_AUG], [ObservedEvent("2026-08-15", "vest",
                                                    "100 shares")]) == []

    def test_reconciliation_survives_reordering(self):
        observed = [ObservedEvent("2026-11-15", "vest", "100 shares"),
                    ObservedEvent("2026-08-15", "vest", "100 shares")]
        assert reconcile([VEST_AUG, VEST_NOV], observed) == []


class TestAnObservationRecordsRatherThanRewrites:
    def _observation(self, deviations=()):
        return PlanObservation(
            plan_id="my-rsu", observed_at="2026-09-30T00:00:00Z",
            data_snapshot="prices@2026-09-30",
            expected_events=(VEST_AUG, VEST_NOV),
            observed_events=(ObservedEvent("2026-08-15", "vest", "100 shares"),),
            deviations=tuple(deviations),
        )

    def test_an_observation_names_its_data_snapshot(self):
        """Two observations disagreeing is meaningless without knowing the data."""
        assert self._observation().data_snapshot == "prices@2026-09-30"

    def test_a_clean_observation_says_so_plainly(self):
        assert "agree on every event" in self._observation().summary()

    def test_drift_is_summarised_by_kind_not_just_counted(self):
        observation = self._observation(reconcile(
            [VEST_AUG, VEST_NOV],
            [ObservedEvent("2026-08-15", "vest", "100 shares"),
             ObservedEvent("2026-09-02", "bonus", "one-off")],
        ))

        assert observation.has_drifted
        assert "missing" in observation.summary()
        assert "unexpected" in observation.summary()

    def test_observations_are_addressable_and_hashed(self):
        observation = self._observation()

        assert observation.artifact_id.startswith("observation/my-rsu@")
        assert len(observation.content_hash) == 64

    def test_the_plan_is_not_part_of_the_observation(self):
        """The plan is a declaration and does not change when reality does."""
        payload = self._observation().to_json()

        assert "scenario" not in payload
        assert "plan" not in payload
        assert payload["plan_id"] == "my-rsu"


def proposal(**kw) -> Proposal:
    defaults = dict(
        proposal_id="p1", plan_id="my-rsu", generated_at="2026-08-17",
        generated_from="disposition: sell_all_and_diversify",
        reason="your plan sells vested shares and buys SPY",
        event="vest 2026-08-15", ticker="SPY", notional=5000.0,
    )
    defaults.update(kw)
    return Proposal(**defaults)


class TestProposalsArePaperOnly:
    def test_placed_is_a_property_and_cannot_be_set(self):
        """A field defaulting to False trusts every future caller not to set it."""
        p = proposal()

        assert p.placed is False
        with pytest.raises((AttributeError, TypeError)):
            object.__setattr__  # sanity: frozen dataclass
            p.placed = True

    def test_execution_mode_is_an_enum_not_free_text(self):
        from src.mission import ExecutionMode

        assert proposal().execution_mode is ExecutionMode.NONE
        assert list(ExecutionMode) == [ExecutionMode.NONE], (
            "a second mode must be a reviewed change to a type, not a new string"
        )

    def test_the_payload_says_no_orders_are_placed(self):
        payload = proposal().to_json()

        assert payload["placed"] is False
        assert payload["execution_mode"] == "NONE"
        assert "places no orders" in payload["note"]

    def test_a_blocked_proposal_is_retained_and_not_actionable(self):
        p = proposal(eligibility=Eligibility.BLOCKED_BY_WINDOW,
                     detail="inside a blackout window until 2026-08-31")

        assert not p.actionable
        assert "blackout" in p.detail

    def test_eligibility_is_summarised_on_the_observation(self):
        observation = PlanObservation(
            plan_id="p", observed_at="2026-09-30T00:00:00Z", data_snapshot="s",
            paper_action_proposals=(
                proposal(proposal_id="a"),
                proposal(proposal_id="b", eligibility=Eligibility.BLOCKED_BY_WINDOW),
            ),
        )
        assert observation.to_json()["action_eligibility"] == \
            ["BLOCKED_BY_WINDOW", "ELIGIBLE"]

    def test_a_proposal_traces_to_the_plan_clause_that_produced_it(self):
        """A consequence of what the user wrote, not of what the system decided."""
        assert proposal().generated_from.startswith("disposition:")

    def test_no_proposal_language_reads_as_a_recommendation(self):
        p = proposal()
        found = {k: v for k, v in scan_language(str(p.to_json())).items() if v}

        assert not found, f"proposal reads as advice: {found}"


class TestProposalLifecycle:
    def test_accepted_records_the_person_acting_not_the_platform(self):
        accepted = proposal().resolve(ProposalStatus.ACCEPTED)

        assert accepted.status is ProposalStatus.ACCEPTED
        assert accepted.placed is False, (
            "accepting a proposal must never mean an order was placed"
        )

    def test_a_resolved_proposal_does_not_change_again(self):
        ignored = proposal().resolve(ProposalStatus.IGNORED)
        with pytest.raises(ValueError, match="historical fact"):
            ignored.resolve(ProposalStatus.ACCEPTED)

    def test_resolving_returns_a_copy_rather_than_mutating(self):
        original = proposal()
        original.resolve(ProposalStatus.IGNORED)

        assert original.status is ProposalStatus.OPEN

    def test_a_superseded_proposal_must_name_its_successor(self):
        with pytest.raises(ValueError, match="superseded by nothing"):
            proposal(status=ProposalStatus.SUPERSEDED)

    def test_naming_a_successor_without_the_status_is_refused(self):
        with pytest.raises(ValueError, match="names a successor"):
            proposal(superseded_by="p2")

    def test_an_overdue_proposal_expires_and_is_kept(self):
        from src.mission import expire_overdue

        [expired] = expire_overdue(
            [proposal(expires="2026-08-31",
                      eligibility=Eligibility.BLOCKED_BY_WINDOW)],
            as_of="2026-09-15")

        assert expired.status is ProposalStatus.EXPIRED

    def test_expiry_in_a_blackout_is_reported_as_a_measurable_cost(self):
        """Three proposals expiring behind a closed window is evidence, and it
        is only visible if they were kept."""
        from src.mission import expire_overdue, lifecycle_summary

        expired = expire_overdue(
            [proposal(proposal_id=f"p{i}", expires="2026-08-31",
                      eligibility=Eligibility.BLOCKED_BY_WINDOW)
             for i in range(3)],
            as_of="2026-09-15")
        summary = lifecycle_summary(expired)

        assert summary["expired_in_blackout"] == 3
        assert "what the constraint cost" in summary["note"]

    def test_the_lifecycle_note_proposes_nothing(self):
        from src.mission import expire_overdue, lifecycle_summary

        expired = expire_overdue(
            [proposal(expires="2026-08-31",
                      eligibility=Eligibility.BLOCKED_BY_WINDOW)],
            as_of="2026-09-15")
        found = {k: v for k, v in
                 scan_language(lifecycle_summary(expired)["note"]).items() if v}

        assert not found


class TestTheCounterfactualMeasuresTheConstraint:
    BASE = dict(flow_schedule_hash="h1", starting_capital=0.0,
                cash_policy_rate=0.0, tax_treatment="NONE_APPLIED",
                cost_bps=10.0, execution_lag=1,
                period_start="2021-01-01", period_end="2023-01-01")

    def test_only_timing_differing_isolates_the_constraint(self):
        from src.mission import RunConditions, classify_counterfactual, ComparisonClass

        verdict = classify_counterfactual(
            RunConditions(**self.BASE),
            RunConditions(**{**self.BASE, "execution_lag": 0}),
            constraint="the blackout window")

        assert verdict.comparison_class is ComparisonClass.CONSTRAINT_EFFECT
        assert verdict.attribution_isolated
        assert verdict.isolates == "the blackout window"

    def test_it_does_not_claim_to_be_evidence_about_strategy(self):
        from src.mission import RunConditions, classify_counterfactual

        verdict = classify_counterfactual(
            RunConditions(**self.BASE),
            RunConditions(**{**self.BASE, "execution_lag": 0}),
            constraint="the blackout window")

        assert "not evidence about the strategy" in verdict.required_disclosure

    def test_a_counterfactual_that_changes_more_stops_measuring_the_constraint(self):
        from src.mission import RunConditions, classify_counterfactual

        verdict = classify_counterfactual(
            RunConditions(**self.BASE),
            RunConditions(**{**self.BASE, "execution_lag": 0, "cost_bps": 50.0}),
            constraint="the blackout window")

        assert not verdict.attribution_isolated
        assert "no longer measures" in verdict.detail

    def test_observation_summaries_never_recommend(self):
        observation = PlanObservation(
            plan_id="p", observed_at="2026-09-30T00:00:00Z", data_snapshot="s",
            deviations=reconcile([VEST_AUG], []))
        found = {k: v for k, v in scan_language(observation.summary()).items() if v}

        assert not found


class TestConstraintIsolationRejectsEverythingElse:
    """The class survives only when the constraint is the sole difference."""

    BASE = dict(flow_schedule_hash="h1", starting_capital=0.0,
                cash_policy_rate=0.0, tax_treatment="NONE_APPLIED",
                cost_bps=10.0, execution_lag=1,
                period_start="2021-01-01", period_end="2023-01-01",
                allocation_rule_hash="r1", data_snapshot="prices@2026-07-31")

    @pytest.mark.parametrize("field,value", [
        ("allocation_rule_hash", "r2"),
        ("data_snapshot", "prices@2026-09-30"),
        ("flow_schedule_hash", "h2"),
        ("starting_capital", 5000.0),
        ("cash_policy_rate", 0.04),
        ("tax_treatment", "LONG_TERM_CG"),
        ("cost_bps", 50.0),
        ("period_end", "2022-01-01"),
    ])
    def test_any_other_difference_defeats_the_class(self, field, value):
        from src.mission import RunConditions, classify_counterfactual

        verdict = classify_counterfactual(
            RunConditions(**self.BASE),
            RunConditions(**{**self.BASE, "execution_lag": 0, field: value}),
            constraint="the blackout window")

        assert not verdict.attribution_isolated, (
            f"{field} differing still claimed to isolate the constraint"
        )
        assert verdict.isolates == ""

    def test_a_restated_data_snapshot_is_not_a_decision(self):
        """A revision changes the answer without the user changing anything."""
        from src.mission import RunConditions, classify

        verdict = classify(
            RunConditions(**self.BASE),
            RunConditions(**{**self.BASE, "data_snapshot": "prices@2026-09-30"}))

        assert "data_snapshot" in verdict.differing_dimensions
        assert not verdict.attribution_isolated
