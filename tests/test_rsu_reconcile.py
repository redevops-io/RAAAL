"""Planned against observed, with four kinds of "not there" kept apart.

    not yet observed      the window is still open
    observed late         outside its expected date, inside tolerance
    observed differently  it arrived, and not as predicted
    confirmed absent      someone with evidence says it did not happen

The generic tracker matched `(date, kind)` exactly and had no pending state, so
a vest due on the 15th examined on the 10th reported MISSING, and one that
settled four days late reported MISSING *and* UNEXPECTED — two deviations for
one event that happened once.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from decimal import Decimal

from src.db.decimals import to_decimal

from src.mission.rsu_reconcile import (
    MATCHING_POLICY_VERSION,
    UNKNOWN_STATES,
    CounterfactualRun,
    MatchingPolicy,
    ObservedEvent,
    PlannedEvent,
    ReconciliationStatus,
    reconcile,
)

JUNE = PlannedEvent(
    event_id="plan-jun", grant_ref="grant/g1", expected_date="2026-06-15",
    employer_asset="ACME", expected_gross_shares="100.0",
    expected_withheld_shares="22.0", expected_delivered_shares="78.0",
    source_declaration="declaration/rsu@1", version_pin="pin-abc")


def seen(**overrides) -> ObservedEvent:
    base = dict(observation_id="obs-1", observed_date="2026-06-16",
                effective_date="2026-06-15", grant_ref="grant/g1",
                employer_asset="ACME", gross_shares="100.0",
                withheld_shares="22.0", delivered_shares="78.0",
                evidence_ref="statement/june")
    base.update(overrides)
    return ObservedEvent(**base)


def only(planned, observed, *, as_of, policy=None):
    rows = reconcile(planned, observed, as_of=as_of, policy=policy)
    assert len(rows) == 1, [r.status.value for r in rows]
    return rows[0]


class TestUnknownNeverBecomesAbsent:

    def test_before_the_date_it_is_pending(self):
        """A vest due on the 15th, examined on the 10th, has not gone wrong."""
        row = only([JUNE], [], as_of="2026-06-10")
        assert row.status is ReconciliationStatus.PENDING
        assert row.is_unknown

    def test_inside_the_grace_window_it_is_still_pending(self):
        assert only([JUNE], [], as_of="2026-06-17").status \
            is ReconciliationStatus.PENDING

    def test_after_the_window_it_is_overdue_not_missing(self):
        row = only([JUNE], [], as_of="2026-06-25")
        assert row.status is ReconciliationStatus.UNOBSERVED_OVERDUE
        assert row.is_unknown
        assert "not a claim that it did not happen" in row.detail

    def test_overdue_is_not_missing_confirmed(self):
        assert only([JUNE], [], as_of="2026-07-30").status \
            is not ReconciliationStatus.MISSING_CONFIRMED

    def test_only_an_explicit_confirmation_reaches_missing(self):
        row = only([JUNE], [seen(confirms_absence=True,
                                 evidence_ref="payroll/none")],
                   as_of="2026-07-30")
        assert row.status is ReconciliationStatus.MISSING_CONFIRMED
        assert "payroll/none" in row.evidence_refs

    def test_a_confirmed_absence_is_not_an_unknown(self):
        row = only([JUNE], [seen(confirms_absence=True)], as_of="2026-07-30")
        assert not row.is_unknown

    def test_silence_never_sets_confirms_absence(self):
        assert seen().confirms_absence is False


class TestALateVestIsOneEvent:

    def test_four_days_late_is_late_not_missing_plus_unexpected(self):
        rows = reconcile([JUNE], [seen(effective_date="2026-06-19")],
                         as_of="2026-06-30")
        assert len(rows) == 1
        assert rows[0].status is ReconciliationStatus.LATE

    def test_the_planned_event_is_still_referenced(self):
        row = only([JUNE], [seen(effective_date="2026-06-19")],
                   as_of="2026-06-30")
        assert row.planned_ref == "plan-jun"
        assert row.observed_ref == "obs-1"

    def test_the_date_variance_is_recorded_with_its_size(self):
        row = only([JUNE], [seen(effective_date="2026-06-19")],
                   as_of="2026-06-30")
        [variance] = row.variances
        assert variance.dimension == "date"
        assert variance.delta == 4

    def test_outside_tolerance_it_becomes_two_events(self):
        """Far enough apart, they might genuinely be different events, and
        saying so is the right answer."""
        rows = reconcile([JUNE], [seen(effective_date="2026-08-20")],
                         as_of="2026-09-01")
        statuses = {row.status for row in rows}
        assert statuses == {ReconciliationStatus.UNOBSERVED_OVERDUE,
                            ReconciliationStatus.UNEXPECTED}

    def test_the_tolerance_is_declared_and_versioned(self):
        policy = MatchingPolicy()
        assert policy.version == MATCHING_POLICY_VERSION
        assert policy.to_json()["date_tolerance_days"] == 7

    def test_a_narrower_policy_changes_the_answer(self):
        strict = MatchingPolicy(date_tolerance_days=1)
        rows = reconcile([JUNE], [seen(effective_date="2026-06-19")],
                         as_of="2026-06-30", policy=strict)
        assert {row.status for row in rows} == {
            ReconciliationStatus.UNOBSERVED_OVERDUE,
            ReconciliationStatus.UNEXPECTED}

    def test_the_report_date_does_not_decide_lateness(self):
        """A vest reported in July may have settled in June."""
        row = only([JUNE], [seen(observed_date="2026-07-20",
                                 effective_date="2026-06-15")],
                   as_of="2026-07-30")
        assert row.status is ReconciliationStatus.MATCHED


class TestVarianceIsDimensionSpecific:

    def test_more_withheld_is_a_variance_not_an_unexpected_event(self):
        row = only([JUNE], [seen(withheld_shares="30.0", delivered_shares="70.0")],
                   as_of="2026-06-20")
        assert row.status is ReconciliationStatus.MATCHED_WITH_VARIANCE

    def test_each_dimension_is_reported_separately(self):
        row = only([JUNE], [seen(withheld_shares="30.0", delivered_shares="70.0")],
                   as_of="2026-06-20")
        dimensions = {one.dimension: one.delta for one in row.variances}
        assert dimensions["withheld_shares"] == pytest.approx(8.0)
        assert dimensions["delivered_shares"] == pytest.approx(-8.0)
        assert "gross_shares" not in dimensions

    @pytest.mark.parametrize("expected,observed", [
        ("5000", None),   # planned a value, none reported
        (None, "5000"),   # reported a value, none planned
    ])
    def test_an_unknown_on_one_side_is_not_a_variance(self, expected,
                                                      observed):
        """Reporting one would invent a difference from an absence of
        information — a $5,000 shortfall from a figure nobody supplied.

        Both sides must not be None here: with neither known, a treat-None-as-
        zero implementation skips the comparison anyway and the test cannot
        tell the two apart.
        """
        planned = replace(JUNE, expected_value=expected)
        row = only([planned], [seen(value=observed)], as_of="2026-06-20")

        assert not any(one.dimension == "value" for one in row.variances)
        assert row.status is ReconciliationStatus.MATCHED

    def test_an_exact_match_reports_no_variance(self):
        row = only([JUNE], [seen()], as_of="2026-06-20")
        assert row.status is ReconciliationStatus.MATCHED
        assert row.variances == ()


class TestThePlanNeverMoves:

    def test_reconciling_does_not_alter_the_planned_event(self):
        before = JUNE.to_json()
        reconcile([JUNE], [seen(delivered_shares="70.0")], as_of="2026-06-20")
        assert JUNE.to_json() == before

    def test_reconciling_does_not_alter_the_observation(self):
        observation = seen(delivered_shares="70.0")
        before = observation.to_json()
        reconcile([JUNE], [observation], as_of="2026-06-20")
        assert observation.to_json() == before

    def test_the_expectation_survives_being_wrong(self):
        """Rewriting it to 70 destroys the evidence that the prediction was
        wrong, which is the only thing tracking is for."""
        row = only([JUNE], [seen(delivered_shares="70.0")], as_of="2026-06-20")
        assert to_decimal(JUNE.expected_delivered_shares) == Decimal("78.0")
        assert row.variances


class TestAmbiguityIsNotResolvedAutomatically:

    def test_two_plans_that_could_explain_one_observation(self):
        second = replace(JUNE, event_id="plan-jun-b",
                         expected_date="2026-06-17")
        rows = reconcile([JUNE, second], [seen(effective_date="2026-06-16")],
                         as_of="2026-06-30")
        assert all(row.status is ReconciliationStatus.AMBIGUOUS
                   for row in rows)

    def test_it_names_the_candidates(self):
        second = replace(JUNE, event_id="plan-jun-b",
                         expected_date="2026-06-17")
        rows = reconcile([JUNE, second], [seen(effective_date="2026-06-16")],
                         as_of="2026-06-30")
        assert set(rows[0].candidates) == {"plan-jun", "plan-jun-b"}

    def test_it_is_not_attached_to_the_nearer_date(self):
        """Attaching it silently moves shares between two grants."""
        second = replace(JUNE, event_id="plan-jun-b",
                         expected_date="2026-06-17")
        rows = reconcile([JUNE, second], [seen(effective_date="2026-06-16")],
                         as_of="2026-06-30")
        assert not any(row.status is ReconciliationStatus.MATCHED
                       for row in rows)

    def test_ambiguity_counts_as_unknown(self):
        second = replace(JUNE, event_id="plan-jun-b",
                         expected_date="2026-06-17")
        rows = reconcile([JUNE, second], [seen(effective_date="2026-06-16")],
                         as_of="2026-06-30")
        assert all(row.is_unknown for row in rows)

    def test_two_observations_for_one_plan_conflict(self):
        rows = reconcile([JUNE], [seen(), seen(observation_id="obs-2")],
                         as_of="2026-06-30")
        assert rows[0].status is ReconciliationStatus.CONFLICTING
        assert set(rows[0].candidates) == {"obs-1", "obs-2"}

    def test_a_different_grant_does_not_create_ambiguity(self):
        other = replace(JUNE, event_id="plan-other", grant_ref="grant/g2")
        rows = reconcile([JUNE, other], [seen()], as_of="2026-06-20")
        statuses = {row.planned_ref: row.status for row in rows}
        assert statuses["plan-jun"] is ReconciliationStatus.MATCHED


class TestUnexpectedEvents:

    def test_an_observation_matching_no_plan_stays_visible(self):
        rows = reconcile([], [seen()], as_of="2026-06-20")
        assert rows[0].status is ReconciliationStatus.UNEXPECTED

    def test_it_is_not_attached_to_the_nearest_plan(self):
        far = replace(JUNE, expected_date="2026-01-15")
        rows = reconcile([far], [seen()], as_of="2026-06-20")
        assert {row.status for row in rows} == {
            ReconciliationStatus.UNOBSERVED_OVERDUE,
            ReconciliationStatus.UNEXPECTED}

    def test_a_confirmed_absence_is_not_reported_as_unexpected(self):
        rows = reconcile([], [seen(confirms_absence=True)], as_of="2026-06-20")
        assert rows == []


class TestReplayability:

    def test_the_policy_version_travels_on_every_row(self):
        rows = reconcile([JUNE], [seen()], as_of="2026-06-20")
        assert all(row.matching_policy_version == MATCHING_POLICY_VERSION
                   for row in rows)

    def test_the_same_records_reconcile_the_same_way(self):
        first = reconcile([JUNE], [seen()], as_of="2026-06-20")
        second = reconcile([JUNE], [seen()], as_of="2026-06-20")
        assert [r.status for r in first] == [r.status for r in second]
        assert [r.to_json()["variances"] for r in first] == \
            [r.to_json()["variances"] for r in second]

    def test_altering_an_observation_changes_only_the_reconciliation(self):
        """The plan hash is unchanged; the derived relationship moves."""
        before = JUNE.to_json()
        matched = only([JUNE], [seen()], as_of="2026-06-20")
        varied = only([JUNE], [seen(delivered_shares="70.0")],
                      as_of="2026-06-20")

        assert JUNE.to_json() == before
        assert matched.status is not varied.status

    def test_removing_an_observation_returns_to_unknown_not_absent(self):
        assert only([JUNE], [], as_of="2026-06-25").status \
            is ReconciliationStatus.UNOBSERVED_OVERDUE


class TestCounterfactualsStayHypothetical:

    def test_it_is_labelled(self):
        run = CounterfactualRun(counterfactual_id="cf-1",
                                observed_state_ref="obs-1",
                                changed_dimension="blackout_timing")
        assert run.hypothetical is True
        assert "did not happen" in run.to_json()["label"]

    def test_it_is_not_isolated_without_a_verdict(self):
        run = CounterfactualRun(counterfactual_id="cf-1",
                                observed_state_ref="obs-1",
                                changed_dimension="blackout_timing")
        assert not run.is_isolated

    def test_a_constraint_counterfactual_names_what_it_isolates(self):
        run = CounterfactualRun(
            counterfactual_id="cf-1", observed_state_ref="obs-1",
            changed_dimension="blackout_timing",
            comparability_verdict={"attribution_isolated": True,
                                   "status": "CONSTRAINT_EFFECT"},
            isolates="blackout timing")
        assert run.is_isolated
        assert run.isolates == "blackout timing"

    def test_an_unisolated_verdict_defeats_the_claim(self):
        run = CounterfactualRun(
            counterfactual_id="cf-1", observed_state_ref="obs-1",
            changed_dimension="blackout_timing",
            comparability_verdict={"attribution_isolated": False})
        assert not run.is_isolated

    def test_it_holds_no_copy_of_the_observation(self):
        """A counterfactual that could be mistaken for what happened is worse
        than none."""
        import dataclasses

        names = {f.name for f in dataclasses.fields(CounterfactualRun)}
        assert "observed_state_ref" in names
        for embedded in ("observation", "observed_event", "shares", "value"):
            assert embedded not in names


class TestTheStatesAreAllDistinct:

    def test_every_status_is_reachable_and_named(self):
        assert len(set(ReconciliationStatus)) == 9

    def test_unknown_states_are_declared(self):
        assert UNKNOWN_STATES == {
            ReconciliationStatus.PENDING,
            ReconciliationStatus.UNOBSERVED_OVERDUE,
            ReconciliationStatus.AMBIGUOUS,
            ReconciliationStatus.CONFLICTING}

    def test_missing_confirmed_is_not_an_unknown_state(self):
        assert ReconciliationStatus.MISSING_CONFIRMED not in UNKNOWN_STATES
