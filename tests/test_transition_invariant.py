"""The transition write must change exactly one row.

This is an invariant guard, tested at its own boundary. It is *not* a claim that
today's PostgreSQL protocol can produce a zero row count — the row lock and the
post-lock re-read make a loser refuse before it ever reaches the write. The
claim under test is narrower and does not depend on reachability:

    if the persistence layer reports that no transition occurred,
    the application cannot report that one did

Four protections sit in front of the durable write, and each answers a different
question:

    row lock            no competing transaction can decide concurrently
    post-lock re-read   the proposal still authorizes acceptance
    `status='PROPOSED'` the stored state still permits the transition
    row count == 1      *this* transaction actually performed it

Only the last is about what the write did. A future refactor, a trigger, an
alternate caller or a changed isolation level could open a path the first three
do not cover, and the failure would be an accepted response for a transition
that never happened.

So the transition method is stubbed to report zero while everything before it
succeeds normally. That is a legitimate unit-level falsification of the
invariant; the real concurrency schedule is proven separately in
`tests/test_postgres_concurrency.py`.
"""
from __future__ import annotations

import pytest

from src.workspace.apply import (
    ApplyRefused,
    ProposalConflict,
    ProposalStatus,
    TransitionIntegrityError,
    accept,
)
from src.workspace.intent import plan
from src.workspace.proposal import propose
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create, from_json

OWNER = "pilot"
RESULT = {"modelling_scope": {"excludes": ["dividends"]}, "final_value": 1.0}


@pytest.fixture
def store(tmp_path):
    from src.mission.compiler import compile_scenario
    from src.mission.scenario import ScenarioSpecification
    from src.mission.spec import Inference, Provenance

    store = WorkspaceStore(tmp_path / "w.db")
    compiled = compile_scenario(
        "I put $2,000 into SPY every month in my Roth IRA, on the first trading "
        "day of the period, reinvesting the dividends, and I never sell.",
        name="plan-1", version=1,
        benchmark_rule="benchmark-policy/public-default@1")
    provenance = compiled.scenario.provenance
    scenario = ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=provenance.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in provenance.inferred),
            contradictions=provenance.contradictions, unresolved=())})
    store.save_plan(plan_id="plan-1", owner=OWNER, scenario=scenario,
                    stated_text="seed", saved_at="2026-01-01T00:00:00Z")
    store.record_run(run_id="run-0", plan_id="plan-1",
                     ran_at="2026-01-01T00:00:00Z", result=RESULT, comparison={})
    store.save_worksheet(create(worksheet_id="ws-1", owner_id=OWNER,
                                scenario_ref="plan-1", primary_run_ref="run-0",
                                created_at="2026-01-01T00:00:00Z"))
    return store


def staged(store, instruction="Try SPY, VTI and VT and keep the best"):
    worksheet = from_json(store.get_worksheet("ws-1", OWNER)["payload"])
    intent = plan(instruction, intent_id="i",
                  source_revision=worksheet.revision, history=[],
                  target_run="run-0")
    proposal = propose(intent, worksheet)
    store.save_worksheet_proposal(
        proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
        proposal=proposal, created_at="2026-01-01T00:00:00Z")
    return proposal


def report_rowcount(store, count):
    """Let the transition run, then report a different row count.

    The write still happens, so this isolates the guard rather than also
    removing the update it guards.
    """
    original = store.resolve_worksheet_proposal

    def reporting(proposal_id, owner, **kwargs):
        original(proposal_id, owner, **kwargs)
        return count

    store.resolve_worksheet_proposal = reporting


class TestZeroRowsIsNotSuccess:
    def test_the_caller_is_refused(self, store):
        proposal = staged(store)
        report_rowcount(store, 0)
        with pytest.raises(ProposalConflict):
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=lambda candidate: dict(RESULT))

    def test_no_revision_survives(self, store):
        """Refusing without rolling back would leave an applied edit that no
        proposal records."""
        proposal = staged(store)
        report_rowcount(store, 0)
        with pytest.raises(ProposalConflict):
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=lambda candidate: dict(RESULT))
        assert len(store.worksheet_revisions("ws-1", OWNER)) == 1

    def test_no_candidate_runs_survive(self, store):
        proposal = staged(store)
        report_rowcount(store, 0)
        with pytest.raises(ProposalConflict):
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=lambda candidate: dict(RESULT))
        assert [r["run_id"] for r in store.runs_for("plan-1", OWNER)] == ["run-0"]

    def test_the_error_carries_no_database_internals(self, store):
        """Operators get the original by chaining or logs; callers do not."""
        proposal = staged(store)
        report_rowcount(store, 0)
        with pytest.raises(ProposalConflict) as caught:
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=lambda candidate: dict(RESULT))
        message = str(caught.value)
        for leak in ("UPDATE", "SELECT", "worksheet_proposal", "sqlite3",
                     "psycopg", "constraint", "rowcount"):
            assert leak not in message, f"{leak!r} leaked into {message!r}"


class TestMoreThanOneRowIsAnIntegrityFailure:
    """Impossible under the primary key, which is why saying so is cheap."""

    def test_it_is_not_reported_as_success(self, store):
        proposal = staged(store)
        report_rowcount(store, 2)
        with pytest.raises(TransitionIntegrityError):
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=lambda candidate: dict(RESULT))

    def test_it_is_not_a_refusal(self, store):
        """A refusal invites a retry. This does not: the stored state no longer
        matches what the code can reason about."""
        proposal = staged(store)
        report_rowcount(store, 2)
        with pytest.raises(TransitionIntegrityError) as caught:
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=lambda candidate: dict(RESULT))
        assert not isinstance(caught.value, ApplyRefused)

    def test_nothing_is_committed(self, store):
        proposal = staged(store)
        report_rowcount(store, 2)
        with pytest.raises(TransitionIntegrityError):
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=lambda candidate: dict(RESULT))
        assert len(store.worksheet_revisions("ws-1", OWNER)) == 1


class TestExactlyOneIsTheOnlySuccess:
    def test_one_row_proceeds(self, store):
        proposal = staged(store)
        result = accept(store, proposal_id="wp-1", owner=OWNER,
                        proposal=proposal, worksheet_id="ws-1",
                        at="2026-01-02T00:00:00Z",
                        run_candidate=lambda candidate: dict(RESULT))
        assert result.status == ProposalStatus.ACCEPTED.value
        assert len(store.worksheet_revisions("ws-1", OWNER)) == 2

    def test_the_real_path_reports_one(self, store):
        """The guard would be vacuous if the live path never returned 1."""
        staged(store)
        assert store.resolve_worksheet_proposal(
            "wp-1", OWNER, status=ProposalStatus.ACCEPTED.value,
            resolved_at="2026-01-02T00:00:00Z", result_revision=2) == 1


class TestRunOwnershipIsDerivedNotDefaulted:
    """`record_run` resolves an owner or refuses; it never picks one.

    Removing the refusal left every test green, because every existing caller
    passes a plan that exists and has one owner. The failing conditions have to
    be constructed: a run naming no plan at all, and a plan id held by two
    owners. Both would otherwise get an owner assigned by whichever row the
    lookup happened to return.
    """

    def test_a_run_for_an_unknown_plan_is_refused(self, store):
        from src.workspace.store import NotSaveable

        with pytest.raises(NotSaveable, match="resolves to 0 owners"):
            store.record_run(run_id="r-x", plan_id="p-missing",
                             ran_at="2026-01-01T00:00:00Z", result=RESULT,
                             comparison={})

    def test_nothing_is_written_when_the_owner_cannot_be_resolved(self, store):
        from src.workspace.store import NotSaveable

        with pytest.raises(NotSaveable):
            store.record_run(run_id="r-x", plan_id="p-missing",
                             ran_at="2026-01-01T00:00:00Z", result=RESULT,
                             comparison={})
        with store._conn() as conn:
            assert conn.execute(
                "SELECT COUNT(*) AS n FROM plan_run WHERE run_id = ?",
                ("r-x",)).fetchone()["n"] == 0

    def test_an_explicit_owner_is_honoured(self, store):
        """The caller may state it, which is what the apply path will do once
        two tenants can hold the same plan id."""
        store.record_run(run_id="r-explicit", plan_id="plan-1",
                         ran_at="2026-01-01T00:00:00Z", result=RESULT,
                         comparison={}, owner=OWNER)
        assert store.get_run("r-explicit", OWNER)["owner"] == OWNER

    def test_a_plan_id_held_by_two_owners_is_refused(self, store):
        """The lookup cannot choose, so it must not."""
        from src.mission.compiler import compile_scenario
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance
        from src.workspace.store import NotSaveable

        compiled = compile_scenario(
            "I put $2,000 into SPY every month in my Roth IRA, on the first "
            "trading day of the period, reinvesting the dividends, and I never "
            "sell.", name="plan-1", version=1,
            benchmark_rule="benchmark-policy/public-default@1")
        provenance = compiled.scenario.provenance
        scenario = ScenarioSpecification(**{
            **compiled.scenario.__dict__,
            "provenance": Provenance(
                stated=provenance.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in provenance.inferred),
                contradictions=provenance.contradictions, unresolved=())})
        # A second tenant with the same plan id — legal since the ownership
        # migration, and precisely the case the derivation cannot resolve.
        store.save_plan(plan_id="plan-1", owner="other", scenario=scenario,
                        stated_text="seed", saved_at="2026-01-01T00:00:00Z")

        with pytest.raises(NotSaveable, match="resolves to 2 owners"):
            store.record_run(run_id="r-ambiguous", plan_id="plan-1",
                             ran_at="2026-01-01T00:00:00Z", result=RESULT,
                             comparison={})
