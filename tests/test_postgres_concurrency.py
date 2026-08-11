"""Concurrent acceptance, settled by the database rather than by luck.

Everything here is a **PostgreSQL-only guarantee**. SQLite has one effective
writer, so a green SQLite lane says nothing about any of it — that is the point
of separating these tests rather than parametrising the existing ones over both
engines. What is proven here:

    concurrent writer settlement
    row locking and conditional state transitions
    race translation into typed conflicts
    transaction isolation of the authorizing read

**Every assertion is made from a third connection.** Reading through a session
that just failed shows rolled-back in-memory state, not what another process
would see. The question is always what is durably true, so the check opens its
own connection.

**The divergence is created outside the code that detects it.** Two independent
sessions, synchronized at a barrier, are what makes the race real; a fixture
that pre-set a status would test the check against a state the check itself had
arranged.
"""
from __future__ import annotations

from tests.market_fixture import NO_MARKET_DATA
import os
import threading
from typing import List

import pytest

from src.db.engine import Database
from src.workspace.apply import (
    ApplyRefused,
    ProposalStatus,
    StaleProposal,
    accept,
)
from src.workspace.intent import plan
from src.workspace.proposal import propose
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create, from_json, revise

POSTGRES_URL = os.environ.get("QUANTIFY_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set QUANTIFY_TEST_POSTGRES_URL; these are PostgreSQL-only "
           "guarantees and SQLite cannot evidence them")

OWNER = "pilot"
RESULT = {"market_data": NO_MARKET_DATA.to_json(), "modelling_scope": {"excludes": ["dividends"]}, "final_value": 1.0, "market_data": NO_MARKET_DATA.to_json()}


def fresh_database():
    from sqlalchemy import text

    database = Database(POSTGRES_URL)
    engine = database.sqlalchemy_engine()
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    engine.dispose()
    return database


def seeded_store(owner=OWNER, worksheet_id="ws-1", plan_id="plan-1"):
    """A plan, a run and a worksheet, on a real PostgreSQL database."""
    from src.mission.compiler import compile_scenario
    from src.mission.scenario import ScenarioSpecification
    from src.mission.spec import Inference, Provenance

    store = WorkspaceStore(POSTGRES_URL)
    compiled = compile_scenario(
        "I put $2,000 into SPY every month in my Roth IRA, on the first trading "
        "day of the period, reinvesting the dividends, and I never sell.",
        name=plan_id, version=1,
        benchmark_rule="benchmark-policy/public-default@1")
    provenance = compiled.scenario.provenance
    scenario = ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=provenance.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in provenance.inferred),
            contradictions=provenance.contradictions, unresolved=())})
    store.save_plan(plan_id=plan_id, owner=owner, scenario=scenario,
                    stated_text="seed", saved_at="2026-01-01T00:00:00Z")
    store.record_run(run_id=f"{plan_id}-run-0", plan_id=plan_id,
                     ran_at="2026-01-01T00:00:00Z", result=RESULT, comparison={})
    store.save_worksheet(create(
        worksheet_id=worksheet_id, owner_id=owner, scenario_ref=plan_id,
        primary_run_ref=f"{plan_id}-run-0", created_at="2026-01-01T00:00:00Z"))
    return store


def proposal_for(store, worksheet_id="ws-1", owner=OWNER,
                 instruction="show me the drawdown as a chart"):
    worksheet = from_json(store.get_worksheet(worksheet_id, owner)["payload"])
    intent = plan(instruction, intent_id="i",
                  source_revision=worksheet.revision, history=[],
                  target_run=f"{worksheet.scenario_ref}-run-0")
    return propose(intent, worksheet)


def observe(query, params=()):
    """Read durable state from a connection of its own."""
    database = Database(POSTGRES_URL)
    conn = database.connect()
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


@pytest.fixture
def store():
    fresh_database()
    return seeded_store()


class TestTheTwoMechanismsIndependently:
    """Each guard tested on its own, because together they mask each other.

    Acceptance is protected twice: the row lock makes a second session wait,
    and the conditional `status = 'PROPOSED'` update makes a second write
    match nothing. Either alone settles the race, so removing one and running
    the end-to-end race proves nothing — it passes on the other. Each is
    therefore checked directly, and the race below checks that they compose.
    """

    def test_the_lock_really_holds_the_row(self, store):
        """Asserted by trying to take the row from another session.

        This first checked that `inspect.getsource` contained "FOR UPDATE" —
        and passed with the clause deleted, because the docstring says
        "FOR UPDATE" too. That is the prose-matching failure this codebase has
        hit repeatedly: the assertion must be against what the code does, never
        against the text that describes it.

        `NOWAIT` turns "would block" into an immediate error, so the test
        proves a lock is held without waiting on one.
        """
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        holder = WorkspaceStore(POSTGRES_URL)
        blocked: List[object] = []
        released = threading.Event()
        reached = threading.Event()

        def hold():
            with holder.transaction():
                holder.lock_worksheet_proposal("wp-1", OWNER)
                reached.set()
                released.wait(timeout=10)

        thread = threading.Thread(target=hold)
        thread.start()
        reached.wait(timeout=10)
        try:
            other = Database(POSTGRES_URL).connect()
            try:
                other.execute(
                    "SELECT proposal_id FROM worksheet_proposal "
                    "WHERE proposal_id = ? AND owner = ? FOR UPDATE NOWAIT",
                    ("wp-1", OWNER))
                blocked.append(None)          # the row was free: no lock held
            except Exception as exc:          # noqa: BLE001
                blocked.append(exc)
            finally:
                other.close()
        finally:
            released.set()
            thread.join(timeout=15)

        assert blocked and blocked[0] is not None, (
            "another session took the proposal row while acceptance held it — "
            "no row lock is being taken, so two sessions can both proceed")

    def test_the_conditional_update_reports_a_loss(self, store):
        """The predicate was always here; the row count was thrown away, so a
        caller that updated nothing carried on as though it had won."""
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        first = store.resolve_worksheet_proposal(
            "wp-1", OWNER, status=ProposalStatus.ACCEPTED.value,
            resolved_at="2026-01-02T00:00:00Z", result_revision=2)
        assert first == 1

        second = store.resolve_worksheet_proposal(
            "wp-1", OWNER, status=ProposalStatus.ACCEPTED.value,
            resolved_at="2026-01-02T00:00:00Z", result_revision=3)
        assert second == 0, (
            "a second resolution must match no rows; discarding this is what "
            "let one review produce two acceptances")

    def test_the_row_count_guard_carries_the_race_without_the_lock(self, store):
        """Isolates the second guard by removing the first.

        With the lock working, the loser blocks and refuses at the status
        check, so the row-count guard never fires and deleting it leaves every
        test green. That makes it look like dead code. It is not: it is what
        settles the race if the lock is ever lost — a dropped clause, a changed
        isolation level, a different engine. So the lock is neutralized here and
        the guard is required to do the job alone.
        """
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        def unlocked(self, proposal_id, owner):
            """The same read, taking no lock."""
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM worksheet_proposal "
                    "WHERE proposal_id = ? AND owner = ?",
                    (proposal_id, owner)).fetchone()
            return None if row is None else dict(row)

        a_read = threading.Event()
        b_committed = threading.Event()
        outcomes: List[object] = []
        lock = threading.Lock()

        def session_a():
            mine = WorkspaceStore(POSTGRES_URL)
            mine.lock_worksheet_proposal = lambda p, o: unlocked(mine, p, o)
            original = mine.save_worksheet

            def pause(worksheet):
                a_read.set()
                b_committed.wait(timeout=10)
                return original(worksheet)

            mine.save_worksheet = pause
            try:
                with lock:
                    outcomes.append(accept(
                        mine, proposal_id="wp-1", owner=OWNER,
                        proposal=proposal, worksheet_id="ws-1",
                        at="2026-01-02T00:00:00Z"))
            except Exception as exc:                     # noqa: BLE001
                with lock:
                    outcomes.append(exc)

        thread = threading.Thread(target=session_a)
        thread.start()
        a_read.wait(timeout=10)
        # B accepts and commits while A is mid-transaction and holds no lock.
        accept(WorkspaceStore(POSTGRES_URL), proposal_id="wp-1", owner=OWNER,
               proposal=proposal, worksheet_id="ws-1",
               at="2026-01-02T00:00:00Z")
        b_committed.set()
        thread.join(timeout=30)

        assert outcomes and isinstance(outcomes[0], ApplyRefused), (
            "with no lock, the row-count guard is the only thing standing "
            f"between one review and two acceptances: {outcomes}")
        assert observe("SELECT COUNT(*) AS n FROM worksheet WHERE owner = %s",
                       (OWNER,))[0]["n"] == 2, (
            "the losing session left a revision behind")

    def test_the_stored_outcome_is_the_first_one(self, store):
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")
        store.resolve_worksheet_proposal(
            "wp-1", OWNER, status=ProposalStatus.ACCEPTED.value,
            resolved_at="2026-01-02T00:00:00Z", result_revision=2)
        store.resolve_worksheet_proposal(
            "wp-1", OWNER, status=ProposalStatus.REJECTED.value,
            resolved_at="2026-01-03T00:00:00Z")

        row = observe("SELECT status, result_revision FROM worksheet_proposal "
                      "WHERE proposal_id = %s", ("wp-1",))[0]
        assert row["status"] == ProposalStatus.ACCEPTED.value
        assert row["result_revision"] == 2

    def test_the_authorizing_reads_happen_inside_the_transaction(self, store):
        """Instrumented, not read from the source.

        The first version split `inspect.getsource(accept)` on the `with`
        statement — and broke the moment the function was refactored into two,
        while saying nothing about what actually ran. Source inspection proves
        what a function mentions; only watching the calls proves what it did.

        A check made before the lock describes state another session is still
        free to change, so both authorizing reads must happen while the
        transaction is open.
        """
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        open_now = {"value": False}
        seen: List[tuple] = []

        original_transaction = store.transaction
        original_lock = store.lock_worksheet_proposal
        original_get = store.get_worksheet

        from contextlib import contextmanager

        @contextmanager
        def watched_transaction():
            open_now["value"] = True
            try:
                with original_transaction() as conn:
                    yield conn
            finally:
                open_now["value"] = False

        def watched_lock(proposal_id, owner):
            seen.append(("lock", open_now["value"]))
            return original_lock(proposal_id, owner)

        def watched_get(worksheet_id, owner, **kwargs):
            seen.append(("read_revision", open_now["value"]))
            return original_get(worksheet_id, owner, **kwargs)

        store.transaction = watched_transaction
        store.lock_worksheet_proposal = watched_lock
        store.get_worksheet = watched_get

        accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
               worksheet_id="ws-1", at="2026-01-02T00:00:00Z")

        assert ("lock", True) in seen, (
            f"the proposal was locked outside the transaction: {seen}")
        assert ("read_revision", True) in seen, (
            f"the authorizing revision was read outside the transaction: {seen}")
        assert not any(inside is False for _, inside in seen), (
            f"an authorizing read happened before the transaction: {seen}")


class TestRacingAcceptance:
    """Two sessions that both read PROPOSED, then both try to accept.

    The interleaving is forced rather than hoped for. An earlier version let
    two threads through a barrier and trusted them to collide; it caught the
    original defect once and then stopped reproducing, so reverting the fix
    left it green. A race that only sometimes races cannot discriminate.
    """

    def test_the_loser_waits_on_the_lock_and_then_refuses(self, store):
        """Deterministic: A holds the lock until B has certainly reached it."""
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        a_holds_the_lock = threading.Event()
        b_has_started = threading.Event()
        outcomes: List[object] = []
        lock = threading.Lock()

        def session_a():
            mine = WorkspaceStore(POSTGRES_URL)
            original = mine.save_worksheet

            def pause_then_write(worksheet):
                # Past the lock and inside the transaction.
                a_holds_the_lock.set()
                b_has_started.wait(timeout=10)
                return original(worksheet)

            mine.save_worksheet = pause_then_write
            try:
                with lock:
                    outcomes.append(("A", accept(
                        mine, proposal_id="wp-1", owner=OWNER,
                        proposal=proposal, worksheet_id="ws-1",
                        at="2026-01-02T00:00:00Z")))
            except Exception as exc:                     # noqa: BLE001
                with lock:
                    outcomes.append(("A", exc))

        def session_b():
            a_holds_the_lock.wait(timeout=10)
            mine = WorkspaceStore(POSTGRES_URL)
            b_has_started.set()
            try:
                result = accept(mine, proposal_id="wp-1", owner=OWNER,
                                proposal=proposal, worksheet_id="ws-1",
                                at="2026-01-02T00:00:00Z")
                with lock:
                    outcomes.append(("B", result))
            except Exception as exc:                     # noqa: BLE001
                with lock:
                    outcomes.append(("B", exc))

        threads = [threading.Thread(target=session_a),
                   threading.Thread(target=session_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        accepted = [(who, o) for who, o in outcomes
                    if not isinstance(o, Exception)]
        refused = [(who, o) for who, o in outcomes if isinstance(o, Exception)]
        assert len(accepted) == 1, f"expected one acceptance: {outcomes}"
        assert len(refused) == 1, f"expected one refusal: {outcomes}"
        assert isinstance(refused[0][1], ApplyRefused), (
            "the loser must receive a typed refusal, not a raw database "
            f"error: {refused[0][1]!r}")

    def test_exactly_one_acceptance_wins(self, store):
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        barrier = threading.Barrier(2)
        outcomes: List[object] = []
        lock = threading.Lock()

        def attempt():
            # A store of its own: a shared one would share a connection, and
            # the race would be between two threads on one session rather than
            # between two sessions.
            mine = WorkspaceStore(POSTGRES_URL)
            mine.get_worksheet_proposal("wp-1", OWNER)   # both read PROPOSED
            barrier.wait(timeout=30)
            try:
                result = accept(mine, proposal_id="wp-1", owner=OWNER,
                                proposal=proposal, worksheet_id="ws-1",
                                at="2026-01-02T00:00:00Z")
                with lock:
                    outcomes.append(result)
            except Exception as exc:                     # noqa: BLE001
                with lock:
                    outcomes.append(exc)

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        accepted = [o for o in outcomes if not isinstance(o, Exception)]
        refused = [o for o in outcomes if isinstance(o, Exception)]

        assert len(accepted) == 1, (
            f"expected one acceptance, got {len(accepted)}: {outcomes}")
        assert len(refused) == 1
        assert isinstance(refused[0], ApplyRefused), (
            "the loser must receive a typed refusal, not a raw database error: "
            f"{refused[0]!r}")

    def test_only_one_revision_exists(self, store):
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")
        self._race(proposal)

        rows = observe("SELECT revision FROM worksheet WHERE owner = %s "
                       "AND worksheet_id = %s ORDER BY revision", (OWNER, "ws-1"))
        assert [r["revision"] for r in rows] == [1, 2], (
            "one review must produce one new revision")

    def test_the_proposal_ends_accepted_once(self, store):
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")
        self._race(proposal)

        rows = observe("SELECT status, result_revision FROM worksheet_proposal "
                       "WHERE proposal_id = %s", ("wp-1",))
        assert len(rows) == 1
        assert rows[0]["status"] == ProposalStatus.ACCEPTED.value
        assert rows[0]["result_revision"] == 2

    def test_the_loser_leaves_no_candidate_work(self, store):
        """Rolling back only the status update would leave the loser's runs
        behind, and they would look like history belonging to nothing."""
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")
        self._race(proposal)

        rows = observe("SELECT run_id FROM plan_run WHERE run_id LIKE %s",
                       ("wp-1-run-%",))
        winner = observe("SELECT result_runs FROM worksheet_proposal "
                         "WHERE proposal_id = %s", ("wp-1",))[0]["result_runs"]
        assert len(rows) == len(winner or []), (
            "runs survive that no accepted proposal cites")

    @staticmethod
    def _race(proposal):
        barrier = threading.Barrier(2)
        errors: List[Exception] = []
        lock = threading.Lock()

        def attempt():
            mine = WorkspaceStore(POSTGRES_URL)
            mine.get_worksheet_proposal("wp-1", OWNER)
            barrier.wait(timeout=30)
            try:
                accept(mine, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                       worksheet_id="ws-1", at="2026-01-02T00:00:00Z")
            except Exception as exc:                     # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        return errors


class TestAtomicApply:
    """Failure after any durable step must leave nothing behind.

    Each assertion opens its own connection. Reading through the session that
    failed would show rolled-back in-memory state rather than what another
    process would observe, which is the only thing that matters here.
    """

    def scenario_proposal(self, store):
        """A proposal whose acceptance runs candidates, so there are runs to
        orphan. A derived-analysis proposal writes no runs and could not
        evidence the ordering."""
        worksheet = from_json(store.get_worksheet("ws-1", OWNER)["payload"])
        intent = plan("Try SPY, VTI and VT and keep the best", intent_id="i",
                      source_revision=worksheet.revision, history=[],
                      target_run="plan-1-run-0")
        return propose(intent, worksheet)

    def durable_state(self):
        return {
            "runs": [r["run_id"] for r in observe(
                "SELECT run_id FROM plan_run WHERE run_id LIKE %s",
                ("wp-1-%",))],
            "revisions": [r["revision"] for r in observe(
                "SELECT revision FROM worksheet WHERE owner = %s "
                "AND worksheet_id = %s", (OWNER, "ws-1"))],
            "status": [r["status"] for r in observe(
                "SELECT status FROM worksheet_proposal WHERE proposal_id = %s",
                ("wp-1",))],
            "derived": [r["event_id"] for r in observe(
                "SELECT event_id FROM confirmation_event WHERE event_id LIKE %s",
                ("wp-1-%",))],
        }

    @pytest.mark.parametrize("fail_after", [0, 1, 2])
    def test_failure_during_candidate_runs_leaves_nothing(self, store,
                                                          fail_after):
        """Inject the failure after run 1, run 2 and run 3 in turn."""
        proposal = self.scenario_proposal(store)
        assert proposal.proposed_scenario_patch is not None, (
            "this test needs an instruction that produces candidate runs; "
            "without them there is nothing for a failure to orphan")
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        calls = {"n": 0}

        def failing_runner(candidate):
            if calls["n"] == fail_after:
                calls["n"] += 1
                raise RuntimeError(f"injected failure at candidate {fail_after}")
            calls["n"] += 1
            return dict(RESULT)

        with pytest.raises(RuntimeError, match="injected failure"):
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=failing_runner)

        state = self.durable_state()
        assert state["runs"] == [], "candidate runs survived a failed apply"
        assert state["revisions"] == [1], "a revision survived a failed apply"
        assert state["status"] == [ProposalStatus.PROPOSED.value]

    def test_failure_while_persisting_the_revision_leaves_nothing(self, store):
        proposal = self.scenario_proposal(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        def failing(_worksheet):
            raise RuntimeError("injected failure persisting the revision")

        store.save_worksheet = failing
        with pytest.raises(RuntimeError, match="injected failure"):
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=lambda candidate: dict(RESULT))

        state = self.durable_state()
        assert state["runs"] == []
        assert state["revisions"] == [1]
        assert state["derived"] == []
        assert state["status"] == [ProposalStatus.PROPOSED.value]

    def test_failure_while_resolving_the_proposal_leaves_nothing(self, store):
        """The last durable step. A revision committed here without its status
        update would be an applied edit no proposal records."""
        proposal = self.scenario_proposal(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        def failing(proposal_id, owner, **kwargs):
            raise RuntimeError("injected failure resolving the proposal")

        store.resolve_worksheet_proposal = failing
        with pytest.raises(RuntimeError, match="injected failure"):
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=lambda candidate: dict(RESULT))

        state = self.durable_state()
        assert state["runs"] == []
        assert state["revisions"] == [1]
        assert state["status"] == [ProposalStatus.PROPOSED.value]

    def test_a_clean_retry_afterwards_succeeds_exactly_once(self, store):
        """Nothing from the failed attempt may block or duplicate the retry."""
        proposal = self.scenario_proposal(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        with pytest.raises(RuntimeError):
            accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
                   worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
                   run_candidate=lambda candidate: (_ for _ in ()).throw(
                       RuntimeError("injected")))

        result = accept(WorkspaceStore(POSTGRES_URL), proposal_id="wp-1",
                        owner=OWNER, proposal=proposal, worksheet_id="ws-1",
                        at="2026-01-03T00:00:00Z",
                        run_candidate=lambda candidate: dict(RESULT))
        assert result.status == ProposalStatus.ACCEPTED.value

        state = self.durable_state()
        assert state["revisions"] == [1, 2]
        assert state["status"] == [ProposalStatus.ACCEPTED.value]

    def test_the_commit_order_is_visible_in_the_database(self, store):
        """Asserted from the durable record, not from an instrumented Python
        call sequence: the question is what another process can observe."""
        proposal = self.scenario_proposal(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")
        accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
               worksheet_id="ws-1", at="2026-01-02T00:00:00Z",
               run_candidate=lambda candidate: dict(RESULT))

        cited = observe("SELECT payload FROM worksheet WHERE owner = %s AND "
                        "worksheet_id = %s AND revision = 2",
                        (OWNER, "ws-1"))[0]["payload"]
        stored_runs = {r["run_id"] for r in observe(
            "SELECT run_id FROM plan_run", ())}
        for reference in cited.get("benchmark_run_refs", []):
            assert reference in stored_runs, (
                f"revision 2 cites {reference}, which is not in the database — "
                "a dangling reference the write ordering exists to prevent")


class TestStaleAcceptanceIsSettledInsideTheLock:
    """A pre-lock check still races. The authorizing read must happen after."""

    def test_a_revision_committed_after_the_read_is_detected(self, store):
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")

        # Session A has read the proposal against revision 1.
        session_a = WorkspaceStore(POSTGRES_URL)
        session_a.get_worksheet_proposal("wp-1", OWNER)

        # Session B advances the worksheet and commits.
        session_b = WorkspaceStore(POSTGRES_URL)
        current = from_json(session_b.get_worksheet("ws-1", OWNER)["payload"])
        session_b.save_worksheet(revise(current, reason="independent edit",
                                        created_at="2026-01-01T12:00:00Z"))
        assert observe("SELECT MAX(revision) AS r FROM worksheet "
                       "WHERE owner = %s", (OWNER,))[0]["r"] == 2

        # Session A now attempts acceptance. The revision it was reviewed
        # against is gone, and the refusal must come from re-reading inside the
        # transaction rather than from the stale value it already holds.
        with pytest.raises(StaleProposal):
            accept(session_a, proposal_id="wp-1", owner=OWNER,
                   proposal=proposal, worksheet_id="ws-1",
                   at="2026-01-02T00:00:00Z")

        assert observe("SELECT MAX(revision) AS r FROM worksheet "
                       "WHERE owner = %s", (OWNER,))[0]["r"] == 2, (
            "a refused acceptance must not have written a revision")

    def test_the_proposal_is_still_proposed_after_a_stale_refusal(self, store):
        """Refusing is not resolving. The user may re-plan against the current
        revision, and the proposal must still be there to see."""
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")
        session_b = WorkspaceStore(POSTGRES_URL)
        current = from_json(session_b.get_worksheet("ws-1", OWNER)["payload"])
        session_b.save_worksheet(revise(current, reason="independent edit",
                                        created_at="2026-01-01T12:00:00Z"))

        with pytest.raises(StaleProposal):
            accept(WorkspaceStore(POSTGRES_URL), proposal_id="wp-1",
                   owner=OWNER, proposal=proposal, worksheet_id="ws-1",
                   at="2026-01-02T00:00:00Z")

        rows = observe("SELECT status FROM worksheet_proposal "
                       "WHERE proposal_id = %s", ("wp-1",))
        assert rows[0]["status"] == ProposalStatus.PROPOSED.value


class TestNoRawDatabaseErrorEscapes:
    def test_a_second_acceptance_is_a_typed_refusal(self, store):
        """A unique-violation reaching the caller would leak constraint names
        and table structure into an application error path."""
        proposal = proposal_for(store)
        store.save_worksheet_proposal(
            proposal_id="wp-1", owner=OWNER, worksheet_id="ws-1",
            proposal=proposal, created_at="2026-01-01T00:00:00Z")
        accept(store, proposal_id="wp-1", owner=OWNER, proposal=proposal,
               worksheet_id="ws-1", at="2026-01-02T00:00:00Z")

        with pytest.raises(ApplyRefused) as caught:
            accept(WorkspaceStore(POSTGRES_URL), proposal_id="wp-1",
                   owner=OWNER, proposal=proposal, worksheet_id="ws-1",
                   at="2026-01-02T00:00:00Z")
        message = str(caught.value)
        for leak in ("psycopg", "DETAIL:", "pg_", "CONSTRAINT", "relation"):
            assert leak not in message, f"{leak!r} leaked into {message!r}"
