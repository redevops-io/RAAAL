"""Every production path that persists a run, and what it may claim.

The write guard refuses a run with no provenance. That is not sufficient on its
own: a live path can satisfy it by labelling its own omission as legacy, and the
stored record would then say "nobody recorded this" when in fact this code
declined to. The two are indistinguishable afterwards, which is exactly the
reconstruction the design refuses everywhere else.

So callers are classified, and only an import path may claim legacy absence.

The caller list is read from the call graph, not from the classification beside
it — a hand-kept list would let a new caller pass by never appearing, which is
the coverage failure this codebase has now hit five times.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.market_data.producers import (
    CallerKind,
    RUN_CALLERS,
    live_callers,
    unclassified_callers,
)
from src.market_data.provenance import ProvenanceStatus, from_json

POLICY = "PILOT_DATA_POLICY"


def functions_calling_record_run():
    """Every function under `src/` that calls `record_run`, from the AST.

    Parsed rather than grepped: a docstring mentioning `record_run` is not a
    call, and this codebase has produced eleven defects from checks that could
    not tell the difference.
    """
    found = {}
    for path in sorted(Path("src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "record_run"):
                    found[node.name] = str(path)
    return found


class TestEveryCallerIsClassified:
    def test_none_is_unclassified(self):
        callers = functions_calling_record_run()
        missing = unclassified_callers(sorted(callers))
        assert missing == (), (
            f"these persist runs and are not classified: "
            f"{[(name, callers[name]) for name in missing]}")

    def test_the_list_comes_from_the_call_graph(self):
        """A hand-kept list would let a new caller pass by never appearing."""
        found = functions_calling_record_run()
        assert found, "the AST scan found no callers at all"
        for name, module in found.items():
            assert RUN_CALLERS[name].module == module, (
                f"{name} is declared in {RUN_CALLERS[name].module} and found "
                f"in {module}")

    def test_each_classification_records_why(self):
        for caller in RUN_CALLERS.values():
            assert caller.reason.strip(), caller.name

    def test_both_production_callers_are_present(self):
        """Given the coverage lesson: assert the scan saw the ones we know.

        `_apply`, not `accept`. `accept` validates and delegates; the call
        graph is what says which function writes, and the first version of the
        classification named the wrong one.
        """
        found = set(functions_calling_record_run())
        assert {"generate", "_apply"} <= found


class TestNoLiveCallerMayClaimLegacyAbsence:
    def test_the_live_callers_are_market_derived(self):
        for caller in live_callers():
            assert caller.kind is CallerKind.MARKET_DERIVED, caller.name

    def test_no_live_caller_may_claim_legacy(self):
        assert not any(one.may_claim_legacy for one in live_callers())

    def test_nothing_currently_claims_legacy(self):
        """The kind classifies nothing today.

        `apply_import` was listed here until the call-graph scan showed it
        never calls `record_run` — the transfer tool writes below the store
        with raw SQL, which is correct for a migration that must carry a legacy
        row through unchanged. The kind stays because the distinction is what
        stops a live caller labelling its own omission as legacy.
        """
        legacy = [one.name for one in RUN_CALLERS.values()
                  if one.may_claim_legacy]
        assert legacy == []

    def test_the_import_path_is_not_reachable_from_a_request(self):
        """If a route ever calls the transfer tool, a request could write a
        row the store never inspected."""
        for path in (Path("src/workspace/routes.py"), Path("src/web/routes.py"),
                     Path("src/api.py")):
            assert "apply_import" not in path.read_text(), path


class TestALiveCallerStoresRealProvenance:
    """What the classification claims, checked against what is stored."""

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        from src.workspace.store import WorkspaceStore

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        return WorkspaceStore(tmp_path / "w.db")

    def test_generate_stores_the_resolver_record(self, store, monkeypatch):
        from src.market_data.access import resolve
        from src.workspace.generate import generate

        from tests.test_producer_inventory import TestInstanceCompleteness

        inventory = TestInstanceCompleteness()
        scenario = inventory.scenario()
        store.save_plan(plan_id="p-1", owner="alice", scenario=scenario,
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")

        access = resolve(context="a run", accessed_at="2026-01-01T00:00:00Z")
        generate(store, plan_id="p-1", owner="alice", scenario=scenario,
                 run={"modelling_scope": {"excludes": []}, "final_value": 1.0,
                      "market_data": access.provenance.to_json()},
                 comparison={}, ran_at="2026-01-01T00:00:00Z", access=access)

        runs = store.runs_for("p-1", "alice")
        assert runs
        carried = from_json(runs[0]["result"]["market_data"])
        assert carried.status is ProvenanceStatus.RECORDED
        assert carried.identifies_data
        assert carried.snapshot_id == access.provenance.snapshot_id

class TestGenerateEnforcesItsOwnClassification:
    """A live producer may not downgrade its omission into legacy absence.

    The store's guard accepts NOT_RECORDED, because a migration must carry a
    legacy row through unchanged. That leniency is right there and wrong here:
    it let `generate` satisfy the guard while storing an absence it had chosen,
    and the record would read as "nobody recorded this" rather than "this code
    declined to".
    """

    @pytest.fixture
    def prepared(self, tmp_path, monkeypatch):
        from src.workspace.store import WorkspaceStore

        from tests.test_producer_inventory import TestInstanceCompleteness

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        store = WorkspaceStore(tmp_path / "w.db")
        inventory = TestInstanceCompleteness()
        scenario = inventory.scenario()
        store.save_plan(plan_id="p-1", owner="alice", scenario=scenario,
                        stated_text="x", saved_at="2026-01-01T00:00:00Z")
        return store, scenario

    def run_with(self, prepared, market_data, omit=False):
        from src.market_data.access import resolve
        from src.workspace.generate import generate

        store, scenario = prepared
        body = {"modelling_scope": {"excludes": []}, "final_value": 1.0}
        if not omit:
            body["market_data"] = market_data
        # A real delivery, so these cases exercise the provenance guard rather
        # than tripping the delivery guard on their way to it.
        access = resolve(context="a run", accessed_at="2026-01-01T00:00:00Z")
        return generate(store, plan_id="p-1", owner="alice", scenario=scenario,
                        run=body, comparison={},
                        ran_at="2026-01-01T00:00:00Z", access=access)

    def test_an_omitted_record_is_refused(self, prepared):
        from src.workspace.generate import UnattributableRun

        with pytest.raises(UnattributableRun, match="no market-data"):
            self.run_with(prepared, None, omit=True)

    def test_a_claimed_legacy_absence_is_refused(self, prepared):
        """The precise downgrade this exists to stop."""
        from src.market_data.provenance import not_recorded
        from src.workspace.generate import UnattributableRun

        with pytest.raises(UnattributableRun, match="never recorded"):
            self.run_with(prepared, not_recorded("no reason").to_json())

    def test_a_denied_decision_is_refused(self, prepared):
        from src.workspace.generate import UnattributableRun

        with pytest.raises(UnattributableRun):
            self.run_with(prepared, {
                "status": "RECORDED", "snapshot_id": "s-1",
                "content_digest": "mdv1:aaa", "access_decision": "DENIED",
                "accessed_at": "2026-01-01T00:00:00Z"})

    def test_a_label_without_a_digest_is_refused(self, prepared):
        from src.workspace.generate import UnattributableRun

        with pytest.raises(UnattributableRun):
            self.run_with(prepared, {
                "status": "RECORDED", "snapshot_id": "prices-2026-01",
                "access_decision": "SYNTHETIC_ALLOWED",
                "accessed_at": "2026-01-01T00:00:00Z"})

    def test_nothing_is_persisted_when_it_refuses(self, prepared):
        from src.workspace.generate import UnattributableRun

        store, _ = prepared
        with pytest.raises(UnattributableRun):
            self.run_with(prepared, None, omit=True)
        assert store.runs_for("p-1", "alice") == []

    def test_a_resolver_record_is_accepted(self, prepared):
        from src.market_data.access import resolve

        access = resolve(context="a run", accessed_at="2026-01-01T00:00:00Z")
        assert self.run_with(prepared, access.provenance.to_json()) is not None


class TestTheCandidatePathCarriesTheRealRecord:
    """Candidate runs are independent artifacts.

    A worksheet citing three of them must be able to say which data each used,
    even while they happen to share one access. This was wired and untested:
    dropping the candidate path back to NOT_RECORDED changed no result, which
    means the wiring was decoration until these existed.
    """

    @pytest.fixture
    def seeded(self, tmp_path, monkeypatch):
        from src.mission.compiler import compile_scenario
        from src.mission.scenario import ScenarioSpecification
        from src.mission.spec import Inference, Provenance
        from src.workspace.store import WorkspaceStore
        from src.workspace.worksheet import create

        monkeypatch.setenv(POLICY, "SYNTHETIC_ONLY")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL",
                           f"sqlite:///{tmp_path}/w.db")
        store = WorkspaceStore(tmp_path / "w.db")

        compiled = compile_scenario(
            "I put $2,000 into SPY every month in my Roth IRA, on the first "
            "trading day of the period, reinvesting the dividends, and I never "
            "sell.", name="plan-1", version=1,
            benchmark_rule="benchmark-policy/public-default@1")
        p = compiled.scenario.provenance
        scenario = ScenarioSpecification(**{
            **compiled.scenario.__dict__,
            "provenance": Provenance(
                stated=p.stated,
                inferred=tuple(Inference(i.field, i.value, i.why,
                                         confirmed=True) for i in p.inferred),
                contradictions=p.contradictions, unresolved=())})
        store.save_plan(plan_id="plan-1", owner="pilot", scenario=scenario,
                        stated_text=compiled.scenario.stated_text
                        if hasattr(compiled.scenario, "stated_text")
                        else "I put $2,000 into SPY every month in my Roth IRA, "
                             "on the first trading day of the period, "
                             "reinvesting the dividends, and I never sell.",
                        saved_at="2026-01-01T00:00:00Z")
        store.save_worksheet(create(
            worksheet_id="ws-1", owner_id="pilot", scenario_ref="plan-1",
            primary_run_ref="r-0", created_at="2026-01-01T00:00:00Z"))
        return store

    def test_a_candidate_result_carries_the_access_it_used(self, seeded):
        from src.market_data.access import resolve
        from src.workspace.routes import _candidate_runner

        access = resolve(context="candidate runs",
                         accessed_at="2026-01-01T00:00:00Z")
        runner = _candidate_runner(access, seeded, "ws-1")
        result = runner(["SPY"])

        carried = from_json(result["market_data"])
        assert carried.status is ProvenanceStatus.RECORDED, (
            "the candidate stored an absence while the access that produced it "
            "was in hand")
        assert carried.identifies_data
        assert carried.snapshot_id == access.provenance.snapshot_id
        assert carried.content_digest == access.provenance.content_digest

    def test_every_candidate_cites_the_same_access(self, seeded):
        """SHARED_ACCESS: one resolver call above the loop, and every candidate
        proven to cite it rather than assumed to."""
        from src.market_data.access import resolve
        from src.workspace.routes import _candidate_runner

        access = resolve(context="candidate runs",
                         accessed_at="2026-01-01T00:00:00Z")
        runner = _candidate_runner(access, seeded, "ws-1")
        # Instruments the synthetic snapshot actually covers. A candidate with
        # no price history is a data gap and raises, which is correct and is
        # not what this test is about.
        records = [from_json(runner([ticker])["market_data"])
                   for ticker in ("SPY", "SH", "TLT")]

        assert len({one.snapshot_id for one in records}) == 1
        assert len({one.content_digest for one in records}) == 1
        assert len({one.access_decision for one in records}) == 1
        assert all(one.identifies_data for one in records)

    def test_the_run_helper_refuses_a_bare_frame(self, seeded):
        """The type guard that stops provenance being attached afterwards by a
        caller who might forget."""
        from src.market_data.access import resolve
        from src.workspace.routes import _run

        access = resolve(context="a run", accessed_at="2026-01-01T00:00:00Z")
        with pytest.raises(TypeError, match="not the frame alone"):
            _run(object(), access.frame)

    def test_a_run_built_by_the_helper_carries_the_record(self, seeded):
        from src.market_data.access import resolve
        from src.mission.compiler import compile_scenario
        from src.workspace.routes import _run

        access = resolve(context="a run", accessed_at="2026-01-01T00:00:00Z")
        compiled = compile_scenario(
            "I put $2,000 into SPY every month in my Roth IRA, on the first "
            "trading day of the period, reinvesting the dividends, and I never "
            "sell.", name="plan-1", version=1,
            benchmark_rule="benchmark-policy/public-default@1")
        outcome = _run(compiled.scenario, access)
        if outcome.get("result") is None:
            pytest.skip("no price history for this scenario in the fixture")
        assert outcome["result"].market_data is access.provenance
