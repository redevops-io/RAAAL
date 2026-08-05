"""Telemetry is recorded by the deployed path, and can still fail freely.

`tests/test_telemetry.py` proves the mechanism across twenty-five cases: spans
nest, decisions name what they rejected, a deleted database costs a trace and
not an edit, one tenant sees none of another's. Every one of those builds a
`Recorder` *with* a store in its own fixture.

Nothing in `src/` ever built one. The single production entry point called
`plan_and_record(...)` without a recorder, and that function substituted
`Recorder(store=None)` — so every span, trace and decision the runtime
assembled was dropped, and `TraceStore` was constructed nowhere outside the
module that defines it.

**That made the independence claim vacuous rather than true.** "Deleting every
trace changes nothing about what a worksheet means" holds trivially when there
are no traces. The property Gate 6 has to establish is the harder one: with
telemetry *actually recording*, breaking it must still cost nothing.

So this lane asks two questions the mechanism lane structurally cannot:

    reachability   does a real request write a real trace
    independence   with recording live, does breaking it change any outcome
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.workspace.store import WorkspaceStore
from tests.test_workspace import COMPLETE

OWNER = "pilot"
AT = "2026-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A deployment whose trace store is a real file in a temp directory."""
    from src.deploy.context import bind, resolve, unbind

    traces = tmp_path / "trace.db"
    context = resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY",
                       "QUANTIFY_TRACE_PATH": str(traces)})
    bind(context)
    try:
        yield context, traces
    finally:
        unbind()


def build_workspace(path):
    """One worksheet, ready for an instruction."""
    from src.mission.compiler import compile_scenario
    from src.mission.scenario import ScenarioSpecification
    from src.mission.spec import Inference, Provenance
    from src.workspace.worksheet import create

    store = WorkspaceStore(path)
    compiled = compile_scenario(
        "I put $2,000 into SPY every month in my Roth IRA, on the first "
        "trading day of the period, reinvesting the dividends, and I never "
        "sell.", name="plan-1", version=1,
        benchmark_rule="benchmark-policy/public-default@1")
    source = compiled.scenario.provenance
    scenario = ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=source.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in source.inferred),
            contradictions=source.contradictions, unresolved=())})
    store.save_plan(plan_id="plan-1", owner=OWNER, scenario=scenario,
                    stated_text="seed", saved_at=AT)
    store.record_run(run_id="run-0", plan_id="plan-1", ran_at=AT, owner=OWNER,
                     result={"modelling_scope": {"excludes": []},
                             "market_data": {"status": "NOT_APPLICABLE"}},
                     comparison={})
    store.save_worksheet(create(worksheet_id="ws-1", owner_id=OWNER,
                                scenario_ref="plan-1", primary_run_ref="run-0",
                                created_at=AT))
    return store


@pytest.fixture
def planned(tmp_path, deployment):
    _, traces = deployment
    return build_workspace(tmp_path / "w.db"), traces


def rows(traces, table):
    import sqlite3

    if not traces.exists():
        return []
    conn = sqlite3.connect(traces)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


_SEQUENCE = iter(range(1, 10_000))


def plan_through_the_route(store, worksheet_id="ws-1",
                           instruction="Replace SPY with VTI", ids=None):
    """The route's own call, including how it builds the recorder.

    Ids are unique per call. Reusing them made the second invocation hit the
    uniqueness constraint on `worksheet_intent` — the workspace correctly
    refusing a duplicate intent — and the failure looked like a telemetry
    defect while being the store doing its job.
    """
    import src.workspace.routes as routes
    from src.workspace.intent_service import plan_and_record

    index = ids if ids is not None else next(_SEQUENCE)
    return plan_and_record(
        store, worksheet_id=worksheet_id, owner=OWNER,
        instruction=instruction, intent_id=f"{worksheet_id}-i{index}",
        proposal_id=f"{worksheet_id}-p{index}", at=AT,
        recorder=routes._recorder(worksheet_id=worksheet_id))


class TestTheProductionPathRecords:
    def test_a_trace_is_written(self, planned):
        store, traces = planned
        plan_through_the_route(store)
        assert rows(traces, "trace"), (
            "the deployed path recorded no trace; the recorder was built "
            "without a store, which is what made every telemetry test vacuous "
            "in production")

    def test_spans_are_written(self, planned):
        store, traces = planned
        plan_through_the_route(store)
        assert len(rows(traces, "span")) > 1

    def test_decisions_are_written(self, planned):
        store, traces = planned
        plan_through_the_route(store)
        assert rows(traces, "decision")

    def test_the_trace_is_scoped_to_the_tenant(self, planned):
        store, traces = planned
        plan_through_the_route(store)
        assert {row["tenant"] for row in rows(traces, "trace")} == {OWNER}

    def test_the_trace_names_the_worksheet(self, planned):
        store, traces = planned
        plan_through_the_route(store)
        assert {row["worksheet_id"] for row in rows(traces, "trace")} == {"ws-1"}

    def test_no_write_failed(self, planned):
        """`Recorder.failures` counts writes that did not land. A path that
        recorded nothing *and* reported no failures is a path with no store."""
        import src.workspace.routes as routes

        store, _ = planned
        recorder = routes._recorder(worksheet_id="ws-1")
        assert recorder.store is not None, (
            "the deployment resolved no trace store")
        from src.workspace.intent_service import plan_and_record

        plan_and_record(store, worksheet_id="ws-1", owner=OWNER,
                        instruction="Replace SPY with VTI", intent_id="i",
                        proposal_id="p", at=AT, recorder=recorder)
        assert recorder.failures == 0


class TestBreakingLiveTelemetryCostsNothing:
    """The claim the mechanism lane could not make.

    Each case plans the *same* instruction against two identical, independent
    workspaces — one recording normally, one with telemetry broken — and
    requires the answers to match.

    Not two sequential calls against one workspace. Planning twice legitimately
    differs: the trial total counts the first attempt, so the second is a
    different situation and comparing them measures the system's own memory
    rather than telemetry's effect on it. That comparison failed, and it was
    right to.
    """

    def plan_in(self, path):
        """Plan the same instruction, with the same ids, in a fresh workspace.

        Fixed ids rather than stripping them afterwards: the stores are
        independent, so both runs can legitimately use them, and a comparison
        that had to exclude fields would be a comparison that could quietly
        exclude the field that mattered.
        """
        store = build_workspace(path)
        rendered = plan_through_the_route(store, ids="fixed").to_json()
        return {key: value for key, value in rendered.items()
                if key != "trace_id"}, store

    def witness(self, traces):
        """Prove the mechanism was active before breaking it.

        A failure-tolerance claim needs a *premise witness*: "deleting the
        traces changed nothing" is free if there were none. This whole gate
        exists because that premise went unstated for the life of the
        telemetry suite, so each case below asserts it for itself rather than
        relying on a sibling test having established it.
        """
        assert rows(traces, "trace"), (
            "no trace existed to destroy; this case would have passed without "
            "telemetry ever running")
        assert rows(traces, "span")

    def test_a_deleted_database_mid_flight(self, tmp_path, deployment):
        _, traces = deployment
        healthy, _ = self.plan_in(tmp_path / "a.db")
        self.witness(traces)

        traces.unlink()
        traces.mkdir()          # a directory where a file must be: unopenable
        broken, _ = self.plan_in(tmp_path / "b.db")
        assert broken == healthy

    def test_a_read_only_store(self, tmp_path, deployment):
        _, traces = deployment
        healthy, _ = self.plan_in(tmp_path / "a.db")
        self.witness(traces)

        traces.chmod(0o400)
        try:
            broken, _ = self.plan_in(tmp_path / "b.db")
        finally:
            traces.chmod(0o600)
        assert broken == healthy

    def test_a_store_that_raises_on_every_write(self, tmp_path, deployment,
                                                 monkeypatch):
        import src.telemetry.trace_store as store_module

        _, traces = deployment
        healthy, _ = self.plan_in(tmp_path / "a.db")
        self.witness(traces)

        def refuse(*args, **kwargs):
            raise RuntimeError("the trace store is unavailable")

        for name in ("start_trace", "end_trace", "record_span",
                     "record_decision"):
            monkeypatch.setattr(store_module.TraceStore, name, refuse)
        broken, _ = self.plan_in(tmp_path / "b.db")
        assert broken == healthy

    def test_the_failures_are_counted_rather_than_silent(self, planned,
                                                          monkeypatch):
        """Swallowed and *unreported* would make a dead trace store
        indistinguishable from a quiet one — which is exactly the state this
        gate found the system in."""
        import src.telemetry.trace_store as store_module
        import src.workspace.routes as routes
        from src.workspace.intent_service import plan_and_record

        store, _ = planned

        def refuse(*args, **kwargs):
            raise RuntimeError("the trace store is unavailable")

        monkeypatch.setattr(store_module.TraceStore, "record_span", refuse)
        recorder = routes._recorder(worksheet_id="ws-1")
        plan_and_record(store, worksheet_id="ws-1", owner=OWNER,
                        instruction="Replace SPY with VTI", intent_id="i",
                        proposal_id="p", at=AT, recorder=recorder)
        assert recorder.failures > 0

    def test_the_persisted_artifacts_match_too(self, tmp_path, deployment,
                                                monkeypatch):
        """Not only the returned payload: what landed in the workspace."""
        import src.telemetry.trace_store as store_module

        _, traces = deployment
        _, healthy_store = self.plan_in(tmp_path / "a.db")
        self.witness(traces)
        expected = healthy_store.worksheet_intents("ws-1", OWNER)

        monkeypatch.setattr(store_module.TraceStore, "record_span",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("unavailable")))
        _, broken_store = self.plan_in(tmp_path / "b.db")
        found = broken_store.worksheet_intents("ws-1", OWNER)

        assert len(found) == len(expected)
        assert [row["edit_effect"] for row in found] == \
            [row["edit_effect"] for row in expected]


class TestTheDeploymentDecidesWhetherToRecord:
    def test_recording_is_on_by_default(self):
        from src.deploy.context import resolve

        assert resolve({}).telemetry.enabled

    def test_an_empty_path_disables_it(self):
        from src.deploy.context import resolve

        assert not resolve({"QUANTIFY_TRACE_PATH": ""}).telemetry.enabled

    def test_a_disabled_deployment_still_serves(self, planned, monkeypatch):
        from src.deploy.context import bind, resolve, unbind

        store, traces = planned
        bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY",
                      "QUANTIFY_TRACE_PATH": ""}))
        try:
            import src.workspace.routes as routes

            assert routes._recorder(worksheet_id="ws-1").store is None
            assert plan_through_the_route(store) is not None
        finally:
            unbind()

    def test_an_unopenable_path_disables_it_rather_than_raising(self, tmp_path):
        """A read-only volume must cost telemetry, not the deployment."""
        from src.deploy.context import resolve

        blocked = tmp_path / "no-such-directory" / "x" / "trace.db"
        (tmp_path / "no-such-directory").mkdir()
        (tmp_path / "no-such-directory").chmod(0o500)
        try:
            assert resolve({"QUANTIFY_TRACE_PATH": str(blocked)}).telemetry.store() \
                is None
        finally:
            (tmp_path / "no-such-directory").chmod(0o700)

    def test_the_startup_proof_reports_whether_it_is_on(self):
        from src.deploy.context import resolve

        rendered = resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}).to_json()
        assert rendered["telemetry"]["enabled"] is True

    def test_the_proof_carries_no_path(self):
        """A filesystem path is a deployment detail, and `to_json` is logged."""
        from src.deploy.context import resolve

        rendered = resolve({"QUANTIFY_TRACE_PATH": "/srv/secret/trace.db"})
        assert "/srv/secret" not in str(rendered.to_json())


class TestTheStoresStayApart:
    def test_the_trace_store_is_not_the_workspace_database(self, planned):
        from src.deploy.context import current

        store, traces = planned
        assert str(traces) != current().database.url
        assert current().telemetry.path != current().database.url

    def test_telemetry_is_resolved_from_its_own_variable(self):
        """One target for both would make retention a per-table convention and
        put deletable rows in the transaction that writes permanent ones."""
        from src.deploy.context import resolve

        context = resolve({"QUANTIFY_DATABASE_URL": "sqlite:///w.db",
                           "QUANTIFY_TRACE_PATH": "t.db"})
        assert context.telemetry.path == "t.db"
        assert context.database.url == "sqlite:///w.db"

    def test_no_production_module_reads_the_trace_path_itself(self):
        """The same single-resolution rule as every other identity."""
        readers = []
        for path in sorted(Path("src").rglob("*.py")):
            if str(path) == "src/deploy/context.py":
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:                                  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and \
                        node.value == "QUANTIFY_TRACE_PATH":
                    readers.append(str(path))
        assert readers == [], readers


class TestRetentionIsReachable:
    """`purge_before` exists and nothing schedules it. That is recorded here
    rather than left implicit: a retention policy no path performs is a
    declaration, and this codebase has spent its time removing those."""

    def test_the_deployment_states_a_retention_period(self):
        from src.deploy.context import resolve

        assert resolve({}).telemetry.retention_days == 90

    def test_it_can_be_configured(self):
        from src.deploy.context import resolve

        assert resolve({"QUANTIFY_TRACE_RETENTION_DAYS": "7"}) \
            .telemetry.retention_days == 7

    def test_a_malformed_value_keeps_the_default(self):
        """Telemetry is the expendable half; a bad retention setting must not
        stop a deployment serving."""
        from src.deploy.context import resolve

        for bad in ("garbage", "", "-1", "0"):
            assert resolve({"QUANTIFY_TRACE_RETENTION_DAYS": bad}) \
                .telemetry.retention_days == 90

    def test_purging_works_when_it_is_called(self, planned):
        store, traces = planned
        plan_through_the_route(store)
        assert rows(traces, "trace")

        from src.telemetry.trace_store import TraceStore

        purged = TraceStore(traces).purge_before("2030-01-01T00:00:00Z")
        assert purged["traces"] >= 1
        assert rows(traces, "trace") == []

    def test_an_operator_can_perform_the_purge(self):
        """A command, not a scheduler.

        A closed pilot's trace volume is small and a cron entry calling this is
        the whole requirement; a generalized scheduling subsystem is post-pilot
        work. What matters is that the retention period the deployment resolves
        is *performable* rather than only stated.
        """
        callers = []
        for path in sorted(Path("src").rglob("*.py")):
            if "trace_store" in str(path):
                continue
            if "purge_before" in path.read_text():
                callers.append(str(path))
        assert callers == ["src/telemetry/purge.py"], callers

    def test_the_command_uses_the_deployment_s_retention(self, planned):
        from src.telemetry.purge import cutoff_for

        import datetime as dt

        from src.deploy.context import current

        days = current().telemetry.retention_days
        now = dt.datetime(2026, 8, 4, tzinfo=dt.timezone.utc)
        assert cutoff_for(days, now=now).startswith("2026-05-06")

    def test_a_dry_run_deletes_nothing(self, planned, capsys):
        from src.telemetry.purge import main

        store, traces = planned
        plan_through_the_route(store)
        before = len(rows(traces, "trace"))
        assert before, "nothing to purge; this case would prove nothing"

        assert main(["--dry-run"]) == 0
        assert len(rows(traces, "trace")) == before
        assert "would purge" in capsys.readouterr().out

    def test_a_disabled_deployment_purges_nothing_and_succeeds(self, capsys):
        from src.deploy.context import bind, resolve, unbind
        from src.telemetry.purge import main

        bind(resolve({"QUANTIFY_TRACE_PATH": ""}))
        try:
            assert main([]) == 0
        finally:
            unbind()
        assert "disabled" in capsys.readouterr().out


class TestTheHttpRouteItselfRecords:
    """Through the real route, not through a helper that repeats its wiring.

    The first version of this file called `routes._recorder(...)` directly and
    handed the result to `plan_and_record` — reproducing what the route does
    rather than exercising it. Deleting `recorder=...` from the route changed
    nothing, and the falsification pass caught it: a test that rebuilds the
    call site it is meant to verify is testing its own copy.

    Sixth instance of that class in this codebase, and the second written by
    someone who had just documented it.
    """

    @pytest.fixture
    def client(self, tmp_path, deployment, monkeypatch):
        from fastapi.testclient import TestClient

        _, traces = deployment
        workspace = tmp_path / "w.db"
        build_workspace(workspace)

        import src.workspace.routes as routes

        # The route resolves its own store from the deployment; point that at
        # the workspace this fixture built.
        monkeypatch.setattr(routes, "_store", lambda: WorkspaceStore(workspace))

        from fastapi import FastAPI

        from src.web.failure import install

        app = FastAPI()
        install(app)
        app.include_router(routes.router)
        return TestClient(app, raise_server_exceptions=False), traces

    def test_a_post_writes_a_trace(self, client):
        http, traces = client
        response = http.post("/workspace/research/ws-1/intent",
                             data={"instruction": "Replace SPY with VTI"})
        assert response.status_code == 200, response.text
        assert rows(traces, "trace"), (
            "the HTTP route recorded no trace; the recorder it builds is not "
            "reaching the service")

    def test_the_post_writes_spans(self, client):
        http, traces = client
        http.post("/workspace/research/ws-1/intent",
                  data={"instruction": "Replace SPY with VTI"})
        assert len(rows(traces, "span")) > 1

    def test_the_trace_names_the_worksheet_from_the_path(self, client):
        http, traces = client
        http.post("/workspace/research/ws-1/intent",
                  data={"instruction": "Replace SPY with VTI"})
        assert {row["worksheet_id"] for row in rows(traces, "trace")} == {"ws-1"}

    def test_the_response_carries_the_trace_id(self, client):
        """So a support conversation can reach the trace without the user
        quoting anything internal."""
        http, traces = client
        body = http.post("/workspace/research/ws-1/intent",
                         data={"instruction": "Replace SPY with VTI"}).json()
        assert body.get("trace_id")
        assert body["trace_id"] in {row["trace_id"] for row in rows(traces, "trace")}

    def test_a_broken_store_still_serves_the_request(self, client, monkeypatch):
        import src.telemetry.trace_store as store_module

        http, _ = client
        monkeypatch.setattr(
            store_module.TraceStore, "record_span",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unavailable")))
        response = http.post("/workspace/research/ws-1/intent",
                             data={"instruction": "Replace SPY with VOO"})
        assert response.status_code == 200, response.text


class TestThePlanBuilderRecords:
    """The routes a pilot user actually walks.

    Gate 6 fixed "the only production entry point" and that was true when it
    was written. The Plan Builder arrived afterwards and did not carry the
    wiring across, so `GET /workspace/new` and `POST /workspace/save` — every
    step of the journey an invited user takes — built no recorder at all. A
    day of live traffic on `quantify.club` produced zero rows in all three
    tables while the class above passed.

    That is the reachability axis again, and it is the reason this class
    asserts on the *store* rather than on the route: "the recorder was
    constructed" is a claim about the code, and the code was never in doubt.
    The observable property is that a journey leaves rows behind.

    Recorded as OBS-4 and wired under a named exception to the freeze: product
    behaviour is frozen, instrumentation needed to evaluate the pilot is not.
    An experiment whose measurement channel is disconnected cannot answer the
    question it was started to ask.
    """

    RUNNABLE = "I buy VOO every month and never sell."
    UNCLEAR = "I put money into tech every so often."

    @pytest.fixture
    def client(self, tmp_path, deployment, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api
        import src.workspace.routes as routes

        _, traces = deployment
        store = WorkspaceStore(tmp_path / "workspace.db")
        monkeypatch.setattr(routes, "_store", lambda: store)
        api._bootstrap()
        return TestClient(api.app), traces

    def draft(self, http, describe):
        response = http.get("/workspace/new", params={"describe": describe})
        assert response.status_code == 200, response.text
        return response.text

    def test_the_empty_form_opens_no_trace(self, client):
        """A blank page is not a journey. Recording one would make the trace
        count a page-view metric and the question 'how many people described
        something' unanswerable."""
        http, traces = client
        assert http.get("/workspace/new").status_code == 200
        assert rows(traces, "trace") == []

    def test_describing_something_writes_a_trace(self, client):
        http, traces = client
        self.draft(http, self.RUNNABLE)
        assert rows(traces, "trace"), (
            "GET /workspace/new recorded nothing; the Plan Builder is not "
            "reaching the recorder")

    def test_describing_something_writes_spans(self, client):
        http, traces = client
        self.draft(http, self.RUNNABLE)
        assert rows(traces, "span")

    def test_describing_something_records_a_decision(self, client):
        http, traces = client
        self.draft(http, self.RUNNABLE)
        assert rows(traces, "decision")

    def test_the_questions_asked_are_recorded_by_name(self, client):
        """Phase 1's actual question. A description with open items must leave
        the field names behind, or the pilot measures traffic and not
        understanding."""
        import re

        http, traces = client
        body = self.draft(http, self.UNCLEAR)
        outcomes = {row["outcome"] for row in rows(traces, "decision")}
        assert "QUESTIONS_PRESENTED" in outcomes, outcomes

        # The expectation comes from the page, not from a list written here.
        # `assert refs.strip()` passed against a recorder writing `[]` — an
        # empty JSON array is a non-empty string, and the test was asserting
        # that a field existed rather than that it named anything.
        on_screen = set(re.findall(r'data-field="([a-z_]+)"', body))
        assert on_screen, "this description asked nothing; the case is vacuous"
        recorded = " ".join(str(row["evidence_refs"])
                            for row in rows(traces, "decision"))
        missing = {field for field in on_screen if field not in recorded}
        assert not missing, (
            f"the user was asked about {sorted(missing)} and the trace does "
            f"not name them: {recorded}")

    def test_saving_writes_its_own_trace(self, client):
        """The second half of the journey. `submit_rendered_confirmation`
        posts the controls the page actually rendered rather than a payload
        the test wishes it had — the same reason the class above stopped
        calling `_recorder` directly."""
        from tests.conftest import submit_rendered_confirmation

        http, traces = client
        before = len(rows(traces, "trace"))
        response, plan_id = submit_rendered_confirmation(http, COMPLETE)
        assert response.status_code == 303, response.text
        assert len(rows(traces, "trace")) > before + 1, (
            "POST /workspace/save recorded nothing beyond the GET that "
            "preceded it")

    def test_a_saved_plan_is_named_on_its_trace(self, client):
        """So an operator holding a plan id can find the journey that made it.
        That is the direction support actually searches: a user reports a
        figure, never a request id."""
        from tests.conftest import submit_rendered_confirmation

        http, traces = client
        response, plan_id = submit_rendered_confirmation(http, COMPLETE)
        assert response.status_code == 303, response.text
        produced = " ".join(str(row["produced"]) for row in rows(traces, "trace"))
        assert plan_id and plan_id in produced

    def test_a_save_that_still_has_questions_records_them(self, client):
        """The round-trip count Phase 1 is trying to measure. A submission
        that comes back for more answers must be distinguishable from one that
        saved, or 'how many rounds did this take' has no answer."""
        from tests.conftest import submit_rendered_confirmation

        http, traces = client
        response, _ = submit_rendered_confirmation(
            http, self.UNCLEAR, acknowledge_all=False)
        assert response.status_code == 200, response.status_code
        outcomes = {row["outcome"] for row in rows(traces, "decision")}
        assert outcomes & {"RETURNED_FOR_ANSWERS", "RETURNED_NOT_EXECUTABLE"}, \
            outcomes

    def test_the_trace_holds_no_description(self, client):
        """The guarantee the other layers already keep. Caddy and uvicorn were
        both leaking this into CloudWatch; the store must not become the third
        place the sentence survives."""
        http, traces = client
        secret = "I buy ZZZCANARY every month and never sell."
        self.draft(http, secret)

        # The premise first. An empty store contains no canary either, and a
        # privacy assertion that passes hardest when nothing was recorded is
        # the shape that let the restore drill certify its own write.
        assert rows(traces, "span"), "nothing was recorded; absence proves nothing"

        dump = " ".join(
            str(tuple(sorted(row.items()))) for table in ("trace", "span", "decision")
            for row in rows(traces, table))
        assert "ZZZCANARY" not in dump
        assert "every month" not in dump

    def test_a_broken_store_still_serves_the_builder(self, client, monkeypatch):
        """Telemetry never breaks the thing it observes — restated for the two
        routes that now carry it."""
        import src.telemetry.trace_store as store_module

        http, _ = client
        monkeypatch.setattr(
            store_module.TraceStore, "record_span",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unavailable")))
        assert http.get("/workspace/new",
                        params={"describe": self.RUNNABLE}).status_code == 200

    @staticmethod
    def form(body):
        import html as html_module
        import re

        found = re.search(r'name="parse" value="([^"]*)"', body)
        payload = {"describe": TestThePlanBuilderRecords.RUNNABLE,
                   "title": "Draft",
                   "parse": html_module.unescape(found.group(1)) if found else ""}
        for field, value in re.findall(
                r'name="confirm:([a-z_]+)"[^>]*value="([A-Z]+)"[^>]*checked', body):
            payload[f"confirm:{field}"] = value
        return payload
