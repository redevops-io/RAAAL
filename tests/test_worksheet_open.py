"""Opening a saved worksheet resolves references. It never reinterprets.

    GET  /workspace/research/{id}            read, side-effect free
    POST /workspace/research/{id}/reinterpret compile the original words again
    POST /workspace/research/{id}/rerun       simulate again

Three different operations, and only the first is what reopening owes. The plan
page already conflated them once — it recompiled its stored prose and simulated
the fresh interpretation while displaying the stored scenario — and it reached
compilation *indirectly*, which is why this test patches every route into it
rather than the one obvious function.

The route should have nowhere to pass original text.
"""
from __future__ import annotations

import re

import pytest

from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import Block, create, revise

OWNER = "pilot"
DESCRIPTION = ("I put $2,000 into SPY every month in my Roth IRA, on the first "
               "trading day of the period, reinvesting the dividends, and I "
               "never sell.")


def text(html: str) -> str:
    body = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import src.api as api
    import src.web.routes as web_routes
    import src.workspace.routes as workspace_routes
    from src.ledger import Ledger

    ledger = Ledger(tmp_path / "public.db")
    monkeypatch.setattr(api, "_ledger", ledger)
    monkeypatch.setattr(web_routes, "Ledger", lambda *a, **k: ledger)
    store = WorkspaceStore(tmp_path / "workspace.db")
    monkeypatch.setattr(workspace_routes, "_store", lambda: store)
    api._bootstrap()
    return TestClient(api.app), store


@pytest.fixture
def saved(client):
    """A worksheet over a stored scenario and a stored run."""
    from src.mission.compiler import compile_scenario
    from src.mission.spec import Inference, Provenance
    from src.mission.scenario import ScenarioSpecification

    _client, store = client
    compiled = compile_scenario(DESCRIPTION, name="plan-1", version=1,
                                benchmark_rule="benchmark-policy/public-default@1")
    provenance = compiled.scenario.provenance
    scenario = ScenarioSpecification(**{
        **compiled.scenario.__dict__,
        "provenance": Provenance(
            stated=provenance.stated,
            inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                           for i in provenance.inferred),
            contradictions=provenance.contradictions, unresolved=()),
    })
    store.save_plan(plan_id="plan-1", owner=OWNER, scenario=scenario,
                    stated_text=DESCRIPTION, saved_at="2026-08-01T00:00:00Z")
    store.record_run(run_id="run-1", plan_id="plan-1",
                     ran_at="2026-08-01T00:00:00Z",
                     result={"modelling_scope": {"excludes": ["dividends"]},
                             "money_weighted": 0.11,
                             "time_weighted_annualized": 0.09,
                             "final_value": 130000.0},
                     comparison={"members": ["SPY", "60/40"],
                                 "comparability": "COMPARABLE"})
    worksheet = create(worksheet_id="ws-1", owner_id=OWNER,
                       scenario_ref="plan-1", primary_run_ref="run-1",
                       title="Monthly SPY", created_at="2026-08-01T00:00:00Z")
    store.save_worksheet(worksheet)
    return worksheet


@pytest.fixture
def no_compiling(monkeypatch):
    """Every route into compilation or simulation raises.

    Patched broadly on purpose. The plan-page defect survived review because the
    route reached compilation through a helper rather than by calling the
    compiler directly.
    """
    import src.mission.compiler as compiler
    import src.mission.parse_model as parse_model
    import src.mission.simulate as simulate
    import src.workspace.routes as routes

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "opening a saved worksheet reached compilation or simulation")

    for module, name in (
        (compiler, "compile_scenario"), (compiler, "parse"),
        (parse_model, "parse_with_model"), (parse_model, "parse_from_stored"),
        (simulate, "simulate"),
        (routes, "compile_scenario"), (routes, "simulate"),
        (routes, "parse_with_model"), (routes, "parse_from_stored"),
        (routes, "_run"), (routes, "_pinned_parse"),
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, forbidden)


class TestOpeningDoesNotReinterpret:

    def test_it_succeeds_without_compiling_or_simulating(self, client, saved,
                                                         no_compiling):
        api_client, _store = client
        response = api_client.get("/workspace/research/ws-1")
        assert response.status_code == 200

    def test_it_shows_the_stored_scenario(self, client, saved, no_compiling):
        api_client, _store = client
        page = text(api_client.get("/workspace/research/ws-1").text)
        assert "Roth IRA" in page or "ROTH" in page
        assert "SPY" in page

    def test_the_route_has_nowhere_to_pass_original_text(self):
        """Structural. A function with no parameter for prose cannot compile
        prose, whatever a later edit does inside it."""
        import inspect

        from src.workspace import routes

        signature = inspect.signature(routes.open_worksheet)
        assert "describe" not in signature.parameters
        assert "stated_text" not in signature.parameters

    def test_opening_creates_no_new_revision(self, client, saved, no_compiling):
        api_client, store = client
        before = store.worksheet_revisions("ws-1", OWNER)
        api_client.get("/workspace/research/ws-1")
        assert store.worksheet_revisions("ws-1", OWNER) == before

    def test_opening_records_no_run(self, client, saved, no_compiling):
        api_client, store = client
        before = store.runs_for("plan-1", OWNER)
        api_client.get("/workspace/research/ws-1")
        assert store.runs_for("plan-1", OWNER) == before

    def test_opening_twice_returns_the_same_page(self, client, saved,
                                                 no_compiling):
        api_client, _store = client
        first = text(api_client.get("/workspace/research/ws-1").text)
        second = text(api_client.get("/workspace/research/ws-1").text)
        assert first == second

    def test_a_changed_compiler_does_not_change_what_is_displayed(
            self, client, saved, no_compiling, monkeypatch):
        """The stored scenario is what the user confirmed. A compiler that has
        since learned to represent more fields does not get to redraw it."""
        import src.mission.evolution as evolution

        api_client, _store = client
        before = text(api_client.get("/workspace/research/ws-1").text)
        monkeypatch.setattr(evolution, "COMPILER_VERSION", "99")
        assert text(api_client.get("/workspace/research/ws-1").text) == before


class TestReferencesArePinned:

    def test_the_worksheet_opens_against_the_version_it_cited(self, client,
                                                              saved, no_compiling):
        """Not the newest. A route that helpfully resolves a concept id to the
        latest artifact silently changes what a saved worksheet means."""
        api_client, store = client
        store.record_run(run_id="run-2", plan_id="plan-1",
                         ran_at="2027-01-01T00:00:00Z",
                         result={"modelling_scope": {"excludes": []},
                                 "final_value": 999999.0},
                         comparison={})
        page = text(api_client.get("/workspace/research/ws-1").text)
        assert "999,999" not in page and "999999" not in page

    def test_a_missing_run_is_a_named_unmet_block(self, client, no_compiling):
        """An omitted panel is invisible; a panel that says why is a fact."""
        api_client, store = client
        store.save_worksheet(create(worksheet_id="ws-bare", owner_id=OWNER,
                                    scenario_ref="plan-1",
                                    created_at="2026-08-01T00:00:00Z"))
        page = text(api_client.get("/workspace/research/ws-bare").text)
        assert "primary_run_ref" in page or "no run" in page.lower()

    def test_an_unresolvable_reference_is_reported_not_dropped(self, client,
                                                               no_compiling):
        api_client, store = client
        store.save_worksheet(create(worksheet_id="ws-broken", owner_id=OWNER,
                                    scenario_ref="plan-that-is-gone",
                                    primary_run_ref="run-1",
                                    created_at="2026-08-01T00:00:00Z"))
        response = api_client.get("/workspace/research/ws-broken")
        assert response.status_code == 200
        assert "could not be resolved" in text(response.text).lower()


class TestRevisions:

    def test_an_old_revision_stays_readable(self, client, saved, no_compiling):
        api_client, store = client
        store.save_worksheet(revise(saved, reason="dropped a benchmark",
                                    benchmark_run_refs=(),
                                    created_at="2026-09-01T00:00:00Z"))
        assert api_client.get("/workspace/research/ws-1?revision=1").status_code == 200
        assert api_client.get("/workspace/research/ws-1").status_code == 200

    def test_a_title_change_does_not_alter_financial_identity(self, saved):
        renamed = revise(saved, reason="renamed it", title="Something else")
        assert renamed.canonical_hash == saved.canonical_hash

    def test_a_changed_reference_needs_a_new_revision(self, saved):
        moved = revise(saved, reason="pointed at a later run",
                       primary_run_ref="run-2")
        assert moved.revision == 2
        assert moved.canonical_hash != saved.canonical_hash


class TestReadAndReinterpretAreDifferentEndpoints:

    def test_get_is_read_only(self):
        """`GET` must not compile, simulate, revise or refresh."""
        import inspect

        from src.workspace import routes

        source = inspect.getsource(routes.open_worksheet)
        for verb in ("compile_scenario", "simulate", "save_worksheet",
                     "record_run", "revise"):
            assert verb not in source, f"open_worksheet calls {verb}"

    def test_reinterpretation_is_its_own_endpoint(self):
        from src.workspace import routes

        assert hasattr(routes, "reinterpret_worksheet")

    def test_a_rerun_is_its_own_endpoint(self):
        from src.workspace import routes

        assert hasattr(routes, "rerun_worksheet")


class TestRenderingIsPresentationalOnly:
    """No new financial semantics, no hidden calculations, no ranking.

    A template that decides which result is better has made an argument the
    engine did not, and it is the argument nobody reviewed.
    """

    BLOCKS = "src/workspace/templates/_worksheet_blocks.html"
    PAGE = "src/workspace/templates/worksheet.html"

    def test_no_template_performs_a_financial_calculation(self):
        body = open(self.BLOCKS).read()
        for pattern in (r"\{\{[^}]*\*\s*100[^}]*\}\}(?!.*macro)",
                        r"\{\{[^}]*\bsum\b", r"\{\{[^}]*\|\s*sort"):
            offending = [m for m in re.findall(pattern, body)
                         if "macro" not in m]
            assert not offending, f"{pattern}: {offending}"

    def test_no_result_is_sorted_or_ranked(self):
        """Sorting by outcome is ranking with extra steps."""
        body = open(self.BLOCKS).read()
        assert "sort(" not in body and "|sort" not in body

    @pytest.mark.parametrize("word", [
        "beats", "wins", "winner", "optimal", "superior", "recommended",
        "best", "outperform",
    ])
    def test_no_recommendation_language(self, word):
        for path in (self.BLOCKS, self.PAGE):
            assert word not in open(path).read().lower(), f"{word} in {path}"

    def test_comparability_renders_before_performance(self):
        """A reader who sees two numbers side by side has already compared
        them, whatever a caption says afterwards."""
        body = open(self.BLOCKS).read()
        assert body.index("BenchmarkComparisonBlock") < \
            body.index("PerformanceSummaryBlock")

    def test_both_return_bases_appear_together(self):
        body = open(self.BLOCKS).read()
        assert "time_weighted_annualized" in body and "money_weighted" in body

    def test_state_is_never_carried_by_colour_alone(self):
        """Every state carries a word as well as a class.

        The state is printed verbatim next to the class, so a reader who cannot
        distinguish the colours reads "not evaluated" rather than inferring it.
        """
        body = open(self.BLOCKS).read()
        assert 'class="small state-' in body
        assert "NOT_EVALUATED" in body
        assert ".replace('_', ' ')|lower" in body

    def test_an_absent_verdict_cannot_render_as_a_negative_one(self):
        """`false` is checked and different. `null` is not checked.

        A page showing the second as the first looks cautious while being
        wrong, and a reader takes it for an actual verdict.
        """
        from src.workspace.comparability_record import DISPLAYED, from_payload

        restored = from_payload({"benchmarks": [{"name": "stored before "
                                                          "verdicts existed"}]})
        assert restored, "a legacy benchmark must still render"
        assert set(restored[0]["dimensions"].values()) == {"NOT_EVALUATED"}
        assert set(restored[0]["dimensions"]) == set(DISPLAYED)

    def test_equality_by_absence_is_not_a_match(self):
        """Two empty account hashes compare equal in the engine. Rendering that
        as "matched" would say the account treatment was compared when neither
        run recorded one."""
        from src.mission.comparability import RunConditions
        from src.workspace.comparability_record import record

        common = dict(flow_schedule_hash="f1", starting_capital=0.0,
                      cash_policy_rate=0.0, tax_treatment="ROTH", cost_bps=10.0,
                      execution_lag=1, period_start="2016-01-04",
                      period_end="2025-11-19", data_snapshot="prices@x")
        verdict = record(RunConditions(**common, allocation_rule_hash="a"),
                         {"passive": RunConditions(**common,
                                                   allocation_rule_hash="b")})[0]
        assert verdict.dimensions["account"] == "NOT_EVALUATED"
        assert verdict.dimensions["flow_schedule"] == "MATCHED"

    def test_a_data_gap_renders_as_a_named_gap(self, client, no_compiling):
        api_client, store = client
        store.save_worksheet(create(worksheet_id="ws-gap", owner_id=OWNER,
                                    scenario_ref="plan-missing",
                                    created_at="2026-08-01T00:00:00Z"))
        page = text(api_client.get("/workspace/research/ws-gap").text)
        assert "could not be resolved" in page.lower()

    def test_an_old_revision_renders(self, client, saved, no_compiling):
        api_client, store = client
        store.save_worksheet(revise(saved, reason="second",
                                    created_at="2026-09-01T00:00:00Z"))
        first = api_client.get("/workspace/research/ws-1?revision=1")
        assert first.status_code == 200
        assert "Revision 1" in text(first.text)
