"""What an accepted proposal declares it will execute, and whether that is true.

    accepted WorksheetProposal -> WorksheetExecutionDeclaration

Neutral by construction: it inherits from no orchestration type and imports no
contract package. The moment a financial declaration inherits an orchestration
type, the orchestrator's vocabulary starts deciding what Quantify can say.

Its value today is that the execution semantics of `apply.py` become a
*statement* rather than an ordering implied by the sequence of lines in a
function — and a statement can be checked against what actually happens.
"""
from __future__ import annotations

import pytest

from src.workspace.apply import accept
from src.workspace.execution import declare
from src.workspace.intent import plan
from src.workspace.proposal import propose
from src.workspace.store import WorkspaceStore
from src.workspace.worksheet import create, from_json

OWNER = "pilot"
RESULT = {"modelling_scope": {"excludes": ["dividends"]}, "final_value": 1.0}


@pytest.fixture
def worksheet():
    return create(worksheet_id="ws-1", owner_id=OWNER, scenario_ref="plan-1",
                  primary_run_ref="run-0", created_at="t0")


@pytest.fixture
def store(tmp_path, worksheet):
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
                    stated_text="seed", saved_at="t0")
    store.record_run(run_id="run-0", plan_id="plan-1", ran_at="t0",
                     result=RESULT, comparison={})
    store.save_worksheet(worksheet)
    return store


def chain(instructions, worksheet):
    history, proposals = [], []
    for index, text in enumerate(instructions):
        intent = plan(text, intent_id=f"i{index}", source_revision=1,
                      history=list(history), target_run="run-0")
        history.append(intent)
        proposals.append(propose(intent, worksheet))
    return proposals


class TestFanOutIsPreserved:

    def test_a_search_is_three_candidates_not_one(self, worksheet):
        """One search over three instruments is three candidates, never one
        candidate holding three instruments — the distinction a flattened graph
        would lose, and the one trial accounting depends on."""
        proposal = chain(["Try SPY, VTI and VT and keep the best"], worksheet)[0]
        declaration = declare(proposal)

        assert declaration.fan_out == 3
        assert [c.candidate_id for c in declaration.candidates] == [
            "candidate/SPY", "candidate/VTI", "candidate/VT"]

    def test_a_substitution_is_one_candidate(self, worksheet):
        declaration = declare(chain(["Replace SPY with VTI"], worksheet)[0])
        assert declaration.fan_out == 1

    def test_analysis_variants_fan_out_too(self, worksheet):
        proposals = chain(["Add 63-day rolling volatility",
                           "Try 21, 63 and 126 day windows"], worksheet)
        assert declare(proposals[1]).fan_out == 3


class TestTheEvaluatedSetAndTheChoiceAreDifferentFacts:

    def test_a_selection_names_a_candidate_it_came_from(self, worksheet):
        proposals = chain(["Add 63-day rolling volatility",
                           "Try 21, 63 and 126 day windows",
                           "Keep 63 because it looks smoothest"], worksheet)
        declaration = declare(proposals[2])
        assert declaration.selected_candidate == "analysis/volatility/63"
        assert declaration.selected_candidate in [
            c.candidate_id for c in declaration.candidates]

    def test_a_search_with_no_winner_selects_nothing(self, worksheet):
        proposals = chain(["Add 63-day rolling volatility",
                           "Try 21, 63 and 126 day windows"], worksheet)
        assert declare(proposals[1]).selected_candidate is None

    def test_the_trial_effect_survives_into_the_declaration(self, worksheet):
        declaration = declare(
            chain(["Try SPY, VTI and VT and keep the best"], worksheet)[0])
        assert declaration.trial_effect == 3


class TestTheDeclarationMatchesWhatApplyDoes:
    """The half of graph equivalence that can be proven today: a declaration
    that disagrees with the code it describes is worse than none."""

    def test_the_candidate_count_matches_the_runs_written(self, store,
                                                          worksheet):
        proposal = chain(["Try SPY, VTI and VT and keep the best"], worksheet)[0]
        declaration = declare(proposal)
        store.save_worksheet_proposal(proposal_id="p1", owner=OWNER,
                                      worksheet_id="ws-1", proposal=proposal,
                                      created_at="t0")

        result = accept(store, proposal_id="p1", owner=OWNER,
                        worksheet_id="ws-1", proposal=proposal, at="t1",
                        run_candidate=lambda c: RESULT)
        assert len(result.runs) == declaration.fan_out

    def test_the_declared_ordering_matches_the_observed_write_order(
            self, store, worksheet):
        proposal = chain(["Try SPY, VTI and VT and keep the best"], worksheet)[0]
        declaration = declare(proposal)
        store.save_worksheet_proposal(proposal_id="p1", owner=OWNER,
                                      worksheet_id="ws-1", proposal=proposal,
                                      created_at="t0")

        observed = []
        original_run = store.record_run
        original_sheet = store.save_worksheet
        original_resolve = store.resolve_worksheet_proposal
        store.record_run = lambda **kw: (observed.append("candidates"),
                                         original_run(**kw))[1]
        store.save_worksheet = lambda w: (observed.append("worksheet_revision"),
                                          original_sheet(w))[1]
        store.resolve_worksheet_proposal = lambda *a, **kw: (
            observed.append("proposal_resolution"),
            original_resolve(*a, **kw))[1]

        accept(store, proposal_id="p1", owner=OWNER, worksheet_id="ws-1",
               proposal=proposal, at="t1", run_candidate=lambda c: RESULT)

        # Collapse the repeated candidate writes; the declaration states phases,
        # not one entry per candidate.
        phases = [stage for index, stage in enumerate(observed)
                  if index == 0 or stage != observed[index - 1]]
        assert phases == list(declaration.ordering)

    def test_a_declaration_requiring_runs_matches_a_proposal_requiring_a_runner(
            self, worksheet):
        for instruction in ("Replace SPY with VTI",
                            "Try SPY, VTI and VT and keep the best"):
            proposal = chain([instruction], worksheet)[0]
            assert declare(proposal).requires_runs is proposal.rerun_required

    def test_terminal_artifacts_name_the_run_only_when_one_is_produced(
            self, worksheet):
        scenario = declare(chain(["Replace SPY with VTI"], worksheet)[0])
        analysis = declare(chain(["Add 63-day rolling volatility"], worksheet)[0])
        assert "run" in scenario.required_terminal_artifacts
        assert "run" not in analysis.required_terminal_artifacts


class TestItStaysNeutral:

    def test_it_imports_no_orchestration_package(self):
        """The moment a financial declaration inherits an orchestration type,
        the orchestrator's vocabulary decides what Quantify can say."""
        import inspect

        from src.workspace import execution

        source = inspect.getsource(execution)
        for package in ("runtime_contracts", "mission_sdk", "redevops_mission"):
            assert package not in source, package

    def test_an_inapplicable_proposal_declares_nothing(self, worksheet):
        """A plan for a thing that does not happen."""
        proposal = chain(["Move the sparkline widget"], worksheet)[0]
        with pytest.raises(ValueError, match="no execution to declare"):
            declare(proposal)

    def test_it_serializes_whole(self, worksheet):
        payload = declare(
            chain(["Try SPY, VTI and VT and keep the best"], worksheet)[0]).to_json()
        assert payload["fan_out"] == 3
        assert payload["ordering"] == ["candidates", "worksheet_revision",
                                       "proposal_resolution"]
