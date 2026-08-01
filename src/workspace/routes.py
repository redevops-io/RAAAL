"""The private scenario workspace.

A separate router with its own templates, mounted at its own prefix, so the
boundary decision is visible in the file tree rather than only in a document.
The intent is that this can be served from a different hostname without moving
any code.

Nothing here is public. Every page is scoped to one owner at the query, and a
plan may cite public artifacts while nothing public may cite a plan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..mission import (
    ISOLATION_DIMENSIONS,
    CashFlow,
    ComparisonClass,
    ExpectedEvent,
    ObservedEvent,
    PlanObservation,
    Proposal,
    ProposalStatus,
    classify_counterfactual,
    expire_overdue,
    lifecycle_summary,
    reconcile,
    CashPolicy,
    ComparisonClass,
    RunConditions,
    buy_and_hold,
    classify,
    compare,
    comparison_payload,
    compile_scenario,
    hold_cash,
    simulate,
)
from ..mission.parse_model import (
    AnthropicClient,
    parse_from_stored,
    parse_with_model,
)
from ..mission.scenario import UNSIMULATED
from ..mission.templates import RSU_TEMPLATE
from ..mission.templates import TEMPLATES as LIFE_EVENT_TEMPLATES
from .chain import SCENARIO_CHAIN_ORDER, build_scenario_chain
from .confirmation import build as build_confirmation
from .store import NotSaveable, WorkspaceStore

#: Two search paths: the workspace's own templates, and the shared design
#: system. Layout stays separate; tokens are one file.
TEMPLATES = Jinja2Templates(directory=[
    str(Path(__file__).parent / "templates"),
    str(Path(__file__).resolve().parents[1] / "web" / "templates"),
])
router = APIRouter(prefix="/workspace", tags=["workspace"])

PRICES = Path("data/history/prices.parquet")
BENCHMARK_RULE = "benchmark-policy/public-default@1"


def _parser_client():
    """A model for stage 1, when one is configured.

    Absent by default. Without a key the compiler falls back to its
    deterministic rules and asks more questions, which is the correct direction
    to fail in: narrower recognition, never a confident wrong reading.
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    return AnthropicClient(model=os.environ.get("QUANTIFY_PARSER_MODEL",
                                                "claude-sonnet-5"))


def _pinned_parse(record):
    """The parse a saved plan was compiled from, re-verified against its text.

    Revisiting a plan must show what the user confirmed. Re-running stage 1
    against a model that has changed since would quietly recompile it into
    something else, so the stored parse is the input and the model is not
    consulted here at all.

    Plans saved before parses were pinned carry none, and recompile
    deterministically — which is exactly what they did when they were saved.
    """
    stored = record.get("parse")
    if not stored:
        return None
    return parse_from_stored(stored, record["stated_text"])

#: Single-user pilot. Real authentication replaces this before the workspace is
#: exposed to anyone; naming it here keeps the substitution obvious rather than
#: letting an implicit "current user" spread through the handlers.
PILOT_OWNER = "pilot"

#: Life-event templates the compiler may hand off to, keyed by the hint it
#: emits. Named apart from the Jinja environment above, which is a different
#: kind of template entirely.
TEMPLATES_BY_HINT = dict(LIFE_EVENT_TEMPLATES)


def _disclosures(compiled, run) -> Dict[str, Any]:
    """Trial accounting and the recommendation verdict, prepared for the page.

    Above the fold because a conversational interface makes trying variants
    nearly free, and a reader who sees the result first will have formed a
    conclusion before learning how many attempts produced it.
    """
    payload = (run or {}).get("payload") or {}
    assessment = payload.get("recommendation_assessment")
    return {
        "selection_basis": "STATED_DIRECTLY",
        "selection_note": (
            "You described this plan yourself, so no alternatives were "
            "measured and discarded on your behalf."
        ),
        "evaluated": 0,
        "trials": 1,
        "hidden_selection": False,
        "is_recommendation": bool(assessment and assessment["is_recommendation"]),
        "recommendation_headline": (assessment["headline"] if assessment
                                    else "not assessed — nothing was rendered"),
        "derivation_complete": bool(assessment and assessment["derivation_complete"]),
    }


def _store() -> WorkspaceStore:
    return WorkspaceStore()


def _prices() -> Optional[pd.DataFrame]:
    if not PRICES.exists():
        return None
    frame = pd.read_parquet(PRICES)
    return frame.sort_index()


def _flows_from(schedule, sessions: pd.DatetimeIndex) -> List[CashFlow]:
    """Turn a declared schedule into dated contributions.

    The day rule is applied here rather than assumed, because "monthly" does not
    name a day and the day moves the money-weighted return.
    """
    if schedule.amount <= 0:
        return ([CashFlow(sessions[0], schedule.starting_capital, "starting capital")]
                if schedule.starting_capital > 0 else [])

    series = sessions.to_series()
    if schedule.cadence == "monthly":
        groups = series.groupby([sessions.year, sessions.month])
    elif schedule.cadence == "weekly":
        iso = sessions.isocalendar()
        groups = series.groupby([iso.year.values, iso.week.values])
    elif schedule.cadence == "biweekly":
        iso = sessions.isocalendar()
        groups = series.groupby([iso.year.values, (iso.week.values // 2)])
    else:
        return [CashFlow(sessions[0], schedule.amount, "one-off")]

    dates = (groups.max() if schedule.day_rule == "last_session_of_period"
             else groups.min())
    return [CashFlow(d, schedule.amount, "contribution") for d in dates]


def _benchmark_specs(prices: pd.DataFrame, assets) -> List[Dict[str, Any]]:
    """The set, declared before anything runs and generated by a named rule.

    Cash is always present. It is the comparison nobody asks for and the one that
    answers "was any of this worth doing?".
    """
    universe = [a for a in assets if a in prices.columns]
    specs: List[Dict[str, Any]] = []
    if universe:
        specs.append({"name": "Your basket, bought and held",
                      "tickers": universe, "program": buy_and_hold(universe),
                      "description": "the same instruments, with no timing rule"})
    for ticker, label in (("SPY", "S&P 500"), ("QQQ", "Nasdaq 100"),
                          ("AGG", "Aggregate bonds")):
        specs.append({
            "name": f"Contribute to {label}", "tickers": [ticker],
            "program": buy_and_hold([ticker]),
            "description": f"the same contributions into {ticker}",
        })
    specs.append({"name": "Hold cash", "tickers": [], "program": hold_cash(),
                  "description": "contribute and never invest"})
    return specs


def scenario_from_stored(stored: Dict[str, Any], fallback):
    """The scenario as it was saved, rebuilt from its stored canonical body.

    Falls back to the freshly compiled one only for plans saved before the
    stored body carried enough to rebuild — and says so rather than pretending
    the replay was faithful.
    """
    from ..mission.evolution import rebuild_scenario

    rebuilt = rebuild_scenario(stored)
    return rebuilt if rebuilt is not None else fallback


def migration_for(plan_id: str, stored: Dict[str, Any], compiled):
    """What today's compiler would make of the same words, and whether to offer
    it. Explained and never performed: adopting a new interpretation changes
    what a saved plan means, and only its owner can agree to that."""
    from ..mission.evolution import (
        COMPILER_VERSION, diff_stored_against, propose_migration)

    if not stored or compiled.scenario is None:
        return None
    diff = diff_stored_against(
        stored, compiled.scenario,
        stored_compiler=str(stored.get("compiler_version", "1")),
        current_compiler=COMPILER_VERSION,
        current_unresolved=[u.field for u in compiled.unresolved])
    proposal = propose_migration(plan_id, diff)
    return proposal.to_json() if proposal else None


def declare_unsimulated(scenario, scope: Optional[Dict[str, Any]]
                        ) -> Dict[str, Any]:
    """Add every declared-but-unsimulated behaviour to the modelling scope.

    Derived from the scenario rather than hardcoded, so a behaviour that becomes
    simulatable stops being disclosed by deleting one entry in `UNSIMULATED`,
    and one that is added starts being disclosed without anyone remembering to
    edit this function.
    """
    scope = dict(scope or {})
    declared = {
        "dividend_policy": scenario.holdings_policy.dividend_policy,
    }
    not_modelled = {
        field: {"declared": value, "why": UNSIMULATED[field]}
        for field, value in declared.items() if field in UNSIMULATED
    }
    if not_modelled:
        scope["declared_but_not_simulated"] = not_modelled
    return scope


def _run(scenario, prices: pd.DataFrame,
         scope: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Simulate a scenario and its benchmarks under identical conditions."""
    sessions = prices.index
    flows = _flows_from(scenario.flow_schedule, sessions)
    policy = CashPolicy.idle()
    assets = list(scenario.allocation_rule.assets)
    tradeable = [a for a in assets if a in prices.columns]

    if not tradeable or not flows:
        return {"result": None, "benchmarks": [], "payload": None,
                "comparability": None,
                "unavailable": (
                    "No price history for "
                    f"{', '.join(assets) or 'the instruments named'} over this "
                    "period, so the scenario cannot be replayed. This is a data "
                    "gap, not a result."
                )}

    # Every declared behaviour the engine cannot honour must reach the result's
    # modelling scope. Representing `dividend_policy` without saying it is not
    # simulated would move the defect one layer up rather than closing it: the
    # scenario would look enforced and the figure would silently ignore it.
    scope = declare_unsimulated(scenario, scope)

    result = simulate(prices, flows=flows, program=buy_and_hold(tradeable),
                      cash_policy=policy, modelling_scope=scope)
    specs = _benchmark_specs(prices, tradeable)
    benchmarks = compare(prices, flows=flows, cash_policy=policy, benchmarks=specs)

    conditions = RunConditions(
        flow_schedule_hash=scenario.flow_schedule.schedule_hash,
        starting_capital=scenario.flow_schedule.starting_capital,
        cash_policy_rate=policy.annual_rate,
        tax_treatment=scenario.tax_treatment,
        cost_bps=10.0, execution_lag=1,
        period_start=str(sessions[0].date()), period_end=str(sessions[-1].date()),
        allocation_rule_hash=scenario.rule_hash,
        data_snapshot=f"prices@{sessions[-1].date()}",
    )
    return {
        "result": result,
        "benchmarks": benchmarks,
        "comparability": classify(conditions, conditions),
        "payload": comparison_payload(
            result, benchmarks,
            declared_order=[s["name"] for s in specs],
            rendered_text="",
            user_originated_rule=True,
            platform_generated_action=False,
            portfolio_selection_performed=False,
        ),
        "unavailable": None,
    }


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse(
        request, "index.html",
        {"plans": _store().list_plans(PILOT_OWNER), "owner": PILOT_OWNER},
    )


@router.get("/new", response_class=HTMLResponse)
def new_plan(request: Request, describe: str = ""):
    """The confirmation screen. Nothing is saved and nothing is committed."""
    if not describe.strip():
        return TEMPLATES.TemplateResponse(request, "new.html", {"result": None})

    stage1 = parse_with_model(describe, client=_parser_client())
    compiled = compile_scenario(describe, name="draft", version=1,
                                benchmark_rule=BENCHMARK_RULE,
                                parsed=stage1.parsed)
    prices = _prices()
    run = (_run(compiled.scenario, prices)
           if compiled.can_simulate and prices is not None else None)

    return TEMPLATES.TemplateResponse(
        request, "new.html",
        {
            "describe": describe,
            "result": compiled,
            "confirmation": compiled.confirmation(),
            "view": build_confirmation(compiled, text=describe),
            "parse": json.dumps(stage1.parsed.to_json()),
            "parse_provenance": stage1.provenance,
            "chain": build_scenario_chain(
                subject="draft", scenario=compiled.scenario,
                result=run["result"] if run else None,
                benchmarks=run["benchmarks"] if run else (),
                comparability=run["comparability"] if run else None,
            ),
            "run": run,
            "chain_order": SCENARIO_CHAIN_ORDER,
        },
    )


@router.post("/save")
def save_plan(describe: str, plan_id: str, confirm_all: str = "",
              parse: str = Form(default="")):
    """Commit. Refuses anything the user has not actually confirmed.

    The parse comes back from the confirmation screen so that what is saved is
    the interpretation the user actually read. It arrives via a browser and is
    therefore not trusted: `parse_from_stored` re-checks every recognition
    against the description, so a tampered field cannot inject a reading the
    text does not support.
    """
    parsed = None
    if parse.strip():
        try:
            parsed = parse_from_stored(json.loads(parse), describe)
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"the submitted interpretation does not match the "
                       f"description: {exc}") from exc

    compiled = compile_scenario(describe, name=plan_id, version=1,
                                benchmark_rule=BENCHMARK_RULE, parsed=parsed)
    scenario = compiled.scenario

    if confirm_all == "yes":
        # Confirming is an act the user performs on each inference. Doing it in
        # bulk is allowed, and recording that it was bulk keeps the difference
        # from a considered confirmation visible later.
        from ..mission.spec import Inference, Provenance
        from ..mission.scenario import ScenarioSpecification

        p = scenario.provenance
        scenario = ScenarioSpecification(**{
            **scenario.__dict__,
            "provenance": Provenance(
                stated=p.stated,
                inferred=tuple(Inference(i.field, i.value, i.why, confirmed=True)
                               for i in p.inferred),
                contradictions=p.contradictions,
                unresolved=p.unresolved,
            ),
        })

    try:
        _store().save_plan(
            plan_id=plan_id, owner=PILOT_OWNER, scenario=scenario,
            stated_text=describe, saved_at=pd.Timestamp.now("UTC").isoformat(),
            parse=parsed.to_json() if parsed is not None else None,
        )
    except NotSaveable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RedirectResponse(f"/workspace/plans/{plan_id}", status_code=303)


@router.get("/plans/{plan_id}", response_class=HTMLResponse)
def plan_detail(request: Request, plan_id: str):
    store = _store()
    record = store.get_plan(plan_id, PILOT_OWNER)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no plan {plan_id!r}")

    # Opening a saved plan is a *replay*, not a reinterpretation. This route
    # used to recompile the stored text and simulate the result, while showing
    # the stored scenario beside it — so after any compiler change the page
    # displayed one plan and the figures came from another.
    #
    # The stored scenario is what the user read and confirmed. It is what runs.
    compiled = compile_scenario(record["stated_text"], name=plan_id, version=1,
                                benchmark_rule=BENCHMARK_RULE,
                                parsed=_pinned_parse(record))
    stored = record["scenario"]
    migration = migration_for(plan_id, stored, compiled)

    prices = _prices()
    scope = (TEMPLATES_BY_HINT[compiled.template_hint].modelling_scope()
             if compiled.template_hint in TEMPLATES_BY_HINT else None)
    run = (_run(scenario_from_stored(stored, compiled.scenario), prices, scope)
           if prices is not None else None)

    return TEMPLATES.TemplateResponse(
        request, "plan.html",
        {
            "record": record,
            "scenario": record["scenario"],
            "migration": migration,
            "run": run,
            "runs": store.runs_for(plan_id, PILOT_OWNER),
            "proposals": [p["payload"] for p in
                          store.list_proposals(plan_id, PILOT_OWNER)],
            "observations": [o["payload"] for o in
                             store.list_observations(plan_id, PILOT_OWNER)],
            "scope": (run.get("result").to_json()["modelling_scope"]
                      if run and run.get("result") else scope),
            "disclosures": _disclosures(compiled, run),
            "chain": build_scenario_chain(
                subject=plan_id, scenario=compiled.scenario,
                result=run["result"] if run else None,
                benchmarks=run["benchmarks"] if run else (),
                comparability=run["comparability"] if run else None,
                saved=True,
            ),
        },
    )


@router.get("/plans/{plan_id}/proposals", response_class=HTMLResponse)
def proposals(request: Request, plan_id: str, as_of: str = ""):
    """Every proposal this plan produced, whatever became of it.

    Expired and ignored proposals stay as visible as accepted ones. Neither is a
    failed record — an expiry is the only evidence that a constraint cost
    something, and hiding it would make the platform look more effective than it
    was.
    """
    store = _store()
    if store.get_plan(plan_id, PILOT_OWNER) is None:
        raise HTTPException(status_code=404, detail=f"no plan {plan_id!r}")

    stored = store.list_proposals(plan_id, PILOT_OWNER)
    return TEMPLATES.TemplateResponse(
        request, "proposals.html",
        {"plan_id": plan_id, "proposals": [p["payload"] for p in stored],
         "as_of": as_of},
    )


@router.get("/plans/{plan_id}/observations", response_class=HTMLResponse)
def observations(request: Request, plan_id: str):
    """Planned, observed and reconciled — three lanes, never merged.

    A delayed vest reads as a missing expectation *and* an unexpected arrival,
    because that is what it is. One shifted row would hide the fact that the
    plan's assumption failed.
    """
    store = _store()
    if store.get_plan(plan_id, PILOT_OWNER) is None:
        raise HTTPException(status_code=404, detail=f"no plan {plan_id!r}")

    stored = store.list_observations(plan_id, PILOT_OWNER)
    return TEMPLATES.TemplateResponse(
        request, "observations.html",
        {"plan_id": plan_id, "observations": [o["payload"] for o in stored]},
    )


@router.get("/plans/{plan_id}/counterfactual", response_class=HTMLResponse)
def counterfactual(request: Request, plan_id: str, constraint: str = "a blackout window"):
    """What a constraint cost, with the isolation stated before any figure.

    The view leads with what is isolated rather than with the difference,
    because a number shown first will be read as a verdict on the strategy.
    """
    store = _store()
    record = store.get_plan(plan_id, PILOT_OWNER)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no plan {plan_id!r}")

    prices = _prices()
    compiled = compile_scenario(record["stated_text"], name=plan_id, version=1,
                                benchmark_rule=BENCHMARK_RULE,
                                parsed=_pinned_parse(record))
    scenario = compiled.scenario
    if prices is None or prices.empty:
        return TEMPLATES.TemplateResponse(
            request, "counterfactual.html",
            {"plan_id": plan_id, "verdict": None, "constraint": constraint,
             "unavailable": "No price history is available to replay against."},
        )

    sessions = prices.index
    common = dict(
        flow_schedule_hash=scenario.flow_schedule.schedule_hash,
        starting_capital=scenario.flow_schedule.starting_capital,
        cash_policy_rate=0.0, tax_treatment=scenario.tax_treatment,
        cost_bps=10.0,
        period_start=str(sessions[0].date()), period_end=str(sessions[-1].date()),
        allocation_rule_hash=scenario.rule_hash,
        data_snapshot=f"prices@{sessions[-1].date()}",
    )
    verdict = classify_counterfactual(
        RunConditions(**common, execution_lag=1),
        RunConditions(**common, execution_lag=0),
        constraint=constraint,
    )
    return TEMPLATES.TemplateResponse(
        request, "counterfactual.html",
        {"plan_id": plan_id, "verdict": verdict, "constraint": constraint,
         "held_identical": [d for d in ISOLATION_DIMENSIONS
                            if d not in verdict.differing_dimensions],
         "unavailable": None},
    )
