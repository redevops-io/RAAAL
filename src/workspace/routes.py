"""The private scenario workspace.

A separate router with its own templates, mounted at its own prefix, so the
boundary decision is visible in the file tree rather than only in a document.
The intent is that this can be served from a different hostname without moving
any code.

Nothing here is public. Every page is scoped to one owner at the query, and a
plan may cite public artifacts while nothing public may cite a plan.
"""
from __future__ import annotations

from dataclasses import replace as dataclasses_replace
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
    DEFAULT_MODEL,
    parse_from_stored,
    parse_with_model,
)
from ..mission.scenario import UNSIMULATED
from ..mission.spec import ScenarioAmendment
from ..mission.templates import RSU_TEMPLATE
from ..mission.templates import TEMPLATES as LIFE_EVENT_TEMPLATES
from . import historical_lots
from .chain import SCENARIO_CHAIN_ORDER, build_scenario_chain
from .confirmation import build as build_confirmation
from .comparability_record import as_payload as comparability_payload
from .environment import pins_for
from .comparability_record import record as comparability_records
from .generate import generate as generate_worksheet
from .generate import new_plan_id
from .store import NotSaveable, WorkspaceStore

#: Two search paths: the workspace's own templates, and the shared design
#: system. Layout stays separate; tokens are one file.
TEMPLATES = Jinja2Templates(directory=[
    str(Path(__file__).parent / "templates"),
    str(Path(__file__).resolve().parents[1] / "web" / "templates"),
])
router = APIRouter(prefix="/workspace", tags=["workspace"])

#: Available to every template, rather than passed by each of eleven render
#: calls. A disclosure that each surface must remember to include is a
#: disclosure that will be missing from the twelfth — and the surface it is
#: missing from will be the one someone reads a figure on.
TEMPLATES.env.globals["data_notice"] = lambda: _data_notice()

PRICES = Path("data/history/prices.parquet")
BENCHMARK_RULE = "benchmark-policy/public-default@1"


#: The pilot data boundary, as disclosure. `market_data.pilot_policy` is what
#: enforces it; this is what says so where a figure is read.
#:
#: A boundary enforced and not stated leaves a user reading a realistic-looking
#: series and reasonably taking it for historical analysis. The synthetic
#: fixture is deliberately shaped like market data so the evaluation stack has
#: something realistic to run on, which is exactly why the disclosure is needed.
def _data_notice():
    from ..deploy.context import current
    from ..market_data.pilot_policy import PilotDataPolicy

    policy = current().market_data.policy
    if policy is not PilotDataPolicy.SYNTHETIC_ONLY:
        return None
    return {
        "headline": "Pilot mode uses synthetic market data.",
        "detail": ("Results are for product evaluation only and are not based "
                   "on licensed live market data. The series are invented, "
                   "shaped like market data so the engine has something "
                   "realistic to run on, and calibrated to no real security."),
    }


def _suggested_title(describe: str) -> str:
    """A first draft of the plan's name, from the user's own words.

    A suggestion only — the field is editable and identity never derives from
    it. Truncated on a word boundary so the box opens with something short
    enough to read rather than the whole description.
    """
    words = describe.strip().split()
    if not words:
        return "Untitled plan"
    draft = " ".join(words[:8])
    return draft[:60].rstrip(" ,.") + ("…" if len(words) > 8 else "")


def _parser_client():
    """A model for stage 1, when one is configured.

    Absent by default. Without a key the compiler falls back to its
    deterministic rules and asks more questions, which is the correct direction
    to fail in: narrower recognition, never a confident wrong reading.

    Asks the deployment rather than the environment. This function used to read
    `ANTHROPIC_API_KEY` and `QUANTIFY_PARSER_MODEL` itself — a request handler
    deciding for itself what the deployment was, and the exact shape that let
    the preflight validate PostgreSQL while the store opened SQLite.
    """
    from ..deploy.context import ParserFallback, current

    model = current().model
    if not model.model_assisted:
        # Declared deterministic. Not "no key found" — a deployment that says
        # what it is rather than one that discovers it.
        return None
    if not model.available:
        if model.fallback is ParserFallback.REFUSE:
            raise HTTPException(
                status_code=503,
                detail="This deployment is configured for model-assisted "
                       "interpretation and cannot currently reach its model. "
                       "Please try again shortly.")
        return None
    return AnthropicClient(model=model.model or DEFAULT_MODEL,
                           api_key=model.api_key())


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
    """The store, opened on the database this deployment resolved.

    Stated at the construction site rather than left to the engine's default.
    `WorkspaceStore()` reaching a correct default is how the previous defect
    hid: the substitution happened one layer down, where no reader of this
    function could see which database a request was about to write to.
    """
    from ..deploy.context import current

    return WorkspaceStore(current().database.url)


def _recorder(*, worksheet_id: str = "", conversation_id: str = ""):
    """A recorder wired to the deployment's trace store.

    Until Gate 6 nothing in `src/` ever constructed a `TraceStore`. The only
    production entry point took `recorder=None` and `plan_and_record`
    substituted `Recorder(store=None)`, so every span, trace and decision the
    runtime carefully assembled was dropped. Twenty-five telemetry tests passed
    because each built a recorder *with* a store in its own fixture — the
    mechanism proven, and nothing reaching it.

    That also made the independence claim vacuous rather than true. "Deleting
    every trace changes nothing" holds trivially when there are no traces.

    Failures here cost a trace and never an edit: `TelemetryTarget.store`
    returns `None` if the store cannot be opened, and `Recorder._guard` counts
    a failed write instead of raising.
    """
    from ..deploy.context import current
    from ..telemetry import Recorder

    return Recorder(store=current().telemetry.store(),
                    conversation_id=conversation_id or None,
                    worksheet_id=worksheet_id or None,
                    tenant=PILOT_OWNER)


def _prices() -> Optional[pd.DataFrame]:
    """Prices for a pilot request, through the pilot data gate.

    This used to read `data/history/prices.parquet` directly — an unmanifested
    file with no snapshot identity, no licence class and no egress check. A
    pilot user describing a scenario received figures from it by an ordinary
    code path, which is precisely the route a licence breach takes.

    The resolution now lives in `market_data.access` and is shared with the
    public router, which held the same bypass until Gate 3. Two copies of a gate
    are not twice as safe; they are a gate that gets updated in one place.
    """
    from ..market_data.access import resolve

    return resolve(context="pilot scenario run").frame


def _market_data(context: str, *, plan_id: str = "", scenario=None,
                 ran_at: str = ""):
    """The frame, its provenance and the record of this delivery, together.

    The run identity is computed *before* the data is resolved, wherever the
    caller knows what it will store. That is what lets the delivery name the
    execution it was made for, without a second write to connect them later —
    and a chain with a second write has a half-connected state that a crash can
    find. `run_id_for` is deterministic over exactly these three inputs, so
    this is the same identity `generate` will derive, not a guess at it.
    """
    from ..market_data.access import resolve
    from .generate import run_id_for

    run_id = (run_id_for(plan_id, scenario.content_hash, ran_at)
              if plan_id and scenario is not None and ran_at else None)
    return resolve(context=context, run_id=run_id,
                   request_id=f"{context}:{plan_id or 'anonymous'}")


def _approved_snapshot():
    """The snapshot an approved policy names. Absent until one is recorded."""
    from ..market_data.loader import production_snapshot

    try:
        return production_snapshot()
    except Exception:                                          # noqa: BLE001
        return None


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


def _now() -> str:
    """One timestamp format for everything this module writes."""
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _candidate_runner(access, store, worksheet_id: str):
    """Simulate one candidate from an accepted proposal.

    Injected into `apply.accept` rather than imported by it, so the apply path
    can be tested against a failing run without a price file and so a caller
    cannot get a scenario change applied without supplying one.

    Each candidate is simulated as the stored scenario with its instruments
    replaced. Rebuilding from the *stored* scenario rather than recompiling the
    original text matters: the stored one is what the user read and confirmed,
    and recompiling would let a compiler change alter what a candidate means.
    """
    from dataclasses import replace

    from .worksheet import from_json as worksheet_from_json

    record = store.get_worksheet(worksheet_id, PILOT_OWNER)
    worksheet = worksheet_from_json(record["payload"])
    plan = store.get_plan(worksheet.scenario_ref, PILOT_OWNER)
    if plan is None:
        raise HTTPException(
            status_code=409,
            detail=(f"worksheet {worksheet_id} cites scenario "
                    f"{worksheet.scenario_ref}, which is not in this workspace"))

    compiled = compile_scenario(plan["stated_text"], name=worksheet.scenario_ref,
                                version=1, benchmark_rule=BENCHMARK_RULE,
                                parsed=_pinned_parse(plan))
    base = scenario_from_stored(plan["scenario"], compiled.scenario)

    def run_one(candidate) -> Dict[str, Any]:
        assets = tuple(candidate) if isinstance(candidate, (list, tuple)) \
            else (candidate,)
        scenario = replace(
            base, allocation_rule=replace(base.allocation_rule, assets=assets))
        outcome = _run(scenario, access)
        if outcome.get("result") is None:
            # A candidate with no price history is a data gap, not a result.
            # Returning an empty payload would let the revision cite a run that
            # simulated nothing.
            raise HTTPException(
                status_code=422,
                detail=outcome.get("unavailable")
                or f"candidate {assets} could not be simulated")

        # Serialized here rather than handed back as a `MissionResult`. The
        # annotation said `Dict` and the object was not one, so `dict(result)`
        # in the apply path would have failed on the first real candidate.
        #
        # Every candidate is an independent artifact: a worksheet citing three
        # of them must be able to say which data each used, even while they
        # happen to share one access. `to_json` carries the provenance the
        # result was built with, so this is a serialization rather than a
        # caller remembering to attach something.
        return outcome["result"].to_json()

    return run_one


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


def _run(scenario, access, scope: Optional[Dict[str, Any]] = None
         ) -> Dict[str, Any]:
    """Simulate a scenario and its benchmarks under identical conditions.

    Takes the `MarketDataAccess` rather than a bare frame, so the record of
    which data produced the figures travels with the data that produced them.
    A caller attaching provenance to the result afterwards is a caller that can
    forget, and the run it forgot on looks exactly like one it did not.
    """
    from ..market_data.access import MarketDataAccess

    if not isinstance(access, MarketDataAccess):
        raise TypeError(
            "_run needs the MarketDataAccess the frame came from, not the "
            "frame alone; the provenance is not reconstructable afterwards")
    prices = access.frame
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

    # Pin what this run actually used, before it runs. Left empty, the
    # classifier compares two absences — and before classifier@2 reported them
    # equal, so a stored verdict claimed the account treatment was checked when
    # nothing was. An unpinnable runtime becomes a declared limitation on the
    # result rather than a blank, which is why this must happen before the scope
    # is handed to `simulate` rather than after.
    snapshot = f"prices@{sessions[-1].date()}"
    pins = pins_for(scenario, snapshot=snapshot)
    if pins.unpinned:
        scope = {**scope, "unpinned_runtimes": pins.limitations()}

    # What the engine was actually given, digested at the moment it is given.
    #
    # The delivery digest proves the resolver returned these rows. It cannot
    # prove that nothing between here and `simulate` dropped, reordered,
    # mutated or substituted them — so this closes the remaining span by
    # digesting the frame at the call itself:
    #
    #     resolved_frame_digest -> execution_input_digest -> result
    #
    # Equal digests mean the transformation was the identity. They are equal
    # today because `simulate` receives the delivered frame unchanged and the
    # instrument selection travels in `program` rather than in the data. If a
    # future path slices or adjusts the frame, this records that it did, and
    # the difference must then name a declared, versioned transformation rather
    # than pass unnoticed.
    from ..market_data.access_event import frame_digest

    execution_input = frame_digest(prices)
    result = simulate(prices, flows=flows, program=buy_and_hold(tradeable),
                      cash_policy=policy, modelling_scope=scope)
    if access.access_event is not None:
        scope = {**scope, "execution_input_digest": execution_input,
                 "execution_input_matches_delivery":
                     execution_input == access.access_event.frame_digest}
        result = dataclasses_replace(result, modelling_scope=scope)
    # Attached here rather than by each caller. `_run` is the only place that
    # knows which access produced these figures, and a caller that has to
    # remember to attach it is a caller that can forget — which is how the live
    # path stored unattributable runs while the provenance sat one frame away.
    result = dataclasses_replace(result, market_data=access.provenance)
    specs = _benchmark_specs(prices, tradeable)
    benchmarks = compare(prices, flows=flows, cash_policy=policy, benchmarks=specs)

    conditions = RunConditions(
        **pins.as_conditions(),
        flow_schedule_hash=scenario.flow_schedule.schedule_hash,
        starting_capital=scenario.flow_schedule.starting_capital,
        cash_policy_rate=policy.annual_rate,
        tax_treatment=scenario.tax_treatment,
        cost_bps=10.0, execution_lag=1,
        period_start=str(sessions[0].date()), period_end=str(sessions[-1].date()),
        allocation_rule_hash=scenario.rule_hash,
        data_snapshot=snapshot,
    )
    # One verdict per benchmark, computed here and stored with the run. The
    # worksheet reads them; it never recomputes. Every benchmark shares the
    # strategy's flows, period and account treatment by construction — `compare`
    # runs them under identical conditions — so only the allocation rule differs.
    benchmark_conditions = {
        spec["name"]: RunConditions(
            **{k: v for k, v in conditions.__dict__.items()
               if k != "allocation_rule_hash"},
            allocation_rule_hash=f"benchmark:{spec['name']}")
        for spec in specs
    }
    verdicts = comparability_records(conditions, benchmark_conditions)

    return {
        "result": result,
        "benchmarks": benchmarks,
        "comparability": classify(conditions, conditions),
        "comparability_records": comparability_payload(verdicts),
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


#: Runtime references a template declaration needs and the text cannot supply.
#: Read from configuration rather than inferred, and absent ones stay absent so
#: the card asks rather than defaulting.
def _template_runtime_refs() -> Dict[str, str]:
    return {"tax_runtime_ref": "tax/us-federal@1",
            "account_runtime_ref": "account/taxable@1",
            "market_data_ref": f"prices@{_now()[:10]}"}


def _declaration_versions():
    from ..mission.rsu_declaration import DeclarationVersions
    from ..runtime.rsu import US_SHARE_WITHHOLDING

    return DeclarationVersions(
        template_version="template/rsu-vesting@1",
        rsu_runtime_version=US_SHARE_WITHHOLDING.name + "@1",
        account_runtime_version="account/taxable@1",
        tax_runtime_version="tax/us-federal@1",
        corporate_action_runtime_version="",
        scope_schema_version="rsu-result-context@1")


#: Template surfaces that recognise a description but cannot yet compile one
#: into anything executable, with what is missing.
#:
#: `RSUDeclaration` is consumed by the route that builds it, the card that
#: renders it, and their tests. Nothing turns a declaration into a scenario, a
#: run or a worksheet — vest events are not cash flows the compiler
#: understands, and there is no `compile_rsu_declaration`. The confirmation
#: card was therefore a polished surface in front of an unimplemented feature,
#: which is precisely the shape this codebase exists to remove: a declaration
#: with no reachable behaviour.
#:
#: Rendering it anyway implied that saving was one step away. It is not, and a
#: user who writes an equity-compensation description deserves to be told that
#: before they invest in refining it.
UNAVAILABLE_TEMPLATES = {
    "rsu-vesting": (
        "Equity-compensation modelling is not available in this pilot. Your "
        "description was recognised as an RSU scenario, and this version "
        "cannot yet compile one into an executable plan — so there is nothing "
        "it could honestly show you. Historical account and contribution "
        "analysis is fully supported; describing your holdings and "
        "contributions directly will work."),
}


def _template_confirmation(request: Request, describe: str, stage1):
    """Answer for a description this build recognises but cannot execute.

    No plan-shaped record is created. A draft that cannot become a plan is a
    record whose only purpose is to look like progress.
    """
    hint = stage1.parsed.template_hint
    detail = UNAVAILABLE_TEMPLATES.get(hint)
    if detail is None:
        raise HTTPException(
            status_code=501,
            detail=f"{hint!r} has no handler in this build")
    return TEMPLATES.TemplateResponse(
        request, "unavailable.html",
        {"describe": describe, "capability": hint, "detail": detail},
        status_code=501,
    )


@router.get("/new", response_class=HTMLResponse)
def new_plan(request: Request, describe: str = ""):
    """The confirmation screen. Nothing is saved and nothing is committed."""
    if not describe.strip():
        return TEMPLATES.TemplateResponse(request, "new.html", {"result": None})

    from ..deploy.context import current as _deployment

    stage1 = parse_with_model(describe, mode=_deployment().model.mode.value,
                              client=_parser_client())

    # Dispatch *before* generic compilation, not after. Compiled first and
    # branched afterwards, a vest would already have been read as cash arriving
    # and then a purchase, and the RSU surface would be describing a scenario
    # built by the wrong semantics.
    if stage1.parsed.template_hint:
        return _template_confirmation(request, describe, stage1)

    compiled = compile_scenario(describe, name="draft", version=1,
                                benchmark_rule=BENCHMARK_RULE,
                                parsed=stage1.parsed)
    access = _market_data("draft scenario preview")
    run = (_run(compiled.scenario, access)
           if compiled.can_simulate and access.usable else None)

    from .feasibility import assess, blockers, classify

    feasibility = assess(compiled.scenario, access.frame)
    open_items = classify(compiled.scenario.provenance.unresolved,
                          executable=feasibility.can_execute)

    return TEMPLATES.TemplateResponse(
        request, "new.html",
        {
            "describe": describe,
            "result": compiled,
            "feasibility": feasibility,
            "open_items": open_items,
            "blockers": blockers(compiled.scenario,
                                 executable=feasibility.can_execute,
                                 stated_text=describe),
            "confirmation": compiled.confirmation(),
            "view": build_confirmation(compiled, text=describe),
            "suggested_title": _suggested_title(describe),
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
async def save_plan(request: Request, describe: str = Form(...),
                    title: str = Form(default=""),
                    parse: str = Form(default="")):
    """Commit what the user read, answered and confirmed.

    Everything arrives in one POST body from one form. It used to arrive in
    three places — `describe` and `plan_id` as query parameters, `parse` in the
    body, and the answer and confirmation radios *outside the form entirely* —
    so a user could read every question, click every answer and press Save
    while none of it was submitted. For any scenario with an open question the
    button did not render at all, and the journey dead-ended with no way
    forward. That was the pilot's blocking defect and no backend test could see
    it: they all construct a confirmed scenario directly.

    The form contract:

        title             the user's own name for the plan
        parse             the pinned stage 1 reading, re-verified on the way in
        answer:<field>    an answer to a question the compiler asked
        confirm:<field>   CONFIRMED or REJECTED for one inference

    **The identity is generated here.** It never derives from the title, the
    description or the route, so two plans may share a title, a title may be
    edited, and runs and worksheets keep pointing at the same opaque id.
    """
    form = await request.form()

    from ..deploy.context import current

    parser = current().model.identity()

    parsed = None
    if parse.strip():
        try:
            parsed = parse_from_stored(json.loads(parse), describe)
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"the submitted interpretation does not match the "
                       f"description: {exc}") from exc

    answered_at = pd.Timestamp.now("UTC").isoformat()
    amendments = tuple(
        ScenarioAmendment(question_id=key[len("answer:"):], answer=str(value),
                          recorded_at=answered_at)
        for key, value in form.multi_items()
        if key.startswith("answer:") and str(value).strip())

    decisions = {key[len("confirm:"):]: str(value)
                 for key, value in form.multi_items()
                 if key.startswith("confirm:")}

    # Acknowledgements: "continue without modelling this". Typed, recorded, and
    # only accepted for items the feasibility service classifies as separable —
    # the form may offer the control, but it does not decide whether the
    # control is permitted.
    acknowledged = {key[len("exclude:"):] for key, value in form.multi_items()
                    if key.startswith("exclude:") and str(value) == "on"}

    # Two passes, because a choice means different things depending on what
    # the compiler had proposed.
    #
    #   the user picks the value the compiler inferred  -> agreement
    #   the user picks a different value                -> a statement
    #
    # Both are decisions and only the second is stated information. Recording
    # agreement as a statement would credit the user with saying something the
    # system suggested; recording a correction as a confirmed inference would
    # credit the system with proposing what the user supplied.
    compiled = compile_scenario(describe, name=title or "plan", version=1,
                                benchmark_rule=BENCHMARK_RULE, parsed=parsed,
                                amendments=amendments)

    exclusions = _permitted_exclusions(compiled, acknowledged, answered_at)
    if exclusions:
        compiled = compile_scenario(describe, name=title or "plan", version=1,
                                    benchmark_rule=BENCHMARK_RULE,
                                    parsed=parsed, amendments=amendments,
                                    exclusions=exclusions)

    proposed = {one.field: one.value for one in compiled.scenario.provenance.inferred}
    corrections = tuple(
        ScenarioAmendment(question_id=field, answer=value,
                          recorded_at=answered_at)
        for field, value in decisions.items()
        if field in proposed and value != proposed[field])
    if corrections:
        amendments = amendments + corrections
        compiled = compile_scenario(describe, name=title or "plan", version=1,
                                    benchmark_rule=BENCHMARK_RULE,
                                    parsed=parsed, amendments=amendments,
                                    exclusions=exclusions)

    agreed = {field for field, value in decisions.items()
              if proposed.get(field) == value}
    scenario = _with_decisions(compiled.scenario, agreed)

    # Refused here, not merely warned about on the screen. A plan that cannot
    # execute saved successfully with zero runs and no worksheet while the
    # confirmation page displayed both "no price history" and "Ready to save".
    from .feasibility import assess, blockers

    verdict = assess(scenario, _market_data("feasibility check").frame)
    # Before the market-data verdict, because this refusal is unconditional
    # and that one is not. Asked to model 500 shares of AAPL, the feasibility
    # check answered "there is no price history for AAPL" — true, and
    # misleading: the plan would not run with prices either. A user takes that
    # message as an instrument problem and tries a different ticker.
    described_holding = historical_lots.detect(describe)
    if described_holding:
        raise HTTPException(
            status_code=422,
            detail=("This describes an existing holding. " +
                    described_holding[0].why_it_matters))

    if not verdict.can_execute:
        raise HTTPException(status_code=422, detail=verdict.detail)

    # Name what is blocking, from the same classification the screen rendered.
    # "still has unconfirmed inferences or open questions" told a user nothing
    # about which item stopped them or whether anything could be done — the
    # blocker was a forward projection the engine does not model, and the
    # message did not say so.
    # The description is passed, not inferred. The scenario does not carry its
    # own text, so a guard reading `scenario.stated_text` would have been a
    # control on a path nothing takes — dead in exactly the place it matters.
    outstanding = blockers(scenario, executable=verdict.can_execute,
                           stated_text=describe)
    if outstanding.any:
        raise HTTPException(status_code=422, detail=outstanding.detail())

    plan_id = new_plan_id()
    saved_at = pd.Timestamp.now("UTC").isoformat()
    try:
        _store().save_plan(
            plan_id=plan_id, owner=PILOT_OWNER, scenario=scenario,
            stated_text=describe, saved_at=saved_at, title=title.strip(),
            parse=parsed.to_json() if parsed is not None else None,
            parser=parser,
        )
    except NotSaveable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # The run is persisted before the worksheet that cites it. A worksheet
    # written first and back-filled would briefly name artifacts that were not
    # there, and "briefly" is exactly when a crash happens.
    access = _market_data("saving a scenario", plan_id=plan_id,
                          scenario=scenario, ran_at=saved_at)
    if access.usable:
        run = _run(scenario, access)
        if run.get("result") is not None:
            generate_worksheet(
                _store(), plan_id=plan_id, owner=PILOT_OWNER, scenario=scenario,
                run=run["result"].to_json(),
                comparison={**(run.get("payload") or {}),
                            **(run.get("comparability_records") or {})},
                ran_at=saved_at, title=title.strip() or plan_id, access=access)

    return RedirectResponse(f"/workspace/plans/{plan_id}", status_code=303)


def _permitted_exclusions(compiled, acknowledged, at: str):
    """Turn acknowledgements into typed exclusions, refusing the impermissible.

    The feasibility service decides what may be dismissed; this only records
    the user's decision about the ones it allows. A required clarification or a
    material item submitted as an exclusion is refused rather than honoured —
    the form is not the authority on whether proceeding is safe, and a
    hand-built POST must not be able to dismiss the thing a result depends on.
    """
    from ..mission.spec import ScenarioExclusion
    from .feasibility import Resolution, classify

    if not acknowledged:
        return ()
    items = {one.field: one
             for one in classify(compiled.scenario.provenance.unresolved)}
    permitted = []
    for field in sorted(acknowledged):
        item = items.get(field)
        if item is None:
            continue                       # already settled; nothing to exclude
        if item.resolution is not Resolution.UNSUPPORTED_SEPARABLE:
            raise HTTPException(
                status_code=422,
                detail=(f"'{item.subject}' cannot be set aside: "
                        + ("it is a required input, so there is nothing to "
                           "proceed without."
                           if item.resolution is Resolution.REQUIRED_CLARIFICATION
                           else "excluding it would change what the result "
                                "answers. Revise the description instead.")))
        permitted.append(ScenarioExclusion(
            item=item.field, reason=item.why_it_matters, acknowledged_at=at))
    return tuple(permitted)


def _with_decisions(scenario, agreed):
    """Mark the inferences the user agreed to, and only those.

    An inference nobody acted on stays unconfirmed and the store refuses the
    plan: silence is not agreement, which is the rule the whole confirmation
    screen exists to enforce.
    """
    from ..mission.scenario import ScenarioSpecification
    from ..mission.spec import Inference, Provenance

    if not agreed:
        return scenario
    source = scenario.provenance
    return ScenarioSpecification(**{
        **scenario.__dict__,
        "provenance": Provenance(
            stated=source.stated,
            inferred=tuple(
                Inference(one.field, one.value, one.why,
                          confirmed=one.field in agreed)
                for one in source.inferred),
            contradictions=source.contradictions,
            unresolved=source.unresolved,
            amended=source.amended)})


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

    access = _market_data("opening a saved plan")
    scope = (TEMPLATES_BY_HINT[compiled.template_hint].modelling_scope()
             if compiled.template_hint in TEMPLATES_BY_HINT else None)
    run = (_run(scenario_from_stored(stored, compiled.scenario), access, scope)
           if access.usable else None)

    return TEMPLATES.TemplateResponse(
        request, "plan.html",
        {
            # From the stored parse, not from the deployment. Reading
            # `current().model` here would re-describe an old plan with today's
            # configuration, which is the failure the whole pinning rule
            # exists to prevent.
            "parser_identity": (record.get("parse") or {}).get("parser"),
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
    # Pinned here too. A counterfactual claims the *only* difference is the
    # constraint, which under classifier @2 requires that every other dimension
    # was actually evaluated rather than absent on both sides.
    snapshot = f"prices@{sessions[-1].date()}"
    pins = pins_for(scenario, snapshot=snapshot)
    common = dict(
        **pins.as_conditions(),
        flow_schedule_hash=scenario.flow_schedule.schedule_hash,
        starting_capital=scenario.flow_schedule.starting_capital,
        cash_policy_rate=0.0, tax_treatment=scenario.tax_treatment,
        cost_bps=10.0,
        period_start=str(sessions[0].date()), period_end=str(sessions[-1].date()),
        allocation_rule_hash=scenario.rule_hash,
        data_snapshot=snapshot,
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


# --- research worksheets ---------------------------------------------------
#
# Three operations that look alike and must stay apart:
#
#     GET    /research/{id}              read the saved worksheet
#     POST   /research/{id}/reinterpret  compile the original words again
#     POST   /research/{id}/rerun        simulate again
#
# Only the first is what reopening owes. The plan page conflated them once —
# recompiling stored prose and simulating the fresh interpretation while
# displaying the stored scenario — and it reached compilation through a helper
# rather than directly, which is why the read path below takes no text at all.

@router.get("/research/{worksheet_id}", response_class=HTMLResponse)
def open_worksheet(request: Request, worksheet_id: str,
                   revision: Optional[int] = None):
    """Resolve stored references and render them. Nothing else.

    Side-effect free by construction: it has nowhere to pass original text, and
    a test asserts it names no compiler, simulator or writer.
    """
    from .worksheet import from_json
    from .worksheet_view import build as build_worksheet_view

    store = _store()
    record = store.get_worksheet(worksheet_id, PILOT_OWNER, revision)
    if record is None:
        raise HTTPException(status_code=404,
                            detail=f"no worksheet {worksheet_id!r}")

    worksheet = from_json(record["payload"])
    view = build_worksheet_view(worksheet, store=store, owner=PILOT_OWNER)
    return TEMPLATES.TemplateResponse(
        request, "worksheet.html",
        {"view": view,
         "revisions": store.worksheet_revisions(worksheet_id, PILOT_OWNER)},
    )


@router.post("/research/{worksheet_id}/reinterpret")
def reinterpret_worksheet(worksheet_id: str):
    """What today's compiler makes of the same words.

    Deliberately a separate, explicit action. Adopting a new interpretation
    changes what a saved worksheet means, and only its owner can agree to that —
    so this returns a proposal and writes nothing.
    """
    from .worksheet import from_json

    store = _store()
    record = store.get_worksheet(worksheet_id, PILOT_OWNER)
    if record is None:
        raise HTTPException(status_code=404,
                            detail=f"no worksheet {worksheet_id!r}")

    worksheet = from_json(record["payload"])
    plan = store.get_plan(worksheet.scenario_ref, PILOT_OWNER)
    if plan is None:
        raise HTTPException(
            status_code=409,
            detail=(f"worksheet {worksheet_id} cites scenario "
                    f"{worksheet.scenario_ref}, which is not in this workspace"))

    compiled = compile_scenario(plan["stated_text"], name=worksheet.scenario_ref,
                                version=1, benchmark_rule=BENCHMARK_RULE,
                                parsed=_pinned_parse(plan))
    return {"worksheet_id": worksheet_id,
            "revision": worksheet.revision,
            "migration": migration_for(worksheet.scenario_ref, plan["scenario"],
                                       compiled),
            "applied": False}


@router.post("/research/{worksheet_id}/intent")
def plan_worksheet_intent(worksheet_id: str, instruction: str = Form(...),
                          source_revision: Optional[int] = Form(None)):
    """Plan one instruction against this worksheet's persisted intent chain.

    Returns a proposal awaiting confirmation. **It never applies anything.**
    Acceptance is the separate endpoint below, on the transaction that was
    already proven, because a route that planned and applied in one call would
    decide on the user's behalf exactly where their judgement is the point.

    The history comes from the store, not the request. That is the whole reason
    this endpoint exists: `intent.plan` has always taken history and the live
    application never had any to give it, so every instruction arrived looking
    like the first one and repeated tuning counted as nothing.
    """
    from .intent_service import (
        IntentRefused,
        StaleInstruction,
        UntrustworthyHistory,
        plan_and_record,
    )

    store = _store()
    stamp = _now()
    try:
        planned = plan_and_record(
            store, worksheet_id=worksheet_id, owner=PILOT_OWNER,
            instruction=instruction,
            intent_id=f"{worksheet_id}-intent-{stamp}",
            proposal_id=f"{worksheet_id}-proposal-{stamp}",
            at=stamp, source_revision=source_revision,
            recorder=_recorder(worksheet_id=worksheet_id))
    except StaleInstruction as stale:
        raise HTTPException(status_code=409, detail=str(stale)) from stale
    except UntrustworthyHistory as broken:
        # 409 rather than 500. The request is well-formed; the stored history
        # is not, and that is a conflict with durable state rather than a bug in
        # handling this call.
        raise HTTPException(status_code=409, detail=str(broken)) from broken
    except IntentRefused as refused:
        raise HTTPException(status_code=404, detail=str(refused)) from refused

    return {"worksheet_id": worksheet_id, "applied": False, **planned.to_json()}


@router.post("/research/{worksheet_id}/proposals/{proposal_id}/accept")
def accept_worksheet_proposal(worksheet_id: str, proposal_id: str):
    """Apply a reviewed proposal through the existing transaction.

    This adds no application logic. `apply.accept` already orders the writes so
    nothing can be orphaned, refuses a stale proposal rather than rebasing it,
    and commits the runs and the revision together — none of which is worth
    reimplementing at the edge.
    """
    from .apply import ApplyRefused, StaleProposal, accept
    from .proposal import from_json as proposal_from_json

    store = _store()
    record = store.get_worksheet_proposal(proposal_id, PILOT_OWNER)
    if record is None:
        raise HTTPException(status_code=404,
                            detail=f"no proposal {proposal_id!r}")

    proposal = proposal_from_json(record["payload"])
    access = _market_data("candidate runs for an accepted proposal")
    try:
        result = accept(
            store, proposal_id=proposal_id, owner=PILOT_OWNER,
            worksheet_id=worksheet_id, proposal=proposal, at=_now(),
            # A scenario change needs a runner, and there is no price history
            # here to give it one. Passing None is what makes the apply path
            # refuse rather than write a revision citing runs that never
            # happened.
            run_candidate=(_candidate_runner(access, store, worksheet_id)
                           if access.usable else None),
            # The same access the runner closes over. Passed explicitly so the
            # delivery is recorded once and every candidate cites it, rather
            # than each candidate carrying its own — a fan-out whose members
            # cited different deliveries would be claiming they were measured
            # on different data, which is the one thing comparing them assumes
            # is untrue.
            access=access)
    except StaleProposal as stale:
        raise HTTPException(status_code=409, detail=str(stale)) from stale
    except ApplyRefused as refused:
        raise HTTPException(status_code=422, detail=str(refused)) from refused

    return {"worksheet_id": worksheet_id, "applied": True, **result.to_json()}


def _reconciliation_view(store, worksheet_id: str, *, as_of: str):
    """The three lanes, from persisted records.

    Verification is best-effort by design. The page shows history whatever
    happens: a re-derivation that cannot run leaves every row NOT_VERIFIABLE
    rather than taking the page down, because a record the user wrote is not
    less real because this build cannot re-judge it.
    """
    from ..mission.rsu_reconcile import ObservedEvent, PlannedEvent, reconcile
    from .reconciliation_view import RSUReconciliationView, verify

    planned = store.planned_events(worksheet_id, PILOT_OWNER)
    observed = store.observed_events(worksheet_id, PILOT_OWNER)
    stored = store.reconciliations(worksheet_id, PILOT_OWNER)

    verification = {}
    try:
        fresh = reconcile(
            [PlannedEvent(**one["payload"]) for one in planned],
            [ObservedEvent(**one["payload"]) for one in observed],
            as_of=as_of)
        verification = verify(stored, fresh)
    except Exception:                                          # noqa: BLE001
        verification = {}

    return RSUReconciliationView.from_records(
        planned, observed, stored, verification=verification)


@router.get("/research/{worksheet_id}/tracking", response_class=HTMLResponse)
def worksheet_tracking(request: Request, worksheet_id: str):
    """Planned, observed and reconciliation, side by side.

    Resolves stored records and arranges them. It matches nothing, computes no
    dates and decides no statuses — those were decided when the reconciliation
    was derived, and deciding them again here would produce a second answer.
    """
    store = _store()
    if store.get_worksheet(worksheet_id, PILOT_OWNER) is None:
        raise HTTPException(status_code=404,
                            detail=f"no worksheet {worksheet_id!r}")

    view = _reconciliation_view(store, worksheet_id, as_of=_now()[:10])
    return TEMPLATES.TemplateResponse(
        request, "tracking.html",
        {"worksheet_id": worksheet_id, "view": view.to_json()})


@router.post("/research/{worksheet_id}/rerun")
def rerun_worksheet(worksheet_id: str):
    """Simulate the stored scenario again, against today's data.

    Also separate, and also not something reopening does. A rerun produces a new
    run and therefore a new worksheet revision; it never overwrites the figures
    a saved worksheet cites.
    """
    raise HTTPException(
        status_code=501,
        detail=("rerun is not implemented yet. It must create a new run and a "
                "new worksheet revision rather than refresh the stored one, so "
                "it is left unbuilt rather than approximated"))
