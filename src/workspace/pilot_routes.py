"""The pilot draft page: the first route that reaches the runtime.

Mounted beside the existing workspace rather than replacing it. `/new` branches
on the deployment's *declared* parser mode, so which interpreter served a
request is a fact the deployment stated and the plan records — not something
inferred from which module happened to import.

    QUANTIFY_PARSER_MODE=RUNTIME    this path
    anything else                   the existing compile_scenario path

**No fallback between them.** A deployment that quietly served the legacy
grammar when the reader was unreachable would hand two users different products
under one name, which is the rule `ParserFallback.REFUSE` already states for the
model. The pilot refuses and says why.

What this page is for is narrow: submit a goal, see what the runtime understood,
answer what it could not settle, see what will execute and what is refused by
name. Everything else — proposals, observations, counterfactuals, rerun — stays
on the legacy path until a pilot user needs it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from ..deploy.context import ParserMode
from ..discovery.schema import QUANTIFY_SCHEMA
from .pilot import InterpreterUnavailable, PilotReading, answer, read, reopen

router = APIRouter()


def deployment_uses_the_runtime() -> bool:
    from ..deploy.context import current

    return current().model.mode is ParserMode.RUNTIME


def configured_reader():
    """The reader this deployment declared.

    The choice is resolved in `deploy.context`, not read here. A first version
    called `os.environ` in this function and `test_single_resolution` caught it
    immediately — which is the rule that module exists for: a request handler
    deciding for itself where its answers come from is how one instance serves
    fixtures while reporting them as the model's.
    """
    from ..deploy.context import PilotReader, current

    model = current().model
    if model.pilot_reader is PilotReader.RECORDED:
        from ..discovery.hosted_recording import RecordedHostedReader

        return RecordedHostedReader()

    from ..discovery.readers_quantify import HostedReader

    return HostedReader(model=model.model or "claude-sonnet-5")


def execute(reading: PilotReading, *, plan_id: str = "") -> Dict[str, Any]:
    """Run an executable plan and return the figure with its evidence.

    Through `execution.execute_compiled_plan`, which both paths call. The pilot
    does not get its own copy of Quantify's execution logic — two copies drift,
    and the drift shows up as two users getting different numbers from the same
    plan.

    A plan that is not executable returns no run at all rather than a partial
    one. A figure beside an unanswered question is a figure for a request
    nobody finished making.
    """
    if not reading.executable or reading.compiled is None:
        return {}

    from .run_boundary import execute_compiled_plan, market_data_for

    scenario = reading.compiled.scenario
    access = market_data_for(scenario, context="pilot", plan_id=plan_id)

    # The same guard the legacy path applies, and for the same reason: `_run`
    # takes the frame's index without checking, so an unusable source reaches
    # it as `None` and raises an AttributeError a user would see as a 500. An
    # absent data source is a refusal with a reason, not a crash.
    if not access.usable:
        return {"result": None, "strategy_not_executed": True,
                "unavailable": "market data is not available in this "
                               "deployment, so no figure can be produced for "
                               "this plan"}

    return execute_compiled_plan(scenario, access, stated_text=reading.text)


def page(reading: PilotReading, *, text: str,
         run: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the template needs, and nothing the template must interpret."""
    compiled = reading.compiled
    run = run or {}
    return {
        "run": run,
        # From the result, not the payload. The payload carries the
        # comparison — benchmarks, notes, the recommendation assessment — and
        # the figure a user came for is on the result itself. Reading the wrong
        # object produced a page that had executed a plan and showed no number.
        "figure": (None if run.get("result") is None
                   else f"{run['result'].final_value:,.2f}"),
        "gain": (None if run.get("result") is None
                 else getattr(run["result"], "gain", None)),
        "coverage": run.get("coverage"),
        "unavailable": run.get("unavailable"),
        "strategy_not_executed": run.get("strategy_not_executed", False),
        "text": text,
        "reading": reading,
        "understood": [f for f in reading.settled if f.value is not None],
        "questions": reading.questions,
        "absent": reading.absent_fields,
        "refusals": reading.refusals,
        "executable": reading.executable,
        "witnesses": reading.profile.to_json(),
        "reader_id": reading.reader_id,
        "interpreter": reading.interpreter_version,
        "intent_hash": None if reading.intent is None else reading.intent.intent_hash,
        "sealed": reading.intent is not None and reading.intent.is_verified,
        "derivation": {} if compiled is None else dict(compiled.derivation),
        "applied_defaults": [] if compiled is None else list(
            compiled.applied_defaults),
    }


def _refuse_unless_declared(request: Request):
    """A deployment that has not declared the runtime cannot reach it.

    Checked in the route rather than by mounting conditionally: the route
    table would otherwise depend on an environment variable, and the boundary
    sweep derives its inventory from that table — an endpoint that vanishes
    from the inventory is an endpoint nothing audits.
    """
    from .routes import TEMPLATES

    if deployment_uses_the_runtime():
        return None
    return TEMPLATES.TemplateResponse(
        request, "pilot.html",
        {"text": "", "reading": None,
         "unavailable": "this deployment does not declare "
                        "QUANTIFY_PARSER_MODE=RUNTIME, so the pilot "
                        "interpreter is not the one it serves"},
        status_code=404)


def draft(request: Request, describe: str = ""):
    """The pilot draft, as one implementation with two callers.

    `/new` reaches this when the deployment declares the runtime, which is how
    a cohort meets it — the entry point is the workspace's own, and the legacy
    path is not the default experience for a pilot that was meant to test the
    runtime. `/pilot` reaches the same function as a diagnostic alias, so there
    is never a second implementation to keep in step with this one.
    """
    from .routes import TEMPLATES

    if not describe.strip():
        return TEMPLATES.TemplateResponse(request, "pilot.html",
                                          {"text": "", "reading": None})
    try:
        reading = read(describe, configured_reader(), schema=QUANTIFY_SCHEMA)
    except InterpreterUnavailable as down:
        return TEMPLATES.TemplateResponse(
            request, "pilot.html",
            {"text": describe, "reading": None, "unavailable": str(down)},
            status_code=503)

    return TEMPLATES.TemplateResponse(
        request, "pilot.html",
        page(reading, text=describe, run=execute(reading)))


@router.get("/pilot", response_class=HTMLResponse)
def pilot_new(request: Request, describe: str = ""):
    """Diagnostic alias for `/new` under the runtime mode.

    Kept for development and deliberately **not** the cohort entry point: two
    URLs serving one journey is two surfaces to describe, and the experiment is
    about whether the workspace people already use is better with the runtime
    underneath it. Delegates rather than duplicating.
    """
    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused
    return draft(request, describe)


@router.post("/pilot/answer", response_class=HTMLResponse)
async def pilot_answer(request: Request, describe: str = Form(...)):
    """One human amendment, authored `USER` and carried into the intent."""
    from .routes import TEMPLATES

    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused

    form = await request.form()
    answers = {k[len("answer_"):]: str(v).strip()
               for k, v in form.items()
               if k.startswith("answer_") and str(v).strip()}

    try:
        reading = read(describe, configured_reader(), schema=QUANTIFY_SCHEMA)
    except InterpreterUnavailable as down:
        return TEMPLATES.TemplateResponse(
            request, "pilot.html",
            {"text": describe, "reading": None, "unavailable": str(down)},
            status_code=503)

    answered = answer(reading, answers)
    return TEMPLATES.TemplateResponse(
        request, "pilot.html",
        page(answered, text=describe, run=execute(answered)))


@router.post("/pilot/save")
async def pilot_save(request: Request, describe: str = Form(...)):
    """Persist the runtime artifact, not a rendering of it.

    What is stored is the pinned intent and the settled record. Reopening
    recompiles from that and never re-reads the sentence — which is the whole
    property, and the first place a person can exercise it.
    """
    from fastapi.responses import RedirectResponse

    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused

    from .pilot_store import save

    form = await request.form()
    answers = {k[len("answer_"):]: str(v).strip()
               for k, v in form.items()
               if k.startswith("answer_") and str(v).strip()}

    reading = read(describe, configured_reader(), schema=QUANTIFY_SCHEMA)
    if answers:
        reading = answer(reading, answers)

    plan_id = save(reading)
    return RedirectResponse(f"/pilot/plans/{plan_id}", status_code=303)


@router.get("/pilot/plans/{plan_id}", response_class=HTMLResponse)
def pilot_plan(request: Request, plan_id: str):
    """Reopen from the stored intent. No reader is constructed on this path."""
    from .pilot_store import load
    from .routes import TEMPLATES

    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused

    stored = load(plan_id)
    if stored is None:
        return TEMPLATES.TemplateResponse(
            request, "pilot.html",
            {"text": "", "reading": None, "unavailable": "no such plan"},
            status_code=404)

    reading = reopen(stored)
    # Executed from the persisted artifact, not from a fresh reading. The
    # figure a user sees on reopening is the figure their confirmed plan
    # produces, and nothing on this path can consult a model to get there.
    context = page(reading, text=stored.get("text", ""),
                   run=execute(reading, plan_id=plan_id))
    context["plan_id"] = plan_id
    context["reopened"] = True
    return TEMPLATES.TemplateResponse(request, "pilot.html", context)
