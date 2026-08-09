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


def page(reading: PilotReading, *, text: str) -> Dict[str, Any]:
    """What the template needs, and nothing the template must interpret."""
    compiled = reading.compiled
    return {
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


@router.get("/pilot", response_class=HTMLResponse)
def pilot_new(request: Request, describe: str = ""):
    """Submit a goal and see what the runtime made of it."""
    from .routes import TEMPLATES

    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused

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

    return TEMPLATES.TemplateResponse(request, "pilot.html",
                                      page(reading, text=describe))


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

    return TEMPLATES.TemplateResponse(
        request, "pilot.html", page(answer(reading, answers), text=describe))


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
    context = page(reading, text=stored.get("text", ""))
    context["plan_id"] = plan_id
    context["reopened"] = True
    return TEMPLATES.TemplateResponse(request, "pilot.html", context)
