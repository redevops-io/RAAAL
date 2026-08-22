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

from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from ..deploy.context import ParserMode
from ..discovery.schema import QUANTIFY_SCHEMA
from .catalog_assumptions import assume
from .catalog_intent import reading_for
from .pilot import InterpreterUnavailable, PilotReading, answer, read, reopen
from .pilot_events import (answers_already_in_the_prompt, attempts_by, observe,
                           observe_resubmission, observe_save)
from .pilot_consent import may_keep_prose
from .pilot_session import (attach, last_prompt, new_participant,
                            participant_in, record as record_transcript)
from .strategy_library import origin_of

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

    from ..discovery.readers_quantify import configured_hosted_reader

    # One selector, shared. This logic lived here, in the recorder and in the
    # gate, and a fourth place that needed it never got it.
    return configured_hosted_reader()


def _declared_profile():
    """The witness profile this deployment declared, onto every artifact.

    Read from the resolved context rather than inferred from whether a syntax
    reader was constructed: `MODEL_ONLY` and `BOTH` are claims a plan carries
    for its whole life, and inferring one from a local variable is how an
    artifact comes to say it had two witnesses because an import succeeded.
    """
    from ..deploy.context import current

    return current().model.witnesses


def configured_syntax_reader():
    """The deterministic witness, if this deployment declared one.

    `None` when it did not, so the caller can tell "no syntax reader" from "a
    syntax reader that found nothing" — the same distinction `unread` makes for
    dimensions, and for the same reason: a silent witness and an absent one
    look identical in a fused decision unless something records which it was.

    Mirrors the hosted/recorded split. A `RECORDED` deployment replays parses
    from `parses.json`; a `HOSTED` one loads the real parser, which is a ~500MB
    model and seconds to start, so it is constructed once per process rather
    than per request.
    """
    from ..deploy.context import PilotReader, current

    model = current().model
    if not model.syntax_witness:
        return None

    if model.pilot_reader is PilotReader.RECORDED:
        from ..discovery.syntax_stanza import RecordedReader

        return RecordedReader()

    return _live_parser()


_PARSER: Dict[str, Any] = {}


def _live_parser():
    """One Stanza instance per process. Loading it per request would put a
    neural model load in front of every sentence somebody types."""
    if "reader" not in _PARSER:
        from ..discovery.syntax_stanza import StanzaReader

        _PARSER["reader"] = StanzaReader("en")
    return _PARSER["reader"]


def _answers_in(form) -> Dict[str, str]:
    """Which submitted values are the person's own words.

    One reader for every route that needs them, because two copies of "what
    counts as an answer" is how the save path and the answer path come to
    disagree.

    **Authorship comes from change, not from emptiness.** The form states what
    each row was offered as (`original_*`) and whose it was (`author_*`), and
    this compares them:

        submitted differs from what was offered   -> the person's, authored USER
        already theirs, carried back unchanged    -> stays theirs
        offered as an assumption, unchanged       -> stays an assumption

    The rule it replaces was "a non-empty box is the user's". That is
    correlated with authorship and is not the same fact, and the cost was
    real: to keep an assumed value ours it had to be rendered as an empty box,
    so a page with a dozen assumptions asked somebody to retype all twelve. The
    safety property is unchanged — pressing the button without touching a row
    still cannot turn a guess into a stated preference — but now the value is
    visible while it stays ours.

    A row carrying no `original_` is treated as an edit. Older forms and
    hand-made requests have no such field, and the safe reading of "somebody
    typed this and I cannot tell what it was offered as" is that it is theirs:
    it over-attributes to the person, never the other way.
    """
    answers: Dict[str, str] = {}
    for key, raw in form.items():
        if not key.startswith("answer_"):
            continue
        name = key[len("answer_"):]
        submitted = str(raw).strip()
        if not submitted:
            continue

        author = str(form.get(f"author_{name}", "") or "")
        if author == "USER":
            # Theirs already. Carried back so a later submission does not drop
            # it — the reading is rebuilt from scratch each time, and a value
            # left out of the answers would silently revert to whatever the
            # reader says.
            answers[name] = submitted
            continue

        original = form.get(f"original_{name}")
        if original is None or submitted != str(original).strip():
            answers[name] = submitted
    return answers


def _observe_attempt(request, describe: str, reading: PilotReading,
                     answers: Optional[Dict[str, str]] = None,
                     run: Optional[Dict[str, Any]] = None,
                     picked: str = "") -> str:
    """Everything one submission implies about the person making it.

    Returns the participant token so the caller can put it on the response.
    Ordered so the *previous* prompt is read before this one is written —
    reversing those two makes every submission look like a reword of itself.

    A submission after a first one is a resubmission whether the text changed
    or not: someone who resends the identical sentence after answering a
    question has still been sent round the loop, and that is the thing worth
    counting.
    """
    participant = participant_in(request) or new_participant()
    answers = answers or {}

    prior = attempts_by(participant)
    # Gated on the same predicate the recorder uses, so a declining
    # participant is never compared against a sentence that should not exist.
    previous = last_prompt(participant) if may_keep_prose(participant) else ""

    if prior:
        observe_resubmission(
            participant=participant, attempt=prior + 1,
            # `None`, not `False`, when there is nothing to compare against.
            changed=(None if not previous else previous.strip() != describe.strip()),
            answered=tuple(answers),
            repeated=answers_already_in_the_prompt(describe, answers))

    observe(reading, participant=participant, run=run)
    # Derived from the sentence and the pick, not taken from the form. Without
    # it the cohort measures the catalogue: sentences we wrote, read by a
    # reader we wrote, is a closed loop, and the rate it produces would say
    # nothing about whether anybody's own words work.
    record_transcript(
        participant, describe, attempt=prior + 1,
        origin=origin_of(describe, picked),
        questions=list(reading.questions),
        answered=dict(answers),
        refused=[getattr(r, "dimension", "") for r in reading.refusals],
        executable=bool(reading.executable))
    return participant


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
    from .ledger_view import lines as ledger_lines
    from .ledger_view import unfilled as unfilled_orders
    from .parameters import rows as parameter_rows
    from .parameters import unanswered as unanswered_parameters

    compiled = reading.compiled
    run = run or {}
    return {
        "run": run,
        # The ledger the figure was derived from.
        #
        # It has been built and reconciled on every run since the ledger
        # existed — `run_boundary` refuses to show a figure at all when the
        # rows and the result disagree — and no page has ever rendered it. A
        # person was shown a number and a chart derived from a state they
        # could not see.
        #
        # Rows rather than a summary: "what actually happened, at the price
        # that was actually available" is the claim, and a total cannot be
        # checked against a market while a line can.
        "ledger": ledger_lines(run),
        "unfilled": unfilled_orders(run),
        # The parameter table: settled, asked, refused and defaulted in one
        # list, and only the dimensions this sentence actually touched. Built
        # here rather than in the template because deciding which rows exist
        # is a reading of the reading, and a template that decided it would be
        # a second place where "what this plan has" is defined.
        "parameters": parameter_rows(reading),
        # The same list the table renders, so the button cannot offer to
        # answer a question the table does not show.
        "needed": unanswered_parameters(reading),
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
        # The comparison, drawn. Every run already computes the plan's path and
        # five benchmark paths; the page printed one number and discarded them.
        "chart": _chart(run),
        # Risk-adjusted performance, from the time-weighted return so
        # contributions neither flatter nor distort it. None when no figure ran.
        "performance": _performance(run),
    }


def _performance(run):
    """Sharpe, volatility and drawdown for the run, or None if it did not run.

    None rather than a raise, for the same reason `_chart` returns None: the
    figure is the point, and a statistic that could not be computed is not worth
    withholding it for."""
    result = run.get("result")
    if result is None:
        return None
    try:
        from ..mission.performance import from_path

        return from_path(result.path).as_dict()
    except Exception:  # noqa: BLE001
        return None


def _chart(run):
    """None rather than a raised error. A figure is the best part of the page
    and not the part worth failing it for: a plan that ran and produced a
    number should still show the number if the drawing fails."""
    try:
        from .comparison_chart import build

        return build(run)
    except Exception:  # noqa: BLE001
        return None


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


def draft(request: Request, describe: str = "", picked: str = ""):
    """The pilot draft, as one implementation with two callers.

    `/new` reaches this when the deployment declares the runtime, which is how
    a cohort meets it — the entry point is the workspace's own, and the legacy
    path is not the default experience for a pilot that was meant to test the
    runtime. `/pilot` reaches the same function as a diagnostic alias, so there
    is never a second implementation to keep in step with this one.
    """
    from .routes import TEMPLATES

    if not describe.strip():
        # A token on the empty page, before anything is typed.
        #
        # Issuing it only on submission would make the protocol unrunnable:
        # consent is recorded against a token, so there would be nothing to
        # record it against until after the participant's first sentence had
        # already been discarded — and the unprompted first phrasing is the
        # most informative thing they produce all session.
        empty = TEMPLATES.TemplateResponse(
            request, "pilot.html",
            {"text": "", "reading": None, "picked": picked})
        attach(empty, participant_in(request) or new_participant())
        return empty
    # A picked strategy is structured evidence, so it does not go to a model.
    #
    # The product knows which entry it offered; pasting that entry's sentence
    # into a box and asking what it means discards the one fact it was certain
    # of. `reading_for` returns None for an entry the table does not describe,
    # and then the sentence is read as any typed one would be — a fallback that
    # is measured rather than silent.
    reading = reading_for(picked, describe) if picked else None
    if reading is None:
        try:
            reading = read(describe, configured_reader(),
                           schema=QUANTIFY_SCHEMA,
                           profile=_declared_profile(),
                           syntax_reader=configured_syntax_reader())
        except InterpreterUnavailable as down:
            return TEMPLATES.TemplateResponse(
                request, "pilot.html",
                {"text": describe, "reading": None, "unavailable": str(down)},
                status_code=503)

    # The catalogue supplies what its own sentence does not say, and only for a
    # strategy that was picked from it. Typed prose gets nothing: we know which
    # family we offered, and we do not know what somebody meant by their own
    # words — inferring a family from free text would be a second classifier
    # deciding what to assume on the user's behalf.
    reading = assume(reading, picked) if picked else reading

    run = execute(reading)
    participant = _observe_attempt(request, describe, reading, run=run,
                                   picked=picked)
    response = TEMPLATES.TemplateResponse(
        request, "pilot.html",
        dict(page(reading, text=describe, run=run), picked=picked))
    attach(response, participant)
    return response


@router.get("/pilot", response_class=HTMLResponse)
def pilot_new(request: Request, describe: str = "",
              picked: str = ""):
    """Diagnostic alias for `/new` under the runtime mode.

    Kept for development and deliberately **not** the cohort entry point: two
    URLs serving one journey is two surfaces to describe, and the experiment is
    about whether the workspace people already use is better with the runtime
    underneath it. Delegates rather than duplicating.
    """
    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused
    return draft(request, describe, picked)


@router.post("/pilot/answer")
async def pilot_answer(request: Request, describe: str = Form(...),
                       picked: str = Form(""),
                       from_review: str = Form("")):
    """One human amendment, authored `USER`, persisted, then redirected to.

    **Post-Redirect-Get, and the redirect is the point.** This handler used to
    render HTML at its own URL. Nothing was stored, so the answers lived only
    in the request body: a refresh, a Back, or a pasted link issued a GET
    against a POST-only route and got `Method Not Allowed`, and Back returned
    to the last real GET — the empty form — discarding everything typed.

    So the write happens here and the rendering happens on the GET that follows
    it. The browser is never left sitting on a URL it cannot re-request, which
    is what makes refresh, Back and Forward ordinary navigation rather than
    ways to lose work or repeat it.
    """
    from fastapi.responses import RedirectResponse

    from .pilot_store import save_review

    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused

    form = await request.form()
    answers = _answers_in(form)

    # Answers are edits to the selection, so they enter the structured path as
    # `USER` values rather than being applied afterwards to a model's reading.
    reading = reading_for(picked, describe, edits=answers) if picked else None
    if reading is None:
        try:
            reading = read(describe, configured_reader(),
                           schema=QUANTIFY_SCHEMA,
                           profile=_declared_profile(),
                           syntax_reader=configured_syntax_reader())
        except InterpreterUnavailable as down:
            from .routes import TEMPLATES

            # Rendered rather than redirected: there is no state to persist,
            # so there is nothing for a GET to read. A redirect here would
            # point at a review that was never written.
            return TEMPLATES.TemplateResponse(
                request, "pilot.html",
                {"text": describe, "reading": None, "unavailable": str(down)},
                status_code=503)

    # Assumptions first, the person's answers second, so an edit wins. `settle`
    # appends, so the assumed entry survives underneath: the record says the
    # value is now theirs *and* that it did not start that way. Re-applying the
    # assumptions here rather than carrying them in the form is what stops a
    # posted field from being able to claim it was assumed.
    reading = assume(reading, picked) if picked else reading
    answered = answer(reading, answers)

    # Persist before redirecting. A 303 pointing at a row that was not written
    # is a worse failure than the one being fixed: the person's answers would
    # be gone *and* the URL would look like it held them.
    if answered.intent is None:
        from .routes import TEMPLATES

        # Nothing to review. A reading with no intent cannot be rebuilt by the
        # GET — `reopen` refuses it rather than re-reading the sentence — so
        # redirecting would point at a page that could only 404. Rendered
        # here, like the unavailable case, for the same reason.
        return TEMPLATES.TemplateResponse(
            request, "pilot.html",
            {"text": describe, "reading": None,
             "unavailable": "this reading produced no intent to review"},
            status_code=503)

    review_id = save_review(answered, picked)

    # A lap, not a step. The review is addressed by its content, so landing on
    # the id we came from means this submission settled nothing — the answers
    # were supplied and the state is identical. Redirecting silently would put
    # the person back on the page they just left with no explanation, which
    # reads as a broken button and is how a clarification loop feels from the
    # outside.
    #
    # Named rather than blocked: the answers may be genuinely unusable, and
    # refusing the submission would lose them. The page says which dimensions
    # were sent and did not move.
    # Two ways to arrive back where you started, and they are different
    # situations for the person:
    #
    #   nothing was edited        the form was sent as it was rendered, so
    #                             there was no new evidence to settle anything
    #   answers were sent         and the state is still identical, so the
    #                             values were unusable or already recorded
    #
    # Both land on the page they submitted from, and both read as a broken
    # button without a word. The marker is separate from the list because the
    # first case has no dimensions to name.
    stalled = bool(from_review) and from_review == review_id
    unmoved = ",".join(sorted(answers)) if stalled and answers else ""

    participant = _observe_attempt(request, describe, answered, answers,
                                   run=None, picked=picked)
    target = f"/pilot/reviews/{review_id}"
    if stalled:
        target += "?stalled=1"
        if unmoved:
            target += f"&unchanged={quote(unmoved)}"
    response = RedirectResponse(target, status_code=303)
    attach(response, participant)
    return response


@router.get("/pilot/reviews/{review_id}", response_class=HTMLResponse)
def pilot_review(request: Request, review_id: str,
                 stalled: str = "", unchanged: str = ""):
    """The persisted clarification state, read and rendered. Nothing else.

    **No reader is constructed on this path**, exactly as on
    `/pilot/plans/{plan_id}`. `reopen` takes a dict and cannot reach one, so
    refreshing this page cannot produce a Discovery call, cannot re-apply the
    answers, and cannot mint a second plan. Reopening a clarification is a read
    of what was settled, the same kind of operation as reopening a saved plan.
    """
    from .pilot_store import load_review
    from .routes import TEMPLATES

    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused

    stored = load_review(review_id)
    if stored is None:
        # 404 rather than a redirect to a fresh form. A person who followed a
        # stale link should be told the state is gone, not silently handed an
        # empty page that looks like their answers were never submitted.
        return TEMPLATES.TemplateResponse(
            request, "pilot.html",
            {"text": "", "reading": None,
             "unavailable": "this review is no longer available"},
            status_code=404)

    reading = reopen(stored)
    participant = participant_in(request) or new_participant()
    run = execute(reading)
    context = page(reading, text=stored.get("text", ""), run=run)
    context["picked"] = stored.get("picked", "")
    context["review_id"] = review_id
    # A property of the transition that reached this page, not of the state
    # itself — which is why it arrives in the query and is not stored. The GET
    # remains a read: the same URL without it renders the same page.
    context["stalled"] = bool(stalled)
    context["unchanged"] = [d for d in unchanged.split(",") if d]
    response = TEMPLATES.TemplateResponse(request, "pilot.html", context)
    attach(response, participant)
    return response


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
    answers = _answers_in(form)

    reading = read(describe, configured_reader(), schema=QUANTIFY_SCHEMA,
                   profile=_declared_profile(),
                   syntax_reader=configured_syntax_reader())
    if answers:
        reading = answer(reading, answers)

    participant = participant_in(request) or new_participant()
    plan_id = save(reading)
    # After the write, never before. A handler recording SAVED first would
    # report a save that failed.
    observe_save(reading, plan_id, participant)
    response = RedirectResponse(f"/pilot/plans/{plan_id}", status_code=303)
    attach(response, participant)
    return response


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
    participant = participant_in(request) or new_participant()
    run = execute(reading, plan_id=plan_id)
    observe(reading, plan_id=plan_id, participant=participant, reopened=True,
            run=run)
    context = page(reading, text=stored.get("text", ""), run=run)
    context["plan_id"] = plan_id
    context["reopened"] = True
    response = TEMPLATES.TemplateResponse(request, "pilot.html", context)
    attach(response, participant)
    return response


@router.get("/pilot/plans/{plan_id}/runtime-artifact")
def pilot_plan_runtime_artifact(request: Request, plan_id: str):
    """Export a saved plan as a canonical runtime artifact (dual identity).

    This is the boundary crossing: a downstream runtime (wealth-manager) fetches
    the plan's canonical runtime artifact — the native `intent_hash` carried as
    `source_intent_hash`, plus the `rcv1` `runtime_artifact_hash`. Derived from the
    stored intent on demand; nothing is recomputed or mutated on RAAAL's side.
    """
    from fastapi import Response
    from fastapi.responses import JSONResponse

    from .pilot_store import load
    from .runtime_export import runtime_artifact_for

    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused
    stored = load(plan_id)
    if stored is None:
        return JSONResponse({"error": "no such plan"}, status_code=404)
    reading = reopen(stored)
    if reading.intent is None:
        return JSONResponse(
            {"error": "plan has no sealed intent to export"}, status_code=409)

    artifact = runtime_artifact_for(reading.intent, label=plan_id)
    # The ETag IS the canonical identity (freeze plan §6.2): a strong validator,
    # so a consumer that already holds this exact runtime artifact can revalidate
    # with `If-None-Match` and get 304 instead of re-reading a byte-identical body.
    # The identity is content-addressed, so this can never serve a stale artifact
    # under a matching tag.
    etag = f'"{artifact["runtime_artifact_hash"]}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return JSONResponse(artifact, headers={"ETag": etag})
