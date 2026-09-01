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
from . import abuse, telemetry
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


def _emit_evaluation_funnel(reading: PilotReading,
                            run: Optional[Dict[str, Any]] = None, *,
                            route: str = "", participant: str = "",
                            latency_ms: Optional[float] = None) -> None:
    """Emit the §10 clarification + evaluation funnel events for one reading.

    Best-effort and text-free: it reads the *shape* of the reading and the run —
    is it still asking questions, did it refuse, did it produce a figure — and
    emits ids/counts/latency only. Never the prompt: the sentence is reduced to a
    length via `question_count`/nothing here carries it.
    """
    run = run or {}
    # Clarification: still asking, or sealed and settled.
    if reading.questions:
        telemetry.emit(telemetry.CLARIFICATION_REQUESTED, route=route,
                       participant=participant,
                       question_count=len(reading.questions))
    elif reading.intent is not None:
        telemetry.emit(telemetry.CLARIFICATION_COMPLETED, route=route,
                       participant=participant,
                       settled_count=len(reading.settled))

    # Evaluation outcome, from the run's shape.
    if reading.refusals:
        telemetry.emit(telemetry.EVALUATION_ABSTAINED, route=route,
                       participant=participant, outcome="refused",
                       refusal_count=len(reading.refusals),
                       latency_ms=latency_ms)
    elif run.get("result") is not None:
        telemetry.emit(telemetry.EVALUATION_COMPLETED, route=route,
                       participant=participant, outcome="figure",
                       latency_ms=latency_ms)
    elif run.get("unavailable") or run.get("strategy_not_executed"):
        telemetry.emit(telemetry.EVALUATION_FAILED, route=route,
                       participant=participant, outcome="no_figure",
                       latency_ms=latency_ms)


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
                # A data gap the person cannot close by editing a value: the
                # deployment holds no priced series for this plan at all.
                "refusal_kind": "data_gap",
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
        # The period the figure covers, when the plan named a trailing window the
        # engine restricted the replay to — so §6.B captions the return with it
        # ("over the past 5 years, this would have been the return") instead of
        # letting it read as the whole history. `short` says the snapshot did not
        # reach the full window back, so a figure is never labelled "5 years"
        # over three years of data. None when the plan ran over all history.
        "period": _period(run),
        "unavailable": run.get("unavailable"),
        # Which dimension the refusal lives in, so the page can tell the person
        # whether editing a value fixes it (`plan`), whether it is a data gap
        # they cannot close (`data_gap`), or neither (`internal`). None on a run
        # that produced a figure.
        "refusal_kind": run.get("refusal_kind"),
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
        # §6.D — the benchmarks as *labeled alternatives*, kept separate from the
        # user's interpreted strategy. Read straight off the run the page already
        # holds; nothing is recomputed and no model is consulted. The page had
        # these paths (they feed `_chart`) and never named them as the distinct
        # analytical alternatives they are.
        "alternatives": _alternatives(run),
        # The comparison's own words — that every benchmark received identical
        # contributions, so the only difference is the rule — and no ranking.
        "comparison_note": _comparison_note(run),
        # §6.C — evidence/reproducibility, all surfaced from provenance that
        # already travels with the result. Absent fields are None here and marked
        # (not fabricated) in the template.
        "market_snapshot": _market_snapshot(run),
        "disclosures": _disclosures(run),
        # Not classified on the public evaluation surface — the publication gate
        # that assigns a performance class is not on this path. Stated as None so
        # the panel can say "not classified here" rather than invent one.
        "performance_class": None,
        # Nor is a pilot evaluation bound to a single declared methodology/protocol
        # version. None, and marked as such; the panel still links to the public
        # methodology surface for the concept where one applies (§7).
        "methodology_id": None,
        "methodology_version": None,
        "protocol": None,
        # The revision this deployment is serving, from git state — a fact about
        # the build, not a recomputation of the strategy.
        "source_revision": _source_revision(),
    }


def _alternatives(run):
    """The benchmark comparisons as labeled alternatives — never merged into the
    user's strategy (§6.D).

    Each benchmark received the *same* contributions on the same days under the
    same costs and calendar, so the only difference is what the money bought.
    Read off `run["benchmarks"]`, which the run already carries and the chart
    already draws; this only names them. An incomparable benchmark keeps its row
    with `figure` None so a missing comparison never reads as a zero return.
    """
    out = []
    for b in run.get("benchmarks") or ():
        res = getattr(b, "result", None)
        out.append({
            "name": getattr(b, "name", ""),
            "description": getattr(b, "description", ""),
            "comparable": bool(getattr(b, "comparable", False)),
            "figure": (None if res is None else f"{res.final_value:,.2f}"),
            "gain": (None if res is None else getattr(res, "gain", None)),
        })
    return out


def _comparison_note(run):
    """The comparison's own disclaimer — identical flows, no ranking — if a
    payload was built. None when no figure ran."""
    payload = run.get("payload") or {}
    return payload.get("note")


def _market_snapshot(run):
    """The market-data snapshot id these figures were produced under, from the
    result's own recorded provenance. None when no figure ran or none was
    recorded — never a fabricated id."""
    result = run.get("result")
    if result is None:
        return None
    try:
        return result.market_data_json().get("snapshot_id")
    except Exception:  # noqa: BLE001
        return None


def _disclosures(run):
    """The declared-but-not-simulated limitations that already travel with the
    figure, read from the result's modelling scope (§6.C). Empty when none —
    an empty list, so the panel says "none recorded" rather than nothing."""
    result = run.get("result")
    scope = getattr(result, "modelling_scope", None) if result is not None else None
    if not scope:
        return []
    out = []
    for entry in scope.get("not_modelled") or ():
        if isinstance(entry, dict):
            out.append(entry.get("reason") or entry.get("field") or "")
    return [d for d in out if d]


_SOURCE_REVISION: Dict[str, Any] = {}


def _source_revision():
    """The short commit this deployment serves, best-effort and cached.

    Read from git state — a property of the build, not the strategy engine and
    not a model call — so an inspectable public result can name the revision that
    produced it (§1 fact 3). None when git is unavailable, and marked as such
    rather than guessed.
    """
    if "rev" not in _SOURCE_REVISION:
        import subprocess

        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5, check=False)
            _SOURCE_REVISION["rev"] = (
                out.stdout.strip() if out.returncode == 0 else None)
        except (OSError, subprocess.SubprocessError):
            _SOURCE_REVISION["rev"] = None
    return _SOURCE_REVISION["rev"]


#: The two published methodology concepts (`methodologies/*.yaml`) and the
#: strategy families each one documents. Only a family a published methodology
#: genuinely covers maps to a concept; everything else resolves to None and the
#: evidence panel links to the methodology index instead of claiming a specific
#: page. Honest by omission — a link is offered only where the page exists.
FAMILY_METHODOLOGY = {
    "cross_sectional_momentum": "xsmom",
    "time_series_momentum": "xsmom",
    "dual_momentum": "xsmom",
    "regime_momentum": "xsmom",
    "risk_parity": "hrp",
    "equal_risk_contribution": "hrp",
    "minimum_variance": "hrp",
    "max_diversification": "hrp",
}


def methodology_concept_for(picked: str) -> Optional[str]:
    """The public methodology concept a picked strategy belongs to, or None.

    Used by the evidence panel's back-link (§7): the daily research graphs and
    the evaluator are two faces of one engine, so a result points back at the
    methodology page for its concept where a published one exists. Resolved from
    the catalogue entry's declared family; typed prose and unmapped families get
    None (the panel then links to the methodology index).
    """
    if not picked:
        return None
    from .strategy_library import entry

    chosen = entry(picked)
    if chosen is None:
        return None
    return FAMILY_METHODOLOGY.get(chosen.family)


def _period(run):
    """The trailing window the figure covers, as `{label, short, start, end}`,
    or None when the plan ran over the whole history.

    Read from the resolved window on the result — the sessions actually
    evaluated, not the phrase asked — so the caption names the period the figure
    is really for. `short` (the snapshot did not reach the full window back) is
    carried through so §6.B can say "as far back as the data goes" rather than
    labelling a three-year figure "the past five years"."""
    resolved = run.get("resolved_window") if isinstance(run, dict) else None
    window = getattr(resolved, "window", None) if resolved is not None else None
    label = getattr(window, "label", "") if window is not None else ""
    if not label:
        return None
    return {
        "label": label,
        "short": bool(getattr(resolved, "short", False)),
        "start": getattr(resolved, "start", None),
        "end": getattr(resolved, "end", None),
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

    # Normalize before anything reads the sentence (§11): trim, collapse runs of
    # whitespace, cap the length. An ordinary sentence is unchanged, so a normal
    # evaluation keeps its exact content-addressed identity; what this removes is
    # the padded-out or whitespace-bomb input an abuser sends.
    describe = abuse.normalize_prompt(describe)

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
    _emit_evaluation_funnel(reading, run, route="/evaluate",
                            participant=participant)
    # When the evaluation is complete — sealed and asking nothing — persist the
    # content-addressed review so the Save button can carry its *id* rather than
    # the sentence. Saving then binds this stored artifact and re-reads nothing.
    # Idempotent and deterministic: the same evaluation rewrites the same row and
    # yields the same id, so two renders of one draft stay byte-identical.
    review_id = ""
    if reading.intent is not None and not reading.questions:
        from .pilot_store import save_review

        review_id = save_review(reading, picked)
    response = TEMPLATES.TemplateResponse(
        request, "pilot.html",
        dict(page(reading, text=describe, run=run), picked=picked,
             review_id=review_id,
             methodology_concept=methodology_concept_for(picked)))
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

    # Normalize before reading (§11), the same normalization the draft applies,
    # so the two entry points cannot disagree about what was submitted.
    describe = abuse.normalize_prompt(describe)

    # A prompt was submitted (§10). The event carries a *digest and length*, never
    # the words: `prompt_digest` is what keeps raw strategy text out of analytics.
    telemetry.emit(telemetry.PROMPT_SUBMITTED, route="/pilot/answer",
                   **telemetry.prompt_digest(describe))

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
    _emit_evaluation_funnel(answered, route="/pilot/answer",
                            participant=participant)
    target = f"/pilot/reviews/{review_id}"
    if stalled:
        target += "?stalled=1"
        if unmoved:
            target += f"&unchanged={quote(unmoved)}"
    response = RedirectResponse(target, status_code=303)
    attach(response, participant)
    return response


@router.get("/evaluate", response_class=HTMLResponse)
def evaluate(request: Request, describe: str = "", picked: str = ""):
    """The canonical public evaluator — a thin presentation/controller alias.

    `/evaluate` is the one public URL for the describe → clarify → evaluate flow
    (§3 of the public strategy-lab plan). It is presentation only: it holds no
    parser, compiler, clarification or evaluation logic of its own and delegates
    straight to `pilot_new`, the same entrypoint `/pilot` reaches. Two
    implementations of one evaluator would drift, and the drift shows up as two
    visitors getting different numbers from the same words — so there is exactly
    one implementation and this is a second name for its door.

    The older public URLs — `/workspace/new`, `/pilot`, `/pilot/answer` — keep
    serving the same journey unchanged, so nothing that links to them breaks
    while `/evaluate` becomes the name the site leads with.

    It does not accept holdings, tax profile, income or account state (§1 hard
    rule): the only inputs are a strategy description and an optional catalogue
    pick, exactly as the pilot draft takes. Keeping evaluation impersonal is
    what keeps the publisher position intact.
    """
    # The evaluator was opened (§10). Emitted here rather than in `draft` so it
    # counts a visit to the canonical public URL, not every internal render.
    telemetry.emit(telemetry.EVALUATOR_OPENED, route="/evaluate")
    return pilot_new(request, describe, picked)


@router.post("/evaluate")
async def evaluate_answer(request: Request, describe: str = Form(...),
                          picked: str = Form(""), from_review: str = Form("")):
    """Clarify and evaluate under the canonical name. Delegates to `pilot_answer`.

    The same handler `/pilot/answer` uses, so the evaluation an anonymous
    visitor gets from `/evaluate` is the one the legacy path produces from the
    same submission — there is no second strategy parser in the website layer.
    """
    return await pilot_answer(request, describe=describe, picked=picked,
                              from_review=from_review)


@router.get("/for-advisors", response_class=HTMLResponse)
def for_advisors(request: Request):
    """The public advisor narrative (§8 of the public strategy-lab plan, Gate 4).

    Informational only. It manages no households, reads no account state and
    takes no parameters — the same evaluated `SavedStrategyPlan` a person can
    already produce on the public evaluator is the portable input to Wealth
    Manager, and this page explains that lifecycle without becoming part of it.

    It is `PUBLIC_RESEARCH`: reachable without an account, carrying nothing a
    user wrote. Crucially it does not gate the evaluator — a demo/contact path
    is offered, never required, and `/evaluate` stays free whether or not anyone
    writes in.

    The stage labels are grounded in *deployed* status, not aspiration (this is
    Gate 4's honesty requirement). `Evaluate` and `Save strategy plan` are LIVE
    — they are exactly the Gate 1–3 public evaluator and the exact-save
    `SavedStrategyPlan` handoff. The four downstream stages — connect account,
    apply constraints, governed execution, continuous supervision — are Wealth
    Manager capabilities that run *simulation-first* and are not live: governed
    execution is simulated and live brokerage execution is gated on external
    broker/RIA authorization that does not yet exist. They are labelled roadmap
    / in development, never as live money movement, so the page's claims match
    the same declared-vs-realized rule the rest of the system enforces.
    """
    from .routes import TEMPLATES

    return TEMPLATES.TemplateResponse(request, "for_advisors.html", {})


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
    from .pilot_store import load_review, load_review_under
    from .owner import SHARED
    from .routes import TEMPLATES

    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused

    stored = load_review(review_id)
    if stored is None:
        # The anonymous → signed-in handoff. An evaluation run before signing in
        # is saved under the shared workspace (`owner.SHARED`), because that is
        # who the request was. The instant the visitor signs in to save it, the
        # current owner becomes their subject and this scoped read finds nothing
        # — so the review they just ran, with its figure and its description,
        # reads as "no longer available" the moment they authenticate. It is not
        # gone: it is content-addressed under SHARED, so fall back to reading it
        # there by its own id. This is the same handoff the exact-save flow
        # already relies on (see load_review_under); a review id is a hash of the
        # reading, so this reads a specific evaluated artifact the requester can
        # already name — never a way to browse another tenant's private plans,
        # which live in pilot_plans and stay scoped to the current owner.
        stored = load_review_under(SHARED, review_id)
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
    context["methodology_concept"] = methodology_concept_for(
        stored.get("picked", ""))
    context["review_id"] = review_id
    # A property of the transition that reached this page, not of the state
    # itself — which is why it arrives in the query and is not stored. The GET
    # remains a read: the same URL without it renders the same page.
    context["stalled"] = bool(stalled)
    context["unchanged"] = [d for d in unchanged.split(",") if d]
    response = TEMPLATES.TemplateResponse(request, "pilot.html", context)
    attach(response, participant)
    return response


def _unavailable(request: Request, message: str, *, status: int = 404):
    from .routes import TEMPLATES

    return TEMPLATES.TemplateResponse(
        request, "pilot.html",
        {"text": "", "reading": None, "unavailable": message},
        status_code=status)


async def _csrf_refusal(request: Request):
    """A 403 when CSRF is enforced and this state-changing POST lacks a valid
    token, or `None` to proceed (§11).

    Off unless `QUANTIFY_CSRF_ENFORCE` is set, so every existing caller keeps
    working; when on, a cross-site POST that cannot echo the cookie's token is
    refused. The token rides in the `csrf_token` form field, read from the same
    parsed form the handler already used — Starlette caches it, so this does not
    re-read the body.
    """
    if not abuse.csrf_enforced():
        return None
    form = await request.form()
    if abuse.verify_csrf(request, str(form.get(abuse.CSRF_FIELD, "") or "")):
        return None
    abuse.log_event("csrf_rejected", request, outcome="403")
    return _unavailable(
        request, "This save could not be verified as coming from the Quantify "
        "page you were on. Reload the evaluation and try Save again.",
        status=403)


def _bind_the_exact_artifact(request: Request, session):
    """Bind an already-evaluated review to the authenticated owner. No reader.

    This is the exact-save invariant made mechanical (§2/§4). The review named by
    the session is content-addressed, so loading it returns the *exact* evaluated
    artifact rather than a re-derivation; `reopen` takes a dict and cannot reach a
    reader, so nothing here interprets the sentence again. The plan identity is
    re-derived from the reopened artifact and asserted equal to the one the
    evaluation already determined — an equality a model call on this path could
    only break, which is why its holding is the proof no such call happened.

    Owner binding changes only the *envelope*: `save` scopes the write to the
    current (now-authenticated) owner, while the artifact's content hash — its
    identity — is untouched.
    """
    from fastapi.responses import RedirectResponse

    from .pilot_store import (load_review, load_review_under, plan_id_for, save)

    stored = load_review_under(session.review_owner, session.compiled_plan_hash)
    if stored is None:
        # The review may have been written under the current owner (a signed-in
        # visitor who evaluated their own strategy before saving it).
        stored = load_review(session.compiled_plan_hash)
    if stored is None:
        return _unavailable(
            request, "the evaluated plan behind this save is no longer "
            "available; evaluate it again to save it")

    reading = reopen(stored)                       # dict-only: no reader exists
    minted = plan_id_for(reading)
    # The structural invariant. `reopen` cannot construct a reader, so `minted`
    # is the plan the pre-login evaluation already fixed; a save that recomputed
    # the strategy would land on a different hash and trip this.
    assert minted == session.evaluated_plan_id, (
        "the reopened review yielded a different plan identity than was "
        "evaluated — this save is not binding the exact artifact")

    plan_id = save(reading)                        # owner is the envelope
    assert plan_id == minted, "the saved plan id is not its content address"

    participant = participant_in(request) or new_participant()
    # After the write, never before. A handler recording SAVED first would
    # report a save that failed.
    observe_save(reading, plan_id, participant)
    # Plan saved (§10). `recomputed=False` is the structural fact this path
    # guarantees: it bound a content-addressed review and ran no parser or
    # evaluator, which is what makes the save-without-recompute rate 100%.
    telemetry.emit(telemetry.PLAN_SAVED, route="/pilot/save/resume",
                   participant=participant, plan_id=plan_id, recomputed=False)
    response = RedirectResponse(f"/pilot/plans/{plan_id}", status_code=303)
    attach(response, participant)
    return response


def _begin_save(request: Request, review_id: str, picked: str):
    """Open a save over an already-evaluated review.

    Signed in (or a deployment with no accounts): bind it now. Signed out with a
    provider present: mint the single-use session, then send the visitor to sign
    in and return to `/pilot/save/resume`, where the exact artifact is bound to
    the owner they just proved. Either way no sentence is re-read — the review is
    already evaluated and content-addressed."""
    from fastapi.responses import RedirectResponse

    from . import evaluation_session as es

    try:
        session = es.create_for_review(review_id, picked)
    except es.SessionError as gone:
        return _unavailable(request, str(gone))

    from .auth_routes import _target, signed_in

    target = _target()
    if not target.configured or signed_in(request) is not None:
        return _bind_the_exact_artifact(request, session)

    # Anonymous, and this deployment has accounts. The Save is the first — and
    # only — authentication boundary: sign in, then come back to bind the exact
    # evaluated artifact. The review id already rode a public, refreshable URL,
    # and the single-use token is what the resume consumes.
    resume = quote(
        f"/pilot/save/resume?session={session.session_id}"
        f"&save_token={session.save_token}", safe="")
    return RedirectResponse(f"/auth/login?next={resume}", status_code=303)


def _complete_from_session(request: Request, session_id: str, save_token: str):
    """Consume the single-use token and bind, or handle a replay idempotently.

    The first arrival spends the token and saves. A second arrival — a
    double-click, a refreshed resume URL, a deliberate replay — finds the token
    already consumed and is sent to the plan that first save minted, rather than
    minting a second one or erroring. A token that matches no live session, or
    one whose session expired or was tampered with, is refused."""
    from fastapi.responses import RedirectResponse

    from . import evaluation_session as es

    session = es.consume_save_token(save_token)
    if session is not None and session.session_id == session_id:
        return _bind_the_exact_artifact(request, session)

    # Not consumable now. Distinguish a genuine replay of a completed save (send
    # them to the plan) from a stale, expired or forged link (refuse).
    from hmac import compare_digest

    existing = es.resolve(session_id)
    if (existing is not None and es.is_consumed(session_id)
            and compare_digest(save_token, existing.save_token)):
        return RedirectResponse(
            f"/pilot/plans/{existing.evaluated_plan_id}", status_code=303)
    return _unavailable(
        request, "this save link has already been used or has expired; open "
        "the plan from your saved plans, or evaluate the strategy again",
        status=409)


@router.post("/pilot/save")
async def pilot_save(request: Request, describe: str = Form(""),
                     review_id: str = Form(""), picked: str = Form(""),
                     session: str = Form(""), save_token: str = Form("")):
    """Persist the runtime artifact, not a rendering of it.

    **The exact-save path (§2/§4).** When the request names an already-evaluated
    artifact — a `review_id`, or a `session` + `save_token` — the save binds that
    stored, content-addressed review to the owner and never re-reads `describe`.
    No provider/model call is required merely to save an already-evaluated plan;
    `_bind_the_exact_artifact` proves it structurally.

    **The legacy direct path.** With only a `describe`, there is no prior review
    to bind, so the reading is built here. This is retained for the no-provider
    pilot and existing callers; the exact-save invariant governs the id path
    above, which is the one the evaluator's Save button now uses.
    """
    from fastapi.responses import RedirectResponse

    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused

    blocked = await _csrf_refusal(request)
    if blocked is not None:
        return blocked

    if session and save_token:
        return _complete_from_session(request, session, save_token)
    if review_id:
        telemetry.emit(telemetry.SAVE_CLICKED, route="/pilot/save",
                       review_id=review_id)
        return _begin_save(request, review_id, picked)

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


@router.post("/evaluate/save")
async def evaluate_save(request: Request, review_id: str = Form(...),
                        picked: str = Form("")):
    """Save an evaluated strategy — the public entry to the auth boundary.

    Reachable without an account (it is where the account is asked for): it names
    an already-evaluated, content-addressed review and either binds it now (a
    signed-in visitor) or mints the single-use session and redirects to sign in
    (an anonymous one). It never re-reads the sentence — the review is already
    evaluated — so the click that starts a save costs no model call."""
    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused
    blocked = await _csrf_refusal(request)
    if blocked is not None:
        return blocked
    # Save was clicked (§10). The auth-boundary entry; no prompt text travels.
    telemetry.emit(telemetry.SAVE_CLICKED, route="/evaluate/save",
                   review_id=review_id)
    return _begin_save(request, review_id, picked)


@router.get("/pilot/save/resume")
def pilot_save_resume(request: Request, session: str = "", save_token: str = ""):
    """Finish a save after signing in. Binds the exact evaluated artifact.

    The `next` an anonymous Save redirected through login. It sits behind the
    session gate, so it can only run for a now-authenticated visitor; it consumes
    the single-use token and binds the exact review this session named to that
    owner — with no reader on the path. It is idempotent under replay: the token
    is single-use and the plan is content-addressed, so a re-request lands on the
    same plan rather than minting a second."""
    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused
    if not (session and save_token):
        return _unavailable(
            request, "this save link is incomplete; evaluate the strategy again "
            "to save it", status=400)
    return _complete_from_session(request, session, save_token)


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
    # Reopening a saved plan re-executes it from the stored artifact (§10 rerun).
    telemetry.emit(telemetry.EVALUATION_RERUN, route="/pilot/plans",
                   participant=participant, plan_id=plan_id)
    context = page(reading, text=stored.get("text", ""), run=run)
    context["plan_id"] = plan_id
    context["methodology_concept"] = methodology_concept_for(
        stored.get("picked", ""))
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


@router.post("/pilot/plans/{plan_id}/monitor")
async def pilot_plan_monitor(request: Request, plan_id: str,
                             holdings_source: str = Form("SIMULATED")):
    """**Monitor This Strategy** — turn a saved plan into a durable monitored portfolio.

    Builds a versioned ``SavedStrategyPlan`` from the plan's sealed intent, hands its
    wire form to wealth-manager (``POST /app/portfolios/monitor``) which instantiates
    a monitored portfolio (simulated holdings now; imported/linked later), and redirects
    the user to that portfolio's Portfolio Operations workspace view. RAAAL builds +
    hands over a verified plan and navigates; it re-implements no portfolio logic.

    Degrades gracefully: if wealth-manager is not configured/reachable, the user is
    returned to the plan with a message rather than an error page — the save/evaluate
    surface is unaffected.
    """
    from fastapi.responses import HTMLResponse, RedirectResponse

    from . import owner as owner_mod
    from ..deploy.login import SESSION_COOKIE
    from .monitor_handoff import MonitorUnavailable, monitor_plan
    from .pilot_store import load
    from .routes import TEMPLATES

    refused = _refuse_unless_declared(request)
    if refused is not None:
        return refused
    blocked = await _csrf_refusal(request)
    if blocked is not None:
        return blocked

    stored = load(plan_id)
    if stored is None:
        return TEMPLATES.TemplateResponse(
            request, "pilot.html",
            {"text": "", "reading": None, "unavailable": "no such plan"},
            status_code=404)
    reading = reopen(stored)
    # Forward the signed-in user's own Zitadel token (the session cookie IS the
    # verified ID token) so wealth-manager authorizes the policy under the real
    # person, not a shared service principal — and no service secret is needed.
    user_token = request.cookies.get(SESSION_COOKIE, "")
    try:
        result = monitor_plan(stored, reading, plan_id=plan_id,
                              holdings_source=holdings_source,
                              owner_id=owner_mod.current(), user_token=user_token)
    except MonitorUnavailable:
        return HTMLResponse(
            f"<p>Monitoring isn't available yet — the Portfolio Operations service "
            f"is not reachable. <a href='/pilot/plans/{plan_id}'>Back to your plan</a>.</p>",
            status_code=503)
    except ValueError as exc:
        return HTMLResponse(
            f"<p>This plan can't be monitored: {exc}. "
            f"<a href='/pilot/plans/{plan_id}'>Back to your plan</a>.</p>",
            status_code=409)

    target = result.get("workspace_url") or f"/pilot/plans/{plan_id}"
    return RedirectResponse(target, status_code=303)
