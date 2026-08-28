"""Quantify investment-agent API.

The public surface for quantify.club. Two rules shape it:

**Impersonal by construction.** The product's regulatory position rests on the
publisher's exclusion (*Lowe v. SEC*, 472 U.S. 181; the 2024 Seeking Alpha
dismissal). That holds only while output is impersonal and disinterested. There
is therefore no endpoint that accepts a user's holdings, and none that tailors a
response to a caller. Adding one is not a feature decision — it changes which
body of law applies to every backtest already published.

**No number without its class.** Every performance figure is served as a record
carrying its `performance_class` and the disclosure bound to it, so a figure
cannot be copied out of the API without its caveat. Mixed-class series are
refused rather than rendered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from contextlib import asynccontextmanager

import logging

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from .evaluation import ProtocolRegistry
from .ledger import Ledger
from .methodology import MethodologyRegistry, merge
from .methodology.spec import PerformanceClass
from .policy import HARD_BLOCKERS, Decision, PolicyRegistry, Surface
from .web import router as ui_router
from .workspace.routes import router as workspace_router

API_VERSION = "0.2.0"

SOURCE_REPOSITORY = "https://github.com/redevops-io/RAAAL"


def source_url() -> str:
    """The corresponding source for *the revision running here*.

    AGPL §13 offers users the source of the version they are interacting with.
    A bare repository link is an offer for whatever the default branch holds
    when they click it, which after a few merges is a different program.

    Falls back to the repository when the deployment cannot say what it is —
    an offer that is too broad, rather than one that is silently wrong. The
    build manifest reports `observable: False` in that case, so the two
    statements agree.
    """
    from .deploy.context import current

    commit = (current().build.private() or {}).get("commit")
    if commit:
        return f"{SOURCE_REPOSITORY}/tree/{commit}"
    return SOURCE_REPOSITORY


SOURCE_URL = SOURCE_REPOSITORY

#: AGPL §13 requires that users interacting with the software over a network are
#: offered its source. Serving it from the API root is how that offer is made
#: visible rather than merely satisfied in the repository.
def license_notice() -> dict:
    """The notice, with the source offer resolved for the running revision.

    A function rather than the module-level constant it used to be, because
    the constant was built at import time and the deployment identity is not
    known until the context resolves. An offer computed once at import is an
    offer for whatever the repository held when the process started, which is
    the moving-target problem one level in.
    """
    return {**LICENSE_NOTICE, "source": source_url()}


LICENSE_NOTICE = {
    # The §13 entitlement is unchanged by the Commons Clause — that condition
    # removes the right to *sell*, and says nothing about offering source. Both
    # are named because naming only the first would now be inaccurate, and this
    # object is the machine-readable statement of what the service is.
    "license": "AGPL-3.0-or-later WITH Commons-Clause",
    "spdx": "LicenseRef-AGPL-3.0-or-later-with-Commons-Clause",
    "source": SOURCE_REPOSITORY,
    "notice": (
        "This service is AGPL-3.0-or-later with the Commons Clause condition. "
        "You are entitled to the complete corresponding source of the version "
        "running here. The Commons Clause additionally withholds the right to "
        "sell this software."
    ),
}

DEMO_NOTICE = (
    "DEMO — decision support only, not investment advice. Paper trading; no real "
    "orders are ever placed."
)

_registry = MethodologyRegistry()
_protocols = ProtocolRegistry()
_policies = PolicyRegistry()
_ledger = Ledger()


def _bootstrap() -> None:
    """Publish on-disk methodologies and protocols into the ledger at startup.

    Idempotent: publishing no-ops on identical content and raises on a changed
    body under an existing id, so an edited published file is caught here rather
    than silently diverging from what results cite.
    """
    for m in _registry.load_all():
        _ledger.publish_methodology(m)
    for p in _protocols.load_all():
        _ledger.publish_protocol(p)


#: The startup preflight's outcome, held for the readiness endpoint.
#:
#: Readiness and liveness are separate on purpose. A migration mismatch should
#: make the service *unready* — visible to a load balancer, still diagnosable by
#: an operator — rather than crash-looping a container that cannot be inspected.
#: What must not happen is a user-facing request being served in that state.
_PREFLIGHT: Dict[str, Any] = {"outcome": None}


def preflight_outcome():
    return _PREFLIGHT["outcome"]


@asynccontextmanager
async def _lifespan(_: FastAPI):
    import datetime as dt

    from .deploy.context import bind, bound, resolve as resolve_context
    from .deploy.preflight import Profile, Result, run as run_preflight

    # In production `create_app` has already resolved, judged and bound one,
    # and this reuses it: resolving again here would reintroduce the second
    # reader the context exists to remove, and a deployment whose environment
    # changed between the two calls would be preflighted as one thing and
    # served as another. Resolving only happens on the imported-application
    # path, where nothing has established a deployment at all.
    context = bound() or bind(resolve_context())

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    # Run again rather than reusing `create_app`'s outcome: the preflight is a
    # check, not a resolution, and re-checking the same context is idempotent.
    # Reusing a held outcome would let a process that has served before report
    # the previous deployment's verdict for this one.
    outcome = run_preflight(checked_at=stamp, context=context)
    _PREFLIGHT["outcome"] = outcome

    if not outcome.ready:
        # The detail is for logs and the private proof record, never for a
        # client. It names the host, the revision or the drifted column.
        LOG.error("preflight %s: %s", outcome.result.value, outcome.detail)
        if outcome.profile is Profile.PRODUCTION or \
                outcome.result is Result.MIGRATION_MISMATCH:
            # A schema mismatch stops any profile. Serving a database whose
            # columns this code does not expect moves the failure onto the
            # first request that touches one, which is how it reaches a user
            # instead of an operator.
            raise RuntimeError(
                f"preflight {outcome.result.value}; refusing to serve. "
                "See the operator log for detail.")

    _bootstrap()
    try:
        yield
    finally:
        # A bound context outlives the application object, and a process that
        # serves several — every test that builds a client — would otherwise
        # inherit the previous deployment's answer.
        from .deploy.context import unbind

        unbind()


LOG = logging.getLogger(__name__)


class DeploymentRefused(RuntimeError):
    """The deployment did not establish what it must before serving.

    Raised by `create_app` rather than by a request handler, so a refusal
    happens before the process is capable of accepting traffic at all — not on
    the first request that reaches whichever code path noticed.
    """

    def __init__(self, outcome) -> None:
        self.outcome = outcome
        super().__init__(
            f"deployment preflight {outcome.result.value}; refusing to serve. "
            "See the operator log for detail.")


def create_app() -> FastAPI:
    """The production entrypoint.

    Uvicorn is pointed at this rather than at the module-level `app`, so the
    preflight runs while the process is still starting and a refusal prevents
    the server binding at all. Importing a ready-made application would run the
    checks in a lifespan hook, after the socket is open.

    Until Gate 3 nothing served this application at all — no uvicorn invocation
    existed anywhere, and the container ran the Bokeh dashboard instead. Every
    control Gate 2 built sat on a path no deployment took.
    """
    import datetime as dt
    import json as _json

    from .deploy.context import bind, resolve as resolve_context
    from .deploy.preflight import Profile, run as run_preflight

    # Uvicorn installs its own logging config and does not propagate this
    # module's logger, so a proof written with `LOG.info` never reached the
    # container log. It is attached to uvicorn's own logger, which is the one
    # an operator is actually reading.
    log = logging.getLogger("uvicorn.error")

    # And the same for everything else under `src`, not just this module.
    #
    # Borrowing uvicorn's logger by hand fixed the one line whose absence was
    # noticed. Every other module still called `getLogger(__name__)` into a
    # logger with nowhere to send anything, so `parse_model`'s "stage1
    # provider call" — written, with a comment saying so, expressly to be
    # counted against a live server — was never emitted by one. Three reopens
    # of a saved plan logged zero provider calls, and a fresh draft that
    # certainly made one logged zero as well. The measurement said what we
    # wanted to hear because it could not say anything else.
    #
    # Attached at the package so a module needs to do nothing to be heard.
    # The per-module alternative is a convention, and this is what happened to
    # the convention.
    package = logging.getLogger(__package__ or "src")
    if not package.handlers:
        source = log
        while source and not source.handlers:
            source = source.parent
        package.handlers = list(source.handlers) if source else []
    package.setLevel(logging.INFO)
    package.propagate = False

    # Emitted through a deep module's own logger, not through `log`. That is
    # the whole claim: a module far from this file, using
    # `getLogger(__name__)` and doing nothing special, is heard.
    #
    # It exists so a count taken from these logs has a premise. "Zero provider
    # calls on reopen" was reported from a stream that could not carry the
    # line at all, and read as a clean result. A measurement is worth nothing
    # until something has shown it can produce a non-zero.
    logging.getLogger("src.mission.parse_model").info(
        "instrumentation self-test: package logging reaches the operator log")

    # Resolve, judge *that object*, then serve under it. The preflight once
    # validated PostgreSQL while the store opened a local SQLite file, because
    # each read the environment for itself and neither was wrong on its own
    # terms. Passing the resolved context to both is what makes them the same
    # answer rather than two answers that usually agree.
    context = resolve_context()

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    outcome = run_preflight(checked_at=stamp, context=context)
    _PREFLIGHT["outcome"] = outcome

    if not outcome.ready:
        # Detail goes to the operator, never to a client: it names the host,
        # the revision or the drifted column.
        log.error("preflight %s: %s", outcome.result.value, outcome.detail)
        LOG.error("preflight %s: %s", outcome.result.value, outcome.detail)
        if outcome.profile is Profile.PRODUCTION:
            raise DeploymentRefused(outcome)

    # Only now, and only this object: nothing serves under a context the
    # preflight did not judge.
    bind(context)
    app.state.deployment = context

    proof = _json.dumps(outcome.proof(), sort_keys=True)
    log.info("deployment proof %s", proof)
    LOG.info("deployment proof %s", proof)
    log.info("deployment context %s",
             _json.dumps(context.to_json(), sort_keys=True))
    return app


app = FastAPI(
    title="investment-agent (Quantify Investment OS)",
    version=API_VERSION,
    description=(
        "Versioned, evidence-backed investment methodologies. Every result links to "
        "its methodology version, run manifest, trial count and disclosure."
    ),
    lifespan=_lifespan,
)


# One translation for every route. Installed on the application rather than
# written into handlers: there are 49 handlers and one boundary, and a handler
# that sanitises its own database exception is a handler that can forget.
from .web.failure import install as install_failure_handling  # noqa: E402

install_failure_handling(app)


#: The response headers that must be on every answer this application gives.
#:
#: They used to be Caddy's, and moving the deployment to Kubernetes removed
#: Caddy and took all four with it — silently, because nothing serves worse for
#: their absence and no check looked. That is the argument for putting them
#: here: a security header that exists only while a particular proxy is in
#: front is a header that disappears exactly when the fronting changes, which
#: is the moment nobody is reading response headers.
#:
#: HSTS is safe to assert because every hostname is HTTPS-only at Cloudflare
#: already; it tells the browser to stop trying the other scheme.
SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


@app.middleware("http")
async def _security_headers(request, call_next):
    """Set them, rather than default them.

    Assignment and not `setdefault`: if something upstream has already put a
    weaker value on the response, the weaker value is the one that would
    survive, and "a header is present" is not the property worth having.
    """
    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers[name] = value
    return response


# --- the research dashboard ------------------------------------------------
#
# Served from a directory the deployment writes rather than from the image.
# The dashboard is a 6MB Bokeh document rebuilt from the day's history, so it
# is data, not code: baking it into an image would pin research output to a
# release, and every rebuild would ship a stale chart until the next deploy.
#
# It went missing when `daily-deploy.yml` was replaced. That workflow ran the
# backtest, built this document and rsynced it onto the Proxmox host, which
# mounted it at /research. The replacement deployed the application and nothing
# else, so the route 404'd and the graphs simply stopped existing.
#
# `/var/lib/quantify` is already mounted into the container, so the host's cron
# writes here and the application serves whatever is present. When nothing has
# been built the page says so rather than 404ing: a missing dashboard is a
# pipeline that has not run, and "not built yet" is a different thing to tell
# somebody than "no such page".
# The research → evaluator affordance (§7 of the public strategy-lab plan).
#
# The daily graphs and the evaluator are two faces of one engine, not separate
# dashboards, so every research view offers a contextual move into the
# evaluator. It carries only *public strategy semantics* — an optional catalogue
# pick that lands in the evaluator's own textarea — and never any holdings, tax
# or account state (§1 hard rule): personalization is the Wealth Manager
# boundary, and implying it here would be the exact confusion the boundary
# exists to prevent.
_RESEARCH_TO_EVALUATE = (
    '<div style="font:14px/1.5 system-ui,sans-serif;padding:12px 18px;'
    'border-bottom:1px solid #d9dee5;background:#f5f7fa;color:#16202b;'
    'display:flex;gap:16px;align-items:baseline;flex-wrap:wrap">'
    '<span>This is public research, refreshed daily.</span>'
    '<a href="/evaluate" style="font-weight:600;color:#1f6feb">'
    'Describe your own strategy &rarr;</a>'
    '<a href="/evaluate?picked=scheduled-funding" style="color:#1f6feb">'
    'Evaluate a variation</a>'
    '</div>'
)


def _with_evaluate_affordance(html: str) -> str:
    """Prepend the research → evaluate banner (§7). Public semantics only.

    Inserted after the document's opening ``<body>`` where there is one, so the
    banner sits inside the rendered page; otherwise prepended. String work on the
    served document only — the daily build is untouched.
    """
    lower = html.lower()
    marker = lower.find("<body")
    if marker != -1:
        close = html.find(">", marker)
        if close != -1:
            return html[: close + 1] + _RESEARCH_TO_EVALUATE + html[close + 1 :]
    return _RESEARCH_TO_EVALUATE + html


@app.get("/research", response_class=HTMLResponse)
def research() -> HTMLResponse:
    from .deploy.context import current

    built = Path(current().research_directory) / "regime_dashboard.html"
    if not built.exists():
        return HTMLResponse(
            _with_evaluate_affordance(
                "<!doctype html><meta charset=utf-8>"
                "<title>Research dashboard — Quantify</title>"
                "<body>"
                "<h1>The research dashboard has not been built yet.</h1>"
                "<p>It is rebuilt from the day's history by a scheduled job on the "
                "host. If this persists past the next run, the job is failing "
                "rather than the page being missing.</p>"),
            status_code=503)
    return HTMLResponse(_with_evaluate_affordance(built.read_text()))


# --- service metadata ------------------------------------------------------

app.include_router(ui_router)

# The private workspace. A separate router with its own templates and its
# own store, mounted at its own prefix, so it can move to a different
# hostname without any code moving with it.
app.include_router(workspace_router)

# The pilot surface. Mounted always and *served* only when the deployment
# declares `QUANTIFY_PARSER_MODE=RUNTIME` — the routes themselves check, so
# that a deployment which has not made the declaration cannot reach the runtime
# by knowing a URL. Mounting conditionally would have made the route table
# depend on an environment variable, and `test_boundary_sweep` derives its
# inventory from that table.
from .workspace.pilot_routes import router as pilot_router  # noqa: E402
from .workspace.pilot_session import note_departures  # noqa: E402

app.include_router(pilot_router)

# Sign-in. Mounted always; each route checks whether this deployment declares a
# provider, so a build without one refuses with a sentence rather than a 404.
from .workspace.auth_routes import router as auth_router  # noqa: E402

app.include_router(auth_router)


#: Paths that require a verified session where this deployment has accounts,
#: **derived from the boundary manifest** rather than hand-written here.
#:
#: This pair used to be two standalone string lists that the middleware trusted
#: and the manifest knew nothing about — so a route could be public in one and
#: private in the other, which is exactly the "path-list fragility" §5 of the
#: public strategy-lab plan set out to remove. The login decision now lives in
#: one place: `boundary_manifest`, where each route declares its `BoundaryClass`.
#: `require_a_signed_in_viewer` reads that classification directly (via
#: `login_required`), and these two tuples are computed *from the same manifest*,
#: so they cannot diverge from it.
#:
#: They are kept as module constants because the deploy acceptance check
#: (`deploy/acceptance.py`) and `test_private_surface_is_derived` import
#: `PRIVATE_PREFIXES` to probe each gated mount once. `gated_prefixes()` returns
#: the minimal umbrellas — today `("/pilot", "/workspace")` — so those consumers
#: see the same shape they always did, now provably in step with the manifest.
#:
#: `/auth/*` is deliberately open — a login that required a login could not be
#: started — and so are `/ui`, `/research`, the evaluator (`/evaluate`,
#: `/workspace/new`, `/pilot/answer`) and the health endpoints.
#:
#: `/pilot/plans/{plan_id}` is the one that historically mattered: it resolves a
#: stored plan for `owner.current()`, which for a viewer with no session is the
#: shared `pilot` workspace — so anybody with a plan id could read a plan saved
#: there. It is AUTHENTICATED_PERSISTENCE in the manifest and gated by the
#: `/pilot` umbrella, exactly as before.
from .workspace.boundary_manifest import (  # noqa: E402
    gated_prefixes, login_required, public_within_gated)

PRIVATE_PREFIXES = gated_prefixes()

# Evaluating a plan is public — anyone can try the evaluator without an account.
# An account is the price of *keeping* a plan, not of seeing what one does, so
# the describe→evaluate→answer flow is exempt from the login while saving a plan
# and reading saved ones stay behind it. Derived from the manifest's
# PUBLIC_EVALUATION carve-outs that sit under a gated mount.
PUBLIC_WITHIN_PRIVATE = public_within_gated()


@app.middleware("http")
async def require_a_signed_in_viewer(request, call_next):
    """The workspace, for whoever proved who they are.

    This replaces the shared basic-auth credential that guarded these paths
    from the proxy. That credential was a door, not an identity: everybody
    holding it was the same person as far as the application was concerned, so
    a plan could not belong to anybody in particular.

    Only where accounts exist. A deployment that declares no issuer keeps the
    proxy's password — it is not identity, but it is the difference between a
    private workspace and a public one, and removing it there would publish
    every plan.

    Redirects rather than refusing, because somebody arriving at a private page
    without a session has not done anything wrong; they are not signed in yet.
    The path they asked for is carried in `next` so they land where they meant
    to rather than at the front door.

    **The public/private decision is the manifest's**, read through
    `login_required`, not recomputed from a local string list. That is what
    makes it impossible for a newly mounted route to be classified one way here
    and the other way in `boundary_manifest`.
    """
    if login_required(request.url.path):
        from .workspace.auth_routes import _target, signed_in

        target = _target()
        if target.configured and signed_in(request) is None:
            import urllib.parse

            from fastapi.responses import RedirectResponse

            destination = urllib.parse.quote(
                request.url.path
                + (f"?{request.url.query}" if request.url.query else ""),
                safe="/?=&")
            return RedirectResponse(f"/auth/login?next={destination}",
                                    status_code=303)
    return await call_next(request)


@app.middleware("http")
async def _note_the_viewer(request, call_next):
    """Establish who is looking, once, for the page header.

    Middleware rather than a dependency on every handler: the header is
    rendered by every page, and the sign-in link spent a week existing in the
    routing table while no page displayed it — a per-handler value is one the
    next handler forgets.

    Never raises and never refuses. This only decides which link the header
    shows; the workspace is still guarded by the proxy, and every page that
    reads a token verifies it there. A cookie that cannot be verified simply
    means nobody is signed in.
    """
    from .workspace.routes import _LOOKING

    try:
        from .workspace.auth_routes import signed_in

        token = _LOOKING.set(signed_in(request))
    except Exception:  # noqa: BLE001 - a header must not be able to fail a page
        token = _LOOKING.set(None)
    try:
        return await call_next(request)
    finally:
        _LOOKING.reset(token)

# A pilot participant reaching the legacy workspace is the strongest negative
# signal the study can collect, and the only one no counter would otherwise
# show: `plan_compiled` staying flat looks the same whether people tried the
# runtime and went back to what they knew, or never arrived at all. Middleware,
# so the legacy handlers stay unaware there is an experiment.
note_departures(app)


@app.get("/health/live")
def live() -> Dict[str, Any]:
    """Liveness. The process exists and is answering.

    Deliberately says nothing about whether it can serve. A dashboard
    responding on a port has never been evidence of pilot readiness, and a
    liveness probe that reported readiness would make an unready instance
    indistinguishable from a working one.
    """
    return {"live": True}


@app.get("/health/ready")
def ready() -> Dict[str, Any]:
    """Whether this instance may serve requests.

    Carries the outcome and nothing about why: a client learns that the service
    is not ready, not that its migration head is behind or which host it could
    not reach. The reasons are in the operator log and the startup proof.
    """
    from fastapi.responses import JSONResponse

    outcome = preflight_outcome()
    if outcome is None:
        # The factory has not run. Either the application was imported rather
        # than created, or the preflight has not completed.
        return JSONResponse({"ready": False}, status_code=503)
    return JSONResponse(outcome.public(),
                        status_code=200 if outcome.ready else 503)


@app.get("/ready")
def ready_alias() -> Dict[str, Any]:
    """Kept so an existing probe does not silently start failing."""
    return ready()


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness plus the compatibility facts a client needs.

    The public view only. An image digest and a migration head describe how to
    attack the deployment, not how to interoperate with it.
    """
    from .deploy.context import current

    return {
        "status": "ok",
        "version": API_VERSION,
        "build": current().build.public(),
        "paper_only": True,
        "external_execution_path": False,
        "notice": DEMO_NOTICE,
    }


def _service_data_policy() -> Optional[Dict[str, Any]]:
    """What to say about the data behind every figure, derived from the
    snapshot the deployment actually resolved — synthetic, vendor-attributed,
    or none configured. It was once "the synthetic disclosure, or None when
    licensed"; the vendor branch made silence the wrong answer."""
    from .workspace.routes import _data_notice

    return _data_notice()


@app.get("/info")
def info() -> Dict[str, Any]:
    concepts = _registry.concepts()
    return {
        "service": "investment-agent (Quantify Investment OS)",
        "version": API_VERSION,
        "spec_version": "0.1",
        "paper_only": True,
        "external_execution_path": False,
        "notice": DEMO_NOTICE,
        "personalization": {
            "enabled": False,
            "reason": (
                "Output is impersonal by design. Personalized output would forfeit "
                "the publisher's exclusion under Lowe v. SEC and bring every "
                "published backtest under the SEC Marketing Rule."
            ),
        },
        # The single most important caveat about every figure this service
        # produces, on the surface that describes the service.
        #
        # `/info` is linked from every page footer as "Service details" and is
        # what an integrator reads. It carried the demo notice and the licence
        # and said nothing about the numbers being invented, so the disclosure
        # on every HTML page stopped at the one endpoint that exists to answer
        # "what is this".
        #
        # Read from `_data_notice` rather than restated: a second copy of a
        # disclosure is the one that goes stale when the policy changes.
        "data_policy": _service_data_policy(),
        "methodologies": concepts,
        "license": license_notice(),
    }


# --- methodologies ---------------------------------------------------------


@app.get("/methodologies")
def list_methodologies() -> Dict[str, Any]:
    """All published methodology versions. Versions coexist; none replaces another."""
    return {"methodologies": _ledger.list_methodologies()}


@app.get("/methodologies/{concept}")
def get_concept(concept: str, version: Optional[int] = None) -> Dict[str, Any]:
    """Resolve a concept. Omit `version` for latest, supply it to pin."""
    try:
        m = _registry.get(concept, version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    versions = _registry.versions(concept)
    return {
        "methodology": m.to_json(),
        "resolved": "pinned" if version is not None else "latest",
        "available_versions": [v.version for v in versions],
        "trials_observed": _ledger.trial_count(concept),
    }


@app.get("/methodologies/{concept}/diff")
def diff_versions(concept: str, base: int, target: int) -> Dict[str, Any]:
    """Structural diff between two versions of one concept."""
    try:
        a = _registry.get(concept, base)
        b = _registry.get(concept, target)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    left, right = a.canonical_form(), b.canonical_form()
    changed = {
        key: {"base": left.get(key), "target": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }
    contract_breaks = b.contract.breaks_compatibility_with(a.contract)
    return {
        "base": a.version_id,
        "target": b.version_id,
        "changed_fields": changed,
        "contract_breaks": contract_breaks,
        "comparability": "broken" if contract_breaks else "comparable",
        "change_rationale": b.change_rationale,
    }


@app.get("/methodologies/{concept}/compare")
def compare_versions(concept: str) -> Dict[str, Any]:
    """Compare published figures across versions of one concept.

    Refuses to present a naive ranking. Two versions can differ in contract *and*
    in the period they were evaluated over — a longer lookback needs more warmup,
    so it starts later — and a table sorted by return would silently compare
    strategies measured on different data. Both conditions are reported.
    """
    try:
        versions = _registry.versions(concept)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not versions:
        raise HTTPException(status_code=404, detail=f"unknown concept {concept!r}")

    entries: List[Dict[str, Any]] = []
    for m in versions:
        for record in _ledger.list_performance(m.version_id):
            entries.append(
                {
                    "version_id": m.version_id,
                    "metric": record["metric"],
                    "value": record["value"],
                    "period_start": record["period_start"],
                    "period_end": record["period_end"],
                    "performance_class": record["performance_class"],
                    "trials_at_publication": record["trials_at_publication"],
                    "cost_model": record["cost_model"],
                }
            )

    blockers: List[str] = []
    periods = {(e["period_start"], e["period_end"]) for e in entries}
    if len(periods) > 1:
        blockers.append(
            "evaluation periods differ across versions: "
            + "; ".join(f"{s} .. {e}" for s, e in sorted(periods))
        )

    for older, newer in zip(versions, versions[1:]):
        breaks = newer.contract.breaks_compatibility_with(older.contract)
        if breaks:
            blockers.append(f"{older.version_id} -> {newer.version_id}: {'; '.join(breaks)}")

    classes = {e["performance_class"] for e in entries}
    if len(classes) > 1:
        blockers.append(f"mixed performance classes: {sorted(classes)}")

    return {
        "concept_id": f"methodology/{concept}",
        "entries": entries,
        "directly_comparable": not blockers,
        "comparability_blockers": blockers,
        "note": (
            "Figures are reported side by side, not ranked. Ranking versions "
            "measured over different periods or under different contracts would "
            "compare different things."
        )
        if blockers
        else None,
        "trials_observed": _ledger.trial_count(concept),
    }


@app.post("/methodologies/{concept}/merge")
def merge_versions(concept: str, base: int, ours: int, theirs: int) -> Dict[str, Any]:
    """Three-way merge of two independent revisions.

    Returns a verdict per layer rather than a boolean — see `methodology.merge`.
    """
    try:
        result = merge(
            _registry.get(concept, base),
            _registry.get(concept, ours),
            _registry.get(concept, theirs),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result.to_json()


# --- performance -----------------------------------------------------------


@app.get("/performance")
def list_performance(
    version_id: Optional[str] = None,
    include_superseded: bool = Query(
        False, description="Superseded figures are retained, not deleted."
    ),
) -> Dict[str, Any]:
    """Published figures. Each carries its class, disclosure and trial count."""
    records = _ledger.list_performance(version_id, include_superseded)
    classes = {r["performance_class"] for r in records}
    return {
        "performance": records,
        "performance_classes_present": sorted(classes),
        "linking_prohibited": len(classes) > 1,
        "linking_note": (
            "Records of different performance classes must not be combined into a "
            "single series. GIPS prohibits linking actual to theoretical performance."
        )
        if len(classes) > 1
        else None,
    }


@app.get("/performance/series")
def performance_series(version_id: str, metric: str) -> Dict[str, Any]:
    """A single-class series for charting.

    Refuses to return a mixed-class series. This is the data-model enforcement of
    the GIPS rule — a chart that flows from backtest into live cannot be built
    from this endpoint because the endpoint will not serve the underlying data.
    """
    records = [
        r for r in _ledger.list_performance(version_id) if r["metric"] == metric
    ]
    if not records:
        raise HTTPException(status_code=404, detail="no performance records match")

    classes = {r["performance_class"] for r in records}
    if len(classes) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"refusing to serve a mixed-class series ({sorted(classes)}). "
                "Request one class at a time; linking them is prohibited."
            ),
        )
    return {
        "version_id": version_id,
        "metric": metric,
        "performance_class": records[0]["performance_class"],
        "disclosure": records[0]["disclosure"],
        "points": [
            {"value": r["value"], "period_end": r["period_end"], "run_id": r["run_id"]}
            for r in records
        ],
    }


# --- runs, trials, errata --------------------------------------------------


@app.get("/runs")
def list_runs(version_id: Optional[str] = None) -> Dict[str, Any]:
    return {"runs": _ledger.list_runs(version_id)}


@app.get("/trials/{concept}")
def trials(concept: str) -> Dict[str, Any]:
    """Platform-observed trial count for a methodology lineage.

    This is the number an individual researcher cannot be relied upon to report,
    because they see only the result they chose to publish. It is the input to
    the Deflated Sharpe Ratio, and it only increases.
    """
    if concept not in _registry.concepts():
        raise HTTPException(status_code=404, detail=f"unknown concept {concept!r}")
    breakdown = _ledger.trial_breakdown(concept)
    return {
        "concept": concept,
        "trials_observed": _ledger.trial_count(concept),
        "breakdown": breakdown,
        "note": (
            "Counted by the platform across every version of this lineage AND every "
            "evaluation protocol those versions ran under, not self-reported. "
            "Searching over protocols is multiple testing exactly as searching over "
            "parameters is. Required for multiple-testing correction."
        ),
    }


@app.get("/policies")
def list_policies() -> Dict[str, Any]:
    """Versioned statistical policies.

    Thresholds live here rather than in code because they are product decisions
    that will be revised. A published result cites the policy version it was
    judged under, so raising a bar later does not retroactively re-judge it.
    """
    return {
        "policies": [p.to_json() for p in _policies.load_all()],
        "available": _policies.names(),
    }


@app.get("/policies/{name}")
def get_policy(name: str, version: Optional[int] = None) -> Dict[str, Any]:
    try:
        p = _policies.get(name, version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "policy": p.to_json(),
        "resolved": "pinned" if version is not None else "latest",
        "available_versions": [v.version for v in _policies.versions(name)],
    }


@app.get("/surfaces")
def list_surfaces() -> Dict[str, Any]:
    """Publication surfaces and what each asserts.

    Different surfaces make different claims, so one universal threshold cannot
    serve them. A private draft asserts nothing; a validated badge asserts a
    strict standard was met.
    """
    return {
        "surfaces": [s.value for s in Surface],
        "decisions": [d.value for d in Decision],
        "hard_blockers": list(HARD_BLOCKERS),
        "note": (
            "Failed results are not hidden. The gate controls whether a result may "
            "be represented as validated and what disclosure travels with it."
        ),
    }


@app.get("/protocols")
def list_protocols() -> Dict[str, Any]:
    """Published evaluation protocols.

    `methodology + evaluation protocol = performance`. A figure that cites only a
    methodology is not reproducible: the costs, execution lag, walk-forward grid
    and data snapshot all live here.
    """
    return {
        "protocols": _ledger.list_protocols(),
        "available": _protocols.names(),
        "note": (
            "Protocols are versioned and hashed like methodologies. Searching over "
            "protocols is multiple testing and is counted as such."
        ),
    }


@app.get("/protocols/{name}")
def get_protocol(name: str, version: Optional[int] = None) -> Dict[str, Any]:
    try:
        p = _protocols.get(name, version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "protocol": p.to_json(),
        "resolved": "pinned" if version is not None else "latest",
        "available_versions": [v.version for v in _protocols.versions(name)],
    }


@app.get("/holdout-unlocks")
def holdout_unlocks() -> Dict[str, Any]:
    """Every sealed-holdout opening, with reason and authorizer.

    A sealed holdout opens once. This log is what makes that claim checkable
    rather than asserted.
    """
    return {"unlocks": _ledger.list_holdout_unlocks()}


@app.get("/errata")
def list_errata() -> Dict[str, Any]:
    """Published corrections. Superseded results are linked, never removed."""
    return {"errata": _ledger.list_errata()}


# --- discovery (v8 Discovery Runtime interfaces) ---------------------------


@app.get("/project/discoveries")
def discoveries() -> Dict[str, Any]:
    """Detected changes worth acting on, as governed mission proposals.

    Shapes follow the Context Runtime v8 Discovery interfaces (Signal →
    Detection → Correlation → Hypothesis → MissionProposal) so this surface can
    be backed by that runtime rather than a parallel implementation.

    Detection is deterministic and evidence-cited; nothing here proposes a change
    to a published methodology directly. A proposal is an input to review.
    """
    proposals: List[Dict[str, Any]] = []

    for concept, versions in _registry.concepts().items():
        if len(versions) < 2:
            continue
        latest = _registry.get(concept)
        previous = _registry.get(concept, sorted(versions)[-2])
        breaks = latest.contract.breaks_compatibility_with(previous.contract)
        if breaks:
            proposals.append(
                {
                    "kind": "COMPARABILITY_BROKEN",
                    "subject": latest.concept_id,
                    "hypothesis": {
                        "claim": (
                            f"{previous.version_id} results are not comparable to "
                            f"{latest.version_id}"
                        ),
                        "supporting_evidence": breaks,
                        "contradicting_evidence": [],
                        "verification_plan": [
                            f"re-run {previous.version_id} under the {latest.version_id} contract",
                            "compare against the published figure",
                        ],
                    },
                    "suggested_template": "SupersessionMission",
                    "approval_policy": "human_required",
                    "evidence": {"change_rationale": latest.change_rationale},
                }
            )

    for erratum in _ledger.list_errata():
        proposals.append(
            {
                "kind": "ERRATUM_PUBLISHED",
                "subject": erratum.get("version_id") or "platform",
                "hypothesis": {
                    "claim": erratum["title"],
                    "supporting_evidence": [erratum["summary"]],
                    "contradicting_evidence": [],
                    "verification_plan": ["confirm superseded figures are flagged"],
                },
                "suggested_template": "ErratumMission",
                "approval_policy": "published",
                "evidence": {"supersedes": erratum["supersedes"]},
            }
        )

    return {
        "discoveries": proposals,
        "note": (
            "Proposals only. Discovery never mutates a published methodology; it "
            "produces evidence-backed work for review."
        ),
    }


@app.get("/project/learning")
def learning() -> Dict[str, Any]:
    """What the platform has learned about its own published results."""
    errata = _ledger.list_errata()
    superseded = sum(len(e["supersedes"]) for e in errata)
    return {
        "errata_published": len(errata),
        "performance_records_superseded": superseded,
        "methodology_versions": len(_ledger.list_methodologies()),
        "trials_by_concept": {
            c: _ledger.trial_count(c) for c in _registry.concepts()
        },
    }


# --- current strategies ----------------------------------------------------


@app.get("/current-strategies")
def current_strategies() -> Dict[str, Any]:
    """Latest published version of each methodology, with its integrity context."""
    out = []
    for concept, versions in _registry.concepts().items():
        m = _registry.get(concept)
        perf = _ledger.list_performance(m.version_id)
        out.append(
            {
                "concept_id": m.concept_id,
                "version_id": m.version_id,
                "title": m.title,
                "objective": m.objective,
                "content_hash": m.content_hash,
                "grounded_in": [c.to_json() for c in m.grounded_in],
                "assumptions": list(m.assumptions),
                "limitations": list(m.limitations),
                "cost_model": m.cost_model,
                "trials_observed": _ledger.trial_count(concept),
                "published_performance": perf,
                "deprecation_date": m.deprecation_date,
            }
        )
    return {"strategies": out, "notice": DEMO_NOTICE}


@app.get("/")
def root(request: Request):
    # The graphs are the front door. A browser landing on the site is sent to
    # the research dashboard; a client that asked for JSON — an API caller, a
    # health probe — still gets the service descriptor, so nothing
    # machine-readable moved. The descriptor also stays reachable at /info.
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/research", status_code=307)
    return JSONResponse(
        {
            "service": "investment-agent (Quantify Investment OS)",
            "version": API_VERSION,
            "notice": DEMO_NOTICE,
            "license": license_notice(),
            "docs": "/docs",
            "dashboard": "/research",
        }
    )
