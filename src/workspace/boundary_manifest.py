"""Every endpoint declares which side of the boundary it serves.

An undeclared endpoint is the failure mode. A new route added next quarter will
be written by someone thinking about the feature, not about the publisher
position, and it will default to whatever the framework defaults to. So the
manifest is checked against the live application: a route that is not declared
here fails the test suite until someone decides which side it is on.

This is the same rule the artifact model applies everywhere else — a declaration
with no realization, or a realization with no declaration, is a defect — pointed
at the HTTP surface.

**Two axes, one declaration.** A route carries two facts that used to live in two
places that could disagree:

    exposure         does this endpoint carry a *private artifact* — one
                     person's plan — or only impersonal published research?
                     This is the publisher-position axis, and it is what the
                     `/ui` and `/workspace` sweeps have always checked.

    access           may an anonymous visitor reach it, or does it require a
                     signed-in session? This is what `require_a_signed_in_viewer`
                     in `src.api` decides on every request.

They correlate but are not the same fact, and the gap between them is real: the
public evaluator (`/workspace/new`, `/pilot/answer`, `/evaluate`) is reachable
without an account — an account is the price of *keeping* a plan, not of seeing
what one does — yet `/workspace/new` is mounted under the private workspace store
and its artifact lineage is private. The old code encoded that gap as two string
lists in `src.api` (`PRIVATE_PREFIXES` + `PUBLIC_WITHIN_PRIVATE`) that the
middleware trusted and the manifest knew nothing about — so a new route could be
public in one and private in the other. Now both facts are declared here, once,
per route, and the middleware *derives* its decision from this manifest rather
than from a parallel list it could drift from.

`BoundaryClass` is the finer, plan-named access taxonomy (§5 of the public
strategy-lab plan). `Exposure` is the coarse publisher-position projection the
existing sweeps read; it is derived from the class, with a per-route override for
the evaluation surfaces that live under a private mount.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

from ..mission.boundary import Visibility


class Exposure(str, Enum):
    PUBLIC = "PUBLIC"
    """Impersonal. May be read by anyone; may never contain private artifacts."""

    PRIVATE = "PRIVATE"
    """One person's. May cite public artifacts; never reachable from public ones."""

    INFRASTRUCTURE = "INFRASTRUCTURE"
    """Docs, schema, health. Carries no artifacts of either kind."""


class BoundaryClass(str, Enum):
    """The access taxonomy: which of the four plan-named surfaces a route is.

    The login decision is a property of the *class*, not of a path string, so a
    route cannot be public in the middleware and private in the manifest — there
    is only one place the answer lives.
    """

    PUBLIC_RESEARCH = "PUBLIC_RESEARCH"
    """Published methodology, protocol, policy, runs, performance, the research
    dashboard. Impersonal by construction; no login."""

    PUBLIC_EVALUATION = "PUBLIC_EVALUATION"
    """The anonymous evaluator: describe a strategy, clarify it, see what it
    does. `/evaluate`, `/pilot/answer`, `/workspace/new`. No login — evaluation
    is free; an account is the price of keeping a plan, not of running one."""

    AUTHENTICATED_PERSISTENCE = "AUTHENTICATED_PERSISTENCE"
    """Saving and reading a plan that belongs to somebody: `/pilot/save`,
    `/pilot/plans/{id}`, `/workspace/plans/{id}`, and the workspace mounts
    themselves. Requires a signed-in session."""

    PRIVATE_FINANCIAL_STATE = "PRIVATE_FINANCIAL_STATE"
    """Exports and anything account-specific: the runtime-artifact export a
    downstream runtime fetches. Requires a signed-in session."""

    INFRASTRUCTURE = "INFRASTRUCTURE"
    """Docs, schema, health, and the login itself. Carries no artifact; the
    login cannot require a login."""

    @property
    def requires_login(self) -> bool:
        """Whether an anonymous request to this class must be sent to sign in.

        The single source of truth the middleware reads. PUBLIC_* and
        INFRASTRUCTURE are open; the two private classes are gated.
        """
        return self in (BoundaryClass.AUTHENTICATED_PERSISTENCE,
                        BoundaryClass.PRIVATE_FINANCIAL_STATE)

    @property
    def default_exposure(self) -> Exposure:
        """The coarse publisher-position projection of this class.

        Overridable per route: a PUBLIC_EVALUATION surface mounted under a
        private workspace prefix keeps PRIVATE artifact lineage even though its
        access is public.
        """
        if self is BoundaryClass.INFRASTRUCTURE:
            return Exposure.INFRASTRUCTURE
        if self.requires_login:
            return Exposure.PRIVATE
        return Exposure.PUBLIC


@dataclass(frozen=True)
class EndpointBoundary:
    path: str
    boundary_class: BoundaryClass
    why: str = ""
    #: Overrides the class's coarse exposure where the two axes legitimately
    #: diverge — the evaluation surfaces under a private mount. `None` means
    #: "take the class's default", which is the common case.
    exposure_override: Optional[Exposure] = None

    @property
    def exposure(self) -> Exposure:
        return self.exposure_override or self.boundary_class.default_exposure

    @property
    def requires_login(self) -> bool:
        return self.boundary_class.requires_login

    @property
    def visibility(self) -> Visibility:
        return (Visibility.PRIVATE_WORKSPACE if self.exposure is Exposure.PRIVATE
                else Visibility.PUBLIC_LIBRARY)


#: Prefixes are matched longest-first, so a specific route overrides its parent.
MANIFEST: Sequence[EndpointBoundary] = (
    # --- the private workspace mounts -------------------------------------
    #
    # The umbrella entries. Everything under them requires a session unless a
    # longer, more specific entry below carves it out as public evaluation.
    EndpointBoundary("/workspace", BoundaryClass.AUTHENTICATED_PERSISTENCE,
                     "The scenario workspace. Everything here is one person's."),
    EndpointBoundary("/pilot", BoundaryClass.AUTHENTICATED_PERSISTENCE,
                     "The pilot mission workspace. One person's goal, one "
                     "person's plan, and a stored intent that names the "
                     "sentence it came from — private for the same reason the "
                     "scenario workspace is."),
    # --- the public evaluator ---------------------------------------------
    #
    # Reachable without an account. `/workspace/new` and `/pilot/answer` are
    # carved out of the private mounts above (this is what `PUBLIC_WITHIN_PRIVATE`
    # used to be); `/evaluate` is the canonical public name for the same flow.
    #
    # `/workspace/new` and `/pilot/answer` keep PRIVATE artifact lineage — they
    # are mounted under a private store — while their *access* is public. That
    # gap is the whole reason the two axes are declared separately.
    EndpointBoundary("/evaluate", BoundaryClass.PUBLIC_EVALUATION,
                     "The canonical public evaluator. A thin controller over "
                     "the existing pilot evaluation flow — describe a strategy, "
                     "clarify it, see what it does — with no login. Delegates to "
                     "the same entrypoints `/pilot` and `/pilot/answer` use; it "
                     "reimplements no parser, compiler or evaluator."),
    EndpointBoundary("/workspace/new", BoundaryClass.PUBLIC_EVALUATION,
                     "Start the evaluator. Public: trying a strategy needs no "
                     "account. Artifact lineage stays private because it is "
                     "mounted under the private workspace store.",
                     exposure_override=Exposure.PRIVATE),
    EndpointBoundary("/pilot/answer", BoundaryClass.PUBLIC_EVALUATION,
                     "Clarify and evaluate. Public: answering the interpreter's "
                     "questions is part of evaluation, not of keeping a plan. "
                     "Artifact lineage stays private (mounted under /pilot).",
                     exposure_override=Exposure.PRIVATE),
    EndpointBoundary("/evaluate/save", BoundaryClass.PUBLIC_EVALUATION,
                     "Begin saving an evaluated strategy — the public entry to "
                     "the one authentication boundary. It names an already-"
                     "evaluated, content-addressed review and either binds it "
                     "(signed in) or redirects to sign in (anonymous); it reads "
                     "no sentence and accepts no account state, so the click "
                     "that starts a save is still public evaluation.",
                     exposure_override=Exposure.PRIVATE),
    # --- authenticated persistence ----------------------------------------
    #
    # Saving and reading an owned plan. Declared explicitly so the finer class
    # is visible even though the umbrella `/pilot` already gates them.
    EndpointBoundary("/pilot/save", BoundaryClass.AUTHENTICATED_PERSISTENCE,
                     "Persist a plan to its owner. The first authentication "
                     "boundary: evaluation is public, keeping the result is not."),
    EndpointBoundary("/pilot/save/resume", BoundaryClass.AUTHENTICATED_PERSISTENCE,
                     "Finish a save after signing in — the `next` an anonymous "
                     "Save redirects through login. Behind the session gate so "
                     "it can only run for a now-authenticated visitor; it binds "
                     "the exact evaluated review to that owner and re-reads "
                     "nothing."),
    EndpointBoundary("/pilot/plans/{plan_id}", BoundaryClass.AUTHENTICATED_PERSISTENCE,
                     "Read a saved plan. Resolves to its owner, so it must be "
                     "behind the session gate."),
    EndpointBoundary("/workspace/plans/{plan_id}", BoundaryClass.AUTHENTICATED_PERSISTENCE,
                     "Read a saved scenario plan. One person's."),
    # --- private financial state ------------------------------------------
    #
    # The export a downstream runtime fetches. Account-specific; the most
    # private surface, kept a distinct class from persistence.
    EndpointBoundary("/pilot/plans/{plan_id}/runtime-artifact",
                     BoundaryClass.PRIVATE_FINANCIAL_STATE,
                     "Export a saved plan as a canonical runtime artifact for a "
                     "downstream runtime. Account-specific financial state; "
                     "reachable only by the owner."),
    # --- public research --------------------------------------------------
    EndpointBoundary("/for-advisors", BoundaryClass.PUBLIC_RESEARCH,
                     "The advisor narrative (§8). Informational only — it "
                     "manages no household, reads no account state and takes no "
                     "parameters. It explains that the same evaluated "
                     "SavedStrategyPlan becomes the input to Wealth Manager, and "
                     "labels each downstream stage by deployed status; "
                     "impersonal by construction, so no login."),
    EndpointBoundary("/ui", BoundaryClass.PUBLIC_RESEARCH,
                     "The research library. Impersonal by construction."),
    EndpointBoundary("/methodologies", BoundaryClass.PUBLIC_RESEARCH, "Published rules."),
    EndpointBoundary("/protocols", BoundaryClass.PUBLIC_RESEARCH, "Published evaluation protocols."),
    EndpointBoundary("/policies", BoundaryClass.PUBLIC_RESEARCH, "Published statistical policies."),
    EndpointBoundary("/runs", BoundaryClass.PUBLIC_RESEARCH, "Published run records."),
    EndpointBoundary("/performance", BoundaryClass.PUBLIC_RESEARCH, "Published figures."),
    EndpointBoundary("/errata", BoundaryClass.PUBLIC_RESEARCH, "Published corrections."),
    EndpointBoundary("/trials", BoundaryClass.PUBLIC_RESEARCH, "Trial counts for published lineages."),
    EndpointBoundary("/compatibility", BoundaryClass.PUBLIC_RESEARCH, "Comparability verdicts."),
    EndpointBoundary("/holdout", BoundaryClass.PUBLIC_RESEARCH, "Sealed-holdout unlock records."),
    EndpointBoundary("/holdout-unlocks", BoundaryClass.PUBLIC_RESEARCH,
                     "Every sealed-holdout opening, with reason and authorizer. "
                     "Impersonal: it concerns published methodologies."),
    EndpointBoundary("/surfaces", BoundaryClass.PUBLIC_RESEARCH,
                     "What each publication surface asserts. Describes the "
                     "policy, names no user."),
    EndpointBoundary("/current-strategies", BoundaryClass.PUBLIC_RESEARCH,
                     "Latest published version of each methodology."),
    EndpointBoundary("/project/discoveries", BoundaryClass.PUBLIC_RESEARCH,
                     "Changes detected in *published* artifacts. Must stay "
                     "impersonal: the moment a discovery is derived from one "
                     "user's plan it belongs to the workspace instead."),
    EndpointBoundary("/project/learning", BoundaryClass.PUBLIC_RESEARCH,
                     "What the platform has concluded about its own published "
                     "results."),
    # The nightly research dashboard, built on the host and served from the
    # mounted volume. PUBLIC_RESEARCH because it always was: it is the same
    # Bokeh document the platform published before the daily job was replaced,
    # it contains published research rather than anything a user wrote, and it
    # is the surface the library's figures are drawn from.
    EndpointBoundary("/research", BoundaryClass.PUBLIC_RESEARCH,
                     "The research dashboard, rebuilt daily from the run "
                     "history. Published output; carries nothing a user "
                     "entered and nothing from the private workspace."),
    # --- infrastructure ---------------------------------------------------
    EndpointBoundary("/info", BoundaryClass.INFRASTRUCTURE,
                     "Build and capability metadata. Carries no artifacts."),
    EndpointBoundary("/health", BoundaryClass.INFRASTRUCTURE, "Liveness."),
    # Signing in. INFRASTRUCTURE rather than a private class: these routes carry
    # no artifact of either kind, and they have to be reachable by somebody who
    # is not yet anybody — a login behind the private boundary is a door that
    # can only be opened from inside.
    #
    # What they do carry is a token, which is why the cookie is HttpOnly,
    # Secure and SameSite=Lax, and why the callback verifies before it writes.
    EndpointBoundary("/auth/login", BoundaryClass.INFRASTRUCTURE,
                     "Starts a sign-in. Redirects to the identity provider "
                     "with a PKCE challenge; carries no artifact."),
    EndpointBoundary("/auth/callback", BoundaryClass.INFRASTRUCTURE,
                     "Completes a sign-in. Verifies the token before any "
                     "session exists; carries no artifact."),
    EndpointBoundary("/auth/logout", BoundaryClass.INFRASTRUCTURE,
                     "Ends the session on this deployment. Clears the cookie "
                     "and carries no artifact."),
    EndpointBoundary("/health/live", BoundaryClass.INFRASTRUCTURE,
                     "Liveness. The process exists. Says nothing about whether "
                     "it can serve — a port answering is not readiness."),
    EndpointBoundary("/health/ready", BoundaryClass.INFRASTRUCTURE,
                     "Readiness. Whether the deployment preflight passed, and "
                     "nothing about why it did not."),
    EndpointBoundary("/ready", BoundaryClass.INFRASTRUCTURE,
                     "Readiness. Reports whether the startup preflight passed, "
                     "and nothing about why it did not."),
    EndpointBoundary("/docs", BoundaryClass.INFRASTRUCTURE, "Generated API docs."),
    EndpointBoundary("/redoc", BoundaryClass.INFRASTRUCTURE, "Generated API docs."),
    EndpointBoundary("/openapi.json", BoundaryClass.INFRASTRUCTURE, "Schema."),
    EndpointBoundary("/", BoundaryClass.INFRASTRUCTURE, "Root."),
)


class UndeclaredEndpoint(KeyError):
    """A route exists that nobody has assigned a side."""


def boundary_for(path: str) -> EndpointBoundary:
    """The declaration covering a path, longest prefix first.

    The root is matched exactly and never as a prefix. Treating `/` as a prefix
    would make it match every path in the application, which is precisely the
    catch-all default this manifest exists to prevent — the check would pass
    forever and mean nothing.
    """
    for entry in sorted(MANIFEST, key=lambda e: -len(e.path)):
        if path == entry.path:
            return entry
        stem = entry.path.rstrip("/")
        if stem and path.startswith(stem + "/"):
            return entry
    raise UndeclaredEndpoint(
        f"{path} has no declared boundary. Every endpoint must state whether it "
        "serves public research or a private workspace before it can ship — a "
        "route that defaults to whatever the framework defaults to is how the "
        "publisher position is lost quietly."
    )


def undeclared(paths: Sequence[str]) -> List[str]:
    out: List[str] = []
    for path in paths:
        try:
            boundary_for(path)
        except UndeclaredEndpoint:
            out.append(path)
    return out


def login_required(path: str) -> bool:
    """Whether the auth middleware must send an anonymous request to sign in.

    Derived from the manifest, so the middleware and the classification cannot
    disagree — the property §5 of the plan asks for. An undeclared path is not
    gated: it will 404 anyway, and the boundary sweep fails CI the moment a real
    route is added without a declaration, so "undeclared" never reaches
    production as a served route. This preserves the previous behaviour exactly,
    where a path matching no private prefix was passed straight through.
    """
    try:
        return boundary_for(path).requires_login
    except UndeclaredEndpoint:
        return False


def gated_prefixes() -> Sequence[str]:
    """The minimal set of login-requiring mount prefixes, derived from the
    manifest.

    This is what `src.api.PRIVATE_PREFIXES` is now built from, rather than a
    hand-written tuple that the middleware and the deploy acceptance check both
    trusted and that could drift from the classification. "Minimal" means the
    umbrellas only: a login-requiring entry that already sits under another one
    (`/pilot/save` under `/pilot`) is dropped, so the deploy acceptance script
    probes each mount once rather than probing `{plan_id}` template paths it
    cannot resolve.
    """
    login_paths = sorted({e.path for e in MANIFEST if e.requires_login}, key=len)
    minimal: List[str] = []
    for path in login_paths:
        if any(path.startswith(m.rstrip("/") + "/") for m in minimal):
            continue
        minimal.append(path)
    return tuple(sorted(minimal))


def public_within_gated() -> Sequence[str]:
    """The public routes that live under a gated mount, derived from the
    manifest.

    The successor to `PUBLIC_WITHIN_PRIVATE`: the evaluation carve-outs
    (`/workspace/new`, `/pilot/answer`) that are reachable without a session
    even though they sit under `/workspace` and `/pilot`. `/evaluate` is not
    here because it is not under a gated prefix — it is public at the root.
    """
    prefixes = gated_prefixes()
    out: List[str] = []
    for entry in MANIFEST:
        if entry.requires_login:
            continue
        if any(entry.path == p or entry.path.startswith(p.rstrip("/") + "/")
               for p in prefixes):
            out.append(entry.path)
    return tuple(sorted(out))
