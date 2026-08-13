"""Every endpoint declares which side of the boundary it serves.

An undeclared endpoint is the failure mode. A new route added next quarter will
be written by someone thinking about the feature, not about the publisher
position, and it will default to whatever the framework defaults to. So the
manifest is checked against the live application: a route that is not declared
here fails the test suite until someone decides which side it is on.

This is the same rule the artifact model applies everywhere else — a declaration
with no realization, or a realization with no declaration, is a defect — pointed
at the HTTP surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence

from ..mission.boundary import Visibility


class Exposure(str, Enum):
    PUBLIC = "PUBLIC"
    """Impersonal. May be read by anyone; may never contain private artifacts."""

    PRIVATE = "PRIVATE"
    """One person's. May cite public artifacts; never reachable from public ones."""

    INFRASTRUCTURE = "INFRASTRUCTURE"
    """Docs, schema, health. Carries no artifacts of either kind."""


@dataclass(frozen=True)
class EndpointBoundary:
    path: str
    exposure: Exposure
    why: str = ""

    @property
    def visibility(self) -> Visibility:
        return (Visibility.PRIVATE_WORKSPACE if self.exposure is Exposure.PRIVATE
                else Visibility.PUBLIC_LIBRARY)


#: Prefixes are matched longest-first, so a specific route overrides its parent.
MANIFEST: Sequence[EndpointBoundary] = (
    EndpointBoundary("/workspace", Exposure.PRIVATE,
                     "The scenario workspace. Everything here is one person's."),
    EndpointBoundary("/pilot", Exposure.PRIVATE,
                     "The pilot mission workspace. One person's goal, one "
                     "person's plan, and a stored intent that names the "
                     "sentence it came from — private for the same reason the "
                     "scenario workspace is."),
    EndpointBoundary("/ui", Exposure.PUBLIC,
                     "The research library. Impersonal by construction."),
    EndpointBoundary("/methodologies", Exposure.PUBLIC, "Published rules."),
    EndpointBoundary("/protocols", Exposure.PUBLIC, "Published evaluation protocols."),
    EndpointBoundary("/policies", Exposure.PUBLIC, "Published statistical policies."),
    EndpointBoundary("/runs", Exposure.PUBLIC, "Published run records."),
    EndpointBoundary("/performance", Exposure.PUBLIC, "Published figures."),
    EndpointBoundary("/errata", Exposure.PUBLIC, "Published corrections."),
    EndpointBoundary("/trials", Exposure.PUBLIC, "Trial counts for published lineages."),
    EndpointBoundary("/compatibility", Exposure.PUBLIC, "Comparability verdicts."),
    EndpointBoundary("/holdout", Exposure.PUBLIC, "Sealed-holdout unlock records."),
    EndpointBoundary("/holdout-unlocks", Exposure.PUBLIC,
                     "Every sealed-holdout opening, with reason and authorizer. "
                     "Impersonal: it concerns published methodologies."),
    EndpointBoundary("/surfaces", Exposure.PUBLIC,
                     "What each publication surface asserts. Describes the "
                     "policy, names no user."),
    EndpointBoundary("/current-strategies", Exposure.PUBLIC,
                     "Latest published version of each methodology."),
    EndpointBoundary("/project/discoveries", Exposure.PUBLIC,
                     "Changes detected in *published* artifacts. Must stay "
                     "impersonal: the moment a discovery is derived from one "
                     "user's plan it belongs to the workspace instead."),
    EndpointBoundary("/project/learning", Exposure.PUBLIC,
                     "What the platform has concluded about its own published "
                     "results."),
    EndpointBoundary("/info", Exposure.INFRASTRUCTURE,
                     "Build and capability metadata. Carries no artifacts."),
    EndpointBoundary("/health", Exposure.INFRASTRUCTURE, "Liveness."),
    # The nightly research dashboard, built on the host and served from the
    # mounted volume. PUBLIC because it always was: it is the same Bokeh
    # document the platform published before the daily job was replaced, it
    # contains published research rather than anything a user wrote, and it is
    # the surface the library's figures are drawn from.
    EndpointBoundary("/research", Exposure.PUBLIC,
                     "The research dashboard, rebuilt daily from the run "
                     "history. Published output; carries nothing a user "
                     "entered and nothing from the private workspace."),
    # Readiness is separate from liveness: a failed preflight makes an
    # instance unready without making it indistinguishable from a dead
    # process. It carries the outcome and nothing about why — a client learns
    # that the service cannot serve, not which host it could not reach or
    # which revision its schema is at.
    EndpointBoundary("/health/live", Exposure.INFRASTRUCTURE,
                     "Liveness. The process exists. Says nothing about whether "
                     "it can serve — a port answering is not readiness."),
    EndpointBoundary("/health/ready", Exposure.INFRASTRUCTURE,
                     "Readiness. Whether the deployment preflight passed, and "
                     "nothing about why it did not."),
    EndpointBoundary("/ready", Exposure.INFRASTRUCTURE,
                     "Readiness. Reports whether the startup preflight passed, "
                     "and nothing about why it did not."),
    EndpointBoundary("/docs", Exposure.INFRASTRUCTURE, "Generated API docs."),
    EndpointBoundary("/redoc", Exposure.INFRASTRUCTURE, "Generated API docs."),
    EndpointBoundary("/openapi.json", Exposure.INFRASTRUCTURE, "Schema."),
    EndpointBoundary("/", Exposure.INFRASTRUCTURE, "Root."),
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
