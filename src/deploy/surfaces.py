"""What this repository can serve, and which of it a deployment actually runs.

`bokeh_app.py` was classified "the standalone Bokeh demo, served separately and
not part of the pilot surface". That was written from the name and was wrong:
`scripts/service.py` imports it, and `scripts/service.py` is the Dockerfile's
`CMD`. It is what the container runs.

The same look also found the reverse. **Nothing serves `src/api.py`.** There is
no `uvicorn` invocation in the Dockerfile, in `scripts/`, or anywhere else — so
the pilot application, its routers, and the entire Gate 2 startup preflight have
no deployment entrypoint. A preflight nothing starts is a control nothing
reaches, which is the invariant this codebase keeps re-learning, applied to the
deployment rather than to a function.

So surfaces are declared here as data, with their entrypoint named, and
`tests/test_surfaces.py` checks each declaration against the deployment
artifacts rather than against its own description.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple


class DataPolicy(str, Enum):
    GATED = "GATED"
    """Reaches market data only through `market_data.access.resolve_prices`."""

    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    """Reads files directly, and is restricted to synthetic data."""

    NONE = "NONE"
    """Reads no market data."""


@dataclass(frozen=True)
class Surface:
    """One thing this repository can serve."""

    name: str
    module: str
    entrypoint: Optional[str]
    """What starts it, or None if nothing does."""

    production_reachable: bool
    data_policy: DataPolicy
    reason: str
    dockerfile: Optional[str] = None
    requires_preflight: bool = False
    """Whether its entrypoint must run the deployment preflight before
    accepting traffic. True for every production surface."""

    def to_json(self) -> dict:
        return {"surface": self.name, "module": self.module,
                "entrypoint": self.entrypoint, "dockerfile": self.dockerfile,
                "production_reachable": self.production_reachable,
                "requires_preflight": self.requires_preflight,
                "data_policy": self.data_policy.value, "reason": self.reason}


#: What each image's `CMD` runs. Read by the tests that check these
#: declarations against the Dockerfiles rather than trusting them.
PRODUCTION_ENTRYPOINT = "src.api:create_app"
DASHBOARD_ENTRYPOINT = "scripts/service.py"

PRODUCTION_DOCKERFILE = "Dockerfile"
DASHBOARD_DOCKERFILE = "Dockerfile.dashboard"

SURFACES: Tuple[Surface, ...] = (
    Surface(
        name="pilot-api", module="src/api.py",
        entrypoint=PRODUCTION_ENTRYPOINT,
        dockerfile=PRODUCTION_DOCKERFILE,
        production_reachable=True,
        requires_preflight=True,
        data_policy=DataPolicy.GATED,
        reason="The product. Served by the production image through "
               "`create_app`, which runs the deployment preflight while the "
               "process is starting — so a refusal prevents the server binding "
               "rather than being noticed after the socket is open. Nothing "
               "served this at all until Gate 3."),
    Surface(
        name="regime-dashboard", module="src/visualization/bokeh_app.py",
        entrypoint=DASHBOARD_ENTRYPOINT,
        dockerfile=DASHBOARD_DOCKERFILE,
        production_reachable=False,
        requires_preflight=False,
        data_policy=DataPolicy.SYNTHETIC_ONLY,
        reason="A development and demo surface. It was the production "
               "container until Gate 3, which meant the deployed product was a "
               "dashboard while the pilot went unserved. It reads history "
               "parquet files directly, which is tolerable for a demo and is "
               "exactly why it must not be production-reachable."),
    Surface(
        name="mission-cli", module="scripts/mission.py",
        entrypoint="scripts/mission.py",
        production_reachable=False, requires_preflight=False,
        data_policy=DataPolicy.NONE,
        reason="An operator command line, run deliberately. It is not served "
               "and accepts no request from a user."),
)


def by_module() -> Mapping[str, Surface]:
    return {one.module: one for one in SURFACES}


def production_surfaces() -> Sequence[Surface]:
    return tuple(one for one in SURFACES if one.production_reachable)


def unserved_surfaces() -> Sequence[Surface]:
    """Surfaces with no entrypoint. Currently the pilot application itself."""
    return tuple(one for one in SURFACES if one.entrypoint is None)
