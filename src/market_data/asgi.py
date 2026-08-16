"""`quantify-data` as something a container can start.

    uvicorn src.market_data.asgi:create_app --factory

The service itself takes its adapter and object store as arguments, because
which provider a deployment reads and where its bytes live are deployment
questions. This is the one place that answers them — from the resolved
deployment context, which is the only module permitted to read the
environment.

**One image, three commands.** `quantify-web`, `quantify-evaluate` and
`quantify-data` currently ship as the same image with different entrypoints.
That means the evaluator's image *contains* market-data code it may not use,
and the guards against using it are the import graph and, once deployed, IAM —
not the absence of the file. Separate images would be stricter and are a later
step; conflating them with the cluster migration would mean debugging three
builds and a new control plane at once.

**It refuses to start without a snapshot root.** A service that defaulted to a
temporary directory would come up healthy, serve, and lose every payload it
wrote on the next restart — leaving descriptors in PostgreSQL with nothing
behind them. That state is already named `PAYLOAD_MISSING`, and it should
arrive from an interrupted write rather than from a service that was never told
where to put anything.
"""
from __future__ import annotations

from typing import Any


class NotDeployable(RuntimeError):
    """The service cannot start, and says which setting is missing."""


def create_app() -> Any:
    """The market-data service, configured from the deployment."""
    from pathlib import Path

    from ..deploy.context import current
    from .adapters import LocalParquetAdapter
    from .object_store import ObjectStore
    from .server import create_app as build

    deployment = current()

    root = getattr(deployment.market_data, "snapshot_root", None)
    if not root:
        raise NotDeployable(
            "QUANTIFY_SNAPSHOT_ROOT is not set. This service stores immutable "
            "observation payloads and has nowhere to put them; defaulting to a "
            "temporary directory would serve happily and lose every payload on "
            "restart, leaving descriptors recorded with nothing behind them")

    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)

    return build(
        adapter=LocalParquetAdapter(),
        store=ObjectStore(root=path),
        # The manifest already maps `QUANTIFY_BUILD_COMMIT` to `build_commit`.
        # Read rather than re-derived, so this service and the application
        # cannot disagree about which revision they are.
        build=dict(getattr(deployment.build, "deployment", {}) or {}))
