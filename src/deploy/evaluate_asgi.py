"""`quantify-evaluate` as something a container can start.

    uvicorn src.deploy.evaluate_asgi:create_app --factory

**Why this is in `deploy` and not in `evaluation`.** It is the composition
root, and a composition root is the one place allowed to know about both sides
of a boundary. The evaluator needs an engine, and the engine's entry point —
`execute_compiled_plan` — is still in `workspace`, which is the leftover from
Step 7 that the decomposition named and did not finish. Putting this file in
`src/evaluation` would have made that import the evaluator's, and the boundary
test would have been right to fail it.

So the package stays clean and the wiring lives here. When the engine moves out
of `workspace`, this file gets shorter and nothing else changes — which is what
a boundary is for.

**It talks to quantify-data over HTTP and nowhere else.** No adapter, no symbol
table, no object store, no descriptor table. It is handed a URL and a snapshot
hash, and if the data service is unreachable it says so rather than reaching
for a file — there is no file to reach for, and no IAM role that would let it
read one if there were.
"""
from __future__ import annotations

from typing import Any


class NotDeployable(RuntimeError):
    """The service cannot start, and says which setting is missing."""


def _http(base: str):
    """A market-data client over real HTTP.

    `urllib` rather than a client library: this makes two request shapes, one
    JSON and one binary, and a dependency whose only job is to be more
    ergonomic than forty lines is a dependency in the image forever.
    """
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    from ..market_data.client import HttpMarketData

    def post(url, body):
        request = urllib.request.Request(
            url, method="POST" if body is not None else "GET",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"} if body else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as answer:
                return answer.status, json.loads(answer.read())
        except urllib.error.HTTPError as refused:
            try:
                return refused.code, json.loads(refused.read())
            except Exception:                                  # noqa: BLE001
                # A body that is not JSON carries no failure kind, and the
                # client turns that into "unreachable" rather than guessing
                # which refusal it was.
                return refused.code, None

    def fetch(url, params):
        full = url + ("?" + urllib.parse.urlencode(params) if params else "")
        try:
            with urllib.request.urlopen(full, timeout=120) as answer:
                return answer.status, answer.read(), dict(answer.headers)
        except urllib.error.HTTPError as refused:
            return refused.code, refused.read(), dict(refused.headers)

    return HttpMarketData(post=post, fetch=fetch, base=base.rstrip("/"))


def create_app() -> Any:
    """The evaluation service, pointed at quantify-data and nothing else."""
    from ..deploy.context import current
    from ..evaluation.server import create_app as build
    from ..workspace.run_boundary import execute_compiled_plan

    deployment = current()
    base = getattr(deployment, "market_data_service_url", "") or ""
    if not base:
        raise NotDeployable(
            "QUANTIFY_DATA_URL is not set. This service evaluates against "
            "snapshots it fetches by hash and has nowhere to fetch them from; "
            "starting without it would produce a service that is Ready and "
            "refuses every request that needs data")

    client = _http(base)

    def resolve_access(spec, snapshot_id):
        """A verified snapshot, as the delivery the engine understands.

        The descriptor hash is not in the request — the contract carries a
        snapshot hash — so it is looked up by content. That is the one place
        this composition needs to know both addresses, and it belongs here
        rather than in the contract: a request naming both would let a caller
        pair a descriptor with observations it does not describe.
        """
        from ..market_data.client import as_delivery

        descriptor_hash = _descriptor_for(client, snapshot_id)
        snapshot = client.get(snapshot_id, descriptor_hash)
        return as_delivery(snapshot, policy_version="", accessed_at="")

    return build(resolve_access=resolve_access, run_plan=execute_compiled_plan,
                 build=dict(getattr(deployment.build, "deployment", {}) or {}))


def _descriptor_for(client, snapshot_hash: str) -> str:
    """Which description of these observations to verify against.

    Asked of the data service rather than assumed. A snapshot can carry several
    descriptors — a licence re-reviewed, an adapter corrected — and the
    evaluator has no basis for choosing between them, so it takes the one the
    owner of the data offers.
    """
    status, payload = client.post(
        f"{client.base}/snapshots/{snapshot_hash}/descriptor", None)
    if status != 200 or not payload:
        from ..market_data.client import MarketDataUnreachable

        raise MarketDataUnreachable(
            f"the data service did not name a descriptor for {snapshot_hash} "
            f"(status {status}). No figure is produced: verifying observations "
            "against a description nobody supplied would be checking them "
            "against themselves")
    return payload["descriptor_hash"]
