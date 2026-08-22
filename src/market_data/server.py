"""The market-data service. It owns the whole chain; nothing else may.

    ResolutionRequest -> resolve instrument -> adapter -> MarketSnapshot
                      -> immutable storage -> hash

Everything a caller gets back is a hash. No symbols leave, no provider names, no
URIs, no adapter internals — because anything that crossed would be a way to
reach the data without the hash, and the whole architecture rests on there being
no such way.

**Transport only.** Resolution, fetching, describing, storing and verifying all
exist already and are unchanged. What is here is routing and status codes.

**Failures keep their kinds.** Four of them arrive as 422 and they are four
different repairs; a caller reading only the status would retry an ambiguous
symbol forever and go looking for a snapshot that is sitting right there.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .service_contract import (MARKET_DATA_CONTRACT_VERSION, STATUS, Failure,
                               create_response, failure)
from .snapshot_read import SnapshotProblem


def build_identity(build: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """What this service is. The build is given, never read from the process.

    One module resolves the deployment; a service reading `os.environ` for its
    own identity would be a second, and that is how two halves of one
    deployment come to disagree about which build they are.
    """
    from .snapshot_contract import SNAPSHOT_CONTRACT_VERSION

    supplied = dict(build or {})
    return {"service": "quantify-market-data",
            "contract_version": MARKET_DATA_CONTRACT_VERSION,
            "snapshot_contract_version": SNAPSHOT_CONTRACT_VERSION,
            "build_commit": supplied.get("build_commit", ""),
            "image_digest": supplied.get("image_digest", ""),
            "adapters": supplied.get("adapters", [])}


#: How a read-path problem becomes a wire failure. One mapping, so a new
#: problem kind cannot quietly arrive as a generic error.
_READ_FAILURES = {
    SnapshotProblem.DESCRIPTOR_MISSING: Failure.DESCRIPTOR_MISSING,
    SnapshotProblem.DESCRIPTOR_MISMATCH: Failure.DESCRIPTOR_MISMATCH,
    SnapshotProblem.PAYLOAD_MISSING: Failure.PAYLOAD_MISSING,
    SnapshotProblem.EMPTY_OBSERVATIONS: Failure.EMPTY_OBSERVATIONS,
    SnapshotProblem.PAYLOAD_DIGEST_MISMATCH: Failure.PAYLOAD_CORRUPT,
}


def create_app(*, adapter, store, build=None):
    """The service, with its adapter and object store injected.

    Both given rather than constructed: which provider this deployment reads
    and where its bytes live are deployment questions, and a service that chose
    either would be deciding something the contract says it is handed.
    """
    from .adapters import AdapterRefused, snapshot_from
    from .object_store import to_bytes
    from .snapshot_read import get as read_snapshot
    from .snapshot_store import descriptor as read_descriptor
    from .snapshot_store import latest_descriptor  # noqa: F401
    from .snapshot_store import record

    app = FastAPI(title="Quantify market data",
                  version=MARKET_DATA_CONTRACT_VERSION)

    def _refuse(kind: Failure, detail: str, **extra) -> JSONResponse:
        return JSONResponse(status_code=STATUS[kind],
                            content=failure(kind, detail, **extra))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/version")
    def version():
        return build_identity(build)

    @app.post("/snapshots")
    async def create(request: Request):
        """Resolve, fetch, describe, store — and return only hashes."""
        payload = await request.json()
        if payload.get("contract_version") != MARKET_DATA_CONTRACT_VERSION:
            return _refuse(
                Failure.CONTRACT_MISMATCH,
                f"this service implements {MARKET_DATA_CONTRACT_VERSION} and "
                f"was sent {payload.get('contract_version')!r}")

        symbols = list(payload.get("symbols") or ())
        reinvested = bool(payload.get("reinvested"))

        try:
            fetched = adapter.fetch(symbols, reinvested=reinvested)
        except AdapterRefused as refused:
            # The adapter's own words, classified. It already refuses for
            # exactly these reasons; what this adds is a kind, so a caller can
            # tell an ambiguous name from an untradable one without parsing
            # prose.
            detail = str(refused)
            if "not resolved" in detail and "names a family" in detail:
                kind = Failure.AMBIGUOUS_SYMBOL
            elif "not resolved" in detail:
                kind = Failure.UNKNOWN_SYMBOL
            elif "index" in detail or "indices" in detail:
                kind = Failure.NOT_TRADABLE
            else:
                kind = Failure.NO_COVERAGE
            return _refuse(kind, detail)

        snapshot = snapshot_from(fetched, reinvested=reinvested)
        record(snapshot, recorded_at=payload.get("at", ""))
        store.put(snapshot.snapshot_hash, to_bytes(fetched.observations))
        return JSONResponse(status_code=201, content=create_response(snapshot))

    @app.get("/descriptors/{descriptor_hash}")
    def describe(descriptor_hash: str):
        body = read_descriptor(descriptor_hash)
        if body is None:
            return _refuse(Failure.DESCRIPTOR_MISSING,
                           f"no descriptor {descriptor_hash} is recorded")
        return {"contract_version": MARKET_DATA_CONTRACT_VERSION,
                "descriptor": body}

    @app.post("/snapshots/{snapshot_hash}/descriptor")
    @app.get("/snapshots/{snapshot_hash}/descriptor")
    def current_descriptor(snapshot_hash: str):
        """Which description to verify these observations against.

        The evaluator holds a snapshot hash and needs a descriptor hash to
        check the bytes against. It cannot choose one — several may exist and
        it has no basis for preferring any — so the service that owns the data
        says which is current.
        """
        from .snapshot_store import latest_descriptor

        found = latest_descriptor(snapshot_hash)
        if found is None:
            return _refuse(
                Failure.DESCRIPTOR_MISSING,
                f"no descriptor is recorded for observations {snapshot_hash}, "
                "so there is nothing to verify them against")
        return {"contract_version": MARKET_DATA_CONTRACT_VERSION,
                "snapshot_hash": snapshot_hash, "descriptor_hash": found}

    @app.get("/prices")
    def prices(reinvested: int = 0):
        """The whole current snapshot's price frame, as parquet bytes.

        The one place the price frame crosses to a pod that has no S3
        credentials of its own. `resolve()` runs on every consumer — web
        included — but only this service holds the vendor URI and the IAM role
        that reads it, so a consumer asks here rather than loading locally and
        failing closed with `SnapshotUnavailable: ... has no URI`.

        The snapshot is resolved from *this service's own* policy, exactly as
        `resolve()` resolves it: SYNTHETIC_ONLY yields the committed fixture, the
        approved vendor policy yields the reviewed vendor snapshot. That match is
        the point — the frame that crosses must be the frame the consumer's
        provenance already names, and choosing a snapshot here by any other route
        would serve bytes under an identity nobody recorded.

        Provenance does not travel with the bytes. The consumer reads it from the
        manifest, and the digest headers below are how it checks that the bytes
        it received are the ones that identity names.
        """
        from ..deploy.context import current
        from .access import approved_snapshot
        from .loader import load_prices, synthetic_snapshot
        from .pilot_policy import PilotDataPolicy

        # The policy the deployment resolved, read the same way `resolve()`
        # reads it. An absent policy denies here for the same reason it denies
        # there: a snapshot resolved without one is a snapshot nobody authorised.
        policy = current().market_data.policy
        if policy is None:
            return _refuse(
                Failure.NO_COVERAGE,
                "no market-data policy is configured, so there is no snapshot "
                "to serve prices from")

        if policy is PilotDataPolicy.SYNTHETIC_ONLY:
            snapshot = synthetic_snapshot()
        else:
            snapshot = approved_snapshot()
        if snapshot is None:
            return _refuse(
                Failure.NO_COVERAGE,
                "no snapshot was resolved for this policy, so there are no "
                "prices to serve")

        want_reinvested = bool(reinvested)
        try:
            frame = load_prices(snapshot, reinvested=want_reinvested)
        except Exception as unavailable:                       # noqa: BLE001
            # The same direction `resolve()` fails in: no frame rather than a
            # substitute. A snapshot with no total-return twin cannot answer a
            # reinvested request, and PAYLOAD_MISSING carries that across the
            # wire as a refusal the consumer can classify — rather than a 500 it
            # can only guess at, which is what an unhandled load failure becomes.
            return _refuse(
                Failure.PAYLOAD_MISSING,
                f"snapshot {snapshot.snapshot_id} could not be loaded"
                f"{' with dividends reinvested' if want_reinvested else ''}: "
                f"{unavailable}")

        headers = {"x-snapshot-id": str(snapshot.snapshot_id)}
        # The content digest identifies the *price* series only. The reinvested
        # twin is a different series with a different digest, so it travels
        # without this header — exactly as the local twin path skips the digest
        # check — and the consumer verifies the price frame while leaving the
        # twin to the client's transport-level guard.
        if not want_reinvested and snapshot.content_digest:
            headers["x-content-digest"] = str(snapshot.content_digest)

        return Response(content=to_bytes(frame),
                        media_type="application/octet-stream",
                        headers=headers)

    @app.get("/snapshots/{snapshot_hash}/observations")
    def observations(snapshot_hash: str, descriptor: str):
        """Bytes, verified before they leave.

        The service does not hand out a payload it has not checked against its
        descriptor. A caller could verify too — and the point is that it does
        not have to trust a store that skipped it.
        """
        read = read_snapshot(snapshot_hash, descriptor, store=store)
        if not read.ok:
            kind, why = read.problems[0]
            return _refuse(_READ_FAILURES[kind], why)

        return Response(
            content=store.get(snapshot_hash),
            media_type="application/octet-stream",
            headers={"x-snapshot-hash": snapshot_hash,
                     "x-descriptor-hash": descriptor})

    return app
