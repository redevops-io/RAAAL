"""Reaching the market-data service, and verifying what it says.

The client re-verifies. That is not distrust of the service — it is the same
rule the whole system runs on: a digest is only worth something if the party
relying on it checks it. The service verifies before sending so it cannot serve
a payload it knows is wrong; the consumer verifies on arrival so a transport,
a proxy or a cache cannot change what it relied on.

**Failures keep their kinds across the wire.** A caller that could not tell an
ambiguous symbol from a corrupted payload would retry one forever and go
looking for the other in the wrong place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

from .service_contract import MARKET_DATA_CONTRACT_VERSION, Failure, create_request


class MarketDataRefused(ValueError):
    """The service will not answer, and says which input it could not use."""

    def __init__(self, kind: Failure, detail: str) -> None:
        self.kind, self.detail = kind, detail
        super().__init__(f"{kind.value}: {detail}")


class MarketDataUnreachable(RuntimeError):
    """The service could not be reached. Not a statement about the request."""


@dataclass(frozen=True)
class Snapshot:
    """A verified snapshot as a consumer sees it: hashes and observations."""

    snapshot_hash: str
    descriptor_hash: str
    observations: Any
    descriptor: dict


@dataclass(frozen=True)
class HttpMarketData:
    """The market-data service, over HTTP."""

    post: Callable
    """`(url, json) -> (status, payload)`."""

    fetch: Callable
    """`(url, params) -> (status, body_bytes, headers)`. Separate from `post`
    because observations cross as bytes: putting them through a JSON decoder
    would place a transport's choices between the data and its digest."""

    base: str = ""

    def create(self, symbols: Sequence[str], *, reinvested: bool
               ) -> Tuple[str, str]:
        """Resolve, fetch and store. Returns the two addresses and nothing else."""
        status, payload = self.post(f"{self.base}/snapshots",
                                    create_request(symbols,
                                                   reinvested=reinvested))
        if status == 201:
            return payload["snapshot_hash"], payload["descriptor_hash"]
        raise self._refusal(status, payload)

    def get(self, snapshot_hash: str, descriptor_hash: str) -> Snapshot:
        """Fetch by hash, and verify before returning.

        Verified here as well as in the service. A consumer that trusted the
        sender's word would have no answer when the two disagree, and the
        digest exists precisely so it does not have to.
        """
        from .object_store import from_bytes
        from .snapshot_contract import from_json

        status, body, _headers = self.fetch(
            f"{self.base}/snapshots/{snapshot_hash}/observations",
            {"descriptor": descriptor_hash})
        if status != 200:
            raise self._refusal(status, _as_json(body))

        described, descriptor_body = self._descriptor(descriptor_hash)
        observations = from_bytes(body)

        problems = described.verify(observations)
        if problems:
            raise MarketDataRefused(
                Failure.PAYLOAD_CORRUPT,
                "the observations that arrived are not the ones this snapshot "
                "names: " + "; ".join(problems))
        return Snapshot(snapshot_hash=snapshot_hash,
                        descriptor_hash=descriptor_hash,
                        observations=observations, descriptor=descriptor_body)

    def _descriptor(self, descriptor_hash: str):
        from .snapshot_contract import from_json

        status, payload = self.post(
            f"{self.base}/descriptors/{descriptor_hash}", None)
        if status != 200:
            raise self._refusal(status, payload)
        body = payload["descriptor"]
        return from_json(body), body

    @staticmethod
    def _refusal(status: int, payload) -> Exception:
        if not isinstance(payload, dict) or "kind" not in payload:
            return MarketDataUnreachable(
                f"the market-data service answered {status} with no failure "
                "kind. No snapshot is produced: a request assembled from an "
                "unclassified error would be guessing what went wrong")
        return MarketDataRefused(Failure(payload["kind"]),
                                 payload.get("detail", ""))


def _as_json(body):
    import json

    try:
        return json.loads(body)
    except Exception:                                          # noqa: BLE001
        return None


def as_delivery(snapshot: Snapshot, *, policy_version: str, accessed_at: str):
    """A verified snapshot, as the delivery the engine already understands.

    The engine takes a `MarketDataAccess` — a frame with the record of where it
    came from — and a verified snapshot carries exactly those two facts under
    different names. Building it here rather than in the evaluator is the point:
    this module owns the descriptor, so the provenance is *read* from what the
    service said rather than reconstructed by whoever happens to be holding the
    data. A caller assembling provenance afterwards is a caller that can get it
    wrong, and the run it got wrong on looks exactly like one it did not.

    Nothing is invented. Every field comes from the descriptor, and the ones the
    descriptor does not carry stay absent rather than being filled with a
    plausible value.
    """
    from .access import MarketDataAccess
    from .access_event import (ACCESS_EVENT_VERSION, MarketDataAccessEvent,
                               ResolutionRequest, TimeRange)
    from .provenance import AccessDecision, MarketDataProvenance, ProvenanceStatus

    body = snapshot.descriptor
    span = body["session_range"]
    # The snapshot's identity is its hash, not the adapter tag the descriptor
    # carries under `snapshot_id`. That field names the *source dataset* —
    # `local-parquet@1` — and using it here made `evaluate` refuse, comparing
    # the hash it was asked for against a provider name. Content addressing is
    # the point of the whole path, so the identity that travels is the address.
    # The source tag is not lost; it stays in the descriptor, which is where a
    # question about provenance is answered.
    provenance = MarketDataProvenance(
        status=ProvenanceStatus.RECORDED,
        snapshot_id=snapshot.snapshot_hash,
        content_digest=snapshot.snapshot_hash,
        content_digest_version=body["content_digest_version"],
        license_class=body["license_class"],
        license_review_status=body["license_review_status"],
        policy_version=policy_version,
        access_decision=AccessDecision.SYNTHETIC_ALLOWED,
        accessed_at=accessed_at)
    event = MarketDataAccessEvent(
        access_event_id=f"mdae-{snapshot.descriptor_hash[5:21]}",
        request_id=f"snapshot:{snapshot.snapshot_hash}",
        run_id=None,
        snapshot_id=snapshot.snapshot_hash,
        provenance_digest="",
        frame_digest=snapshot.snapshot_hash,
        selected_columns=tuple(body["symbols"]),
        row_count=int(span["sessions"]),
        time_range=TimeRange(span["start"], span["end"]),
        policy_version=policy_version,
        access_decision=AccessDecision.SYNTHETIC_ALLOWED,
        accessed_at=accessed_at,
        version=ACCESS_EVENT_VERSION,
        resolution=ResolutionRequest(
            reinvested=bool(body["resolution"].get("reinvested"))))
    return MarketDataAccess(snapshot.observations, provenance, event)
