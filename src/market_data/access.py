"""The one way a production request may obtain market data.

    configured policy -> approved snapshot -> authorise -> load by identity

`src/workspace/routes.py` used to read `data/history/prices.parquet` directly:
an unmanifested file with no snapshot identity, no licence class and no egress
check. That was fixed there, and the identical read stayed in
`src/web/routes.py` — the *public* router — because the fix was applied to the
consumer that was found rather than to the class of consumer.

So there is now one function, and the inventory in `tests/test_data_access.py`
requires every production reader to be it. A second copy of the gate is not
safer than no gate; it is a gate that will be updated in one place.

**There is no fallback.** A denied or unresolvable snapshot yields no prices,
never the synthetic one. A figure drawn from data the plan did not name is worse
than no figure: it renders, it looks ordinary, and nothing about it says which
data produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .access_event import MarketDataAccessEvent
from .provenance import MarketDataProvenance, ProvenanceStatus

#: Every file a production request is permitted to obtain market data from is
#: reached through `load_prices`, keyed by snapshot identity. This path is not
#: one of them, and is named here so a test can assert nothing reads it.
UNMANIFESTED_PRICES = "data/history/prices.parquet"


@dataclass(frozen=True)
class MarketDataAccess:
    """A frame and the record of where it came from, as one value.

    Returned together, and carried together. Passing `frame` and `provenance`
    as separate arguments through several functions is how one of them gets
    dropped — which is exactly what the live run path did: it used the frame
    and discarded the provenance, so every stored figure was unattributable
    while the mechanism to attribute it already existed.
    """

    frame: Optional[Any]
    provenance: "MarketDataProvenance"

    access_event: Optional["MarketDataAccessEvent"] = None
    """The factual record of *this* delivery, digested from the frame beside
    it. `None` only where nothing was delivered — a denial, a missing snapshot,
    an unloadable file — because an event describing no frame would be a claim
    with nothing behind it. Provenance says which source was authorised; this
    says what arrived."""

    @property
    def usable(self) -> bool:
        return self.frame is not None

    @property
    def access_event_id(self) -> Optional[str]:
        return self.access_event.access_event_id if self.access_event else None

    def matches_delivery(self, frame: Any) -> bool:
        """Whether `frame` is byte-for-byte the canonical content delivered.

        How execution proves it simulated what it was given rather than
        something it later became. A digest recomputed by the caller over the
        caller's own frame would answer a different question.
        """
        from .access_event import frame_digest

        if self.access_event is None:
            return False
        return frame_digest(frame) == self.access_event.frame_digest

    def __iter__(self):
        """Unpacks as a pair, for callers that want only one half.

        Deliberately still a pair. Widening it to three would silently give
        every existing `frame, provenance = resolve(...)` a third value or an
        unpacking error, and the event has a named accessor precisely so it is
        reached on purpose.
        """
        return iter((self.frame, self.provenance))


def resolve(*, context: str, accessed_at: Optional[str] = None,
            request_id: Optional[str] = None, run_id: Optional[str] = None,
            reinvested: bool = False) -> "MarketDataAccess":
    """Prices, their provenance and the record of this delivery, together.

    Returned as one object so a caller cannot obtain a figure without the
    record of where it came from. A separate "and also fetch the provenance"
    call is one a producer can forget, and the figure it forgot on looks
    exactly like one it did not.

    `run_id` is accepted so the execution that will consume this frame can be
    named *before* it runs. Binding afterwards would need a second write to
    connect them, and a chain with a second write has a state where it is half
    connected — which is the state a crash finds.

    The frame digest is computed here, over the frame about to be returned.
    A caller computing it later would digest whatever the caller was holding,
    and whether that is still what was delivered is the whole question.
    """
    import datetime as dt
    import uuid

    from .provenance import (
        AccessDecision,
        not_recorded,
        recorded,
    )
    from .loader import synthetic_snapshot
    from ..deploy.context import current
    from .access_event import ResolutionRequest
    from .access_event import build as build_event
    from .pilot_policy import PilotDataDenied, PilotDataPolicy, authorise

    stamp = accessed_at or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z")

    # The policy the deployment resolved, not one this gate looks up for
    # itself. Fails closed unchanged: an absent policy still denies, and it
    # now denies for the reason the resolver recorded rather than for the
    # reason this call site guessed.
    policy = current().market_data.policy
    if policy is None:
        return MarketDataAccess(
            None, not_recorded("no market-data policy is configured"))

    if policy is PilotDataPolicy.SYNTHETIC_ONLY:
        snapshot = synthetic_snapshot()
        decision = AccessDecision.SYNTHETIC_ALLOWED
    else:
        snapshot = approved_snapshot()
        decision = AccessDecision.PILOT_VENDOR_APPROVED
    if snapshot is None:
        return MarketDataAccess(
            None, not_recorded("no snapshot was resolved for this policy"))

    try:
        authorise(snapshot, policy=policy, context=context)
    except PilotDataDenied as refusal:
        # No data, and a provenance that records the refusal rather than
        # pretending the question was never asked.
        return MarketDataAccess(None, MarketDataProvenance(
            status=ProvenanceStatus.RECORDED, snapshot_id=snapshot.snapshot_id,
            content_digest=getattr(snapshot, "content_digest", None),
            content_digest_version=getattr(snapshot, "content_digest_version",
                                           None),
            license_class=getattr(snapshot, "license_class", None),
            license_review_status=getattr(snapshot, "license_review_status",
                                          None),
            policy_version=policy.value,
            access_decision=AccessDecision.DENIED,
            access_decision_reason=str(refusal)[:200], accessed_at=stamp))

    try:
        frame = _frame_for(snapshot, reinvested=reinvested)
    except Exception as unavailable:
        # The reason, not a generic failure. A snapshot with no total-return
        # twin cannot answer a plan that reinvests dividends, and "could not be
        # loaded" would send whoever reads it looking for a missing file rather
        # than at the series the plan asked for.
        #
        # A delegated fetch fails into this same except: the data service being
        # unreachable, refusing the request, or returning bytes that do not
        # match the snapshot's digest all raise, and all produce the same "could
        # not be loaded" provenance — because from the caller's point of view a
        # frame that did not arrive is a frame that did not arrive, however far
        # away the S3 credentials that would have produced it happen to live.
        return MarketDataAccess(None, not_recorded(
            f"snapshot {snapshot.snapshot_id} could not be loaded"
            f"{' with dividends reinvested' if reinvested else ''}: "
            f"{unavailable}"))

    # Sorted first, then digested, then described — in that order, over one
    # object. The frame handed to the caller and the frame the digest describes
    # are the same value, not two values that ought to be equal.
    delivered = frame.sort_index()
    provenance = recorded(snapshot, policy_version=policy.value,
                          decision=decision, accessed_at=stamp, reason=context)
    event = build_event(
        access_event_id=f"mdae-{uuid.uuid4().hex}",
        request_id=request_id or f"unattributed-{uuid.uuid4().hex[:12]}",
        run_id=run_id, frame=delivered, provenance=provenance,
        policy_version=policy.value, decision=decision, accessed_at=stamp,
        # The request, recorded beside the digest it produced. The snapshot
        # says which observations exist; this says which of them were
        # delivered, and only the pair determines the frame — asking the same
        # snapshot with `reinvested` flipped yields a different digest. An
        # event naming only the snapshot cannot be recomputed, and a verifier
        # that tried would compare against a frame nobody was given.
        resolution=ResolutionRequest(reinvested=bool(reinvested)))
    return MarketDataAccess(delivered, provenance, event)


def _frame_for(snapshot, *, reinvested: bool):
    """The price frame, from wherever this pod is allowed to get it.

    Two deployments, one call site. On the data service itself
    (`market_data_service_url` empty) the frame is loaded locally from the
    pinned object — that pod holds the S3 credentials and the vendor URI, and
    `load_prices` is the whole of the local gate. On every other pod
    (`market_data_service_url` set) there are no credentials to load with, so the
    frame is fetched from quantify-data over HTTP; a local `load_prices` there
    fails closed with "has no URI", which is the defect this delegation exists to
    remove.

    Provenance is unchanged either way. `resolve()` reads it from the manifest —
    `approved_snapshot()`/`synthetic_snapshot()` — not from whatever produced the
    bytes, so which pod fetched the frame is not a fact any figure records.

    The delegated price frame is re-verified against the snapshot's content
    digest, exactly as `load_prices` verifies it locally. The client already
    guards the transport (a truncated or altered body is a `PAYLOAD_CORRUPT`
    refusal); this adds the same *data* check the local path has, so a frame that
    crossed the wire is held to the identity the plan named rather than trusted
    for having arrived. The reinvested twin is deliberately not digest-checked,
    mirroring the local twin path in `load_prices`: that digest identifies the
    price series and would fail on the twin by design.
    """
    from ..deploy.context import current
    from .loader import load_prices

    url = current().market_data_service_url
    if not url:
        return load_prices(snapshot, reinvested=reinvested)

    frame = _market_data_client(url).prices(reinvested=reinvested)
    if not reinvested:
        from .integrity import verify

        verify(frame,
               expected_content_digest=getattr(snapshot, "content_digest", None),
               source=f"the market-data service for {snapshot.snapshot_id}")
    return frame


def _market_data_client(base: str):
    """A market-data client over real HTTP, built the way evaluate builds its own.

    The transport is mirrored from `src/deploy/evaluate_asgi.py::_http` — the
    composition root that already talks to quantify-data — rather than imported
    from it. `access` is reached from every consumer of market data; importing a
    deploy composition root here would point the data path at the wiring instead
    of leaving the wiring pointed at the data path. `urllib` rather than a client
    library for the same reason it gives there: this makes two request shapes,
    one JSON and one binary, and a dependency whose only job is ergonomics is one
    that lives in the image forever.
    """
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    from .client import HttpMarketData

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
                # A body that is not JSON carries no failure kind, and the client
                # turns that into "unreachable" rather than guessing which
                # refusal it was.
                return refused.code, None

    def fetch(url, params):
        full = url + ("?" + urllib.parse.urlencode(params) if params else "")
        try:
            with urllib.request.urlopen(full, timeout=120) as answer:
                return answer.status, answer.read(), dict(answer.headers)
        except urllib.error.HTTPError as refused:
            return refused.code, refused.read(), dict(refused.headers)

    return HttpMarketData(post=post, fetch=fetch, base=base.rstrip("/"))


def resolve_prices(*, context: str) -> Optional[Any]:
    """Prices only, for callers that do not persist the figure.

    Every caller that *stores* a number must use `resolve` and keep the
    provenance with it.
    """
    return resolve(context=context).frame


#: The vendor snapshot this deployment may use, and the record that authorises
#: it. Both named here so the pair is reviewable in one place.
VENDOR_MANIFEST = "data/snapshots/prices-yahoo-20260822.yaml"


def approved_snapshot():
    """The vendor snapshot a non-synthetic policy may use, or None.

    Was deliberately unimplemented while the pilot was synthetic-only. It is
    implemented now because the six licensing questions carry recorded
    answers — but it **re-checks that record** rather than trusting that a
    manifest exists.

    That distinction is the point. A manifest is a file anyone can add; the
    gate is whether every question has an answer, a source, a reviewer, a date,
    an audience, a permitted derived output, an export rule and a retention
    rule. Returning a snapshot on the strength of the file alone would make the
    check decorative, which is how a control becomes a comment.

    Returns None on any failure — missing manifest, missing record, incomplete
    answers. The caller then records "no snapshot was resolved for this policy"
    and serves nothing, which is the correct direction to fail in.
    """
    import yaml

    from .loader import REPO_ROOT, load_manifest
    from .pilot_policy import licensing_resolved

    manifest = REPO_ROOT / VENDOR_MANIFEST
    if not manifest.exists():
        return None
    try:
        snapshot = load_manifest(manifest)
    except Exception:                                       # pragma: no cover
        return None

    record_path = (snapshot.raw or {}).get("license_record")
    if not record_path:
        return None
    record_file = REPO_ROOT / str(record_path)
    if not record_file.exists():
        return None
    try:
        answers = (yaml.safe_load(record_file.read_text()) or {}).get("answers")
    except Exception:                                       # pragma: no cover
        return None
    if not answers or not licensing_resolved(answers):
        return None
    return snapshot
