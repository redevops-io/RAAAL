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
            request_id: Optional[str] = None, run_id: Optional[str] = None
            ) -> "MarketDataAccess":
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
    from .loader import load_prices, synthetic_snapshot
    from ..deploy.context import current
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
        frame = load_prices(snapshot)
    except Exception:
        return MarketDataAccess(None, not_recorded(
            f"snapshot {snapshot.snapshot_id} could not be loaded"))

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
        policy_version=policy.value, decision=decision, accessed_at=stamp)
    return MarketDataAccess(delivered, provenance, event)


def resolve_prices(*, context: str) -> Optional[Any]:
    """Prices only, for callers that do not persist the figure.

    Every caller that *stores* a number must use `resolve` and keep the
    provenance with it.
    """
    return resolve(context=context).frame


def approved_snapshot():
    """The vendor snapshot a non-synthetic policy would use.

    Deliberately unimplemented while the pilot is synthetic-only: there is no
    approved vendor snapshot until the six licensing questions are resolved and
    recorded, and returning one here would make `PILOT_VENDOR_APPROVED` mean
    something the licence does not yet permit.
    """
    return None
