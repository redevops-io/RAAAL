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

from typing import Any, Optional, Tuple

from .provenance import MarketDataProvenance, ProvenanceStatus

#: Every file a production request is permitted to obtain market data from is
#: reached through `load_prices`, keyed by snapshot identity. This path is not
#: one of them, and is named here so a test can assert nothing reads it.
UNMANIFESTED_PRICES = "data/history/prices.parquet"


def resolve(*, context: str, accessed_at: Optional[str] = None):
    """Prices and the provenance of the data behind them, together.

    Returned as a pair so a caller cannot obtain a figure without the record of
    where it came from. A separate "and also fetch the provenance" call is one
    a producer can forget, and the figure it forgot on looks exactly like one
    it did not.
    """
    import datetime as dt

    from .provenance import (
        AccessDecision,
        not_recorded,
        recorded,
    )
    from .loader import load_prices, synthetic_snapshot
    from .pilot_policy import (
        PilotDataDenied,
        PilotDataPolicy,
        PilotPolicyMissing,
        authorise,
        configured_policy,
    )

    stamp = accessed_at or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z")

    try:
        policy = configured_policy()
    except PilotPolicyMissing:
        return None, not_recorded("no market-data policy is configured")

    if policy is PilotDataPolicy.SYNTHETIC_ONLY:
        snapshot = synthetic_snapshot()
        decision = AccessDecision.SYNTHETIC_ALLOWED
    else:
        snapshot = approved_snapshot()
        decision = AccessDecision.PILOT_VENDOR_APPROVED
    if snapshot is None:
        return None, not_recorded("no snapshot was resolved for this policy")

    try:
        authorise(snapshot, context=context)
    except PilotDataDenied as refusal:
        # No data, and a provenance that records the refusal rather than
        # pretending the question was never asked.
        return None, MarketDataProvenance(
            status=ProvenanceStatus.RECORDED, snapshot_id=snapshot.snapshot_id,
            content_digest=getattr(snapshot, "content_digest", None),
            content_digest_version=getattr(snapshot, "content_digest_version",
                                           None),
            license_class=getattr(snapshot, "license_class", None),
            license_review_status=getattr(snapshot, "license_review_status",
                                          None),
            policy_version=policy.value,
            access_decision=AccessDecision.DENIED,
            access_decision_reason=str(refusal)[:200], accessed_at=stamp)

    try:
        frame = load_prices(snapshot)
    except Exception:
        return None, not_recorded(
            f"snapshot {snapshot.snapshot_id} could not be loaded")

    return frame.sort_index(), recorded(
        snapshot, policy_version=policy.value, decision=decision,
        accessed_at=stamp, reason=context)


def resolve_prices(*, context: str) -> Optional[Any]:
    """Prices only, for callers that do not persist the figure.

    Every caller that *stores* a number must use `resolve` and keep the
    provenance with it.
    """
    frame, _ = resolve(context=context)
    return frame


def approved_snapshot():
    """The vendor snapshot a non-synthetic policy would use.

    Deliberately unimplemented while the pilot is synthetic-only: there is no
    approved vendor snapshot until the six licensing questions are resolved and
    recorded, and returning one here would make `PILOT_VENDOR_APPROVED` mean
    something the licence does not yet permit.
    """
    return None
