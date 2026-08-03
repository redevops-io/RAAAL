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

from typing import Any, Optional

#: Every file a production request is permitted to obtain market data from is
#: reached through `load_prices`, keyed by snapshot identity. This path is not
#: one of them, and is named here so a test can assert nothing reads it.
UNMANIFESTED_PRICES = "data/history/prices.parquet"


def resolve_prices(*, context: str) -> Optional[Any]:
    """Prices for a production request, or None.

    `context` is recorded on the authorisation decision, so a denial says which
    request wanted the data rather than only that something did.
    """
    from .loader import load_prices, synthetic_snapshot
    from .pilot_policy import (
        PilotDataDenied,
        PilotDataPolicy,
        PilotPolicyMissing,
        authorise,
        configured_policy,
    )

    try:
        policy = configured_policy()
    except PilotPolicyMissing:
        # Fails closed. Guidance in a runbook is not a gate.
        return None

    snapshot = (synthetic_snapshot()
                if policy is PilotDataPolicy.SYNTHETIC_ONLY
                else approved_snapshot())
    if snapshot is None:
        return None

    try:
        authorise(snapshot, context=context)
    except PilotDataDenied:
        return None

    try:
        frame = load_prices(snapshot)
    except Exception:
        # A snapshot that will not load is a missing snapshot. Falling through
        # to any other source would be the bypass this function replaced.
        return None
    return frame.sort_index()


def approved_snapshot():
    """The vendor snapshot a non-synthetic policy would use.

    Deliberately unimplemented while the pilot is synthetic-only: there is no
    approved vendor snapshot until the six licensing questions are resolved and
    recorded, and returning one here would make `PILOT_VENDOR_APPROVED` mean
    something the licence does not yet permit.
    """
    return None
