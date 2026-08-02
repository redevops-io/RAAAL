"""Resolving a grant for tests, through the real runtime.

Not a bypass. `resolved_for` calls the same `resolve_for_vest` production calls,
handing it an explicit empty action history — which is an answer ("this snapshot
reports no actions") rather than the absence of one ("no snapshot was pinned").
The two behave differently and must keep doing so.

Tests that care about corporate actions pass their own events.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from src.runtime.corporate_action import (
    US_CORPORATE_ACTIONS,
    CorporateActionEvent,
    RealizedCorporateActions,
)
from src.runtime.rsu import resolve_for_vest

ISSUER = "issuer/test"


def resolved_for(vest, *, granted: float | None = None,
                 events: Sequence[CorporateActionEvent] = (),
                 snapshot: str | None = None):
    """The `ResolvedGrant` a vest needs, from an explicit action history."""
    history = RealizedCorporateActions(
        snapshot_ref=snapshot or vest.corporate_action_ref or "actions/test@1",
        events=tuple(events))
    return resolve_for_vest(
        vest, granted_shares=(vest.gross_shares if granted is None else granted),
        issuer_ref=ISSUER, realized=history, runtime=US_CORPORATE_ACTIONS)
