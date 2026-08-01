"""Turning a confirmed scenario into a saved worksheet.

    confirmed scenario -> persisted run -> worksheet revision 1

The order matters and is enforced: **a worksheet is created only after the
references it will cite exist.** A worksheet written first and back-filled would
briefly name artifacts that were not there, and "briefly" is exactly when a
crash happens.

Generation is deterministic. Fixed block order, explicit configuration, no
model-chosen layout, and no result payloads copied onto the worksheet — the
worksheet names the run, and the run owns the figures.

Re-saving a plan whose scenario changed produces a **new revision** rather than
overwriting revision 1. That is the same rule as everywhere else here: the
figures someone already read stay readable.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Optional, Sequence

from .worksheet import ResearchWorksheet, create, revise


def run_id_for(plan_id: str, scenario_hash: str, ran_at: str) -> str:
    """A run id that says what produced it.

    Derived rather than sequential so two runs of the same scenario at the same
    moment collide instead of quietly becoming two records of one event.
    """
    digest = hashlib.sha256(
        f"{plan_id}|{scenario_hash}|{ran_at}".encode()).hexdigest()[:12]
    return f"run-{plan_id}-{digest}"


def worksheet_id_for(plan_id: str) -> str:
    return f"ws-{plan_id}"


def generate(store, *, plan_id: str, owner: str, scenario, run: Mapping[str, Any],
             comparison: Mapping[str, Any], ran_at: str,
             title: str = "") -> Optional[ResearchWorksheet]:
    """Persist the run, then create or revise the worksheet that cites it.

    Returns `None` when there is no run to cite. A worksheet whose performance
    block can never be filled is worse than no worksheet: it looks like a
    result that has not loaded.
    """
    if not run:
        return None

    identifier = run_id_for(plan_id, scenario.content_hash, ran_at)
    store.record_run(run_id=identifier, plan_id=plan_id, ran_at=ran_at,
                     result=dict(run), comparison=dict(comparison or {}))

    worksheet_id = worksheet_id_for(plan_id)
    existing = store.get_worksheet(worksheet_id, owner)
    if existing is None:
        worksheet = create(worksheet_id=worksheet_id, owner_id=owner,
                           scenario_ref=plan_id, primary_run_ref=identifier,
                           title=title or plan_id, created_at=ran_at)
        store.save_worksheet(worksheet)
        return worksheet

    from .worksheet import from_json

    previous = from_json(existing["payload"])
    if previous.primary_run_ref == identifier:
        # The same scenario replayed at the same instant. Nothing moved, so
        # there is nothing to record — a revision per page view would bury the
        # changes that matter.
        return previous

    worksheet = revise(previous, reason="a new run was recorded for this plan",
                       primary_run_ref=identifier, created_at=ran_at)
    store.save_worksheet(worksheet)
    return worksheet
