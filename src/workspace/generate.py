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
import uuid
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


def new_worksheet_id() -> str:
    """An opaque, server-generated identity.

    It was `ws-{plan_id}`, which put a user's own scenario name into an
    identifier that appears in URLs and is shared across tenants. Two problems,
    and the second outlives the first: the id was guessable, and it was derived
    from something the user chose, so knowing a plan name was enough to name
    another tenant's worksheet.

    The human-readable name lives on as `title`, which is a field rather than an
    identity — an identifier that carries meaning is an identifier that leaks it.
    """
    return f"ws-{uuid.uuid4().hex}"


class UnattributableRun(ValueError):
    """A live producer tried to store a figure it could not attribute."""


def _require_recorded_provenance(run_id: str, market_data) -> None:
    """`generate` is a live producer, so it may not store an absence.

    The store's own guard accepts NOT_RECORDED, because a migration must be
    able to carry a legacy row through unchanged. That leniency is correct
    there and wrong here: a live path satisfying the guard by labelling its own
    omission as historical absence produces a record saying "nobody recorded
    this" when in fact this code declined to, and the two are indistinguishable
    afterwards.

    Enforced at this boundary rather than left to the route. A caller that must
    remember to supply something is a caller that can forget, and the route is
    not the only thing that will ever call this.
    """
    from ..market_data.provenance import (
        ProvenanceStatus,
        from_json,
        verify,
    )

    if market_data is None:
        raise UnattributableRun(
            f"run {run_id} has no market-data provenance. This is a live "
            "producer: it must supply the record the resolver returned, and "
            "may not fall back to stating that none exists.")

    problems = verify(market_data)
    if problems:
        raise UnattributableRun(
            f"run {run_id} has incoherent provenance: " + "; ".join(problems))

    provenance = from_json(market_data)
    if provenance.status is ProvenanceStatus.NOT_RECORDED:
        raise UnattributableRun(
            f"run {run_id} claims its provenance was never recorded. Only an "
            "import path may say that; a live producer holding the resolver's "
            "answer is declining to store it.")
    if provenance.status is ProvenanceStatus.RECORDED:
        if not provenance.identifies_data:
            raise UnattributableRun(
                f"run {run_id} names a snapshot without a content digest, so "
                "two snapshots sharing a label cannot be told apart.")
        if not provenance.permitted:
            raise UnattributableRun(
                f"run {run_id} was produced under a denied access decision, "
                "which should not have yielded data at all.")


def generate(store, *, plan_id: str, owner: str, scenario, run: Mapping[str, Any],
             comparison: Mapping[str, Any], ran_at: str,
             title: str = "", provenance=None) -> Optional[ResearchWorksheet]:
    """Persist the run, then create or revise the worksheet that cites it.

    Returns `None` when there is no run to cite. A worksheet whose performance
    block can never be filled is worse than no worksheet: it looks like a
    result that has not loaded.
    """
    if not run:
        return None

    identifier = run_id_for(plan_id, scenario.content_hash, ran_at)
    # Carried into the result rather than passed beside it, because the
    # provenance belongs to the figure and travels wherever the figure does.
    body = dict(run)
    if provenance is not None and "market_data" not in body:
        body["market_data"] = provenance.to_json()
    _require_recorded_provenance(identifier, body.get("market_data"))
    store.record_run(run_id=identifier, plan_id=plan_id, ran_at=ran_at,
                     result=body, comparison=dict(comparison or {}))

    # Found by what it cites, not by an id computed from the plan name. With an
    # opaque identity there is nothing to recompute, and looking it up by
    # (owner, scenario) is the question actually being asked: does this user
    # already have a worksheet for this scenario?
    existing = store.worksheet_for_scenario(plan_id, owner)
    if existing is None:
        worksheet = create(worksheet_id=new_worksheet_id(), owner_id=owner,
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
