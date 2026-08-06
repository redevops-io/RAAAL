"""Recompile a saved plan under a newer compiler, with the owner's authority.

    python -m src.workspace.migrate_plan --plan PLAN_ID --authorized-by NAME
    python -m src.workspace.migrate_plan --plan PLAN_ID --dry-run

A plan saved before the funding policy existed rebuilds as not-event-funded, so
the engine that now executes conditional rules refuses it exactly as the engine
that could not. The stored scenario is the obstacle, and it is the obstacle *on
purpose*: `plan.scenario` is immutable and pinned to the parse the user read
and confirmed.

So this does not rewrite it. It records an authorisation, stores the
recompiled interpretation beside that authorisation, produces a run against it,
and leaves the withdrawn run attached to the original. The plan then carries
its own history:

    v1   pinned parse, compiler 2, buy-and-hold engine, withdrawn
    v2   same parse, compiler 3, event runtime, reconciled, authoritative

**The parse is not re-derived.** Only the compiler changes. A migration that
re-parsed would change what the user's words were taken to mean, and that is a
different act requiring a different consent — the model may have moved since.

**Nothing is authorised by this file.** `--authorized-by` is recorded, not
checked; the authority is the operator running it on the owner's behalf, and
the record says who. A migration with no name on it would be the system
deciding what a saved plan means, which is precisely what `migration_for`
refuses to do.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import uuid
from typing import Any, Optional, Sequence

#: Why this migration class exists, in one sentence a person can read later.
REASON = ("funding policy introduced: conditional purchase rules are executed "
          "by the engine rather than recorded and replayed as buy-and-hold")

FROM_ENGINE = "engine/buy-and-hold-only@1"


def new_migration_id() -> str:
    return f"mig-{uuid.uuid4().hex[:16]}"


def candidates(store, owner: str) -> Sequence[dict]:
    """Plans holding a withdrawn run whose stored scenario cannot execute.

    Derived, like the withdrawal sweep: the plan that prompted this was found
    by one user opening one page, and a list typed from that page would fix the
    instance and leave the class.
    """
    from .invalidate import declares_a_rule

    found = []
    for plan in store.list_plans(owner):
        record = store.get_plan(plan["plan_id"], owner)
        if record is None or not declares_a_rule(record.get("scenario")):
            continue
        withdrawn = [run for run in store.runs_for(plan["plan_id"], owner)
                     if run.get("invalidation")]
        if not withdrawn:
            continue
        # Already migrated: a second authorisation for the same change would
        # be a second answer to a question the owner already settled.
        if store.migrations_for(plan["plan_id"], owner):
            continue
        allowed, refusal = migratable(record.get("scenario"))
        found.append({"plan_id": plan["plan_id"],
                      "withdrawn_run": withdrawn[0]["run_id"],
                      "stored_scenario": record.get("scenario"),
                      "migratable": allowed, "refusal": refusal})
    return tuple(found)


def migratable(scenario_body) -> tuple:
    """Whether a stored plan's decisions can be replayed, and why not.

    **Migration may replay only persisted structured decisions.** It may not
    infer, parse or reconstruct them from display text.

    A `provenance@1` body dropped `amended` on the way to disk, so its answers
    survive only as rendered sentences under `stated` —
    `"account_type: TAXABLE (answered)"`. Reading those back would reverse the
    direction of authority:

        structured answer → rendered sentence → reconstructed answer

    and the reconstruction would then sit inside the record that exists to say
    what the owner agreed to. The two-minute cost of re-entering a plan is far
    smaller than a reverse-parser in the most sensitive part of the chain.
    """
    from ..mission.spec import (
        LEGACY_PROVENANCE_INCOMPLETE,
        provenance_shape_of,
    )

    body = (scenario_body or {}).get("provenance") if isinstance(
        scenario_body, dict) else None
    shape = provenance_shape_of(body)
    if shape == "provenance@1":
        return False, LEGACY_PROVENANCE_INCOMPLETE
    return True, ""


def stored_amendments(scenario_body) -> tuple:
    """The answers the user gave, replayed from the saved plan.

    Recompiling from the pinned parse alone is not enough, and the production
    plan proved it: without the amendments the identity of "SP500 ETF" is
    unresolved again, the amount reads as unclear, and the recompiled plan has
    no funding policy at all — so the migration would silently produce nothing
    and look like the engine still could not execute the rule.

    The parse and the amendments together are exactly what the user confirmed.
    Replaying both and changing only the compiler is what makes this a
    migration rather than a re-interview.
    """
    from ..mission.spec import ScenarioAmendment

    if not isinstance(scenario_body, dict):
        return ()
    provenance = scenario_body.get("provenance") or {}
    return tuple(
        ScenarioAmendment(question_id=one.get("question_id", ""),
                          answer=str(one.get("answer", "")),
                          recorded_at=one.get("recorded_at", ""))
        for one in (provenance.get("amended") or ())
        if one.get("question_id"))


def recompile(store, plan_id: str, owner: str):
    """Today's compiler applied to yesterday's parse and yesterday's answers."""
    from ..mission.compiler import compile_scenario

    from . import routes

    record = store.get_plan(plan_id, owner)
    if record is None:
        raise SystemExit(f"no plan {plan_id!r}")

    access = routes._market_data("migrating a saved plan")
    return record, compile_scenario(
        record["stated_text"], name=plan_id, version=2,
        benchmark_rule=routes.BENCHMARK_RULE,
        parsed=routes._pinned_parse(record),
        amendments=stored_amendments(record.get("scenario")),
        priceable=tuple(access.frame.columns)
        if access.usable else ()), access


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ..deploy.context import current
    from ..mission.evolution import COMPILER_VERSION
    from ..mission.ledger import EXECUTION_ENGINE_VERSION
    from .store import WorkspaceStore
    from . import routes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="")
    parser.add_argument("--authorized-by", default="")
    parser.add_argument("--owner", default="pilot")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    store = WorkspaceStore(current().database.url)
    targets = candidates(store, args.owner)
    if args.plan:
        targets = [one for one in targets if one["plan_id"] == args.plan]
    if not targets:
        print("no plans awaiting migration")
        return 0

    if not args.dry_run and not args.authorized_by:
        print("--authorized-by is required: adopting a new interpretation "
              "changes what a saved plan means, and the record must name who "
              "agreed", file=sys.stderr)
        return 2

    for target in targets:
        plan_id = target["plan_id"]
        if not target["migratable"]:
            print(f"{plan_id}: {target['refusal']} — this plan predates "
                  f"structured amendment persistence. Its displayed answers "
                  f"are a rendering and cannot be used as migration input. "
                  f"Create a replacement plan through the builder.")
            continue
        record, compiled, access = recompile(store, plan_id, args.owner)
        scenario = compiled.scenario

        if not scenario.is_event_funded:
            print(f"{plan_id}: recompiles to a plan this engine still cannot "
                  f"execute; not migrated")
            continue

        run = routes._run(scenario, access)
        if run.get("result") is None:
            print(f"{plan_id}: {run.get('unavailable')}")
            continue

        ledger = run["ledger"]
        print(f"{plan_id}: {ledger.summary()}")
        if args.dry_run:
            print("  (dry run; nothing written)")
            continue

        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        migration_id = new_migration_id()
        # The authorisation first, then the run, then the link. A run written
        # before its migration would cite a record that does not exist.
        store.record_migration(
            migration_id=migration_id, plan_id=plan_id, owner=args.owner,
            from_compiler=str(record["scenario"].get("compiler_version", "2")),
            to_compiler=COMPILER_VERSION,
            from_engine=FROM_ENGINE, to_engine=EXECUTION_ENGINE_VERSION,
            reason=REASON, authorized_by=args.authorized_by,
            migrated_at=stamp, scenario=scenario,
            old_run=target["withdrawn_run"])

        run_id = f"{plan_id}-run-{uuid.uuid4().hex[:12]}"
        store.record_run(run_id=run_id, plan_id=plan_id, owner=args.owner,
                         ran_at=stamp, result=run["result"].to_json(),
                         comparison={**(run.get("payload") or {}),
                                     **(run.get("comparability_records") or {})})
        store.attach_migration_run(migration_id=migration_id,
                                   owner=args.owner, run_id=run_id)
        print(f"  migration {migration_id}")
        print(f"  replacement run {run_id}")
        print(f"  fingerprint {ledger.fingerprint()}")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
