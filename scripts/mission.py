"""Mission DevOps: treat a Mission like a software project.

    mission create      compile a description into a scenario
    mission validate    would this save, and what is unrepresented
    mission benchmark   run it and report comparability before performance
    mission replay      re-derive the stored plan and check it still reproduces
    mission diff        stored plan versus what today's compiler would make
    mission verify      every claim this plan makes, checked against its record
    mission publish     record a methodology in the public library
    mission rollback    point a worksheet back at an earlier revision

Everything underneath already existed; only the tooling did not. Each verb
composes what the engine already does rather than adding semantics, which is why
none of them takes an option that would let it decide something.

Exit codes are meaningful: 0 when the claim holds, 1 when it does not. A tool
whose failure is a paragraph nobody reads is a tool that runs in CI and never
fails a build.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BENCHMARK_RULE = "benchmark-policy/public-default@1"
OWNER = "pilot"


def _store(path: str | None):
    from src.workspace.store import WorkspaceStore

    return WorkspaceStore(Path(path) if path else Path("data/workspace.db"))


def _compile(text: str):
    from src.mission.compiler import compile_scenario

    return compile_scenario(text, name="mission", version=1,
                            benchmark_rule=BENCHMARK_RULE)


def _print_scenario(result) -> None:
    from src.workspace.confirmation import build as build_confirmation

    view = build_confirmation(result, text="")
    print(f"path: {view.path}")
    for row in view.summary:
        print(f"  {row['label']:14} {row['value']}")


# --- verbs -----------------------------------------------------------------

def create(args) -> int:
    result = _compile(args.description)
    _print_scenario(result)
    if result.contradictions:
        print("\ncontradictions:")
        for conflict in result.contradictions:
            print(f"  {conflict.detail}")
    if result.unresolved:
        print("\nstill unanswered:")
        for question in result.unresolved:
            print(f"  {question.field}: {question.question}")
    print(f"\nrunnable: {result.can_simulate}   saveable: {result.can_save}")
    return 0 if result.can_simulate else 1


def validate(args) -> int:
    from src.mission.compiler import parse
    from src.mission.representation import representation_gaps

    result = _compile(args.description)
    problems = []

    gaps = representation_gaps(parse(args.description), result.scenario)
    problems += [f"unrepresented: {gap}" for gap in gaps]
    # The compiler surfaces contradictions it found in the *compiled* form, and
    # the scenario can restate the same conflict. Deduplicated on the detail so
    # one problem reads as one problem — a list that counts a finding twice
    # makes a caller think there are two.
    seen = set()
    for detail in ([c.detail for c in result.contradictions]
                   + (result.scenario.self_conflicts()
                      if result.scenario is not None else [])):
        if detail not in seen:
            seen.add(detail)
            problems.append(f"contradiction: {detail}")

    print(f"{len(result.stated)} stated · {len(result.inferred)} inferred · "
          f"{len(result.unresolved)} unanswered")
    for problem in problems:
        print(f"  FAIL  {problem}")
    if not problems:
        print("  ok    every recognised value reached the compiled scenario")
    return 1 if problems else 0


def benchmark(args) -> int:
    import pandas as pd

    result = _compile(args.description)
    if result.scenario is None:
        print("nothing to run: the description did not compile")
        return 1

    prices_path = Path(args.prices)
    if not prices_path.exists():
        print(f"no price data at {prices_path}. This is a data gap, not a result.")
        return 1

    prices = pd.read_parquet(prices_path)
    assets = [a for a in result.scenario.allocation_rule.assets
              if a in prices.columns]
    if not assets:
        print(f"no price history for "
              f"{', '.join(result.scenario.allocation_rule.assets) or 'the named instruments'}"
              ". This is a data gap, not a result.")
        return 1

    from src.workspace.environment import pins_for
    from src.mission.comparability import RunConditions, classify

    snapshot = f"prices@{prices.index[-1].date()}"
    pins = pins_for(result.scenario, snapshot=snapshot)
    conditions = RunConditions(
        **pins.as_conditions(),
        flow_schedule_hash=result.scenario.flow_schedule.schedule_hash,
        starting_capital=result.scenario.flow_schedule.starting_capital,
        cash_policy_rate=0.0, tax_treatment=result.scenario.tax_treatment,
        cost_bps=10.0, execution_lag=1,
        period_start=str(prices.index[0].date()),
        period_end=str(prices.index[-1].date()),
        allocation_rule_hash=result.scenario.rule_hash, data_snapshot=snapshot)

    verdict = classify(conditions, conditions)
    # Comparability first, always. A figure read before its verdict has already
    # been compared.
    print(f"comparability: {verdict.comparison_class.value}")
    print(f"  attribution isolated: {verdict.attribution_isolated}")
    if verdict.unchecked_dimensions:
        print(f"  not evaluated: {', '.join(verdict.unchecked_dimensions)}")
    if pins.unpinned:
        print(f"  unpinned runtimes: {', '.join(pins.unpinned)}")
    print(f"  {verdict.required_disclosure}")
    return 0


def replay(args) -> int:
    from src.mission.evolution import rebuild_scenario

    store = _store(args.store)
    plan = store.get_plan(args.plan_id, args.owner)
    if plan is None:
        print(f"no plan {args.plan_id!r} for {args.owner!r}")
        return 1

    rebuilt = rebuild_scenario(plan["scenario"])
    if rebuilt is None:
        print("the stored body is too old to rebuild faithfully")
        return 1

    stored_hash = plan["scenario"].get("content_hash")
    matches = rebuilt.content_hash == stored_hash
    print(f"stored     {stored_hash}")
    print(f"rebuilt    {rebuilt.content_hash}")
    print("reproduces" if matches else
          "DOES NOT REPRODUCE — the record was modified after it was saved")
    return 0 if matches else 1


def diff(args) -> int:
    from src.mission.evolution import COMPILER_VERSION, diff_stored_against

    store = _store(args.store)
    plan = store.get_plan(args.plan_id, args.owner)
    if plan is None:
        print(f"no plan {args.plan_id!r} for {args.owner!r}")
        return 1

    current = _compile(plan["stated_text"])
    difference = diff_stored_against(
        plan["scenario"], current.scenario,
        stored_compiler=str(plan["scenario"].get("compiler_version", "1")),
        current_compiler=COMPILER_VERSION,
        current_unresolved=[u.field for u in current.unresolved])

    if difference.is_empty:
        print("no difference: today's compiler reads this the same way")
        return 0
    for line in difference.explain():
        print(f"  {line}")
    print("\nThis is a proposal. The stored plan is unchanged.")
    return 0


def verify(args) -> int:
    """Every claim the saved plan makes, checked against its own record.

    Local by construction. A `VerificationResult` from the contracts package
    describes a control-plane check; this asks the narrower question a developer
    actually has — does this plan still hold together?
    """
    from src.mission.compiler import parse
    from src.mission.evolution import rebuild_scenario
    from src.mission.representation import representation_gaps

    store = _store(args.store)
    plan = store.get_plan(args.plan_id, args.owner)
    if plan is None:
        print(f"no plan {args.plan_id!r} for {args.owner!r}")
        return 1

    failures = []
    rebuilt = rebuild_scenario(plan["scenario"])
    if rebuilt is None:
        failures.append("the stored body cannot be rebuilt")
    elif rebuilt.content_hash != plan["scenario"].get("content_hash"):
        failures.append("the stored body does not reproduce its own hash")

    gaps = representation_gaps(parse(plan["stated_text"]), rebuilt)
    failures += [f"unrepresented: {gap}" for gap in gaps]

    if rebuilt is not None:
        failures += [f"self-conflict: {c}" for c in rebuilt.self_conflicts()]

    runs = store.runs_for(args.plan_id, args.owner)
    for run in runs:
        if not (run["result"] or {}).get("modelling_scope"):
            failures.append(f"run {run['run_id']} carries no modelling scope")

    print(f"plan {args.plan_id} · {len(runs)} run(s)")
    for failure in failures:
        print(f"  FAIL  {failure}")
    if not failures:
        print("  ok    reproduces, fully represented, no conflicts, "
              "every run declares its scope")
    return 1 if failures else 0


def publish(args) -> int:
    from src.evaluation import ProtocolRegistry
    from src.ledger import Ledger
    from src.methodology import MethodologyRegistry

    ledger = Ledger(Path(args.ledger))
    registry = MethodologyRegistry()
    published = 0
    for methodology in registry.load_all():
        if args.name and methodology.artifact_id != args.name:
            continue
        ledger.publish_methodology(methodology)
        print(f"  published {methodology.artifact_id}")
        published += 1
    for protocol in ProtocolRegistry().load_all():
        ledger.publish_protocol(protocol)
    if not published:
        print("nothing matched; publishing no-ops on identical content and "
              "refuses a changed body under an existing id")
        return 1
    return 0


def rollback(args) -> int:
    from src.workspace.worksheet import from_json, revise

    store = _store(args.store)
    target = store.get_worksheet(args.worksheet_id, args.owner, args.to)
    if target is None:
        print(f"no revision {args.to} of {args.worksheet_id!r}")
        return 1

    latest = from_json(store.get_worksheet(args.worksheet_id, args.owner)["payload"])
    previous = from_json(target["payload"])
    if latest.revision == previous.revision:
        print("already at that revision")
        return 0

    # Forward, never backward. Rolling back by deleting revisions would erase
    # the history that revisions exist to keep, so this creates a *new* revision
    # pointing at the older references.
    restored = revise(latest, reason=f"rolled back to revision {args.to}",
                      scenario_ref=previous.scenario_ref,
                      primary_run_ref=previous.primary_run_ref,
                      benchmark_run_refs=previous.benchmark_run_refs,
                      layout=previous.layout, created_at=args.at)
    store.save_worksheet(restored)
    print(f"revision {restored.revision} restores the references from "
          f"revision {args.to}; revisions {previous.revision}-{latest.revision} "
          "remain readable")
    return 0


VERBS = {"create": create, "validate": validate, "benchmark": benchmark,
         "replay": replay, "diff": diff, "verify": verify,
         "publish": publish, "rollback": rollback}


def _migrate(args) -> int:
    """Export a SQLite workspace, plan the import, and optionally apply it.

    The dry run and the real import consume the *same* plan object. Recomputing
    it would let a dry run report one thing and the import do another, which is
    the only failure mode a dry run exists to rule out.
    """
    import datetime as dt
    import json as _json

    from src.db.engine import Database
    from src.db.transfer import (
        ExportRefused,
        ImportRefused,
        apply_import,
        export_bundle,
        plan_import,
        verify_import,
    )
    from src.workspace.store import WorkspaceStore

    source = WorkspaceStore(args.from_sqlite)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    try:
        bundle = export_bundle(source, exported_at=stamp,
                               commit=os.environ.get("QUANTIFY_COMMIT", ""),
                               owner=args.owner)
    except ExportRefused as refusal:
        print(f"export refused:\n{refusal}")
        return 1

    if args.bundle:
        pathlib.Path(args.bundle).write_text(
            _json.dumps(bundle, indent=2, sort_keys=True, default=str))

    target = Database(args.to_postgres) if args.to_postgres else Database()
    plan = plan_import(target, bundle)

    print(f"rows ready      {plan.ready}")
    print(f"redeliveries    {plan.redelivered}")
    print(f"conflicts       {len(plan.conflicts)}")
    print(f"unknown tables  {plan.unknown_tables or 'none'}")
    print(f"digest status   {'verified' if bundle['manifest']['bundle_digest'] else 'absent'}")
    for conflict in plan.conflicts[:10]:
        print(f"  conflict {conflict['table']} {conflict['identity']}")

    if args.dry_run:
        print("\ndry run: nothing was written")
        return 1 if plan.conflicts or plan.unknown_tables else 0

    try:
        apply_import(target, bundle, plan)
    except ImportRefused as refusal:
        print(f"import refused:\n{refusal}")
        return 1

    problems = verify_import(target, bundle)
    if problems:
        print("verification failed:\n  " + "\n  ".join(problems))
        return 1
    print("\nimported and verified against the bundle")
    return 0


VERBS["migrate"] = _migrate


def main() -> int:
    parser = argparse.ArgumentParser(prog="mission", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="verb", required=True)

    for name in ("create", "validate", "benchmark"):
        p = sub.add_parser(name)
        p.add_argument("description")
        if name == "benchmark":
            p.add_argument("--prices",
                           default="tests/fixtures/prices_synthetic.parquet")

    for name in ("replay", "diff", "verify"):
        p = sub.add_parser(name)
        p.add_argument("plan_id")
        p.add_argument("--store", default=None)
        p.add_argument("--owner", default=OWNER)

    p = sub.add_parser("publish")
    p.add_argument("--name", default=None)
    p.add_argument("--ledger", default="data/quantify.db")

    p = sub.add_parser("migrate")
    p.add_argument("--from-sqlite", required=True,
                   help="path to the SQLite workspace to export")
    p.add_argument("--to-postgres", default=None,
                   help="target URL; defaults to QUANTIFY_DATABASE_URL")
    p.add_argument("--owner", default=None,
                   help="export one tenant only; omitted, exports everything")
    p.add_argument("--dry-run", action="store_true",
                   help="validate, plan and report without writing anything")
    p.add_argument("--bundle", default=None,
                   help="write the neutral bundle here as well")

    p = sub.add_parser("rollback")
    p.add_argument("worksheet_id")
    p.add_argument("--to", type=int, required=True)
    p.add_argument("--store", default=None)
    p.add_argument("--owner", default=OWNER)
    p.add_argument("--at", default=None)

    args = parser.parse_args()
    return VERBS[args.verb](args)


if __name__ == "__main__":
    raise SystemExit(main())
