"""Find stored runs whose declared rule was never executed, and mark them.

    python -m src.workspace.invalidate [--dry-run]

The defect: `_run` called `simulate(..., program=buy_and_hold(tradeable))`
whatever the scenario declared, and nothing converted `event_program` into an
`EventProgram`. A plan describing "buy $1,000 every time SPY crosses below its
200-day average" was replayed as one purchase held to the end, and returned a
figure identical to the buy-and-hold benchmark beside a disclosure saying the
difference was attributable to the rule.

**The inventory is derived, not listed.** The affected plan was found by a user
opening one page; naming that plan here would fix the instance and leave the
class. Every run for the tenant is read, and each is tested against a property
of the artifacts themselves.

**The predicate is forward-compatible.** A run is affected when its plan
declares a non-empty `event_program` *and* the stored result carries no
evidence that any rule event executed. Once the engine executes the program and
records `rule_events`, a correct run stops matching without this file being
edited — which is the difference between a sweep and a hardcoded list of
victims.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence

#: The runtime's vocabulary, not prose. Grouped by an operator asking how far
#: the defect reached.
CLASSIFICATION = "RULE_NOT_EXECUTED"

#: What was wrong with the figure, in a sentence a person can read. Deliberately
#: names what the number *is* rather than only what it is not: "invalid" alone
#: leaves a reader assuming a small error rather than a different question.
REASON = (
    "The plan declared a conditional purchase rule. The engine replayed a "
    "buy-and-hold program instead, so this figure is the result of holding the "
    "instruments and not of following the rule."
)

#: The engine that produced the affected runs. Recorded so a replacement run can
#: say what changed rather than merely being newer.
ENGINE_VERSION = "engine/buy-and-hold-only@1"


def executed_rule_events(result: Any) -> Optional[int]:
    """How many rule events a stored result claims to have executed.

    `None` means the result predates the field entirely, which is not the same
    as zero and must not be read as it. A run recorded before the engine could
    say anything about rule events is affected because of what its plan
    declared, not because it reported a zero it never had the vocabulary for.
    """
    if not isinstance(result, dict):
        return None
    if "rule_events" not in result:
        return None
    events = result.get("rule_events")
    if isinstance(events, int):
        return events
    if isinstance(events, (list, tuple)):
        return len(events)
    return None


def declares_a_rule(scenario: Any) -> bool:
    """Whether the stored scenario declares an event program.

    Reads the stored JSON rather than recompiling the description. The
    recompiled scenario is today's reading of the text; what was shown to the
    user is what was stored, and this asks which figures were wrong.
    """
    if not isinstance(scenario, dict):
        return False
    methodology = scenario.get("methodology")
    if isinstance(methodology, dict) and methodology.get("event_program"):
        return True
    return bool(scenario.get("event_program"))


def affected(store, owner: str) -> List[Dict[str, Any]]:
    """Every stored run that showed a figure for a rule that never ran."""
    plans = {p["plan_id"]: p for p in store.list_plans(owner)}
    found = []
    for run in store.all_runs(owner):
        plan = plans.get(run["plan_id"])
        if plan is None:
            continue
        record = store.get_plan(run["plan_id"], owner)
        if record is None or not declares_a_rule(record.get("scenario")):
            continue
        if executed_rule_events(run.get("result")):
            continue        # the rule ran; this figure is about the rule
        found.append({"run_id": run["run_id"], "plan_id": run["plan_id"],
                      "ran_at": run["ran_at"]})
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    import datetime as dt

    from ..deploy.context import current
    from .store import WorkspaceStore

    arguments = list(argv if argv is not None else sys.argv[1:])
    dry_run = "--dry-run" in arguments
    owner = "pilot"

    store = WorkspaceStore(current().database.url)
    rows = affected(store, owner)
    if not rows:
        print("no affected runs")
        return 0

    at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    written = 0
    for row in rows:
        print(f"{'would invalidate' if dry_run else 'invalidating'} "
              f"{row['run_id']} (plan {row['plan_id']}, ran {row['ran_at']})")
        if not dry_run:
            wrote = store.invalidate_run(
                run_id=row["run_id"], plan_id=row["plan_id"], owner=owner,
                classification=CLASSIFICATION, reason=REASON,
                engine_version=ENGINE_VERSION, at=at)
            written += int(wrote)
            if not wrote:
                print(f"  already withdrawn; the first notice stands")
    if dry_run:
        print(f"{len(rows)} run(s) would be invalidated")
    else:
        print(f"{written} run(s) invalidated, "
              f"{len(rows) - written} already withdrawn")
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
