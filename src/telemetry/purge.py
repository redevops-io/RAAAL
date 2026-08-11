"""Delete expired telemetry. Run by an operator, not by a scheduler.

    python -m src.telemetry.purge [--dry-run]

Retention is resolved by the deployment (`QUANTIFY_TRACE_RETENTION_DAYS`,
default 90) and performed here. There is deliberately no scheduling subsystem:
a closed pilot's trace volume is small, and a cron entry calling this is the
whole requirement.

    0 * * * *  cd /srv/quantify && python -m src.telemetry.purge

Financial artifacts are never touched. This opens the trace database only, and
`purge_before` consults nothing in the workspace — nothing in the workspace is
allowed to need a trace.
"""
from __future__ import annotations

import datetime as dt
import sys
from typing import Optional, Sequence


def cutoff_for(retention_days: int, *, now: Optional[dt.datetime] = None) -> str:
    moment = now or dt.datetime.now(dt.timezone.utc)
    return (moment - dt.timedelta(days=retention_days)).isoformat(
        timespec="seconds")


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ..deploy.context import current

    arguments = list(argv if argv is not None else sys.argv[1:])
    dry_run = "--dry-run" in arguments

    telemetry = current().telemetry
    if not telemetry.enabled:
        print("telemetry is disabled for this deployment; nothing to purge")
        return 0

    store = telemetry.store()
    if store is None:
        # Consistent with every other telemetry path: a store that cannot be
        # opened costs telemetry, never the caller. An operator running this by
        # hand still needs to be told, so it says so and exits non-zero.
        print("the trace store could not be opened", file=sys.stderr)
        return 1

    cutoff = cutoff_for(telemetry.retention_days)
    if dry_run:
        print(f"would purge traces started before {cutoff}")
        return 0

    removed = store.purge_before(cutoff)
    print(f"purged before {cutoff}: " +
          ", ".join(f"{count} {name}" for name, count in sorted(removed.items())))
    return 0


if __name__ == "__main__":                                   # pragma: no cover
    raise SystemExit(main())
