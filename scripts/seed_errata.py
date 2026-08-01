"""Register the published errata into the ledger.

Both errata are backfilled explicitly rather than through an ``UNKNOWN`` member.
There are two of them and both causes are known; introducing an undeclared value
into a system that insists on declared meaning would be the wrong trade for a
little convenience.

Idempotent. The erratum documents under `docs/errata/` are the record of
authority; this makes them queryable so the UI and API can surface them next to
the figures they affect.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ledger import Ledger  # noqa: E402

ERRATA = [
    {
        "erratum_id": "2026-07-30-01",
        "title": "Execution lag and transaction costs",
        "correction_type": "NUMERICAL",
        "cause_type": "EXECUTION",
        "severity": "material",
        "summary": (
            "Backtested performance was overstated: weights earned the return of the "
            "day they were decided, and no transaction costs were charged. The "
            "headline Regime (Unrestricted) figure changes from 13.00% to -2.83%; "
            "11.15pp of the 15.83pp swing is the look-ahead alone. The ensemble "
            "regime classifier's >80% accuracy claim is withdrawn — it was measured "
            "on a shuffled split of a time series."
        ),
        "document_path": "docs/errata/2026-07-30-execution-lag-and-costs.md",
    },
    {
        "erratum_id": "2026-07-30-02",
        "title": "Trading calendar contamination",
        "correction_type": "NUMERICAL",
        "cause_type": "DATA",
        "severity": "material",
        "summary": (
            "BTC-USD forced a seven-day price index, so ETF prices were forward-filled "
            "across weekends and 31.1% of daily returns were zeros that are not market "
            "data. Volatility was understated and the 252-period annualization "
            "mis-scaled. Supersedes the absolute levels in 2026-07-30-01. The trading "
            "calendar is now a referenced artifact (calendar/nyse@1)."
        ),
        "document_path": "docs/errata/2026-07-30-02-trading-calendar.md",
    },
]


def main() -> int:
    ledger = Ledger()
    existing = {e["erratum_id"] for e in ledger.list_errata()}
    for erratum in ERRATA:
        if erratum["erratum_id"] in existing:
            continue
        ledger.publish_erratum(supersedes=[], **erratum)
    for e in ledger.list_errata():
        print(f"{e['erratum_id']}  {e['severity']:9s}  {e['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
