from __future__ import annotations

import argparse
from datetime import datetime

from src.pipeline import run_allocation
from src.reporting import allocation_table, save_reports


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-adjusted allocation MVP")
    parser.add_argument("--start", type=_parse_date, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=_parse_date, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--refresh", action="store_true", help="Force refetch of price data")
    args = parser.parse_args()

    result = run_allocation(start=args.start, end=args.end, force_refresh=args.refresh)

    print(f"Detected regime: {result.regime.name}")
    for k, v in result.regime.diagnostics.items():
        print(f"  {k}: {v:.4f}")
    print()

    print("Portfolio metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.4f}")
    print()

    print(allocation_table(result))

    files = save_reports(result)
    print("\nArtifacts saved:")
    for label, path in files.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
