"""Publish a completed backtest into the ledgers.

Bridges the corrected engine to the Release 1 ledgers: reads the run manifest and
performance figures produced by `python -m src.history`, records them against a
pinned methodology version, and lets the ledger assign the trial ordinal.

Usage::

    python -m scripts.publish_run --methodology hrp@1 --metric-key hrp_restricted

Deliberately not automatic. Publishing is an act with regulatory weight, so it is
an explicit command rather than a side effect of running a backtest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ledger import Ledger  # noqa: E402
from src.methodology import MethodologyRegistry  # noqa: E402
from src.methodology.spec import PerformanceClass  # noqa: E402

MANIFEST = Path("data/history/run_manifest.json")
SUMMARY = Path("data/history/history_summary.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methodology", required=True, help="e.g. hrp@1")
    parser.add_argument(
        "--metric-key",
        required=True,
        help="key in history_summary.json performance map, e.g. hrp_restricted",
    )
    parser.add_argument("--metric", default="annualized_return")
    parser.add_argument("--db", default="data/quantify.db")
    args = parser.parse_args()

    if not MANIFEST.exists() or not SUMMARY.exists():
        print(
            "No completed run found. Run `python -m src.history ...` first.",
            file=sys.stderr,
        )
        return 1

    manifest = json.loads(MANIFEST.read_text())
    summary = json.loads(SUMMARY.read_text())
    performance = summary.get("performance", {})

    if args.metric_key not in performance:
        print(f"{args.metric_key!r} not in performance map.", file=sys.stderr)
        print(f"Available: {sorted(performance)[:15]}", file=sys.stderr)
        return 1

    registry = MethodologyRegistry()
    methodology = registry.resolve(args.methodology)

    ledger = Ledger(args.db)
    ledger.publish_methodology(methodology)

    run_id = manifest["run_id"]
    # The digest is recomputed from the manifest rather than trusted from it, so
    # a hand-edited manifest cannot masquerade as a reproducible run.
    from src.reproducibility import RunManifest

    digest = RunManifest(**manifest).digest

    ordinal = ledger.record_run(
        run_id=run_id,
        version_id=methodology.version_id,
        manifest=manifest,
        manifest_digest=digest,
        notes=f"metric_key={args.metric_key}",
    )

    record = ledger.record_performance(
        performance_id=f"{run_id}:{args.metric_key}:{args.metric}",
        run_id=run_id,
        version_id=methodology.version_id,
        performance_class=PerformanceClass.BACKTEST_HYPOTHETICAL,
        metric=args.metric,
        value=float(performance[args.metric_key]),
        cost_model=methodology.cost_model,
        period_start=str(manifest.get("params", {}).get("start", "")),
        period_end=str(manifest.get("params", {}).get("end", "")),
    )

    print(f"Published {methodology.version_id}")
    print(f"  run           {run_id}  (trial #{ordinal})")
    print(f"  manifest      {digest[:16]}...  dirty={manifest['git']['dirty']}")
    print(f"  {args.metric:<13} {record.value:.4%}")
    print(f"  class         {record.performance_class.value}")
    print(f"  trials        {record.trials_at_publication}")
    print(f"  disclosure    {record.disclosure[:70]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
