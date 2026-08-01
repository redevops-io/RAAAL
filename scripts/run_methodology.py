"""Execute a methodology from its specification and publish the result.

This is the spec-driven path: the methodology AST drives the computation, so the
published figure is bound to the version that produced it by construction rather
than by assertion.

Usage::

    python -m scripts.run_methodology --methodology hrp@1
    python -m scripts.run_methodology --methodology hrp@2

Requires a price panel from a completed engine run (``data/history/prices.parquet``).
Publishing is explicit rather than a side effect: it is an act with regulatory
weight.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import compute_returns  # noqa: E402
from src.history import _annualize, strategy_daily_returns  # noqa: E402
from src.ledger import Ledger  # noqa: E402
from src.methodology import MethodologyRegistry  # noqa: E402
from src.methodology.executor import backtest  # noqa: E402
from src.methodology.spec import PerformanceClass  # noqa: E402
from src.reproducibility import (  # noqa: E402
    DEFAULT_SEED,
    build_run_manifest,
    frame_digest,
    seed_everything,
)

PRICES = Path("data/history/prices.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methodology", required=True, help="e.g. hrp@1 or hrp")
    parser.add_argument("--db", default="data/quantify.db")
    parser.add_argument("--prices", default=str(PRICES))
    parser.add_argument("--publish", action="store_true", help="write to the ledger")
    args = parser.parse_args()

    prices_path = Path(args.prices)
    if not prices_path.exists():
        print(
            f"No price panel at {prices_path}. Run `python -m src.history ...` first.",
            file=sys.stderr,
        )
        return 1

    seed_everything(DEFAULT_SEED)
    prices = pd.read_parquet(prices_path)
    returns = compute_returns(prices)

    registry = MethodologyRegistry()
    methodology = registry.resolve(args.methodology)

    weights = backtest(methodology, prices)
    daily = strategy_daily_returns(weights, returns, "weight")
    annualized = _annualize(daily)

    manifest = build_run_manifest(
        run_id=pd.Timestamp.now("UTC").strftime("run_%Y%m%dT%H%M%S%fZ"),
        params={
            "methodology": methodology.version_id,
            "content_hash": methodology.content_hash,
            "rebalance_frequency": methodology.contract.rebalance_frequency,
            **{k: v.value for k, v in methodology.params.items()},
        },
        inputs={"prices": frame_digest(prices)},
        outputs={"weights": frame_digest(weights)},
    )

    print(f"{methodology.version_id}  ({methodology.title})")
    print(f"  content hash    {methodology.content_hash[:16]}...")
    print(f"  cadence         {methodology.contract.rebalance_frequency}")
    print(f"  rebalances      {weights['date'].nunique()}")
    print(f"  period          {daily.index[0].date()} .. {daily.index[-1].date()}")
    print(f"  annualized      {annualized:.4%}  (net of costs, execution-lagged)")
    print(f"  manifest        {manifest.digest[:16]}...")

    if not args.publish:
        print("\n  not published (pass --publish)")
        return 0

    ledger = Ledger(args.db)
    ledger.publish_methodology(methodology)
    ordinal = ledger.record_run(
        run_id=manifest.run_id,
        version_id=methodology.version_id,
        manifest=manifest.__dict__ | {"git": manifest.git},
        manifest_digest=manifest.digest,
        notes="spec-driven execution",
    )
    record = ledger.record_performance(
        performance_id=f"{manifest.run_id}:annualized_return",
        run_id=manifest.run_id,
        version_id=methodology.version_id,
        performance_class=PerformanceClass.BACKTEST_HYPOTHETICAL,
        metric="annualized_return",
        value=float(annualized),
        cost_model=methodology.cost_model,
        period_start=str(daily.index[0].date()),
        period_end=str(daily.index[-1].date()),
    )

    print(f"\n  published       trial #{ordinal}")
    print(f"  class           {record.performance_class.value}")
    print(f"  trials observed {record.trials_at_publication}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
