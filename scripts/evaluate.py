"""Evaluate a methodology under an evaluation protocol, and publish the pair.

    methodology + evaluation protocol = performance

Usage::

    python -m scripts.evaluate --methodology hrp@1 --protocol standard@1 --publish
    python -m scripts.evaluate --methodology hrp@1 --protocol sealed@1

A sealed protocol truncates the price panel before execution, so the holdout is
unreachable rather than merely unreported.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import ProtocolRegistry  # noqa: E402
from src.evaluation.runner import assess_compatibility, evaluate  # noqa: E402
from src.ledger import Ledger  # noqa: E402
from src.methodology import MethodologyRegistry  # noqa: E402
from src.methodology.spec import PerformanceClass  # noqa: E402
from src.trial import build_trial_identity  # noqa: E402
from src.reproducibility import (  # noqa: E402
    DEFAULT_SEED,
    build_run_manifest,
    seed_everything,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methodology", required=True, help="e.g. hrp@1")
    parser.add_argument("--protocol", required=True, help="e.g. standard@1")
    parser.add_argument("--prices", default="data/history/prices.parquet")
    parser.add_argument("--db", default="data/quantify.db")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument(
        "--acknowledge-flags",
        action="store_true",
        help="publish despite diagnostic flags; the flags are recorded either way",
    )
    args = parser.parse_args()

    prices_path = Path(args.prices)
    if not prices_path.exists():
        print(f"No price panel at {prices_path}.", file=sys.stderr)
        return 1

    seed_everything(DEFAULT_SEED)
    prices = pd.read_parquet(prices_path)

    methodology = MethodologyRegistry().resolve(args.methodology)
    protocol = ProtocolRegistry().resolve(args.protocol)

    # Assess the pairing first and persist the verdict either way. A refused
    # pairing is still an attempted configuration and belongs in trial accounting.
    identity = build_trial_identity(
        methodology_hash=methodology.content_hash,
        protocol_hash=protocol.content_hash,
        objective="annualized_return",
        data_partition="holdout_sealed" if protocol.holdout.sealed else "full",
        execution_assumptions={
            "cost_bps": protocol.transaction_costs.bps,
            "execution_lag_days": protocol.transaction_costs.execution_lag_days,
            "purge": protocol.walk_forward.purge,
            "embargo": protocol.walk_forward.embargo,
        },
    )
    compat = assess_compatibility(methodology, protocol)

    ledger = Ledger(args.db)
    ledger.publish_methodology(methodology)
    ledger.publish_protocol(protocol)
    ledger.record_compatibility(
        compatibility_id=f"{identity.trial_id}:compat",
        concept=methodology.concept,
        version_id=methodology.version_id,
        protocol_id=protocol.protocol_id,
        trial_id=identity.trial_id,
        compatible=compat.compatible,
        blockers=compat.blockers,
    )

    if not compat.compatible:
        print(f"{methodology.version_id}  ×  {protocol.protocol_id}")
        print(f"  trial             {identity.trial_id}")
        print("  INCOMPATIBLE — recorded as an attempted configuration")
        for b in compat.blockers:
            print(f"    - {b['code']}: required {b['required']}, provided {b['provided']}")
            print(f"      {b['detail']}")
        return 2

    result, effective = evaluate(methodology, protocol, prices)

    print(f"{result.methodology_version_id}  ×  {result.protocol_id}")
    print(f"  methodology hash  {result.methodology_hash[:16]}...")
    print(f"  protocol hash     {result.protocol_hash[:16]}...")
    print(f"  snapshot hash     {effective.data_snapshot.content_hash[:16]}...")
    print(f"  costs             {effective.transaction_costs.bps}bps, "
          f"lag {effective.transaction_costs.execution_lag_days}d")
    print(f"  grid              {effective.walk_forward.scheme}, "
          f"warmup {effective.walk_forward.warmup}, "
          f"purge {effective.walk_forward.purge}, "
          f"embargo {effective.walk_forward.embargo}")
    if result.sealed_period_excluded:
        print(f"  HOLDOUT SEALED    {effective.holdout.start} .. {effective.holdout.end} "
              "(excluded from this evaluation)")
    print(f"  period            {result.period_start} .. {result.period_end}")
    print(f"  rebalances        {result.n_rebalances}")
    print(f"  annualized        {result.annualized_return:.4%}")
    print(f"  volatility        {result.volatility:.4%}")
    print(f"  sharpe            {result.sharpe:.4f}")
    print(f"  max drawdown      {result.max_drawdown:.4%}")

    diag = result.diagnostics or {}
    audit = result.execution_audit or {}
    status = result.result_status or {}

    print(f"  top holding       {diag.get('top_asset')} "
          f"({diag.get('top_asset_mean_weight', 0):.1%} mean weight)")
    print(f"  effective breadth {diag.get('effective_n_assets', 0):.2f} assets")
    print(f"  fallback usage    {audit.get('fallback_share', 0):.1%} of rebalances"
          + (f" ({audit.get('fallback_by_rule')})" if audit.get("fallback_by_rule") else ""))
    if audit.get("requested_turnover_cap") is not None:
        print(f"  turnover          requested cap {audit['requested_turnover_cap']}, "
              f"realized mean {audit.get('realized_turnover_mean', 0):.4f}, "
              f"max {audit.get('realized_turnover_max', 0):.4f}")
    print(f"  precedence overrides {audit.get('precedence_override_count', 0)}")

    print("\n  RESULT STATUS")
    for key in (
        "computation_valid", "contract_valid", "statistical_valid",
        "economically_degenerate", "publication_eligible",
    ):
        print(f"    {key:26s} {status.get(key)}")

    if result.flags:
        print("\n  DIAGNOSTIC FLAGS")
        for flag in result.flags:
            print(f"    - {flag}")

    if not args.publish:
        print("\n  not published (pass --publish)")
        return 0

    if result.flags and not args.acknowledge_flags:
        print(
            "\n  REFUSING TO PUBLISH: the result is flagged. Review the flags above.\n"
            "  Publish anyway with --acknowledge-flags; the flags are recorded either way.",
            file=sys.stderr,
        )
        return 2

    manifest = build_run_manifest(
        run_id=pd.Timestamp.now("UTC").strftime("run_%Y%m%dT%H%M%S%fZ"),
        params={
            "methodology": methodology.version_id,
            "methodology_hash": methodology.content_hash,
            "protocol": effective.protocol_id,
            "protocol_hash": effective.content_hash,
        },
        inputs={"prices": effective.data_snapshot.content_hash},
        outputs={"annualized_return": f"{result.annualized_return:.10f}"},
    )

    # Persist the full picture at execution time. Recomputing any of it on read
    # would mean a run page shows today's code applied to yesterday's execution —
    # a rendering, not a record. It is also what lets Discovery traverse rather
    # than re-run.
    ordinal = ledger.record_run(
        run_id=manifest.run_id,
        version_id=methodology.version_id,
        protocol_id=effective.protocol_id,
        protocol_hash=effective.content_hash,
        manifest=manifest.__dict__ | {"git": manifest.git},
        manifest_digest=manifest.digest,
        trial_id=identity.trial_id,
        outcome="completed",
        notes="methodology x protocol evaluation",
        result_status=result.result_status,
        diagnostics=result.diagnostics,
        execution_audit=result.execution_audit,
    )

    record = ledger.record_performance(
        performance_id=f"{manifest.run_id}:annualized_return",
        run_id=manifest.run_id,
        version_id=methodology.version_id,
        protocol_id=effective.protocol_id,
        protocol_hash=effective.content_hash,
        performance_class=PerformanceClass.BACKTEST_HYPOTHETICAL,
        metric="annualized_return",
        value=result.annualized_return,
        cost_model=f"{effective.transaction_costs.model}:{effective.transaction_costs.bps}",
        period_start=result.period_start,
        period_end=result.period_end,
    )

    breakdown = ledger.trial_breakdown(methodology.concept)
    print(f"\n  published         {identity.trial_id}")
    print(f"  attempted trials  {breakdown['attempted_trials']} "
          f"({breakdown['blocked_before_execution']} blocked before execution)")
    print(f"  DSR-countable     {breakdown['dsr_countable_trials']}")
    print(f"  repeat executions {breakdown['repeat_executions']} "
          "(reproducibility checks, not new searches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
