"""Run the three assessment layers over an evaluated methodology.

    statistics  → what does the evidence say?
    policy      → does it meet a declared standard?
    publication → who may see it, and labelled how?

Usage::

    python -m scripts.assess --methodology hrp@3 --protocol long-warmup@1 \
        --policy library-default@1 --surface PUBLIC_LIBRARY

Each layer's output is printed separately, because collapsing them is the thing
this design exists to prevent.
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
from src.policy import PolicyRegistry, Surface, decide  # noqa: E402
from src.statistics.assessment import assess  # noqa: E402
from src.statistics.neutralize import FactorModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methodology", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--policy", default="library-default@1")
    parser.add_argument("--surface", default="PUBLIC_LIBRARY")
    parser.add_argument("--prices", default="data/history/prices.parquet")
    parser.add_argument("--db", default="data/quantify.db")
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    methodology = MethodologyRegistry().resolve(args.methodology)
    protocol = ProtocolRegistry().resolve(args.protocol)
    policy = PolicyRegistry().resolve(args.policy)
    surface = Surface(args.surface)

    compat = assess_compatibility(methodology, protocol)
    if not compat.compatible:
        print(f"INCOMPATIBLE: {compat.blockers}")
        return 2

    result, effective = evaluate(methodology, protocol, prices)
    ledger = Ledger(args.db)
    # The run being assessed is itself a trial. Counting only previously recorded
    # runs would report zero for the first evaluation of a lineage, which reads
    # as "no trial count available" rather than "one trial".
    trial_count = max(ledger.trial_count(methodology.concept, dsr_countable_only=True), 1)

    # Build a lineage frame so PBO has comparable configurations to work with.
    lineage = {}
    for version in MethodologyRegistry().versions(methodology.concept):
        pairing = assess_compatibility(version, protocol)
        if not pairing.compatible:
            continue
        try:
            other, _ = evaluate(version, protocol, prices)
            lineage[version.version_id] = other.daily_returns
        except Exception:
            continue
    lineage_frame = pd.DataFrame(lineage).dropna() if len(lineage) > 1 else None

    # Market factor from the protocol's declared benchmark.
    factor_returns = None
    factor_model = None
    if protocol.benchmark and protocol.benchmark in prices.columns:
        bench = prices[protocol.benchmark].pct_change().dropna()
        factor_returns = pd.DataFrame({"market": bench})
        factor_model = FactorModel(
            name="market-only", version=1, factors=("market",), estimation_window=252
        )

    assessment = assess(
        result.daily_returns,
        trial_count=trial_count,
        lineage_returns=lineage_frame,
        factor_returns=factor_returns,
        factor_model=factor_model,
    )

    evaluation = policy.evaluate(assessment, now=pd.Timestamp.now("UTC").isoformat())

    status = dict(result.result_status)
    status["statistical_assessment_complete"] = assessment.complete

    decision = decide(
        surface=surface,
        result_status=status,
        assessment=assessment,
        policy_evaluation=evaluation,
        compatibility_ok=compat.compatible,
    )

    print(f"{methodology.version_id}  ×  {protocol.protocol_id}")
    print(f"  annualized {result.annualized_return:.4%}  sharpe {result.sharpe:.4f}")

    print("\n1. STATISTICAL ASSESSMENT (facts, no verdict)")
    print(f"   computation_status  {assessment.computation_status}")
    print(f"   observations        {assessment.observations}")
    print(f"   trial_count         {assessment.trial_count}  ({assessment.count_policy})")
    for key in ("psr", "dsr", "pbo"):
        payload = getattr(assessment, key)
        if payload:
            print(f"   {key:19s} {payload.get('value'):.4f}")
    if assessment.factor_neutralization:
        fn = assessment.factor_neutralization
        print(f"   neutralization      R²={fn['r_squared']:.3f} "
              f"betas={ {k: round(v,3) for k,v in fn['betas'].items()} }")

    print(f"\n2. POLICY EVALUATION ({evaluation.policy_id})")
    print(f"   status              {evaluation.status.value}")
    print(f"   evidence_grade      {evaluation.evidence_grade.value}")
    for finding in evaluation.findings:
        mark = "ok  " if finding.passed else finding.severity.value.ljust(4)
        print(f"   [{mark}] {finding.code:32s} {finding.detail}")

    # Complete the run record with the three assessment layers, so the run is a
    # standalone artifact rather than something a page has to re-derive.
    run_id = pd.Timestamp.now("UTC").strftime("run_%Y%m%dT%H%M%S%fZ")
    ledger.publish_methodology(methodology)
    ledger.publish_protocol(effective)
    ledger.record_run(
        run_id=run_id,
        version_id=methodology.version_id,
        protocol_id=effective.protocol_id,
        protocol_hash=effective.content_hash,
        manifest={"source": "scripts.assess"},
        manifest_digest=effective.content_hash,
        notes="three-layer assessment",
        result_status=status,
        diagnostics=result.diagnostics,
        execution_audit=result.execution_audit,
        assessment=assessment.to_json(),
        policy_evaluation=evaluation.to_json(),
        publication_decision=decision.to_json(),
    )

    print(f"\n3. PUBLICATION DECISION ({surface.value})")
    print(f"   decision            {decision.decision.value}")
    print(f"   may_claim_validated {decision.may_claim_validated}")
    if decision.hard_blockers:
        print(f"   hard_blockers       {decision.hard_blockers}")
    for disclosure in decision.disclosures:
        print(f"   disclosure          {disclosure}")
    print(f"   {decision.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
