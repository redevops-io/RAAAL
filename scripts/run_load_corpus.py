"""Run the strategy corpus through the compiler and report what broke.

    python3 scripts/run_load_corpus.py --per-strategy 100

Writes a JSON report beside a human summary. The summary is the point: a wall of
numbers nobody reads is not a test result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loadtest.catalog import load_strategies                # noqa: E402
from src.loadtest.harness import Report, run_corpus             # noqa: E402
from src.loadtest.paraphrase import corpus                      # noqa: E402

OUT = Path("reports/loadtest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-strategy", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N strategies, for a quick pass")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    strategies = load_strategies()[:args.limit]
    prompts = corpus(strategies, args.per_strategy)
    print(f"{len(strategies)} strategies x {args.per_strategy} paraphrases "
          f"= {len(prompts):,} prompts\n")

    started = time.perf_counter()
    outcomes = run_corpus(
        prompts, progress=lambda n: print(f"  {n:,} ...", flush=True))
    elapsed = time.perf_counter() - started
    report = Report(outcomes)

    print(f"\ncompiled {len(outcomes):,} in {elapsed:.1f}s "
          f"({len(outcomes) / elapsed:,.0f}/s)\n")

    print("latency (microseconds)")
    for stage, stats in report.latency().items():
        print(f"  {stage:12} p50 {stats['p50']:>8,.0f}  p95 {stats['p95']:>8,.0f}  "
              f"p99 {stats['p99']:>9,.0f}  max {stats['max']:>9,.0f}")

    print("\nby paraphrase class")
    print(f"  {'class':22} {'n':>6} {'ok':>6} {'crash':>6} {'disagree':>9} "
          f"{'saveable':>9} {'asked':>7}")
    for klass, stats in sorted(report.by_class().items()):
        print(f"  {klass:22} {stats['n']:>6} {stats['ok']:>6} "
              f"{stats['crashed']:>6} {stats['disagreed']:>9} "
              f"{stats['saveable']:>9} {stats['asked']:>7}")

    problems = report.distinct_problems()
    print(f"\n{len(problems)} distinct problem(s), "
          f"{sum(len(v) for v in problems.values()):,} occurrence(s)")
    for message, ids in problems.items():
        print(f"\n  {len(ids):,} x  {message}")
        print(f"     e.g. {', '.join(ids[:4])}")

    worst = sorted(report.by_family().items(),
                   key=lambda kv: -kv[1]["problems"] / max(kv[1]["n"], 1))
    print("\nfamilies by problem rate")
    for family, stats in worst[:8]:
        rate = stats["problems"] / max(stats["n"], 1)
        print(f"  {rate:6.1%}  {stats['problems']:>5}/{stats['n']:<5} {family}")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategies": len(strategies), "per_strategy": args.per_strategy,
        "prompts": len(outcomes), "elapsed_s": round(elapsed, 2),
        "latency": report.latency(), "by_class": report.by_class(),
        "by_family": report.by_family(),
        "problems": {k: v[:50] for k, v in problems.items()},
        "crashes": [o.as_row() for o in report.crashes[:50]],
    }
    (args.out / "corpus-report.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    rows = args.out / "corpus-outcomes.jsonl"
    with rows.open("w") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome.as_row(), default=str) + "\n")
    print(f"\nwrote {args.out}/corpus-report.json and {rows.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
