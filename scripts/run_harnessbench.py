"""HarnessBench: canonical versus Polars, results compared before timings.

    python3 scripts/run_harnessbench.py --scales 1000 10000 100000 1000000

Reports the crossover per workload. Not one global constant: a grouped replay
and a latency aggregation cross at different sizes, and a single threshold would
be wrong for both.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loadtest.events import generate                        # noqa: E402
from src.loadtest.harnessbench import (                         # noqa: E402
    WORKLOADS, crossover, measure, with_parquet_projection, write_projection,
)

OUT = Path("reports/loadtest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=int, nargs="+",
                        default=[1_000, 10_000, 100_000, 1_000_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--from-dicts", action="store_true",
                        help="feed Polars from Python rows instead of a Parquet "
                             "projection, to show what the adapter costs")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp())
    measurements = []

    for scale in args.scales:
        print(f"\n{'=' * 78}\n{scale:,} events")
        built = time.perf_counter()
        rows = generate(scale)
        print(f"  generated in {time.perf_counter() - built:.1f}s")

        if args.from_dicts:
            with_parquet_projection(None)
        else:
            path = workdir / f"events-{scale}.parquet"
            written = time.perf_counter()
            write_projection(rows, str(path))
            size = path.stat().st_size / 1024 / 1024
            print(f"  projection written in "
                  f"{time.perf_counter() - written:.1f}s ({size:.1f} MiB)")
            with_parquet_projection(str(path))

        print(f"\n  {'workload':20} {'backend':14} {'p50 ms':>10} {'peak MiB':>9} "
              f"{'rows out':>9}  equivalent")
        for workload in WORKLOADS:
            for m in measure(workload, rows, repeats=args.repeats):
                measurements.append(m)
                verdict = ("-" if m.matches_canonical is None
                           else "yes" if m.matches_canonical else "NO")
                print(f"  {m.workload:20} {m.backend:14} {m.p50_ms:>10.2f} "
                      f"{m.peak_kib / 1024:>9.1f} {m.result_count:>9,}  {verdict}")
                if m.mismatch:
                    print(f"      MISMATCH: {m.mismatch[:150]}")

    mismatches = [m for m in measurements if m.matches_canonical is False]
    print(f"\n{'=' * 78}")
    if mismatches:
        print(f"{len(mismatches)} EQUIVALENCE FAILURE(S) — Polars is not a "
              f"second owner of semantics:")
        for m in mismatches:
            print(f"  {m.workload}/{m.backend} @ {m.rows:,}: {m.mismatch[:160]}")
    else:
        print("all backends agree on every workload at every scale")

    print("\ncrossover (first scale where a Polars backend beats canonical)")
    for workload, scale in crossover(measurements).items():
        print(f"  {workload:22} "
              + (f"{scale:,} events" if scale else
                 "never — canonical wins at every scale measured"))

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "scales": args.scales, "repeats": args.repeats,
        "source": "python-dicts" if args.from_dicts else "parquet-projection",
        "crossover": crossover(measurements),
        "measurements": [m.as_row() for m in measurements],
    }
    (OUT / "harnessbench.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT}/harnessbench.json")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
