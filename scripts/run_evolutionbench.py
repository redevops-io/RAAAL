"""EvolutionBench: one plan, lived with over five years.

    python3 scripts/run_evolutionbench.py

Reports replay, reinterpretation and migration separately, because they are
three different questions and only one of them is what a saved plan owes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loadtest.evolutionbench import run, summarize                # noqa: E402
from src.mission.evolution import COMPILER_VERSION                    # noqa: E402

OUT = Path("reports/loadtest")


def main() -> int:
    checkpoints = run()
    summary = summarize(checkpoints)

    print(f"current compiler: version {COMPILER_VERSION}\n")
    print(f"  {'when':12} {'what':34} {'replay':>9} {'reinterp':>9}  outcome")
    for c in checkpoints:
        outcome = ("migration offered" if c.migration_recommended
                   else "differs" if c.reinterpretation_differs else "unchanged")
        print(f"  {c.at:12} {c.what:34} {c.replay_us:>7}us {c.reinterpret_us:>7}us"
              f"  {outcome}")

    print(f"\n{summary['reinterpretations_differing']}/{summary['checkpoints']} "
          f"revisions reinterpret differently under the current compiler")

    for c in checkpoints:
        if not c.diff:
            continue
        print(f"\n  {c.at} — {c.what}")
        for line in c.diff.explain():
            print(f"      {line}")
        break

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evolutionbench.json").write_text(json.dumps({
        "compiler_version": COMPILER_VERSION, **summary,
        "checkpoints": [c.as_row() for c in checkpoints],
    }, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT}/evolutionbench.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
