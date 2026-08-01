"""Semantic stability: do paraphrases of one plan compile to one plan?

    python3 scripts/run_stability.py --wordings 40

Every wording in a family means the same thing. They must all produce the same
`rule_hash`. A difference means the compiler read something from the phrasing
rather than the meaning — the defect that makes a system feel arbitrary to a
user who rephrased one sentence and got a different answer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loadtest.catalog import load_strategies                # noqa: E402
from src.loadtest.stability import facts_for, run, summarize    # noqa: E402

OUT = Path("reports/loadtest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wordings", type=int, default=40)
    parser.add_argument("--model", action="store_true")
    args = parser.parse_args()

    stage1 = None
    if args.model:
        from src.mission.parse_model import AnthropicClient, parse_with_model

        client = AnthropicClient()
        stage1 = lambda text: parse_with_model(text, client=client).parsed  # noqa: E731

    strategies = [s for s in load_strategies() if facts_for(s)]
    print(f"{len(strategies)} families x {args.wordings} wordings "
          f"= {len(strategies) * args.wordings:,} descriptions\n")

    families = run(strategies, args.wordings, parser=stage1)
    summary = summarize(families)
    print(f"stability {summary['stability_rate']}%  "
          f"({summary['stable']}/{summary['families']} families, "
          f"{summary['wordings']:,} wordings)")

    unstable = [f for f in families if not f.stable]
    for family in unstable[:10]:
        print(f"\n  {family.strategy_id}: {len(family.distinct)} distinct rule "
              f"hashes, {len(family.errors)} error(s)")
        if family.divergence():
            print("    " + family.divergence())
        for error in family.errors[:2]:
            print(f"    error: {error}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stability.json").write_text(json.dumps({
        "stage1": "model" if args.model else "deterministic",
        **summary,
        "unstable": [{"strategy_id": f.strategy_id,
                      "distinct": len(f.distinct),
                      "errors": f.errors[:5]} for f in unstable],
    }, indent=2) + "\n")
    print(f"\nwrote {OUT}/stability.json")
    return 1 if unstable else 0


if __name__ == "__main__":
    raise SystemExit(main())
