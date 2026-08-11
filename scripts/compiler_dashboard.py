"""Compiler quality, as metrics rather than a pass count.

    python3 scripts/compiler_dashboard.py

Run after any parser, compiler or stage-1 change. "980 tests passed" says
nothing about whether recognition improved or regressed; these numbers do, and
every one of them exists because the corpus found the matching defect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loadtest.catalog import load_strategies                # noqa: E402
from src.loadtest.dashboard import as_json, build, render       # noqa: E402
from src.loadtest.harness import Report, run_corpus             # noqa: E402
from src.loadtest.paraphrase import corpus                      # noqa: E402

OUT = Path("reports/loadtest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-strategy", type=int, default=100)
    parser.add_argument("--model", action="store_true",
                        help="use model-assisted stage 1 instead of the "
                             "deterministic phrase rules")
    args = parser.parse_args()

    stage1 = None
    if args.model:
        from src.mission.parse_model import AnthropicClient, parse_with_model

        client = AnthropicClient()
        stage1 = lambda text: parse_with_model(text, client=client).parsed  # noqa: E731

    strategies = load_strategies()
    report = Report(run_corpus(corpus(strategies, args.per_strategy),
                               parser=stage1))
    metrics = build(report, strategies=len(strategies))
    print(render(metrics))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compiler-quality.json").write_text(
        json.dumps({"stage1": "model" if args.model else "deterministic",
                    "prompts": len(report.outcomes),
                    "metrics": as_json(metrics)}, indent=2) + "\n")
    print(f"\nwrote {OUT}/compiler-quality.json")
    return 0 if all(m.meets_target for m in metrics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
