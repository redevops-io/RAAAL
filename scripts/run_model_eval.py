"""Bounded stage-1 model evaluation: 41 families x 5 wordings = 205 calls.

    python3 scripts/run_model_eval.py --max-calls 215

One pinned model. Hard call cap. Full capture. A semantic failure is a result
and is never retried; only transport faults and unparseable output are.

Cross-model comparison comes after this harness produces a clean report —
running several providers before knowing whether the evaluation is calibrated
multiplies the bill without multiplying the information.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loadtest.catalog import load_strategies                      # noqa: E402
from src.loadtest.modeleval import Case, pins, run_case, summarize    # noqa: E402
from src.loadtest.paraphrase import Klass, paraphrases                # noqa: E402
from src.loadtest.stability import facts_for, wordings                # noqa: E402

OUT = Path("reports/modeleval")


def build_cases() -> list:
    """Five wordings per family, chosen to reach every metric in one budget.

    Three semantically identical wordings measure extraction accuracy and
    paraphrase stability; one contradictory and one underspecified reach
    contradiction recall and false inference, which the stability set cannot.
    """
    cases = []
    for strategy in load_strategies():
        facts = facts_for(strategy)
        if facts is None:
            continue
        for index, text in enumerate(wordings(facts, 3)):
            cases.append(Case(case_id=f"{strategy.strategy_id}#s{index}",
                              family=strategy.strategy_id, klass="STABILITY",
                              text=text))
        extra = {p.klass: p for p in paraphrases(strategy, 16)}
        for klass, flags in ((Klass.CONTRADICTORY, {"expects_contradiction": True}),
                             (Klass.UNDERSPECIFIED, {"expects_questions": True})):
            prompt = extra.get(klass)
            if prompt:
                cases.append(Case(case_id=f"{strategy.strategy_id}#{klass.value[:4]}",
                                  family=strategy.strategy_id,
                                  klass=klass.value, text=prompt.text, **flags))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-calls", type=int, default=215,
                        help="hard cap; the run stops and reports partial "
                             "coverage rather than silently exceeding it")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from src.mission.parse_model import AnthropicClient

    client = AnthropicClient(model=args.model)
    cases = build_cases()[:args.limit]
    print(f"{len({c.family for c in cases})} families, {len(cases)} cases, "
          f"cap {args.max_calls} calls\n")

    outcomes, calls, started = [], 0, time.perf_counter()
    for index, case in enumerate(cases, 1):
        if calls >= args.max_calls:
            print(f"\nSTOPPED at the cap: {calls} calls made, "
                  f"{len(cases) - index + 1} cases not run. "
                  "Reporting partial coverage.")
            break
        outcome = run_case(case, client)
        calls += 1 + outcome.retries
        outcomes.append(outcome)
        if index % 20 == 0:
            print(f"  {index}/{len(cases)}  {calls} calls  "
                  f"{time.perf_counter() - started:.0f}s", flush=True)

    elapsed = time.perf_counter() - started
    summary = summarize(outcomes)
    summary["calls_made"] = calls
    summary["cases_run"] = len(outcomes)
    summary["cases_planned"] = len(cases)
    summary["elapsed_s"] = round(elapsed, 1)

    print(f"\n{'=' * 74}\n{len(outcomes)}/{len(cases)} cases, {calls} calls, "
          f"{elapsed:.0f}s\n")
    for key in ("rule_exact", "schedule_exact", "content_exact",
                "provenance_exact", "false_inference_rate",
                "contradiction_recall", "family_convergence_rate",
                "schema_failure_rate", "retry_rate"):
        value = summary[key]
        print(f"  {key:26} " + ("    n/a" if value is None else f"{value:>7}%"))
    print(f"\n  latency p50 {summary['latency_p50_ms']:.0f}ms  "
          f"p95 {summary['latency_p95_ms']:.0f}ms  "
          f"p99 {summary['latency_p99_ms']:.0f}ms")
    print(f"  tokens in {summary['input_tokens']:,} "
          f"out {summary['output_tokens']:,}")
    print(f"\n  failure classes: {summary['failure_classes']}")
    print(f"  saveable with open questions: "
          f"{summary['saveable_with_open_questions']}  (hard gate: 0)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(
        {"pins": pins(client, outcomes), "summary": summary},
        indent=2, default=str) + "\n")
    with (OUT / "bundle.jsonl").open("w") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(outcome.as_row(), default=str) + "\n")
    print(f"\nwrote {OUT}/summary.json and bundle.jsonl "
          f"({len(outcomes)} full capture records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
