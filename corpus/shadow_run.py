"""Phase 3's measurement: two readers over the corpus, disagreements recorded.

    python corpus/shadow_run.py --corpus strategies        # 35 labelled
    python corpus/shadow_run.py --corpus catalogue         # 144 generated
    python corpus/shadow_run.py --corpus both --out out/

Runs the deterministic compiler and the hosted reader over the same sentences
and records, per dimension, whether they agreed, disagreed, or only one of them
looked. It resolves nothing — the point of a shadow phase is to make the
disagreements countable *before* anything depends on the new reader.

**What this is not a measurement of.** Not accuracy: neither reader is ground
truth, and a disagreement says one of them is wrong without saying which. The
adjudication step — a person reading the contested rows — is what turns this
into a corpus with answers in it. Until that happens, the honest output is a
list of things to look at, ordered by how often they happen.

**Cost and egress.** One provider call per prompt, carrying the sentence, the
schema and the instructions. Nothing else, per
`data/licensing/discovery-egress@1.yaml` — and `--dry-run` exercises the whole
path with the hosted reader disabled, so the shape can be checked for free.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from src.discovery import (  # noqa: E402
    AGREED,
    CONTESTED,
    ONE_SIDED,
    UNREAD,
    QUANTIFY_SCHEMA,
    compare,
)
from src.discovery.readers_quantify import CompilerReader, HostedReader  # noqa: E402


def prompts(which: str):
    rows = []
    if which in ("strategies", "both"):
        rows += [(r["ref"], r["prompt"], r.get("expectation"))
                 for r in json.loads((HERE / "strategies.json").read_text())]
    if which in ("catalogue", "both"):
        rows += [(r["ref"], r["prompt"], None)
                 for r in json.loads((HERE / "catalogue.json").read_text())]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="strategies",
                        choices=("strategies", "catalogue", "both"))
    parser.add_argument("--out", type=Path, default=HERE / "shadow")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="skip the provider; proves the path without cost")
    args = parser.parse_args()

    rows = prompts(args.corpus)
    if args.limit:
        rows = rows[:args.limit]

    compiler = CompilerReader()
    hosted = HostedReader()
    if args.dry_run:
        hosted.api_key_env = "DEFINITELY_NOT_SET"

    records, states, per_dimension = [], Counter(), {}
    unusable = 0

    for index, (ref, text, expectation) in enumerate(rows, start=1):
        sets = [compiler.read(text, QUANTIFY_SCHEMA),
                hosted.read(text, QUANTIFY_SCHEMA)]
        result = compare(text, sets, QUANTIFY_SCHEMA)
        if not result.usable:
            unusable += 1

        for one in result.fields:
            if one.state is UNREAD or one.state == UNREAD:
                continue
            states[one.state] += 1
            per_dimension.setdefault(one.dimension, Counter())[one.state] += 1

        records.append({"ref": ref, "expectation": expectation,
                        **result.to_json()})
        if index % 10 == 0:
            print(f"  {index}/{len(rows)}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"shadow-{args.corpus}.json").write_text(
        json.dumps(records, indent=1))

    total = sum(states.values())
    print(f"\n{len(rows)} prompts · {total} dimension readings\n")
    for state in (AGREED, CONTESTED, ONE_SIDED):
        share = f"{100 * states[state] / total:.0f}%" if total else "-"
        print(f"  {state:10s} {states[state]:5d}  {share}")

    if unusable:
        print(f"\n  {unusable} prompts produced no usable comparison — fewer "
              "than two readers contributed. These are NOT agreements and are "
              "excluded from every count above.")

    contested = [(d, c[CONTESTED]) for d, c in per_dimension.items()
                 if c[CONTESTED]]
    if contested:
        print("\ncontested by dimension — the adjudication queue, worst first:")
        for dimension, count in sorted(contested, key=lambda x: -x[1]):
            counts = per_dimension[dimension]
            print(f"  {count:4d} contested  {dimension:24s} "
                  f"(agreed {counts[AGREED]}, one-sided {counts[ONE_SIDED]})")

    one_sided = sorted(((d, c[ONE_SIDED]) for d, c in per_dimension.items()
                        if c[ONE_SIDED] and not c[CONTESTED]),
                       key=lambda x: -x[1])[:6]
    if one_sided:
        print("\nread by one reader only — coverage difference, not conflict:")
        for dimension, count in one_sided:
            print(f"  {count:4d}  {dimension}")

    print(f"\nwrote {args.out}/shadow-{args.corpus}.json")
    print("\nNeither reader is ground truth. A disagreement says one of them is"
          "\nwrong, not which. Adjudication is what turns this into answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
