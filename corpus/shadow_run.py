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
    matrix,
    render,
)
from src.discovery.readers_quantify import CompilerReader, HostedReader  # noqa: E402


def schema_fingerprint(schema) -> str:
    """What the comparison meant, when it ran.

    Frozen before the large run and recorded in every output. Two runs are
    comparable only if they asked the same questions and compared the answers
    the same way, and this project has already had one measurement invalidated
    by changing the instrument mid-flight — the first 35-prompt run scored
    `"200"` against `"200-day"` as a conflict and the second did not.

    Hashes `Schema.semantic_surface()` — every property capable of changing
    what Discovery can be asked or can say: dimension names, vocabularies and
    comparison modes, and the relation kinds with their roles, required and
    repeatable roles, qualifier and attribute names, and ordering semantics.

    The relation half is not optional. The first version covered dimensions
    only, and schema@2 could have added two relation kinds — materially
    changing the instrument — while presenting the same digest, so two
    incomparable runs would have looked like one experiment.

    Prose descriptions stay out. Rewording what a model is told is a real
    change and the run records it through the model's own readings; hashing it
    would invalidate every baseline over a typo.
    """
    import hashlib
    import json

    digest = hashlib.sha256(
        json.dumps(schema.semantic_surface(), sort_keys=True,
                   separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"{schema.version}/{digest}"


def adjudicated_provenance():
    """What adjudication concluded, per dimension, for the matrix's last column.

    Read from `corpus/adjudicated.json` rather than recomputed, because the
    conclusion is a human judgement about which reader was right — and a
    number that re-derived it would be claiming to know what only a person
    decided.
    """
    path = HERE / "adjudicated.json"
    if not path.exists():
        return {}
    record = json.loads(path.read_text())
    out = {}
    for row in record.get("rows", []):
        current = out.get(row["dimension"])
        verdict = {"COMPILER": "compiler defect", "MODEL": "model miss",
                   "SCHEMA": "schema gap"}[row["defect"]]
        # A dimension with mixed verdicts keeps both — collapsing them would
        # hide that `assets` has failures on both sides.
        if current and verdict not in current:
            out[row["dimension"]] = f"{current} + {verdict}"
        elif not current:
            out[row["dimension"]] = verdict
    for entry in record.get("catalogue_findings", {}).get("classes", []):
        name = entry["dimension"]
        verdict = {"COMPILER": "compiler defect", "MODEL": "model miss",
                   "SCHEMA": "schema gap", "BOTH": "both readers"}[entry["defect"]]
        out[name] = out.get(name) or verdict
    return out


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
    parser.add_argument("--refreeze", action="store_true",
                        help="record the current comparison schema as the "
                             "baseline. Invalidates every earlier run.")
    args = parser.parse_args()

    rows = prompts(args.corpus)
    if args.limit:
        rows = rows[:args.limit]

    compiler = CompilerReader()
    hosted = HostedReader()
    if args.dry_run:
        hosted.api_key_env = "DEFINITELY_NOT_SET"

    fingerprint = schema_fingerprint(QUANTIFY_SCHEMA)
    frozen = HERE / "schema-frozen.txt"
    if frozen.exists():
        expected = frozen.read_text().strip()
        if expected != fingerprint:
            print(f"SCHEMA CHANGED\n  frozen  {expected}\n  now     {fingerprint}\n"
                  "\nA run under a changed comparison schema is not comparable "
                  "with the frozen baseline.\nRe-freeze deliberately "
                  "(`--refreeze`) and re-run every corpus, or revert.")
            if not args.refreeze:
                return 2
    if args.refreeze or not frozen.exists():
        frozen.write_text(fingerprint + "\n")
        print(f"schema frozen at {fingerprint}\n")

    records, states, per_dimension = [], Counter(), {}
    comparisons = []
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

        comparisons.append(result)
        records.append({"ref": ref, "expectation": expectation,
                        "schema": fingerprint, **result.to_json()})
        if index % 10 == 0:
            print(f"  {index}/{len(rows)}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"shadow-{args.corpus}.json").write_text(
        json.dumps({"schema": fingerprint, "records": records}, indent=1))

    readers = (compiler.id, hosted.id)
    by_dimension = matrix(comparisons, QUANTIFY_SCHEMA, readers)
    (args.out / f"matrix-{args.corpus}.json").write_text(
        json.dumps({"schema": fingerprint, "readers": list(readers),
                    "matrix": by_dimension}, indent=1))

    total = sum(states.values())
    print(f"\n{len(rows)} prompts · {total} dimension readings\n")
    for state in (AGREED, CONTESTED, ONE_SIDED):
        share = f"{100 * states[state] / total:.0f}%" if total else "-"
        print(f"  {state:10s} {states[state]:5d}  {share}")

    if unusable:
        print(f"\n  {unusable} prompts produced no usable comparison — fewer "
              "than two readers contributed. These are NOT agreements and are "
              "excluded from every count above.")

    provenance = adjudicated_provenance()
    print("\nby dimension — agree · contested · read by one only · neither\n")
    print(render(by_dimension, readers, provenance))

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
