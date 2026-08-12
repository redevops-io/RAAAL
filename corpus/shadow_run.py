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
from src.discovery.readers_quantify import (CompilerReader,
                                            configured_hosted_reader)  # noqa: E402


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


def run_provenance(schema, prompts_rows, hosted, compiler, mode: str,
                   out_name: str) -> dict:
    """How this measurement was produced, carried with what it measured.

    Two evidence-layer failures produced this, and neither was visible in the
    values a result file contained:

        a `--limit 2 --dry-run` probe overwrote a completed 35-prompt run,
        and the file that replaced it looked like a small valid measurement

        16 of 144 hosted replies were cut mid-JSON by this file's own token
        ceiling, and the parser recorded them the same way it records a reader
        with nothing to say

    In both, the harness confused its own operating conditions with properties
    of the readers. A file carrying only values cannot refuse to be compared
    with a file produced under different conditions — so it carries the
    conditions, and `is_comparable_with` refuses on its behalf.
    """
    import hashlib
    import subprocess

    prompts_digest = hashlib.sha256(
        json.dumps([r[0] for r in prompts_rows], sort_keys=True).encode()
    ).hexdigest()[:16]

    try:
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(HERE.parent),
            capture_output=True, text=True, timeout=10).stdout.strip()[:12]
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(HERE.parent),
            capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:                                             # noqa: BLE001
        tree, dirty = "", True

    return {
        "schema_fingerprint": schema_fingerprint(schema),
        "prompt_count": len(prompts_rows),
        "prompts_digest": prompts_digest,
        "readers": {
            compiler.id: {"enabled": True},
            hosted.id: {"enabled": hosted.available(),
                        "max_tokens": hosted.max_tokens,
                        "api_key_env": hosted.api_key_env},
        },
        "mode": mode,
        "output_name": out_name,
        "commit": tree,
        "tree_dirty": dirty,
        "truncated": 0,
    }


def is_comparable_with(left: dict, right: dict) -> list:
    """Why two result files may not be compared, or an empty list.

    Everything here changes what the numbers mean rather than merely how they
    were reached, which is why `commit` is absent: two runs from different
    commits are comparable if the schema, prompts, readers and mode match, and
    refusing there would forbid the before-and-after a fix is judged by.
    """
    reasons = []
    for key in ("schema_fingerprint", "prompts_digest", "mode"):
        if left.get(key) != right.get(key):
            reasons.append(f"{key}: {left.get(key)!r} != {right.get(key)!r}")
    if set(left.get("readers", {})) != set(right.get("readers", {})):
        reasons.append("different readers")
    for name, spec in left.get("readers", {}).items():
        other = right.get("readers", {}).get(name, {})
        if spec.get("enabled") != other.get("enabled"):
            reasons.append(f"{name} enabled differs")
        if spec.get("max_tokens") != other.get("max_tokens"):
            reasons.append(f"{name} max_tokens differs")
    return reasons


VALIDITY_GATE = (
    "schema fingerprint fixed", "hosted reader active", "no truncation",
    "no output-path collision", "both readers contributed where expected",
    "matrix includes fields and relations")


def validity(provenance: dict, unusable: int, prompt_count: int,
             matrix_by_dimension: dict) -> dict:
    """Whether any semantic number in this run may be quoted at all.

    Checked before the counts are printed, because a number reported under a
    failed gate is read and remembered whatever caveat follows it.
    """
    readers = provenance["readers"]
    hosted = next((v for k, v in readers.items() if "compiler" not in k), {})
    return {
        "schema fingerprint fixed": bool(provenance["schema_fingerprint"]),
        "hosted reader active": bool(hosted.get("enabled")),
        "no truncation": provenance["truncated"] == 0,
        "no output-path collision": provenance["mode"] == "full",
        "both readers contributed where expected":
            prompt_count > 0 and unusable == 0,
        "matrix includes fields and relations":
            any(k.startswith("rel:") for k in matrix_by_dimension),
    }


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
    hosted = configured_hosted_reader()
    suffix = ""
    if args.dry_run:
        hosted.api_key_env = "DEFINITELY_NOT_SET"
        # A dry run must not write where a real run writes. Mine did, and a
        # two-prompt `--limit 2 --dry-run` check silently replaced a completed
        # 35-prompt measurement with two rows in which the hosted reader was
        # switched off. The numbers had already been reported by then.
        #
        # Self-contaminating evidence: the harness overwrote its own result
        # with a probe of itself. Separating the paths is the whole fix.
        suffix = "-dryrun"
    if args.limit and not args.dry_run:
        # A truncated real run is not the corpus either.
        suffix = f"-limit{args.limit}"

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

    mode = "dryrun" if args.dry_run else (
        f"limit{args.limit}" if args.limit else "full")
    provenance = run_provenance(QUANTIFY_SCHEMA, rows, hosted, compiler, mode,
                                f"shadow-{args.corpus}{suffix}.json")

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
    provenance["truncated"] = sum(
        1 for r in records
        if "unparseable" in json.dumps(r.get("failed_readers", {})))

    (args.out / f"shadow-{args.corpus}{suffix}.json").write_text(
        json.dumps({"schema": fingerprint, "provenance": provenance,
                    "records": records}, indent=1))

    readers = (compiler.id, hosted.id)
    by_dimension = matrix(comparisons, QUANTIFY_SCHEMA, readers)
    checks = validity(provenance, unusable, len(rows), by_dimension)
    (args.out / f"matrix-{args.corpus}{suffix}.json").write_text(
        json.dumps({"schema": fingerprint, "provenance": provenance,
                    "validity": checks, "readers": list(readers),
                    "matrix": by_dimension}, indent=1))

    failed_checks = [name for name, ok in checks.items() if not ok]
    print("\nvalidity gate\n")
    for name, ok in checks.items():
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}")
    if failed_checks:
        print("\n  NO SEMANTIC NUMBER FROM THIS RUN MAY BE QUOTED.\n"
              "  A number reported under a failed gate is read and remembered\n"
              "  whatever caveat follows it. Fix and re-run.\n")

    total = sum(states.values())
    print(f"\n{len(rows)} prompts · {total} dimension readings\n")
    for state in (AGREED, CONTESTED, ONE_SIDED):
        share = f"{100 * states[state] / total:.0f}%" if total else "-"
        print(f"  {state:10s} {states[state]:5d}  {share}")

    truncated = sum(1 for r in records
                    if "unparseable" in json.dumps(r.get("failed_readers", {})))
    if truncated:
        print(f"\n  {truncated} prompts lost the hosted reading to TRUNCATION, "
              "not to a\n  disagreement. Raise max_tokens and re-run; these "
              "rows are not evidence\n  about either reader's recall.")

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

    print(f"\nwrote {args.out}/shadow-{args.corpus}{suffix}.json")
    print("\nNeither reader is ground truth. A disagreement says one of them is"
          "\nwrong, not which. Adjudication is what turns this into answers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
