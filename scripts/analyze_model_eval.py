"""Re-analyse the captured evaluation bundle. No API calls.

Written because the first pass reported two numbers that were the harness being
wrong rather than the model:

    family convergence 0%    measured across all five wordings per family,
                             including the contradictory and underspecified ones
                             that are *meant* to differ

    provenance exact 19%     compared quoted spans rather than field provenance,
                             and counted an `unclear:` note about a phrase
                             outside the vocabulary as a provenance error

Full capture is what made this fixable without spending the budget twice.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUNDLE = Path("reports/modeleval/bundle.jsonl")


def field_provenance(entries: dict) -> dict:
    """Provenance of *fields*, not of quoted text.

    Spans are the words the parser matched. Two parsers reading the same
    sentence legitimately quote different substrings, so comparing spans
    measures phrasing and calls it provenance.
    """
    return {k: v for k, v in entries.items()
            if not k.startswith("span:") and not k.startswith("unclear:")}


def main() -> int:
    rows = [json.loads(line) for line in BUNDLE.open()]
    stability = [r for r in rows if r["klass"] == "STABILITY"]
    contradictory = [r for r in rows if r["klass"] == "CONTRADICTORY"]
    underspecified = [r for r in rows if r["klass"] == "UNDERSPECIFIED"]

    n = len(rows)
    print(f"{n} cases  ({len(stability)} stability, {len(contradictory)} "
          f"contradictory, {len(underspecified)} underspecified)\n")

    print("extraction accuracy")
    for key in ("rule_exact", "schedule_exact", "content_exact"):
        hit = sum(1 for r in rows if r[key])
        print(f"  {key:22} {100 * hit / n:6.1f}%   ({hit}/{n})")

    prov_hit = sum(1 for r in rows
                   if field_provenance(r["expected_provenance"])
                   == field_provenance(r["actual_provenance"]))
    print(f"  {'field provenance':22} {100 * prov_hit / n:6.1f}%   ({prov_hit}/{n})")

    print("\nquestions")
    vocab_extra = [r for r in rows
                   if {q for q in r["actual_questions"]
                       if not q.startswith("unclear:")}
                   - set(r["expected_questions"])]
    unclear_extra = [r for r in rows
                     if any(q.startswith("unclear:")
                            for q in r["actual_questions"])]
    settled = [r for r in rows
               if set(r["expected_questions"]) - set(r["actual_questions"])]
    print(f"  false inference (settled what the text did not)   "
          f"{100 * len(settled) / n:5.1f}%   ({len(settled)}/{n})")
    print(f"  extra questions about vocabulary fields           "
          f"{100 * len(vocab_extra) / n:5.1f}%   ({len(vocab_extra)}/{n})")
    print(f"  extra 'unclear' notes outside the vocabulary      "
          f"{100 * len(unclear_extra) / n:5.1f}%   ({len(unclear_extra)}/{n})")

    # The headline trust metric, measured properly. The first definition only
    # counted questions the compiler *asked* — so a field it never asked about,
    # because no clause triggered the question, could be invented freely. That
    # is exactly what happened: a description with no funding clause came back
    # with `funding_source: additional_cash`, which changes how much money the
    # plan invests.
    invented = []
    for r in rows:
        for change in r["changes"]:
            if change.startswith("flows.") or change.startswith("methodology."):
                invented.append((r["case_id"], change))
    print(f"\n  values changed vs the deterministic reading: {len(invented)}")
    for case_id, change in invented[:6]:
        print(f"    {case_id}: {change[:96]}")

    print("\ncontradiction recall")
    caught = sum(1 for r in contradictory if r["actual_contradictions"])
    print(f"  {100 * caught / (len(contradictory) or 1):5.1f}%   "
          f"({caught}/{len(contradictory)})")

    print("\nparaphrase convergence (stability wordings only)")
    families: dict = {}
    for r in stability:
        families.setdefault(r["family"], set()).add(r["actual_content"])
    converged = [f for f, hashes in families.items() if len(hashes) == 1]
    print(f"  {100 * len(converged) / (len(families) or 1):5.1f}%   "
          f"({len(converged)}/{len(families)} families 3/3)")
    for family, hashes in families.items():
        if len(hashes) > 1:
            print(f"    {family}: {len(hashes)} distinct missions")

    print("\nhard gates")
    saveable_open = [r for r in rows if r["can_save"] and r["actual_questions"]]
    print(f"  saveable with open questions   {len(saveable_open)}   (must be 0)")
    print(f"  schema failures                "
          f"{sum(1 for r in rows if not r['model_available'])}   (must be 0)")
    print(f"  retries                        "
          f"{sum(r['retries'] for r in rows)}")

    print("\noperational")
    lat = sorted(r["latency_ms"] for r in rows)
    print(f"  latency p50 {lat[len(lat)//2]:.0f}ms  "
          f"p95 {lat[int(0.95*len(lat))]:.0f}ms  p99 {lat[int(0.99*len(lat))]:.0f}ms")
    tin = sum(r["input_tokens"] or 0 for r in rows)
    tout = sum(r["output_tokens"] or 0 for r in rows)
    print(f"  tokens in {tin:,} out {tout:,}  "
          f"({tin/n:.0f} / {tout/n:.0f} per call)")
    models = {r["resolved_model"] for r in rows if r["resolved_model"]}
    print(f"  resolved model(s): {sorted(models)}")

    print("\nwhat the model contributed")
    accepted = Counter(f for r in rows for f in r["accepted_from_model"])
    print(f"  fields read that the phrase rules missed: {dict(accepted)}")
    rejected = Counter(x["why"][:52] for r in rows for x in r["rejected"])
    print(f"  proposals refused by the quarantine: {sum(rejected.values())}")
    for why, count in rejected.most_common(4):
        print(f"    {count:>4}  {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
