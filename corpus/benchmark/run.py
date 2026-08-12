"""Run the strategy benchmark and produce an improvement queue.

    python corpus/benchmark/run.py            # writes findings.json
    python corpus/benchmark/run.py --print    # and shows the queue

**Checkpoints, not final returns.** Each prompt is carried through the serving
path and what happened at every stage is recorded, because "two equivalent
prompts gave different answers" is useless until you know whether Discovery
read them differently, Mission compiled them differently, or execution
diverged. The layer is part of the finding.

**Outcomes are split by danger, not by success.**

    CORRECT_EXECUTION       supported, executed, agrees with its siblings
    CORRECT_REFUSAL         unsupported, refused by name
    CORRECT_CLARIFICATION   under-specified, asked about

    SILENT_REDUCTION        unsupported and executed anyway
    UNSTABLE_EXECUTION      siblings executed different plans
    UNSTABLE_SAFE           siblings disagreed, and at most one executed
    WRONG_EXECUTABLE_MEANING  a contrast pair compiled to one plan
    FALSE_CLAIM_OF_SUPPORT  refused for the wrong capability
    UNNECESSARY_REFUSAL     supported and refused
    UNNECESSARY_QUESTION    supported, fully specified, and asked about

A count of executions is not a score. An unsupported strategy refused by name
is a pass, and a benchmark that rewarded execution would reward exactly the
defect this project spent months removing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE / "suite.json"
OUT = HERE / "findings.json"

CORRECT_EXECUTION = "CORRECT_EXECUTION"
CORRECT_REFUSAL = "CORRECT_REFUSAL"
CORRECT_CLARIFICATION = "CORRECT_CLARIFICATION"

SILENT_REDUCTION = "SILENT_REDUCTION"
UNSTABLE_EXECUTION = "UNSTABLE_EXECUTION"
UNSTABLE_SAFE = "UNSTABLE_SAFE"
WRONG_EXECUTABLE_MEANING = "WRONG_EXECUTABLE_MEANING"
FALSE_CLAIM_OF_SUPPORT = "FALSE_CLAIM_OF_SUPPORT"
UNNECESSARY_REFUSAL = "UNNECESSARY_REFUSAL"
UNNECESSARY_QUESTION = "UNNECESSARY_QUESTION"
INCOMPLETE_REFUSAL = "INCOMPLETE_REFUSAL"
READER_FAILED = "READER_FAILED"

#: `UNSTABLE_SAFE` is deliberately absent. Two phrasings of one request where
#: one executes and the other is refused by name is a recognition gap and not a
#: danger — nobody is shown a plan that is not theirs. It stays in the queue,
#: ranked below everything here, because it is still something to fix.
DANGEROUS = {SILENT_REDUCTION, UNSTABLE_EXECUTION, WRONG_EXECUTABLE_MEANING,
             FALSE_CLAIM_OF_SUPPORT}

#: Which layer a finding belongs to. Recorded on the finding so the queue says
#: where to look rather than only what went wrong.
DISCOVERY, FUSION, MISSION, SURFACE = "Discovery", "Fusion", "Mission", "Surface"


def _readers():
    from src.discovery.hosted_recording import RecordedHostedReader
    from src.discovery.syntax_stanza import RecordedReader

    return RecordedHostedReader(), RecordedReader()


def checkpoints(text: str, model, syntax) -> dict:
    """Everything one prompt did on the way through, stage by stage."""
    from hashlib import sha256

    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.witnesses import BOTH
    from src.workspace.pilot import read

    try:
        reading = read(text, model, schema=QUANTIFY_SCHEMA, profile=BOTH,
                       syntax_reader=syntax)
    except Exception as failure:                                # noqa: BLE001
        return {"text": text, "stage_failed": f"{type(failure).__name__}",
                "detail": str(failure)[:160]}

    plan = None
    scenario = getattr(reading.compiled, "scenario", None)
    if scenario is not None:
        canonical = json.dumps(scenario.execution_form(), sort_keys=True,
                               default=str)
        plan = sha256(canonical.encode()).hexdigest()[:16]

    return {
        "text": text,
        "settled": {f.field: str(f.value) for f in reading.settled
                    if f.value is not None},
        "open": list(reading.open_fields),
        "questions": list(reading.questions),
        # Split by kind. `UNRESOLVED_INPUT` is a question wearing a refusal's
        # shape — Mission saying "the intent names nothing to hold" — and
        # counting it as a capability refusal reports a reader asking a
        # reasonable question as a reader refusing a supported strategy.
        "refusals": sorted({getattr(r, "dimension", "") for r in reading.refusals
                            if getattr(r, "kind", "") != "UNRESOLVED_INPUT"}),
        "needs_input": sorted({getattr(r, "dimension", "") for r in reading.refusals
                               if getattr(r, "kind", "") == "UNRESOLVED_INPUT"}),
        "sealed": reading.intent is not None and reading.intent.is_verified,
        "executable": bool(reading.executable),
        "plan": plan,
    }


def _disposition(point: dict) -> str:
    if point.get("stage_failed"):
        return "FAILED"
    if point["refusals"]:
        return "REFUSES"
    if point["executable"]:
        return "EXECUTES"
    return "CLARIFIES"


def _asks_about(point: dict) -> list:
    return sorted(set(point.get("questions", ())) | set(point.get("needs_input", ())))


def classify_class(entry: dict, points: dict) -> list:
    """Findings for one equivalence class.

    The class is its own oracle: every phrasing must reach the same disposition
    and, where it executes, the same plan. No known portfolio value is needed
    and none is used.
    """
    findings = []
    expected = entry["disposition"]

    for phrasing in entry["phrasings"]:
        point = points[phrasing]
        got = _disposition(point)

        if got == "FAILED":
            findings.append({"kind": READER_FAILED, "layer": DISCOVERY,
                             "class": entry["id"], "prompt": phrasing,
                             "detail": point.get("detail", "")})
            continue

        if expected == "REFUSES":
            if got == "EXECUTES":
                findings.append({
                    "kind": SILENT_REDUCTION, "layer": MISSION,
                    "class": entry["id"], "prompt": phrasing,
                    "detail": ("unsupported and executed anyway; read "
                               f"{sorted(point['settled'])}")})
            elif got == "REFUSES" and entry["refuses"]:
                if not set(point["refusals"]) & set(entry["refuses"]):
                    findings.append({
                        "kind": FALSE_CLAIM_OF_SUPPORT, "layer": MISSION,
                        "class": entry["id"], "prompt": phrasing,
                        "detail": (f"refused {point['refusals']}, expected one "
                                   f"of {entry['refuses']}")})
                # Every unsupported thing the sentence states must be named.
                # Refusing one and staying silent about the other leaves the
                # person believing the rest was fine.
                unmentioned = sorted(set(entry["states"])
                                     - set(point["refusals"])
                                     - set(point["settled"]))
                if unmentioned:
                    findings.append({
                        "kind": INCOMPLETE_REFUSAL, "layer": DISCOVERY,
                        "class": entry["id"], "prompt": phrasing,
                        "detail": (f"refused {point['refusals']} and never "
                                   f"mentioned {unmentioned}, which the "
                                   "sentence also asks for")})

        if expected == "EXECUTES":
            if got == "REFUSES":
                findings.append({
                    "kind": UNNECESSARY_REFUSAL, "layer": MISSION,
                    "class": entry["id"], "prompt": phrasing,
                    "detail": f"refused {point['refusals']}"})
            elif got == "CLARIFIES":
                findings.append({
                    "kind": UNNECESSARY_QUESTION, "layer": DISCOVERY,
                    "class": entry["id"], "prompt": phrasing,
                    "detail": f"asked about {point['questions']}"})

    # Siblings that executed must have executed the same plan.
    plans = {points[p]["plan"] for p in entry["phrasings"]
             if points[p].get("plan")}
    if len(plans) > 1:
        findings.append({
            "kind": UNSTABLE_EXECUTION, "layer": FUSION,
            "class": entry["id"], "prompt": entry["phrasings"][0],
            "detail": (f"{len(plans)} different compiled plans across "
                       f"{len(entry['phrasings'])} phrasings of one strategy")})
    return findings


def classify_contrast(pair: dict, points: dict) -> list:
    """A pair a word apart that must not agree."""
    left, right = points[pair["left"]], points[pair["right"]]
    if left.get("stage_failed") or right.get("stage_failed"):
        return []
    if left.get("plan") and left["plan"] == right["plan"]:
        return [{"kind": WRONG_EXECUTABLE_MEANING, "layer": FUSION,
                 "class": pair["name"], "prompt": pair["left"],
                 "detail": (f"compiles identically to {pair['right']!r} — "
                            + pair["why"])}]
    return []


def _executable_identity(point: dict) -> tuple:
    """What executes: the disposition, the compiled plan, and the refusals.

    Deliberately excludes the settled values, because a `SAME` relation is
    about meaning and `$1,000` against `$1k` may legitimately differ in surface
    form while compiling identically. If they do *not* compile identically that
    is the finding, and the plan digest is what says so.
    """
    return (_disposition(point), point.get("plan"),
            tuple(point.get("refusals", ())))


def _semantic_identity(point: dict) -> tuple:
    """Everything the reader concluded, settled values included.

    Used for `DIFFER`, because two prompts that are both refused have no plan
    and must still be distinguishable — `60/40` and `40/60` are refused
    identically and the reader does tell them apart, in `stated_weights`.

    The first version compared only the plan. Both refused prompts had none, so
    "the plan did not change" was trivially true and the pair was reported as a
    conflation the reader had not made. That is the fourth time in this work a
    convenient proxy stood in for the property being measured, and the fourth
    time it manufactured a finding.
    """
    return (_disposition(point), point.get("plan"),
            tuple(sorted((point.get("settled") or {}).items())),
            tuple(point.get("refusals", ())))


def classify_metamorphic(relation: dict, points: dict) -> list:
    """A transformation, and what it must do to the meaning."""
    before, after = points[relation["from"]], points[relation["to"]]
    if before.get("stage_failed") or after.get("stage_failed"):
        return []

    if relation["relation"] == "SAME":
        same = _executable_identity(before) == _executable_identity(after)
    else:
        same = _semantic_identity(before) == _semantic_identity(after)

    if relation["relation"] == "SAME" and not same:
        # Safe or dangerous, the distinction the drift lane already draws.
        #
        # Two phrasings that mean one thing and compile to two *executable*
        # plans is the dangerous shape: whichever the person happened to type
        # decided what was run, and both looked like an answer. Two phrasings
        # where one executes and the other is refused by name is a gap in what
        # the runtime recognises — real, worth closing, and not the same thing,
        # because nobody is shown a plan that is not theirs.
        #
        # Collapsing the two would have been convenient here and wrong: it is
        # how a taxonomy starts reporting the number that is easiest to move.
        both_executed = bool(before.get("plan")) and bool(after.get("plan"))
        return [{"kind": UNSTABLE_EXECUTION if both_executed else UNSTABLE_SAFE,
                 "layer": FUSION,
                 "class": relation["name"], "prompt": relation["to"],
                 "detail": (f"{relation['name']} changed the compiled plan; "
                            f"{relation['from']!r} and {relation['to']!r} "
                            "mean the same thing"
                            if both_executed else
                            f"{relation['name']}: {relation['from']!r} "
                            f"executes and {relation['to']!r} is refused, "
                            "though they mean the same thing")}]
    if relation["relation"] == "DIFFER" and same:
        return [{"kind": WRONG_EXECUTABLE_MEANING, "layer": FUSION,
                 "class": relation["name"], "prompt": relation["to"],
                 "detail": (f"{relation['name']} left the plan unchanged; "
                            f"{relation['from']!r} and {relation['to']!r} "
                            "mean different things")}]
    return []


def run() -> dict:
    sys.path.insert(0, str(HERE.parent.parent))
    suite = json.loads(SUITE.read_text())
    model, syntax = _readers()

    points = {p: checkpoints(p, model, syntax) for p in suite["prompts"]}

    findings = []
    for entry in suite["classes"]:
        findings += classify_class(entry, points)
    for pair in suite["contrasts"]:
        findings += classify_contrast(pair, points)
    for relation in suite["metamorphic"]:
        findings += classify_metamorphic(relation, points)

    # What went right, counted separately so the queue is not read as the
    # whole picture.
    correct = {CORRECT_EXECUTION: 0, CORRECT_REFUSAL: 0,
               CORRECT_CLARIFICATION: 0}
    flagged = {f["prompt"] for f in findings}
    for entry in suite["classes"]:
        for phrasing in entry["phrasings"]:
            if phrasing in flagged:
                continue
            got = _disposition(points[phrasing])
            if got == "EXECUTES":
                correct[CORRECT_EXECUTION] += 1
            elif got == "REFUSES":
                correct[CORRECT_REFUSAL] += 1
            elif got == "CLARIFIES":
                correct[CORRECT_CLARIFICATION] += 1

    # The improvement queue: by failure class and recurrence, not by prompt.
    queue: dict = {}
    for finding in findings:
        key = (finding["kind"], finding["layer"], finding["class"])
        row = queue.setdefault(key, {"kind": finding["kind"],
                                     "layer": finding["layer"],
                                     "area": finding["class"],
                                     "instances": 0, "examples": []})
        row["instances"] += 1
        if len(row["examples"]) < 3:
            row["examples"].append({"prompt": finding["prompt"],
                                    "detail": finding["detail"]})

    ranked = sorted(queue.values(),
                    key=lambda r: (r["kind"] not in DANGEROUS, -r["instances"]))

    return {
        "schema": "quantify-benchmark-findings@1",
        "prompts": len(points),
        "correct": correct,
        "dangerous_instances": sum(r["instances"] for r in ranked
                                   if r["kind"] in DANGEROUS),
        "queue": ranked,
        "findings": findings,
        "scoring_note": (
            "There is no pass rate here, deliberately. An unsupported strategy "
            "refused by name is correct, and a score that counted executions "
            "would reward silent reduction."),
        "checkpoints": points,
    }


def main(show: bool = False) -> int:
    report = run()
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"{report['prompts']} prompts")
    for name, total in sorted(report["correct"].items()):
        print(f"  {name:24} {total}")
    print(f"  {'dangerous instances':24} {report['dangerous_instances']}")

    if show and report["queue"]:
        print("\nimprovement queue — by failure class and recurrence\n")
        print(f"  {'finding':28} {'inst':>4}  {'layer':10} area")
        for row in report["queue"]:
            print(f"  {row['kind']:28} {row['instances']:>4}  "
                  f"{row['layer']:10} {row['area']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(show="--print" in sys.argv))
