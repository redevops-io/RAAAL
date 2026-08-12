"""Whether nondeterminism can change what gets executed.

    python corpus/parser/drift_lane.py                 # 3 draws, escalate to 5
    python corpus/parser/drift_lane.py --canary        # the daily boundary check
    python corpus/parser/drift_lane.py --longitudinal  # 1 draw, for schedule
    python corpus/parser/drift_lane.py --draws 5       # override

**Not in the ordinary suite.** Every commit runs recorded readers: deterministic
and free. This lane calls the provider live, in the exact serving profile —
hosted model *and* Stanza, `WitnessProfile.BOTH` — because the question it
answers cannot be answered from a recording. A recording replays one stored
answer however many times you ask, so measuring stability against it would
report perfect stability for a reader that has none.

**The property is not determinism.** Lean does not need Discovery to give the
same answer twice. It needs Discovery to stop a stochastic reader from silently
changing *executable meaning* between runs. So the classification is over the
final fused artifact, never over raw model output:

    STABLE_EXECUTABLE      every draw produced the same executable plan,
                           compared by compiled-plan digest, not outcome class
    STABLE_REFUSAL         every draw refused, by the same capabilities
    STABLE_CLARIFICATION   every draw asked, and asked the same things
    UNSTABLE_SAFE          draws disagreed, and none of them executed
    UNSTABLE_EXECUTABLE    draws disagreed and at least one executed

The last two are split because they are different problems. A prompt that
refuses on one draw and asks a question on the next is annoying. A prompt that
executes on one draw and refuses on the next — or executes *a different plan* —
is a correctness blocker, and it is the only kind that can put a wrong number
in front of somebody.

Two executable draws with different compiled plans are `UNSTABLE_EXECUTABLE`
even though both "worked". That is the case a naive outcome-class comparison
misses entirely, and it is the worst one: two people typing the same sentence
get different strategies, both confidently.

**Escalation.** Three draws by default. Any prompt whose draws disagree is
re-run to five, because a 2-1 split is weak evidence about which behaviour is
the outlier and the cases that disagree are exactly the ones worth spending on.
Stable prompts cost three calls and nothing more.

**The artifact is self-describing** and the gate rejects it when the versions it
was produced against no longer match, or when it is older than a week. A drift
run from three months ago proves nothing about today's schema and would
otherwise become a guarantee nobody re-earned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "strategy_families.json"
OUT = HERE / "drift.json"

STABLE_EXECUTABLE = "STABLE_EXECUTABLE"
STABLE_REFUSAL = "STABLE_REFUSAL"
STABLE_CLARIFICATION = "STABLE_CLARIFICATION"
UNSTABLE_SAFE = "UNSTABLE_SAFE"
UNSTABLE_EXECUTABLE = "UNSTABLE_EXECUTABLE"

#: How long a drift artifact may be cited for. Not a technical limit — the
#: provider can change under a fixed model id, so evidence about stochastic
#: behaviour has a shelf life whether or not anything in this repository moved.
VALID_FOR_DAYS = 7


def _plan_digest(reading) -> str:
    """What identifies the plan that would actually run.

    The canonical form of the compiled `ScenarioSpecification`, not the intent
    hash. The first version used `intent_hash` and reported the flagship
    sentence — *invest $500 monthly into VTI* — as execution-unsafe across
    three draws whose settled fields were identical. The hash covers more than
    the executed semantics, so it moves when nothing about the simulation does.

    That is the third time in this work a convenient proxy stood in for the
    property being measured, and each time it manufactured findings: carrier
    presence instead of refusal, field names instead of settled values, and now
    intent identity instead of plan identity. A measurement that reports
    working behaviour as broken cannot be trusted about the broken kind either.

    Falls back to the intent hash only when there is no compiled scenario,
    which cannot happen for an executable reading and is here so a future change
    that makes it possible does not silently digest `None`.
    """
    from hashlib import sha256

    compiled = reading.compiled
    scenario = getattr(compiled, "scenario", None)
    if scenario is None:
        intent = reading.intent
        return "no-plan:" + (intent.intent_hash[:16] if intent else "?")
    canonical = json.dumps(scenario.canonical_form(), sort_keys=True,
                           default=str)
    return sha256(canonical.encode()).hexdigest()[:16]


def _outcome(reading) -> dict:
    """One draw, reduced to what actually differs for a user.

    `identity` is what makes two draws the same draw. For an executable plan it
    is the compiled plan's digest, so two draws that both execute but execute
    *different strategies* compare unequal — the failure a comparison over
    outcome classes alone cannot see.
    """
    refusals = tuple(sorted(getattr(r, "dimension", "") for r in reading.refusals))
    questions = tuple(sorted(reading.questions))

    if refusals:
        return {"class": "REFUSAL", "identity": "refused:" + ",".join(refusals),
                "refusals": list(refusals)}
    if reading.executable:
        return {"class": "EXECUTABLE", "identity": "plan:" + _plan_digest(reading),
                "refusals": []}
    return {"class": "CLARIFICATION",
            "identity": "asks:" + ",".join(questions),
            "questions": list(questions), "refusals": []}


def classify(draws: list) -> str:
    identities = {d["identity"] for d in draws}
    if len(identities) == 1:
        one = draws[0]["class"]
        return {"EXECUTABLE": STABLE_EXECUTABLE,
                "REFUSAL": STABLE_REFUSAL,
                "CLARIFICATION": STABLE_CLARIFICATION}[one]
    # Disagreed. The only question left is whether any draw could have put a
    # figure in front of somebody.
    if any(d["class"] == "EXECUTABLE" for d in draws):
        return UNSTABLE_EXECUTABLE
    return UNSTABLE_SAFE


def _provenance(model_reader, syntax_reader, texts: list, draws: int) -> dict:
    """Everything needed to say whether this artifact still applies.

    Version fields are read from the objects that produced the run rather than
    restated here, for the reason `derived_from` exists in the manifest: a
    constant that has to be remembered is a constant that goes stale silently.
    """
    from datetime import datetime, timezone
    from hashlib import sha256

    sys.path.insert(0, str(HERE.parent))
    from shadow_run import schema_fingerprint                   # noqa: E402

    from src.discovery.hosted_recording import PROMPT_VERSION
    from src.discovery.pipeline import PIPELINE_VERSION
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.witnesses import BOTH

    # Who produced this, so a development run cannot stand in for the
    # scheduled lane. Read from the environment because that is the only place
    # that knows — this is a corpus script, not a serving consumer, and the
    # rule `deploy.context` enforces is about request handlers deciding where
    # their answers come from.
    import os

    on_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    return {
        "producer": "github-actions" if on_ci else "local",
        "workflow": os.environ.get("GITHUB_WORKFLOW", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "mode": ("longitudinal" if draws < 3
                 else os.environ.get("GITHUB_EVENT_NAME", "local")),
        "schema_fingerprint": schema_fingerprint(QUANTIFY_SCHEMA),
        "prompt_set_digest": sha256(
            "\n".join(sorted(texts)).encode()).hexdigest()[:16],
        "prompt_count": len(texts),
        "hosted_model_id": model_reader.id,
        "hosted_model": getattr(model_reader, "model", ""),
        "prompt_version": PROMPT_VERSION,
        "syntax_witness_version": getattr(syntax_reader, "id", "")
                                  or f"stanza@{getattr(syntax_reader, '_version', '?')}",
        "pipeline_version": PIPELINE_VERSION,
        "fusion_version": "quantify-fusion@1",
        "witness_profile": sorted(w.value if hasattr(w, "value") else str(w)
                                  for w in BOTH.available),
        "draws_per_prompt": draws,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def run(draws: int = 3, escalate_to: int = 5, texts=None) -> dict:
    """Every prompt, several times, in the profile a deployment serves.

    Also reports silent reductions *across draws*. Measuring them from a single
    recording is the same sampling error this lane exists to expose: on one
    draw `sell VTI and buy BND` carries `sell_action` and is refused, on the
    next it does not and executes. A gate reading a one-draw number would open
    or close on the luck of which recording was current.
    """
    sys.path.insert(0, str(HERE.parent.parent))

    from src.discovery.readers_quantify import configured_hosted_reader
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.syntax_stanza import StanzaReader
    from src.discovery.witnesses import BOTH
    from src.workspace.pilot import read

    cases = json.loads(CASES.read_text())["cases"]
    supported = {c["text"]: c["must_be"] == "RECOGNISED" for c in cases}
    texts = texts or [c["text"] for c in cases]
    model = configured_hosted_reader()
    if not model.available():
        raise SystemExit(f"{model.api_key_env} is not set; this lane calls the "
                         "provider by design and cannot run without it")
    syntax = StanzaReader("en")

    def draw(text: str) -> dict:
        reading = read(text, model, schema=QUANTIFY_SCHEMA, profile=BOTH,
                       syntax_reader=syntax)
        return _outcome(reading)

    results, by_class = [], {}
    for index, text in enumerate(texts, start=1):
        got = []
        for _ in range(draws):
            try:
                got.append(draw(text))
            except Exception as failure:                        # noqa: BLE001
                # Recorded, never dropped. A draw that vanished would make an
                # unstable prompt look stable by having fewer opinions.
                got.append({"class": "ERROR", "identity": f"error:{failure!r}"[:80],
                            "refusals": []})

        verdict = classify(got)
        if verdict in (UNSTABLE_SAFE, UNSTABLE_EXECUTABLE) and escalate_to > draws:
            for _ in range(escalate_to - draws):
                try:
                    got.append(draw(text))
                except Exception as failure:                    # noqa: BLE001
                    got.append({"class": "ERROR",
                                "identity": f"error:{failure!r}"[:80],
                                "refusals": []})
            verdict = classify(got)

        # A silent reduction on *any* draw: the family is unsupported and this
        # draw produced an executable plan for it anyway.
        reduced = [i for i, d in enumerate(got, start=1)
                   if not supported.get(text, False) and d["class"] == "EXECUTABLE"]
        results.append({"text": text, "classification": verdict,
                        "draws": got,
                        "silently_reduced_on_draws": reduced,
                        "distinct_outcomes": sorted({d["identity"] for d in got})})
        by_class[verdict] = by_class.get(verdict, 0) + 1
        print(f"  [{index:2}/{len(texts)}] {verdict:22} {text[:48]}")

    return {
        "schema": "quantify-drift-lane@1",
        "provenance": _provenance(model, syntax, texts, draws),
        "by_classification": by_class,
        "execution_unsafe": [r["text"] for r in results
                             if r["classification"] == UNSTABLE_EXECUTABLE],
        "silently_reduced_any_draw": [r["text"] for r in results
                                      if r["silently_reduced_on_draws"]],
        "gate_note": (
            "The pre-Lean gate requires zero UNSTABLE_EXECUTABLE here and zero "
            "SILENTLY_REDUCED in strategy_closure.json. Discovery need not be "
            "deterministic; it must stop nondeterminism from changing what "
            "executes without anybody noticing."),
        "results": results,
    }


#: The daily lane's sentences: the boundaries that have actually broken.
#:
#: The daily run used to re-ask all 36 families, which is a statistically
#: meaningless sample repeated at cost. Its question is not "how stable is the
#: reader" — that is Monday's three-draw run — but "has the provider started
#: violating a boundary we already know is dangerous". Those are different
#: questions and only the second needs asking every day.
#:
#: Every entry earned its place by having changed executable meaning or exposed
#: reader instability at least once. A sentence nobody has ever seen go wrong
#: is not a canary; it is a bill.
CANARY = [
    # Crossing versus persistent. Conflating an event with a state changes how
    # often a strategy fires, and these two are the pair that showed it.
    "buy VOO when SPY falls below its 200-day moving average",
    "add to BND while TLT is under its 200-day",

    # Buy/sell role inversion. Getting the roles the wrong way round produces
    # a formally perfect calculation of the opposite portfolio.
    "sell VTI and buy BND",

    # Negation. The `rather than` case reads backwards under the current
    # reader — `assets='ETF'` for a sentence rejecting the ETF — accepted on a
    # single witness because syntax was silent.
    "buy the index rather than through an ETF",
    "do not reinvest the dividends",

    # Asset omission. The dominant real-world shape: everything stated except
    # what to hold.
    "a 60/40 portfolio",

    # Cadence attachment, and a live incomplete-refusal finding.
    "rebalance back to 60/40 every year",

    # Structural relations, where a flattened mapping loses the whole request.
    "hold the bonds in the IRA and the stocks in the taxable account",
    "keep three years of expenses in cash and the rest in stocks",

    # The withdrawal case that refuses for the wrong capability under the
    # current reader, and the factor tilt that once reduced silently.
    "take $40,000 a year out of the portfolio",
    "tilt 20% toward small cap value",
]


def main(argv: list) -> int:
    draws = 1 if "--longitudinal" in argv else 3
    if "--draws" in argv:
        draws = int(argv[argv.index("--draws") + 1])
    escalate = draws if draws == 1 else 5

    # `--canary` narrows the sentences, never the draws. A daily run that also
    # cut draws would answer neither question: one draw of ten prompts says
    # nothing about stability, and it would still cost money to say it.
    texts = CANARY if "--canary" in argv else None
    report = run(draws=draws, escalate_to=escalate, texts=texts)
    # `scope` only. `prompt_count` is already set by `run()` from the sentences
    # it actually asked about, and overwriting it here with `None` for full
    # runs would have blanked a field the gate compares.
    report["provenance"]["scope"] = "canary" if texts else "full"
    out = OUT if (draws > 1 and not texts) else OUT.with_name(
        "drift_canary.json" if texts else "drift_longitudinal.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{out.name}")
    for name, total in sorted(report["by_classification"].items()):
        print(f"  {name:22} {total}")
    unsafe = report["execution_unsafe"]
    print(f"\nexecution-unsafe instability: {len(unsafe)}")
    for text in unsafe:
        print(f"    {text[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
