"""How each strategy family fares, measured rather than asserted.

    python corpus/parser/strategy_closure.py          # writes strategy_closure.json
    python corpus/parser/strategy_closure.py --print  # and shows the table

**What is measured is the refusal, not a dimension.** The first version scored
a case CARRIED when one nominated dimension appeared in the reading. That is a
proxy for the thing that matters and it was wrong in both directions: it called
asset location understood because `account_type` happened to appear, and it
called a withdrawal unhandled when the model had read
`objective=assess_withdrawal` and Mission would refuse on it. The rule this
tier exists to enforce is:

    if the engine cannot model the semantic, Discovery must preserve enough of
    it for Mission to refuse it BY NAME

so the question asked of each case is now `refusals_for(reading)` — does asking
Mission about what was read produce a refusal. One state per case:

    REFUSED             Mission refuses this, by name. Success for an
                        unsupported family regardless of which dimension
                        carried it.

    EXECUTABLE          the family is supported and Mission raises no
                        objection. Success for a supported family.

    SILENTLY_REDUCED    something was read, the family is unsupported, and
                        Mission refuses nothing. The sentence produces a plan
                        built from whatever fragment survived, and every
                        surviving fragment in this build is
                        accumulation-shaped. This is the dangerous state: no
                        refusal, no approximation flagged as one, and a figure
                        at the end of it.

    NOTHING_READ        nothing at all was recognised. Honest failure. The
                        person is told the sentence could not be read, which is
                        wrong-but-safe rather than wrong-and-confident.

`SILENTLY_REDUCED` is ranked worse than `NOTHING_READ` on purpose. A sentence
that produces nothing sends somebody back to rephrase; a sentence that produces
the wrong plan sends them away with a number.

**What this measures.** Everything Mission would be asked about on the serving
path: the model's readings, the relation kinds folded in, and Quantify's own
deterministic readers run the way `pilot.read` runs them — with no parse,
because the deployment declares no syntax witness.

The derived readers were absent from this for one release and the omission was
invisible, because while every one of them needed a parse they could not run in
production either, so measuring without them measured the truth. The moment the
family readers were made to read the text, this lane went on reporting a silent
reduction for a sentence the serving path refuses by name — an instrument
describing a configuration that had stopped existing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "strategy_families.json"
OUT = HERE / "strategy_closure.json"

REFUSED = "REFUSED"
EXECUTABLE = "EXECUTABLE"
SILENTLY_REDUCED = "SILENTLY_REDUCED"
NOTHING_READ = "NOTHING_READ"

#: Mission cannot build the plan because the sentence did not say something it
#: needs — no holding, no amount — rather than because it named a semantic this
#: build will not run. A question, not a refusal.
#:
#: Split out when the lane started asking the serving path instead of
#: reconstructing its answer. The reconstruction called `refusals_for` on the
#: model's declared dimensions, and a dimension nobody stated is not declared,
#: so "contribute $1,000 a quarter" — which names no holding — scored
#: EXECUTABLE. It is not: the serving path asks which holdings were meant, and
#: it is right to. Counting that as REFUSED would report five false refusals of
#: supported families; counting it as EXECUTABLE was the flattering error it
#: replaced.
NEEDS_INPUT = "NEEDS_INPUT"

#: An unsupported family that produced no plan and no named refusal: the
#: sentence became a question instead.
#:
#: Weaker than SILENTLY_REDUCED and not the same thing, which matters because
#: the gate's condition is about plans. `SILENTLY_REDUCED` means "a plan is
#: produced and it is the wrong one", and these produce none — calling them
#: that would over-report by the same margin that calling them REFUSED would
#: under-report.
#:
#: Still a finding. "rebalance whenever an allocation drifts more than 5
#: points" is asked about rather than refused, and the dimension it would be
#: refused on — `periodic_rebalancing` — is EXECUTED in the manifest, so it is
#: the *value* that cannot run and an empty reading produced no value to
#: refuse. Somebody answering the question gets a plan the engine can run and
#: not the strategy they described, one step later than a silent reduction.
ASKED_NOT_REFUSED = "ASKED_NOT_REFUSED"

#: The schema has no dimension for this, so recognition cannot be the problem
#: and cannot be the fix. Scored apart from the rest because counting it as a
#: recognition defect would send the work to the wrong layer.
SCHEMA_GAP = "SCHEMA_GAP"


def _read(reader, schema, text: str) -> dict:
    """What Mission would be asked about, dimensions and relations together.

    Relation kinds are folded in through the serving path's own helper rather
    than a copy here. The first version read only `result.readings`, so a model
    that returned `asset_location(holding=REITs, account=Roth)` — exactly right
    — was scored as having silently reduced the sentence. The measurement was
    blind to the half of the reading that carried the answer, which is the same
    defect as the code it was measuring.
    """
    result = reader.read(text, schema)
    if getattr(result, "failed", ""):
        return {}

    from src.discovery.derived_readers import DERIVED_READERS
    from src.workspace.pilot import _relation_fields

    # Quantify's own deterministic readers, on the same terms the serving path
    # runs them: candidates and parse absent, because the deployment declares
    # no syntax witness and `pilot.read` calls them exactly like this.
    #
    # Without them this measured a surface narrower than the one served, while
    # its docstring said it measured "what Mission would be asked about". That
    # was true while every derived reader needed a parse and none could run in
    # production. It stopped being true when the family readers were made to
    # read the text, and the lane went on reporting a silent reduction for a
    # sentence the serving path refuses by name.
    derived = {}
    for _reader_id, derive in DERIVED_READERS:
        found = derive((), None, text)
        if found is not None:
            derived[found.dimension] = found.value

    return {**{r.dimension: r.value for r in result.readings},
            **_relation_fields(result),
            **derived}


#: The reader a deployment actually serves. `compiler` is retained only as a
#: historical comparator: nothing in `src/` constructs `CompilerReader`, and
#: `test_strategy_families` asserts that structurally.
SERVING = "hosted"


def _witness(name: str):
    """Either reader, so the same measurement can be run against both.

    The whole point of running it twice: 11 silent reductions against the
    deterministic reader is a finding about `quantify-compiler@2`, which is the
    reader being deleted. Whether it is also a finding about Discovery is a
    different question with a different answer and a different fix.
    """
    if name == "hosted":
        from src.discovery.hosted_recording import RecordedHostedReader

        return RecordedHostedReader()

    from src.discovery.readers_quantify import CompilerReader

    return CompilerReader()


def _serving(reader, schema, text: str):
    """What the serving path read, and what it refused — asked of it directly.

    This used to reconstruct the answer: read the model, fold in relations,
    and call `refusals_for` on the result. That reconstruction drifted twice in
    one sitting. It first omitted Quantify's derived readers, and then, once
    those were included, it still omitted the syntax guards — so it reported a
    silent reduction for `annuitize a third of the portfolio at 70` while the
    serving path refused it by name, because the guard that proves the
    predicate runs in `pilot.read` and not here.

    A reconstruction of a path is a second implementation of it, and it goes
    stale exactly the way a second implementation does. So the question is put
    to the path itself: `pilot.read` with the profile the deployment declares.

    `refusals_for` stays imported by `measure` for the compiler comparator,
    which has no serving path to ask.
    """
    from src.deploy.context import current
    from src.workspace.pilot import read

    profile = current().model.witnesses
    syntax = None
    if getattr(current().model, "syntax_witness", False):
        from src.workspace.pilot_routes import configured_syntax_reader

        syntax = configured_syntax_reader()

    reading = read(text, reader, schema=schema, profile=profile,
                   syntax_reader=syntax)

    declared = {f.field: f.value for f in reading.settled}
    declared.update({r.dimension: getattr(r, "stated_value", "") or ""
                     for r in reading.refusals})
    return declared, tuple(reading.refusals)


def measure(witness: str = SERVING) -> dict:
    sys.path.insert(0, str(HERE.parent.parent))

    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.mission.capability import refusals_for

    document = json.loads(CASES.read_text())
    reader = _witness(witness)

    results, by_state, by_family = [], {}, {}
    for case in document["cases"]:
        # The serving witness is asked; the compiler comparator is
        # reconstructed, because there is no serving path to ask it about.
        #
        # Nothing in `src/` constructs `CompilerReader` — `test_strategy_
        # families` asserts that structurally — so `pilot.read` would be
        # measuring a reader through a path that reader never takes. Its
        # numbers are a frozen defect report whose whole worth is
        # comparability, and this file's own history records the number moving
        # twice for reasons that were not the compiler. Routing it through the
        # new path moved it from 17 to 1 and would have been a third.
        if witness == SERVING:
            got, refusals = _serving(reader, QUANTIFY_SCHEMA, case["text"])
        else:
            got = _read(reader, QUANTIFY_SCHEMA, case["text"])
            refusals = refusals_for(got)

        # A refusal that names a semantic, as distinct from a request for
        # something the sentence never said. `kind` is Mission's own
        # distinction and is not re-derived here.
        semantic = tuple(r for r in refusals
                         if getattr(r, "kind", "") != "UNRESOLVED_INPUT")
        wanted = tuple(r for r in refusals
                       if getattr(r, "kind", "") == "UNRESOLVED_INPUT")
        supported = case["must_be"] == "RECOGNISED"

        if not case["carriers"] and not supported:
            state = SCHEMA_GAP
        elif semantic:
            # A named refusal. Success for an unsupported family, and a real
            # finding for a supported one.
            state = REFUSED
        elif wanted:
            # The sentence did not say something the plan needs. For a
            # supported family that is the product working — "contribute
            # $1,000 a quarter" names no holding and being asked is correct.
            # For an unsupported one it is not enough: a question is not a
            # refusal, and answering it would produce the wrong plan.
            state = NEEDS_INPUT if supported else ASKED_NOT_REFUSED
        elif supported:
            state = EXECUTABLE
        elif got:
            state = SILENTLY_REDUCED
        else:
            state = NOTHING_READ

        results.append({**case, "state": state, "read": got,
                        "refused": [f"{r.dimension}={r.stated_value!r}"
                                    for r in semantic],
                        "asked": [r.dimension for r in wanted]})
        by_state[state] = by_state.get(state, 0) + 1
        family = by_family.setdefault(case["family"], {})
        family[state] = family.get(state, 0) + 1

    reduced = [r for r in results if r["state"] == SILENTLY_REDUCED]
    return {
        "schema": "quantify-strategy-closure@1",
        "witness": reader.id,
        "witness_note": (
            "The serving path, asked directly rather than reconstructed: the "
            "hosted reader and the deterministic syntax witness, in the "
            "profile the deployment declares. Production serves BOTH — the "
            "guards that prove a dropped predicate run only on that branch, "
            "and until Stanza shipped they had never run for a user."),
        "count": len(results),
        "by_state": by_state,
        "by_family": by_family,
        "silently_reduced": [
            {"id": r["id"], "text": r["text"],
             "should_carry": "/".join(r["carriers"]), "read_instead": r["read"]}
            for r in reduced],
        "cases": results}


def main(show: bool = False, witness: str = "compiler") -> int:
    report = measure(witness)
    out = OUT if witness == SERVING else OUT.with_name(
        "strategy_closure_compiler.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"{report['count']} cases, witness {report['witness']}")
    for state in (REFUSED, EXECUTABLE, SILENTLY_REDUCED, NOTHING_READ,
                  SCHEMA_GAP):
        print(f"  {state:18} {report['by_state'].get(state, 0)}")

    if show:
        print("\nsilently reduced — a plan is produced and it is the wrong one:")
        for one in report["silently_reduced"]:
            print(f"  {one['text'][:58]:60}")
            print(f"    should carry {one['should_carry']:22} "
                  f"read instead {one['read_instead']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(
        show="--print" in sys.argv,
        witness="compiler" if "--compiler" in sys.argv else SERVING))
