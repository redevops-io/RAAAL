# Strategy benchmark — improvement queue

Run against `Baseline-v1`. Not a pass rate: an unsupported strategy refused by
name is correct, and a score counting executions would reward the silent
reduction this project spent months removing.

    prompts                103
    correct executions     31
    correct refusals       45
    correct clarifications 6
    dangerous instances    2      (was 12)

| finding | instances | severity | layer | area |
|---|---:|---|---|---|
| incomplete refusal | 2 | high | Discovery | rebalance-annual |
| unstable execution | 1 | high | Fusion | thousands shorthand |
| unstable execution | 1 | high | Fusion | holding order, equal weight |
| unnecessary question | 1 | medium | Discovery | ma-cross-below |

## What each one is

**A refusal that names one of the two things asked for.** "rebalance back to
60/40 every year" is refused for `stated_weights` and never mentions
`periodic_rebalancing`. The person is told one part of their sentence is
unsupported and left believing the other part was fine. An incomplete refusal
is more misleading than a blunt one, because its specificity reads as
completeness.

**Order and shorthand change the compiled plan.** `VTI and BND` against `BND
and VTI` for an equal-weight strategy, and `$1,000` against `$1k`, produce
different plan digests. Whether the *figure* differs is a separate question;
that the plan identity moves on surface form is the finding, because plan
identity is what the replay property rests on. This is the canonicalization
boundary: `VerifiedIntent` identity may legitimately retain the original
expression, and execution identity must not.

**A question about an asset the sentence names.** "whenever VTI crosses under
the 200 DMA, invest $1,000" asks which assets to hold. VTI is the observed
asset and the held one, and reading the first without the second is a gap in
binding rather than a genuine ambiguity.

## Closed since the first run

**Rotation executed as buy-and-hold** (3 instances, critical). "each month hold
whichever of VTI and BND performed best" read as two holdings and a monthly
cadence and *executed* — the selection, and the selling a rotation implies,
silently gone.

Not closed by adding `rotate`, `hold whichever` and `the stronger of` to the
syntax guard's lemma set. Those words are witnesses of a missing semantic, not
the semantic itself, and a guard built from them passes the benchmark while the
next synonym reduces silently. Closed by giving Discovery a `selection_rule`
dimension that can represent the concept, and Mission a `NOT_MODELLED` entry
that refuses it by name.

**Moving average against holding period** (1, critical). "buy VTI below its
200-day moving average" and "hold VTI for 200 days" compiled to one plan.
Discovery told them apart; Mission dropped both. Closed by `holding_period`
plus a general rule: any dimension Discovery settles that no part of the
compiler consults is refused as `UNSUPPORTED_DIMENSION` rather than discarded.

That general rule immediately found a **second silent reduction nobody was
looking for** — a stated cadence beside an event trigger. "contribute $500
monthly when VTI crosses below its average" built an event-triggered schedule
and threw the *monthly* away. Recorded in
`tests/test_mission_from_intent.py::test_a_calendar_stated_beside_a_trigger_is_refused_not_dropped`.

**False claims of support** (4, high) and **factor-tilt silent reduction** (1).
Absorbed by the same general rule and by widening the corpus's declared
expectations where the corpus, not the system, was wrong.

## Open, and deliberately not fixed yet

**`moving_average_window` has no unit.** Syntax reads `12` from "the 12-month
moving average"; the hosted reader reads `252`, the same twelve months in
trading sessions. Both are defensible readings of a field that never says what
it counts, and fusion refuses to settle it — the safe outcome, and the reason
this is not on the dangerous list. Fixing it means giving the dimension a unit,
which is a schema change; held until the harvested corpus says which units real
language actually uses. Tracked in `LEFT_THE_ANSWERABLE_SET` in
`tests/test_semantics.py`.

## What this does not authorise

The queue is a counterexample generator, not a fifth reopen trigger. A wrong
executable meaning and a silent reduction already activate the first two
triggers. An unsupported strategy appearing repeatedly is evidence for the
fourth — counted demand — and not a decision on its own.
