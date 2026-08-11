# Strategy benchmark — improvement queue

Run against `Baseline-v1`. Not a pass rate: an unsupported strategy refused by
name is correct, and a score counting executions would reward the silent
reduction this project spent months removing.

    prompts                103
    correct executions     30
    correct refusals       37
    correct clarifications 8
    dangerous instances    12

| finding | instances | severity | layer | area |
|---|---:|---|---|---|
| silent reduction | 3 | critical | Mission | momentum-rotation |
| false claim of support | 2 | high | Mission | swr-fixed-amount |
| false claim of support | 1 | high | Mission | rebalance-annual |
| false claim of support | 1 | high | Mission | glidepath |
| silent reduction | 1 | critical | Mission | factor-tilt |
| unstable execution | 1 | high | Fusion | momentum-rotation |
| wrong executable meaning | 1 | critical | Fusion | moving average versus a holding period |
| unstable execution | 1 | high | Fusion | thousands shorthand |
| unstable execution | 1 | high | Fusion | holding order, equal weight |
| unnecessary question | 2 | medium | Discovery | ma-cross-below |

## What each one is

**Rotation executes as buy-and-hold.** "each month hold whichever of VTI and
BND performed best" reads as two holdings and a monthly cadence, and executes.
The selection — and the selling that a rotation implies — is gone. The syntax
presence guards did not fire because no disposal verb appears in the sentence:
`rotate`, `hold whichever`, and `the stronger of` are not in the guard's lemma
set. This is the silent-reduction class alive in a family the earlier corpus
did not contain.

**Fixed-amount withdrawal refused for the wrong capability.** "withdraw $40,000
a year" is refused, which is right, but not for `sell_action`. A refusal naming
the wrong capability tells somebody the system cannot do a thing it in fact
never considered.

**Order and shorthand change the compiled plan.** `VTI and BND` against `BND
and VTI` for an equal-weight strategy, and `$1,000` against `$1k`, produce
different plan digests. Whether the *figure* differs is a separate question;
that the plan identity moves on surface form is the finding, because plan
identity is what the replay property rests on.

**Moving average against holding period.** "buy VTI below its 200-day moving
average" and "hold VTI for 200 days" compile to one plan. A threshold and a
duration are not the same instruction.

## What this does not authorise

The queue is a counterexample generator, not a fifth reopen trigger. A wrong
executable meaning and a silent reduction already activate the first two
triggers. An unsupported strategy appearing repeatedly is evidence for the
fourth — counted demand — and not a decision on its own.
