# Baseline v1 — semantic correctness complete

A fixed reference point, written the day before the closed pilot opens, so
that six months of user feedback can be measured against something rather than
remembered against nothing.

    build in production at close of engineering   cf596f4
    build deployed for the pilot                  (this document's commit)
    licence                                       AGPL-3.0-or-later
                                                  with Commons Clause
    data policy                                   SYNTHETIC_ONLY

    suites          SQLite 3,609 passed / 280 skipped
                    PostgreSQL 3,889 passed
    corpus          35 strategies — 14 RESULT, 21 REFUSAL,
                    0 PRODUCT_ERROR, 0 HARNESS_ERROR, 16 need a human

## The closure criterion

Not "the corpus is green". The stronger one:

> Every semantic defect found in this slice either executes faithfully, or
> prevents a financial figure from being published while naming the mismatch.

Which gives the failure boundary the product now has:

    understands and executes faithfully   → shows a result
    understands, cannot represent         → refuses, by name
    material semantic disagreement        → asks
    unknown or unsupported domain         → readable refusal

And the state that must never return:

    the user said X
      → the system quietly executed Y
      → coverage reported 1/1
      → a figure appeared

## What was closed

Eleven product findings, then five more after the baseline was first declared
complete. The later five all had one shape — a semantic dimension the compiler
cannot represent falling out of the coverage denominator, so a figure was
published for a strategy nobody described.

| element | what was being lost |
|---|---|
| `evaluation_period` | a stated period replayed over the whole snapshot |
| `event_triggered_funding` | a conditional purchase whose reading was unsettled |
| `scheduled_funding` | a second funding mode dropped in silence |
| `sell_action` | an exit leg discarded |
| `conditional_amount` | "double it when…" ignored, base plan reported |
| `allocation_method` | inverse volatility executed as equal weight |
| `periodic_rebalancing` | a rebalancing schedule with nowhere to go |
| `stated_weights` | a 60/40 portfolio executed as 50/50 |

Plus, on the compiler itself: crossing versus persistent trigger semantics
(4.6× the money), cadence versus evaluation window, cadence versus rebalancing
frequency, and execution timing honoured-or-refused rather than overwritten.

## Known architectural debt

**Compiler-derived semantic inventory — validated, P2.** `coverage.assess`
enumerates supported constructs by hand. Five entries were added in a single
slice, which is the evidence that the list is structurally wrong rather than
merely incomplete. Implement when a sixth would otherwise be needed.

**`ReaderDecision` — triggered candidate.** A reader that has *declined* to
read a field is indistinguishable from one that never saw it, so a second
reader can reintroduce exactly what the first rejected. Three instances.
Bounded span and context checks are holding; revisit when a third field needs
one.

**The pressure behind both, in one line:**

> Absence is not always ignorance. Sometimes it is a deliberate rejection, and
> collapsing the two lets another reader reintroduce what was rejected.

## What found the defects

The record is unambiguous and worth keeping, because it contradicts where the
effort would naturally go.

| lane | found |
|---|---|
| ~3,900 unit and integration tests | regressions in fixes — not one of the semantic defects |
| mutation testing | defects in the tests themselves, repeatedly |
| SQLite / PostgreSQL parity | none |
| provenance and replay verification | none |
| deterministic strategy corpus | 3 product defects in under two minutes |
| browser journeys | the rule-description mismatch that opened the whole line |
| reading a rendered page | five, including every one after the baseline |

The suite was not inadequate. It was answering a different question. The whole
stack agreed with itself — parser, compiler, engine, ledger, tests — and agreed
on the wrong reading of English. Internal consistency cannot detect a shared
misunderstanding of the input.

    implementation is verified by the suite
    meaning is verified by an observer reading the output

The permanent shape, where each level answers a question the one above cannot:

    unit and integration
      → mutation
        → database parity
          → deterministic semantic corpus
            → browser journeys
              → human acceptance
                → pilot users

## Where the corpus is blind

Recorded because a green corpus should not be read as coverage it does not
have. No corpus strategy states bare percentage weights, which is exactly
where `stated_weights` lived — that fix was verified by direct probes, and the
corpus only proved nothing else moved. Percentage-allocation prompts should
join the set.

## What the pilot is for

The remaining uncertainties are no longer whether the runtime executes the
strategies it claims to. They are whether people understand the questions,
trust the refusals, and describe strategies in ways nobody here anticipated.
That evidence cannot be produced by another internal pass.
