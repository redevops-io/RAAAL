# Rules for evidence

Four rules, each written after a defect that the rule would have caught. They
are about the evidence rather than the runtime, because every one of these
failures leaves the test suite green — that is what makes them worth writing
down rather than remembering.

## 1. A regression set may grow. It may not silently shrink.

Once a case enters a regression set because it demonstrated a property, its
disappearance is an event that requires an explicit disposition. Removing it is
not a fix, and neither is letting it fall out.

The failure this closes is subtle because nothing in it looks like a deletion.
`tests/test_semantics.py` derives its case list from `closure.json`, which is
regenerated: cases whose two readers agree are collected, and cases that
disagree simply are not. So a case that *stops* agreeing does not fail. It
stops being collected, and the suite goes green with one fewer thing tested.

It happened. Re-recording under schema `@6` moved
`sema-window-moving_average-013` from AGREE to DISAGREE and moved two `day_rule`
cases the other way. The total went 41 → 42. A number went up, the suite passed,
and a case had left the tested set underneath it.

The shape of the fix, wherever this pattern occurs:

    corpus/parser/answerable.json     the recorded set, committed
    LEFT_THE_ANSWERABLE_SET           the only way out, one reason per entry
    a staleness test                  an entry that starts passing must be removed

The exception list is the boundary. An entry in it is a decision somebody made
and signed; an empty denominator is not.

**Where this is applied.** The semantics tier
(`corpus/parser/answerable.json`), and the strategy benchmark
(`corpus/benchmark/recorded_prompts.json`). Both are evaluators whose case
lists are derived rather than declared.

## 2. An answer key may not be read off the system it grades.

An expectation produced by running the runtime and writing down what it said
measures self-consistency. It does so while looking exactly like evidence,
which is why it cannot be caught downstream — every result agrees with every
other result.

In the harvested corpus this is enforced by vocabulary. The material-semantic
concepts are named `how often money goes in`, not `cadence`, and a test asserts
the two vocabularies do not overlap. Naming them identically is how the
shortcut gets taken without anyone deciding to take it.

The mapping between the vocabularies is written out in `MAPS_TO`, where it can
be argued with. It was wrong twice, both times manufacturing findings *against*
the runtime — one caught by checking a finding before reporting it, one by a
test requiring every mapped name to exist in the schema. An answer key is not
trustworthy because it is independent; it is trustworthy because it is
independent and checked.

## 3. A metric must say when its own number is misleading.

Material-semantic survival currently reads 18/18. It is not a product claim and
must not become one: no attested sentence reached a plan, so nothing could be
reduced, and the denominator contains only the cases that got as far as the
comparison.

`survival.json` carries that caution in the artifact, and a test requires the
caution to be present whenever nothing runs. The reason it lives in the
artifact rather than in a person's memory is that the artifact is what gets
quoted.

The same rule produced the benchmark's `UNSTABLE_SAFE` category. It was
introduced in the change that took dangerous instances to zero, which is
exactly the circumstance in which a new category deserves distrust — so three
tests require the downgraded finding to stay in the queue, require that no
`UNSTABLE_SAFE` pair has two executable sides, and pin the dangerous count to
the taxonomy.

## 4. Missing is not zero.

    missing material quantity   ->  unresolved
    explicitly zero quantity    ->  zero

Zero is a substantive instruction. Absence is the lack of one. Code that writes
`value or 0` has decided they are the same thing, and the decision is invisible
at the call site.

This project has now made that mistake four times: two worksheet templates
rendering an undefined return as `+0.00%`, a compiler defaulting an unreadable
`$1k` to zero, and a compiler defaulting an unstated amount to zero on a
recurring cadence — which produced the only attested sentence that executed,
holding I-Bonds annually and contributing nothing.

It was then nearly made in the opposite direction: the first version of the
recurring-cadence check refused an explicitly stated `$0`, rejecting something
the person had said in order to prevent something they had not.

Materiality is contextual, which is why this is a rule about *material*
quantities rather than a global "required fields" list. An amount may be
legitimately absent for a one-off or evaluation request; `every year` asserts a
recurring action whose quantity must be settled before anything executes. That
is the `seal()` result-changing rule doing its job — a field is required when
its absence would change the result, not because a schema said so.
