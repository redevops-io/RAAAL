# Authority is earned per semantic claim, not per parser

Stanza is not authoritative. The hosted model is not authoritative. Even
`quantify-trigger-semantics@1` is not *generally* authoritative.

A deterministic derivation may author a contract field only inside the grammar
its contract recognises and its falsification suite constrains. Outside that
envelope it returns no claim at all.

This is the rule that stops a narrow reader becoming a compiler one field at a
time, which is how `quantify-compiler@2` happened and why deleting it took
months.

## What "no privileged reader" does and does not mean

    it means      provenance does not settle a disagreement
    it does not   every material fact needs two witnesses to exist

The second reading is the tempting one and it is wrong in an expensive way. If
a field may only be settled when two independent readers speak, then stochastic
model participation becomes *mandatory* for understanding sentences whose
grammar states the answer outright — and the product asks a follow-up question
every time a provider happens to omit a dimension.

That is not hypothetical. The live drift lane measured "buy VOO when SPY falls
below its 200-day moving average" executing on two draws of five and asking on
the other three, purely because the hosted reader emits `trigger_semantics`
inconsistently. Neither obvious response was acceptable:

- **Always ask.** Converges, and turns a supported journey into a follow-up on
  every event-triggered sentence. Nineteen tests failed when it was tried.
- **Let syntax carry the field.** Converges, and makes the parser an authority
  on meaning.

The third way is to name the thing in between: a reader with an id, a version,
a contract naming exactly one field, and a falsification suite that says where
it must stay silent.

## Deterministic is not the same as authoritative

The falsification suite is what converts a derivation into a reader, and it
earned that role immediately by finding two defects worse than the feature.

**Negation.** `"buy VOO when SPY does not fall below its 200-day moving
average"` derived `crossing_event`. A dynamic verb and a comparison
preposition were both present and nothing looked at `not`.

That defect had existed for as long as the derivation had, and it was
*harmless* — because the derivation was evidence, and evidence never carried a
field. Promoting the same rule to authority would have converted a latent
evidence bug into an authoritative inverted trigger: the plan fires on exactly
the condition the sentence excludes.

> A derivation's defects are bounded by its authority. Granting authority
> retroactively promotes every one of them.

**Hidden candidates.** `"crosses below and stays below"` was supposed to be
caught by two readings disagreeing. It was not. The level binds to one
governing verb, so exactly one family fires and a sentence carrying both
readings arrives looking unanimous.

> Candidate agreement is insufficient when the extraction can hide a candidate.

The check moved to the parse, which is the only place both verbs are visible.

## The contract this reader holds

    id            quantify-trigger-semantics@1
    authors       trigger_semantics, and nothing else
    speaks when   the grammar states a transition or a state, unambiguously
    declines when both readings appear, the clause is negated, or neither fires

Fusion then weighs its claim against the hosted reader's by the ordinary rules:

    both agree                     settle
    derived speaks, model silent   settle
    both speak and disagree        ask
    neither speaks                 ask

The restriction to one field is asserted from the AST rather than remembered,
and the id is versioned because a derivation whose rules changed under a fixed
id would make two runs look comparable when they are not.

## The second one: `quantify-weight-binding@1`

Authors `stated_weights`, and the field it authors is the argument for it. The
hosted reader returns the split — `60/40` — which is the whole fact for anybody
reading the sentence and half of it for anything executing one. Which holding
takes which share is not in that value, and the engine divides each purchase by
weights it has to be able to attach.

So this reader reads the pairing off the sentence: a percentage, at most a
preposition, then the holding. `60% in VTI and 40% in BND` binds; `a 60/40
portfolio` does not, and neither does a ratio sitting beside a list of
instruments. That silence is the point. Pairing positionally would mean
deciding that the first number belongs to the first instrument, and getting it
backwards runs 40/60 under the name 60/40 — a wrong executable meaning, on a
figure nothing downstream can check, because both readings produce a perfectly
ordinary number.

It reads the **text**, not the parse, and that is deliberate rather than
convenient. The deployment that serves users has no deterministic parser
installed; a reader that needed one would be correct in the suite and absent in
production, which is the shape of every gap this project has found in its own
deployment.

Where it and the hosted reader agree, fusion now keeps *its* value rather than
the model's. That is not authority over a disagreement — `same_value` has
already established the two are the same reading — it is keeping the one that
carries the binding, because settling the model's `60/40` discarded it and left
the compiler refusing a split it had just been handed.

## Adding another one

The same shape generalises — `from IRA → to Roth` could feed an
`AccountTransitionReader` while generic parser output stays evidence. The bar
is the falsification suite, not the plausibility of the rules:

1. Name the single field it may author, and assert it structurally.
2. Write the cases where it must **decline** before the ones where it decides.
   A falsification set in which everything resolves proves the reader answers,
   not that it knows when not to.
3. Include negation, coordination, and any construction where the extraction
   could hide a competing reading.
4. Prove that deleting the reader reintroduces the instability it was added
   for. A reader that changes nothing when removed is one nothing depends on.


## The third one: `quantify-day-of-month@1`

Authors `day_rule`, and it exists because the vocabulary could not say what
somebody wrote. The schema offers three day rules — first session of the
period, last session, and a calendar-first variant — and none of them can
express "the 15th".

So a person who wrote *"I invest $200 into NVDA every month, on the same day
each month — the 15th for the past 5 years"* had it read as
`calendar_first_rolled_forward`, the **first** of the period, and was then
refused for asking for a rule this build does not run. They had not asked for
it. That is worse than a refusal and worse than a silence: a wrong reading
arrived wearing a refusal's clothes, and the record showed them requesting a
plan they never described.

The hosted reader was not being careless. Asked for a value from a closed
vocabulary that has no word for the thing in the sentence, it answered with the
nearest thing it could say. The gap was representational, which is the same
diagnosis that produced schema `@4`.

It reads an **ordinal** — `the 15th`, `on the 3rd` — because every neighbouring
dimension in these sentences is a bare number and the cost of confusing them is
money landing on a date nobody named:

    $200 into NVDA                an amount
    the past 5 years              an evaluation period
    its 200-day moving average    a window
    every month                   a cadence

None of those wears an ordinal suffix. It is silent on two ordinals — "the 1st
and the 15th" is twice a month, not a day — and silent on "the 1st **trading**
day", which is the first-session rule this build already executes and which
this reader must not overwrite.

The engine executes it. A named day lands on the first session on or after that
date, rolled forward off a closed market, because the 15th is a weekend about
two months in seven. A month that never reaches the day — the 31st of a
thirty-day month — takes that month's last session rather than rolling into the
next one, because "monthly" means once a month and landing late within it keeps
that true.

### It had never run

All three of these readers are invoked by `pipeline.read`, and `pilot.read`
only calls that when a deterministic parser is present. No deployment this
project serves declares one. So none of them had ever run for a single user —
including `quantify-weight-binding@1`, which had been rewritten to read the
*text* rather than the parse precisely because production has no Stanza, and
then sat behind the branch that requires one.

Reachability is the recurring defect here and it is never visible in a test
that calls the thing directly. `tests/test_day_of_month.py` now reads a
sentence through the path production takes, with no parser, and fails if the
day is not read.
