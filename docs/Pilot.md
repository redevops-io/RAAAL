# The pilot

What is being studied, under what protocol, and what it has shown.

> Consolidated from `Pilot.md`, `Pilot.md`, `Pilot.md`, `Pilot.md`, `Pilot.md`, `Pilot.md`.
>
> The protocol, what it observed, the model evaluation and three separate baseline documents that had begun contradicting each other.


---

## Pilot protocol

Five to ten people. The objective is not conversion and not a demo — it is to
find out whether governed, replayable execution makes a workflow more useful
than a normal chatbot or agent.

## What the deployment must declare

    QUANTIFY_PARSER_MODE=RUNTIME
    QUANTIFY_PILOT_READER=HOSTED
    QUANTIFY_PARSER_MODEL=claude-sonnet-5
    ANTHROPIC_API_KEY=…
    QUANTIFY_PILOT_TRANSCRIPTS=yes      # see "What is retained" below

`RUNTIME` is a declaration, not a flag: without it `/workspace/new` serves the
legacy interpreter and the pilot measures the wrong product. The preflight
refuses a `RUNTIME` deployment with no key rather than passing startup and
refusing every request.

`HOSTED`, not `RECORDED`. A deployment replaying fixtures would serve answers
from a file and report them as the model's.

## The supported journey

    describe → clarify → save → figure → reopen → the same figure

That is the Era 1 launch journey, unchanged, and
`tests/test_pilot_journey.py` runs it. Anything outside it — proposals,
observations, counterfactuals, rerun — is on the legacy path and is not part of
this experiment.

## Freeze

No new behaviour before the cohort, with two exceptions: a correctness defect,
or something that blocks the supported journey. Everything else waits for
evidence that it matters.

**The freeze begins at the deployment proof, not at the invitation.** What it
protects is the join between an observation and the code that produced it, and
that join is fixed the moment the proof succeeds. Changing the measured system
between proof and observation defeats it whether or not anybody has typed
anything yet.

`swr-fixed-amount` is the live case. It is a real correctness defect — one
dangerous instance in the strategy benchmark, a refusal naming the wrong
capability — and it therefore qualifies under the first exception. The choice
is between two honest orders and not between fixing and not fixing:

    fix it, then re-run the deployment proof, then invite
    or leave it known and out of scope for the whole cohort, and fix after

What is not available is fixing it quietly between the proof and the cohort.
That produces observations attributed to a revision that was not serving them.

## Retaining the proof, not just the result

The successful output of `scripts/verify_deployment_identity.py` is kept with
the cohort evidence rather than read once and discarded.

    cohort event -> serving_commit -> deployment proof -> repository revision

Every pilot event already carries `serving_commit`, so this needs no
instrumentation — only the discipline of keeping the proof. Without it the
chain depends on the service still running that revision when somebody asks,
which it will not be in three months. The proof is the durable half; the
running service is the perishable one.

## The six observations, and where each one comes from

These are the things worth knowing about a participant. They do not all come
from the same place, and the table exists because treating them as if they did
is how a study ends up with a number standing in for a conversation.

| Observation | Source | Honest? |
|---|---|---|
| the exact prompt they entered | `pilot_transcripts`, verbatim | yes, if retention was declared |
| whether the runtime asked an unnecessary clarification | **interview**, with `clarifications_answered_from_the_prompt` pointing at which transcripts to read | the counter is a proxy, not the finding |
| whether the refusal matched what they expected | **interview only** | nothing mechanical can answer this |
| whether they edited and resubmitted | `plan_resubmitted`, with `text_changed` | yes |
| whether they reopened saved plans | `plan_reopened` | yes |
| whether they went back to the legacy workspace | `left_for_legacy` | yes |

Four are counted, one is a pointer, one is a conversation.

### The proxy, stated plainly

`clarifications_answered_from_the_prompt` counts the times somebody answered a
question by repeating a phrase their sentence already contained. If a
participant wrote *invest $500 monthly into VTI*, was asked what to hold, and
typed *VTI*, the reading missed something that was in front of it — a fact about
the parse, not a judgement about the person.

It cannot see the opposite error, a question that should have been asked and was
not, and it misses anyone who answers a redundant question in different words.
So it is not the finding. A capability appearing there repeatedly is a parser
defect with a name attached, and it is worth chasing precisely because it
arrives without anybody being asked anything.

## What is retained, and what people are told

Two stores, kept apart on purpose.

    pilot_events        counts and codes. No prose. Needs no permission.
    pilot_transcripts   what someone actually typed. Off unless declared.

    QUANTIFY_PILOT_TRANSCRIPTS=yes      keep participants' sentences
    QUANTIFY_PILOT_TRANSCRIPT_DAYS=30   and drop them after this long

Off by default, and only an explicit affirmative turns it on — a typo in that
variable must fail towards *not* keeping what people typed.

That variable says the study *may* keep prose. It cannot say whether a given
person agreed, so it is not the whole gate. Consent is recorded per
participant, and both gates must open before a sentence is stored.

### The three modes

Each participant is in exactly one, and the system can tell them apart:

| Mode | What is collected | How |
|---|---|---|
| events only | counts and codes | the default; nothing to do |
| events + transcript | their sentences too | `grant(participant)` |
| events + transcript + interview | and the conversation | `grant` + the five questions |

The third is not a system state — it is whether a conversation happens. It is
in the table because conflating it with the second is how a study ends up
believing it interviewed everyone whose transcript it kept.

### The notice

The exact words live in `pilot_consent.NOTICE`, in the repository, versioned.
Read them out before the session starts:

> **Pilot notice**
>
> During this pilot we may record the prompts you enter and the system's
> responses to help us improve the interpreter. These transcripts are used only
> for product improvement, are retained for up to 30 days, and can be deleted on
> request at any time. Participation is voluntary, and transcript recording is
> disabled unless you explicitly agree.

Then, verbally:

> "We'll ask whether we may keep the conversation transcript for analysis. If
> you'd rather not, we'll still collect anonymous usage events and run the
> session normally."

A test asserts the notice's promises against the mechanism: thirty days is the
default retention, withdrawal exists, and recording is off unless declared. It
is the one document here a non-engineer relies on, so its claims are checked
rather than trusted.

### Running it

    participant opens the pilot page          → a token is issued
    read the notice, ask                      → grant(token) or decline(token)
    they work                                 → prose kept only if granted
    they ask to be removed, ever              → withdraw(token)

The token is issued by the empty page view, before anything is typed, so the
unprompted first sentence — the most informative thing anyone produces all
session — is inside the consented window rather than discarded before consent
could exist.

**Consent is not retroactive.** Sentences typed before someone agreed stay
unkept, and no code path can reach back for them. Agreeing at the end of a
session is agreeing to what happens next.

**Consent is versioned.** Change the notice, bump `NOTICE_VERSION`, and every
earlier grant reads as `UNKNOWN` until those people are asked again.

**An empty transcript store is never ambiguous.** `pilot_consent.by_state()`
separates "declined" from "nobody asked" — the same distinction as zero events
versus zero usage, and the same reason for making it.

Requests carry an opaque participant token in a cookie. It is not identity: no
name, no address, nothing derived from the request. It exists because a revision
chain is invisible without it, and because forty compiles is a busy pilot or one
person struggling and those are opposite conclusions.

The wording read to participants, and the consent it records, are below under
*The three modes*. `pilot_consent.withdraw(participant)` is the deletion, and it
works today — written before the pilot rather than after somebody asks.

## What the cohort is for, beyond counting questions

The obvious analysis — average follow-ups, then automate the frequent ones —
would be wrong, and wrong in the direction that costs safety. Frequency alone
does not distinguish a question that wastes somebody's time from one that
decides what runs.

Three independent questions per frequently-clarified dimension:

    was it frequent                        asked_by_dimension
    did the answer change what would run   answering_changed_the_outcome
    was the answer already recoverable     from deterministic evidence

Only **high / low / yes** argues for another deterministic reader. That is the
combination `quantify-trigger-semantics@1` had: asked often, answers rarely
altering the plan, and the grammar stating the answer when the model omitted
it.

The inverse is the part worth guarding. A dimension asked constantly whose
answers *do* change execution identity is the most valuable question in the
product — it is information acquisition rather than friction, and automating
it away would remove the thing that makes the plan correct. The distinction
this analysis draws is exactly that one.

## What the telemetry answers

`pilot_events.summary()`, at a prompt. Counts and names, each carrying the
deployment profile so a model-only cohort stays separable from a later
dual-witness one:

- which capabilities were refused, by name and count
- plans saved, and plans saved with questions still open
- plans reopened
- results produced, against refusals produced
- resubmissions, and how many changed the wording
- departures to the legacy workspace
- how many participants all of the above is spread across

Every one of those is a thing that happened. None is a claim about what anyone
thought of it.

Read `participants` first. Every other number is ambiguous without it.

## What only the interview answers

Same five questions for everyone, asked after the task, in this order. The
order matters: expectation before interpretation, because asking what they
thought the result meant first will make them reconstruct an expectation to
match it.

1. **What did you expect it to do** when you typed your sentence?
2. **What did you think the result meant?**
3. **Was any question or refusal confusing?** Which one, and what did you think
   it was asking?
4. **Did the evidence make the result more trustworthy** — the reader, the
   pinned intent, the fact that reopening recomputes from what you confirmed?
   Or was it noise?
5. **What would you try next** if you were using this for real?

Ask them the same way each time and write the answers down verbatim. A summary
written from memory is a summary of what the interviewer expected to hear.

**Do not** turn these into a form, a rating, or a telemetry field. A number
derived from a five-point scale on "was this trustworthy" is a number that will
be quoted later without the conversation that produced it, and the conversation
is the evidence.

## What would make this a failure worth having

The pilot succeeds as an *experiment* if it produces a clear answer, including
a negative one. Specifically these are all useful outcomes:

- people ask for capabilities the manifest refuses, repeatedly and by name —
  the refusal boundary is right and the manifest is too small
- people do not care that reopening replays — the runtime's distinctive
  property is not the one that matters to them
- people abandon at the clarification question — the interpretation is right
  and the interaction is wrong
- nobody wants to type a sentence at all — the surface is wrong, which no
  amount of runtime work would have discovered

The outcome that would waste the pilot is a set of numbers with no
conversations attached, because the questions worth answering are not the ones
the system can count about itself.


---

## Pilot observations

Things seen, not things owed. An observation records what happened and what is
not yet known about whether it matters; it becomes work when evidence says so.

The distinction is load-bearing. "TODO: fix worksheet generation" asserts that
something is broken and should be repaired. "Plan execution completed, figures
rendered, worksheet absent for this plan shape, no evidence users depend on
it" asserts only what was seen — and leaves the decision to the people who
will actually be affected.

Format: what was observed, under what conditions, and what would make it
actionable.

---

## OBS-1 — Worksheet absent for a conditional-trigger plan

**Observed** 2026-08-05, live deployment `90ccc99`.

The original-prompt journey completed end to end: nine questions answered, the
plan saved, and the plan page rendered figures — $1,000 contributions, $5,160
accumulated, $4,248. `worksheet_present` was `False` for that plan.

The run happened; the figures are on the page. Whatever renders a worksheet
did not produce one for this plan shape. Both supported launch journeys
(contribution replay, Roth) produce worksheets, and the deploy-time journey
check asserts it — so this is specific to the conditional-trigger shape rather
than general.

**Not yet known.** Whether any user opens a worksheet, and whether the figures
on the plan page are what they actually read.

**Would become actionable if** a pilot user asks where the detail is, or
reports the figures without being able to say how they were reached. Both are
severity 2 — misleading certainty — and neither will arrive as a bug report.

---

## OBS-2 — Model re-parse on every round trip

**Observed** 2026-08-04, live deployment.

Each submission in the Plan Builder re-parses the description with the model:
the save route verifies the posted parse by re-parsing, and the re-render
parses again. A round trip is therefore two provider calls, taking tens of
seconds.

`moving_average_kind` disappeared between two passes without being answered,
and `cadence` appeared. Some of that is legitimate — supplying an amount makes
"how often" newly relevant — but model non-determinism across calls means the
question set is not wholly attributable to what the user did.

**Not yet known.** Whether users notice, and whether the drift is large enough
to break the sense that answering made progress.

**Would become actionable if** a user says the questions changed for no
reason, or if latency is the thing they complain about. The fix is available
and cheap: the pinned `parse` token is already posted and already verified, so
the re-render can use it rather than re-deriving it. It is not done because
nothing yet says it matters more than the things ahead of it.


---

## OBS-3 — The description travels in the URL

**Observed** 2026-08-05, closing the log leak above.

`/workspace/new?describe=…` puts the user's financial description in the query
string. The proxy log and the uvicorn access log both recorded it verbatim,
and both are now redacted — but a query string also reaches places no
server-side redaction can follow: browser history, the address bar, a
screenshot, a pasted link, and any intermediary that keeps URLs.

For synthetic-data evaluation this is minor. For a user describing real
holdings it is not, and OBS-1 aside, this is the surface most likely to matter
once the pilot lifts its data boundary.

**Not yet known.** Whether pilot users share links, and whether they type
descriptions they would mind appearing in their own browser history.

**Would become actionable if** the pilot moves past synthetic data, or a user
shares a plan link. The fix is a POST for the describe form with a redirect to
an opaque draft id — a change to one route and one template, not to the
compiler.

---

## OBS-4 — The pilot's own journeys record no telemetry

**Observed** 2026-08-05, checking whether Phase 1's continuous-observation
loop has anything to observe.

`QUANTIFY_TRACE_PATH` is set, `trace.db` exists with the right schema, and the
`trace`, `span` and `decision` tables hold **zero rows** after a day of live
journeys including the original-prompt walkthrough.

`_recorder()` is called from one place — `routes.py:1346`, the worksheet and
intent path. `GET /workspace/new` and `POST /workspace/save`, which is every
step a pilot user takes, construct no recorder. Gate 6's comment says it fixed
"the only production entry point"; it fixed a real one, and the Plan Builder
was added later.

**Unlike the other observations, this one cannot wait for evidence**, because
it is the thing that gathers evidence. Waiting would be circular.

**What is still observable without it.** CloudWatch keeps the request line for
every call: which routes were hit, in what order, with what status and
duration. That yields journeys attempted, abandonment between `/new` and
`/save`, error rates and latency.

**What is not.** Anything inside the clarification loop — which questions were
asked, which were answered, which answer settled which field, how many round
trips a user took, which unsupported capability they requested. Precisely the
Phase 1 list.

**Decision needed before invitations,** and it is not an engineering question
so much as a scope one: accept a pilot that measures traffic but not
understanding, or wire the recorder into the two Plan Builder routes first.
The second is small — `_recorder()` already exists and already fails safe —
but it is engineering work during a declared freeze, so it is the operator's
call rather than an implementation detail.

### Resolved — 2026-08-05, under a named exception to the freeze

The operator classified this as an **instrumentation defect** rather than a
bug, and amended the freeze to read:

> Product behavior is frozen. Operational correctness and instrumentation
> required to evaluate the pilot remain in scope until the first external users
> are invited.

The distinction that made it decidable: a defect in what the product *does*
waits for users; a defect in what the product *observes* cannot, because it is
what the waiting is for. "Closer to fixing a broken thermometer before starting
an experiment than changing the experiment."

Scope held to the two routes. No schema change, no new decision kinds, no new
event types, no dashboard. `DecisionKind.CONFIRMATION` already existed and
already meant this.

**What is recorded.** One trace per journey; `plan_draft` and `plan_save`
spans; one decision per screen naming the fields the user was asked about, with
an outcome from a fixed set — `QUESTIONS_PRESENTED`, `READY_TO_SAVE`,
`RETURNED_FOR_ANSWERS`, `RETURNED_NOT_EXECUTABLE`, `TEMPLATE_DISPATCH`. A saved
plan's id is written to `produced`, so a trace can be found from the artifact
rather than only from a request id.

**Field names only, never answers.** `cadence` and `asset_identity` are the
compiler's vocabulary; an answer may carry an amount, an employer or an
instrument nobody has heard of. This store must not become the third place a
user's sentence survives after Caddy and uvicorn were closed.

**Two defects found by wiring it**, both by the falsification pass rather than
by the change itself:

- The privacy assertion passed hardest when nothing was recorded — an empty
  store contains no canary either. It now witnesses its premise before
  claiming absence proves anything. Same shape as the restore drill certifying
  its own write.
- `assert refs.strip()` passed against a recorder writing `[]`, because an
  empty JSON array is a non-empty string. Replaced with a comparison against
  the fields the *page* rendered — and that immediately showed the recorder
  naming three of five controls: inferences awaiting confirmation are
  questions the user receives, and were not being counted as any.

An empty `GET /workspace/new` opens no trace. A blank form is not a journey,
and recording one would turn the trace count into a page-view metric.

### The production canary found a leak the suite could not — 2026-08-05

The first request against the newly-wired recorder proved reachability
(`trace 0→1, span 0→1, decision 0→1`) and in the same breath wrote this:

```
unclear:every so often (unclear cadence)
unclear:tech (unspecified asset/sector, not a ticker)
```

The user's own words, in the store documented to hold none. Not every
unresolved item is vocabulary: an unplaceable phrase becomes
`unclear:{phrase}`, the user's text with a model-written reason appended — and
the reason string appears nowhere in this codebase, because it comes back from
the model. **"Record field names, never answers" was not the guarantee it
sounded like.** The leak arrived through the name.

Fixed in `b0f529e` by hashing anything not in `vocabulary.FIELDS`. An allowlist
rather than a rule against `unclear:`, because the next dynamic field id would
arrive without one, added by whoever is least likely to be thinking about this.

**Why no test could have caught it, which is the part worth keeping.** The
privacy case described something the compiler fully understood, so it produced
no unplaceable phrase. Choosing a better description would not have helped: the
deterministic parse emits no `unclear` list *at all*. Only `MODEL_ASSISTED`
does, and that is what production runs.

So this was not a missing assertion or a badly chosen input. **The test
environment lacked the capability that produces the dangerous input.** Three
vacuous checks were found in this one slice — an expectation derived from
itself, an observation derived from itself, and now an environment that cannot
generate the case. The third is the one to watch for, because reading the test
carefully does not reveal it; only running the real configuration does.

The new case stubs the model client so the live route builds the field id out
of the canary, then asserts both that it reached the recorder and that the
words did not.

---

## OBS-5 — A declared rule was never executed, and the figure looked authoritative

**Observed** 2026-08-05 by a pilot user opening a saved plan, not by any test.

    I buy $1,000 of SP500 ETF every time the S&P 500 crosses below its
    200-day moving average for the past 5 years.

The page showed $5,160 and +18.09% — identical, to the penny, to "Your basket,
bought and held" and to "Contribute to S&P 500" — beside a disclosure reading
*"every dimension outside the investment rule was held identical… a difference
between these figures is attributable to the rule."*

There was no difference and no rule. `_run` called
`simulate(..., program=buy_and_hold(tradeable))` regardless of what the
scenario declared, and nothing ever converted `event_program` into an
`EventProgram`. The user noticed because $1,000 contributed could not support a
rule that fires repeatedly — the arithmetic was visible even though the defect
was not.

Fifth instance of the reachability shape in this codebase, and the first that
moves money. `simulate` takes a `program` argument; the engine has always been
able to run one. The live path did not reach it.

### Two further defects the fix uncovered

**The disclosure that should have caught it had never rendered.**
`declare_unsimulated` writes `declared_but_not_simulated`; `_scope.html` reads
`scope.not_modelled`. Computed correctly, attached to the result, stored, and
displayed by a template reading a different key — so both columns of "What this
simulation models" were empty, and the `dividend_policy` disclosure had been
invisible for its entire life too.

**`declare_unsimulated` derived its inventory from a one-entry dict** while its
docstring claimed the opposite: *"derived from the scenario rather than
hardcoded, so… one that is added starts being disclosed without anyone
remembering to edit this function."* The claim to be exhaustive is what made
the omission dangerous rather than merely incomplete.

### Why no test caught any of it

`test_the_original_prompt.py` is a permanent fixture for this exact sentence
and passed throughout. It asserts parsing, asset identity and the time window —
whether the system *understood* the user, never whether it *did* what it
understood.

The missing-producer class again, and sharper this time: the compiler only
builds an `event_program` once `trigger_semantics` is settled, so a test using
the bare description compiles to zero steps and exercises none of this. The new
suite settles the trigger by amendment and asserts the premise first.

**Deployment 1** (this commit) stops the wrong number: no figure, no benchmark
table, the rule named under "Not modelled", `STRATEGY_EFFECT` refused at the
classifier, and affected runs invalidated from an inventory derived from stored
artifacts rather than a list of plan ids someone read off a page.

**Deployment 2** executes the rule and reports a timeline — which is also the
independent witness that it ran, because one purchase and $1,000 total cannot
support a repeating rule and would be visible as such.


## Follow-up burden (added before the cohort)

The harvested corpus changed what the pilot should watch. 16 of 29 attested
strategy statements never reach a plan — they stop at a question about
holdings, because real financial intent is routinely incomplete on first
utterance. Instrumentation built only on plan events would record those
sessions as near-silence and could not say whether the runtime asked well or
badly.

So the interaction is captured, not only the outcome:

    original utterance          pilot_transcripts, under a declaration
    unresolved dimensions       discovery_asked
    questions asked             discovery_asked, by dimension
    the answer supplied         discovery_answered, by dimension
    sealed intent               intent_sealed
    disposition and result      the plan events already recorded

`discovery_answered` is **one event per dimension that goes from unresolved to
answered**, emitted by whichever route performed the transition:

    /pilot/answer resolves X          ->  discovery_answered(X)
    /pilot/save   resolves X and Y    ->  discovery_answered(X), (Y)
    X already answered                ->  nothing

Tying it to a state change rather than a UI action is what makes undercounting
and double-counting both impossible, and it makes the emitter idempotent — so
it can be called from anywhere a reading exists without anyone reasoning about
overlap. A dimension only counts if it was asked about first: something the
first sentence already supplied was never a follow-up, and counting it would
inflate the burden the runtime is measured on with work it never asked for.

`src/workspace/pilot_burden.py` joins them into three questions that are
deliberately not one question:

**How many follow-ups were needed.** `asked_by_dimension` — distinct material
dimensions raised, counted once per participant however many times the question
was shown. Someone who reloads five times is one person who has not answered,
not five who could not. `answered_by_dimension` is what the asking bought, and
`asked_and_never_settled` is the difference.

**Which were unnecessary.** `answer_was_already_in_the_prompt` — dimensions the
participant answered with something their original sentence already contained.
A proxy, and named for what it measures: somebody may restate a thing the
runtime was right to be unsure about.

**Which missing material facts were never asked about.**
`never_asked_by_dimension`, computed as a dimension Mission refuses as
`UNRESOLVED_INPUT` that Discovery never raised. This is the one a question
count cannot reach: a runtime that asks nothing scores perfectly on burden and
may be failing worse than one that asks twice, because the person is refused at
the end having never been given the chance to supply what was missing.

No rates. A ten-person cohort makes a percentage look like a measurement and
behave like one participant's afternoon.

### The blind spot that shaped the semantics

`/pilot/save` accepts `answer_<dimension>` fields and recorded no answer event,
so anybody who supplied the missing holding *and* saved in one step counted as
having answered nothing. Emitting from the save route as well would have
double-counted anyone who answered and then saved.

Neither is a defect in the emitter. Both come from counting form submissions
and calling them answers, which is why this was settled as a definition rather
than patched — and why the metric now means the same thing regardless of which
buttons somebody presses.


## The seven questions, audited before the cohort rather than after

Instrumentation added later cannot describe what already happened, so each
question was checked against what is actually recorded. Three were not
answerable and they differed in kind.

| question | answerable from |
|---|---|
| which dimensions most often cause clarification | `asked_by_dimension` |
| how many questions before a seal | `intent_sealed.questions_before_sealing` |
| which questions changed the execution identity | **added**: `intent_sealed.execution_identity` |
| which answers Discovery could have derived | `answer_was_already_in_the_prompt` (a proxy) |
| which material facts were never asked about | `never_asked_by_dimension` |
| abandonment after a clarification | **derived**: asked, never sealed |
| whether phrasings differ in burden | **not from telemetry** — see below |

**The one real gap was the third.** The ledger could say how many questions
were asked and not whether any of them changed what would run, and that is the
question deciding whether a dimension deserves a deterministic reader or simply
a better default. `intent_sealed` now carries the *execution* identity — a
digest, so it stays countable, and the execution form specifically, so two
seals differing only in how somebody spelled a holding do not read as a
follow-up having changed the outcome.

**Abandonment needed no new event.** It is an absence, and nothing can be
emitted by the thing that did not happen. A participant asked something who
never sealed is found by subtraction, in analysis.

**Phrasing equivalence cannot come from telemetry at all.** Deciding which
utterances are "the same strategy" is a judgement, and no event can carry it.
That is a transcript-and-annotation task under the retention declaration —
said here so it is not mistaken later for a gap somebody forgot to close.

## What would justify another deterministic reader

`quantify-trigger-semantics@1` earned its existence from measured stochastic
omission plus decisive structural evidence. The bar for the next one is the
same and the cohort is what supplies it:

    the dimension is asked about often                asked_by_dimension
    the answer rarely changes what runs              answering_changed_the_outcome
    the grammar states it when the model does not    a falsification suite

A dimension asked constantly whose answers *do* change the outcome does not
want a reader — it wants a better question.


---

## Stage-1 model evaluation — first bounded run

**Date:** 2026-08-01 · **Model:** `claude-sonnet-5` (resolved id as returned by
the provider) · **Compiler:** commit `7b3f9f9`, version 3 · **Cases:** 205 ·
**Calls:** 205 · **Retries:** 0 · **Cap:** 215

41 families × 5 wordings: three semantically identical (extraction accuracy and
paraphrase convergence), one contradictory, one underspecified.

Two things were true before this ran, and they are why the result is
interpretable rather than anecdotal: paraphrases already converged to one
canonical Mission (100%), and specifications already round-tripped without
identity or provenance drift (100%). A failure here is attributable to **model
extraction**.

---

## 1. Against the targets

| Metric | Target | Measured | |
|---|---:|---:|---|
| `rule_hash` exact | ≥ 95% | **99.5%** | 204/205 |
| `schedule_hash` exact | ≥ 95% | **99.0%** | 203/205 |
| `content_hash` exact | ≥ 90% | **98.0%** | 201/205 |
| Field provenance exact | — | **99.5%** | 204/205 |
| False inference | ≤ 1% | **0.98%** | 2/205 |
| Contradiction recall | 100% | **100%** | 41/41 |
| Family convergence (3/3) | ≥ 90% | **92.7%** | 38/41 |

**Hard gates, all clean:** 0 saveable Missions with open questions, 0 schema
failures, 0 retries, 0 silent recommendations, 100% of responses passed
deterministic validation before simulation.

Operational: p50 5.7 s, p95 12.1 s, p99 15.5 s. 661 input / 577 output tokens
per call; 135,568 / 118,283 total.

---

## 2. The quarantine did real work

145 model proposals were refused:

| Refusals | Why |
|---:|---|
| 90 | a ticker that does not appear in the description |
| 36 | the field was already recognised |
| 14 | a value outside the vocabulary *(this one was our bug — see §4)* |
| 5 | not a field stage 1 recognises |

Ninety attempts to resolve a company name to a symbol the user never wrote. That
is the check that exists to stop a scenario pricing the wrong security, and it
fired on 44% of calls.

The model contributed 34 readings the phrase rules missed — 33 `funding_source`
and one `weighting`.

---

## 3. Three divergences, three different causes

**Two are genuine false inferences.** `WM-0012#s1` and `WM-0037#s1` describe a
plain monthly contribution with no conditional buying at all, and the model set
`funding_source: additional_cash`. Nothing in either description mentions where
extra money comes from. This is the field with the clearest financial
consequence — as additional cash the plan invests more, and more money in a
rising market always looks like a better rule.

**One says the deterministic compiler is wrong.** `WM-0009#CONT` — "I buy $500
of SPY every week" — compiled to *no assets at all* under the phrase rules,
because `SPY` sits on a reserved list that stops it being read as a holding. It
is on that list because it is usually the *signal* in a trend rule. The model
read it as the holding, which is what the sentence says. The deterministic
reading is the defective one.

---

## 4. Two harness defects, and one product one

The first summary reported **family convergence 0%** and **provenance exact
19%**. Both were the harness.

Convergence was measured across all five wordings per family, including the
contradictory and underspecified ones that are *meant* to differ. Provenance
compared quoted spans rather than field provenance — two parsers reading one
sentence legitimately quote different substrings, so it measured phrasing.

Full capture is what let both be corrected without spending the budget twice.

A third defect was real: the quarantine rejected `sells_allowed: "False"` on
capitalization, because the vocabulary is lowercase and a JSON boolean
serializes capitalized. It refused a **correct** reading of "I don't sell
anything" fourteen times. Values now compare case-insensitively; field names
stay exact.

---

## 5. The finding with product consequences

**80.5% of cases (165/205) gained an extra question** the deterministic compiler
does not ask — almost always a note that the account type ("my brokerage
account") maps to nothing in the vocabulary.

The model is not wrong. Account type genuinely is not a field stage 1 can
represent, so it correctly declines to place it. But every one of those notes
becomes a user-facing question, so four in five plans would show a question
about something the user already said clearly.

That is a vocabulary gap, not a model failure, and it is the single largest
difference between the deterministic and model-assisted paths.

---

## 6. Verdict

On the error distribution: **mostly exact, with rare uncertainty** — so model
stage 1 is usable behind confirmation, which is where it already sits.

Before exposing it in the pilot:

1. add `account_type` to the vocabulary, or suppress `unclear` notes for
   phrases the compiler deliberately does not model — 80.5% extra questions is
   not shippable friction;
2. decide whether `funding_source` should be model-readable at all, given both
   false inferences landed there;
3. remove `SPY` from the reserved list when it is the object of a purchase.

Cross-model comparison comes after those, on the same 205 cases.

---

## 7. What was fixed as a result

All three, in the commit after the run.

**Account type is now in the vocabulary.** `tax_treatment` had always been on
the scenario and in the content hash, and nothing ever set it — so every plan
compiled from prose was `NONE_APPLIED`, and a Roth compared as identical to a
taxable account. This project's founding example of a defect was still live for
the entire conversational path. Taxable, Roth IRA, Roth 401(k), traditional IRA
and 401(k) are now read and represented; an account the compiler cannot place
(a donor-advised fund, an inherited IRA, "my retirement accounts") is asked
about rather than guessed, because guessing between traditional and Roth is
precisely the defect.

**SPY is a holding when it is bought.** Written as a signal test rather than a
list of purchase verbs — the first attempt enumerated buy/put/invest and missed
"goes into", which the stability benchmark caught within one run.

**`funding_source` was kept model-readable.** Both false inferences landed
there, but that is an argument that the field is economically important, not
that it should be removed: it changes invested capital, benchmark equivalence,
the TWR/MWR reading and the schedule hash. Better to ask one more question than
to assume.

### The metric that replaced the hash comparison

    accepted without a question      69.3%   (1,597 of 2,304)
    accepted after 1-2 questions     30.7%
    needs three or more questions     0.0%
    silently changed                  0.0%

"rule_hash exact 99.5%" answers a question nobody asks. This one is what a user
lives with.

---

## 8. Reproducing

```bash
python3 scripts/run_model_eval.py --max-calls 215   # billable, ~21 minutes
python3 scripts/analyze_model_eval.py               # offline, from the bundle
```

`reports/modeleval/bundle.jsonl` holds one full capture record per case: source,
expected Mission, model response hash, accepted and rejected proposals, actual
Mission, typed diff, tokens, latency and pins.


---

## Quantify v1 — the baseline the pilot starts from

Frozen before any pilot evidence changes anything, so there is a clean
before/after point. Every version below is read out of the tree by
`tests/test_baseline_v1.py` rather than typed here, because a baseline that
records what somebody believed is not a baseline.

    commit                  89959bd
    Discovery schema        quantify-discovery-schema@6/eb01a824e4f43d02
    capability manifest     quantify/capability-manifest@1
    hosted prompt           quantify-hosted-prompt@1
    serving reader          gpt-4.1-2025-04-14@1
    fusion pipeline         quantify-pipeline@1
    MWR contract            quantify/mwr-contract@1
    drawdown semantics      drawdown@2
    Formal Core             v1 — 89 theorems, 57 guards

The schema is at **@6**, not @3. It moved three times: `@4` added
`reserve_policy`, `bucket_policy` and a `leverage_multiplier` qualifier; `@5`
added `asset_location`; `@6` added `selection_rule` and `holding_period`. Each
was a bump because the *content* changed, which is the rule that keeps two runs
from looking comparable when they are not.

## The serving reader changed provider after this baseline was frozen

Recorded for the same reason the schema move is, and it matters more: every
number in this document was measured through a reader that is no longer the one
serving.

    frozen with             claude-sonnet-5@1
    now                     gpt-4.1-2025-04-14@1
    moved by                the pilot having no Anthropic credential

The corpus was re-recorded under the new reader — both sets are kept, keyed by
reader id, so the old readings remain as history rather than being overwritten.

**What survived the change, and what did not.** This is the strongest available
test of the claim that the guarantees are properties of the architecture rather
than of one model, and the answer is mixed in an informative way.

Held:

- **No silent reduction.** Nothing executes a meaning the person did not ask
  for, which is the class this project exists to remove.
- **No wrong executable meaning.** No contrast pair collapsed to one plan.
- **42 of 43** answerable corpus cases produce the same value as before.
- Every Lean theorem, every accounting identity, every refusal-by-name path —
  none of it touches the reader.

Did not hold:

- **One false claim of support.** "an annual $40,000 withdrawal" is refused for
  `objective` rather than `sell_action`: the right outcome for the wrong stated
  reason, which tells somebody the system cannot do a thing it never considered.
- **Two unnecessary refusals**, on `conditional_amount`, for moving-average
  sentences the previous reader did not emit that dimension for.
- **Two unnecessary questions** about an amount the sentence states.
- **One case reads backwards.** "buy the index rather than through an ETF"
  settles `assets='ETF'` — the instrument the sentence rejects — accepted on a
  single witness because syntax was silent. Recorded in `MODEL_IS_WRONG`.

The pattern is worth stating plainly: **the safety properties are
reader-independent and the precision properties are not.** Changing provider did
not make the runtime execute anything wrong; it made it refuse and ask less
accurately.

## The schema moved after this baseline was frozen

Recorded rather than rewritten. `@5` was the version frozen here; `@6` came
after, and a baseline that quietly renumbered itself would destroy the
before/after point it exists to provide.

    frozen at               quantify-discovery-schema@5/ca8f3b7785ff5d70
    now                     quantify-discovery-schema@6/eb01a824e4f43d02
    moved by                the strategy evaluation benchmark, before any
                            pilot evidence

The benchmark found `momentum-rotation` executing as buy-and-hold: "hold
whichever performed best" produced a plan with two holdings and a monthly
cadence, the selection silently gone. Closing it needed a dimension that could
*represent* the selection so Mission could refuse it by name — `selection_rule`
— and `holding_period` alongside it, because the same investigation found "hold
VTI for 200 days" and "buy VTI below its 200-day moving average" compiling to
the identical plan.

This is the authored corpus doing the job it was kept for. The pilot has not
started; the change is attributable to the benchmark and to nothing else, and
that attribution is the reason to write it down here rather than edit a line.

## What is settled

- **Formal Core v1** — `docs/Measures.md`, with every public claim mapped to
  a named theorem.
- **Discovery serving profile** — hosted model plus the Stanza syntax witness,
  `WitnessProfile.BOTH`, with presence guards so a dropped material dimension
  becomes a question rather than a silence.
- **Mission** — capability manifest refuses by name; relations reach the
  refusal path; `objective` classified.
- **Pilot instrumentation** — seven events, structured context only, per
  participant consent, transcripts off unless declared.

## The Discovery baseline, frozen on admissible evidence

The pre-Lean gate is **open**, and what makes that citable is not the verdict
but the artifact behind it: the first one ever produced by CI from the actual
serving stack, carrying every identity the gate checks.

    run                     31603973383
    commit                  a982959
    schema                  quantify-discovery-schema@6/eb01a824e4f43d02
    serving reader          gpt-4.1-2025-04-14@1
    hosted prompt           quantify-hosted-prompt@1
    fusion pipeline         quantify-pipeline@1
    producer                github-actions
    mode                    workflow_dispatch
    draws                   3 per prompt, 36 prompts

    execution_unsafe            0
    silently_reduced_any_draw   0
    silently_reduced            0
    watched_crossed             0
    closure_witness             gpt-4.1-2025-04-14@1

    STABLE_EXECUTABLE 4 · STABLE_REFUSAL 12 · STABLE_CLARIFICATION 2
    UNSTABLE_SAFE 18

`corpus/parser/drift.json` is that artifact, committed rather than described.
It expires: the gate refuses evidence about stochastic behaviour older than
seven days, so this freezes what was measured without letting it be cited
forever.

**Why the two former blockers matter more than the count.** Both now produce
*one distinct executable identity across all three draws*, which is a stronger
statement than "never unsafe". It says the serving path reproduces the same
executable meaning, rather than merely containing stochasticity behind
refusals.

**`UNSTABLE_SAFE` is 18, and the decomposition is the useful part.**

    12  REFUSAL <-> REFUSAL            different stated reasons
     4  CLARIFICATION <-> REFUSAL      different journey shape
     2  CLARIFICATION <-> CLARIFICATION  a different ambiguity asked about

Only the last two vary which question a person is asked. None of it reopens
the safety gate, and none of it is a pre-pilot zero target — it is a UX queue.

## What is open, and stays open

- **Closed, and left here as the shape of the problem.** The gate was blocked
  for weeks on `producer='unknown'`, which read as an operational to-do —
  press *Run workflow*. It was not: `drift-lane.yml` lived on a feature branch
  and GitHub registers `workflow_dispatch` only from the default branch, so
  the lane had never been startable. Merging it, configuring
  `OPENAI_API_KEY`, and dispatching produced the artifact above.
- The gate now pins `hosted_model_id` and checks the closure report's witness.
  It was blind to both, and a provider swap would have walked straight through
  with a GPT drift artifact and a Claude closure report reaching one verdict.
- Mission computes no volatility and no drawdown. `max_drawdown` is declared
  absent in `worksheet_view` with a reason.
- Six prompts are `UNSTABLE_SAFE` and watched.

## What reopens engineering

Four triggers, and nothing else. Everything outside them is pilot evidence, not
a coding task:

1. **wrong executable meaning** — a plan that does not mean what was said
2. **unsafe or silent reduction** — an unsupported intent producing a figure
3. **supported journey blocked** — describe → clarify → save → figure → reopen
4. **repeated user demand crossing a roadmap trigger** — counted, not felt

The fourth is deliberately about frequency rather than architecture. If six of
ten people ask for withdrawals, that earns withdrawal capability. If nobody
asks for factor tilts, the schema being able to represent them is not a reason
to build them.

## The corpora, and which one now matters more

    authored / synthetic     the regression laboratory
    real pilot prompts       the evidence for what deserves investment

The authored corpus becomes secondary in **product prioritisation**, not in
**correctness**. It stays the adversarial and rare-case suite, and nothing
below it may be thinned because a cohort did not happen to type those
sentences.

It found the accumulation bias, the crossing-versus-persistent defect, the
moving-average off-by-one, the MWR two-root case and the drawdown opening-level
gap. Ten people would have produced none of them. What it cannot do is say what
anyone wants, which is the only thing the pilot prompts are better at — so they
become primary for deciding what to build, and for nothing else.

## What no amount of this proves

Whether real people express intentions the schema has not anticipated,
understand a refusal when they meet one, and trust the explanation enough to
approve what is about to be evaluated. That is the pilot's question and nothing
above answers it.


---

## Baseline v1 — semantic correctness complete

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


---

## Baseline complete

The point at which a defect stops being an architecture question.

After this line, an issue is **product**, **UX**, **capability**, **performance**
or **pilot evidence** — and is answered by contact with users rather than by
stating another invariant. Before it, the foundations could not be trusted to
support learning, so internal work was the fastest way to improve the system.

The distinction matters because there is no natural end to inventing
invariants. Each one this branch applied found something real; that is exactly
why a stopping rule has to be written down rather than felt.

## The six criteria

| | Criterion | Witnessed by |
|---|---|---|
| 1 | **Compiler** — a description compiles to a plan, or names what it cannot read | `tests/test_the_original_prompt.py`, the vocabulary and registry suites |
| 2 | **Provenance** — every decision records who made it, and survives storage | `tests/test_provenance_persistence.py` · **pending in production, see below** |
| 3 | **Execution** — a declared rule runs, or no figure is produced | `tests/test_event_triggered_execution.py`, `tests/test_declared_rule_not_executed.py` |
| 4 | **Telemetry** — journeys are observed, and observe nothing they should not | `tests/test_telemetry_reachability.py`, canary-verified in production |
| 5 | **Verification** — evidence reports PASS, FAIL *or* VACUOUS | the falsification passes; `deploy/provenance_gate.py` exits 2 when untested |
| 6 | **Deployment** — the running system states its own identity and refuses when it cannot | `deploy/acceptance.py`, the startup proof, schema parity |

## What is not yet witnessed

**Criterion 2 has never been demonstrated in production.**

The code is deployed (`3eaa5eb`, migration `a91c4e7b2f05`) and both database
lanes pass. But every plan in the production workspace predates the fix, so the
gate that would prove it reports:

    VACUOUS: no plan on the current shape exists, so the check that matters —
    prose answers without structured records — was never evaluated.

That is the honest state, and it is one browser session from closing: a plan
created through the builder after this deploy, then inspected in storage for
structured `amended`, `asset_resolutions` and `time_window`, reopened without
questions reappearing, and executed to a reconciled ledger.

**Until that run exists, this document describes a baseline that is complete in
code and unwitnessed in production.** Declaring it done on the strength of a
green suite would be the failure this branch spent its length removing.

## The three properties that were not true before

**A plan can no longer silently produce the wrong result.** It compiles and
executes, or it refuses and says exactly why. The middle case — a figure for a
strategy that never ran — is gone, and a stored one from before is withdrawn
rather than deleted.

**User decisions are first-class data.** An answer is a `ScenarioAmendment`
with a source and a timestamp, stored structurally and rendered from that
record. Previously it became a sentence, and a sentence cannot be replayed.

**Evidence distinguishes three outcomes, not two.** PASS, FAIL and VACUOUS. The
third is the one that took longest to learn: a check that reports success
without having evaluated anything is worse than a missing check, because it
converts an unknown into a false known.

## The layering that produced them

    code → tests → SQLite lane → PostgreSQL lane → deployment → production gate

Each layer found defects the previous one structurally could not. The
PostgreSQL lane is not the same tests on another engine: `JsonText` and
`sa.Text()` both render TEXT on SQLite, so a whole class of type-parity defect
can only exist where they differ. The production gate is not the suite again:
it reads rows that real journeys wrote.

The last two entries are why "the tests pass" was never the standard here.
