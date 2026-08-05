# Pilot observations

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
