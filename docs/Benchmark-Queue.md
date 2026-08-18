# Strategy benchmark — improvement queue

Run against `Baseline-v1`. Not a pass rate: an unsupported strategy refused by
name is correct, and a score counting executions would reward the silent
reduction this project spent months removing.

    reader                 gpt-4.1-2025-04-14@1   (was claude-sonnet-5@1)
    prompts                103
    correct executions     28
    correct refusals       42
    correct clarifications 8
    dangerous instances    0      (was 12; 1 under this reader until the
                                  withdrawal expectation was corrected)

| finding | instances | severity | layer | area |
|---|---:|---|---|---|
| incomplete refusal | 2 | high | Discovery | rebalance-annual |
| unnecessary question | 2 | medium | Discovery | ma-cross-below |
| unnecessary refusal | 2 | medium | Mission | ma-cross/persistent-below |
| unstable safe | 1 | low | Fusion | thousands shorthand |

**`swr-fixed-amount` was not a runtime defect.** The class declared
`refuses=["sell_action"]` and three of its four phrasings are refused for
exactly that. "an annual $40,000 withdrawal" is a noun phrase with no verb, so
Discovery settles `objective=assess_withdrawal` and Mission refuses *that* —
saying "this build only buys, so it cannot assess what taking money out would
do". Nothing executes on either path and the person is told the same true
thing.

The benchmark was asserting which *dimension* carried the refusal, when the
property that matters is whether the missing capability is **named**. Widening
a declaration to make a runtime pass is the cardinal sin, so it is paired with
a check that is strictly stronger than the one relaxed: the refusal message
must mention the withdrawal, whichever dimension carries it. A refusal naming
the right dimension with a message that never mentions taking money out would
now fail, where before it passed.

**These numbers were re-measured after the serving reader changed provider.**
Under `claude-sonnet-5@1` this read zero dangerous instances; under
`gpt-4.1-2025-04-14@1` it reads one. Nothing in Mission changed between the two
runs. That is the point of recording the reader beside the count: a benchmark
whose headline moves when the model moves is measuring the pair, and a number
quoted without the reader is not a fact about the system.

What did *not* move is the class that matters most — no silent reduction and no
wrong executable meaning under either reader. What moved is precision: refusals
naming the wrong capability, and questions about things the sentence states.

One of the findings below was *reclassified* rather than fixed. That is said here rather than left to be
discovered, because a headline that goes to zero in the same change that adds
a new category is the shape of a number being managed.

## What each one is

**A refusal that names one of the two things asked for.** "rebalance back to
60/40 every year" is refused for `stated_weights` and never mentions
`periodic_rebalancing`. The person is told one part of their sentence is
unsupported and left believing the other part was fine. An incomplete refusal
is more misleading than a blunt one, because its specificity reads as
completeness.

**`$1k` is refused where `$1,000` runs.** A recognition gap: the same request
in two notations, one executed and one refused by name. Classified
`UNSTABLE_SAFE` rather than dangerous because nobody is shown a plan that is
not theirs. Which notations to recognise is a question for the harvested
corpus rather than a guess, so it is deliberately still open.

**A question about an asset the sentence names.** "whenever VTI crosses under
the 200 DMA, invest $1,000" asks which assets to hold. VTI is the observed
asset and the held one, and reading the first without the second is a gap in
binding rather than a genuine ambiguity.

## Closed since the first run

**Order changed the compiled plan** (1, high) — and the diagnosis was worse
than the symptom. Mission's `_assets` split holdings on commas only, so "split
equally between VTI and BND" produced *one* holding literally named
`"VTI and BND"`, weighted at 100%. The two prompts were never two orderings of
a portfolio; they were two different single-instrument portfolios of an
instrument no market lists. `AllocationRule.canonical_form` sorts its assets,
so the sort ran over a one-element list and reported nothing wrong. Discovery's
fusion had already agreed the sentence named two assets — the member-splitting
rule exists in both layers, Mission may not import Discovery, and only one copy
knew about `and`. Now pinned equal by a cross-layer test.

**`$1k` compiled a plan that invested nothing** (1, critical). This was in the
queue as "plan identity moves on surface form", which undersold it. `_decimal`
returned `None` for `$1k` and the call site read `_decimal(...) or Decimal(0)`,
so the amount became zero while the asset, cadence and day rule were all
correct. The plan was indistinguishable from the one asked for except that it
invested nothing, and a backtest of it would have reported a portfolio that
never grew — with the market as the plausible-looking explanation.

Closed by refusing any stated number that cannot be read, for every numeric
dimension, rather than by teaching `_decimal` about `k`. Teaching it the
notation would have closed this sentence and left the class open for the next
one, which is the same mistake as closing rotation by adding `rotate` to a
lemma set.

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

## Found by the first real user, and open

Two dimensions the flagship pilot sentence states that Mission cannot consume.
Discovery reads both correctly; the stranded-dimension check refuses them
rather than dropping them, so the sentence terminates in an honest refusal —
and cannot run.

    i buy 1000 usd of SP500 etf every time SPY index trades under it's 200DMA
    - only ones on the next business day. what would be my total return and
    the final cash amount over the past 5 years

**`evaluation_period`.** "over the past 5 years". The manifest claimed
EXECUTED and nothing consulted it. Honouring a stated window means *selecting*
the prices a run sees — `period_start` and `period_end` are currently reported
from whatever index came back, not chosen — so it is an execution-path change,
not a compiler one. Refused by name until then.

**`cadence` beside a trigger.** "only ones" is read as `cadence=once`, and the
event-funded path does not consult cadence, so it is stranded. But "only once"
here is not a calendar competing with the trigger — it is a *repeat limit on
the trigger*: buy once per qualifying episode rather than on every session the
condition holds. The model has no representation for that, and the two
readings differ enormously in how much is contributed.

Both are capability gaps rather than defects in the reading, and the second is
the more interesting: `EventTriggered` exists, the Lean composition work
already formalises trigger-to-flow ordering, and what is missing is the
repeat semantics between them.

## Open, and deliberately not fixed yet

**`moving_average_window` has no unit.** Syntax reads `12` from "the 12-month
moving average"; the hosted reader reads `252`, the same twelve months in
trading sessions. Both are defensible readings of a field that never says what
it counts, and fusion refuses to settle it — the safe outcome, and the reason
this is not on the dangerous list. Fixing it means giving the dimension a unit,
which is a schema change; held until the harvested corpus says which units real
language actually uses. Tracked in `LEFT_THE_ANSWERABLE_SET` in
`tests/test_semantics.py`.

**`conditional_amount` fires on an amount that does not vary.** "invest $2,000
into VTI whenever it drops 10% below its highest close of the last year" is a
fixed contribution on a condition. gpt-4.1-2025-04-14 settles both `amount` and
`conditional_amount` to `$2,000`, and the build refuses `conditional_amount` by
name — so a plan it can run in full is refused for varying an amount that is
the same every time. claude-sonnet-5 does not do this, which is why it went
unnoticed: the sentence was written into the strategy selector, passed under
the reader the recording happened to use, and was caught only once the
catalogue was checked under both.

This is a false refusal rather than a silent reduction, so it is safe and not
on the dangerous list. It is still the most user-visible defect in the queue:
buying after a drawdown is an ordinary thing to want, and the refusal says the
build cannot do something it can. Held because the fix is in the reader's
proposal rather than the compiler — `conditional_amount` should not be proposed
when its value equals `amount` — and that is a fusion-level rule that wants a
falsification set before it ships. The entry is out of the selector until then;
`tests/test_strategy_library.py` is what keeps it out.

**`dividend_income` lost the refusal that was covering it.** "live off the
dividends and never touch the principal" was declined because `dividend_policy`
was refused wholesale. It is not any more — reinvested distributions are
credited from the total-return series — and the family now passes the legacy
compiler unrefused, taking the silent-reduction baseline from 9 to 10.

The refusal was right and its reason was wrong. Living off income is a
withdrawal strategy: what should decline it is `sell_action`, or an objective of
`assess_withdrawal`, neither of which the reader currently returns for this
phrasing. Held because the fix is in what the reader recognises rather than in
what the engine refuses, and a dimension added to the manifest to re-catch it
would be restoring an accident.

## What this does not authorise

The queue is a counterexample generator, not a fifth reopen trigger. A wrong
executable meaning and a silent reduction already activate the first two
triggers. An unsupported strategy appearing repeatedly is evidence for the
fourth — counted demand — and not a decision on its own.

---

## Found by running the suite against PostgreSQL

The suite runs on SQLite by default and reports clean. Roughly two hundred
PostgreSQL-only guarantees skip for want of `QUANTIFY_TEST_POSTGRES_URL`, and
that is the same blind spot that let `row[0]` — correct on `sqlite3.Row`,
`KeyError: 0` on psycopg's `dict_row` — break every saved plan in production
while every test passed.

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=x -e POSTGRES_DB=quantify \
        -p 55444:5432 postgres:16-alpine
    QUANTIFY_TEST_POSTGRES_URL=postgresql://postgres:x@127.0.0.1:55444/quantify \
        python -m pytest -q

    5 failed, 6688 passed, 3 skipped, 19 errors

### 1. Tenancy invariant, three tables — caused by declaring them

`test_tenancy_invariant.py::TestTheSchemaLayer` fails on `pilot_consent`,
`pilot_events` and `pilot_transcripts`: "tenant-owned and has no `owner`
column".

`tenant_owned_tables()` returns every table in `TABLE_MUTABILITY`, and those
three were added to that classification when the four runtime tables were
declared. They scope by `participant` — an anonymous study token that is
deliberately *not* a user identity — so they carry no `owner` column, and for
`pilot_events` and `pilot_transcripts` the participant is not in the primary
key either.

The retention registry already models this correctly, with
`owner_scope=DIRECT, owner_column="participant"`. So two registries disagree
about what "owned" means. Two candidate fixes, and the choice is a real one:

* Teach the invariant to read the declared owner column rather than assume the
  name `owner`. Aligns the registries; leaves `pilot_events` and
  `pilot_transcripts` failing, because their identity still omits the
  participant.
* Make the identity say what it is: composite keys `(participant, event_id)`
  and `(participant, entry_id)`. A migration, and the honest shape — the ids
  are already sha256 of the participant and the moment, so the scoping exists
  and is merely implicit.

Not exempted. `test_the_exception_list_is_empty` exists to stop exactly that,
and it is right.

### 2. Two failures that are test pollution, not defects

`test_pilot_events.py::TestWhatIsDeliberatelyNotMeasured` and
`test_pilot_session.py::TestTheUnnecessaryClarificationProxy` pass in isolation
and fail in the full run. Every test shares one PostgreSQL database, where on
SQLite each gets its own `tmp_path` file. The Postgres lane needs per-test
isolation — a schema per test, or a truncate between them — before its results
can be trusted as product signal.

### 3. The provenance digest does not verify — and has never run here

`test_provenance_journey.py::TestTheStoredRunCitesTheDeliveryItConsumed::test_the_stored_digest_is_the_resolver_digest`
fails on a *clean* database, and is skipped entirely on SQLite. It recomputes
`frame_digest(resolve(...).frame)` and compares it with the digest stored on
the access event; the two differ.

This is the most serious of the five. The stored digest is the mechanism that
answers "which observations produced this figure", and it is the foundation the
data-lake design in `Architecture.md` rests on. If a recomputed digest does not
match a stored one, the provenance chain records something that cannot be
checked against the data — which is the failure the mechanism exists to
prevent. It may be that resolution is not deterministic across calls, in which
case the digest identifies a frame nobody can reproduce.

Diagnose before building any of the snapshot work; the snapshot-by-hash design
assumes this property already holds.

---

## Parsing resolves; the first click still rarely draws a graph

Measured over all 43 offered strategies under the deployed reader:

| outcome | count |
|---|---|
| runs straight to a graph | 3 |
| asks one question first | 37 |
| refuses by name | 3 |

Nothing is silent, every question appears as a row with an example, and every
refusal names its dimension — the page contract holds. The parser is not the
problem here.

The dominant question is `assets`, asked for 22 strategies, and **in all 22 the
sentence genuinely names no holding**: "I withdraw $20,000 from the portfolio
each year", "I spend the taxable account first, then the IRA, then the Roth",
"I take the required minimum distribution starting at 73". A human advisor
would ask the same thing. The reading is right.

But it makes for a poor first click. Somebody chooses a withdrawal rule to
model *withdrawals* and is asked to invent a portfolio before the question they
came with can be answered. Two ways out, and neither is parser work:

* The catalogue sentence names a holding where the strategy does not care —
  "…from a portfolio of VTI and BND". Honest, and it makes the entry concrete.
* The table pre-fills a default and marks it as one, so the answer is a click
  and the provenance says `assumed, not stated`. Better, because it keeps the
  distinction between what somebody said and what we supplied — which is the
  distinction the whole parameter table exists to hold.

The second, with the default drawn from the strategy kind. Until then the
honest description of the product is: it reads what you say, tells you what it
still needs, and needs something roughly nine times in ten.

---

## Open upstream: `same_value` answers `SET` before it reads `normalizers`

Found while restoring Quantify's holding comparison after the internal fusion
was deleted.

`discovery_runtime.fusion.same_value` advertises `normalizers` as the way a
caller supplies what a mode means, and then answers `SET` from its own token
comparison before consulting the mapping:

```python
if mode == "SET":
    return _tokens(left) == _tokens(right)

reader = (normalizers or {}).get(mode)
```

So a rule registered under `SET` is accepted, never reached, and never
complained about. That is the same class of defect the adapter-completeness
guard exists for — a seam that exists and cannot be used — except that this one
is worse, because supplying the rule *looks* like it worked.

**Quantify is not blocked on it.** The domain rule is registered as `HOLDINGS`
and `compare_modes` maps the SET dimensions onto it, which is the better
arrangement anyway: dropping `a|an|the` is a fact about English, and the
runtime compares sets in any language. Overriding the generic mode would have
been wrong even if the seam worked.

What the fix upstream should be, when the pin next moves: consult `normalizers`
first for every mode and keep `_tokens` as the fallback for `SET`, so the
precedence is the same for all modes rather than special-cased for one. Worth
doing because the next consumer will register a `SET` rule and get silence.

**What it cost here:** two corpus cases, `the S&P 500 tracker` against `S&P 500
tracker` and `an SPX ETF` against `SPX ETF`, each reported as a disagreement
between readers that had read the same holding.
