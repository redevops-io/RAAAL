# Reading a sentence

How language becomes an intent: authority, corpora, and evidence.

> Consolidated from `Semantics.md`, `Semantics.md`, `Semantics.md`, `Semantics.md`.
>
> Authority, the corpora that test it, the harvest that stresses it, and the rules evidence must meet. One subject: how a sentence becomes an intent nobody has to trust.


---

## Authority is earned per semantic claim, not per parser

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


---

## The parser corpora, and what the web pass found

**Status: Phase 6 frozen.** `AWAITING_A_PARSER = 2`, and both are
deliberate — a units-policy decision and an asymmetry witness. Every
other case has an owner and a proven producer. See *Phase 6, frozen*
for the boundary and the three explicit non-goals.

Three files, and the difference between them is the whole point.

| | cases | provenance | asserts |
|---|---|---|---|
| [`corpus/parser/cases.json`](../corpus/parser/cases.json) | 218 | self-authored, plus 12 observed | correctness |
| [`corpus/parser/real_phrasings.json`](../corpus/parser/real_phrasings.json) | 43 | attested where marked | current behaviour |
| [`corpus/parser/stackexchange_candidates.json`](../corpus/parser/stackexchange_candidates.json) | 29 | verbatim, CC-BY-SA | nothing — awaiting review |

`cases.json` says what the layer *should* do. I wrote both the sentences and
the expected values, which is the weakest evidence this project recognises; the
one-property-per-case rule limits the damage by making a wrong expectation
visible rather than burying it in a conjunction.

`real_phrasings.json` says what the layer *does* with language people actually
write. It carries no expected plans on purpose — labelling attested phrasings
with the answers I think they deserve would reproduce the self-authorship
problem one layer up. `tests/test_real_phrasings.py` therefore asserts current
behaviour and marks which of those behaviours are defects, so a green run means
nothing moved and a red run says on which sentence.

## Where it came from

`bogleheads.org` returns HTTP 402 to this environment's fetcher and
`reddit.com` blocks it. Those are stated preferences about automated reading,
not obstacles to route around, so neither was scraped — swapping a user agent
to get past a block a site deliberately put up would be evading a wish rather
than solving a problem.

**Stack Exchange publishes an API for exactly this, and licenses its content
CC-BY-SA.** So that is the route:
[`harvest_stackexchange.py`](../corpus/parser/harvest_stackexchange.py) queries
`money` and `quant` for descriptions of what someone does with a portfolio,
keeps sentences between 20 and 120 characters that name a first-person action,
and tags each with the *pattern* that matched — never with a meaning. It writes
candidates to a separate file and merges nothing automatically, because an
unreviewed sentence is a sentence nobody has read.

Provenance is per entry: **13 `stackexchange`** (verbatim, with the question
URL, which is both the attribution the licence requires and the provenance the
pack needs), **11 `user_reported`**, **8 `search_summary`**, **11 `variant`**
written by me and marked so. A test holds the attested share above half.

It is still not a sample of how people write — the searches were chosen to
surface descriptions of actions, and that is a selection. But the attested part
is now real text with a link rather than recollection.

## The finding

> In *"I contribute monthly and rebalance at year end"* the parser attaches
> `year end` to **contribute**, not to **rebalance**.

Stanza makes `end` an `obl` of `contribute` and makes `rebalance` a conjunct of
`monthly`. So it reports that the year-end timing belongs to the contribution —
confidently, and wrongly, on a sentence anybody might write.

The constructed sentence it was modelled on, *"invest $500 monthly and
rebalance annually"*, parses correctly. That is exactly why a corpus of
invented sentences said the layer worked, and it is the case for building the
attested pack before writing fusion rules rather than after.

And it cannot be patched with a rule. The attested longer form — *"maintain
60/40 by contributions and rebalance at year end"* — attaches `end` to
`rebalance` **correctly**. Same phrase, same two verbs, opposite result,
decided by how much else is in the clause.

**Consequence for Phase 5.** The plan already says syntax must never win by
itself. This is the concrete argument: syntax here is not neutral or absent, it
is *wrong*, and it is wrong with a chain the scorer walks happily — so the
fusion layer receives a confident score rather than an abstention. A rule of
the form "strong syntax overrules the model" would adopt the error. The
disagreement between a wrong parse and a right model reading has to surface as
a disagreement, not be resolved by whichever number is larger.

## What the harvest broke, immediately

Twenty-nine sentences, and four defect classes the invented corpus had no
instance of. All four are now cases in `cases.json` with `origin: observed`.

**Hyphenated splits were read as nothing.** Four of the first twenty-nine wrote
`60-40` or `80-20` — *"I start with a 60-40 equity to fixed asset allocation"*,
*"my allocation becomes 80-20"*, *"rebalance down to 60-40"*. Every invented
case had used a slash. The sums-to-100 test turned out to do the same work for
both, so `80-20` is a split and `2012-2015` is not.

**A percentage range collapsed to its upper bound.** *"(10-20% of my
allocation?)"* came back as 20%. Silently.

**A money range lost a factor of a thousand.** *"currently make ~$200-$220k per
year"* came back as two amounts, `200` and `220000` — the multiplier at the far
end governs both, and a reader taking the first match cannot know that. Both
ranges are now refused rather than resolved: a range is something to ask about,
and a plausible wrong amount is worse than a question. The mechanism is the one
already there — the range claims its span before anything else can read inside
it, the same rule that stops *"90-day moving average"* becoming a holding
period.

`between $500 and $800` needed its own pattern, because `and` is a range marker
only after `between`. On its own it joins two separate amounts, and *"contribute
$500 and $200 to the second sleeve"* must keep reading both.

**An age was read as an investment horizon.** *"Me - 32 years old"* came back as
an 11,680-day duration. `50 years` in *"an investment horizon of 50 years"* is
the same shape minus one word, and is still read — that case is in the corpus
as the discriminating opposite, so the fix cannot quietly eat real horizons.

## The other defects the pass recorded

**A rebalancing band is invisible.** `5/25` is the standard Bogleheads band —
rebalance when a holding is 5 percentage points, or 25% relative, from target —
and it is shaped exactly like an allocation. The sums-to-100 rule drops it,
which is the correct answer for `12/25` and the wrong one here. No rule over
digits alone separates them; it needs the surrounding words, which means it
belongs above tier 1.

**Nothing binds a ratio to its account.** *Resolved — see "The relation
reader" below.* `401k (50/50), Roth IRA (85/15), taxable brokerage (70/30)`
normalises to three ratios and no accounts, and normalisation was never going
to bind them: this is the `RelationSpec` rule verbatim, that meaning depends on
which value belongs to which participant. The binder now establishes all three.

**And the parser tokenises that sentence three different ways.** *Resolved by
`align()`.* In one string Stanza keeps `50/50` as a single token, splits
`85/15` into `85`, `/`, `15`, and splits `70/30` into `70/` and `30`. An
extractor written against any one shape is wrong about the other two, and no
scoring rule reaches this — it is below the level the scorer sees, which is why
the fix was an ordering change rather than a rule.

**Two ratios arrive with nothing marking which is the target.** In *"my 60/40
is acting like 70/30"* the first is the target; in *"70/30 vs 60/40"* neither
is. Order does not settle it, so a consumer taking the first ratio is right by
accident.

**And the ambiguity is in the language, not only in the reader.** A Bogleheads
thread is titled *"Don't Know How To Rebalance/Reallocate"*. People writing
about their own portfolios do not reliably separate "rebalance back to target"
from "change the target", which means no parser and no model can separate them
from the sentence alone. That is a case for Discovery's clarification gate
rather than for a better score.

## What the corpus is evidence of

> The parser corpus is not evidence that syntax is correct. It is evidence of
> where syntax is informative, where it is unstable, and where clarification is
> required.

That is a stronger basis for a fusion layer than a large self-authored fixture
set would have been, and it is why 43 sourced phrasings were worth more here
than pushing the invented corpus to 300.

Each of the three characterization cases marks a different kind of limit, and
they need different answers rather than better scores:

| case | limit | what it implies |
|---|---|---|
| `5/25` vs `12/25` | digit shape alone cannot decide split from band from date | needs the surrounding words; belongs above tier 1 |
| `401k (50/50), Roth IRA (85/15)…` | extracting tokens is not binding them | needs relation binding, not a better extractor |
| `50/50` whole, `85/15` split, `70/` + `30` | the scorer never receives the unit consistently | **normalise before scoring**, and align values to spans |

The last one is an ordering constraint, not a tuning problem. A scorer that
reasons over tokens is reasoning over whatever the tokenizer happened to do;
one that reasons over normalised values aligned to character spans is not.

## Not installing DeerFlow yet, and the condition that would change that

[`bytedance/deer-flow`](https://github.com/bytedance/deer-flow) (MIT) is a
reasonable tool for a job this project does not have yet: finding broader
strategy language across many sources and proposing candidate fixtures. It is
not the tool for the job just done — twenty-nine sentences from a free,
licensed API broke four assumptions in about a minute, which no agent stack
would have improved on.

The evidence hierarchy this pass established, weakest first:

    invented fixtures            good for falsification, nothing else
    search-summary snippets      useful, weaker — recollection, not text
    verbatim licensed user text  much stronger, and cheap
    long-horizon agent research  potentially useful, not yet necessary

**The trigger:** reach for DeerFlow when the free and licensed sources stop
producing new structural parser failures, or when breadth is needed across
domains the direct APIs cannot reach. Right now the direct APIs are still
paying.

If it is installed, backend-only — `make install` builds the frontend through
pnpm/corepack and there is no reason to take on that dependency surface for a
headless research run. `DeerFlowClient` runs in-process without the HTTP
services. Wire it to **Brave or Bing**, both of which already have keys here;
Perplexity synthesises rather than returning raw results, which is the wrong
shape when the point is verbatim language. Do not add Tavily just for this.

## Phase 5, written from the above

[`src/discovery/fusion.py`](../src/discovery/fusion.py). Four outcomes, and
only one of them proceeds:

| outcome | when | repair |
|---|---|---|
| `AGREE` | the model proposed a value and syntax did not contradict it | — |
| `DISAGREE` | syntax argues otherwise, or syntax proposed alone | adjudication |
| `INSUFFICIENT_RELATION` | the value needs a binding nobody supplied | a reader or schema that binds |
| `AMBIGUOUS_BY_LANGUAGE` | the words carry both readings in attested usage | ask the user |

They are four rather than one `UNRESOLVED` because they call for different
repairs, and a ledger that recorded only "unresolved" would stop before saying
which.

**Nothing reads a score's magnitude.** `contradicts()` looks at the sign only —
a negative score on the model's own value, or a positive score on a different
one. The `year end` case is why: the parser was confident and wrong, so size is
not a signal about correctness, and any threshold would have adopted the error.
Syntax cannot promote a reading either; the value that proceeds is always the
model's.

`AMBIGUOUS_BY_LANGUAGE` is checked **first**, before agreement. Two readers
agreeing on a word that carries two meanings are two readers making the same
assumption, which looks like confirmation and is not. Its term list requires a
source per entry and a test enforces that — a list anyone may add a hunch to
becomes a list of things nobody wants to implement.

Open decisions map onto the contract rather than into a parallel vocabulary:
material ones become `Unresolved(result_changing=True)`, which is what `seal()`
refuses on, so nothing downstream can execute a guess. An
`AMBIGUOUS_BY_LANGUAGE` becomes `NOT_ASKED` rather than
`UNRESOLVED_DISAGREEMENT` — nobody disagreed, nobody has asked yet, and
recording it as a disagreement would misdescribe both the cause and the repair.

## The relation reader

[`src/discovery/binding.py`](../src/discovery/binding.py). The layer fusion
could refuse without but never supply: **who does this value modify?**

    401k (50/50), Roth IRA (85/15), taxable brokerage (70/30)

    ratio@6-11   appositive_of  BOUND  -> 401k
    ratio@24-29  appositive_of  BOUND  -> Roth IRA
    ratio@51-56  appositive_of  BOUND  -> taxable brokerage

Three ratios, three accounts, through three different tokenisations — the
binder consumes normalised values aligned to spans, so the tokenizer's
inconsistency never reaches it.

**It emits structure and never meaning.** A binding says a value is the
appositive of `401k`; it does not say `401k` is an account. Each rule declares
which semantic pairing it is *evidence for* — `account↔allocation`,
`asset↔weight`, `cadence↔action` — and that declaration lives on the rule, not
in the output, so a consumer cannot mistake a structural fact for a settled
field. A test asserts no binding ever names a semantic field.

Three statuses, and the middle one is the point. `AMBIGUOUS` is not `UNBOUND`
with extra steps: a value with two candidate targets is one where picking
either is a coin toss dressed as a reading. Both reach fusion as
`INSUFFICIENT_RELATION` today; they stay distinct because the repairs differ —
one needs a rule, the other needs a question.

Two things the sentences corrected while it was being written. The rule first
looked *upward* for a shared head, and `40%` reached across the `conj` edge to
match `VTI` as well as `BND`; looking downward at what hangs off the value's
own phrase cannot cross a coordination boundary. And targets were reported as
`k`, because Stanza splits `401k` and makes `k` the head — structurally right
and useless, since a target nobody can identify is a binding nobody can check.
Phrases are now rebuilt by character offset, which knows that `401` and `k` are
adjacent and `Roth` and `IRA` are not.

**The seam with fusion is tested as a dependency, not as a substring.** Two
earlier versions of that test scanned `fuse`'s source for the word "parse" and
matched, first, its own explanation of why it does not parse, and then a
refusal message written for a user — the self-matching scan this project has
now produced three times. A text search cannot tell an access from prose about
the access. The property is that fusion does not depend on the structural
types, and an import graph says that exactly. Mutating either side proves it:
importing `Parse` into fusion fails it, and removing the binder predicate fails
it and the live-binding test together.

## Phase 6 — the field mappers, and the metric

[`src/discovery/semantics.py`](../src/discovery/semantics.py). Consumes
`RelationBinding` and normalised values, emits `SemanticCandidate`, touches no
parse — the binder already turned structure into a typed fact, and a mapper
going back to the tokens would be the third place in the pipeline reading
dependencies.

    normalisation → binding → semantic candidate → fusion → field | Unresolved

A candidate is a proposal, not a field. It carries no confidence and no rank —
only what it proposes, what it came from, and the evidence. Fusion decides
whether it survives, and that separation is what stops the mapper becoming the
authority.

`cadence` joined the normaliser to make the mappings possible: recognising that
`monthly` names the MONTHLY period is lexical work, the same class as `$1k` →
1000. What that period governs is the binder's job; which field it fills is the
mapper's. The payoff is the sentence the layer was built for:

    contribute $500 monthly, rebalanced annually
      amount               500
      cadence              monthly
      rebalancing_cadence  annual

Two cadences, two fields, decided by which verb each one attaches to.

### The metric

[`corpus/parser/closure.json`](../corpus/parser/closure.json), regenerated by
[`closure.py`](../corpus/parser/closure.py) and checked by
`tests/test_closure.py`. Not "how many cases left `AWAITING_A_PARSER`" — a
number rules can move — but why each one is still there:

| state | count | owner |
|---|---|---|
| `MAPPED_AND_AGREED` | 9 | — |
| `AMBIGUOUS_BY_LANGUAGE` | 1 | the user, via clarification |
| `NO_FIELD_MAPPING` | 15 | `semantics.py` — field derivation |
| `NO_LITERAL` | 25 | the semantic reader |
| `NO_PARSE_RECORDED` | 4 | `stanza.download` |

`AWAITING_A_PARSER` went 108 → 54 → 47 → **45**, and the nine that left are run
end to end by `tests/test_semantics.py` rather than counted as handled. The
goal is not zero. It is that every remaining case names a specific owner and a
specific reason instead of sitting in a bucket.

**`STILL_UNSUPPORTED` was split, and the split is the useful part.** It held
two failure modes wanting opposite work. *"weight by inverse volatility"* has
no literal at all — nothing to normalise, nothing to bind, and no rule written
here would change that; it belongs to the semantic reader. *"contribute a fixed
$500"* has both: `amount=500` is recognised and bound, and what is missing is a
rule saying that "fixed" makes `amount_kind` FIXED. One needs a reader, the
other needs field derivation from structure already in hand, and a single state
name made 36 cases look like one queue.

**`INSUFFICIENT_RELATION` went to zero without a binder rule.** All six were
corpus fragments: *"below the 200-day moving average"* has no verb, Stanza
roots it on `average`, and the binder correctly found no governor. Nobody types
a bare prepositional phrase at a runtime. The six were rewritten as complete
utterances with their semantic assertions unchanged and the originals recorded
in the builder — teaching the binder to invent a governor would have let the
corpus's shape drive the grammar. Two of the six then answered correctly; the
other four moved to `NO_FIELD_MAPPING` with a specific reason each.

One of them found a real gap on the way: *"sell when it drops under its 50-day
average"* reads as a 50-day **duration**, because `_WINDOW` requires the words
`moving average` and real writing drops them. It is recorded in the phrasings
pack rather than patched — `X-day average` is a window in this domain and not
in others, and widening tier 1 on the strength of one case is how a normaliser
starts guessing.

### The report shipped with the defect it was built to catch

The first version took `candidates[0]` when no candidate matched the field a
case asserts. So *"when SPY crosses below its 200-day average"* was answered
with a 200-day **holding period** and counted as a success for trigger
semantics. Six of thirteen agreements were that shape — the
comparator-manufactures-agreement defect, reproduced inside the instrument
built to measure the pipeline.

Two things now stop it. A candidate for another field is `STILL_UNSUPPORTED`
and names what *was* proposed. And `MAPPED_AND_AGREED` carries
`matches_expected`, because fusion's outcome and the corpus's expectation are
different axes: agreeing says the pipeline was internally consistent, matching
says it was also right. Restoring the fallback takes agreements from 7 to 12
and fails both guards.

## Phase 6b — derivation families

Six families, not fifteen rules, in
[`semantics.py`](../src/discovery/semantics.py). A family says which evidence
determines which value, and fires only when that evidence is present.

    amount_kind        `fixed` modifying an amount        -> fixed
                       a percentage governed by `of`      -> proportional
    trigger_semantics  comparison + a change-of-state verb -> crossing_event
                       comparison + a copula               -> persistent_condition
    day_rule           `last` modifying a period          -> last_session_of_period
                       `first` modifying a period         -> first_session_of_period

**The verb is the discriminator; the preposition is only the signal.** `below`
alone does not mean a persistent condition — *crosses below* is an event and
*is below* is a state. A family requiring the preposition without the verb
would be inferring from a preposition, so both are required and neither is
sufficient.

The binder grew two fields to make this possible without a mapper reading a
parse: `target_lemma` (the head's lemma, unrebuilt — a verb table compared
against the readable span `"put in"` missed exactly the multiword cases) and
`modifiers` (the lemmas describing the value's phrase). Both are still
structure with no meaning attached: the binder reports that `fixed` modifies
the amount and says nothing about what that implies.

### Movement

    MAPPED_AND_AGREED   9  ->  16    (all 16 with the expected value)
    NO_FIELD_MAPPING   15  ->   9
    NO_LITERAL         25       25
    NO_PARSE_RECORDED   4        4

`AWAITING_A_PARSER`: 108 → 54 → 47 → 45 → **38**.

**A defect surfaced in how fusion was being called, not in fusion.** The first
run put five cases into `MAPPED_AND_AGREED` and one into
`AMBIGUOUS_BY_LANGUAGE` — *"rebalance on the last session of each quarter"*,
flagged because the sentence contains "rebalance". But the caller was passing
the **whole utterance** as the proposal's source span, so any sentence
containing an ambiguous term made *every* field of that sentence ambiguous. A
candidate now carries its own span, and fusion reads that.

**`AMBIGUOUS_BY_LANGUAGE` is now zero, and that is correct.** The
rebalance/reallocate ambiguity is about *what the action does*. It does not
touch how often the action happens or which session of the period it happens
on, which are the fields these cases assert. The corpus has no case asserting
the field the ambiguity actually affects — so the outcome is exercised
synthetically in `test_fusion.py`, and a live instance would be a corpus
addition rather than a code change. Recorded in the report so a zero is not
read as an outcome that never fires.

### The nine that did not move, and why none got a rule

| case | what it needs |
|---|---|
| `a 60/40 portfolio`, `split it 70/30 …` | a `modifies_nominal` **binding** — a ratio modifying an allocation noun. A binder rule, out of scope here, and "unbound ratio → allocation_method" would be a rule firing on absence, which `70/30 vs 60/40` shows is wrong |
| `purchase VTI whenever QQQ drops 10%`, `add to BND while TLT …` | role **pairs**. Tier 2 already establishes subject-of-condition and object-of-action; what is missing is a candidate shape that carries a pair |
| `make a one-off $10,000 investment` | `make` is not a funding verb and adding it would fire on "make a withdrawal". The funding is in the noun `investment` |
| `sell when it drops under its 50-day average` | the elided-head window gap — two attested instances, waiting for a third |
| `invest whatever is left over each month` | no literal for the amount at all |
| `invest 10% of my salary monthly` | the percentage binds to `salary monthly` rather than through `of` |
| `buy a core index fund monthly` | `assets` is a span, not a normalisable literal |

Each is a specific reason rather than a pile, which was the point of splitting
the state. None of them was closed by widening a rule to fit it.

## The schema-alignment pass

**Contract field names are canonical at the fusion boundary.** Readers,
mappers, fusion and corpus assertions all speak them. A parser feature may be
called whatever is clearest locally, but it must be translated before it becomes
a `SemanticCandidate` — because the moment two witnesses name the same thing
differently they can never agree about it, and what that looks like is a
permanent `DISAGREE` that is really a spelling difference.

Five mapper-only names, adjudicated one at a time rather than by growing the
schema to match the mapper. Only one was a rename.

| field | verdict |
|---|---|
| `asset_weight` | → `stated_weights`. A clean rename. |
| `rebalancing_cadence` | → **intermediate**. Right in name, wrong in shape |
| `account_allocation` | → **intermediate**. A relation, not a dimension |
| `amount_kind` | → **intermediate**. The manifest executes an amount, not a kind of amount |
| `holding_period_days` | → **intermediate**. Nothing in the manifest makes it change a result |

The `rebalancing_cadence` case is the instructive one. `periodic_rebalancing`
holds a free-text description — its own schema examples are *"rebalance
quarterly"* and *"when it drifts more than 5 points"* — and the deterministic
path produces the canonical token `annual`. Under the dimension's declared
`TEXT` comparison those are not the same value, so renaming converted a
vocabulary mismatch into a *permanent* false `DISAGREE`. Left intermediate
rather than resolved either way: making the mapper emit prose to match a reader
is worse than the mismatch, and loosening the dimension's comparison is a schema
decision.

Intermediates are kept on the `Read` rather than dropped. `amount_kind=fixed` is
a real reading of a real sentence, and discarding it because no contract field
exists would lose the evidence that the contract might need one.

### What alignment removed

**Every live `DISAGREE`.** The one reported before this pass —
`rebalancing_cadence` with the model silent — was entirely an artifact of the
mismatch: the schema calls that dimension something else, so the model could
only ever have been silent about it.

And no contract field on any recorded sentence is now proposed by syntax alone.
The model reads every contract field the deterministic path does. So the
syntax-alone policy is exercised synthetically in `test_fusion.py` and has no
live instance — asserted as an absence, so the first sentence that produces one
is noticed rather than assumed.

### The ambiguity rule, narrowed

An attested-ambiguous term now fires only when **both of its readings are
available in the sentence**. `AMBIGUOUS_TERMS` names the contract fields each
ambiguity is between, and the outcome requires at least one of the others to
have been proposed too:

    rebalance to 70/30      AMBIGUOUS_BY_LANGUAGE   a target is present, so
                                                    "restore 70/30" and "change
                                                    the target to 70/30" are
                                                    both on the table
    rebalanced annually     AGREE                   same word, one reading —
                                                    the competing one needs a
                                                    target and there is none

Two general rules came out of this, and both are about a check firing on its own
ontology. Never put the field name in the evidence used to decide whether the
*user's* language was ambiguous. And "the word appeared" is not ambiguity —
ambiguity is both readings being on the table.

### Counts after alignment

    MAPPED_AND_AGREED      14   (all 14 with the expected value)
    NOT_A_CONTRACT_FIELD    6   the schema, or nobody
    NO_FIELD_MAPPING        7   semantics.py
    NO_LITERAL             23   the semantic reader
    NO_PARSE_RECORDED       4   stanza.download

`AWAITING_A_PARSER` went 38 → **40**, upward, because two cases that had been
answered were answered under mapper-only field names. A count that moves the
wrong way for a good reason is worth more than one that only ever improves.

## Both witnesses in the report

The closure report runs the whole pipeline now, not the deterministic path
alone. Before the hosted reader was wired, a case whose field nothing
normalises could only be `NO_LITERAL` — a label measuring the absence of *one*
producer and calling it the absence of all of them.

    AGREE                   12   both witnesses, all 12 with the expected value
    MODEL_ONLY_ACCEPTED     23   the model alone, all 23 with the expected value
    DISAGREE                 1   adjudication
    MODEL_ONLY_UNRESOLVED    1   adjudication
    INTERMEDIATE_SEMANTIC    6   pending on nothing
    SCHEMA_GAP               2   pending on nothing
    NO_FIELD_MAPPING         4   semantics.py
    NO_PARSE_RECORDED        4   stanza.download

`AWAITING_A_PARSER`: 40 → **10**.

`AGREE` and `MODEL_ONLY_ACCEPTED` are kept apart deliberately. A field two
readers reached independently and a field one reader settled are different
evidence, and collapsing them would let the report claim agreement it never
observed. `INTERMEDIATE_SEMANTIC` and `SCHEMA_GAP` are excluded from the pending
count — they are not waiting on anything — and the report carries
`previously_counted_by_awaiting_a_parser` with the ids, so correcting the
boundary being measured is visible rather than quiet.

### What the second witness exposed

**Tier-3 cases asserted values outside the schema's own vocabularies** — the
field-name error one level down, and invisible until something else answered
the same question.

| case asserted | the schema says |
|---|---|
| `allocation_method = "60/40"` | `stated_weights`; the weights live in their own dimension |
| `dividend_policy = "cash"` | `held_as_cash` |
| `evaluation_period = "1825"` | `trailing:5y` — the dimension spells the canonical form out |

Eight cases. The model was right every time. A guard now checks every tier-3
assertion against the dimension's declared vocabulary, so the class cannot
recur silently.

**Two readings the contract cannot hold** are `SCHEMA_GAP`, not renamed to the
nearest allowed value: `market-cap weighted` is not among `allocation_method`'s
seven values, and `mid-month` is not among `day_rule`'s three. Both are sayable
and neither is executable. Renaming them would have made the corpus agree with
a schema that cannot express the sentence.

**Three of the four `DISAGREE`s were my own NUMBER coercion.** A regex that kept
digits turned `£1k` into 1 and `12-month` into nothing. Comparison now
canonicalises through the *normaliser*, so one place decides what a written
number means instead of two deciding differently — and the report compares by
the dimension's rule too, which is where the last mismatch was hiding.

One genuine disagreement survives: `12-month` against `12` for
`moving_average_window`, where the schema says "in sessions" and the case says
12. A real units question, left live rather than resolved by picking a side.

And the syntax-alone policy does have a live instance —
*"whenever SPY drops under the 200-day"*, where syntax proposes
`trigger_semantics` and the model is silent. The earlier claim that none existed
was drawn from three sentences.

## The preflight, and the queue that is left

**Corpus expectations are validated against the contract before they may judge
parser output.** `corpus/parser/loader.py` refuses to load a fixture whose
expected field or expected value the contract cannot represent.

The rule earned itself. The same class was caught at three layers in three
passes — wrong field names, wrong field *value* vocabularies, wrong numeric and
unit coercion — and every time it was invisible until a second witness answered
the same question and disagreed. Without a preflight the corpus quietly becomes
a second schema, and a second schema always wins arguments it should lose.

Two escapes, both explicit and both meaning something:

    schema_gap            the reading is right and the contract has no value
                          for it — the finding, not an error
    INTERMEDIATE_FIELDS   semantics this pipeline computes outside the
                          contract boundary

### `NO_FIELD_MAPPING` went to zero without a mapping

All four were corpus vocabulary. Three asserted `{held, observed}` — a private
shape for a distinction the schema already carries as two dimensions, `assets`
and `observed_assets`, where the second's own description is that exact
sentence. Both witnesses had been reading the roles all along, under the
contract's names, and the report could only say "no mapping produces role
pairs" because nothing else used that shape.

The fourth, *"a 60/40 portfolio"*, asserted `allocation_method` for a bare noun
phrase that states weights and names no method. Neither witness made that
inference and the model was right not to.

### Both directions of the asymmetry are live

    model speaks, syntax silent    30 cases, accepted
    syntax speaks, model silent     1 case, unresolved

The second was being reported as `MODEL_ONLY_UNRESOLVED` — the count was right
and the label named the wrong witness, which is worse than either being wrong
alone. It is `SYNTAX_ONLY_UNRESOLVED` now and kept deliberately: it is the only
live proof that fusion handles both directions rather than the model filling
deterministic gaps.

### Multilingual is deferred, not deleted

The four non-English dependency fixtures need a model per language — about two
gigabytes — and multilingual parsing is not in the declared scope. They live in
[`deferred_multilingual.json`](../corpus/parser/deferred_multilingual.json)
marked `NOT PART OF CURRENT COVERAGE`, with why and how to bring them back.
Fetching the models so a counter reached zero would corrupt what the report
means; deleting them would lose the evidence of what was intended.

The multilingual **normalisation** cases stay in the active corpus, because
`normalize` needs no model and they run today. That split is the honest one:
the layer that works multilingually is measured, and the layer never fetched is
not counted as pending.

### The intermediates got a test surface, and four of six failed on it

They were worse than pending — excluded from the queue *and* verified by
nothing. The surface is deliberately short: normalise, bind, `propose()`,
assert the candidate. No fusion and no contract-field assertion, because there
is no second witness to agree with and no field for a decision to be about.

Four had no producer at all. `keep it for six months` and `put in half of any
bonus` carry worded numbers and `_DURATION` needs digits; `invest whatever is
left over each month` has no literal for the amount; `invest 10% of my salary
monthly` binds the percentage to `salary monthly`, so the `of` derivation never
fires.

They had been classified `INTERMEDIATE_SEMANTIC` on the strength of their
**asserted field alone**, without checking that anything computed it, and then
excluded from the pending count. That is the overclaiming shape this report
exists to catch, reproduced inside the report — the fourth appearance of that
class in this stretch. They are `INTERMEDIATE_NOT_PRODUCED` now, owned by
`semantics.py`, and pending: being outside the contract is not a reason for a
case to go unchecked.

### What remains

    AGREE                       12   both witnesses, all with the expected value
    MODEL_ONLY_ACCEPTED         30   the model alone, all with the expected value
    DISAGREE                     1   units adjudication
    SYNTAX_ONLY_UNRESOLVED       1   kept as the asymmetry witness
    INTERMEDIATE_SEMANTIC        2   verified at the mapper boundary
    SCHEMA_GAP                   2   the schema
    INTERMEDIATE_NOT_PRODUCED    4   semantics.py — nothing computes it

`AWAITING_A_PARSER`: **6**. 42 of 52 answered, every one with the value the
corpus expects, and `NO_PARSE_RECORDED` is zero because a case that cannot run
is not a case that is waiting.

Every case now has a boundary that owns it: contract fields end to end through
fusion, intermediates at `propose()`, schema gaps against the contract, the
units dispute unresolved on purpose, and multilingual out of scope. And the
last four are the first genuinely parser-shaped work in a while — worded
numbers are the deterministic layer's job.

The single `DISAGREE` stays unresolved: *"beneath the 12-month moving
average"*, where the schema says `moving_average_window` is measured in
sessions and the phrase says months. Normalising it away needs a declared
trading-calendar convention, and until one exists neither side should win — a
units mismatch resolved by whichever reader is louder is the failure this whole
layer was built to prevent.

## Phase 6, frozen

The parser boundary, as built:

    normalization
    → relation binding
    → semantic derivation
    → independent witnesses
    → fusion
    → contract field | unresolved

Each failure state is owned by the layer that can fix it, and the report names
the owner rather than the symptom.

**Three explicit non-goals, left outside this phase.** Named so a later reader
does not mistake them for oversights:

- **Calendar-month ↔ trading-session conversion.** The one live `DISAGREE`
  needs it and does not get it. Converting months to sessions is a market
  calendar policy, not a parser normalisation, and a units mismatch settled by
  whichever reader is louder is the failure this layer exists to prevent.
- **Schema expansion for unsupported concepts.** `market-cap weighted` and
  `mid-month` stay `SCHEMA_GAP`. The parser understanding something the runtime
  cannot represent is a boundary worth keeping, not a bug to coerce away.
- **Multilingual support.** Deferred with its fixtures preserved. The corpus
  describes the supported scope, not the aspirational one.

### The lesson worth carrying

> **Adding recognition capacity creates new ambiguity surfaces. Every new
> literal class must be tested against neighbouring semantic dimensions, not
> only against the examples it was added to recognise.**

Worded numbers were added so `six months` would read. Both new cases passed, and
two neighbours broke silently: `every two weeks` became a 14-day duration
instead of a biweekly cadence, and `each quarter` became the fraction 0.25.
Neither showed in the totals — the same cases stayed answered with the right
values. What exposed it was that agreements by *two* witnesses fell from 12 to
9 while the answered count held, because three cases had quietly lost their
second witness.

That is the argument for keeping per-state witness counts rather than a single
"answered" number. A total that still sums correctly is the easiest place for a
lost witness to hide.

## Order of work

    real phrasings  ->  tier 1 regressions  ->  tier 2 fixtures  ->  fusion

Written down because the temptation is to invert it. Fusion rules written
against the 144-strategy catalogue, or against sentences we invented, would be
tuned on language that does not stress them — and the finding above is what
that looks like when it is measured instead of assumed.


---

## The harvested corpus — what real language does to Quantify

Attested sentences from Stack Exchange, under CC-BY-SA with every sentence
carrying the URL it came from. Bogleheads returns HTTP 402 to automated
fetchers and reddit blocks them; those are stated preferences, not obstacles to
route around, so neither was scraped.

    harvested              220 sentences, 148 distinct questions
    strategy statements     29
    reached a plan           0
    material semantics      76
    adjudicated             18
    silently dropped         0

## The headline number is not the interesting one

Material-semantic survival is **18/18**. It should not be quoted. Survival is
`HONOURED + NAMED` over everything that got a verdict, and no attested sentence
produced a plan, so nothing could be reduced. The rate says this build is safe
on language of this kind and cannot yet model it. `survival.json` carries that
caution in the artifact itself, and a test requires it to be there whenever
nothing runs.

## Three findings, in the order they matter

**Real strategy statements do not say what to buy.** Of 29 sentences, 16 are
stopped by `assets`. "I contribute about $750/month to my 401k" is a complete
thought to the person writing it: the account is named, the amount is named,
the cadence is named, and the holding is not. The authored corpus never has
this problem because whoever wrote it knew the runtime needed an asset. This is
the single largest gap between the corpus Quantify was built against and the
language it will meet.

It is not obvious that the runtime is wrong to ask. It cannot choose a fund on
someone's behalf — that substitution is what the whole boundary exists to
prevent. But "which holdings?" as the first response to most real sentences is
a product fact worth knowing before the pilot rather than after.

**One sentence executed, and it invested nothing.** "putting a portion of my
cash savings into I-Bonds every year" compiled: I-Bonds, annual cadence,
`amount = 0`, no question asked, and `amount` not even reported among the
applied defaults. The person named a quantity — *a portion* — and would have
been shown a plan that contributes zero.

Closed by refusing the incoherence rather than the sentence: a recurring
cadence says money moves every year and a zero amount says none does, and that
contradiction is visible without reading a word of prose. `once` remains
allowed, because a plan may legitimately model opening capital with nothing
after it. This is the general case of the `$1k` defect — that one was a figure
stated and unreadable, this one a figure implied and never settled — and both
produced a plan indistinguishable from the one asked for except that it
invested nothing.

**Forum prose is a poor proxy for a strategy box.** 191 of 220 sentences that
passed a filter built to admit people describing what they do with money are
not strategy statements at all. They are mortgages, houses, cars, job changes,
questions about tax, and fragments. People writing to a forum describe their
situation; people typing into Quantify describe a strategy. Fourteen searches
aimed specifically at the thin families — triggers, factors, execution timing —
produced six more sentences between them, which is evidence that the language
is not there rather than that the searches were wrong.

The consequence is a limit on what more harvesting can buy. This corpus is
worth keeping and re-running; it is not worth scaling to 500 by loosening the
filters, because the sentences that would let in are not the ones under test.

## What was annotated, and how the answer key is kept honest

All 220 sentences were read. The 29 strategy statements carry a canonical human
interpretation, an expected disposition, and the material semantics the
sentence asserts. None of it was produced by running Discovery and writing down
the answer — an answer key copied from the system under test measures only
self-consistency, and does it while looking exactly like evidence.

The concept vocabulary is deliberately not the schema's: `how often money goes
in`, not `cadence`. A test asserts the two vocabularies do not overlap, because
naming them identically is how the shortcut gets taken without anyone deciding
to take it. The mapping between them lives in `MAPS_TO`, where it can be argued
with.

That mapping was wrong twice, both times in the direction that manufactures
findings against the runtime:

- `which account it sits in` pointed at `asset_location`, which is the mapping
  "bonds in the IRA", when the reader settles `account_type`. Ten sentences
  were reported as dropping a concept that had survived.
- `how often it is put back` pointed at `rebalancing_cadence`, a field the
  proposal layer produces and not a schema dimension at all. Every rebalancing
  sentence would have scored as a drop.

The first was caught by checking a finding before reporting it. The second was
caught by a test that requires every mapped name to exist in the schema. Both
are now in the file where the next person will read them.

## What this does not authorise

The same rule as the authored benchmark. This is a counterexample generator,
not a fifth reopen trigger. The zero-amount plan activated the existing second
trigger — an unsafe silent reduction — and was fixed on the day it appeared.
The unnamed-holding finding activates none of the four: it is evidence for the
pilot to confirm or contradict, and expanding Discovery on the strength of 29
forum sentences would be building for a population this corpus is drawn from
and the product is not.


---

## Rules for evidence

Five rules, each written after a defect that the rule would have caught. They
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

## 4. A workflow definition is not evidence of a check.

Only a reachable execution path that cannot silently skip the check is.

This rule exists because the same failure was found three times in one day, in
three unrelated places, each time wearing a different disguise and each time
leaving the repository looking green.

**Unreachable.** `drift-lane.yml` existed, was correct, and had never run.
GitHub registers `workflow_dispatch` and `schedule` only from the default
branch, and the file lived on a feature branch — so the lane had zero runs in
its lifetime while the pre-Lean gate blocked on "no CI artifact", which read as
an operational to-do rather than as a workflow that could not be started.

**Skippable.** `parser-corpus.yml` mapped `ANTHROPIC_API_KEY`, which was never
configured. Its live-drift step took the `no key configured; exit 0` branch on
every run it ever had: a verification that has never once executed, reporting
success each time.

**Pointed at the wrong thing.** After the serving reader changed provider, the
same step still asked Anthropic and still installed `anthropic`. A check that
runs, passes, and measures a component nothing else uses is worse than one that
does not run, because its green is load-bearing.

The general shape is that a check has three ways to be absent — never started,
started and skipped, started against the wrong subject — and all three look
identical from a build badge. So the questions to ask of any verification are:

    can it start                who or what triggers it, and from which ref
    can it no-op                what happens when a precondition is missing
    what did it measure         which version, which model, which environment

A green tick answers none of them. `tests/test_secret_exposure.py` asserts the
second for provider-calling jobs; the pre-Lean gate asserts the third by
pinning schema, prompt, pipeline, reader and producer onto the artifact; the
first is now enforced by the workflow living on the default branch, which is
the only place the answer can be yes.

**Enforced twice, deliberately.** The structural tests check the repository's
files; two GitHub policies check what the platform will actually run:

    allowed_actions        selected   (github-owned + one named third party)
    sha_pinning_required   true       (every action, including GitHub's own)

Neither layer is sufficient. A test cannot stop a workflow added through the
web UI, and a platform setting cannot explain *why* a job must fail without its
key. They fail in different directions, which is the only reason to have both.

## 5. Missing is not zero.

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
