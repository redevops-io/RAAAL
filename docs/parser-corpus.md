# The parser corpora, and what the web pass found

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

## Order of work

    real phrasings  ->  tier 1 regressions  ->  tier 2 fixtures  ->  fusion

Written down because the temptation is to invert it. Fusion rules written
against the 144-strategy catalogue, or against sentences we invented, would be
tuned on language that does not stress them — and the finding above is what
that looks like when it is measured instead of assumed.
