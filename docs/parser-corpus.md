# The parser corpora, and what the web pass found

Two files, and the difference between them is the whole point.

| | cases | provenance | asserts |
|---|---|---|---|
| [`corpus/parser/cases.json`](../corpus/parser/cases.json) | 206 | self-authored | correctness |
| [`corpus/parser/real_phrasings.json`](../corpus/parser/real_phrasings.json) | 30 | attested where marked | current behaviour |

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

## What could and could not be collected

**Not scraped.** `bogleheads.org` returns HTTP 402 to this environment's
fetcher; `reddit.com` and the Stack Exchange sites are blocked to it outright.
Web search returns prose summaries rather than verbatim user writing.

So the pack is seeded rather than sampled: 11 entries quoted by the user from
cited Bogleheads threads, 8 read in search-result summaries, 11 minimal
variations I wrote on attested forms and marked `variant`. Every entry declares
which. It is not a sample of how people write and the file says so.

Earning that description needs a route to the raw text — an authenticated
fetch, an export, or someone pasting threads in. Until then the pack is a
seed with real roots, not a corpus.

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

## The other defects the pass recorded

**A rebalancing band is invisible.** `5/25` is the standard Bogleheads band —
rebalance when a holding is 5 percentage points, or 25% relative, from target —
and it is shaped exactly like an allocation. The sums-to-100 rule drops it,
which is the correct answer for `12/25` and the wrong one here. No rule over
digits alone separates them; it needs the surrounding words, which means it
belongs above tier 1.

**Nothing binds a ratio to its account.** `401k (50/50), Roth IRA (85/15),
taxable brokerage (70/30)` normalises to three ratios and no accounts. The
binding is a relation, and the contract already has the right shape for it —
this is the `RelationSpec` rule verbatim: meaning depends on which value
belongs to which participant.

**And the parser tokenises that sentence three different ways.** In one string
Stanza keeps `50/50` as a single token, splits `85/15` into `85`, `/`, `15`,
and splits `70/30` into `70/` and `30`. An extractor written against any one
shape is wrong about the other two, and no scoring rule reaches this — it is
below the level the scorer sees.

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

## Order of work

    real phrasings  ->  tier 1 regressions  ->  tier 2 fixtures  ->  fusion

Written down because the temptation is to invert it. Fusion rules written
against the 144-strategy catalogue, or against sentences we invented, would be
tuned on language that does not stress them — and the finding above is what
that looks like when it is measured instead of assumed.
