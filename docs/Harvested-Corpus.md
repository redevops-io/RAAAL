# The harvested corpus — what real language does to Quantify

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
