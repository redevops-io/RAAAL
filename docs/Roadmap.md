# Quantify roadmap v1

Written at the point where the work stopped being architectural. Findings are
classified rather than queued: the bucket decides when something is done, and
most things wait for reality to ask.

## How a finding is classified

| Bucket | Response |
|---|---|
| **Critical correctness** — wrong financial result, privacy leak, data loss | fix immediately |
| **Pilot blocker** — prevents a user completing the intended journey | fix before or during the pilot |
| **Product enhancement** — timelines, charts, conditions, assets | schedule into a phase |
| **Architectural refinement** — stronger invariants, better abstractions | record; implement when duplication or user evidence justifies it |

The fourth bucket is the one that needs the discipline. Every invariant applied
on this branch found something real, which is exactly why "it found something"
cannot be the trigger for the next one.

## What to go looking for

The table above triages something already found. This is the other question —
what to actively look for — and it has **two** answers, not four:

- **Product discoveries.** Things users want. *"Can I backtest RSI?"* *"Can I
  compare two strategies?"* *"Can I import my portfolio?"* These become
  roadmap items.
- **Operational discoveries.** Things that threaten trust. Latency spikes,
  provider outages, confusing explanations, a disclosure nobody reads
  correctly, a recovery taking longer than the drill said.

**There is deliberately no third category for architectural discovery.** Not
because architecture stops mattering, but because an architectural insight that
matters will promote itself: a product feature will need it, an operational
failure will expose it, or a second project will duplicate it. One that does
none of those is recorded and left alone.

This does not contradict the fourth bucket above. That bucket says what to do
with an architectural finding *once you have one* — usually while doing
something else. This says not to go hunting for them. The distinction is the
difference between noticing and searching, and on this branch the searching was
justified because the foundations were still being built. It is not any more.

## Phase 0 — Baseline

See `docs/BaselineComplete.md` for the six criteria. Remaining:

1. Create the first post-provenance production plan through the builder.
2. `deploy/provenance_gate.py` reports PASS rather than VACUOUS — (1) is what
   makes (2) possible; they are one action and one observation.
3. Promote `feat/quantify-repository-baseline` to `main`.

Then compiler work freezes **until pilot evidence names a condition**. Not
permanently: Phase 2 is largely compiler work, and it is gated on someone
asking rather than closed.

## Phase 1 — Pilot

Learn what users want. No architecture changes unless they block users.

**Product:** invite the first cohort · observe · 1-2-4-3 interviews · collect
unsupported requests.

**Operations:** licensing · provider budgets · monitoring · trace retention.

**Metrics:** plans created · plans completed · clarification rounds · abandoned
plans · unsupported conditions · unsupported assets · feature requests.
Nothing else.

### What the telemetry can and cannot answer

Countable today, from `trace`/`span`/`decision`: plans created, plans
completed, clarification rounds (`RETURNED_FOR_ANSWERS` per journey),
abandonment between `/new` and `/save`, and which *fields* were asked about.

**Not** answerable: *which* conditions and assets users asked for that we do
not support. Those arrive as `unclear:` items, and the field id is hashed —
`unclear:#0723c05b67e5` — because it is built out of the user's own words and
this store must not hold them. The hash makes recurrence countable: the same
unsupported thing asked ten times is visibly one thing asked ten times. It does
not say what the thing is.

That is the privacy decision working as intended, and it means two of the seven
metrics come from **interviews, not instrumentation**. Recording the phrases
would answer them faster and would put user financial language back into the
trace store, which is the leak closed on 2026-08-05. If the interviews prove
insufficient, the honest option is a reviewed vocabulary of *categories* —
"volatility condition", "options instrument" — recorded structurally, never the
raw phrase.

## Phase 2 — Product

What users ask for, in the order they ask.

**Conditions:** RSI · MACD · Bollinger · volatility · earnings · recession
indicators. Each is a `SignalGenerator`; the abstraction exists so the first
one did not become the API.

**Assets:** options · futures · crypto · mutual funds · portfolios · RSUs.

**Money flows:** recurring · withdrawals · salary · dividends · rebalancing.
Each is a `FundingPolicy` variant.

**Timeline and charts:** equity curve · cash · contributions · drawdown ·
benchmark · signal markers. The execution ledger already backs the timeline;
these render it rather than recompute it.

## Phase 3 — Platform

Extract only after duplication appears, and only when a second product needs
it — not because the code is elegant. Candidates: compiler primitives,
amendment engine, provenance, telemetry, registry. See
`extract-on-repeated-failure-modes`: the trigger is repeated *failure modes*,
not repeated code.

## Phase 4 — Ecosystem

Research library · public strategy catalog · sharing · version history ·
benchmark library · explain mode · API · SDK.

## Phase 5 — Commercial

Only after pilots. Authentication · billing · organizations · collaboration ·
imports · brokerage integration · licensed market data.

## Deferred, with triggers

An item is inactive until its trigger fires. It does not become active by
existing.

| P | Item | Trigger |
|---|---|---|
| P2 | Worksheet rendering for conditional plans (OBS-1) | a pilot user asks where the detail is |
| P2 | Re-parse per round trip (OBS-2) | evidence of latency complaints or question drift |
| P2 | Description in the URL (OBS-3) | the pilot moves past synthetic data, or a user shares a plan link |
| P3 | Ranking-policy versioning | the first dispute about instrument ordering |
| P3 | Compiled registry artifact | the registry grows enough that compile time shows |
| P3 | Compiler extraction | a second product duplicates compiler logic |
| P3 | `ReaderDecision` — a reader reports `ACCEPTED`, `REJECTED` or `NOT_PRESENT` | a fourth field needs the distinction, or a `DISQUALIFIED_SPAN` entry is added for a third |
| **P2** | Compiler-derived coverage inventory | **trigger met 2026-08-07** — five per-feature entries were added in one slice. Status: validated architectural debt. Implement when the next semantic dimension would otherwise need a sixth manual entry. |

`docs/PilotObservations.md` holds the full observations; this table is the
short form with the trigger made explicit.

## Two architectural candidates, and why they are not being built

Both are justified by repeated failure shapes. Neither is yet justified as a
general abstraction, because every instance so far has come from the same two
semantic dimensions and pilot usage has not yet said which others matter.

**`ReaderDecision`.** Stage 1 has two independent readers and `merge` compares
them. It can express "they disagree" and "only one has an opinion". It cannot
express *why* the other is silent, and there are two opposite reasons:

    NO_VALUE
      ├── NOT_PRESENT      the reader saw nothing; a proposal may fill it
      └── REJECTED         the reader saw the words and ruled them out

Collapsing them let a second reader reintroduce an interpretation the first had
just rejected. `rebalanced monthly` was refused as a contribution cadence by
the regex; the model proposed `monthly` quoting that exact phrase; `merge` read
the refusal as a gap and accepted it, and a $100,000 allocation became
$6,100,000 again.

Shipped instead: a span check, asking whether the words a proposal quotes are
being used in the role it claims. Bounded, closes the live defect, and answers
a different question from the fabrication check that already existed — *did the
model see these words* versus *do those words mean this here*.

**Compiler-derived coverage inventory — trigger met.** `coverage.assess`
enumerates supported constructs: period, conditional purchase, second funding
source, sell leg, conditional amount, allocation method, periodic rebalancing,
stated weights. Each was added after a figure was published for a plan the
compiler could not represent, and five of them arrived in a single slice on
2026-08-07.

That is enough. The status is no longer "possible abstraction" but **validated
architectural debt**: the per-feature list is structurally wrong and the
evidence is in the commit history, not in a prediction. It is still not
pre-pilot work — the entries that exist are correct, and each blocks a figure
by name. Implement when the next semantic dimension would otherwise require a
sixth manual entry. The recurring failure is not a missing entry:

> a semantic dimension exists, the compiler cannot represent it, and it
> therefore disappears from the denominator

The replacement would derive the inventory from the declared semantics and
reconcile each against executed, excluded, unresolved or unsupported. Not built
today because this area has been over-generalised twice on this branch and
narrowed back both times — once counting every unanswered question as a
declaration, which blocked the figure on nearly every first submission.

**The design pressure to carry forward, in one line:**

> Absence is not always ignorance. Sometimes it is a deliberate rejection, and
> collapsing the two lets another reader reintroduce exactly what was rejected.

That has now appeared four times: `provenance@1`'s missing keys against an
empty list, an unstamped body asserting emptiness, an ambiguous sentence
against an unrecognised one, and a refused cadence against an unseen one.
