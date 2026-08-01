# Features

What the system does, by surface. Architecture is in [Architecture.md](Architecture.md);
status and remaining work in [Implementation.md](Implementation.md).

---

## Quantify Library — public research

### Methodologies as executable data

A methodology is a typed AST, not a Python function. Every value that drives a
result comes from the specification: lookback, signal linkage, covariance
estimator, weight bounds, cadence, turnover cap, ordered pipeline, rules,
universe filters.

Sections declare their merge semantics — `unordered_set`, `ordered_sequence`,
`normalized_map`, `conjunction_set`, `weighted_expression`, `scalar` — so a diff
between two versions knows whether order matters.

Currently: `hrp@1..3`, `xsmom@1`. The second family was added specifically to
prove the artifact model generalises; it required **zero new artifact types**.

### Every declaration names its realization

```yaml
- id: concentration_cap
  enforced_by: contract.weight_bounds.max
  expected: "<= 0.25"
```

A rule whose named mechanism does not exist is an `unrealized_declaration`, which
is a **hard publication blocker**. The principle is itself stored as
`assumption/declarations-name-their-realization@1`.

### Evaluation protocols, calendars, policies

`methodology + evaluation protocol = performance`. The protocol declares warmup,
execution lag, cost model, calendar reference, factor model, purging and embargo,
holdout and evaluation period. Sealed holdouts are enforced by panel truncation —
the data is not merely hidden, it is unreachable.

Calendars are versioned artifacts (`calendar/nyse@1`, `calendar/crypto@1`) with
rule-based holidays and declared coverage horizons. They **refuse to
extrapolate** past their coverage rather than guessing.

### Statistical assessment separated from judgement

PSR, DSR, PBO, MinTRL, purged cross-validation with embargo, factor
neutralization. The assessment reports facts and carries no verdict; a versioned
`StatisticalPolicy` applies the standard; a surface-aware `PublicationDecision`
decides what may be said.

Deflation counts every configuration tried, including the ones that produced
nothing.

### Comparability

Two methodology versions are comparable when their output contracts are
interchangeable. `OutputContract.compatibility_breaks()` returns structured
`ContractBreak` objects — field, before, after, **and why it matters** — so the
consequence of a change has one author rather than one per page.

The comparison page follows a fixed order: verdict → where the boundary falls →
blocking differences → why each matters → how comparability could be restored →
only then, eligibility for a performance visual.

### Knowledge model

| | |
|---|---|
| **Claim** | Addressable proposition; status derived from evidence, never stored |
| **Evidence** | Declares its own stance toward a claim, with strength and validity |
| **Assumption** | Declared premise, risk if false, and the test that validates it |
| **Finding** | Synthesis of several evidence items into one conclusion with typed impacts |
| **Investigation** | The question and the work — including inquiries that produced nothing |
| **Erratum** | Correction typed by `correction_type` × `cause_type` × severity |

Evidence declares its stance toward a claim rather than the claim listing its
evidence, so recording disagreement is not gated on the claim's owner.

`Investigation` closes the survivorship gap: `NO_EFFECT_FOUND`, `INCONCLUSIVE`,
`ABANDONED` and `PENDING` are kept apart, and `trials_examined` carries the cost
of a search that concluded nothing into every later deflated statistic.

### Errata are first-class

Published corrections that supersede specific performance records rather than
deleting them. Superseded figures stay reachable; the publication gate blocks
them rather than the library hiding them.

---

## Quantify Scenarios — private workspace

### Describe a plan in prose

The compiler runs ten stages with the language model **quarantined to stage 1**.
Everything downstream is deterministic and would produce the same scenario from
the same parse a year from now.

An unrecognised phrase becomes `unresolved`, never a default. Writing "Google"
produces a question about share class, not a silent `GOOGL`.

### Confirm exactly what will be simulated

```
You stated        verbatim spans, quoted back
We inferred       every default, with why it matters
These conflict    where the description contradicts itself
We still need     open questions, each with its consequence
```

Defaults live in a content-hashed artifact
(`compiler-defaults/us-equity-scenario@1`), pinned rather than "latest", so
recompiling the same words next year produces the same scenario.

`can_simulate` and `can_save` are different gates. An underspecified plan runs
provisionally, because running it is how a reader sees what it means. A
structurally impossible one does not — there is no shape to show for a plan that
cannot execute as written.

**Pairs the compiler must not collapse:** below-200DMA every day vs crossing
below once · equal weight at purchase vs rebalanced · contribution vs additional
cash · earnings date vs first session after · dividends reinvested vs held ·
first calendar day vs first trading day.

### Both returns, always

```
Contributed        $48,000
Final value        $65,941
Time-weighted      +0.30%/yr   "is this a good strategy?"
Money-weighted    +34.89%/yr   "how did I do?"
```

Both correct, for a plan that dollar-cost-averaged through a market that halved
and recovered. Reporting only the first answers a question nobody asked.

### Symmetric benchmarks

Every benchmark receives identical contributions on identical days, under
identical costs, lag and calendar. Returned in **declaration order**, never
sorted by outcome. A benchmark that cannot receive the same flows is reported as
incomparable, never quietly dropped — a set that silently excludes what does not
fit is a curated argument.

Cash is always in the set. It is the comparison nobody asks for and the one that
answers *"was any of this worth doing?"*

### Trial and selection disclosure

Above the fold on every plan: selection basis, candidates evaluated, trials
counted, hidden-selection check, recommendation assessment, derivation
completeness.

> This plan was chosen after comparing 3 candidates' results. All 3 count as
> attempts, and the deflated statistics on this page already account for that —
> the best of several will always look better than it is.

### Life-event templates

`template/rsu-vesting@1` — typed inputs with units, cited assumptions, declared
limitations. Input validation refuses the hundredfold error: a withholding rate
typed as `22` instead of `0.22` is rejected rather than applied.

What it models: vesting in shares (not a purchase), statutory withholding with
the annual threshold split, blackout **deferral** rather than cancellation, sale
proceeds, transaction costs, allocation into the selected rule.

What it declines, with reasons: final income-tax liability, capital-gains
treatment, state and payroll taxes, lot optimisation, plan rules not supplied,
actual brokerage execution.

The 22% statutory default is recorded with its risk: *it is a withholding rate,
not a tax rate*, and anyone above that bracket owes the difference at filing.

### Forward tracking

`PlanObservation` records planned against observed without touching the plan.
Three lanes, so a delayed vest reads as *missing* plus *unexpected* rather than
one shifted row.

Proposals carry `placed: false` and `execution_mode: NONE` **in the payload**,
not in surrounding copy, and each traces to the plan clause that produced it.

### Counterfactuals

*"What would have happened had those executed on the first eligible day?"* leads
with **Constraint isolated: the blackout window**, then lists the dimensions held
identical. A number shown first would read as a verdict on the strategy.

The view downgrades to `PERSONAL_OUTCOME` if any isolation dimension differs.

### Modelling scope travels with the number

Rendered straight from `MissionResult` — the exact statement attached to the
figure is what the reader sees, split into modelled / not modelled / why
excluded, with each exclusion typed and tagged with the runtime that declines it.

---

## Interface

- **Library status matrix** — one glyph row per version, scannable in five
  seconds without opening anything. Symbols carry meaning without colour.
- **Artifact chain** — twelve steps grouped `reasoning │ execution │ judgment`,
  rendered from one payload by both the glyph and the table.
- **Impact graph** — nodes own state, edges own impact, with a fallback table
  carrying identical semantic keys.
- **Run page** — recorded-at-execution beside state-now, never merged, plus
  *"would today's policy agree?"* replaying recorded facts under the current
  standard.
- **Claims, findings, investigations, errata, protocols** pages.
- **Private workspace** — plan list, confirmation, plan detail, proposal
  history, observation timeline, counterfactual, scope panel.

---

## API

`/methodologies` `/protocols` `/policies` `/runs` `/performance` `/errata`
`/trials` `/compatibility` `/holdout-unlocks` `/surfaces` `/current-strategies`
`/project/discoveries` `/project/learning` — all public and impersonal.

`/workspace/*` — private, owner-scoped at the query.

Every endpoint is declared in a **boundary manifest**; an undeclared route fails
the test suite until someone decides which side it is on.
