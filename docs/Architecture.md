# Architecture

**Status:** baseline as of 2026-07-31 · 857 tests passing

Quantify is an event-driven financial simulation and research runtime built on
versioned knowledge and execution artifacts. It is not a portfolio optimizer and
not a backtesting dashboard.

The central principle:

> Every meaningful behaviour is declared and realized; every declaration names
> the mechanism that enforces or checks it.

Two failure modes are closed by construction — **behaviour without declaration**
and **declaration without behaviour**. The second is the harder one, and it is
why nearly every artifact here carries a realization check.

---

## 1. Two product surfaces

### Quantify Library — public, impersonal

Methodologies, claims, evidence, findings, assumptions, runs, assessments,
policies, publication decisions, errata, benchmark results. No holdings,
compensation, taxes, vesting, contribution amounts, age, income or objectives.
No personalized ranking.

### Quantify Scenarios — private workspace

Natural-language intent, personal cash-flow schedules, RSU vesting, account and
tax semantics, historical simulation, forward paper tracking, proposals,
counterfactuals, saved plans, symmetric benchmark sets, trial accounting.

### The boundary is a direction

```
private artifact  ──may cite──▶  public artifact
public artifact   ──never────▶  private artifact
```

Enforced in `src/mission/boundary.py`. An artifact kind with no declared
visibility **raises** rather than defaulting — a new type quietly defaulting to
public is the one mistake that file exists to prevent.

Separate router, templates and store (`src/workspace/`), so the boundary is in
the file tree rather than only in a document. A shared table with a `visibility`
column was rejected: it is one forgotten predicate in one query away from being
no boundary at all, and that query gets written by someone who does not know the
rule exists.

A private scenario reaches the public library only through `extract_rule`, which
strips an enumerated list of personal fields and then **recursively scans** the
result for prohibited keys and private references at every depth. It is not
called `promote`: a Mission is a person's financial situation, a methodology is a
rule, and what someone did with their salary is not research.

---

## 2. System flow

```
Intent
  │  SelectionBasis — how the plan was chosen, which sets the trial count
  ▼
Scenario Compiler          stated / inferred / contradictions / unresolved
  ▼
ScenarioSpecification      the layer between language and execution
  ├── FlowSchedule         this person's amounts and dates
  ├── Methodology          the market rule, as a typed AST
  ├── EvaluationProtocol   warmup, lag, costs, calendar, holdout
  └── ExecutionEnvironment composed runtimes
  ▼
Mission Runtime
  ├── historical simulation      TWR and MWR, never one alone
  ├── assessment                 facts only
  ├── policy evaluation          a standard applied to facts
  ├── publication decision       what may be said, on which surface
  ├── forward observations       what actually happened
  ├── proposals                  never executed
  └── counterfactuals            what a constraint cost
  ▼
Immutable artifacts and graph
```

---

## 3. Artifact model

### Knowledge layer — *what do we believe, why, and what changed?*

| Artifact | Purpose |
|---|---|
| `Investigation` | The question and the work, including inquiries that produced nothing |
| `Finding` | Synthesis of several evidence items into one conclusion with typed impacts |
| `Claim` | Addressable proposition; status **derived** from evidence, never stored |
| `Evidence` | Support, qualification or contradiction, with provenance and stance |
| `Assumption` | Declared premise, what fails if false, and the test that validates it |
| `Erratum` | Correction with `correction_type` × `cause_type` × severity |

**Two senses of "implemented" are worth keeping apart here.** `Investigation`
exists as a *knowledge artifact* — persisted, queryable, versioned, rendered, and
validating its own outcomes. It does not yet exist as a *durable unit of work*:
lifecycle transitions, assignment, Discovery-driven creation and
conclusion-to-Finding routing are not built. A reader who sees "workflow
incomplete" should not infer the artifact is absent. See
[Implementation.md §5](Implementation.md).

`Investigation` exists because a finding can only record that something *was*
concluded. `InvestigationOutcome` keeps `NO_EFFECT_FOUND`, `INCONCLUSIVE`,
`ABANDONED` and `PENDING` apart — flattening them is how a research record
acquires survivorship bias, and `trials_examined` is what stops an unrecorded
search inflating every later Sharpe ratio in a lineage.

### Execution layer — *what happened, under which declared conditions?*

| Artifact | Purpose |
|---|---|
| `Intent` | Objective plus `SelectionBasis` |
| `ScenarioSpecification` | Canonical compiled scenario; splits rule / money / evaluation |
| `Methodology` | Typed executable AST |
| `EvaluationProtocol` | Warmup, lag, cost model, calendar reference, purging, holdout |
| `TradingCalendar` | Versioned session semantics, refusing to extrapolate past coverage |
| `TaxRuntime` | Versioned taxation semantics |
| `AccountRuntime` | Contribution limits, withdrawal rules, employer match |
| `CashFlowRuntime` | Reusable event semantics; personal values stay in `FlowSchedule` |
| `MarketDataRuntime` | Sourcing and interpretation policy |
| `RealizedData` | What a run actually received |
| `ExecutionEnvironment` | Typed composition of runtimes |
| `Run` | Immutable historical fact with diagnostics and verdicts |
| `Assessment` | Statistical facts only — no verdict |
| `StatisticalPolicy` | Versioned interpretation thresholds |
| `PublicationDecision` | Surface-aware |
| `PlanObservation` | Planned versus observed, without mutating the plan |
| `Proposal` | Non-executable analytical opportunity with an immutable lifecycle |
| `ComparabilityVerdict` | Class, blockers, isolated dimension, unchecked dimensions |

---

## 4. RuntimeArtifact lifecycle

Every runtime shares one lifecycle (`src/runtime/base.py`), so a new one cannot
invent its own versioning semantics:

```
family · id · version · content_hash · compatibility_hash
assumptions · limitations · realization_checks
declared_form() · comparable_form()
undefined_without · interpreted_with
```

### Two hashes, deliberately

```
content_hash        exact identity — any change, including prose
compatibility_hash  execution-relevant identity — only what moves a number
```

A citation correction mints a new version whose results stay comparable. A tax
rate change breaks comparability while the prose barely moves. One hash would
have to choose between lying about identity and refusing a valid comparison.

The default is the safe direction: a runtime that has not thought about the split
refuses comparisons it might have allowed.

### Three levels of identity

```
market_data/vendor        family   — one semantic runtime through time
market_data/vendor@2      version  — a specific declaration
RealizedData(...)         instance — what a run received
```

Mirroring methodology family / version / run exactly. Extending a coverage
horizon mints a version and changes nothing about meaning, so results stay
comparable.

### Semantic dependence, not software dependency

```python
undefined_without = ("account",)   # declarations have no truth value without it
interpreted_with  = ("account",)   # meaningful alone; another runtime may restrict it
```

"Gains are not taxed" is correct in a 401(k) and an admission of incompleteness
in a taxable account. The sentence has no truth value until an account runtime is
present — which is a different claim from "instantiate this first".

---

## 5. Composition

`ExecutionEnvironment.validate_composition()` answers the question no runtime can
answer about itself: two individually valid runtimes composing into an invalid
environment.

Rules are **registered artifacts** with `id`, `category`, `severity`, `affects`
and `description`, so the set of known-bad compositions is enumerable rather than
buried in conditionals. Categories: `SEMANTICS`, `TEMPORAL`, `TAX`, `ACCOUNT`,
`CORPORATE_ACTION`, `DATA`, `EXECUTION`.

Current rules:

| Rule | What it catches |
|---|---|
| `UNDEFINED_WITHOUT` | A runtime whose semantic precondition is absent |
| `GAINS_TAXED_IN_SHELTER` | Annual capital-gains tax inside a tax-deferred account |
| `SESSION_ALIGNMENT_MISMATCH` | Crypto-aligned data under an NYSE calendar |
| `ADJUSTMENT_POLICY_CONFLICT` | Splits applied to already-adjusted prices |
| `FLOW_KIND_UNSUPPORTED_BY_ACCOUNT` | Employer shares into an account that cannot hold them |

The environment orchestrates; runtimes own their semantics. Predicates are asked
*of* the runtime (`account:tax_deferred`) rather than the environment inspecting
fields, which is what stops `ExecutionEnvironment` becoming a dependency injector
with finance rules hidden inside it.

---

## 6. Comparability

### Three comparison classes

| Class | Meaning |
|---|---|
| `STRATEGY_EFFECT` | Everything outside the rule is identical; a difference **is** the rule |
| `PERSONAL_OUTCOME` | Personal schedules differ; comparable, attribution **not** isolated |
| `CONSTRAINT_EFFECT` | One named constraint differs; all its dependencies equal |

`PERSONAL_OUTCOME` exists so a user can ask *"monthly contributions or my
year-end bonus?"* — a real question — without the platform claiming the answer
identifies a better strategy. A differing evaluation period defeats comparison
entirely rather than weakening it.

### The dimension registry

`ISOLATION_DIMENSIONS` was a hand-maintained tuple and was wrong twice —
`allocation_rule` and `data_snapshot` were both missing, and both silently let a
comparison claim attribution it did not have. It is now **derived** from
`src/comparison/dimensions.py`, where each dimension declares:

```
id · source_kind · causal_label · extractor
supports{class → MUST_EQUAL | MAY_DIFFER | DEFEATS_COMPARISON}
depends_on · isolation_eligible · disclosure_template
```

Both artifact classes register side by side — `flow_schedule` is a
`SCENARIO_ARTIFACT`, `tax_treatment` is a `RUNTIME` — rather than forcing
personal declarations to masquerade as reusable policy.

`depends_on` is **derived** from each runtime's own declarations —
`undefined_without` plus the `interpreted_with` relations explicitly marked
`affects_causal_isolation` — so the fact lives in one place.
`reconcile_dependencies()` fails the suite if the two diverge, and
`unreconcilable()` reports dimensions whose runtime type is not registered, so
an unchecked dependency cannot hide behind an empty disagreement list.

Not every interpretation relation is causal: an account may *refuse* a cash flow
without changing what the flow means, so a differing account does not make "only
the schedule differs" false. Marking every relation causal would defeat isolation
on relations that do not bear on it.

Isolation succeeds only when exactly one *eligible* dimension differs **and**
every dimension it depends on is equal. Without the second condition, "only tax
differs" could be reported as a cause when tax's meaning was jointly determined
by an account that also differed.

`unchecked_dimensions` reports what a comparison could not examine. A hole in an
attribution claim is bad; an invisible one is worse.

---

## 7. Statistics, policy, publication — three separate decisions

```
Assessment          facts        PSR, DSR, PBO, MinTRL, trial count, warnings
StatisticalPolicy   standard     versioned thresholds applied to those facts
PublicationDecision surface      what may be said, and where
```

The assessment payload contains no `passed`, `valid` or verdict. A low DSR is a
finding, not a defect.

Hard blockers are reserved for conditions that make a number *uninterpretable*:
look-ahead, invalid provenance, failed reproducibility, contract violation,
missing trial record, **unrealized declaration**, incompatible
methodology/protocol, severe economic degeneracy.

A policy `PASS` can still yield publication `BLOCK` when economic diagnostics show
the figure is misleading.

### Trial identity

```
hash(methodology, protocol, objective, data_partition, execution_assumptions)
```

Re-running one configuration is reproducibility testing, not search.

| `SelectionBasis` | Trials counted |
|---|---|
| `STATED_PREFERENCE` | 1 |
| `BEFORE_RESULTS` | 1 |
| `AFTER_RESULTS` | every evaluated candidate |

Platform-generated candidates require declared, non-performance
`generation_constraints`. A candidate evaluated but never shown raises
`HiddenSelection` — measuring alternatives privately and presenting the survivors
makes the platform the researcher, reporting one trial for a search it conducted.

---

## 8. Cash flows and returns

Everything the weight-based engine produces is a **time-weighted return**, which
removes the effect of contribution timing by construction. A Mission has external
cash flows by definition, and contribution timing is exactly what the user is
asking about.

So `src/mission/` accounts in **shares and cash** and reports both bases, always:

```
time_weighted   "is this a good strategy?"    — independent of when money arrived
money_weighted  "how did I do?"               — given when money arrived
```

A vest is an **in-kind grant**, not cash-plus-purchase: modelling it as a
purchase invents a session of slippage and a trading decision nobody made.
Withholding is a reduction in granted shares, not tax modelling.

Benchmarks receive **identical flows**. Comparing a $2,000/month plan against a
lump-sum benchmark compares strategy *and* schedule, and the difference cannot be
attributed to either.

---

## 9. Non-recommendation posture

`is_recommendation` is **derived**, never declared. An earlier version emitted the
literal `False`, which asserted the platform's own compliance and could not be
wrong.

Nine checks, each recorded as `DERIVED` from the payload or `DECLARED` by the
caller, with `derivation_complete` making the weaker verdict visible. The
strongest is order preservation — if the payload does not match the order the
benchmark set was *declared* in, it is a ranking whatever the prose says.

### No execution capability

`ExecutionMode` has one member: `NONE`. Adding a second is a reviewed change to a
type rather than a different string in a payload.

`Proposal.placed` is a read-only **property**, always false. A field defaulting to
`False` trusts every future caller not to set it. `ACCEPTED` means the person
recorded that *they* acted — a resolved proposal still reports `placed: false`.

### Backtest and forward cannot be linked

`SegmentedPerformance` raises `LinkedSeriesRefused` from `linked_series`,
`combined`, `since_inception` and `full_history` alike. One guarded method with
three unguarded aliases is not a guard.

---

## 10. Scope and limitations

Every `MissionResult` carries its modelling scope, and the store **refuses** to
record a run without one:

> A recorded figure without a statement of what it excludes will be read as
> excluding nothing.

Exclusions are typed, because listing them together makes a correct treatment
look like a gap:

| Reason | Meaning |
|---|---|
| `NOT_APPLICABLE` | Does not arise here. Nothing is missing |
| `OUT_OF_SCOPE` | Arises, deliberately not modelled. A real gap, stated |
| `UNRESOLVED` | Arises, needs an input nobody supplied. Answerable by asking |

A limitation is **refined by the composed environment**: capital gains going
untaxed is `NOT_APPLICABLE` inside a 401(k) and `OUT_OF_SCOPE` in a taxable
account. Same runtime, two readings; only the environment can tell them apart.

---

## 11. Forward tracking

Plans are immutable. Mutating a plan when reality diverges destroys the only
thing tracking is for.

```
Plan         declares what should happen
Observation  records what happened
Status       derived from both
```

Reconciliation matches on `(date, kind)`, not order — a vest three weeks late is
a **missing** expectation *plus* an **unexpected** arrival, which is what it is.
Positional matching would call it a match.

```
Proposal lifecycle:  OPEN → ACCEPTED | IGNORED | EXPIRED | SUPERSEDED
```

Resolved proposals are immutable copies. Expired ones are kept: three proposals
expiring behind a closed trading window is the evidence a constraint cost
something, visible only if they are still there.

---

## 12. UI architecture

> State is scannable; reasoning is inspectable. **Visualise the graph, not the
> returns.**

**Shared design tokens, distinct layouts.** `src/web/templates/shared/_tokens.html`
is the single definition of typography, spacing, colour and status colour. Both
surfaces include it; the workspace adds `--private`. Layout is deliberately *not*
shared and a test enforces that — a dense research surface and a decision surface
should not resemble each other in structure.

**Nodes own state; edges own impact.** Relation semantics are declared once in
`src/web/semantics.py`; an undeclared relation raises rather than defaulting to
harmless. Diagram and fallback table expose the same canonical key
`(source, relation, target, effect, directness)`, so equivalence is checked on
*meaning* rather than on counts.

**State and adversity are orthogonal.** `State` drives the glyph; `Adversity`
(`NONE`/`ADVISORY`/`AFFECTING`/`BLOCKING`) drives the consequence sentence.
"Blocked at errata" flattened three different meanings into one.

**Performance visuals require four gates** — comparable verdict, declared
performance class, publication permits the surface, historical and forward series
separated. An absent chart renders as a checklist, so its absence states a fact
rather than looking unfinished.

**Pages compose; they do not derive.** `tests/test_pages.py` feeds each page a
view model contradicting what a template could infer from names — a `REFUTES`
edge carrying `ADVISORY`, a boundary marked comparable that still lists
blockers — and asserts the page renders what it was given.

---

## 13. Regulatory posture — the constraint that shapes the architecture

The public library rests on the **publisher's exclusion** (*Lowe v. SEC*, 472
U.S. 181). The SEC's position is that disseminating hypothetical/backtested
performance to a public audience generally cannot comply with the Marketing Rule
— nine advisers were fined $50k–$175k each on 11 Sep 2023, and
[17 CFR 275.206(4)-1(e)](https://www.law.cornell.edu/cfr/text/17/275.206\(4\)-1)
explicitly includes model portfolios and backtests in "hypothetical performance".

> **Invariant for the public library: no per-user portfolio-tailored output.**

The private workspace crosses that line **deliberately**, which is why it is a
separate surface with its own boundary, not a feature added to the library.
Losing the exclusion retroactively would convert every published backtest into a
potential violation.

Four rules that shaped the data model:

1. **GIPS forbids linking actual to theoretical performance.** Enforced by making
   the linked return uncomputable, not by a style guide.
2. **AI-washing is the most active enforcement theory** — Delphia ($225k), Global
   Predictions ($175k), Rimar (~$524k + officer bar), Presto. The violation is the
   gap between description and system, and two of four turned on undisclosed
   third-party provision or undisclosed human involvement. Mitigation: an AI Use
   Register that marketing copy is generated *from*.
3. **Preserve prior versions of public pages.** Two of the nine 2023 respondents
   were additionally charged for not preserving performance advertisements.
4. **SR 11-7 → [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm)**
   (17 Apr 2026). Its model-inventory and documentation fields *are* the
   methodology-version schema, so institutional diligence is a database export.

**EU AI Act:** strategy generation is not Annex III high-risk; the live obligation
is Art. 50 transparency from 2 Aug 2026.

The engineering posture is enforced where it is enforceable — surfaces, the
derived recommendation assessment, and the endpoint boundary manifest. It is not
a legal conclusion, and securities-counsel review gates external access.

### Licensing

`context-runtime` and `redevops-rag` are AGPL-3.0. AGPL §13 extends copyleft to
**network use**, so serving users over a network with AGPL code obliges offering
the corresponding source of the combined work. This project is AGPL-3.0 for that
reason. `CR-enterprise` is proprietary and separate.

---

## Recognition is not representation

A compiler stage can be correct at both ends and wrong between them:

```
Recognized    the parser read the phrase             correct
Represented   it reached the compiled scenario       MISSING
Validated     checks ran over what was represented   vacuous
Compiled      a scenario was produced                correct
Confirmed     the screen quoted the phrase back      correct
```

Three defects had exactly this shape, and none was a parser bug. A user said
"hold the dividends as cash" and the confirmation screen quoted it back under
*you stated*, while the compiled scenario contained no trace of it. The same was
true of "simple" versus "exponential" moving average, and of taking a conditional
buy out of the contribution versus from additional cash — the compiler's own
documented example of two economically different readings, whose answer it had
asked for and never carried.

Each produced an **identical content hash for two materially different
strategies**, which is the same failure as a Roth and a taxable account
comparing as identical.

The rule that closes it: **every field stage 1 can recognise must either name
where it lands in the compiled form, or say why it does not travel.** A field in
neither list fails a test. Presence is checked against the canonical form rather
than the object, because a value excluded from the canonical form is invisible
to identity, comparison and replay whatever the object holds.

This is the compiler's version of the invariant the rest of the system already
has: *every declaration names the mechanism that realizes it.* Recognition
without representation is declaration without behaviour.

---

## Mechanically falsifiable claims

> Every material claim made by a runtime, benchmark, test, worksheet or
> publication must have a mechanical check capable of disproving **that exact
> claim**.

This is stronger than coverage, and it is not the same as having a test. A test
asserting a claim is itself a claim, and it earns nothing by passing — it earns
credibility only when it can fail for the specific defect it says it detects.

Most of the defects found in this system were being asserted *correct* by a
passing test at the time. A comparison that isolated nothing passed a test named
for isolation. A fixture called "a full comparison reports nothing unchecked"
pinned neither account, calendar nor market data. The tests were not weak; they
were unfalsifiable.

So each claim below names the mutation that must break it:

| Claim | The check that can disprove it |
|---|---|
| This comparison isolates one dimension | Leave a required runtime unpinned; isolation must be refused |
| This plan replays unchanged | Edit the persisted body; replay must fail |
| Every recognised field affects canonical meaning | Add a recogniser with no destination; the test must fail |
| Opening a worksheet does not reinterpret it | Inject compilation into the read path; the suite must go red |
| Polars and canonical execution are equivalent | Seed a divergent implementation; the equivalence check must catch it |
| The evidence record is intact | Edit a stored field and leave its hash; replay must report `HASH_MISMATCH` |
| The corpus measures the compiler | Remove an import; the coverage test must fail |

Each of those mutations was performed, and each produced the failure it was
supposed to. A check that has never been seen to fail is a check nobody has
tested.

The corollary shapes how new work is judged: it should either create a visible
user capability, or turn an existing claim into something mechanically
falsifiable. Adding an assertion that cannot fail adds nothing but confidence.


## Ownership reachability

Every persistent record must have a mechanically provable ownership path, or be
explicitly global. There is no third option, and "obviously it belongs to the
user" is not a path.

```text
GLOBAL                    no owner; survives every deletion

DIRECT_OWNER(owner)       carries the column

INDIRECT_OWNER(
    plan_run.plan_id  ->  plan.plan_id  ->  plan.owner
)
```

The third case is why this is an invariant rather than a convention. `plan_run`
holds every simulated result a user has produced and carries no `owner` column;
it is user-owned entirely through the graph. A deletion written the obvious way
— `WHERE owner = ?` — removes nothing from it and reports success. Export misses
it. Tenant-isolation checks pass because they never look.

**The path is executable metadata, not documentation.** `OwnershipPath` is a
declaration that emits its own SQL, and deletion, export and verification all
consume the same object. An ownership rule written in a comment and hand-coded
in one function is a rule the other consumers will each get subtly wrong.

    every table is classified          checked against sqlite_master
    every indirect table has a path    checked against the classification
    the path actually reaches rows     checked by deleting and re-reading

This is the fourth instance of one shape. A control is not implemented until the
production caller cannot bypass it:

| Declared | Bypassed by |
|---|---|
| account contribution rules | `ACCOUNT_IMPLEMENTED` was empty; nothing enforced them |
| history-aware planning | `intent.plan` had no live caller |
| corporate-action pinning | `corporate_action_ref` was required and never resolved |
| market-data egress policy | `_prices()` read an unmanifested file directly |
| record ownership | deletion reached only tables with an `owner` column |

Each was found the same way: by asking what the live path actually calls, rather
than what the module declares. The generalisation is **reachability of
enforcement** — every declared control needs one end-to-end test proving the
live path reaches it, and one mutation proving that bypassing it fails.

## Constructed invalid state

A guard is not tested until a fixture deliberately creates the state that only
that guard is meant to reject.

This is a different question from reachability, and the two fail in different
places:

```text
Reachability of enforcement   does the live path call the control?
Constructed invalid state     has anything built the forbidden state,
                              and shown that this control rejects it?
```

A control can pass the first and fail the second. It sits on the production
path, it runs on every request, and it has never once been given input it
should refuse — so deleting it changes no test result. Mutation testing does
not find these: the mutation is applied, the suite still passes, and the honest
conclusion looks like "this code is dead" when it is in fact "this code is
untried".

The distinction matters most for guards protecting states the ordinary API
cannot produce. Those states have to be built below the API boundary, which
feels like testing the wrong thing until the day a migration, a new caller or a
changed constraint makes them reachable.

| Guard | State a fixture had to construct |
|---|---|
| migration refuses an unresolvable owner | a `plan_run` orphaned under the old schema |
| migration refuses an ambiguous owner | one `plan_id` held by two owners |
| `record_run` derives or refuses | a run naming a plan that does not exist |
| tenant identity in every key | two owners using identical ids |
| composite `OwnershipPath` | a path joining half its parent's key |
| transition affected one row | a conditional update reporting zero |
| decimal mirror agrees with payload | a payload key renamed or rounded |
| content hash covers the body | a payload edited with its hash left alone |

Each row is a guard whose removal left the suite green until the state in the
right-hand column existed. Four of them were found in a single afternoon by
asking the question directly, which suggests the shape is common rather than
exceptional.

**A correct migration can invalidate a correct consumer.** Widening every
tenant key was right, and it silently broke `OwnershipPath`, which joined on the
single-column identity those keys used to have. The join stayed valid, kept
returning rows, and returned another tenant's. Nothing in the migration was
wrong; the defect was in code that still assumed the old identity shape. So the
standing checks validate not only the keys but the consumers of those keys.
