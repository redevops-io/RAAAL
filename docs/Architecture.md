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

## Coverage evidence

A coverage assertion must derive its evidence from the execution that produced
the coverage, never from a list written alongside it.

```text
executed store calls -> captured method names -> compared with the public
                                                 write-method inventory

not

a hand-written list of method names -> compared with another hand-written list
```

The second form passed while the code it claimed to cover was never called. It
is not a weak test; it is **self-authored evidence** — the test writes down what
it believes happened and then checks its own note.

This has now appeared in eight shapes, and they are one defect:

| Claim | How it was authored by the thing it checked |
|---|---|
| the status vocabulary matches its enum | imported from the enum |
| the migration matches the model | the comparison database was built from the model |
| every diagnostic destination is guarded | parametrized from the list being guarded |
| every write method is exercised | the exercised set was retyped by hand |
| consumers preserve composite identity | reported clean having observed no joins |
| no route reads an operational identity | the scan took its variable list from the assertion |
| every candidate cites one delivery | the journey produced one candidate |
| every error category has a producer | the check called a helper nothing else calls |

The last three arrived in three consecutive slices, which is the argument for
treating this as a class rather than a habit. Each was written *by someone who
had just read the other two* and still landed the same way.

The third of them recurred immediately after being fixed. Rewritten to scan
source instead of calling the helper, the check then counted a docstring
sentence — ``Carries `PublicCode.TRANSITION_INTEGRITY_FAILURE` `` — as a
producer, so deleting the only real one left it green. Prose satisfying a
structural test is the same defect wearing the other failure mode this codebase
keeps hitting. It now walks `ast.Attribute` nodes, and a separate test
constructs a prose-only module to prove comments do not count.

The last is the sharpest. A checker given nothing to inspect returns no
violations, and no violations is exactly what a correct system returns. Silence
means *pass* and *never ran* at the same time, so a run that inspected nothing
must say so:

    assert any("JOIN" in one.upper() for one in issued), (
        "no join was captured, so this check had nothing to inspect")

Applied generally:

- every non-read-only store method must produce captured SQL;
- every consumer check must assert it observed at least one relevant join;
- every registry completeness test must compare against an independent schema
  or type inventory;
- every mutation must be shown to change the behaviour under test, not merely
  to have been applied.

## Single resolution

Any question several components depend on is answered **once**, by one
resolver, into one typed object. Every consumer takes the object. No consumer
re-derives the answer from the inputs.

```text
inputs -> one resolver -> typed object -> every consumer

not

inputs -> consumer A resolves
       -> consumer B resolves
```

Two components that each resolve correctly will still drift, because nothing
compares their answers. Each is right about its own half, and the disagreement
is invisible from inside either one.

This is the shape behind most of the significant refactorings in this codebase,
and it took a deployment defect to see that they were all the same move:

| Question | Was resolved by | Now resolved once as |
|---|---|---|
| which rows does this owner hold | deletion and verification separately | `OwnershipPath`, consumed by delete, verify and export |
| which market data is this | the frame, with provenance fetched beside it | `MarketDataAccess`, carrying frame and provenance together |
| what did this run model | card, worksheet and runtime each rebuilding it | `ScopeDisclosure`, projected into all three |
| which columns are an identity | the schema and each upsert statement | the model, read by the conflict-target translation |
| which database is this | the preflight and the store, independently | `resolve_target` |

The last one is why this is written down. `WorkspaceStore()` substituted a
default before the resolver could read the configured URL, so the preflight
validated PostgreSQL — reachable, migrated, schema parity checked — and the
application wrote to a local SQLite file. Both were correct about the question
they asked. Nothing asked whether they had asked the same question.

**It is not the same as reachability.** Reachability asks whether the live path
calls the control. Constructed invalid state asks whether anything builds the
state the control rejects. This asks a third thing: *do the components that must
agree resolve to the same answer* — and it is invisible to both of the others,
because every component can be individually reachable, individually tested, and
individually right.

**The consequence is that the object is the contract.** Once an answer is a
typed object, "did you remember to fetch the other half" stops being a question
a caller can get wrong — which is why `resolve` returns frame and provenance
together rather than as a pair, and why `OwnershipPath` emits its own SQL rather
than describing a join for each consumer to write.

**Resolution parity is not single resolution.** The database case first funnelled
through `resolve_target`, with `preflight.py` and `engine.py` each reading the
environment and passing the result in. They agreed because both ended at the
same function — not because the question was answered once. That is a weaker
property that looks identical while it holds and gives no warning when it stops.

The full form is `src/deploy/context.py`: one `resolve()`, one frozen
`DeploymentContext` carrying database target, market-data policy, model
configuration and build identity, judged by the preflight and then bound as the
object the process serves under. `create_app` resolves once, preflights *that
object*, and binds it; `tests/test_single_resolution.py` proves the object
bound is the object judged, by identity rather than by equality.

**The enumeration had to come first.** Fixing readers one at a time would have
produced the same partial result as the five tenant-key tables. Two scans of the
syntax tree found ten operational identities across ten modules — including
`ANTHROPIC_API_KEY` and `QUANTIFY_PARSER_MODEL` read by a request handler in
`workspace/routes.py`.

**The invariant test was itself an instance of the defect.** The previous version
scanned `src/` for string constants matching a hard-coded list of three variable
names, and contained an assertion named `test_no_route_reads_an_operational_identity`.
It passed while a route read two identities, because the scan was parametrised
from the same list the assertion was about — self-authored evidence, the failure
described under *coverage evidence*, in the file whose subject is exactly this.
It now derives the inventory from the syntax tree: every environment access in
`src/`, whatever it names, so a variable nobody has thought of is in scope by
default. It found one more the two scans had missed — a module-level cache path
in `market_data/loader.py`, evaluated at import and therefore frozen before any
deployment existed.

**What remains, declared rather than hidden.** Four modules still read the
environment, each recorded with its reason: three vendor or process credentials
that no second component forms a view about, and the manifest expander in
`loader.py`, whose variable names come from data rather than code. That one is
fenced — `RESERVED_NAMES` refuses to expand any identity the context owns, so a
YAML file cannot become a second route to a deployment fact.

## Journey completeness

A guarantee is not established until one real journey crosses every boundary
the way production crosses it.

```text
HTTP -> routing -> runtime -> planner -> store -> database
     -> reload -> view model -> export
```

Every seam can be proven and the chain still broken. That is not a hypothetical
here: the first journey test written against the deployed engine found two
defects before it could assert anything about the subject it was written for.

| Defect | Every component was | The composition was not |
|---|---|---|
| the store opened a default database | preflight correct, store correct, resolver correct | a request reached neither |
| `json_extract` on PostgreSQL | migrations correct, store tests passing | every deployed save failed |
| candidate runs dropped provenance | resolver correct, store correct, verifier correct | the route never carried it |

Each was invisible to unit tests *by construction*, because each unit was
right. What was wrong was the join between them, and nothing that examines one
component at a time can see a join.

**Three lanes, three defect classes.** They are not redundant and one does not
subsume another:

    mechanism tests    logic          does this function do what it says
    journey tests      composition    do the seams compose under real calls
    deployment tests   environment    does the shipped artifact do it

Gate 2 produced defects in all three, and no lane found another lane's.

**A journey must fail by observation, not by comparison.** Restoring the store
defect makes the journey report *no run reached the configured database* —
absent rows. A journey that instead compared two configuration values would be
a unit test with more setup, and would have found nothing the unit test did not.

---

## Content-bound execution

A stored figure must trace to the frame that produced it, not to the source its
producer said it had used.

```text
run_id allocated
  -> resolve() digests the frame it returns
    -> MarketDataAccessEvent
      -> execution_input_digest
        -> run cites the event
```

**Provenance and delivery answer different questions.** `MarketDataProvenance`
describes an authorized source and the decision that permitted it, and is
reusable — two runs a month apart under one policy carry identical records, so
identical records are not evidence of the same delivery.
`MarketDataAccessEvent` is the delivery: this request, these columns, this many
rows, this digest, at this instant.

The gap that closed is narrow and was load-bearing. Until this existed, a run
cited what its *producer declared*, and the producer is the one component whose
claim is not independent evidence — it is precisely what a defect corrupts.
This run path had already been caught dropping the resolver's answer while
every unit test passed.

**The digest is computed inside `resolve`, over the frame it is about to
return.** A caller computing it afterwards would digest whatever the caller was
holding, and whether that is still what was delivered is the entire question.

**What a digest does not prove.** It proves the resolver returned exactly these
canonical rows. It does not prove that downstream code did not drop, reorder,
mutate or substitute them. That span is closed separately by
`execution_input_digest`, taken over the frame handed to the engine: equal
digests mean the transformation was the identity, and an unequal pair must name
a declared, versioned transformation or the run does not verify.

**Three checks, kept apart**, because two can hold while the third fails and one
boolean would hide which:

| check | what a failure means |
|---|---|
| event integrity | the stored body is not the body its hash was taken over |
| run binding | the delivery is evidence about a different execution |
| declared consistency | the run's own claim and the delivery record disagree |

**Canonicalisation is versioned and representation-independent.** The digest is
taken over sorted columns, sorted index and `repr` of Python floats — never over
`to_string()` or a pickle, which change with the library and would report every
stored run as tampered on an upgrade. `.tolist()` rather than cell-by-cell
indexing is a correctness decision as much as a speed one: `repr(np.float64(1.0))`
is `'np.float64(1.0)'` on numpy 2 and `'1.0'` on numpy 1.

**The evidence cannot be deleted out from under a figure.** `plan_run` cites the
event under RESTRICT, so while a run exists the delivery it cites cannot be
removed — a stored figure never becomes unverifiable because something else was
deleted.

**Where the fan-out assertion was vacuous.** The journey produces one run, so
"every run cites the same delivery" held trivially there; the falsification pass
found it by mutating the fan-out and watching nothing fail. The claim now has a
constructed fan-out behind it, with `test_the_fan_out_produced_several_runs`
asserting the premise so the rest cannot quietly become vacuous again.

**The adjacent lifecycle is where a new table actually fails.** The access-event
logic was proven where it is written; deletion, export, transfer, retry and
rollback all iterate table inventories, and the new table joins those by
derivation from the registry and the relationship graph rather than by anyone
remembering it. Derivation is why it should work, and
`tests/test_access_event_lifecycle.py` is why it is known to — the eight cases
each answer a question that could be wrong without any access-event test
failing. Two guards fired on their own: export refused the unclassified table
until retention classified it, and the tenancy lane refused a new store write
method it had never captured.

**The crash mode is chosen, not inherited.** The delivery is written before the
run, so a crash between them leaves an orphan delivery — inert, and cleanable.
The reverse, a stored figure citing evidence that was never written, is what the
constraint makes impossible. Cleanup does not exist yet; when it does, the
database is what stops it removing a cited row, so the guarantee is tested
against the constraint rather than against a policy nobody has written.

> **Closure.** Every market-derived run traces to an immutable access event
> holding the digest of the exact frame delivered, and its modelling scope holds
> the digest of the exact frame consumed. A shared candidate fan-out cites one
> delivery. Stored provenance, delivery and execution input stay distinct and
> independently verifiable, and none of the three is re-derived from current
> configuration when a stored figure is read.

## Two channels

> Every externally visible failure produces two correlated outputs: a bounded
> public failure and a durable private diagnostic. **Neither channel is
> sufficient alone**, and a test that checks only one is incomplete.

```text
driver exception -> classify(SQLSTATE) -> DatabaseFailure -> envelope
                 \_ preserved on __cause__ -> operator log
```

| public | private |
|---|---|
| fixed code | original exception chain |
| safe message | internal reason |
| retry disposition | SQLSTATE where applicable |
| request id | operation and correlation id |
| nothing internal | everything |

Translation lives in `db/engine.py` — the lowest layer that sees the driver
exception, knows the SQLSTATE, can preserve the cause and can normalise
PostgreSQL against SQLite. Routes see application exceptions only, and one
handler serves all 49 of them: a handler that sanitises its own database
exception is a handler that can forget.

**The public category is what a caller may know; the internal reason is what an
operator needs.** A missing parent and a cross-tenant reference are both
`CONSTRAINT_CONFLICT` publicly — separating them is what would disclose that
another tenant holds an id — and `MISSING_PARENT` against
`CROSS_SCOPE_REFERENCE` privately, because one is a client ordering mistake and
the other is an authorization boundary being probed.

**Sanitisation and diagnosability fail independently.** `migrations/env.py`
called `logging.config.fileConfig(path)`, whose `disable_existing_loggers`
defaults to True. That disabled `uvicorn.error` — a logger owned by a different
component — and migrations run in-process at startup, so a migrated deployment
served perfectly sanitised public errors and recorded nothing about any of them:

```text
caller   -> sanitised error
operator -> silence
```

Worse than an obviously broken handler, because it produces the appearance of
correct error hygiene while removing the only evidence an incident could be
diagnosed from. Every public-channel assertion passed. It was found by
asserting on the private channel, and only in a full-file run — a single
isolated request test can never reproduce it, because nothing migrates first.

That makes it a **journey completeness** failure as much as an error-handling
one. Application logging worked. Migration logging worked. Error translation
worked. The startup *sequence* did not, and no component was individually
wrong.

`TestTheExactDefect` is the permanent regression: migrate, provoke a real
PostgreSQL `23503` through a request, and assert both channels in one sequence
— because each assertion was individually satisfiable while the whole was
broken. `TestTheStartupJourney` runs the same shape through `create_app()`, so
the claim covers every stage between process start and first request rather
than the migration alone.

**Three narrower rules, each from a defect this gate produced rather than
found:**

*A wrapper must translate connection failures without swallowing domain
exceptions raised inside its context.* `Ledger._conn` is a `@contextmanager`,
so an exception from the `with` body arrives at the `yield`; catching
`Exception` there turned the ledger's own refusal into an opaque database
failure, and a meaningful 422 into a 500. The engine never had this shape
because it wraps one `execute` call and has no body to capture.

*A compatibility helper must stay valid after the taxonomy moves beneath it.*
`apply.py` asks `is_conflict(exc)` to produce a domain refusal. Once the engine
classified first, that helper was handed a `DatabaseFailure` with no SQLSTATE
attribute, answered no, and the apply path silently stopped refusing. The layer
was correct and every consumer above it was looking for the wrong shape.

*Private logging is best-effort and must never change the public answer.* A
handler raising in `emit` turned a clean 409 into an unhandled 500 — the
private channel failing cost the caller their result. `_record` contains it and
falls back to `stderr`, because falling silent would reproduce the migration
defect from the other direction.

**Enforcement is syntactic, not textual.** Text search failed twice here in
opposite directions: false positives from modules that merely name a driver
while connecting or adapting a type, and false evidence from a docstring
sentence counted as a producer. The syntax tree distinguishes an `except
psycopg.X` from an import, and an `ast.Attribute` reference from prose.

**A forbidden-output vocabulary is itself a thing that can be wrong**, and it
fails in a direction that is easy to mistake for rigour. `DETAIL` was on the
list because PostgreSQL's message contains it — and it collided with FastAPI's
own `{"detail": ...}` envelope key, so an ordinary, correct 404 failed the
check. The pressure at that moment is to exempt the test or broaden the
allow-list, and either would have hollowed out the assertion permanently. The
token is now `DETAIL:`, which is the server's actual field marker.

Three properties make such a vocabulary trustworthy:

- **one authoritative definition** — the startup journey had written its own
  second list, which is how the two drifted;
- **falsified against a real leak**, not merely asserted against safe output —
  injecting the driver text into the envelope fails eleven of these tests,
  including `DETAIL:` itself;
- **exercised against ordinary application responses**, because a list only
  ever run against the error envelope has never met the framework's own
  vocabulary. The false positive was reachable only through a route returning a
  normal 404, which is precisely what the narrow envelope tests never do.

## Closed pilot scope

Two scenarios, both complete through the deployed HTTP journey:

```text
description -> rendered confirmation -> amendments / exclusions
            -> opaque plan id -> run -> worksheet -> reopened plan
```

- historical account and contribution replay
- Roth contribution analysis

**Equity compensation is explicitly unavailable.** `RSUDeclaration` is consumed
by the route that builds it, the card that renders it, and their tests.
Nothing turns a declaration into a scenario, a run or a worksheet: vest events
are not cash flows the compiler understands, and there is no
`compile_rsu_declaration`. The confirmation card was therefore a polished
surface in front of an unimplemented feature — a declaration with no reachable
behaviour, which is the shape this document spends its length removing.

Building the form would have produced a submit button with nothing to submit
to. `/workspace/new` returns `501` with a message naming what was recognised
and what does work, and writes no plan-shaped record: a draft that cannot
become a plan is a record whose only purpose is to look like progress. The 157
component tests are untouched, because the work exists and has not been
assembled into a product path — a different statement from it being absent.

**The confirmation screen was decorative.** The answer and inference radios
rendered *outside* the form that submitted, so a user could read every
question, click every answer, press Save and send none of it. For a scenario
with an open question the button did not render at all and the journey
dead-ended. Every backend test passed, because each built its POST body by
hand — the defect lived exactly in the gap between "the backend accepts a valid
payload" and "a user can produce that payload".

What holds now:

| rule | why |
|---|---|
| one form around the whole surface | a control the browser will not submit is a control the user cannot use |
| answers are `ScenarioAmendment`s | stated by the user, later than the description, never merged into it |
| unsupported prose is `ScenarioExclusion` | proceeding is explicit, preserved, and narrows the stated scope |
| identity is server-generated | two plans may share a title; a title may be edited; runs keep pointing at the same id |
| one feasibility service | the screen and the save path cannot hold different opinions about whether a plan can run |

`assess` refuses a plan that would save with zero runs, and `blockers` names
every outstanding item with what can be done about it — a required
clarification, a separable exclusion, or a material capability limit. "Ready"
may never mean an executable plan does not exist.

## Telemetry is expendable, and was never delivered

```text
route -> Recorder(store=deployment telemetry) -> trace -> span
                                                       -> decision
```

Until Gate 6 nothing in `src/` constructed a `TraceStore`. The one production
entry point called `plan_and_record(...)` without a recorder, and that function
substituted `Recorder(store=None)`, so every span, trace and decision the
runtime assembled was dropped. Twenty-five telemetry tests passed because each
built a recorder *with* a store in its own fixture.

**That made the independence claim vacuous rather than true.** "Deleting every
trace changes nothing about what a worksheet means" holds trivially when there
are no traces. The property worth having is the harder one — with recording
live, breaking it must still cost nothing — and it needed the mechanism to be
reached before it could be asserted at all.

**A second one underneath it.** `Recorder.start()` and `finish()` were called
by nothing in `src/` either. The test helper called them. Spans and decisions
would have been written against a `trace_id` with no `trace` row, so the
correlation spine — the reason a conversation can outlive a request — existed
only in the fixture, and every read API would have returned nothing in a
deployment. Both calls are now idempotent and owned by the service, so a caller
cannot forget and a caller who already opened one is not punished.

**What independence now means, tested:** the same instruction planned in two
identical fresh workspaces, one recording normally and one with telemetry
deleted, read-only, or raising on every write, must produce the same answer and
the same persisted artifacts. Not two sequential plans against one workspace —
planning twice legitimately differs, because the trial total counts the first.

`Recorder.failures` counts writes that did not land. Swallowed *and unreported*
would make a dead trace store indistinguishable from a quiet one, which is
precisely the state this gate found.

**Retention resolves and nothing performs it.** `purge_before` works when
called, the period is configurable, and no deployment schedules it. That is a
strict `xfail` rather than an absence: the gap is a recorded decision.

## Discriminating strictness

> **A control that cannot distinguish prohibited behaviour from valid behaviour
> will eventually be weakened, bypassed or ignored.**

The load-bearing word is *distinguish*. Not "reject everything suspicious" —
correctly separate the two.

This one is different in kind from the six above it. Those ask whether a
control exists, runs, is reached, agrees with its neighbours, or composes.
This asks whether the control **means what it says**, which is a property of
the verifier rather than of the system being verified:

```text
Reachability               the control executes
Constructed invalid state  the control can reject
Single resolution          the control answers one question
Journey completeness       the controls compose
Discriminating strictness  the control rejects only what it intends to
```

Gate 9 found four defects in the system and **five in its own checks**. Each of
the five had the same shape — the weaker version *looked stricter*:

| the check | why it looked stronger | why it was weaker |
|---|---|---|
| grep for a driver name | scans everything | confuses a reference with behaviour |
| call the helper to prove reachability | executes real code | authors its own evidence |
| scan source for a category | reads production files | counts prose as a producer |
| forbid `DETAIL` | catches more strings | collides with framework vocabulary |
| a local token list | self-contained | a second authority that drifts |

The failure mode is not laxity. It is a control whose false positives are
indistinguishable from its true ones, because that is the control someone will
eventually widen, exempt or delete — and the widening is always locally
reasonable. `DETAIL` firing on a correct 404 invites exactly two fixes, exempt
the test or extend an allow-list, and both hollow the assertion out
permanently. Narrowing to the server's own field marker was the only repair
that kept it meaning anything.

This generalises past error messages: policy deny-lists, schema checks,
source-code guards, verification thresholds and anomaly detection all fail this
way, and all of them fail *quietly*, because a control nobody trusts is not a
control anybody reports.

The usual framing of false positives is that they are annoying. Operationally
they are worse than that, and the sequence is mechanical:

```text
noise -> ignored -> weakened -> removed
```

Every developer who touches the control experiences it as an obstacle to
something they know is correct, and each individual widening is locally
reasonable. That is what nearly happened to `DETAIL`.

**The dual principle.** Broadness is not strength; precision is. Nearly every
architectural correction recorded in this document moves the same way:

| from | to |
|---|---|
| hashes equal | canonical values equal |
| `proposal_id` | `(owner, proposal_id)` |
| snapshot label | content digest |
| frame returned | frame consumed |
| provenance from configuration | provenance from the record |
| `grep` | the syntax tree |
| `DETAIL` | `DETAIL:` |

None of those is a broader check. Each is a narrower one that means something
the looser version could not distinguish — which is why the system has become
more precise rather than merely more strict.

**What actually increased is information.** Every replacement above carries
more of it than the thing it replaced. `(owner, proposal_id)` carries a tenant
that `proposal_id` had thrown away; a content digest carries a realization that
a label could not name; a consumed frame carries what an execution received
rather than what a resolver returned. The same movement runs back through the
whole project — typed missions over prompts, mission graphs over chat history,
execution artifacts over conversations, `OwnershipPath` over an inferred join.

> **Correctness comes from preserving semantic information, not from adding
> validation.** Validation is usually compensating for information discarded
> earlier, and the compensation is always weaker than the information would
> have been.

`plan_run` is the clearest case. Without an `owner` column, every ownership
question became a join, deletion needed a hand-maintained order, and a
truncated `OwnershipPath` silently deleted another tenant's rows. Three
separate controls were compensating for one discarded column. Adding the column
did not make any of them stricter; it made two of them unnecessary and the
third mechanical.

**The two kinds are not the same, and the split follows the two planes.** Five
of the seven corrections above preserve more information in the *artifact* —
that is system-plane work, and the validation genuinely shrinks. Two of them,
`grep`→AST and `DETAIL`→`DETAIL:`, preserve more information in the *reading*:
nothing about the system changed, only the verifier's representation of it
stopped discarding structure. That is verification-plane work, and it does not
make anything simpler — it makes the check mean what it claimed.

> Preserve semantic information in the artifact wherever the system owns it.
> Preserve structural information in the observation wherever the verifier
> reads it.

Both are information-preservation; only the first reduces how much checking is
needed, which is why the two are worth keeping apart. Stated as one rule it
would predict that fixing a verifier simplifies the system, and it does not.

**The overlap is where it becomes visible.** Narrow tests keep vocabularies
apart; the startup journey put four of them in one request —

```text
framework response language + application error envelope
    + database driver disclosure + startup logging configuration
```

— and two of the five defects above were reachable only there.

> **Closure.** Every database failure crossing a public boundary is translated
> at the engine into one of six stable semantic categories with explicit
> retryability. The application exposes only fixed safe messages and a
> correlation id, while operators receive the original chained cause and the
> internal reason. Driver text, SQLSTATEs, constraint names, key values, paths
> and tenant-existence details cannot reach callers. Migrations cannot disable
> the operator channel, logging failures cannot alter the public response, and
> every declared category has a real production producer.

**What this does not yet prove.** The tests establish the logic against a real
engine in this process. They do not establish that the *deployed* logging stack
preserves it — uvicorn installs its own configuration, and the container is
where a handler is actually attached. Running the same
migration → request → failure → public response + private log sequence in the
built-image lane is on the backlog, and is the one claim above that currently
rests on a process this codebase configures itself.

## The invariants, and where they came from

These are not principles chosen in advance. Each was named after the same
defect appeared enough times to be a shape rather than an incident.

| Invariant | The question it asks |
|---|---|
| Reachability of enforcement | does the live path call this control? |
| Ownership reachability | can every record's owner be proven, or is it declared global? |
| Constructed invalid state | has anything built the state this control rejects? |
| Coverage evidence | is the proof produced by the thing it certifies, or written beside it? |
| Single resolution | do adjacent components resolve the same question once? |
| Journey completeness | has the composed production path actually run? |
| Discriminating strictness | does this control reject only what it intends to? |

They overlap at the edges and are not a partition. Two observations about the
fit, recorded because a taxonomy that is claimed to be exhaustive and is not
becomes a way of *not* looking:

**One recurring defect is not on the list.** Fixing the instance that was found
is not fixing the class — five tenant-key tables found one at a time, two
routers with the same ungated read, two immutable bodies with the same
overwrite. The remedy is always an inventory derived from an independent
source, and every invariant above ends up prescribing one. It may be the
general case and the seven may be its symptoms; that is not yet clear enough to
write down as a law.

**The seventh is about a different subject.** Six describe the system;
*discriminating strictness* describes the things that check the system:

```text
System plane        does the system do the right thing
  reachability of enforcement
  ownership reachability
  constructed invalid state
  coverage evidence
  single resolution
  journey completeness

Verification plane  can the check tell what it claims to tell
  discriminating strictness
```

That is not a tidy addition to a taxonomy — it is an admission that the
verifiers are software too, with their own defect classes, and that a suite
reporting green is evidence about the suite before it is evidence about the
code. Gate 9 produced more defects in its own checks than in the code they were
written to check.

> **The seventh invariant names a class of verifier defects. It does not
> establish that every existing verifier has been examined for them.**

What is established is narrower and worth stating that way: one lane's
verifiers were not trustworthy, and there is now a repeatable method for
auditing the others — self-authored evidence, prose counted as behaviour,
vacuous premises, duplicated vocabularies, checks that cannot separate valid
from invalid, and mutations that do not alter exercised behaviour. Applying it
to the remaining lanes is outstanding work, not a completed claim.

**Several defects belong to more than one.** The truncated `OwnershipPath` was
ownership reachability, single resolution *and* a consumer invalidated by a
correct migration. Categorising it once would lose two of the three reasons it
happened.
