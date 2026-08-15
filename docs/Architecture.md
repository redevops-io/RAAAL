# Architecture

What the system is, how it is deployed, and where it is going.

> Consolidated from `Architecture.md`, `Architecture.md`, `Architecture.md`, `Architecture.md`, `Architecture.md`, `Architecture.md`.
>
> Six documents described one system from different distances — the architecture, the services it is becoming, an inventory of what is deployed, the implementation, the feature list and the roadmap. A reader had to hold all six to know what is true now.


---

## Architecture

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

**One arrived by a different route entirely, and deserves its own name:**

> A guarantee can disappear through test *metadata*, even when the test itself
> is sound.

Three parser-pinning tests needed the real `_parser_client` and no network. They
were marked `model_stage1`, which `pytest.ini` deselects by default because that
tier calls a live provider. The assertions were discriminating, the fixtures
valid, the implementation real — and none of it mattered, because classification
removed them from every ordinary run. The suite reported three fewer tests and
nothing else, and the deselection looked deliberate.

Every other instance in this table is a check that *ran* and could not tell the
difference. This one could, and never ran. That is why the guard inspects
`pytest.ini` rather than asserting inside the tests: **a deselected test cannot
defend its own inclusion.** "Uses production code" and "requires live
infrastructure" are different claims, and a marker that conflates them removes
exactly the tests most worth keeping.

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

**The sharper form, arrived at from the other end.** Preservation says what to
keep. It does not say when a decision is allowed to be made at all:

> Every irreversible decision must either be deterministic, or become an
> immutable observation.

Parser output and user clarifications are observations; registry resolution,
candidate ranking, execution planning, market replay, holdings resolution and
export are deterministic. Nothing else is permitted to decide anything that
cannot be undone.

That is why silent substitution keeps appearing as the anti-pattern. Rewriting
"SPX ETF" to SPY is neither: not deterministic, because it depends on whatever
the catalogue says today, and not an observation, because nothing recorded it.
A substitution that was recorded would be auditable and one that was
deterministic would be reproducible; being neither leaves the output as the
only trace, and the output looks correct.

**Three axes, and they are not peers.** A preservation mechanism has to be
semantically right, reachable by something that consumes it, and verified by
evidence that can itself fail. Each certifies the one before it, so the danger
increases with the layer: a missing mechanism leaves an unknown, which invites
investigation; a wrong one produces a wrong result, which reality can
sometimes surface; an unfalsifiable verifier produces confidence, which
suppresses both. It is the only failure mode that actively prevents its own
correction.

**Inference is not banished downstream; it is made attributable.** The tidy
version of this architecture would say inference happens once, at the model
boundary, and everything after is deterministic. That is not what is built and
would be worse if it were. `compile_scenario` infers — `dividends =
reinvested`, `contribution_day_rule = first_session_of_period` — deterministically
and from a versioned default set. A compiler that refused to infer would either
ask about every field, which nobody would finish, or apply defaults silently,
which is the failure the whole provenance model exists to prevent.

So the boundary is not inference against no inference. It is *who decided, and
were they told*:

    stated     the user said it
    inferred   the system chose it, deterministically, and says so —
               unconfirmed until they agree
    amended    the user answered a question about it

The confirmation screen exists because of the middle row. `execution_timing`
was nobody's decision but the compiler's, and it is shown, reasoned about and
confirmable rather than applied. What is prohibited is not inference but
*undisclosed* inference — a decision with no author on the page. The model's
non-determinism is bounded to one observation; the compiler's determinism is
bounded to choices the user can see and overturn.

**And the invariant does not cover reachability.** An amendment recorded and
never consumed satisfies it exactly — the decision became an immutable
observation — while the system still behaves as though the user said nothing.
That was the largest class of defect in the pilot work: the registry reader
never called, the answer collected and discarded, the identification computed
after the object it should have informed. An observation nothing reads is
indistinguishable from one never made, and the invariant cannot tell them
apart. Only exercising the live path can.

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


---

## Three services, an evaluation engine, and a market-data lake

A design note, not a plan of record. It exists so the next decision is taken
against checked facts rather than remembered ones, and so the parts that are
*already true* of this repository are not rediscovered.

## What was checked

The premise was "DuckDB 1.4, because it allows multiple writers". Half right,
and the half that is wrong changes the design.

**DuckDB does not give multiple *processes* write access to a database file.**
Its MVCC and optimistic concurrency are within one process, across threads —
[Concurrency](https://duckdb.org/docs/current/connect/concurrency). Multi-process
writing is the `Quack` client/server protocol, beta as of v1.5.2 and expected
mature at v2.0, autumn 2026. Three services on Kubernetes are three processes,
so a shared `.duckdb` file is not the shape.

**What 1.4 actually gives is Iceberg writes.**
[DuckDB 1.4.0 "Andium"](https://duckdb.org/2025/09/16/announcing-duckdb-140) added
`INSERT`, `UPDATE`, `DELETE` and `MERGE` against Iceberg tables. So the version
choice is right and the reason is different: it is not that DuckDB learned to
share a file, it is that DuckDB learned to write to a format that was already
designed for many writers.

**Multiple writers is Iceberg's property, not DuckDB's.** Concurrency comes
from atomic snapshot commits through a catalog, which is also why
[writes require an Iceberg REST catalog](https://duckdb.org/docs/current/core_extensions/iceberg/writing) —
on AWS, S3 Tables or SageMaker Lakehouse. Files in a bucket are not enough.
This is the one hard requirement to design around.

**1.4 is the LTS line.** Codename Andium, community support to 16 September
2026, currently at
[1.4.5](https://duckdb.org/2026/06/17/announcing-duckdb-145). 1.5.5 is newer
and not LTS. For a market-data lake that outlives a release cycle, pinning the
LTS is the defensible choice, and `v1.4-andium` is that branch.

## What is already true here

The service split maps onto module boundaries this repository already has,
which is the cheapest kind of migration:

    src/workspace   the front end — pages, plans, the parameter table
    src/mission     the evaluation engine — compile, simulate, refuse by name
    src/market_data the data engine — access, provenance, licensing

QuantLib is already installed and in `requirements-core.txt`, so the evaluation
service starts with the vocabulary rather than acquiring it.

**The licensing gate is the constraint nobody should route around.**
`market_data.access.approved_snapshot()` re-reads the recorded answers to six
vendor licensing questions on every resolve, and returns nothing if one is
missing; `pilot_data_policy` is `SYNTHETIC_ONLY` today and Terraform will
refuse a value the application does not recognise. A market-data lake means
real vendor data at rest, and that is exactly what those questions are about —
redistribution, derived works, retention, egress. Standing up the lake is a
licensing decision before it is an engineering one, and the gate should govern
the *data engine*, not be re-implemented beside it.

## What QuantLib buys as an engine

Fixed income, annuities and structured products, which is the stated reason and
a real one — bonds, swaps, swaptions, caps, floors, options, and the term
structures they price against. None of that is expressible today.

Worth stating plainly: the current simulator is not a subset of QuantLib. It
executes a *plan* — contributions on a schedule, rebalancing, benchmarks
receiving the same money on the same days — and QuantLib prices *instruments*.
A migration is not a port; it is one service calling the other for the
instrument half while keeping the plan half. The capability manifest, the
refusals by name and 5600 tests are built around the plan half, and they are
the thing that makes a refusal honest rather than a crash.

## The shape

    quantify-web        pages, sessions, plans          → users database (Postgres)
    quantify-evaluate   QuantLib + the plan simulator   → stateless
    quantify-data       ingest, provenance, licensing   → Iceberg lake (S3 + REST catalog)

Two databases, as described: the existing Postgres keeps users and their plans;
market data lives in the lake. They should not be the same store — one is
somebody's private workspace and the other is licensed vendor data, and the
retention, erasure and egress rules differ for exactly that reason.

## What it costs

Honesty about the jump: today this is one `t3.small` running docker compose
behind a Cloudflare tunnel, with an internal ALB and RDS. Managed Kubernetes
plus an Iceberg catalog plus a data engine is a different operational class —
more moving parts, a real monthly bill, and a deployment story that has already
cost this project several evenings at its current size.

The staged path that keeps the deployment working throughout:

1. **Split evaluation out first.** It is stateless, it already has QuantLib,
   and it is the only service that can be extracted without touching data
   licensing or session handling. If the split is going to be painful, this is
   where it shows, and it is reversible.
2. **Stand up the lake read-only.** S3 Tables as the REST catalog, DuckDB 1.4
   LTS as the reader, synthetic data first. The licensing gate stays
   `SYNTHETIC_ONLY` and nothing about vendor terms is decided yet.
3. **Answer the licensing questions for data at rest**, then let the data
   engine write. This is the step that needs a person and not a deploy.
4. **Kubernetes last.** Three services on compose on one host is a valid
   intermediate state and tests the split without the cluster. Moving to EKS
   is then a deployment change rather than a redesign.

## The trigger for the lake is reproducibility, not volume

This section replaces an earlier one that said the trigger was "data volume or
a second consumer". That was wrong, and wrong in a way this project has spent
months learning to recognise elsewhere.

When Quantify says *"8.7%, a 19.2% maximum drawdown, ending at $413,280"*, the
questions that decide whether the number means anything are: which SPY
observations, adjusted or unadjusted, which corporate actions, which calendar,
which FX, which curve, which inflation series, which snapshot — and **what was
known on each date**. Without answers, the arithmetic can be formally proved
while the economic history fed into it is wrong.

That is the same defect class as the one Discovery exists to prevent, one
layer down. Discovery refuses to guess what a sentence meant; the data
substrate currently guesses what the market did. Volume is irrelevant: a
hundred rows nobody can reproduce is a worse position than a billion rows that
anybody can.

## What is already true, and it is more than expected

**The ledger exists.** `accounting.Fill` is a line — date, ticker, shares,
price, notional, cost, reason — described in its own docstring as "what
actually happened, at the price that was actually available".
`PortfolioPath` carries end-of-day value, cash, holdings per ticker, external
flows, the fills, and the orders that *could not* execute. Time-weighted and
money-weighted returns are computed from it.

So the engine is already a historical portfolio accounting engine. What it is
not is one whose ledger anybody can see: nothing renders the fills. The page
shows a figure and a chart derived from a ledger the person is never shown,
which is a presentation gap rather than an engine rewrite.

**The reproducibility question is already asked, and cannot be answered.**
Every run records a `market_data_access_event` carrying `snapshot_id`,
`provenance_digest` and `frame_digest` — "the digest of the exact canonical
frame that was handed over". The schema comment is explicit that a snapshot id
alone is insufficient because two provenances differing only in access time are
different records.

So the system already knows it must identify the exact bytes it computed on.
What it cannot do is *rebuild* them: the digest names a frame that no store can
reconstruct from raw observations. The lake is not a new idea being introduced
here — it is the missing half of a mechanism that is already load-bearing.

## The layering

    RAW          vendor observations, as received, never edited
                 Yahoo / Polygon / Nasdaq / FRED / Treasury / EDGAR
        |
    NORMALIZED   instrument identity, calendar, currency,
                 corporate actions, prices, rates
        |
    CANONICAL    total-return series, cash rates, FX, inflation,
                 yield curves, benchmark series
        |
    SNAPSHOT     market-snapshot:<hash>   immutable, published
        |
    EVALUATION   strategy + snapshot + engine version
        |
    MissionResult

The property this buys is that an evaluation becomes close to a pure function:

    evaluate(strategy_hash, market_snapshot_hash, engine_version) -> MissionResult

`quantify-evaluate` must not query vendor tables. It consumes a published
snapshot and nothing else — otherwise "which observations" becomes a question
about when the query ran, which is exactly the state `frame_digest` was added
to escape.

RAW is kept unedited on purpose. A normalisation that overwrites its input
destroys the only evidence that could settle a disagreement about what the
vendor actually said.

## Where QuantLib sits

    Mission strategy
          |
    Portfolio simulator          <- executes the plan, writes the ledger
          |-- equities/ETFs/cash  -> canonical observations
          |-- bonds               -> QuantLib
          |-- options             -> QuantLib
          |-- annuities           -> QuantLib
          |-- rates/curves        -> QuantLib + canonical data
          |
    Ledger -> Formal Core -> MissionResult

QuantLib prices instruments; the simulator executes plans and produces the
ledger. Neither replaces the other, and the ledger is where they meet.

What this opens: four strategies compared against *the same* snapshot rather
than against separately assembled series, which is the difference between a
comparison and a coincidence.

## Free sources, and the licensing gate they still meet

Much of a first data layer is available without a vendor contract:

| Data | Source | Suitability |
|---|---|---|
| Treasury rates and curves | US Treasury | excellent, public domain |
| CPI, Fed rates, macro | FRED / originating agency | excellent; check each series' terms |
| Company fundamentals | SEC EDGAR / XBRL | excellent, public domain |
| US equity and ETF daily prices | open datasets | fine for development; rights vary |
| Dividends and splits | open datasets | usable, provenance needs care |
| ETF holdings and metadata | issuer publications | often usable, terms vary |
| Index levels | index owner | frequently licensed |
| Options, intraday equities | commercial | exchange licensing applies |
| Corporate bonds and credit | fragmented | governments easy, corporates hard |

"Free to download" is not "free to redistribute, derive from, or retain", and
those are exactly the six questions `approved_snapshot()` re-reads on every
resolve. A public-domain Treasury series and a scraped index level are not the
same licensing object, and the lake must carry the distinction per series
rather than per bucket.

## Storage for the pods that run DuckDB

Per pod, never shared, and that is a correctness decision rather than a
convenience one.

[DuckDB's own guidance](https://duckdb.org/docs/lts/guides/performance/environment)
is that it must not run read-write on network-attached storage — NFS, SMB, and
therefore EFS. It takes file locks the filesystem does not honour, which
produces "could not set lock on file" at best and unpredictable behaviour at
worst. Read-only over NAS is supported; read-write is not. Network-backed
*block* storage such as EBS is fine for both.

In Kubernetes terms:

    ReadWriteMany (EFS)     no. A shared PVC holding a DuckDB file is the
                            unsupported configuration, and two pods writing
                            one file is not something DuckDB claims to do
                            in any storage class.
    ReadWriteOnce (EBS)     yes, one per pod, via a StatefulSet's
                            volumeClaimTemplates.
    emptyDir                probably enough — see below.

**Ask first whether it needs to be a PVC at all.** In this architecture the
durable state is the Iceberg lake; DuckDB is the query engine over it. Local
disk then holds spill for larger-than-memory queries and cached remote files —
both reconstructible, neither authoritative. `emptyDir` covers that without a
volume lifecycle to manage, without orphaned EBS volumes after a rescheduling,
and without the constraint below.

A PVC earns its place when the cache is expensive enough to rebuild that losing
it on every restart hurts, or larger than the node's ephemeral storage.

**The constraint that surprises people:** an EBS volume lives in one
availability zone, so a pod bound to one can only be scheduled in that zone. A
service that was multi-AZ becomes single-AZ the moment it acquires a PVC, and
the failure shows up as pods pending rather than as anything about storage.

If a pod ever does need a *writable* DuckDB database that outlives it — a
materialisation rather than a cache — that is one pod with one RWO volume and
no second writer, by construction. The moment two want to write the same
database, the answer is the lake and its catalog, which is what the catalog is
for.

## Skipping EBS, and what cannot be skipped

The wish is to avoid EBS. It is mostly achievable, and the reason it is only
mostly is worth separating from the reason people usually give.

**EBS volumes and local scratch are different things.** An `emptyDir` is the
*node's* disk, not a provisioned volume: no PVC, no zone pinning, no orphaned
volumes after a rescheduling. On instance-store node types — `m6id`, `c6id`,
`i4i` and their kin — that disk is local NVMe, which is faster than EBS as well
as cheaper to reason about. So "no EBS" is achievable; "no local disk" is not.

**Spill is not an ingest problem, so streaming does not remove it.** The
concern was staging a whole dataset in memory before submitting it, and that
part streaming does solve: S3 multipart upload never holds a whole object,
DuckDB reads remote Parquet without downloading it whole and writes Parquet
straight to S3 over httpfs. What streaming does not solve is a sort, a join or
an aggregation larger than `memory_limit`, which spills to `temp_directory`
regardless of how the data arrived. That is the disk that cannot be skipped,
and it is scratch.

**The question is not EBS or not.** It is whether anything on that disk must
survive the pod. For the interim DuckDB the answer is no: the lake is the
authority and the local files are spill and cache. For Doris later the answer
is also no —
[compute-storage decoupled mode](https://doris.apache.org/blog/doris-compute-storage-decoupled/)
keeps the full dataset in S3-compatible storage and caches only hot data on the
BE nodes. Two different engines, the same shape, and the same answer: node
scratch, sized deliberately.

    emptyDir with sizeLimit          scratch that dies with the pod
    ephemeral-storage requests       so the kubelet evicts rather than the
                                     node filling and taking its neighbours
    SET memory_limit                 where spilling starts
    SET temp_directory               where it spills to

**Doris as the central manager changes what this repository owns, not this
answer.** If Doris writes the lake, the data engine stops being a DuckDB
process and becomes a client of Doris — but the reproducibility requirement is
unchanged, because a snapshot identified by hash is a property of what is
published, not of what published it. The `frame_digest` a run already records
has to keep meaning the same thing across that switch, which is the migration
test worth writing before the migration.

## What would make this wrong

- If the instrument work stays hypothetical, the evaluation split buys
  operational cost and no capability. The trigger for stage 1 is a real
  instrument somebody wants modelled, not the diagram.
- If the ledger is never shown and no second strategy is ever compared on the
  same snapshot, the layering is bookkeeping nobody reads. The cheapest test of
  this whole direction is to render the fills that already exist.
- If the licensing answers do not permit vendor data at rest, the lake stays
  synthetic — and reproducibility of a *synthetic* snapshot is still worth
  having, because it is what makes two runs comparable. That outcome shrinks
  the lake; it does not remove the reason for it.


---

## Repository and deployment inventory

**Date:** 2026-07-31 · **Updated:** 2026-08-01 (Mission SDK reconciliation, §2)
· **Purpose:** establish what is real and running before any contract work
begins.

Everything below is observed from the filesystem and git metadata, not inferred
from architecture documents. Where a fact could not be established from a source,
it is recorded as `unknown` rather than guessed.

---

## 1. Summary

The three runtime components named in the contract plan resolve as follows:

| Named component | What it actually is |
|---|---|
| `context-runtime` | **A real repository**, vendored as a submodule, at **v7.0** |
| `redevops-rag` | **A real repository**, vendored as a submodule — and separately checked out at a **different commit** |
| `sidekick` | **Not a repository.** An integration module inside context-runtime |
| `mission-runtime` | **Does not exist.** Zero occurrences in code, docs or config |
| `discovery-runtime` | **Does not exist.** Zero occurrences in code, docs or config |
| `agentic-os` | **Not located** anywhere on this machine |
| `mission-sdk` | **A real remote repository**, `redevops-io/mission-sdk` — not cloned here. Its execution `MissionProgram` is a different type from the lifecycle one in `runtime-contracts` (§2) |

`mission_runtime`, `MissionRuntime`, `discovery_runtime` and `DiscoveryRuntime`
return **zero matches** across `rag-saas-platform` in `.py`, `.go`, `.ts`, `.md`,
`.yaml` and `.yml`. They are architecture-document terminology, not code.

---

## 2. Components

```yaml
components:

  quantify:
    repository: redevops-io/RAAAL
    local_path: /projects/RAAAL
    remote_url: git@github.com:redevops-io/RAAAL.git
    branch: main
    commit: dd2b860
    last_commit: 2026-06-05
    deployed: true                  # Cloudflare Pages, daily-deploy.yml
    deployment_reference: .github/workflows/daily-deploy.yml
    importers: []                   # imports neither runtime
    contract_status: canonical-consumer

  context-runtime:
    repository: redevops-io/context-runtime
    local_path: /projects/rag-saas-platform/context-runtime
    remote_url: git@github.com:redevops-io/context-runtime.git
    branch: submodule (detached)
    commit: 8fe5f3a
    describe: v7.0-2-g8fe5f3a       # v7.0 plus two commits
    deployed: unknown               # library, not a compose service
    deployment_reference: imported by backend/services/cr_runtime.py, cr_media.py
    importers:
      - rag-saas-platform/backend/services/cr_runtime.py
      - rag-saas-platform/backend/services/cr_media.py
      - redevops-rag/benchmarks/*
    modules: 157
    contract_status: SPECIFIED
    implementation_adoption: NOT_STARTED
    predecessor_mapping: documented

  redevops-rag:
    repository: redevops-io/redevops-rag
    local_paths:
      - /projects/redevops-rag                     # main @ ceec853
      - /projects/rag-saas-platform/redevops-rag   # submodule @ e3e37df (v0.2.0-31)
    remote_url: git@github.com:redevops-io/redevops-rag.git
    commit_divergence: true         # see §4
    deployed: unknown               # library, not a compose service
    importers:
      - rag-saas-platform/backend/services/cr_ingest.py
      - context-runtime/context_runtime/integrations/redevops_rag.py
      - context-runtime/context_runtime/adapters/store_{semantic,redevops,diver}.py
    modules: 9
    contract_status: SPECIFIED
    implementation_adoption: NOT_STARTED
    predecessor_mapping: documented

  rag-saas-platform:
    repository: redevops-io/rag-saas-platform
    local_path: /projects/rag-saas-platform
    branch: feat/context-runtime-migration
    commit: 7d562c2
    last_commit: 2026-07-29
    deployed: true                  # docker-compose: postgres, backend, frontend,
                                    # botfather-automation
    role: the deployed control plane that consumes both runtimes
    contract_status: integration-host

  sidekick:
    repository: none
    local_path: context-runtime/context_runtime/integrations/sidekick.py
    deployed: unknown
    contract_status: module-not-repository

  mission-runtime:
    repository: not-located
    evidence: zero occurrences in code, docs or config
    contract_status: NOT_LOCATED

  discovery-runtime:
    repository: not-located
    evidence: zero occurrences in code, docs or config
    contract_status: NOT_LOCATED

  agentic-os:
    repository: not-located-on-this-machine
    pinned_as: mission-sdk dependency runtime @ d261825
    source: reported 2026-08-01, not observed here
    contract_status: NOT_LOCATED

  mission-sdk:
    repository: redevops-io/mission-sdk
    branch: main
    tested_commit: 6aeb4e6
    dependency_runtime: agentic-os@d261825
    local_path: null
    source: reported 2026-08-01, not observed here
    integration_state: NOT_STARTED
    intended_mode: advisory shadow compilation
    contract_status: NOT_CLONED
```

### Two different `MissionProgram` types

They are not interchangeable, and conflating them produces confident wrong
conclusions about what the SDK can represent:

| | `runtime-contracts` | `mission-sdk` |
|---|---|---|
| Orientation | **lifecycle** | **execution** |
| Models | Investigation state transitions | candidate step chains |
| Entry point | constructed directly | `MissionProgram.from_proposal()` |
| Proposal type | none — no `MissionProposal` exists | `MissionProposal` |
| Located here | yes, `/projects/runtime-contracts` | no |

> The existing `runtime-contracts` Quantify adapter
> (`adapters/quantify/adapter.py`) **is not** the Mission SDK adapter, and its
> lifecycle `MissionProgram` **is not** the execution program targeted by this
> integration.

A third, unrelated `MissionProposal` appears in `src/api.py` (`discoveries()`),
naming the last stage of the Context Runtime v8 Discovery chain — Signal →
Detection → Correlation → Hypothesis → MissionProposal. It is a docstring
describing a shape that surface follows, not a type Quantify imports. Three
distinct things share this name; none of them is a synonym for another.

The lifecycle program validates that terminal states carry outcomes and that
transitions do not leave terminal states; it has no candidate fan-out because
fan-out is not what it is for. Reading a fan-out limitation from it and
attributing that limitation to the SDK — as this repository's notes did until
this commit — compares Quantify against the wrong package.

`src/workspace/execution.py` stays Quantify's authoritative execution
declaration either way. It is deliberately neutral, so it is the input an
adapter would translate rather than something an adapter would replace.

**Resume the integration when** discovery must emit executable Quantify
missions; or the pilot needs portable case bundles or cross-process replay; or
Go/Kotlin portability becomes active; or Quantify's orchestration logic starts
duplicating generic SDK functionality. The order is then: clone/install
`mission-sdk` → optional `quantify[missions]` extra → execution-declaration
adapter → `MissionProposal` → shadow validate/profile/ci → graph-equivalence
gate. The plan is valid and retained; it is simply not the next highest-value
Quantify task.

---

## 3. The v8/v10 gap

`context-runtime` is at **v7.0**. None of the v10 contract types exist in any
reachable repository:

```
ArtifactHandle · ContextPreviewPlan · ContextView · DereferenceEvent
CapabilityDescriptor · MissionProgram · EvidenceSpanHandle
GraphNeighborhoodHandle
```

Zero files across `context-runtime`, `redevops-rag` and `RAAAL`.

The v8/v10 implementations are either in a repository not present on this machine
or not yet written. This is consistent with the earlier finding that
`CR-enterprise` is proprietary and separate, and that whitepaper v8's
implementation-status table described intent ahead of publication.

---

## 4. Two risks found

### 4.1 `redevops-rag` is checked out twice, at different commits

```
submodule pin  /projects/rag-saas-platform/redevops-rag   e3e37df  (v0.2.0-31)
standalone     /projects/redevops-rag                     ceec853  (main)
```

The integration host pins one commit; a developer working in the standalone
checkout sees another. Whichever is deployed, one of the two is not it — and this
is precisely the "validating one branch while production runs another" risk the
contract plan warns about, present today and unrelated to contracts.

**Action:** establish which commit is deployed before any adapter is written
against either.

### 4.2 The integration host is on a feature branch

`rag-saas-platform` is on `feat/context-runtime-migration`, not `main`, with its
most recent commit 2026-07-29. If that branch is what runs, then `main` is not a
meaningful conformance target.

---

## 5. Deployment facts that could not be established

Recorded as unknown rather than assumed:

- **Whether context-runtime or redevops-rag is deployed at all.** Neither is a
  `docker-compose` service. Both are libraries imported by `backend/services/`,
  so their deployed version is whatever the backend image was built with — which
  is not discoverable from the source tree.
- **Which commit the running backend was built from.** No CI workflow directory
  exists in `rag-saas-platform`, no image digest is pinned in a manifest reachable
  from here, and no service exposes build metadata.

### Recommended fix

Neither runtime can currently disclose its own source commit, which makes every
downstream conformance claim unverifiable. Add to the backend's authenticated
diagnostics endpoint and startup log:

```json
{
  "service": "context-runtime",
  "version": "7.0",
  "git_commit": "8fe5f3a",
  "git_branch": "...",
  "contract_versions": {
    "artifact_handle": "0.1",
    "runtime_event": "0.1"
  }
}
```

Until that exists, "deployed" is a claim rather than an observation.

---

## 6. Consequence for the contract work

`mission-runtime` and `discovery-runtime` are not missing repositories — they are
**unbuilt components**. Phase B therefore does not need a repository created to
satisfy an architecture diagram. It can be implemented in the deployed control
plane (`rag-saas-platform/backend`) or in Quantify, provided:

- the `MissionProgram` contract is canonical and externally owned;
- `Investigation` keeps one artifact representation;
- lifecycle transitions emit canonical events;
- the implementation is marked as the current adapter;
- later extraction would change neither artifact identity nor wire contracts.

Repository boundaries should follow deployment and ownership, not nouns from a
whitepaper.

### Known defect: generic forward reconciliation

`src/mission/observation.py::reconcile` matches expected against observed on
`(date, kind)` exactly and has no pending state. Two consequences, both of the
"unknown silently becomes absent" kind:

- An event whose date has not arrived reports `MISSING`. A plan examined before
  its first milestone looks like a plan going wrong.
- An event that happens a few days late reports `MISSING` *and* `UNEXPECTED` —
  two deviations describing one event that happened once.

`src/mission/rsu_reconcile.py` solves both for the RSU domain with nine typed
states, a declared and versioned matching tolerance, and `effective_date` kept
apart from `observed_date`. The generic tracker was deliberately left alone
rather than widened during that work.

It should eventually adopt the same temporal vocabulary or be deprecated in
favour of a generic primitive extracted from the RSU one. Until then, any plan
using generic forward tracking carries the defect above.

### Adoption status

```yaml
implementations:
  quantify:            {status: CANONICAL_CONSUMER, adapter: adapters/quantify}
  context-runtime:     {status: SPECIFIED,          adapter: adapters/context_runtime}
  redevops-rag:        {status: SPECIFIED,          adapter: adapters/redevops_rag}
  rag-saas-platform:   {status: PLANNED}
  mission-runtime:     {status: NOT_LOCATED}
  discovery-runtime:   {status: NOT_LOCATED}
  sidekick:            {status: NOT_LOCATED}
  agentic-os:          {status: NOT_LOCATED}

# A separate axis. Adoption above is of runtime-contracts; this is of the
# execution SDK, and the two share a type name but not a type.
mission_sdk_integration:
  quantify: {status: NOT_STARTED, mode: advisory shadow compilation,
             seam: src/workspace/execution.py, deferred_for: closed-pilot-v1}

release_gate:
  quantify:        {minimum: CONFORMANT}
  context-runtime: {minimum: CONFORMANT}
  redevops-rag:    {minimum: CONFORMANT}
  mission-runtime: {minimum: NOT_LOCATED}
```

A component below its gate fails the release. A component whose gate is
`NOT_LOCATED` is a visible roadmap gap, not a red build.


---

## Implementation

**Baseline:** 2026-07-31 · 857 tests passing, 2 skipped · 37 test files

---

## 1. How this system was built

Every release made one hidden choice explicit, and each one immediately exposed a
real defect. That pattern is the project's identity and is worth stating before
the status table, because it is the argument for the architecture rather than a
consequence of it.

| Hidden choice made explicit | Defect it exposed |
|---|---|
| Execution lag and transaction costs | Reported 13.00% was actually **−2.83%** |
| Trading calendar as an artifact | **31.1%** weekend padding inflated annualized figures |
| Covariance estimator in the AST | Estimator divergence between spec and executor |
| Rules naming their realization | Declared rules and universe filters were **inert** |
| Constraint precedence declared | A hard bound silently lost to a soft turnover cap |
| Trial counting | Configurations tried and dropped were never counted |
| `tax_treatment` as a runtime | A Roth and a taxable account compared as **identical** |
| Comparison dimensions registered | `allocation_rule` and `data_snapshot` were unchecked |

Two errata are published and reachable: [execution lag and
costs](errata/2026-07-30-execution-lag-and-costs.md) and [trading
calendar](errata/2026-07-30-02-trading-calendar.md). Superseded figures are
flagged, never deleted.

---

## 2. Completed

### Engine and correctness
- Execution lag and cost model; annualization on a declared session basis
- Chronological train/test split with embargo; models refuse future-trained loads
- `seed_everything()`, `RunManifest`, `frame_digest()` for reproducibility
- Degeneracy diagnostics — concentration, degenerate volatility, implausible
  Sharpe, effective breadth — that refuse publication of a flagged result

### Artifact layers
- Methodology AST with merge semantics; immutable versions; content hashing
- `EvaluationProtocol`, versioned calendars, statistics module, statistical
  policies, publication gates
- Claims, evidence, assumptions, findings, **investigations**
- Persisted run records with stored verdicts
- Declaration realization checks, with `unrealized_declaration` as a hard blocker
- Second methodology family (`xsmom`) proving generality — zero new artifact types

### Mission layer
- Share-level accounting; TWR **and** MWR; flow-matched benchmarks
- `Intent` with `SelectionBasis`; `ScenarioSpecification`; `Mission`
- Ten-stage compiler with the model quarantined to stage 1; versioned defaults
- RSU vesting template; forward tracking; `PlanObservation`; `Proposal`;
  counterfactuals
- Private workspace: separate router, templates, store; owner-scoped queries

### Runtime layer
- `RuntimeArtifact` base with `content_hash` / `compatibility_hash`
- `TaxRuntime`, `AccountRuntime`, `MarketDataRuntime` + `RealizedData`,
  `CashFlowRuntime`
- `ExecutionEnvironment` with composition validation and a registered rule catalog
- Comparison-dimension registry; `ISOLATION_DIMENSIONS` derived

### Interface and boundaries
- UI milestones 1–2: chain, glyph, impact graph, timeline, relation badge,
  eligibility
- Shared design tokens with layout deliberately unshared
- Endpoint boundary manifest with a manifest-driven sweep
- Eleven-step acceptance journey, end to end

### Artifacts on disk

```
methodologies 4 · protocols 3 · calendars 2 · policies 1
claims 3 · assumptions 7 · evidence 6 · findings 3 · investigations 6
```

---

## 3. Acceptance criteria

Testable questions a reader must answer **without reading source**
(`tests/test_acceptance.py`):

1. Why `hrp@3` exists
2. Why `hrp@1` is blocked despite strong statistics
3. What invalidated Erratum 01's absolute figures
4. Whether two methodologies are comparable, and why not
5. Which assumptions a methodology inherits versus declares
6. Which mechanism realizes every declared rule
7. What verdict a run received **historically**
8. Whether current policy would differ
9. What evidence changed a claim's status
10. Whether an investigation concluded with no finding
11. Whether a reader can judge a methodology's state from the library page in
    under five seconds, without opening it

### Architectural invariants

1. Every result identifies its methodology, protocol, environment, realized
   data, assessment, policy, publication decision and modelling scope
2. Every declaration names its realization
3. Every relation has declared semantics
4. Every comparison reports checked **and unchecked** dimensions
5. Every causal attribution names what is isolated
6. Historical facts are persisted; current derivations are recomputed
7. Public artifacts never cite private artifacts
8. Personal scenarios never promote directly to the public library
9. Every benchmark set is symmetric and order-preserving
10. Hidden candidate selection is structurally impossible
11. Backtest and forward results cannot be linked
12. The platform cannot represent an executed order
13. Inconclusive and no-material-impact investigations are first-class
14. UI state and graph renderings derive from the same payload as accessible text
15. A new runtime or artifact kind fails until visibility, realization,
    comparison, composition and semantics are declared

---

## 4. Architecture freeze — 2026-08-01

The architecture is now good enough. The next meaningful risk is no longer "can
the system represent truth correctly?" — it is "will anyone understand it, trust
it, and find the result useful enough to return?", and no amount of further
invariant work answers that.

**Frozen until Closed Pilot v1 ships:**

- new artifact types
- new runtime abstractions
- new comparison semantics
- new lifecycle refinements
- additional canonicalization rules
- edge-case hardening beyond release blockers

**Changes still allowed**, and only these:

| Allowed | Because |
|---|---|
| Data corruption | The record is the product |
| Security or privacy boundary violations | One-way boundary, personal data |
| Materially wrong financial results | Wrong money |
| Broken replay | A result that cannot be reproduced is a claim |
| Deployment failure | Nothing ships |

**The test to apply before adding anything:**

> Will a pilot user see or benefit from this in the next two weeks?

If not, it needs one of the three justifications above. Otherwise it goes to
§6 backlog.

---

## 5. Closed Pilot v1 — the one milestone

A user can:

1. enter a scenario conversationally
2. confirm the compiler's interpretation
3. run a historical simulation
4. compare against 3–5 valid benchmarks
5. see TWR, MWR, modelling scope, tax and account assumptions, and trial count
6. save the plan
7. revisit it
8. start forward paper tracking

Nothing else is required for v1.

### Implementation order

1. ~~**Wire the model into compiler stage 1.**~~ **Done.** A model proposes
   readings; a deterministic quarantine checks every one against the text and a
   vocabulary derived from the phrase rules. Fabricated quotations, invented
   tickers, values outside the vocabulary and figures absent from the
   description are all refused, and anything it cannot place becomes a question
   rather than a default. The parse is **pinned to the saved plan**, so
   revisiting never re-derives it against a model that has changed. No key, no
   network or a bad response falls back to the phrase rules.
2. **Make the UI one product.** Scenario entry, interpretation checklist, result
   summary, benchmark comparison, modelling scope, plan page, forward timeline.
   No further backend ontology until these are usable.
3. ~~**Load-test the main journey.**~~ **Partly done.** The 144-strategy corpus
   runs 14,400 compiles as a test and found five compiler defects on its first
   pass; HarnessBench measures the Polars crossover per workload. See
   [Performance.md](Performance.md). Remaining: the Mission Evolution workload
   and the concurrent scenarios L01-L08.
4. **Tax and account depth for three launch scenarios only:** taxable investing,
   401(k)/Roth accumulation, RSU vesting and diversification.
5. **Recruit pilot users.** Five real users will expose more than another two
   hundred invariant tests.

---

## 6. Backlog — after pilot validation

Deferred deliberately, not abandoned. Each was in progress or planned and is
paused because it does not change what a pilot user sees.

- Full Discovery Runtime; automated Investigation generation
- Complete Finding-production workflow (`FINDING_PRODUCED` route, promotion
  rules, conclusion-to-Finding routing)
- `NO_MATERIAL_IMPACT` journey against persisted evidence
- Elaborate evidence-graph UI
- Second-domain expansion (debt payoff versus investing)
- Corporate-action completeness; generalized estate runtime
- Jurisdiction-specific tax runtime versions; RMD rules
- `TradingCalendar` retrofit to `RuntimeArtifact`
- Cross-agent harness enhancements; deeper v10 optimizations
- Chart artifacts with provenance blocks (Phase E)

### What was completed before the freeze

The Investigation workflow reached durable, replayable evidence state in
`rag-saas-platform`: the transition ledger, canonical replay, the `INCONCLUSIVE`
journey against a reference-first working set, and persisted `ContextView`
declarations and materializations with tamper detection. That work stands; it is
the finished floor under the backlog above, not a half-built room.

---

## 7. Phase detail (paused)

Retained for when the backlog is picked up again.

### Phase B — Investigation workflow and runtime integration

The artifact, persistence, graph queries, examples and UI are implemented (see
§5). What remains is the **governed workflow**: lifecycle transitions, Mission
Runtime handoff, Discovery integration, evidence collection, deduplication, and
conclusion-to-Finding routing.

> **Exit:** Discovery can open work that concludes without a positive finding,
> and the result is as visible as a conclusive one.

### Phase C — Discovery Runtime

Evidence ingestion, claim-status change detection, restatement detection,
affected-artifact traversal, missed-event and expired-proposal detection,
proposal deduplication, significance scoring, human review gates, audit trail.

> **Exit:** Discovery proposes reviewable investigations from typed graph changes
> without inferring from prose or recommending action.

Discovery may surface public methodologies matching user-stated constraints. It
may **not** rank them, use peer behaviour, or recommend action.

### Phase D — Remaining runtimes

1. `CorporateActionRuntime` — market data declares where action facts come from;
   the corporate-action runtime declares how they alter holdings and cash
2. Full cash-flow event library
3. Retirement-account rules
4. Jurisdiction-specific tax runtime versions
5. Estate transfer — only after benchmark-baseline viability is proven

Each must declare family and version, comparable form, realization checks,
limitations, semantic dependencies, composition rules, comparison dimensions and
visibility.

### Phase E — Chart artifacts

Charts become artifacts with a provenance block: producing run, methodology,
protocol, calendar, environment, data snapshot, performance class, publication
decision, comparability requirements. Research pages restyled with shared tokens
while keeping dense analytical layouts.

> **Exit:** every chart is provenance-linked, and none can bypass performance
> eligibility.

### Phase F — Pilot readiness review

Securities-counsel review · privacy and retention review · API boundary sweep ·
threat model · audit logging · model and provider disclosure · prompt-injection
and tool-abuse checks · full non-recommendation journey review · scenario
deletion and export · data-provider licensing · paper-only tracking verification.

### Phase G — Second domain

**Debt payoff versus investing.** Chosen because it preserves the comparative
strength: standardized cash flows, clear counterfactual baselines, interest and
tax assumptions, liquidity constraints — and no new ontology expected.

Domains without standardized counterfactual baselines (estate planning,
insurance) compile fine and lose the comparison, which is currently the strongest
part of the product. Choose a second domain on whether it has baselines, not on
whether it has events.

Do not start until the workspace documentation and closed-pilot surface are
complete.

---

## 8. Investigation: artifact complete, workflow incomplete

Worth stating precisely, because "workflow incomplete" reads as "artifact
absent" and the two are very different states.

```
Implemented:  Investigation as a knowledge artifact
Incomplete:   Investigation as a durable unit of work
```

### Implemented

- `Investigation` artifact with `InvestigationOutcome`
  (`PENDING`, `FINDING_RECORDED`, `NO_EFFECT_FOUND`, `INCONCLUSIVE`, `ABANDONED`)
- Registry support, loading and identity validation
- Six persisted YAML instances covering the real history
- Graph relationships and queries — `open_inquiries`, `null_results`,
  `investigation_for_finding`, `unattributed_findings`, `recorded_trials`,
  `investigation_provenance`
- Investigation UI page at `/ui/investigations`
- Self-validating outcomes: claiming a finding requires citing one, a null
  result may not cite one, a null result must name what it examined
- `trials_examined` flowing into deflation
- 22 tests

### Remaining

- Lifecycle transition enforcement
- Mission Runtime handoff
- Discovery Runtime proposal creation
- Investigation deduplication
- Assignment and ownership
- Pause, resume and cancellation semantics
- Evidence-gathering workflow
- Finding creation from concluded investigations
- Audit events for transitions
- Queue and review UI

## 9. Other known open items

- **`TradingCalendar` is not yet a `RuntimeArtifact`.** It lives in
  `src/calendars/` with its own hashing, so the `calendar` comparison dimension
  names a runtime kind with no registered type and its `depends_on` cannot be
  derived. `unreconcilable()` reports this rather than letting the registry look
  reconciled while one dimension stays unchecked. Converting it is the last
  runtime retrofit

---

## 10. Running it

```bash
python3 -m pytest tests/ -q          # full suite
python3 -m pytest tests/test_journey.py -q    # the eleven-step user journey
uvicorn src.api:app --reload         # /ui public library, /workspace private
python3 scripts/run_methodology.py   # execute a methodology under a protocol
python3 scripts/evaluate.py          # assessment + policy + publication
python3 scripts/assess.py            # statistical assessment only
python3 scripts/publish_run.py       # record a run in the ledger
```

Deployment notes for the Cloudflare Pages dashboard are in `deploy_cloudflare.sh`
and `.github/workflows/daily-deploy.yml`.


---

## Features

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


---

## Quantify roadmap v1

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

See `docs/Pilot.md` for the six criteria. Remaining:

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

`docs/Pilot.md` holds the full observations; this table is the
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
