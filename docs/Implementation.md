# Implementation

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
