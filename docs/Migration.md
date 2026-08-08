# Migrating Quantify to the ReDevOps runtime architecture

**Status:** proposed. Nothing below has been built.
**Date:** 2026-08-08
**Supersedes:** `docs/Roadmap.md` for everything above the execution engine.

---

## 1. The diagnosis, stated precisely

The decision to stop is right, and the reason is worth stating more narrowly
than "too many nuances", because the narrow version tells us what to keep.

Quantify understands a user's sentence with a hand-written recogniser: ~1,585
lines of ordered regex tables in `src/mission/compiler.py`, plus a model layer
in `parse_model.py` that votes against them. Every new way of saying a thing it
already supports is a code change, a deploy, and a suite run. The record bears
this out — the last four defects closed were:

| Defect | Nature |
|---|---|
| "whenever it crosses below" read as a crossing, not a state | new phrasing, supported dimension |
| "rebalanced monthly" read as a contribution cadence | new phrasing, wrong dimension |
| `60/40` written as a ratio rather than percentages | new notation, supported dimension |
| "on first negative month" | new phrasing, supported dimension |

Four fixes, four deploys, one dimension of meaning between them. That is the
treadmill, and an LLM reader ends it. This is the correct call.

### But the treadmill was not the expensive failure

On the same day, the catalogue sweep found this:

```
"I put $1,000 every year into VTI ... over the past 5 years"
    contributed: $1,000        <- one payment, not five
```

The parser was **completely correct** — it emitted `cadence='annual'`. The
executor, `_flows_from`, matched `monthly`/`weekly`/`biweekly` and let
everything else fall through to a single one-off contribution. `quarterly`,
`annual` and `daily` are offered to the user in the product's own confirmation
menu and rendered back as "every quarter" and "every year". A user could pick
"Every year" from a list we showed them and receive a figure computed over a
plan nobody described. No refusal, no caveat, no coverage flag.

**An LLM mission controller would not have caught this. It would have made it
more likely.** A regex table fails to recognise things; a model recognises
everything, including dimensions the engine cannot run, and emits them with
confidence. The gap between *what was understood* and *what was executed* is
the defect class that has cost this project the most — the 61× capital error,
the 4.6× money error, the 60/40 portfolio shown as 50/50 — and moving to a
model widens it.

So the migration has two jobs, and only the first is the one that motivated it:

1. **Replace the recogniser with a model.** Ends the phrasing treadmill.
2. **Make executability a contract rather than a hope.** The engine must
   declare exactly which dimensions and values it can execute; anything the
   controller emits outside that set must refuse, not degrade.

Job 2 is new work. None of the three target repositories has it (§3.4).

---

## 2. What the three repositories actually are

I read all three. Two are not quite what the brief assumed, and the difference
changes the plan.

### 2.1 `context-runtime` — a query planner, not a context manager

It decides *how to answer*: retrieval method, model tier, compression,
verification. It emits an inspectable `Plan`, executes it, records a `Trace`,
and learns from the outcome.

```python
def run(self, goal, *, sources=None, constraints=None) -> RunResult:
    g = self._coerce_goal(goal, sources, constraints)
    plan = self.plan(g)
    ctx = self.build_context(plan, g)
    return self.verify(self.execute(ctx, g))
```

**It has no durable context.** `BuiltContext` is a per-request, in-memory,
frozen dataclass and is never persisted. There is no context id, no context
record, no lifecycle, no store. What it durably writes is one JSON file per
trace plus an overwritten statistics blob.

It also has no decision provenance in the sense this project needs: no actor on
a `Plan` or `Trace`, no supersession chain, no revision history, and
`RunResult.citations` is populated from *every* retrieved hit rather than the
ones the answer actually used.

**Consequence for the plan.** Context Runtime cannot be the system of record
for a user's plan, run or worksheet. Quantify needs durable, versioned,
provenance-carrying decisions — that requirement is not negotiable and it is
not what this component does. Use it for what it is good at: choosing retrieval
and model tier for the understanding step, under a cost and latency budget.

### 2.2 `agentic-os` mission runtime — the right system of record

This is the strongest fit of the three, and it is closer to what Quantify
already believes than anything we would have designed.

- **Event-sourced and append-only.** `EventStore` has no update and no delete
  API. State is a fold over events. `WorldState` stores *observations*, never
  facts, and re-fuses beliefs from the log on every read — so replay
  reconstructs decisions rather than trusting a snapshot.
- **Fail-closed permissions.** A capability with no grant is a `CompileError`,
  not a warning. Capability binding has a trust boundary: plugin-sourced
  capabilities are discoverable but not bindable until trusted.
- **Human gates are a state, not a callback.** `MissionState.WAITING_HUMAN`,
  `HumanTask`, and resolution as an *authoritative observation* that dominates
  fusion — not an approve/reject boolean.
- **One LLM call, at the top.** `planner.py` is explicitly "the one place the
  LLM runs; everything below it is deterministic," and the compiler carries the
  matching rule: "If the model ever needs to run here, the layering has leaked."

That last principle is the one this migration is about, already written down.

### 2.3 `sidekick` — a coding-agent orchestrator, not the user-facing agent

The brief describes sidekick as "the agent running user interactions". It does
run user interactions — but they are *developer* interactions with a codebase.
It takes a task string, decomposes it into a DAG of subtasks, and fans out
headless coding agents on one git worktree per subtask, merging the green
branches.

Concretely, it has:

- no conversation model — every interaction is one task string → one run → a
  report; nothing carries across turns
- no web UI, no templates, no HTTP server other than a webhook receiver
- no users, no sessions, no auth, no tenancy — it runs as the invoking OS user
  against one repo
- no persistence of model turns; only clipped summaries reach the transcript

**Consequence for the plan.** Sidekick cannot be the pilot surface for
quantify.club. Pointing pilot users at it would be a category error. Two
further facts make this firm: its chat gateway binds `0.0.0.0:8787` with no
TLS and explicitly out-of-scope Slack signature verification, and an authorised
chat message triggers auto-approved code execution; and its planner falls back
silently to a single subtask with **empty acceptance checks**, which its own
check runner treats as an automatic pass — a dead model endpoint produces a run
reporting 1/1 accepted having verified nothing.

Sidekick is genuinely useful, as the tool it is: it should build the migration,
not serve it.

### 2.4 `runtime-contracts` — the spine, unmentioned in the brief

`/projects/runtime-contracts` already exists locally and already names this
exact set of implementations:

```
| quantify         | SPECIFIED | CONFORMANT | gap |
| context-runtime  | SPECIFIED | CONFORMANT | gap |
| mission-runtime  | NOT_LOCATED | NOT_LOCATED | ok |
```

> `runtime-contracts` is the canonical, application-neutral contract package
> for ReDevOps runtime interoperability. Quantify.club provides the initial
> fixtures and design discipline but does not own the contracts semantically.

It has already generalised the discipline this project paid for:

```python
class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"                # checked, and it did not hold
    INDETERMINATE = "INDETERMINATE"   # could not be checked
    NOT_APPLICABLE = "NOT_APPLICABLE" # does not arise — a correct answer, not a gap
```

and

```python
class Disposition(str, Enum):
    PRODUCED_NOTHING = "PRODUCED_NOTHING"
    """Concluded, carefully, with nothing to show. A finding of absence, not an
    absence of finding."""
```

That is `PASS/FAIL/VACUOUS/INVALIDATED` and "absence is not always ignorance",
written as a contract by someone else. It should be the schema layer, and this
migration is the event that moves it from `NOT_LOCATED` to real.

---

## 3. Target architecture

```
      user's sentence
            │
┌───────────▼──────────────────────────────────────────────┐
│  sidekick-class conversational surface  (TO BE DECIDED §6)│  Quantify web UI stays
└───────────┬──────────────────────────────────────────────┘
            │  goal: str
┌───────────▼──────────────────────────────────────────────┐
│  MISSION RUNTIME            agentic_os.mission           │
│  · the ONE LLM call: sentence → ExecutionIntent          │
│  · DecisionEvidence per reader, no reader privileged     │
│  · material disagreement → WAITING_HUMAN                 │
│  · event-sourced, append-only, replayable                │
└───────────┬──────────────────────────────────────────────┘
            │  ExecutionIntent (logical)
┌───────────▼──────────────────────────────────────────────┐
│  EXECUTABILITY GATE                        ★ NEW, §3.4   │
│  intent ∩ engine capability manifest                     │
│  anything outside → REFUSE, never degrade                │
└───────────┬──────────────────────────────────────────────┘
            │  ScenarioSpecification (physical, validated)
┌───────────▼──────────────────────────────────────────────┐
│  QUANTIFY EXECUTION ENGINE        kept, unchanged        │
│  simulate · accounting · funding · signals · benchmarks  │
│  market data + provenance                                │
└──────────────────────────────────────────────────────────┘

  context-runtime ── retrieval & model-tier planning for the LLM step only
  runtime-contracts ── schema identity, hashing, canonical serialisation
```

### 3.1 The boundary that matters

**The LLM replaces the recogniser. It never touches arithmetic.**

No model computes a return, a contribution schedule, a weight or a tax
treatment. The model's entire job is `str → ExecutionIntent`. Everything below
stays deterministic, and the existing `Verdict`/coverage discipline applies to
its output rather than being replaced by it.

This is not caution for its own sake. A money-weighted return that a model
produced is a number nobody can reproduce, and the whole provenance chain this
project built exists to make figures reproducible.

### 3.2 `ScenarioSpecification` is already the target schema

The migration is smaller than it looks because the structured target exists and
is good:

```python
@dataclass(frozen=True)
class ScenarioSpecification:
    """Everything the user described, separated into what each layer owns."""
    name: str
    version: int
    objective: Objective
    event_program: Sequence[Dict[str, Any]]
    flow_schedule: FlowSchedule
    allocation_rule: AllocationRule
    holdings_policy: HoldingsPolicy = field(default_factory=HoldingsPolicy)
    ...
    funding: Optional[Any] = None   # the authority: Scheduled | EventTriggered
```

Frozen, versioned, content-hashable, and already the thing the engine consumes.
It becomes the mission's physical plan. The LLM emits an `ExecutionIntent`; a
deterministic compiler turns that into a `ScenarioSpecification`; the gate
in §3.4 decides whether it may run.

### 3.3 The disagreement branch is our own rule, generalised

`feat/disagreement-decision-evidence` should be the baseline, not `main`. It
independently arrived at the rule this project reached through the parse-model
work:

```python
@dataclass
class DecisionEvidence:
    """... No ``source_type`` is privileged over another — the runtime never
    encodes 'regex wins' or 'model wins'; when readers disagree on a material
    field the disagreement is routed to the controller, not silently resolved."""
    field: str
    value: Any
    source_type: str = "prior"   # regex | model | policy | retrieval | prior
    source_ref: str = ""
    confidence: float = 1.0
```

Compare Quantify's `MATERIAL_EXECUTION_FIELDS` and the `merge()` that drops
both readings when they contest a material field. Same rule, better home. The
branch's `cosmetic_inputs` is our materiality list with the polarity chosen
correctly: **everything is material by default; cosmetic is an explicit
opt-out.**

Three gaps in the branch we will hit immediately and should fix upstream:

1. `DecisionEvidence` is not exported from `agentic_os.mission.__init__`.
2. The evidence never reaches the human. `_park_disambiguation` still builds
   `HumanTask.evidence` as the old string list, so the cockpit shows
   `key=/current=/sources=` and not the per-reader packet. The controller is
   routed the disagreement without the evidence it is supposed to weigh — which
   is the *point* of the branch.
3. Human resolution lands with `source_type="prior"`, so the authoritative
   answer is indistinguishable from a guess in the evidence list.

### 3.4 ★ The executability gate — the one genuinely new component

**This is the part that does not exist anywhere and without which the migration
makes correctness worse.**

Today the engine's capabilities are implicit — spread across a regex table, a
`_flows_from` if-chain, a vocabulary dict and a renderer's word map, which is
exactly how three of eight offered cadences came to be un-executable without
anyone noticing. The gate makes them explicit and machine-checkable.

**The engine publishes a capability manifest** naming every semantic dimension
it can execute and, for closed sets, every value:

```yaml
cadence:
  executable: [weekly, biweekly, monthly, quarterly, annual, daily]
  refuses:    [payroll]            # a pay cycle is not a calendar period
day_rule:
  executable: [first_session_of_period, last_session_of_period]
allocation_method:
  executable: [equal_weight_at_purchase]
  refuses:    [inverse_volatility, risk_parity, min_variance, max_diversification]
stated_weights:
  executable: false                # declared 60/40 cannot be honoured — refuse
rebalancing:
  executable: false
```

**Three rules bind it:**

1. **The manifest is derived, never hand-maintained.** It is generated from the
   executor and asserted against it. A value in the manifest with no code path
   fails the build; a code path with no manifest entry fails the build. Had
   this existed, `annual` could not have been offered without being executable.
2. **The confirmation UI is generated from the manifest.** The product may not
   offer a user a choice it cannot execute. This alone closes the defect found
   today.
3. **Outside the manifest is a refusal, never a degradation.** The engine
   already has the right shape for this — the `unavailable` channel that
   returns no figure and states the reason. Every unexecutable dimension routes
   there.

`agentic-os`'s weakest seam is precisely here: `Mission.constraints` and
`IntentStep.constraints` are free-text `list[str]`, interpreted by substring
matching for `"human"`/`"review"` and otherwise decorative. A shipped template
carries `constraints=["never contact a customer twice in 24h"]` which is never
enforced *and never reported as unenforceable*. We should not adopt that seam.
The gate is what we contribute back.

---

## 4. What carries over

### 4.1 Kept and moved (high value, architecture-independent)

| Asset | Where | Size | Why it survives |
|---|---|---|---|
| Execution engine | `src/mission/{simulate,accounting,funding,signals,spec,scenario}.py` | ~2,400 lines | Deterministic arithmetic. Unchanged by the migration. |
| Market data + provenance | `src/market_data/` | ~2,100 lines | `MarketDataProvenance`, `NOT_RECORDED` vs `DENIED`, the egress/licensing gate. Maps onto `runtime_contracts.models.evidence`. |
| Vendor snapshot | `data/snapshots/prices-yahoo-20260807.*` | 2,664 × 44, 2016→2026 | Real prices, manifest, licensing record. Copy verbatim. |
| 144-prompt catalogue | `quantify-ui-agent/catalog_prompts.json` | 144 × 18 families | Labelled with `declares`. Becomes the mission controller's regression set. |
| 35-strategy corpus | `quantify-ui-agent/strategies.py` | EXECUTE 6 / CLARIFY 2 / REFUSE 16 / UNKNOWN 11 | A four-way *expectation* vocabulary that maps 1:1 onto the new states. The most valuable single artefact. |
| Sweep harness | `catalog_sweep.py`, `record.py` | ~400 lines | Response classification (RESULT/REFUSAL/PRODUCT_ERROR/HARNESS_ERROR) is reusable as-is. |
| 11 sweep evidence sets | `quantify-ui-agent/evidence_*/` | — | Before/after baselines. The only way to prove the migration did not regress meaning. |
| Runbook + PostgreSQL guarantees | `docs/` | — | Deployment discipline is not affected by the pivot. |

### 4.2 Kept as knowledge, not as code

The regexes go. What they encode must not.

- **The four defect classes** — reachability, partial reconstruction, authority
  inversion, verification defects. These are review criteria for the new
  system, and three of the four are *more* likely with a model in the loop.
- **The coverage state vocabulary** — `EXECUTED_AND_EVIDENCED`,
  `EXCLUDED_BY_USER`, `DECLARED_NOT_EXECUTED`, `EXECUTED_NOT_EVIDENCED`,
  `NOT_DECLARED`. This is the executability gate's output alphabet.
- **"Declared means the user expressed it, not that the compiler instantiated
  it."** The single most important sentence in the project.
- **"Absence is not always ignorance"** — already contract-encoded as
  `Disposition.PRODUCED_NOTHING` and `Verdict.NOT_APPLICABLE`.
- **The five-row stopping rule** — with the LLM, rows 1 and 2 (alternate
  notation / alternate wording) stop being work at all. Rows 3–5 remain, and
  become the gate's job.
- **The semantic dimension list itself** — cadence, day_rule, trigger
  semantics, execution timing, allocation method, stated weights, rebalancing,
  evaluation period, conditional amount, sell action. Extracted from
  `coverage.py`, this *is* the capability manifest's first draft.

### 4.3 Deleted

| Component | Lines | Replaced by |
|---|---|---|
| `src/mission/compiler.py` regex tables | ~1,585 | the LLM planner |
| `src/mission/parse_model.py` | 630 | `DecisionEvidence` + belief fusion |
| `src/mission/vocabulary.py` hand-written menus | 174 | generated from the capability manifest |
| Regex-driven parts of `coverage.py` | ~200 of 464 | the executability gate |

Roughly 2,600 lines deleted, ~400 lines of gate and manifest added, and the
phrasing treadmill ends.

### 4.4 Explicitly not carried

The ~3,900-test suite does not migrate wholesale. It found **zero** of the
semantic defects closed in the last slice; the deterministic corpus and
rendered-page inspection found all of them. Tests of the execution engine keep
their value. Tests that assert regex behaviour die with the regexes. That is a
saving, not a loss — but it means the corpus in §4.1 becomes the primary
correctness instrument, so it must be ported *first*, not last.

---

## 5. Phases

Each phase has an acceptance gate that is a measurement, not a judgement.
Phases 0–2 change nothing a user sees.

### Phase 0 — Contracts and baseline (1 week)

- Adopt `runtime-contracts` as a dependency; move `mission-runtime` from
  `NOT_LOCATED` by pointing it at `agentic_os.mission`.
- Port the 35-strategy corpus and the 144-prompt catalogue into a standalone
  harness that can run against *either* implementation.
- Freeze a baseline: run both corpora against today's build and store the
  results as the reference.

**Gate:** the harness reproduces the current build's results exactly, from a
clean clone, with no manual data fetching. *(Note: the vendor manifest and
licensing record were gitignored until today — this gate exists because that
class of "passes only on my machine" already happened once.)*

### Phase 1 — The capability manifest (1–2 weeks)

Build §3.4 against the **existing** compiler, before any LLM work.

- Derive the manifest from the executor.
- Assert manifest ↔ code both ways in the build.
- Generate the confirmation UI's choices from it.
- Route everything outside it to the existing `unavailable` channel.

**Gate:** a mutation that adds a value to the manifest with no code path fails
the build; a mutation that adds a code path with no manifest entry fails the
build. And the defect found today — offering "Every year" while executing one
payment — is unreachable by construction.

This phase pays for itself even if the migration stops here.

### Phase 2 — Mission runtime alongside (2–3 weeks)

- Stand up `agentic_os.mission` on the `feat/disagreement-decision-evidence`
  branch, with the three fixes in §3.3 contributed upstream.
- Model Quantify's plan as a `Mission`; the execution engine as an `Operator`
  declaring capabilities via `operator_sdk`.
- Wire `ContextRuntime` for the understanding step's retrieval and model tier.
- Run it in **shadow**: every user sentence goes to both the old compiler and
  the new controller; both readings are recorded as `DecisionEvidence`; the old
  compiler's answer is what the user sees.

**Gate:** shadow disagreement rate measured per semantic dimension across all
179 corpus prompts, with every material disagreement inspected by hand. No
cutover while any material disagreement is unexplained.

Shadow mode is the phase that makes this migration honest. It is also the
phase agentic-os's planner does not currently support — its deterministic
`TemplatePlanner` is a *fallback* on exception, never a cross-check, and
planner failures are swallowed without an event. Fixing that is part of this
phase and belongs upstream.

### Phase 3 — Cutover (1–2 weeks)

- The controller becomes authoritative for understanding; the compiler is
  removed.
- Material disagreement between readers routes to `WAITING_HUMAN` and surfaces
  as the existing confirmation question — the mechanism the product already
  has, and which pilot users already understand.

**Gate:** both corpora at parity or better against the Phase 0 baseline, on
meaning rather than on counts. A prompt that moves from EXECUTE to REFUSE is a
regression to explain; a prompt that moves from REFUSE to EXECUTE needs its
figure checked by hand before it counts as an improvement.

### Phase 4 — The surface (undecided, §6)

---

## 6. Open decisions

**1. What serves the pilot users.** Sidekick is not it (§2.3). Three options:

- *Keep Quantify's web workspace* and put the mission runtime behind it. Lowest
  risk; the confirmation UI, the evidence summary and the disclosure already
  work and were built for exactly this interaction.
- *Build a conversational surface* on the mission runtime's `/inbox` and
  `/cockpit`. More faithful to the new architecture; discards a working UI.
- *LibreChat*, which context-runtime already has a deployment for. Fastest to a
  chat interface; no notion of a plan, a run or a worksheet.

My recommendation is the first, and to treat the surface as the *last* thing to
change rather than the first. Nothing about the phrasing treadmill is a UI
problem.

**2. Where the plan lives.** The mission event log is append-only and
replayable, which is what a plan's history wants. But Quantify's plans are
currently PostgreSQL rows with migrations, parity tests and a restore drill.
Running both is two systems of record — the failure this project has already
catalogued as authority inversion. This needs deciding before Phase 2, not
during it.

**3. Which model, and where it runs.** `agentic-os` defaults to a local
`Qwen3-Coder-Next-NVFP4` over an OpenAI-compatible endpoint. Quantify's
licensing record permits "small snippets during testing... never the full
series" to a model provider, recorded as operator-only with no automated path.
A hosted model in the understanding path sends *user sentences*, not prices —
outside what that record covers either way. It needs its own answer before
Phase 2 ships to a user.

**4. Whether `context-runtime` earns its place in v1.** Its value here is
retrieval and model-tier planning under budget. Quantify's understanding step
retrieves almost nothing — the user's sentence and a capability manifest. The
honest read is that it is the *right* component for a later problem (grounding
against a strategy library, a documents corpus, a user's own history) and adds
a dependency without a job in Phase 2. Recommend deferring it to Phase 4 and
building Phase 2 against the mission runtime alone.

---

## 7. What this does not fix

Stated plainly, because the migration will be judged against expectations set
now.

- **Unsupported dimensions stay unsupported.** Risk parity, inverse volatility,
  direct indexing, tax-loss harvesting and explicit 60/40 weights are refused
  today because the *engine* cannot execute them. A model that understands them
  perfectly does not make them run. Of the 144 catalogue prompts, ~83 are about
  accounts, liabilities and cashflows rather than instruments, and remain
  refusals after every phase above.
- **The refusal rate will not move much at first.** Today: 6 figures, 5
  questions, 133 refusals out of 144. Most of that is engine capability, not
  understanding. Phase 1 will make the refusals *honest and legible*; only
  engine work makes them into results.
- **Model non-determinism enters the system.** Two identical sentences may
  produce different intents. The event log makes this *auditable* rather than
  invisible, and the executability gate makes it *safe*, but neither makes it
  go away. A plan's compiled form must be pinned and re-run from the pinned
  form, never re-derived from the sentence.
- **A model will confidently mis-read a sentence.** The disagreement gate
  catches contested readings, not confidently wrong unanimous ones. The corpus
  is the only instrument for those, which is why §4.4 puts it first.

---

## 8. Recommended immediate next step

Phase 1 — the capability manifest — against the existing compiler.

It is the smallest piece of work that is valuable whether or not the rest
happens, it closes a defect class that has produced every expensive error in
this project's history, and it produces the artefact the mission controller
needs to be safe. It requires no new dependency, no model, and no decision from
§6.
