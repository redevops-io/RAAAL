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
   declare exactly which dimensions and values it can execute; anything
   understood outside that set must refuse, not degrade.

And those two jobs belong to **two different runtimes**, which is the single
most important structural decision in this document (§3):

> **Discovery may understand more than Mission can execute. Mission may never
> execute less than the verified intent while pretending they are equivalent.**

That sentence is the whole project restated. `cadence='annual'` was understood
correctly and executed as one payment, and nothing said so. Under the split
below, understanding it is Discovery's job and being unable to run it is
Mission's job to *say*, out loud, as a refusal.

Job 2 is new work. Nothing in the organisation has it (§3.6).

---

## 2. What the repositories actually are

All 16 public repositories in the organisation were read or triaged, plus the
private `runtime-contracts`. Two of the three named in the brief are not quite
what it assumed, and the difference changes the plan. §2.5 covers the rest.

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
not what this component does. Its real job here is to support Discovery later,
once meaning must be grounded in more than a sentence — see §3.7, and §8.2 for
the measurement that says it should wait.

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

**It is private.** `api.github.com/repos/redevops-io/runtime-contracts` returns
404 while `RAAAL` itself is a public repository under AGPL-3.0 + Commons
Clause. Making a private package the schema spine of a public product is a
decision with consequences — a clone cannot build — and it needs answering
before Phase 0, not during it. The options are: publish it, vendor the subset
we use, or depend on it only in tooling that never ships.

`/projects/runtime-contracts` already names this exact set of implementations:

```
| quantify         | SPECIFIED | CONFORMANT | gap |
| context-runtime  | SPECIFIED | CONFORMANT | gap |
| mission-runtime  | NOT_LOCATED | NOT_LOCATED | ok |
| discovery-runtime | NOT_LOCATED | NOT_LOCATED | ok |
```

It also names **`discovery-runtime`**, and says what to do about it:

> `mission-runtime` and `discovery-runtime` are not missing repositories — they
> are **unbuilt components**. Phase B therefore does not need a repository
> created to satisfy an architecture diagram. It can be implemented in the
> deployed control plane or **in Quantify**, provided the `MissionProgram`
> contract is canonical and externally owned.

That is an explicit sanction for the split in §3: Quantify becomes the
reference **Discovery Runtime** implementation. It is also the most natural
reading of Quantify's own history — the understanding layer we are replacing
*is* a discovery runtime in embryo, and everything it learned about ambiguity,
materiality and refusal is discovery knowledge.

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

### 2.5 The rest of the organisation

All 16 public repositories were read or triaged. Three are forks of third-party
projects (`HippoRAG`, `graphiti`, `openclaw`) and one is unrelated
(`ingress-nginx`); `RAAAL` is this repository. The remainder:

| Repo | Licence | Verdict |
|---|---|---|
| `mission-sdk` | Apache-2.0 | **Adopt.** The intended front door. §2.6 |
| `agentic-os` | AGPL-3.0-or-later | **Adopt** (mission kernel only). §2.2 |
| `runtime-contracts` | private | **Adopt with a caveat.** §2.4 |
| `context-runtime` | AGPL-3.0 | **Defer to Phase 5.** §2.1, §3.7, §8.2 |
| `redevops-rag` | AGPL-3.0-or-later | **Defer.** A Context Runtime *store adapter* — the adapter tunes RAG rather than being tuned by it. Its own README says small structured config "belongs in git, not in a RAG", which is the answer for a capability manifest. Base install pulls `sentence-transformers` and therefore torch. |
| `agent-harness` | AGPL-3.0-or-later | **Do not adopt.** §2.7 |
| `sidekick` | AGPL-3.0-or-later | **Use as a dev tool, not a component.** §2.3 |
| `personal-tax-runtime` | — | **Not adoptable.** §7.1 |
| `accounting-runtime` | — | **Not relevant.** §7.1 |
| `multiagent-orchestration-benchmark` | MIT | **Evidence, and it cuts against part of this plan.** §8.1 |
| `DataOpsBench` | Apache-2.0 | **Borrow one pattern.** §4.2 |
| `redevops-demo` | — | **Deployment reference.** §5, Phase 3 |

Note the licence split. `RAAAL` is public under AGPL-3.0 + Commons Clause, so
AGPL dependencies are compatible; `mission-sdk` being Apache-2.0 is a bonus,
not a requirement. The item that needs a decision is `runtime-contracts` being
private (§2.4).

### 2.6 `mission-sdk` — adopt, and someone has already sketched our integration

`mission-sdk` is the curated boundary over `agentic_os.mission`: you author a
mission, hold **one versioned artifact** (`MissionProgram`), and operate it
without importing runtime internals.

It matters here for a specific reason. `examples/from_proposal/mission.py`
already names Quantify in its docstring and maps a financial scenario onto
exactly our pipeline:

```
market_data_loaded → portfolio_simulated → metrics_computed → worksheet_saved
```

with `research.save_worksheet` carrying
`constraints=["persists a research record — requires human confirmation"]`.
Someone has already thought about this integration; we should start from their
sketch rather than a blank file.

What it gives us that we would otherwise write:

- `MissionProgram.from_proposal(...)` — the compiled plan as one hashable,
  versioned artifact.
- `CaseBundle` — a portable run record with a sha256 content digest,
  `integrity_ok()` tamper detection, `replay_bundle` (rebuild a fresh runtime,
  assert the same terminal state) and `diff_bundles` (report first divergence).
  This is the export/reproduce story we would otherwise build by hand.
- `rdo mission ci` — `feasibility · budget · run · regression · replay` as a
  deploy gate, with `golden` as expected final world-state.

Two cautions: it depends on `agentic-os` via a **bare git commit pin with no
tag**, and it is `0.1.0a0`.

### 2.7 `agent-harness` — do not adopt

Its README offers "tools, permissions, evals, guardrails". Only permissions is
substantial. The LLM client is an offline stub that echoes the prompt; the
agent loop is a single hardcoded step; `redact()` and `validate_output()` are
never called by anything.

The eval module — the part that looked like a home for our corpora — is 16
lines whose scorer is `passed = t.get("expected", "") in str(out)`, with no
corpus format, no label vocabulary, no tests and no callers. Since its default
agent echoes its input, every task whose `expected` is a substring of its
`input` passes. We would be replacing that file, not extending it.

---

## 3. Target architecture

### 3.1 `VerifiedIntent` — the artifact at the centre

Start here, because everything else in this architecture is defined by its
relationship to this one object.

> **`VerifiedIntent` is the canonical description of what the user asked for.**
> **Everything above creates it. Everything below consumes it.**
> **Nothing below may rewrite it.**

That third line is the whole discipline. A `ScenarioSpecification` is derived
from it, a figure is derived from that, and at no point does anything
downstream get to adjust the intent to suit what it can do. When the engine
cannot honour a field, the answer is a refusal naming the field — never a
quietly adjusted intent that happens to be executable.

It is a **runtime boundary**, so `runtime-contracts` should own it rather than
Quantify inventing the Discovery→Mission payload privately. Sketched from what
Quantify already knows it needs:

```yaml
VerifiedIntent
  content_hash:      sha256:...        # identity; plans re-run from this
  produced_by:       discovery-runtime@0.4.2
  objective:         evaluate_investment_strategy

  assets:            [SPY]             # as written; resolution recorded, not applied
  actions:           [BUY]
  funding:           {amount: 1000, mode: EVENT_TRIGGERED}
  condition:         {type: MOVING_AVERAGE_CROSS, direction: BELOW, window: 200}
  evaluation_period: {trailing: 5 years}

  # every field carries its own attribution
  trigger_semantics:
    value:        crossing
    author:       USER                 # who asserted this value
    produced_by:  discovery-runtime@0.4.2   # which runtime version produced it
    confidence:   1.0
    source_span:  "whenever it crosses below"
    evidence:     [DecisionEvidence, ...]   # every reader, including the losers

  unresolved: []      # dimensions deliberately left open — not the same as absent
  amendments: []      # what the user changed, and when
```

Five properties, each bought with a defect:

- **`author` is explicit.** "Declared means the user expressed it, not that the
  compiler instantiated it." A field the model inferred and a field the user
  stated must not be the same shape, or the product offers its own assumption
  back as the user's choice — which it did, for `execution_timing`.
- **`produced_by` is separate from `author`, and both are needed.** They answer
  different questions: `author` is *who asserted this value* — the user, a
  reader, a policy; `produced_by` is *which runtime version produced this
  artifact*. A user-authored value produced by `discovery-runtime@0.4.2` and
  the same value produced by `@0.5.0` are the same assertion by the same
  author, and may still differ in how they were elicited. Without it, a replay
  that diverges after a Discovery upgrade is undiagnosable, and a migration
  cannot tell which intents were produced by a version with a known reading
  bug. The same field belongs downstream:

  ```yaml
  MissionProgram
    compiled_from: sha256:...          # the VerifiedIntent, by hash
    compiled_by:   mission-runtime@0.6.1
  ```

  which makes the full chain attributable: this figure came from that program,
  compiled by that runtime version, from that intent, produced by that
  Discovery version, authored in those places by the user.
- **Absent ≠ rejected.** `unresolved[]` keeps "we did not ask" distinguishable
  from "the user declined to constrain it". `Verdict` and `Disposition` in
  `runtime-contracts` already carry this distinction; the intent must not throw
  it away.
- **Losing readings are kept.** `evidence[]` holds every reader's view, not
  just the winner. A field that was contested and resolved is a different fact
  from one that was never in doubt, and only the first justifies asking the
  user again when the readers change.
- **It is pinned and re-run from, never re-derived.** A model is
  non-deterministic; the same sentence twice may not produce the same intent.
  The hash makes a plan reproducible; re-parsing the sentence on reopen would
  make history rewritable. This is the migration's version of the rule that
  already governs market-data provenance.

### 3.2 The two runtimes that surround it

```
      user's sentence
            │
┌───────────▼──────────────────────────────────────────────┐
│  sidekick-class conversational surface  (TO BE DECIDED §6)│  Quantify web UI stays
└───────────┬──────────────────────────────────────────────┘
            │  the user's own words
┌───────────▼──────────────────────────────────────────────┐
│  DISCOVERY RUNTIME       "What did the human mean?"      │
│  · natural language → verified, attributable intent      │
│  · the ONE LLM call                                      │
│  · ambiguity detection · competing interpretations       │
│  · DecisionEvidence per reader, no reader privileged     │
│  · source spans · amendments · confirmation              │
│  · material disagreement → WAITING_HUMAN                 │
│         ↳ the question is "what did you mean?"           │
└───────────┬──────────────────────────────────────────────┘
            │  ★ VerifiedIntent  — the runtime boundary
            │    (owned by runtime-contracts: identity, hash,
            │     evidence refs, verdict, disposition, version)
┌───────────▼──────────────────────────────────────────────┐
│  MISSION RUNTIME    "Can I execute exactly that?"        │
│  · capability-bind against the CAPABILITY MANIFEST §3.6  │
│  · executable   → compile                                │
│  · unexecutable → NAMED REFUSAL, never a substitution    │
│  · needs approval → WAITING_HUMAN                        │
│         ↳ the question is "may I do this?"               │
│  · event-sourced, append-only, replayable                │
└───────────┬──────────────────────────────────────────────┘
            │  MissionProgram  (mission-sdk: one versioned artifact,
            │                   validate · simulate · ci · bundle · replay)
            │  ScenarioSpecification (physical, validated)
┌───────────▼──────────────────────────────────────────────┐
│  QUANTIFY EXECUTION ENGINE        kept, unchanged        │
│  simulate · accounting · funding · signals · benchmarks  │
│  market data + provenance                                │
└──────────────────────────────────────────────────────────┘

  mission-sdk ────── the authoring/operating boundary (Apache-2.0)
  runtime-contracts ── owns the VerifiedIntent and MissionProgram contracts
  context-runtime ── DEFERRED; later supports DISCOVERY, see §3.7 and §8.2
```

### 3.2a Two boundaries, not one

**(a) The LLM replaces the recogniser. It never touches arithmetic.**

No model computes a return, a contribution schedule, a weight or a tax
treatment. Its entire job is `str → VerifiedIntent`. Everything below stays
deterministic, and the existing `Verdict`/coverage discipline applies to its
output rather than being replaced by it. A money-weighted return that a model
produced is a number nobody can reproduce, and the whole provenance chain this
project built exists to make figures reproducible.

**(b) Meaning and executability are different questions, asked by different
runtimes.**

The earlier draft of this document put the LLM call inside Mission Runtime.
That was wrong, and wrong in a way this project has already paid for: it makes
one component responsible both for *what the user said* and for *what we can
do about it*, and a component holding both is a component that can quietly
reconcile them. Every expensive defect here has that shape — a declared
dimension silently becoming an executable one.

Split, the two questions cannot be traded off against each other:

```
Discovery:  allocation_method = inverse_volatility
            confidence = high · source span = "by inverse volatility"
                     ↓
Mission:    QuantifyEngine.capabilities.allocation_method
              = [equal_weight_at_purchase]
                     ↓
            UNSUPPORTED_CAPABILITY → named refusal
```

**Mission must never reinterpret `inverse_volatility` as `equal_weight`.** It
has no authority to: the intent is already verified, and its author is the
user. The only moves available are compile, refuse, or ask permission.

The corollary is liberating for Discovery. Its vocabulary does **not** have to
be constrained to what today's engine can run. A model may understand
arbitrarily rich intent; Mission executes only the subset its manifest claims.
That is far safer than the alternative — teaching the reader to only see what
the engine can do — because a reader that cannot express "inverse volatility"
will express it as something else.

### 3.3 Two human gates, asking different things

The split produces a distinction the current product conflates:

| Gate | Question | Raised by | Resolved by |
|---|---|---|---|
| Discovery `WAITING_HUMAN` | *"What did you mean?"* | material reader disagreement, or an unresolved dimension | the user, as an **authoritative observation** that becomes part of the verified intent |
| Mission `WAITING_HUMAN` | *"May I do this?"* | `approval_required` on a capability, budget, policy | an approver, as a **permission** — it does not change the intent |

These must not share a queue or a UI affordance. Answering "I meant crossing,
not persistent" is authoring; answering "yes, save that worksheet" is
authorising. Quantify's existing confirmation questions are all the first kind;
it has no instance of the second yet, which is precisely why they are easy to
merge by accident.

Once an intent reaches Mission, the semantic question is settled. Mission can
still discover *execution* disagreement — two capabilities claiming to satisfy
the same outcome differently — but it should never be adjudicating what the
user's English meant.

### 3.4 `ScenarioSpecification` is the physical form below the boundary

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
It is the **physical** plan and sits *below* the boundary: Discovery emits a
`VerifiedIntent`, Mission capability-binds it and compiles a
`ScenarioSpecification` only if the manifest permits.

Note the two must not be confused. `ScenarioSpecification` is shaped by what
the engine can run — `flow_schedule`, `allocation_rule`, `funding` are all
executable forms. `VerifiedIntent` is shaped by what a person can mean, and may
legitimately contain `allocation_method: inverse_volatility` forever, as a
faithful record of a request this build refuses.

### 3.5 The disagreement branch is our own rule, and it belongs to Discovery

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

**But its home is Discovery, not Mission.** The branch currently sits in the
mission kernel because that is where the world-state blackboard lives. Under
the split, reader disagreement is a *meaning* question:

```
Reader A → crossing        Reader B → persistent
                 ↓
        Discovery: material disagreement
                 ↓
             ask the user
                 ↓
    VerifiedIntent.trigger_semantics = crossing · author = USER
```

By the time Mission sees it, that field has one value and a named author. This
is the same mechanism, moved one layer up — and moving it is what stops Mission
from ever being in a position to resolve English.

Mechanically the kernel's `WorldState` already supports this: it stores
observations and re-fuses on every read, so a Discovery runtime can use the
same event-sourced blackboard while the *question it asks* belongs to a
different runtime.

### 3.6 ★ The capability manifest — the genuinely new part

It lives in **Mission**, and it is what Mission consults to answer "can I
execute exactly that?".

**What is new here, stated precisely.** Not the idea of describing
capabilities — that exists in several places already, and the manifest is
largely an assembly of them:

| Already exists | What it gives |
|---|---|
| Quantify's `coverage.py` | the semantic-dimension inventory, and the declared-vs-executed vocabulary |
| `agentic_os` `CapabilitySpec` + capability binding | fail-closed *positive* declaration: `provides`, `permissions`, `approval_required` |
| `mission-sdk` `validate()` | proof that a compiled program's needs all bind |
| Quantify's `unavailable` channel | a refusal surface that returns no figure and states a reason |

What does not exist anywhere is the narrow thing:

> **an explicit semantic capability contract, derived from the executor and
> asserted against it in both directions.**

Each existing piece is either hand-maintained, positive-only, or downstream of
the decision that matters. `CapabilitySpec` can say a capability *provides*
`portfolio_simulated`; nothing anywhere can say the simulator executes
`cadence ∈ {weekly, monthly, quarterly, annual}` and refuses `payroll`, and
nothing checks that claim against the code. That gap is what let three of eight
offered cadences become un-executable silently, and closing it is the work.

It must **not** be used to constrain Discovery's vocabulary. That would put the
engine's limits inside the reader, and a reader that cannot express "inverse
volatility" does not refuse — it picks the nearest thing it *can* express. The
manifest's job is to make a refusal loud, not to make an intent unspeakable.

Today the engine's capabilities are implicit — spread across a regex table, a
`_flows_from` if-chain, a vocabulary dict and a renderer's word map, which is
exactly how three of eight offered cadences came to be un-executable without
anyone noticing. The manifest makes them explicit and machine-checkable.

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
2. **Any choice the product *offers* must be in the manifest.** Discovery may
   understand anything; a menu is a promise. Where the product presents a
   closed set to choose from, that set is generated from the manifest. This
   alone closes the defect found today, where "Every year" was offered and
   executed once.
3. **Outside the manifest is a refusal, never a degradation.** The engine
   already has the right shape for this — the `unavailable` channel that
   returns no figure and states the reason. Every unexecutable dimension routes
   there, naming the dimension and the value it could not run.

**The negative half, checked.** I searched `mission-sdk` and the whole `agentic_os`
tree for `refuse|refusal|clarify|unsupported|not_supported|executab|forbid` —
no substantive hits. `CapabilitySpec` has `provides`, `permissions`,
`side_effecting`, `approval_required`, `undo`; there is **no negative or
exclusion field anywhere**, and no EXECUTE/CLARIFY/REFUSE/UNKNOWN vocabulary in
either package.

What the SDK does give is the **back half** of the gate, and it is worth having:
`validate(program, operators)` compiles the program and fails closed on an
unbindable need, a missing grant, or a cycle. That answers *"can this compiled
program run?"*. It does not answer *"is this sentence executable, and if not,
is that a clarify or a refusal?"* — and it collapses its three failure modes
into one `Check`, so the reasons are only distinguishable by string-matching.

So the split is three-deep, and only the middle row is ours to build:

```
  Discovery (net-new, ours)   prose         →  VerifiedIntent | WAITING_HUMAN
  capability bind (net-new)   intent        →  EXECUTE | REFUSE(reason) | NEEDS_APPROVAL
  mission-sdk validate()      program       →  compiles / CompileError
```

Note `CLARIFY` has moved. In the earlier draft the gate emitted it; under the
split it cannot, because by then the intent is verified and asking "what did
you mean?" is Discovery's gate, upstream. Mission's three outcomes are run it,
refuse it by name, or ask permission.

`agentic-os`'s weakest seam is precisely the half we are building:
`Mission.constraints` and `IntentStep.constraints` are free-text `list[str]`,
interpreted by substring matching for `"human"`/`"review"` and otherwise
decorative. A shipped template carries
`constraints=["never contact a customer twice in 24h"]` which is never enforced
*and never reported as unenforceable*. We should not adopt that seam. The gate
is what we contribute back.

### 3.7 Where Context Runtime attaches, eventually

The split clarifies this too. Context Runtime **supports Discovery; it does not
own the user's durable intent** — which is fortunate, because §2.1 established
that it cannot: `BuiltContext` is never persisted and there is no decision
record in it.

Today Discovery's inputs are a sentence and a manifest, so there is nothing to
assemble and nothing to route. It earns its place when Discovery needs to
ground meaning in more than the sentence:

```
   strategy library · account history · prior decisions · documents
                              ↓
                      Context Runtime
                              ↓
  user → Discovery Runtime → VerifiedIntent → Mission Runtime → execution
```

At that point "which context does this interpretation need, and which model
tier should read it" is exactly its question. Until then it is a dependency
without a job, and §8.2 has the measurement suggesting the machinery would not
improve accuracy anyway.

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

### 4.2a Where the corpora live, and one pattern worth borrowing

`agent-harness` is not the home (§2.7). Two better answers:

**`agentic_os.mission.evaluation`** (contract `evaluation/v10`) makes a
benchmark "a **versioned Mission program**: dataset + protocol + steps, executed
reference-first, verified, and **published only when it verifies**". It ships
`Dataset`, `EvaluationProtocol`, `Run`, `VerificationCheck`,
`VerificationResult`, `Assessment`, `Finding` and a `Publication` decision
(`PUBLISH` / `NO_MATERIAL_IMPACT` / `HOLD` / `REQUIRE_REVIEW`). Note
`mission-sdk` does **not** re-export it, so this is a direct dependency on the
runtime. It has the harness; it does not have a *response* taxonomy, so
EXECUTE/CLARIFY/REFUSE/UNKNOWN is ours to bring — and to contribute back.

**`DataOpsBench`'s `validate` verb** is the pattern to copy, and it is this
project's own mutation discipline applied to an eval corpus: build the defect,
assert the gate fails, apply the reference fix, assert the gate passes. Its
framing — *"a gate that passed on the defect would be worthless"* — is exactly
"close every slice with a mutation". Its structure (`defect → symptom →
reference fix → deterministic gate → blinded judge only for the subjective
residual`) should shape the corpus port, and the principle "deterministic gates
first, a judge only for the residual" is the right default for a product where
figures are checkable.

The concrete consequence for Phase 0: **every expectation in the 35-prompt
corpus must be proven discriminating before the corpus is trusted as a
baseline.** A REFUSE expectation that would also pass on a broken build is not
evidence of anything.

### 4.3 Deleted

| Component | Lines | Replaced by |
|---|---|---|
| `src/mission/compiler.py` regex tables | ~1,585 | Discovery Runtime |
| `src/mission/parse_model.py` | 630 | `DecisionEvidence` + belief fusion, in Discovery |
| `src/mission/vocabulary.py` hand-written menus | 174 | generated from Mission's capability manifest |
| Regex-driven parts of `coverage.py` | ~200 of 464 | the capability manifest + `VerifiedIntent.unresolved[]` |

Roughly 2,600 lines deleted, ~400 lines of manifest and binding added, plus the
Discovery Runtime — which is net-new construction, not a port (§6.5).

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

- Decide the `runtime-contracts` question (§2.4) and adopt it; move
  `mission-runtime` from `NOT_LOCATED` by pointing it at `agentic_os.mission`.
- **Draft the `VerifiedIntent` contract (§3.1) in `runtime-contracts`**, not
  in Quantify. It is a runtime boundary, and the contracts package explicitly
  sanctions Quantify implementing `discovery-runtime` provided the contract is
  "canonical and externally owned". Drafting it here would forfeit that.
- While there, add `produced_by` to the artifacts that lack it. Neither
  `MissionProgram` nor Quantify's own records carry the runtime version that
  produced them today, and the first cross-version replay divergence is the
  wrong moment to discover that.
- Port the 35-strategy corpus and the 144-prompt catalogue into a standalone
  harness that can run against *either* implementation, shaped by §4.2a.
- Freeze a baseline: run both corpora against today's build and store the
  results as the reference.

**Gate, two parts:**

1. The harness reproduces the current build's results exactly, from a clean
   clone, with no manual data fetching. *(The vendor manifest and licensing
   record were gitignored until today — this gate exists because that class of
   "passes only on my machine" already happened once.)*
2. **Every expectation discriminates**, DataOpsBench-style: for each corpus
   entry, a mutation exists that flips it. An expectation nothing can fail is
   removed or rewritten, not counted.

*Cautionary note for this phase.* `accounting-runtime` ships a `SHA256SUMS`
that records the hash of the string `404: Not Found` for two data files whose
download had silently failed — so a checksum pass reports clean over corrupt
data. Verify that our fixtures are what they claim to be, not merely that they
hash to what someone recorded.

### Phase 1 — The capability manifest (1–2 weeks)

Build §3.6 against the **existing** compiler, before any LLM work. This is
Mission's half, and it is buildable with today's parser standing in for
Discovery — which is the reason to do it first.

- Derive the manifest from the executor.
- Assert manifest ↔ code both ways in the build.
- Generate from it any closed set the product offers the user.
- Route everything outside it to the existing `unavailable` channel, naming the
  dimension and the value.

**Gate:** a mutation that adds a value to the manifest with no code path fails
the build; a mutation that adds a code path with no manifest entry fails the
build. And the defect found today — offering "Every year" while executing one
payment — is unreachable by construction.

This phase pays for itself even if the migration stops here.

### Phase 2 — Mission binding, against today's parser (1–2 weeks)

Mission first, because it can be built and proven while Discovery is still the
old compiler. Nothing here needs an LLM.

- Stand up `agentic_os.mission` on the `feat/disagreement-decision-evidence`
  branch, with the three fixes in §3.5 contributed upstream.
- Adopt `mission-sdk` as the boundary, starting from
  `examples/from_proposal/mission.py` (§2.6). Quantify's plan becomes a
  `MissionProgram`; the execution engine an `Operator` declaring capabilities
  via `operator_sdk`.
- Have today's compiler emit a `VerifiedIntent` — it already produces every
  field, and this proves the contract carries what the engine needs before a
  model is involved. It stamps `produced_by: quantify-compiler@<version>`,
  which is exactly the point: when Discovery replaces it in Phase 3, the two
  eras of intent are distinguishable in the record rather than merged.
- Record `compiled_from` (the intent hash) and `compiled_by` on every
  `MissionProgram`, so a figure traces to a program, to a runtime version, to
  an intent, to an author.
- Adopt `CaseBundle` for run export and `rdo mission ci` as a deploy gate.
- **Do not** wire `ContextRuntime` (§3.7, §8.2).

**Gate:** both corpora produce identical figures through the mission path and
the current path, and every Phase 1 refusal arrives as a named
`UNSUPPORTED_CAPABILITY` rather than a `CompileError` string.

### Phase 3 — Discovery Runtime in shadow (2–3 weeks)

Only now does the model enter, and it enters behind a mirror.

- Build the Discovery Runtime: prose → `VerifiedIntent`, with `DecisionEvidence`
  per reader, materiality, and the "what did you mean?" gate (§3.3).
- Run it in **shadow**: every user sentence goes to both the old compiler and
  Discovery; both readings are recorded as `DecisionEvidence` on the same
  fields; the old compiler's answer is what the user sees and what runs.

**Gate:** shadow disagreement rate measured per semantic dimension across all
179 corpus prompts, with every material disagreement inspected by hand. No
cutover while any material disagreement is unexplained.

Shadow mode is the phase that makes this migration honest, and it is the phase
`agentic-os` does not currently support: its deterministic `TemplatePlanner` is
a *fallback on exception*, never a cross-check, and planner failures are
swallowed without an event. Under the split that is Discovery's problem to fix,
and the fix belongs upstream — a model reading and a deterministic reading are
two readers, and the branch's own rule says neither is privileged.

**Gate:** shadow disagreement rate measured per semantic dimension across all
179 corpus prompts, with every material disagreement inspected by hand. No
cutover while any material disagreement is unexplained.

### Phase 4 — Cutover (1–2 weeks)

- Discovery becomes authoritative for meaning; the regex compiler is removed.
- Its "what did you mean?" gate surfaces as the existing confirmation question
  — the mechanism the product already has and pilot users already understand.
- Mission's "may I do this?" gate gets its own affordance, kept separate (§3.3).

**Gate:** both corpora at parity or better against the Phase 0 baseline, on
meaning rather than on counts. A prompt that moves from EXECUTE to REFUSE is a
regression to explain; a prompt that moves from REFUSE to EXECUTE needs its
figure checked by hand before it counts as an improvement. And the migration's
central invariant holds on every prompt: **nothing Mission executed was less
than what Discovery verified, unless it said so by name.**

### Phase 5 — The surface (undecided, §6)

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

**2. Where the plan lives — and now, where the *intent* lives.** The mission
event log is append-only and replayable, which is what a plan's history wants.
But Quantify's plans are currently PostgreSQL rows with migrations, parity
tests and a restore drill. Running both is two systems of record — the failure
this project has already catalogued as authority inversion.

The split sharpens this into two questions, and they may have different
answers. The `VerifiedIntent` is authored, amended and confirmed by a user, so
its history is the thing that must never be rewritten; the
`ScenarioSpecification` is derived from it and could in principle be recomputed
from a pinned intent. Needs deciding before Phase 2.

**3. Which model, and where it runs.** `agentic-os` defaults to a local
`Qwen3-Coder-Next-NVFP4` over an OpenAI-compatible endpoint. Quantify's
licensing record permits "small snippets during testing... never the full
series" to a model provider, recorded as operator-only with no automated path.
A hosted model in Discovery sends *user sentences*, not prices — outside what
that record covers either way. It needs its own answer before Phase 3 ships to
a user, and Discovery is where it now bites.

**4. Whether `context-runtime` earns its place in v1.** Answered, with
evidence: no. Its value is assembling context for Discovery, and Discovery's
inputs today are one sentence. §8.2 shows the retrieval ladder *reducing*
accuracy while cutting tokens. Defer to Phase 5 (§3.7).

**5. How much of Discovery to build ourselves.** `discovery-runtime` does not
exist anywhere. Quantify would be the reference implementation, which the
contracts package sanctions — but it means Phase 3 is genuinely new
construction, not integration, and the estimate should be read with that in
mind. The mitigating fact is that Quantify has already built most of a
discovery runtime once, badly, and knows exactly what it needs to do.

---

## 7. What this does not fix

Stated plainly, because the migration will be judged against expectations set
now.

- **Unsupported dimensions stay unsupported.** Risk parity, inverse volatility,
  direct indexing, tax-loss harvesting and explicit 60/40 weights are refused
  today because the *engine* cannot execute them. A model that understands them
  perfectly does not make them run. Of the 144 catalogue prompts, ~83 are about
  accounts, liabilities and cashflows rather than instruments, and remain
  refusals after every phase above. §7.1 explains why the organisation's two
  financial repos do not change this.

### 7.1 The two financial runtimes do not close the account/tax gap

`personal-tax-runtime` and `accounting-runtime` look, from their READMEs, like
they might serve the ~83 refused prompts. They do not, and the reason is
structural rather than a matter of coverage.

Neither is a runtime. They are ground-truth **data generators** — 529 and 813
lines, one commit each, **zero tests and zero CI** — that drive somebody else's
engine (Tax-Calculator/PolicyEngine; ERPNext/Odoo) and write JSONL, behind a
read-only dashboard. Despite MANUALs describing a Mission Runtime / Context
Runtime / RAG architecture, neither imports any of it.

The three entities the refused prompts need are all absent:

| Needed for | Missing |
|---|---|
| asset location, withdrawal sequencing, RMDs | **an account.** No account entity exists. A generated `1099-R` carries only `box1_gross_distribution` — no box 2a, no box 7 code — so an RMD, a Roth conversion and an ordinary pension payment are indistinguishable. |
| wash sales, tax-loss harvesting, RSU disposition | **a tax lot.** Capital gains are two aggregate scalars. No lots, no basis, no acquisition dates. |
| Roth ladders, debt-payoff-vs-invest, liquidity | **a time axis.** `YEAR = 2025`. One year, one snapshot, no projection. |

More important than the gap: **they publish numbers nothing validated**, which
is the failure this project's refusals exist to prevent. The headline "refund"
KPI is a `rng.uniform(0.08, 0.16)` withholding *guess* minus a real liability;
it omits self-employment tax entirely; and when the two engines disagree it
silently takes Tax-Calculator's answer while recording `agree_core: false`
beside it. `accounting-runtime`'s cross-validation covers only single-line,
no-tax, USD invoices — excluding the four scenarios where two engines could
plausibly differ — and its "correct-by-construction" check is near-tautological,
since ERPNext will not submit an unbalanced voucher in the first place.

**These are not a governance model to copy.** On the discipline that matters
here they are a step backwards from where Quantify already is.

The one reusable asset is the ~20-line **Tax-Calculator (CC0)** adapter, worth
about a day to lift if tax-aware evaluation is ever wanted. That is reusing
`taxcalc`, not this repo — and `taxcalc` is a static-year microsimulation model
for policy scoring. It returns AGI and liability for one year's inputs. It will
not sequence withdrawals or optimise a conversion. That planning layer does not
exist anywhere in the organisation and would be built from scratch.
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

## 8. Evidence from the organisation's own experiments — including against this plan

Two repositories are controlled experiments. Both point the same direction, and
a plan that proposes adopting machinery should quote them rather than ignore
them.

### 8.1 Formal orchestration did not beat a good supervisor

`multiagent-orchestration-benchmark` pre-registered the question, the decision
rule, and a null hypothesis placing the burden of proof on the formal arm:
*"Ties default to the simpler system."*

| | Arm A (emergent) | Arm B (formal) |
|---|---|---|
| Gate pass | 11/11 | 11/11 |
| Manager cost | $0.318 | $0.320 |
| Worker secs, median | 116 | 119 |
| Blinded pairwise quality | 5 wins | 5 wins (+1 tie) |

Its own conclusion: **"the evidence does not support adopting formal
runtime-managed orchestration as a blanket mechanism."** And its roadmap
advice — keep the cheap plumbing (isolation, provenance traces, budgets, scoped
briefs); don't build generic semantic merge operators or deep agent hierarchies
"on the strength of it feels more principled."

**What it does and does not bear on.** It did **not** test Mission Runtime. The
"formal" arm is ~13 lines of inline Python — a declared-output length check, one
retry, and a `kotlinc` compile probe — and *both* arms ran workers through
Sidekick. Nothing about approval gates, budgets, saga, replay or evidence
capture was varied or measured, and gate-pass was 11/11 on both sides, so the
gates had no discriminating power at all.

So it is legitimate evidence against *"a generic integration/merge layer
improves output quality"* — a claim this plan does not make — and silent on the
governance properties that are the actual reason to adopt the mission kernel.
Three caveats worth carrying: no judge code, prompts or transcripts are
committed, so the 5–5–1 is an asserted value; the pre-registered Wilcoxon was
never computed, so this is a **descriptive** tie, not a statistical one; and
the writeup reports Arm B's median as ~104s where its own data says 119s.

**How this plan answers it.** By adopting the minimum that delivers the audit
and refusal properties, and nothing else. We take the event log, the
disagreement gate, fail-closed capability binding, human gates and `CaseBundle`
replay. We do not take the scheduler, saga machinery, multi-agent fan-out, the
learning loops, or a merge layer — none of which Quantify needs, and for which
the organisation's own data shows no benefit. The tie-breaks-to-simpler rule is
applied to this plan too.

### 8.2 The Context Runtime ladder bought efficiency, not accuracy

`redevops-rag` ships the only end-to-end accuracy data on Context Runtime's
retrieval ladder (`benchmarks/results/ladder_N15.txt`, 45 questions):

```
v1_base    naive dump                     acc 0.533   4014 tok/q
v2_sizer   + gating/sizer/abstain         acc 0.444   1582
v3_online  + online arm bandit            acc 0.444   1647
v4_route   + knowledge routing            acc 0.444   1628
v5_diver   + DIVER reasoning retrieval    acc 0.511   1686
```

Adding the machinery **cut accuracy** from 0.533 to 0.444–0.511 while cutting
tokens ~2.4×. On n=45 that is not significant, but it is the only data there
is, and it points the same way as §8.1: the elaborate mechanism buys cost, not
quality.

This is why §6.4's deferral is evidence-backed rather than a judgement call.
Discovery's inputs today are a sentence and a manifest; there is no corpus for
a retrieval ladder to help with, and the one measurement available says the
ladder would not improve answers if there were. When Discovery does need to
ground meaning in a strategy library or a user's history (§3.7), revisit it —
and re-measure rather than assume.

Also relevant: the other benchmark result in the organisation,
`DataOpsBench`'s S21, is won by an arm with `model_calls: 0` — deterministic
Python. It shows that keying a join correctly beats keying it wrongly, which is
a correctness property, not an AI result. Read it as support for the boundary
in §3.2a, not for any runtime.

---

## 9. Recommended immediate next step

Phase 1 — the capability manifest — against the existing compiler.

It is the smallest piece of work that is valuable whether or not the rest
happens; it closes the defect class that has produced every expensive error in
this project's history; and it produces the artefact **Mission** needs before
**Discovery** can safely understand more than the engine can run. It requires
no new dependency, no model, and no decision from §6.

The ordering matters and is deliberate. The manifest is what makes it safe for
Discovery to be ambitious. Build it first and a rich model reading is contained
by a named refusal; build it last and every dimension the model newly
understands is a dimension that can quietly become something else.

---

## Appendix — how this document changed

The first draft put the LLM call inside Mission Runtime, with an
"executability gate" between it and the engine. That collapsed two questions —
*what did the user mean* and *what can we execute* — into one component, which
is the precise shape of every defect this project has spent months removing. A
component that owns both can reconcile them silently.

The revision splits them: Discovery owns meaning and may understand more than
the engine can run; Mission owns executability and may only compile, refuse by
name, or ask permission. `runtime-contracts` already anticipated both runtimes
and explicitly permits Quantify to implement Discovery, and `mission-sdk`
already ships `MissionProgram.from_discovery(...)` with a test proving a
discovered mission gates identically to a hand-authored one — so the boundary
is a designed seam, not an invention of this document.

What survived unchanged: the capability manifest is still the recommended first
step. Its claim was narrowed — the pieces exist in `coverage.py`,
`CapabilitySpec`, `validate()` and the `unavailable` channel; what is new is a
semantic capability contract *derived from the executor and asserted against it
both ways* (§3.6).
