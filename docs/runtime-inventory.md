# Repository and deployment inventory

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
