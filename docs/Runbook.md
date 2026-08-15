# Operator runbook

Deploying it, running it, and what it guarantees while it runs.

> Consolidated from `Runbook.md`, `Runbook.md`, `Runbook.md`, `Runbook.md`.
>
> The runbook told an operator what to do; the guarantees, the measured performance and the retention rules told them what they would get. Those are the same question asked at deploy time and at three in the morning.


---

## Operator runbook — Quantify closed pilot

**Status:** synthetic-data-only pilot · two supported journeys · RSU unavailable

This is written for someone who was not present when any of it was built. Every
procedure below has been executed against a real PostgreSQL instance and the
real production image; where a step has a test, it is named, because a
procedure nobody has run is a wish.

---

## The pilot boundary

> This pilot uses synthetic market data. It is for product evaluation and does
> not provide analysis based on licensed live market data.

Enforced at runtime by `PILOT_DATA_POLICY=SYNTHETIC_ONLY` and disclosed on
every private page and in every export. Enforcement and disclosure are separate
things and both are required: the synthetic series are deliberately shaped like
market data so the engine has something realistic to run on, which is exactly
why a user would otherwise take them for historical analysis.

The boundary lifts only when all six vendor licensing questions are resolved
and recorded. Until then a vendor policy resolves no snapshot and every run
refuses — the gate fails closed rather than falling back.

### Supported journeys

| journey | state |
|---|---|
| historical account and contribution replay | supported |
| Roth contribution analysis | supported |
| RSU vesting and diversification | **unavailable** — `/workspace/new` returns 501 |

RSU is recognised and refused, not silently misread. There is no
declaration-to-scenario compiler: vest events are not cash flows this build
understands. It is post-pilot feature one.

---

## Required configuration

Copy `deploy/production.env.template`. Every name in it is read by the code.

```
QUANTIFY_DEPLOYMENT_PROFILE=production
QUANTIFY_DATABASE_URL=postgresql://…
QUANTIFY_PARSER_MODE=MODEL_ASSISTED
QUANTIFY_PARSER_MODEL=claude-sonnet-5
QUANTIFY_PARSER_FALLBACK=REFUSE
ANTHROPIC_API_KEY=…
PILOT_DATA_POLICY=SYNTHETIC_ONLY
QUANTIFY_COMMIT / QUANTIFY_RELEASE_REF / QUANTIFY_IMAGE_DIGEST / QUANTIFY_SNAPSHOT_ID
```

Two names differ from what you may expect: the market-data policy is
`PILOT_DATA_POLICY` (not `QUANTIFY_MARKET_DATA_POLICY`) and the prompt pin is
`QUANTIFY_PARSER_PROMPT_VERSION` (not `QUANTIFY_PROMPT_VERSION`).

Secrets belong in the platform's secret store — AWS Secrets Manager, Docker
secrets, whatever the host provides. Never in the image, never in the compose
file, never in git.

---

## Startup refusals

**A refusal is the system working.** Each one names a condition under which
serving would be worse than not serving. `/health/ready` returns 503 and the
operator log carries the detail; the public response never does.

| Result | Cause | Fix |
|---|---|---|
| `REFUSED_CONFIGURATION` | no `QUANTIFY_PARSER_MODE` | declare it — see below |
| `REFUSED_CONFIGURATION` | `MODEL_ASSISTED` with no key or no pinned model | set `ANTHROPIC_API_KEY` and `QUANTIFY_PARSER_MODEL` |
| `REFUSED_CONFIGURATION` | `QUANTIFY_DATABASE_URL` unset, or not PostgreSQL | production has no local fallback by design |
| `BUILD_UNOBSERVABLE` | a build stamp missing | set all four from CI |
| `MIGRATION_MISMATCH` | database head ≠ code head | run migrations, or deploy the matching image |
| `SCHEMA_MISMATCH` | schema drift from the model | investigate before serving; do not "fix" by editing the database |
| `DATABASE_UNAVAILABLE` | unreachable | check network and credentials |
| `UNSUPPORTED_DATABASE` | PostgreSQL major ≠ 16 | 16 is what the lane has been proven against |

**Why production refuses an undeclared parser.** Defaulting to deterministic
would serve a narrower product than the one reviewed — fewer recognitions,
different blockers — while the startup proof reported a valid configuration.
That is configuration drift wearing a valid default. Outside production, unset
means deterministic, so a developer checkout cannot become model-assisted
because a key happens to be in the shell.

---

## Deployment sequence

1. **Provision PostgreSQL 16.** Managed is preferable; RDS if you are on AWS.
2. **Configure automated backups and retention** before any user data exists.
   Verify the schedule is real by listing snapshots, not by reading settings.
3. **Load secrets** into the platform's store.
4. **Run migrations** against the new database:
   `python -m alembic upgrade head` (or start the image with the migration
   entrypoint). The application refuses to serve a database at the wrong head,
   so this cannot be skipped silently.
5. **Start the image.**
   `docker compose -f deploy/docker-compose.production.yml --env-file .env up -d`
6. **Verify the startup proof.** `docker compose logs api | grep "deployment proof"`
   — expect `"result": "READY"`, the build identity, and the parser
   configuration. It carries no credentials and no hostname; that is asserted
   by `test_preflight.py::TestTheStartupProof`.
7. **Verify readiness.** `curl -sf https://HOST/health/ready` → `{"ready": true}`.
   `/health/live` is not a substitute: it answers whether the process exists,
   not whether it may serve.
8. **Run both launch journeys** through the browser. Describe a plan, answer
   the confirmation questions, save, reopen. Both must reach a worksheet with
   a figure.
9. **Verify the disclosure** appears on the confirmation page, the plan page,
   and in an export.
10. **Verify parser identity** on a saved plan: the plan page should name the
    model that interpreted it.
11. **Take and restore one backup** — the procedure below — before inviting
    anyone.
12. **Run the end-to-end fixture** before anything else:

    ```bash
    python -m pytest tests/test_the_original_prompt.py -q
    ```

    One sentence, and it exercises the parser, the clarification loop, the
    field vocabulary, the instrument registry, amendments, the compiler, the
    market snapshot, the moving-average indicator, the replay window,
    execution and persistence. It is not special; it is the widest single
    path through the system, which makes it the cheapest way to find out that
    one of those stopped agreeing with the others.

13. **Run the acceptance checks** against the public URL, not the container:

    ```bash
    python deploy/acceptance.py https://YOUR-HOST --record evidence/acceptance.json
    ```

    Exit 0 or do not invite users. Point it at the **public** address: run
    against `http://localhost:8000` it bypasses the proxy holding the
    credential, and the "private surface requires a credential" check fails
    correctly — that failure means the checklist was aimed wrong, not that the
    deployment is open. It reads only the public surface and writes nothing, so
    it is also the right thing to re-run after any configuration change.

    The last four checks are still yours, by hand and in a browser; the script
    prints them when it passes.

---

## Terraform: do not run a bare `apply`

`infra/terraform` declares eight variables with no defaults and there is no
`terraform.tfvars` in the repository. A plan run without the real values does
not fail — it substitutes whatever you pass and produces a plausible-looking
diff.

On 2026-08-07 the variables were reconstructed from state to change one
CloudWatch alarm. `cloudflare_account_id` was taken from the first resource in
state carrying an `account_id`, which was an **AWS** resource, and the plan
read:

    cloudflare_zero_trust_tunnel_cloudflared.pilot must be replaced
    aws_secretsmanager_secret_version.tunnel_token   must be replaced
    account_id "6b031ff3…" -> "388062344663"

Replacing the tunnel takes `quantify.club` offline. The summary line —
`Plan: 4 to add, 2 to change, 3 to destroy` — did not say so; only the
per-resource list did.

So, until the real values live somewhere reachable:

* **read the per-resource list, never the summary count**;
* apply with `-target` limited to what you actually intend to change, and
  confirm the reduced plan before proceeding;
* treat a Cloudflare or Secrets Manager resource appearing in a plan you did
  not intend to touch as a stop signal.

The eight are `alert_email`, `cloudflare_account_id`, `cloudflare_zone_id`,
`domain_name`, `application_image`, `build_commit`, `build_release_ref`,
`build_snapshot_id`. The last four are in `terraform output ansible_variables`
for the *currently deployed* build; the first four are not recoverable from
state without guessing, which is the mistake above.

## Backup and restore

Automated backups are the host's job. This is the **restore drill**, and it is
the part that is usually never run. Proven end to end by
`tests/test_backup_restore.py`; the commands below are what those tests do.

```bash
# 1. Dump the source.
docker exec SOURCE_PG pg_dump -U quantify -d quantify > backup.sql

# 2. A fresh instance. Never restore over the live one.
docker run -d --name quantify-pg-restore \
  -e POSTGRES_USER=quantify -e POSTGRES_PASSWORD=… \
  -e POSTGRES_DB=quantify -p 5434:5432 postgres:16

# 3. Restore, stopping on the first error.
docker exec -i quantify-pg-restore \
  psql -U quantify -d quantify -v ON_ERROR_STOP=1 < backup.sql

# 4. Point a fresh application process at it and let the preflight judge.
QUANTIFY_DATABASE_URL=postgresql://…:5434/quantify \
QUANTIFY_DEPLOYMENT_PROFILE=production … \
  python -c "from src.deploy.preflight import run; print(run().result)"
```

### What counts as a successful restore

Row counts do not. A restore that copies every row and drops a foreign key
passes a count and fails a user opening their plan. Check what a person would:

- the preflight reports `READY` against the restored database
- `applied_revision == code_head` — the migration head came across
- **the constraints came across** — deleting a cited `market_data_access_event`
  must still be refused
- a plan reopens with its `stated_text` unchanged
- its worksheet and run are present
- the **parser identity** survived (mode, model, prompt version)
- the **market-data provenance** survived and still identifies its data
- `verify_access_chain` returns no problems for every stored run
- **a new plan can be created against the restored database**, and the restored
  plan is untouched by that write

The last one matters most. Reopening proves the data survived; writing proves
the *system* did, and a missing constraint or stale sequence only shows on the
next write. A restore nobody has written to is a restore nobody has finished
testing.

**Expected recovery time.** The local drill above completes in about 15
seconds, and that number is about `psql`, not about recovery. On RDS,
`restore-db-instance-from-db-snapshot` to `available` measured **523 seconds**
on 2026-08-05 for a pilot-sized database, plus about a minute of verification.
Budget ten minutes.

**Name the artifact before you restore.** Write down a plan id, a run id and a
parser identity from production *first*, then restore, then assert those exact
values are present. "One plan came across" is not evidence; "plan-b1ffe… came
across, with MODEL_ASSISTED and its run" is.

The reason is a real failure, recorded in
`evidence/restore-drill-2026-08-05.md`. The first production drill used the
latest automated snapshot, which predated any saved plan. "Plans came across"
failed correctly, the run then wrote a plan to test writing, and re-running
against that same instance passed every data check by reading what it had just
written. The drill certified its own write.

The rule that prevents it, in general form:

> Every recovery verification must identify an artifact known to predate the
> recovery. Anything written during the verification is ineligible as
> evidence.

So: take a manual snapshot after the data exists, record the identifiers you
expect, restore once, and never re-run against an instance the drill has
already written to.

**Backup credentials** live in the same secret store as the database URL.
**Run the drill monthly**, and after any schema migration.

---

## Operations

### Database unavailable

`/health/ready` → 503, `DATABASE_UNAVAILABLE` in the log. The public error says
`The service is temporarily unable to reach its storage.` and nothing else — no
host, no SQLSTATE, no driver text. Check the database, then the network, then
credentials. The application recovers without a restart once the database
returns.

### Migration-head mismatch

Refuses for *any* profile, not only production: serving a database whose
columns this code does not expect moves the failure onto the first request that
touches one, which is how it reaches a user instead of an operator. Either run
the migrations or deploy the image matching the database.

### Parser provider unavailable

With `QUANTIFY_PARSER_FALLBACK=REFUSE`, requests needing interpretation return
503 with a retry message. This is deliberate: falling back silently would give
two users different products under one deployment. To ride out a long outage,
set `EXPLICIT_DETERMINISTIC` and restart — plans saved during that window
record that they were parsed deterministically, so the change is visible
afterwards rather than invisible.

### Missing model key

Production will not start. Restore the key; there is no degraded mode that
starts without one under `MODEL_ASSISTED`.

### Rollback to the previous image

Safe when the migration head is unchanged. If the newer image ran a migration,
**roll the database back first** from the pre-deploy backup, then the image — a
rolled-back application against a migrated database refuses with
`MIGRATION_MISMATCH`, which is the safe outcome but is not service.

### Export a user workspace

```python
from src.db.transfer import export_bundle
from src.workspace.store import WorkspaceStore
bundle = export_bundle(WorkspaceStore(URL), exported_at=STAMP, owner="pilot")
```

Carries the plans, runs, worksheets and delivery records, the parser identity
for each plan, and the synthetic-data notice in the manifest. Refuses to export
an inconsistent workspace rather than copying the inconsistency.

### Delete a user workspace

```python
from src.workspace.erasure import delete_workspace
receipt = delete_workspace(store, owner, requested_at=STAMP)
```

Deletes every classified owner-scoped table in dependency order and then
**verifies none remain**, reading the registry rather than the list the
deletion iterated. Raises `DeletionIncomplete` rather than reporting success
with rows surviving. The receipt reproduces no deleted content.

### A user reports an error with a correlation id

Every response carries `X-Request-ID`, and every response with a 4xx or 5xx
status leaves exactly one operator line under that id. Search for it:

```bash
docker compose logs api | grep req-abc123
```

For a database failure the line holds the SQLSTATE, the internal reason, the
operation and the original driver text — none of which was in the response. For
anything else it holds the status and the route template. The path is logged as
`/workspace/plans/{plan_id}`, never `/workspace/plans/my-divorce-settlement`:
the resolved path carries names users chose, and the log would otherwise be a
second place they have to be redacted from.

If the user quotes no id, ask for the status code and the time instead — a 200
is stamped too but not logged, deliberately, because a channel that records
every request records nothing.

This guarantee is wider than the handlers. The handlers cover failures the code
raises on purpose; a stale or mistyped link is routed to a 404 that reaches
none of them, and that is the error a pilot user is most likely to meet. It
carried no id until `deploy/acceptance.py` was run against a live instance and
disagreed with this page.

### Trace-store failure

Costs traces, never a request. `Recorder.failures` counts writes that did not
land; a request whose telemetry failed still returns the same result and stores
the same artifacts. If the trace volume is full or read-only, the application
keeps serving. Set `QUANTIFY_TRACE_PATH=` (empty) to disable recording
entirely.

Retention is not scheduled. Run it:

```bash
docker compose exec api python -m src.telemetry.purge          # or --dry-run
```

A cron entry calling that is the whole requirement; there is deliberately no
scheduling subsystem.

### Rotate secrets

Database credentials and the model key are read once at startup, into an
immutable deployment context. Rotation therefore requires a restart:
update the secret store, then `docker compose up -d --force-recreate api`.
The startup proof will show the new configuration. Nothing logs either value —
`DatabaseTarget.display` redacts the connection string and `ModelTarget`
exposes only whether a key is present.

---

## Deployment evidence

After the first acceptance run against the public URL, keep two files together.
They are the record of what was actually live when the cohort began, which is
the question nobody can answer retroactively.

```bash
mkdir -p evidence
python deploy/acceptance.py https://YOUR-HOST --record evidence/acceptance.json
docker compose logs api | grep "deployment proof" > evidence/startup-proof.txt
```

**It is deliberately two files.** The acceptance script sees only the public
surface, and the public surface does not carry the build identity, the
migration head or the snapshot id — it is checked four times over that it does
not, because those describe how to attack the deployment. Making one tidy
record would mean publishing the facts the checks exist to keep private. So the
public transcript and the private proof are captured separately and joined by
whoever ran them.

Re-record after any configuration change. A failed run is worth keeping too:
"we re-ran it until it passed" is part of the history, and evidence that only
exists when the answer was good is not evidence.

### Reading the transcript

The record is meant to be readable a year later without the script beside it:

| Field | Meaning |
|---|---|
| `target` | the URL that answered |
| `checked_at` | UTC, to the second |
| `outcome` | `PASSED`, `CHECKS_FAILED` or `UNREACHABLE` |
| `failure_category` | `null` when passed; `INVENTORY_DRIFTED` if the script's own inventory disagreed with what it ran |
| `checks` | every check attempted, with its detail |
| `not_attempted` | the checks that never ran, by name |
| `checks_declared` | how many exist in total |

`not_attempted` is the field that makes an abandoned run legible. An unreachable
deployment stops at the first check, and a record holding one failed check
describes itself as a one-check suite — which reads, later, as a thorough run
that found one problem. It should read as sixteen checks of which fifteen never
happened.

The script reconciles its declared inventory against what it actually ran, and
fails if they differ in either direction. Without that, a check added and not
declared would silently make every abandoned record understate what it
skipped — a wrong number in precisely the file being kept as proof.

---

## Before inviting anyone

The gaps below are pilot constraints, not open development. Each has an
operational mitigation, and the mitigation is the price of running without the
gate closed:

| Gap | Required before the first invitation |
|---|---|
| **Gate 8** — no rate limits or cost caps | Keep the cohort small and invitation-only. Set **provider-side budget alerts** before the first invite; a pilot user can otherwise drive model spend with no ceiling in this code. |
| **Gate 10** — no egress allowlist | Issue **tightly scoped credentials** — the model key should do nothing but call the model — and monitor outbound requests at the host or network layer. |
| ~~**Trace retention** — not scheduled~~ **Closed 2026-08-05** | The Ansible role installs the cron unconditionally (`quantify-trace-purge`, 03:17 daily) and the host confirms it: it ran at 03:17 on 2026-08-05 and logged `purged before 2026-05-07T03:17:05+00:00: 0 decisions, 0 spans, 0 traces`. This row said "not scheduled" for longer than it was true. That run proved only that the command executes, because the store was empty — a purge that silently deleted nothing would have logged the identical line. **Deletion was exercised separately on 2026-08-05**: a chain stamped `2020-01-01` was seeded alongside the live one and the unmodified operator command was run, giving `purged before 2026-05-07T21:22:16+00:00: 1 decisions, 1 spans, 1 traces` — expired chain gone, unexpired chain kept, nothing else removed, `foreign_key_check` clean. Both halves matter: deleting everything would satisfy a test that only counted removals. |
| **Licensing** — six questions unresolved | Stay `PILOT_DATA_POLICY=SYNTHETIC_ONLY`. The gate fails closed, so this is enforced rather than remembered; the disclosure is the part that depends on nobody quietly removing it. |

---

## What is not modelled

Stated as a boundary rather than a gap, because the failure is silent.

**Historical security purchases, tax lots, realized gains, tax-loss
harvesting, direct indexing, and equity-compensation lots are not modelled.**
The two supported journeys describe contributions going forward; neither
accepts a holding acquired in the past, and no surface claims a tax-lot
result.

That boundary is load-bearing. A share count recorded before a corporate
action is recorded in units that no longer exist: ten NVDA bought at $1,209 in
May 2024 are a hundred shares today, and valuing the stored ten against a
split-adjusted price gives $1,210 for a position worth $12,100. Off by the
split ratio, and entirely ordinary on the page.

`src/holdings/` now carries the ledger that answers it — immutable acquisition
lots, corporate-action resolution, disposition allocation, and three price
series that refuse each other's questions by name. It is not wired into a user
surface, and until it is, **the pilot must not accept an existing position.**
The moment it does, a share-count-only model is materially unsafe rather than
merely incomplete.

---

## What this pilot does not have

Stated so nobody discovers it during an incident:

- **No user accounts.** A single `pilot` owner; access control is the reverse
  proxy's basic auth. Tenant isolation is enforced in the schema and tested,
  but the pilot does not yet issue separate identities.
- **No rate limits or cost caps** (Gate 8, outstanding). A pilot user could
  drive model spend.
- **No egress allowlist** (Gate 10, outstanding).
- **No scheduled trace purge** — manual, as above.
- **No orphan cleanup** for delivery records left by an interrupted save. They
  are inert.
- **The container lane reuses a fixed container name**; a failed run can leave
  one behind and make the next run fail for an unrelated reason.

---

## Withdrawing a run whose declared rule was never executed

    python -m src.workspace.invalidate --dry-run
    python -m src.workspace.invalidate

Reads every stored run for the tenant and withdraws the ones whose plan
declares an `event_program` and whose result records no executed rule events.
The inventory is derived from the artifacts, never from a list of plan ids read
off a page: the affected plan was found by one user opening one page, and there
is no way to know from a page how many others carried the same defect.

**Re-running is a no-op, not a rewrite.** `run_invalidation` is classified
`IMMUTABLE_ARTIFACT`, and `invalidate_run` returns `False` rather than replacing
a row that already exists. A second sweep must not move `invalidated_at` to
today — that would erase when users were first told — and must not let a reason
be softened on a later pass. The command reports how many it wrote and how many
were already withdrawn.

**The run itself is kept.** Deleting it would destroy the evidence that the
defect happened and what was shown to whom. A user who remembers a figure must
be able to find the record of having been shown it.

**Forward compatible.** Once the engine executes event programs and records
`rule_events`, a correct run stops matching without this command being edited.
An absent `rule_events` reads as unknown rather than zero: a run recorded
before the field existed is affected by what its plan declared, not by a count
it never had the vocabulary for.


---

## PostgreSQL deployment guarantees

PostgreSQL is not another supported dialect. It is the only environment in which
several of Quantify's production claims have been established, and for a
specific reason: some of these properties do not merely go *unproven* on SQLite,
they are actively **misrepresented** by it. Three defects in this list were
invisible for exactly as long as the tests ran on SQLite alone.

> A green SQLite lane is not evidence for any guarantee whose meaning depends on
> PostgreSQL storage, locking, constraints, or dialect adaptation.

SQLite remains the right tool for fast deterministic tests, local development
and the standalone demo. It is not a smaller PostgreSQL.

---

## Proven only on PostgreSQL

### Concurrency and transactions

- Row-level proposal locking (`SELECT ... FOR UPDATE`), asserted by taking the
  row from a second session with `NOWAIT` and requiring the failure.
- Exactly one winner under racing acceptance, with the loser receiving a typed
  refusal rather than a raw driver error.
- Stale-revision checks performed *after* the lock and re-read inside the
  transaction, because a pre-lock check describes state another session is free
  to change.
- Atomic commit of candidate runs, the worksheet revision and the proposal
  status, with rollback proven at every durable boundary.
- Contention translated by SQLSTATE — `40001`, `40P01`, `23505`, `55P03` — not
  by driver exception class.

The isolation level is **`READ COMMITTED`**, PostgreSQL's default, unchanged.
Nothing required `SERIALIZABLE`. The property that matters is not an isolation
label but a protocol: *every decision is made after locking and re-reading the
state that authorizes it.*

### Schema enforcement

- Alembic migration head, and a startup refusal when the database is at a
  different schema than the code expects.
- Foreign-key enforcement, including `RESTRICT` refusing a parent-first delete.
- Composite tenant keys.
- `CHECK` and uniqueness constraints.
- Model/migration parity covering foreign keys, unique constraints, check
  constraints and indexes — not only tables, columns, types and primary keys.
- Deletion order enforced by the database, so the application's ordering is a
  guarantee rather than a convention.

### Tenant identity

- `owner` in every production identity.
- `owner` in every tenant-owned foreign key.
- `owner` in every upsert conflict target.
- Composite identities preserved through joins and consumers.
- Identical ids coexisting across tenants — plan, run, worksheet, proposal,
  observation, reconciliation and revision number.
- Cross-tenant existence never disclosed: an absent record and another tenant's
  record are indistinguishable from outside.

### Immutable artifacts

- Redelivery accepted only for an identical body.
- Divergent redelivery refused.
- Immutable payload columns never updated.
- The **adapted** conflict action captured and classified.
- No in-place JSONB mutation (`jsonb_set`, `||`, `-`, `#-`) on an artifact body.

### Representation-sensitive verification

- `NUMERIC(38, 12)` compared semantically after read, because the column pads to
  its declared scale.
- Canonical spelling checked *before* persistence, where both sides are in hand.
- Exact database aggregation (`SUM` over `NUMERIC`).
- Payload, mirror column and derived state each verified independently, with
  each test requiring the other two verifiers to pass.

### Dialect-translation semantics

- `INSERT OR REPLACE` is never trusted by how it reads in the source.
- The `ON CONFLICT` form PostgreSQL receives is captured and classified.
- Conflict targets and assigned columns are checked after translation.

---

## What SQLite proves

- Deterministic compilation and domain behaviour.
- Financial arithmetic.
- Canonical serialization.
- Worksheet and result semantics.
- Local development compatibility.
- Most non-concurrent store behaviour.

## What SQLite does not prove

- Row locking.
- Concurrent writer correctness.
- PostgreSQL upsert semantics.
- JSONB behaviour.
- Production constraint enforcement.
- Representation-sensitive read verification.
- Composite-key consumer behaviour under the deployed dialect.
- Alembic migration parity.

---

## The defects that earned this boundary

Each of these was found by running against PostgreSQL, and each had been present
while the SQLite suite was green.

**Foreign keys were never enforced.** SQLite ignores them unless
`PRAGMA foreign_keys = ON`, which nothing set. The shipped schema declared
`plan_run.plan_id REFERENCES plan(plan_id)`, and on the engine every test ran
against, that constraint asserted something about the data that was not true.

**`NUMERIC` padding hid a broken verifier.** `verify_decimal_columns` compared
canonical spellings. SQLite stores canonical text verbatim, so it passed;
PostgreSQL pads to scale, so `152.260000000000` against a payload holding
`"152.26"` was reported as drift on *every clean row*. The verifier was failing
open on the engine that matters, and its only tests were on the engine where the
comparison happened to work.

**`INSERT OR REPLACE` concealed an overwrite.** It reads as idempotent. It is
translated to `ON CONFLICT ... DO UPDATE SET`, which is close to the opposite for
an immutable artifact. `save_plan` replaced a pinned scenario and parse;
`record_run` replaced the verdict a saved worksheet cites. Both had docstrings
promising immutability.

**Racing acceptance could not be evidenced at all.** SQLite admits one writer, so
two sessions both accepting one proposal — which happened, returning two
`ACCEPTED` results — is not reproducible there.

**A correct migration invalidated a correct consumer.** Widening every tenant key
was right, and it silently broke `OwnershipPath`, which still joined on the
scalar identity those keys used to have. The join stayed valid, kept returning
rows, and returned another tenant's.

---

---

## The production preflight

A production instance establishes all of this before it serves anything:

```text
build identity -> URL -> connect -> version -> migration head -> schema parity
```

Each step has its own outcome — `REFUSED_CONFIGURATION`,
`DATABASE_UNAVAILABLE`, `UNSUPPORTED_DATABASE`, `MIGRATION_MISMATCH`,
`SCHEMA_MISMATCH`, `BUILD_UNOBSERVABLE` — because collapsing them into one
"startup failed" would discard the only thing an operator needs. The public
surface still reports nothing but `ready: false`.

    QUANTIFY_DEPLOYMENT_PROFILE=production
    QUANTIFY_DATABASE_URL=postgresql://...

**The profile decides whether a refusal stops the service, not whether the
question is asked.** The database checks run under any profile pointed at
PostgreSQL — a developer with an unmigrated database wants to know — and a
schema mismatch stops every profile, because serving a database whose columns
this code does not expect moves the failure onto the first request that touches
one.

**The URL is judged before anything opens it.** `Database` creates the parent
directory of a SQLite path on construction, so a check made after building one
would already have written to disk.

**There is no production fallback.** Absent configuration, the target resolves
to `data/workspace.db` — correct for a checkout, and in production the same
shape as the `_prices()` bypass: a live path quietly reading something nobody
authorised.

`PROVEN_POSTGRES_MAJOR` is stated as *proven*, not as a ceiling. A later major
is unsupported until this same lane passes against it, which is a day's work
rather than a re-architecture.

Readiness and liveness are separate endpoints. A failed preflight makes an
instance unready — visible to a load balancer, still diagnosable — rather than
crash-looping a container nobody can inspect. No user-facing route is served in
that state.

The startup proof record carries the profile, engine, version, migration head,
parity result, build provenance and timestamp, and no credentials or network
detail.

---

## Running the PostgreSQL lane

```bash
docker run -d --name quantify-pg \
  -e POSTGRES_USER=quantify -e POSTGRES_PASSWORD=quantify_dev \
  -e POSTGRES_DB=quantify -p 5433:5432 postgres:16

export QUANTIFY_TEST_POSTGRES_URL="postgresql://quantify:quantify_dev@localhost:5433/quantify"
python -m pytest tests/ -q
```

Without `QUANTIFY_TEST_POSTGRES_URL` the PostgreSQL-gated files **skip**. They do
not fail, and they do not silently pass — a skipped guarantee is reported as
skipped. Roughly 140 tests are gated this way; a run reporting no skips and no
PostgreSQL is a run that proved none of the above.


---

## Performance — measured

**Date:** 2026-08-01 · **Hardware:** 32 cores · **Polars:** 1.43.2

Every number here is measured on this machine, not budgeted. The load-test plan
set engineering budgets to validate; this replaces them with observations.
Where a budget and a measurement disagree, the measurement is what shipped.

---

## 1. The compiler is effectively free

14,400 descriptions — 144 catalog strategies × 100 paraphrase variants — through
the full deterministic pipeline.

| Stage | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| Stage 1 parse | 34 µs | 55 µs | 65 µs | 1.2 ms |
| Stages 2–8 compile | 9 µs | 11 µs | 12 µs | 0.4 ms |
| **Total per description** | **45 µs** | **66 µs** | **80 µs** | 1.3 ms |

The plan's budget for the deterministic acceptance chain was **150 ms p95**.
Measured: **0.066 ms**. Three orders of magnitude inside it.

That reshapes where optimization belongs. The architecture is

```
language model      expensive, one call, quarantined to stage 1
      ↓
deterministic       ~45 µs, and it is the part that decides anything
compiler
      ↓
runtime
```

"AI compiler" reads as expensive. Only the first step is. Everything that
determines what actually gets simulated costs less than a network round trip's
jitter, which is why it can afford to be exhaustive rather than approximate.

---

## 2. HarnessBench — canonical versus Polars

Four analytical workloads, three backends, p50 milliseconds. Results are
compared before any timing is reported: **all backends agreed on every workload
at every scale.** Polars is an execution backend, never a second owner of
semantics.

Read from a Parquet projection, which is what the architecture specifies.

| Workload | Scale | canonical | Polars eager | Polars lazy | best speedup |
|---|---:|---:|---:|---:|---:|
| latency_summary | 1K | **0.31** | 2.20 | 2.38 | 0.14× |
| | 10K | **1.87** | 3.71 | 2.08 | 0.90× |
| | 100K | 25.10 | 17.71 | **4.24** | 5.9× |
| | 1M | 315.70 | 71.57 | **26.93** | 11.7× |
| mission_replay | 1K | **2.45** | 3.85 | 5.10 | 0.64× |
| | 10K | 5.95 | 7.30 | **5.31** | 1.1× |
| | 100K | 54.39 | 22.66 | **9.28** | 5.9× |
| | 1M | 691.21 | 109.02 | **28.62** | 24.2× |
| denial_scan | 1K | **0.81** | 3.72 | 4.07 | 0.22× |
| | 10K | **4.99** | 9.93 | 7.76 | 0.64× |
| | 100K | 41.80 | 27.76 | **16.56** | 2.5× |
| | 1M | 385.63 | 63.77 | **20.04** | 19.2× |
| secret_then_egress | 1K | **0.13** | 2.06 | 3.40 | 0.06× |
| | 10K | **1.28** | 2.34 | 3.22 | 0.55× |
| | 100K | 24.16 | **4.31** | 6.62 | 5.6× |
| | 1M | 284.49 | **11.02** | 12.43 | 25.8× |

### Crossover, per workload

Measured and stored per workload, never one global constant:

```
mission_replay        10,000 events
latency_summary      100,000 events
denial_scan          100,000 events
secret_then_egress   100,000 events
```

`mission_replay` crosses an order of magnitude earlier than the rest. A single
threshold would be wrong for both sides of that gap — which is the argument for
measuring it rather than picking one.

Below the crossover Polars is *slower*, by up to 16×. Constant setup cost
dominates small queries, so routing a 1,000-event replay through Polars would
make an interactive path measurably worse.

---

## 3. The finding that mattered most

**How the data reaches Polars decides whether Polars helps at all.**

The same four workloads at 100,000 events, fed from Python dicts instead of a
Parquet projection:

| Workload | canonical | Polars eager | Polars lazy |
|---|---:|---:|---:|
| latency_summary | **26.7** | 248.8 | 246.3 |
| mission_replay | **55.0** | 247.3 | 241.5 |
| denial_scan | **41.9** | 251.4 | 248.5 |
| secret_then_egress | **23.0** | 238.2 | 239.6 |

Polars loses every workload, by 4–10×, and the ranking inverts completely.

Constructing a DataFrame from a list of dicts costs ~240 ms at this scale and
swamps every query. Timing that path and calling it "Polars" measures the
adapter, not the engine — and would have produced the confident and entirely
wrong conclusion that Polars is not worth adopting.

The plan already specified partitioned Parquet projections written once by the
event buffer and scanned many times. This is the measurement that says why that
detail is load-bearing rather than incidental.

---

## 4. What this means for the roadmap

Polars belongs **nowhere near** the interactive path:

```
parser · compiler · runtime validation · authorization · ledger writes
    already microsecond-scale; Polars would make them slower
```

It belongs where the volumes are:

```
replay · aggregation · trajectory mining · Discovery sweeps
validation scans · fleet monitoring · Mission evolution
    5-25x at a million events, and growing with scale
```

---

## 5. Semantic stability

Different from recognition accuracy, and a stronger property. Accuracy asks
whether the compiler understood; stability asks whether it understood **the same
thing every time**. A parser can be perfectly accurate on a benchmark and still
be unusable if a synonym changes the answer.

    one meaning -> 40 wordings -> compile each -> one rule_hash

41 families x 40 wordings = 1,640 descriptions, varying the verb, the ordering
of holdings, the cadence phrase, the day-rule phrase, the dividend phrase, the
no-selling phrase and the account phrase.

**Stability: 100%.** Every wording of a plan compiles to the same market rule.

This is the harness a language model in stage 1 must be measured against. The
interesting claim is not that a model reads more wordings — it is that different
models, or the same model on different days, still land on one canonical
Mission.

### What it found on its first run

A choice that was **recognised, confirmed to the user, and never represented**.

The compiler read "hold the dividends as cash", the confirmation screen quoted
it back under "you stated", and the compiled scenario contained no trace of it.
Reinvesting and holding as cash produced an identical `content_hash` — two
materially different strategies sharing one identity, the same shape as the
earlier defect where a Roth and a taxable account compared as identical.

Fixed in two parts, because one without the other moves the defect rather than
closing it:

- `dividend_policy` now reaches `HoldingsPolicy` and therefore the rule hash, so
  the two strategies are distinguishable;
- the engine runs on price series only and cannot honour it, so every result
  declares it under `declared_but_not_simulated`. Representing a choice without
  saying it is not simulated would leave the scenario looking enforced while the
  figure ignored it.

---

## 6. Round-trip fidelity

    stability      many texts   -> one Mission
    round-trip     one Mission  -> text -> the same Mission

Different directions, different failures. Stability catches a compiler that
reads wording as meaning. Round-trip catches one that cannot *say* what it
understood — a field that survives compilation with no way back into language is
a field no user can ever correct.

Three declared purposes, and only one claims losslessness:

| Purpose | Exact | Rule identity | Claims lossless |
|---|---:|---:|---|
| SPECIFICATION | **100.0%** | 1728/1728 | yes |
| SUMMARY | 7.8% | 428/1728 | no — reports what it drops |
| EXPLANATION | n/a | n/a | no — disclaims itself |

Identity drift over three cycles: **0/400**.

The declaration is the point. A concise summary that looks like prose is exactly
what someone will paste back in expecting identical behaviour, so it names the
fields it omitted rather than being held to a standard it never claimed.

### What it found

**Values can round-trip while provenance does not.** The first renderer wrote out
every field, including inferred ones — reproducing the values and destroying the
record of who chose them. An inference restated is a decision the user never
made, and the confirmation screen then asks them to confirm nothing.

**A specification must be able to express an open question.** A description that
mentions a market condition without saying how it behaves leaves
`trigger_semantics` unresolved. Dropping the mention made the regenerated text
stop asking — the open question answered by omission, which is the one outcome a
specification must never produce.

Four more losses followed the same shape, each a stated value with no clause to
live in: a weighting on a single holding, an estimator, a funding source with no
trigger, and a funding source with an unresolved one. All four changed what the
plan would simulate.

---

## 7. Reproducing

```bash
python3 scripts/run_load_corpus.py --per-strategy 100     # compiler corpus
python3 scripts/run_harnessbench.py                       # Polars crossover
python3 scripts/run_harnessbench.py --from-dicts          # the adapter cost
python3 scripts/compiler_dashboard.py                     # quality metrics
python3 scripts/run_stability.py                          # semantic stability
python3 scripts/run_roundtrip.py                          # round-trip fidelity
python3 scripts/run_evolutionbench.py                     # compiler evolution
```

All four run on committed synthetic data. No credentials, no network.


---

## Privacy and retention

**Version:** `retention/workspace@1` · **Date:** 2026-08-02 · **Status:** closed
pilot v1

Three data systems with different policies, kept apart because one policy over
all three would be wrong for at least two of them.

| System | Contains | Governed by | Lifetime |
|---|---|---|---|
| Workspace store | the user's research record | user ownership | while the account is active |
| Trace store | operational telemetry | retention schedule | 90 days, configurable |
| Market data | prices and snapshots | the vendor agreement | per agreement |

---

## 1. What "deletion" means here

Two things are routinely called deletion and only one of them is:

```
artifact no longer visible      a query predicate changed
artifact removed                the rows are gone and something checked
```

`delete_workspace` does the second. It enumerates every classified table,
removes the rows, and then **re-reads the schema to verify none survive**. A
deletion that silently missed a table looks exactly like one that worked, so the
verification reads the classified inventory rather than the deletion code — the
two cannot agree by construction.

Backups are the exception, and it is stated rather than glossed: deletion
propagates through backup expiry, not immediate physical removal. Until a backup
cycles, a copy exists.

---

## 2. Workspace store

The durable user record: scenarios, confirmed declarations, account and tax
information, plans, runs, worksheet revisions, intents, proposals, vest
schedules, observations, reconciliations and evidence references.

**Retention** — while the account is active.
**Deletion** — removes the user's records and derived artifacts.
**Export** — `export_workspace(owner)` returns every owner-scoped table, so a
user can see what they hold before deciding to lose it.

### Ownership is not always direct

`plan_run` carries no `owner` column and is reachable only through its plan:

```
plan_run.plan_id -> plan.plan_id -> plan.owner
```

A deletion written around `WHERE owner = ?` removes nothing from it and reports
success. Every classification states its scope and, where indirect, the join
that finds it.

### Sensitive content

None of this is a name or a national identifier and all of it is sensitive:

- employer name and ticker
- compensation and vest schedule
- contribution amounts
- account type
- holdings
- tax assumptions
- raw user instructions
- evidence references
- model prompts and responses

### What a user's deletion does **not** remove

Public methodologies, public runtime declarations, synthetic datasets, and
findings promoted through the public boundary all survive. A personal
*reference* to one does not.

### The receipt

Written outside the deleted scope, holding none of what it deleted: a request
id, an irreversible owner reference, a timestamp, per-table counts, the policy
version and the status. A receipt reproducing the personal data it certifies as
gone would be the one surviving copy.

---

## 3. Trace store

Operational and expendable. Execution never depends on it — proven by deleting
the database mid-flight and requiring every financial path to continue.

| Item | Default |
|---|---|
| structured spans | 90 days |
| decision records | 90 days |
| raw prompt and completion content | **not stored** |
| redacted previews | off |
| aggregate metrics | may outlive traces only if de-identified |

Spans hold structured fields and hashes: an instruction's digest, never its
text. An error records its exception class rather than its message, because a
message can quote the input that caused it.

`purge_tenant` erases one tenant immediately — a deletion request is not a
retention policy and must not wait for one to come round.

A workspace record keeps only `trace_id` after expiry. It is a reference that
may dangle by design, not a foreign key.

---

## 4. Market data

Governed by licence, not by user ownership.

| Layer | Policy |
|---|---|
| snapshot objects | per the vendor agreement and audit needs |
| local cache | bounded by snapshot retirement |
| derived personal results | deleted with the user |
| exports | governed by the snapshot egress policy |

**A personal result is not retained merely because its source data is shared.**
Deleting a user removes their results and scenarios; the shared synthetic
snapshot stays.

Closed pilot v1 is `SYNTHETIC_ONLY`, enforced at the live read path rather than
in deployment guidance. See `src/market_data/pilot_policy.py`.

---

## 5. What is checked mechanically

| Claim | Where |
|---|---|
| every table is classified | `test_retention.py` — reads `sqlite_master`, not the registry |
| deletion reaches indirectly scoped rows | `test_it_reaches_the_indirectly_scoped_runs` |
| verification is independent of deletion | `test_verification_is_independent_of_the_deletion` |
| an incomplete deletion raises | `test_a_deletion_that_removes_nothing_raises` |
| one tenant's deletion spares another | `test_deleting_one_owner_leaves_the_other_intact` |
| the receipt holds no personal content | `test_it_names_no_owner_and_no_content` |
| traces can vanish without affecting artifacts | `test_workspace_export_survives_the_trace_store_being_deleted` |

The inventory test reads the tables SQLite reports rather than the
classification registry. Parametrised from the registry, a new table would pass
by never appearing.

---

## 6. Known gaps

Stated rather than omitted:

- **Backups** are not yet implemented, so deletion is currently immediate and
  complete by default. When backups exist, the propagation delay above becomes
  real and must be disclosed to users.
- **Account closure grace period** is not implemented. Deletion is immediate on
  request.
- **Legal-hold exceptions** are not implemented. There is no mechanism to retain
  a record against a deletion request, and none is claimed.
- **The trace store's expiry is manual.** `purge_before` exists and nothing
  schedules it.
