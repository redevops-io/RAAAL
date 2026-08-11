# Operator runbook — Quantify closed pilot

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
