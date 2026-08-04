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
12. **Run the acceptance checks** against the public URL, not the container:

    ```bash
    python deploy/acceptance.py https://YOUR-HOST
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

**Expected recovery time:** minutes for a pilot-sized database — the drill
above completes in about 15 seconds against a workspace with a handful of
plans. Budget for provisioning the replacement instance, not for the restore.

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
