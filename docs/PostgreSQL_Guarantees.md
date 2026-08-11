# PostgreSQL deployment guarantees

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
