# Privacy and retention

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
