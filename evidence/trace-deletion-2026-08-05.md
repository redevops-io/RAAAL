# Deletion of one trace chain — 2026-08-05

**Reason:** privacy guarantee violation in a pre-pilot canary.

**Affected rows:** 1 trace, 1 span, 1 decision.

**Commit introducing the fix:** `b0f529e`.

**Raw content retained:** no.

## What was removed

| | |
|---|---|
| `trace_id` | `trace-a6695d0c2523431d` |
| `span_id` | `span-f5f49a9c12864de1` |
| `decision_id` | `dec-e04ab461d0a84b30` |
| `request_id` | `req-509de92833fe48ec` |
| `conversation_id` | `conv-80efa4b0b21b4317` |
| `created_at` | `2026-08-05T21:05:49+00:00` |
| tenant | `pilot` |

## Why

The chain was written by the first canary request against the newly-wired Plan
Builder recorder, running image `e3e7208`. It proved the instrumentation
reachable and, in the same request, wrote two `unclear:` field ids into the
trace store. Those ids are built as `unclear:{phrase}`, where the phrase is the
user's own words with a model-written reason appended — so the store documented
to hold no raw instruction text held some, from its first three rows.

`b0f529e` hashes any identifier not drawn from `mission.vocabulary.FIELDS`.
The rule it enforces:

> Only identifiers drawn from a closed, reviewed vocabulary may enter
> telemetry. Everything else is represented by a digest and a typed category.

## How

By exact id, in dependency order — decision, span, trace — inside one
transaction, with each statement required to change exactly one row and the
transaction rolled back otherwise. Not `DELETE WHERE trace_id = ...`, which
would be a wipe wearing a filter, and not a removal of `trace.db`.

The identifiers were read with a query that selects no column able to hold a
phrase. Naming what had to be deleted should not reproduce the content that
made it deletable.

## Verified after

```
BEFORE {'trace': 1, 'span': 1, 'decision': 1}
AFTER  {'trace': 0, 'span': 0, 'decision': 0}

ABSENT trace.trace-a6695d0c2523431d: True
ABSENT span.span-f5f49a9c12864de1:   True
ABSENT decision.dec-e04ab461d0a84b30: True

DELTA_IS_ONE_EACH True
FOREIGN_KEY_CHECK clean
INTEGRITY_CHECK   ok
SCHEMA_TABLES     ['decision', 'span', 'trace']
```

No unrelated rows existed to disturb, which is stated here rather than claimed
as a stronger result than it is: the store held exactly this chain. The
per-statement row-count guard is what would catch over-deletion once it does
not.
