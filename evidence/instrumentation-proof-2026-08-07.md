# Provider-call instrumentation, proven before use — 2026-08-07

Against `https://quantify.club` serving `540c4dc`. Every count below was taken
from CloudWatch over an explicit time window.

The point of this document is not the counts. It is that the counts can be
wrong in the direction that flatters them, and were.

## What was wrong

`parse_model` logs one line per provider call, and its comment says the line
exists so calls "can be measured against a live server rather than a fixture".
It was never emitted by one. Uvicorn configures its own handlers and leaves
the root logger alone, so every module using `getLogger(__name__)` wrote into
a logger with nowhere to send anything.

The failure was silent and flattering. Three reopens of a saved plan reported
**zero** provider calls, which is the correct answer. The claim would have gone
into the record as evidence. It was only caught because the producer was
checked before the count was believed — a fresh draft that certainly called
the provider also reported zero.

A measurement that returns the expected number for every input is not a
measurement.

## The premise, established first

| Check | Result |
|---|---|
| self-test line at startup, from `src.mission.parse_model` | present, twice (two container starts) |
| a fresh draft — must produce a call | **1** |

Only after the second row may a zero mean anything.

## The counts

| Action | Provider calls |
|---|---|
| fresh draft (new description) | **1** |
| three reopens of a saved plan | **0** |
| full journey: draft + 2 clarification rounds → saved | **1** |
| two submissions carrying a tampered pinned parse | **0** (both refused, HTTP 422) |

A whole journey costs one call. Clarification rounds replay the pinned parse
and do not re-read the user's words; reopening a saved plan reads nothing at
all.

The tampered-parse rows are the pin being *verified* rather than trusted: a
parse belonging to a different description, and a syntactically broken one,
are both refused rather than silently re-parsed, and neither costs a call.

## What is not established here

The `stage1 fallback to deterministic` line cannot be exercised against this
deployment. Production runs `parser_fallback=REFUSE`, so a provider failure
refuses rather than falling back, and inducing one on a live deployment to
observe a log line is not a trade worth making. It is covered at test level
only, and this document does not claim otherwise.

## Startup proof

```
result           READY
profile          production
commit           540c4dc
image_digest     sha256:dd3ccc8abcddb9c2a6b9819da781cf0d1f63c0932fcafdd17b5937012ee3e177
migration_head   a91c4e7b2f05   (unchanged; no migration in this deploy)
schema_parity    PASS
database         postgresql 16.13
snapshot         syn-2026-08
```

Acceptance: 16 of 16 checks passed, recorded to `evidence/acceptance.json`.
Four checks remain that need a browser and a person, listed by the tool itself.

## Stored plans

Seven plans existed before this deploy; all seven are unchanged in id, title
and save time. Nothing was withdrawn, migrated or overwritten.

Three plans in the pilot workspace are verification artifacts rather than user
data — two `canary control` and one `pinned-replay probe`, created by the
post-deploy checks on this and the previous deploy. They are named here
because the workspace has one owner and no per-plan deletion, so they will
appear in the pilot's list until the workspace is erased as a whole. That
limitation is also why the playbook now skips its own deploy-time journeys
against a non-empty workspace.
