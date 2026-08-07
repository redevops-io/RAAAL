# Recovery matrix — seven production plans, 2026-08-07

Produced by `src/workspace/recovery.py` against the live `quantify-test` RDS
instance, read through the running `quantify-api-1` container over SSM.
Read-only: one `SELECT` per plan, nothing written, the serving container
unmodified. Raw output in `recovery-matrix-2026-08-07.json`.

Supersedes `recovery-matrix-2026-08-06.md`, which was produced before
`CONFIRMATION_DISCARDED_PROVENANCE` was known and therefore reported the four
`provenance@2` plans as structurally complete. They are not.

## Three independent conditions

A plan may carry any combination. They are separate defects with separate
consequences, and collapsing them loses the distinction that decides what
recovery is possible.

| Condition | What happened | What it costs |
|---|---|---|
| `SAVE_RECOMPILE_DIVERGED` | the preview and the save compiled the same description under different inputs | the stored plan is not the plan the owner approved |
| `FUNDING_POLICY_NOT_PERSISTED` | a derived execution field disappeared on the way to storage | the stored plan cannot execute |
| `CONFIRMATION_DISCARDED_PROVENANCE` | `_with_decisions` rebuilt the provenance from five of eight names | owner decisions were deleted |

The third is the serious one. The first two lost a derivation, which the
description and the pinned parse can rebuild. The third lost **authority** —
which fund the owner chose, which limitations they accepted, which period they
confirmed — and no amount of recompiling reconstructs consent.

## The seven plans

| Plan | Saved | Shape | Conditions | Needs owner | Stored vs fresh |
|---|---|---|---|---|---|
| `SPX 200DMA` | 08-05 14:12 | `@1` | funding, confirmation | `funding` `time_window` `amendments` `exclusions` `asset_resolutions` | `held_assets` `tax_treatment` |
| `agent ORIG-2` | 08-06 19:25 | `@2` | funding, confirmation | `time_window` `exclusions` `asset_resolutions` | `flows` `inferred` |
| `agent ORIG-5` | 08-06 19:27 | `@2` | funding, confirmation | same three | `flows` `inferred` |
| `agent ORIG-6` | 08-06 19:28 | `@2` | funding, confirmation | same three | `flows` `inferred` |
| `agent ORIG-8` | 08-06 19:29 | `@2` | funding, confirmation | same three | `flows` `inferred` |
| `canary` (`8f43680`) | 08-06 23:54 | `@2` | confirmation | same three | — |
| `canary` (`968fa8e`) | 08-07 00:40 | `@2` | **none** | **none** | — |

`SAVE_RECOMPILE_DIVERGED` applies to the first six by construction: every one
was saved by a build whose save path compiled without a priceable set. It is
not derivable from a stored body — the divergence was between two compiles,
and only one of them was ever written down — so it is recorded from the build
identity rather than inferred from the artifact.

The last row is the point of the exercise. A plan saved by `968fa8e` carries
its funding policy, its time window, its asset resolutions and its exclusions,
and needs nothing from anyone.

## Absence is not emptiness, and now has three possible readings

For `SPX 200DMA`, three fields are **ambiguous**: `time_window`, `exclusions`
and `asset_resolutions` are each explained by *both* `THE_PLAN_PREDATES_THE_
FIELD` and `DISCARDED_WHEN_AN_INFERENCE_WAS_CONFIRMED`. Two readings means the
absence carries no information at all — a stronger statement than "the field
is missing", and a different action.

Reading such an absence as an empty value would assert that the owner accepted
no limitation, chose no fund and stated no period. Three claims about consent,
made on their behalf, out of fields that were deleted.

**How a lossy rebuild is detected.** From a stored `confirmed` marker, not a
date or a build number — those need a table mapping builds to behaviour, and
the table is what goes stale. The old rebuild dropped all three fields
together, so one surviving with a value proves the build kept all three and
the empty ones are genuinely empty. When all three are empty the question
stays open, which is the honest answer.

That refinement was not defensive. Without it the fix is worse than the defect
it reports: the first version flagged the `968fa8e` canary — a plan written by
the corrected build ten minutes earlier — as needing owner confirmation for
two fields that are correctly empty, and would have done so for every plan
saved from then on.

## Nothing may migrate automatically

Six of seven report `automatic: false`. Two distinct reasons, and they call
for different conversations:

* **the four agent plans and the pilot plan** — the stored reading and a fresh
  reading of the same words differ (`flows`, `inferred`, `held_assets`,
  `tax_treatment`), because this branch changed the compiler: the asset-role
  split, the canonical keys, the time window actually being applied. Replaying
  them substitutes a new interpretation for the one the owner read.
* **all six** — fields that may have held decisions were deleted, so even a
  complete-looking `provenance@2` body cannot be replayed as if it were the
  owner's full statement.

`agrees` deliberately ignores the `confirmed` flag. It answers whether today's
compiler reads the same words the same way; confirming an inference is not a
reading, and the recompile never replays confirmations. Comparing the flag
reported every confirmed plan as drifted and blamed a compiler change that
never happened — a false reason on a true refusal is still a false reason.

## The owner-authorised sequence

1. reconstruct what is safely derivable;
2. identify every decision potentially lost to the confirmation rebuild;
3. ask the owner only for those;
4. show the complete reconstructed scenario — held and observed instruments,
   funding policy, time window, trigger semantics, execution timing, and every
   material difference from the stored one;
5. record explicit authorisation;
6. create a replacement run under the current compiler.

The original stored scenarios and runs stay as historical artifacts. Nothing
is withdrawn or overwritten merely because a migration became possible.

## Method and its limits

* Only `stated_text` and the pinned stage 1 parse are replayed. Both are
  authoritative original input; neither is display text.
* `provenance@2` bodies have their recorded amendments replayed. The `@1` body
  has none, which is the whole difference between the shapes.
* `stated` is never read. The pilot plan's answers survive there as
  `"account_type: TAXABLE (answered)"`, and a sentence composed for a screen
  may not be turned back into a decision. A blinding test requires the
  classification to be identical with `stated` emptied; a mutation that reads
  those sentences fails it.
* The recompile ran under `SYNTHETIC_ONLY` against snapshot `syn-2026-08`,
  which is the policy and snapshot the deployment declares.

## Two infrastructure alarms that were operator error

Recorded because both looked like platform failures and neither was.

`AWS_REGION=ap-southeast-1` is set in the operator's environment and takes
precedence over `AWS_DEFAULT_REGION`. Exporting the latter changed nothing, so
`describe-instances` reported the app instance as non-existent and
`describe-instance-information` returned an empty list. Both were read as real
— "the instance has been terminated", then "the SSM agent is not registered" —
and both were wrong. Everything was running the entire time.

The lesson is the ordinary one: an empty result from a query is evidence about
the query until the query has been shown to be capable of returning something.
The same discipline caught a genuine defect an hour later, when a provider-call
count of zero was checked against a request that certainly made one and
returned zero as well.
