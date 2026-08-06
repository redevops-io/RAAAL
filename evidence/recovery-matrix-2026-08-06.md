# Recovery matrix — five production plans, 2026-08-06

Produced by `src/workspace/recovery.py` (`plan-recovery@1`) against the live
`quantify-test` RDS instance, read through the running `quantify-api-1`
container over SSM. Read-only: five `SELECT`s, nothing written, the serving
container unmodified. Raw output in `recovery-matrix-2026-08-06.json`.

Build serving at the time: `3eaa5eb`. Assessment run from `9512ff4`.

## What is on disk

| Plan | Shape | Funding policy | Automatic |
|---|---|---|---|
| `agent ORIG-8` | `provenance@2` | **absent** | no |
| `agent ORIG-6` | `provenance@2` | **absent** | no |
| `agent ORIG-5` | `provenance@2` | **absent** | no |
| `agent ORIG-2` | `provenance@2` | **absent** | no |
| `SPX 200DMA` (pilot) | `provenance@1` | **absent** | no |

**F11 is confirmed in production, and it is not confined to the old plan.**
Four of these were saved today, after `3eaa5eb` had deployed the provenance
fix — they carry complete `provenance@2` bodies and still have no funding
policy, because the save path compiled without a priceable set. Every stored
plan on this deployment is an event-triggered strategy that cannot execute.

## The pilot plan, field by field

`SPX 200DMA`, the plan that started this work. `provenance@1`, so four of the
eight provenance fields were never written; no funding policy; and a recompile
still asks four questions.

| Outcome | Fields |
|---|---|
| Recoverable from structure | `held_assets` `weighting` `flows` `event_program` `holdings_policy` `tax_treatment` `benchmark_set` `spec_version` `inferred` `time_window` |
| Requires owner confirmation | `funding` `amendments` `exclusions` `asset_resolutions` |
| Unrecoverable without prose | none |

Open questions a recompile raises: `funding_source`, `account_type`,
`asset_identity:sp500-etf`, `unclear:total-amount-and-return-now`.

`funding` sits under *requires owner confirmation* rather than *recoverable*
for a reason worth stating: it is a derivation, not a decision, and nothing
about it was lost. It cannot be rebuilt only because the instrument is
unresolved, and resolving `SP500 ETF` is the owner's choice between two funds.
Answer that and the policy recompiles from the description alone.

The first version of this matrix reported `funding` as **unrecoverable
without prose**, which is the one reading that would have caused harm — it
tells an operator to abandon a field that four answers restore. The rule now
distinguishes *absent and lost* from *absent and blocked on a question the
recompile is still asking*; `test_a_derivation_blocked_on_a_question_is_not_
called_historical` holds it, and a mutation collapsing the two fails it.

## Nothing may migrate automatically

Every plan, including the four modern ones, reports `automatic: false`. For
the four that is not a gap in what was persisted — all fourteen fields are
recoverable — it is `agrees: false` on `flows`, `inferred` and `time_window`.
The stored reading and a fresh reading of the same words differ, because this
branch changed the compiler: the role split, the canonical keys, the time
window actually being applied.

That is the intended outcome. Replaying those plans would not restore what the
user confirmed; it would substitute a new interpretation of their words for the
one they read. It is a separate act needing separate consent, which is what
`migrate_plan`'s `--authorized-by` exists to record.

`agent ORIG-2` additionally still asks `asset_identity:s-p-500`.

## Method and its limits

* The description and the pinned stage 1 parse are the only inputs replayed.
  Both are authoritative original input, not display text.
* `provenance@2` bodies have their recorded amendments replayed (2, 2, 2 and 3
  respectively). The `@1` body has none to replay, which is the whole
  difference between the shapes and why it alone reports fields as needing
  their owner.
* `stated` is never read. The pilot plan's answers survive there as
  `"account_type: TAXABLE (answered)"`, and a sentence composed for a screen
  may not be turned back into a decision. `test_the_stated_sentences_are_not_
  consulted` blinds that field and requires the classification to be
  identical; a mutation that reads those sentences fails it.
* The recompile ran under `SYNTHETIC_ONLY` against snapshot `syn-2026-08`,
  which is the policy and snapshot the deployment itself declares.
