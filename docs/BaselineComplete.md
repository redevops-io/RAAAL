# Baseline complete

The point at which a defect stops being an architecture question.

After this line, an issue is **product**, **UX**, **capability**, **performance**
or **pilot evidence** — and is answered by contact with users rather than by
stating another invariant. Before it, the foundations could not be trusted to
support learning, so internal work was the fastest way to improve the system.

The distinction matters because there is no natural end to inventing
invariants. Each one this branch applied found something real; that is exactly
why a stopping rule has to be written down rather than felt.

## The six criteria

| | Criterion | Witnessed by |
|---|---|---|
| 1 | **Compiler** — a description compiles to a plan, or names what it cannot read | `tests/test_the_original_prompt.py`, the vocabulary and registry suites |
| 2 | **Provenance** — every decision records who made it, and survives storage | `tests/test_provenance_persistence.py` · **pending in production, see below** |
| 3 | **Execution** — a declared rule runs, or no figure is produced | `tests/test_event_triggered_execution.py`, `tests/test_declared_rule_not_executed.py` |
| 4 | **Telemetry** — journeys are observed, and observe nothing they should not | `tests/test_telemetry_reachability.py`, canary-verified in production |
| 5 | **Verification** — evidence reports PASS, FAIL *or* VACUOUS | the falsification passes; `deploy/provenance_gate.py` exits 2 when untested |
| 6 | **Deployment** — the running system states its own identity and refuses when it cannot | `deploy/acceptance.py`, the startup proof, schema parity |

## What is not yet witnessed

**Criterion 2 has never been demonstrated in production.**

The code is deployed (`3eaa5eb`, migration `a91c4e7b2f05`) and both database
lanes pass. But every plan in the production workspace predates the fix, so the
gate that would prove it reports:

    VACUOUS: no plan on the current shape exists, so the check that matters —
    prose answers without structured records — was never evaluated.

That is the honest state, and it is one browser session from closing: a plan
created through the builder after this deploy, then inspected in storage for
structured `amended`, `asset_resolutions` and `time_window`, reopened without
questions reappearing, and executed to a reconciled ledger.

**Until that run exists, this document describes a baseline that is complete in
code and unwitnessed in production.** Declaring it done on the strength of a
green suite would be the failure this branch spent its length removing.

## The three properties that were not true before

**A plan can no longer silently produce the wrong result.** It compiles and
executes, or it refuses and says exactly why. The middle case — a figure for a
strategy that never ran — is gone, and a stored one from before is withdrawn
rather than deleted.

**User decisions are first-class data.** An answer is a `ScenarioAmendment`
with a source and a timestamp, stored structurally and rendered from that
record. Previously it became a sentence, and a sentence cannot be replayed.

**Evidence distinguishes three outcomes, not two.** PASS, FAIL and VACUOUS. The
third is the one that took longest to learn: a check that reports success
without having evaluated anything is worse than a missing check, because it
converts an unknown into a false known.

## The layering that produced them

    code → tests → SQLite lane → PostgreSQL lane → deployment → production gate

Each layer found defects the previous one structurally could not. The
PostgreSQL lane is not the same tests on another engine: `JsonText` and
`sa.Text()` both render TEXT on SQLite, so a whole class of type-parity defect
can only exist where they differ. The production gate is not the suite again:
it reads rows that real journeys wrote.

The last two entries are why "the tests pass" was never the standard here.
