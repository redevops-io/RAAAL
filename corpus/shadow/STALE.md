# The shadow matrices are stale

    built under   quantify-discovery-schema@2/103c025ac629946f
    current       quantify-discovery-schema@8/bb28984534416245

## What changed

@8 gave `allocation_method` the computed research strategies as named values —
risk parity, minimum variance, the momentum and factor families, and the rest of
the twenty-one the research engine already ran. The engine gained an executor for
them (`mission.rebalance.strategy_driven`), so a sentence describing one now
reads to the method that runs it rather than to silence, and the reader's prompt
changed with the vocabulary. The hosted recordings were refreshed under the new
question (`record_hosted.py --refresh`, all three readers); the shadow matrices
were not.

@7 added the `factor_tilt` and `age_based_allocation` dimensions. The live drift
lane found both families executing under gpt-5.4 where they had refused under
gpt-4.1 — and the old refusal was an accident, an unrelated `portfolio_sleeves`
relation failing first on a model that reported it consistently. Both are
dimensions the schema could not represent, so nothing could refuse them by name.
Neither is asked of the hosted reader; a deterministic reader authors them.

@6 added the `selection_rule` and `holding_period` dimensions. The strategy
evaluation benchmark found "hold whichever performed best" compiling to two
holdings on a monthly cadence — the selection gone, and the plan executing
anyway — and separately found "hold VTI for 200 days" and "buy VTI below its
200-day moving average" producing the identical plan. Both are dimensions the
schema could not represent, so Discovery had nothing to hand Mission and Mission
had nothing to refuse by name.

@5 added the `asset_location` relation — the last schema gap the strategy sweep
left standing, where `account_type` returned TAXABLE for a sentence whose whole
request was a mapping.

@4 added `reserve_policy` and `bucket_policy` relations and a
`leverage_multiplier` qualifier on `portfolio_sleeves`, because the live drift
lane found those three families both silently reduced and execution-unstable —
what a representational gap looks like from the outside.

@3, before that: `objective` gained two values, `assess_conversion` and `assess_debt_repayment`.
The schema's own examples already promised *"should I convert to a Roth"* while
the vocabulary had no value for it, so the reader answered `other` — correctly,
and uselessly, because `other` is what a reader says when a sentence names no
objective at all. Mission then executed a Roth conversion as an ordinary
contribution plan.

## Why the matrices cannot simply be re-frozen

They record AGREE/DISAGREE counts computed while asking readers about a
different set of possible answers. Re-pointing the frozen fingerprint at @3
without re-running would leave numbers that describe @2 wearing an @3 label —
the same defect as a reader whose behaviour changed under an unchanged id, which
is why `quantify-compiler@2` was bumped from `@1` in the first place.

## What must happen before these numbers are cited again

    python corpus/shadow_run.py            # re-runs both corpora against @3

That costs provider calls over the full shadow corpora, which is why it has not
been done as a side effect of a vocabulary change. Until it is, the Phase 3 exit
numbers stand as a record of what was true at @2 and must not be quoted as
current.

## What is *not* stale

`corpus/parser/strategy_closure.json` was recorded after the bump and measures
the serving reader against @3 directly.
