# The shadow matrices are stale

    built under   quantify-discovery-schema@2/103c025ac629946f
    current       quantify-discovery-schema@6/eb01a824e4f43d02

## What changed

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
