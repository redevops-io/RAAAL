# Pilot workspace export — 2026-08-07

Taken immediately before the pre-invite erasure, so that clearing the
workspace is an operational cleanup rather than a data-loss event.

    file      pilot-workspace-export-20260807T220430Z.json
    sha256    7ed5ce76fea595faefbe3ccc3d7e39387a16c56ec12b193e8dadb4cfb5c78095
    size      954,571 bytes
    location  /home/groot/quantify-pilot-exports/
    build     cf596f4 (serving at the time of export)
    method    src/workspace/erasure.export_workspace(store, "pilot")

## What it contains

    plan                        8
    plan_run                    4
    worksheet                   4
    run_invalidation            1
    market_data_access_event    4

Eight plans, of which **one is a real pilot user's** — the original
`SPX 200DMA`, saved 2026-08-05, the `provenance@1` plan that drove the
recovery matrix and is still pending owner-authorised recovery. The other
seven are harness artifacts: four created by the browser agent
(`agent ORIG-2/5/6/8`), two `canary control` plans and one
`pinned-replay probe`, all created during this engineering stretch.

That count matters: the erasure was first described as removing three test
artifacts. It removes eight plans, and erasure is owner-scoped —
`delete_workspace(store, owner)` — with no per-plan deletion, so the pilot
user's plan could not be kept while removing the rest.

## What this file deliberately does not contain

No plan descriptions, no rendered prose, no figures. The export holds the
user's own words; this receipt names the object and its digest so the export
can be located and verified without reproducing what it holds. Same rule as
the trace-deletion receipt of 2026-08-05.

The export itself is **outside both repositories** and is not committed. It
was staged through `s3://quantify-test-deploy-transfer-…` — encrypted,
private, and lifecycle-expired after one day — which is a transfer path and
not an archive. The durable copy is the local one above.

## Recovering from it

`export_workspace` produces the owner-scoped table dump that
`docs/Privacy_and_Retention.md` describes. Reinstating a plan from it is a
write against the same tables; nothing in this build reads the file back
automatically, and any such replay is subject to the same rule as a migration:
only persisted structured decisions may be replayed, never display text.
