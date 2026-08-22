# Quantify / RAAAL / wealth-manager — v0.2.x runtime integration, FROZEN

**Status:** frozen · **Date:** 2026-08-21 · **Canonical plan:**
`QUANTIFY_WEALTH_MANAGER_V0.2X_FREEZE_AND_V0.3_HANDOFF_PLAN.md`

This declares the v0.2.x runtime-integration boundary between **RAAAL** (intent
authoring) and **wealth-manager** (governed execution authority) **frozen** on the
proven dual-identity + governed-execution design. No speculative v0.3.0 work
begins until this line has shipped and held; the v0.3.0 handoff is a separate
program (see the plan's §11 exclusions).

The freeze is not a wish — it is a set of refusals that run where the deployment
runs. Each item below names the invariant, where it is enforced, and the test that
would fail if it regressed.

## What is frozen (the boundary)

- **Dual identity.** RAAAL carries its native `source_intent_hash` **verbatim**;
  the canonical `runtime_artifact_hash` is computed under `rcv1`
  (runtime-contracts 0.3.x, Decimal canonical). The two are different kinds and
  are never interchangeable.
  → `src/runtime_boundary.py`; `tests/test_runtime_boundary.py`,
  `tests/test_runtime_export.py`.
- **Authority.** RAAAL authors intent; wealth-manager holds execution authority.
  A declared rule is not executed by the author; workflows never self-fire.
  → `tests/test_declared_rule_not_executed.py`,
  `tests/test_workflows_do_not_fire_by_themselves.py`.
- **Money policy.** USD, minor-unit 2, settlement rounding FLOOR; cash authority
  is integer minor units. → wealth-manager `tests/test_money.py`.
- **Consumer fail-closed.** wealth-manager refuses an artifact under an
  unsupported canonicalization (never rehashes it) and refuses a payload whose
  hash does not match. → wealth-manager `verify_runtime_artifact`;
  `redevops-conformance/python/test_dual_identity.py`,
  `tests/test_wm_boundary_conformance.py`.

## Deployment hardening added for the freeze

| # | Invariant | Enforced in | Test |
|---|-----------|-------------|------|
| §4.1 | The serving image refuses to become ready if its installed runtime-contracts is below the floor (`0.3.0`) or under the wrong canonicalization (`rcv1`) — `RUNTIME_CONTRACT_MISMATCH`. | `src/deploy/preflight.py` | `tests/test_preflight.py` |
| §5 | An immutable **release manifest** binds code + image digest + exact dependency versions; promotion **refuses** an image that does not match it (the stale-digest incident). | `deploy/release/manifest.py`, emitted by `scripts/build_image.sh`, verified in `.github/workflows/deploy-aws.yml` before `terraform apply` | `tests/test_release_manifest.py` |
| §6.1 | The exported artifact makes the producing runtime-contracts version and producer build revision **explicit** in `protocol`/`provenance` — a consumer never infers them from the hash shape. | `src/runtime_boundary.py` | `tests/test_runtime_boundary.py` |
| §6.2 | The runtime-artifact route serves its canonical identity as a strong **ETag**; `If-None-Match` revalidates to `304`. Content-addressed, so a matching tag can never serve a stale artifact. | `src/workspace/pilot_routes.py` | `tests/test_runtime_artifact_route.py` |
| §7 | The clean-container build check pins runtime-contracts to the frozen floor — **no downgrade path** (a vendored submodule or cached wheel below the floor fails the build). | `scripts/build_image.sh` (step 4) | build-time gate |
| §8 | The Phase-2.5 acceptance evidence is preserved **untouched** and byte-guarded, so the post-2.5 rc-0.3.x migration cannot silently regenerate the record. | wealth-manager `deploy/frozen-original/` | wealth-manager `tests/test_phase25_evidence_frozen.py` |

The floor `(0, 3, 0)` and canonicalization `rcv1` are enforced at **three**
independent points — build (image), promotion (manifest), and runtime (readiness)
— so no single bypass can ship code below the frozen contract.

## Regression evidence (§9)

`scripts/regression.sh` runs the curated freeze suite across the seven dimensions
(contracts, authority, persistence, money, execution, security, deployment) in the
clean serving image, offline.

- **RAAAL:** 678 passed, 24 skipped in the offline clean image. The only
  non-passing cases require what an offline mounted-tree run cannot provide and
  the pipeline validates instead: the network parser (weight extraction), a
  **freshly built** image (installed-version and submodule-tag identity — asserted
  by `build_image.sh` steps 4–5), a live database, and a deployment with build
  metadata (the app's fail-closed startup, which *correctly* refuses to serve as
  `BUILD_UNOBSERVABLE` offline). None are boundary-logic regressions.
- **wealth-manager:** 115 passed.

## Explicitly NOT in scope (deferred to v0.3.0)

GPU/cuOpt/cuDF/cuVS/TensorRT/NIM/Jetson; a Governance plane; live Robinhood
writes; public routing for wealth-manager; and any RAAAL model migration. See the
plan's §11. Reopening the freeze requires a release-blocking defect, not an
enhancement.

## Release identity

- RAAAL live line: `deploy-eks-recreate` (serving quantify.club); runtime-contracts
  `0.3.0`, canonicalization `rcv1`.
- wealth-manager: Phase 2.5 frozen at `482da0d`; migrated onto rc 0.3.x post-2.5.
- The freeze is tagged `v0.2.x-runtime-integration-frozen` at the commit that
  lands this document.
