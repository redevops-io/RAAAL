#!/usr/bin/env bash
#
# The freeze regression suite (freeze plan §9).
#
# Not the whole test tree — that carries strategy, parser, and benchmark tests
# whose churn is unrelated to what the freeze protects. This is the curated set
# that must stay green for the v0.2.x runtime-integration boundary to remain
# frozen, grouped by the seven dimensions the freeze declares: contracts,
# authority, persistence, money, execution, security, deployment.
#
# Run it in the clean serving image, offline, so the result reflects what ships:
#
#     docker run --rm --network=none -v "$PWD:/app" -w /app \
#       --entrypoint bash quantify-test:git \
#       -c 'git config --global --add safe.directory /app; scripts/regression.sh'
#
# Tests that need PostgreSQL or the network parser skip (not fail) offline; the
# live/CI run exercises those. A non-zero exit means a frozen invariant broke.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTEST="${PYTEST:-python3 -m pytest}"

contracts=(
  test_canonical_contracts test_runtime_boundary test_runtime_export
  test_runtime_artifact_route test_wm_boundary_conformance
  test_canonicalisation_exit_gate test_the_contract_is_not_re_vendored)
authority=(
  test_declared_rule_not_executed test_declared_survives_failure_to_compile
  test_workflows_do_not_fire_by_themselves test_verified_intent_emission
  test_confirmation_preserves_provenance test_event_triggered_execution)
persistence=(
  test_provenance_persistence test_reconciliation_persistence
  test_snapshot_store test_snapshot_lifecycle test_plan_recovery
  test_replay_from_pinned_intent)
money=(
  test_allocation test_total_return test_twr_conformance test_mwr_conformance
  test_drawdown_conformance test_rebalance)
execution=(
  test_execution test_executor test_worksheet_execution)
security=(
  test_secret_exposure test_security_headers test_access_chain
  test_scope_disclosure test_boundary_sweep)
deployment=(
  test_preflight test_release_manifest test_deployed_revision_identity
  test_serving_image_contract test_image_pins_the_runtime_it_was_tested_against
  test_container_parser_pinning)

paths=()
for group in contracts authority persistence money execution security deployment; do
  declare -n names="$group"
  for n in "${names[@]}"; do
    f="tests/${n}.py"
    [[ -f "$f" ]] && paths+=("$f") || printf 'note: %s not present, skipped from the suite\n' "$f" >&2
  done
done

printf '\n=== freeze regression: %d test files across 7 dimensions ===\n' "${#paths[@]}" >&2
exec $PYTEST -q -p no:cacheprovider --ignore=tests/test_cross_runtime_replay.py "${paths[@]}"
