#!/usr/bin/env bash
# Vendor the shared agentic_os runtime into this repo's build context (not committed).
# Usage: scripts/vendor_agentic_os.sh [path-to-agentic-os-src]
#
# The vendored copy is a mirror of the source's agentic_os/. It MUST carry the
# current mission runtime — in particular `mission/from_intent.py` and the
# `MissionRuntime.create_mission_from_intent` it wires — or the cross-runtime
# replay contract test (tests/test_cross_runtime_replay.py) collects-then-fails.
# A stale source (vendored before `from_intent` landed) silently produced a broken
# shadow; the guard below fails loudly instead so a re-vendor from a current
# agentic-os-src checkout is the obvious fix.
set -euo pipefail
SRC="${1:-../agentic-os-src}"
if [ ! -d "$SRC/agentic_os" ]; then
  echo "agentic_os not found at $SRC/agentic_os — pass the agentic-os-src checkout path." >&2
  exit 1
fi
if [ ! -f "$SRC/agentic_os/mission/from_intent.py" ]; then
  echo "refusing to vendor: $SRC/agentic_os/mission/from_intent.py is absent — the source" >&2
  echo "predates the mission from_intent runtime the cross-runtime contract test needs." >&2
  echo "Check out a current agentic-os-src (with mission/from_intent.py) and re-run." >&2
  exit 1
fi
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  "$SRC/agentic_os/" ./agentic_os/
echo "vendored agentic_os from $SRC (mission runtime incl. from_intent)"
