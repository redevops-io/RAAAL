#!/usr/bin/env bash
# Vendor the shared agentic_os runtime into this repo's build context (not committed).
# Usage: scripts/vendor_agentic_os.sh [path-to-agentic-os-src]
set -euo pipefail
SRC="${1:-../agentic-os-src}"
if [ ! -d "$SRC/agentic_os" ]; then
  echo "agentic_os not found at $SRC/agentic_os — pass the agentic-os-src checkout path." >&2
  exit 1
fi
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  "$SRC/agentic_os/" ./agentic_os/
echo "vendored agentic_os from $SRC (discovery/learning/planner/console + investment binding)"
