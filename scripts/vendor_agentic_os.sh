#!/usr/bin/env bash
# Vendor the shared runtime packages into this repo's build context (not committed).
# Post open-core split the runtime is TWO packages: the public `agentic_os` core and the private
# `agentic_os_enterprise` overlay (discovery/learning/planner/manifest + the investment binding).
# The agentic-os-src checkout holds agentic_os_enterprise natively and the public agentic_os vendored
# in by the deploy's build-push step.
# Usage: scripts/vendor_agentic_os.sh [path-to-agentic-os-src]
set -euo pipefail
SRC="${1:-../agentic-os-src}"
if [ ! -d "$SRC/agentic_os" ] || [ ! -d "$SRC/agentic_os_enterprise" ]; then
  echo "agentic_os / agentic_os_enterprise not both present under $SRC — pass the agentic-os-src checkout path (with the public core vendored in)." >&2
  exit 1
fi
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  "$SRC/agentic_os/" ./agentic_os/
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  "$SRC/agentic_os_enterprise/" ./agentic_os_enterprise/
echo "vendored agentic_os (public core) + agentic_os_enterprise (discovery/learning/planner + investment binding) from $SRC"
