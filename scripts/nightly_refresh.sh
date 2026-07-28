#!/usr/bin/env bash
# Nightly research refresh for the live Agentic Investment OS (quantify.club).
#
# Runs the RAAAL historical backtest + Bokeh dashboard INSIDE the lean investment-agent image (the
# forecaster/sentiment ML stack is optional; this uses exponential-mean + rule/ensemble regimes), writing
# a fresh prices.parquet + timeline/weights + regime_dashboard.html into the host volumes the live
# container serves from. The console reads the parquet on every request and /research is a file read, so
# NO container restart is needed — the site picks up the new data on the next page load.
#
# Installed on proxmox as a daily systemd timer (see deploy/DEPLOY.md §5). Idempotent + self-logging.
set -euo pipefail

RAAAL_DIR="${RAAAL_DIR:-/projects/RAAAL}"
IMAGE="${IMAGE:-localhost:5000/investment-agent:stable}"
START="${START:-2016-01-01}"
END="$(date +%F)"
LOG="${RAAAL_DIR}/reports/nightly.log"

mkdir -p "${RAAAL_DIR}/data/history" "${RAAAL_DIR}/reports"
echo "[$(date -u +%FT%TZ)] nightly refresh START (start=${START} end=${END})" | tee -a "$LOG"

docker run --rm \
  -v "${RAAAL_DIR}/data:/app/data" \
  -v "${RAAAL_DIR}/reports:/app/reports" \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  "$IMAGE" sh -c "python -m src.history --start ${START} --end ${END} --refresh && python -m src.visualization.bokeh_app" \
  >>"$LOG" 2>&1

# quick freshness assertion so a silent failure is visible in the log
python3 - "$RAAAL_DIR" <<'PY' | tee -a "$LOG"
import sys, pathlib
d = pathlib.Path(sys.argv[1])
p = d / "data/history/prices.parquet"
h = d / "reports/regime_dashboard.html"
import time
def age_h(x): return (time.time() - x.stat().st_mtime) / 3600 if x.exists() else None
print(f"[freshness] prices.parquet exists={p.exists()} age_h={age_h(p)}")
print(f"[freshness] regime_dashboard.html exists={h.exists()} age_h={age_h(h)}")
PY

echo "[$(date -u +%FT%TZ)] nightly refresh DONE" | tee -a "$LOG"
