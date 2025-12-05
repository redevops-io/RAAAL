"""Run historical refresh loop and serve the Bokeh dashboard."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import threading
import time
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.history import run_historical_analysis, save_history
from src.visualization.bokeh_app import REPORTS_DIR, build_dashboard

SITE_DIR = Path(os.environ.get("SITE_DIR", "site"))
SITE_DIR.mkdir(parents=True, exist_ok=True)


def rebuild_dashboard(
    start_dt: datetime,
    warmup_days: int,
    step_days: int,
    force_refresh: bool,
) -> Path:
    # Use today's date at midnight to avoid timezone issues with market data
    end_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    logging.info("Running historical analysis up to %s", end_dt.date().isoformat())
    result = run_historical_analysis(
        start=start_dt,
        end=end_dt,
        warmup_days=warmup_days,
        step=step_days,
        force_refresh=force_refresh,
    )
    save_history(result)
    dashboard_path = build_dashboard(output_path=REPORTS_DIR / "regime_dashboard.html")
    target = SITE_DIR / "index.html"
    shutil.copyfile(dashboard_path, target)
    logging.info("Dashboard rebuilt -> %s", target)
    return target


def _scheduler_loop(
    interval_seconds: int,
    start_dt: datetime,
    warmup_days: int,
    step_days: int,
    force_refresh: bool,
) -> None:
    if interval_seconds <= 0:
        logging.info("Scheduler disabled (interval <= 0)")
        return
    logging.info("Starting scheduler loop every %s seconds", interval_seconds)
    while True:
        time.sleep(interval_seconds)
        try:
            rebuild_dashboard(start_dt, warmup_days, step_days, force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Scheduled rebuild failed: %s", exc)


def serve_http(port: int) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(SITE_DIR))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    logging.info("Serving %s on port %s", SITE_DIR, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        logging.info("Shutting down server")
        server.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dashboard service")
    parser.add_argument("--start", default=os.environ.get("START_DATE", "2015-01-01"), help="Historical start date YYYY-MM-DD")
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP_DAYS", 252)), help="Warmup window in trading days")
    parser.add_argument("--step", type=int, default=int(os.environ.get("STEP_DAYS", 5)), help="Evaluate every N trading days")
    parser.add_argument("--interval", type=int, default=int(os.environ.get("REFRESH_INTERVAL", 86400)), help="Seconds between refresh jobs (<=0 disables)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="HTTP port to serve dashboard")
    parser.add_argument("--force-refresh", action="store_true", default=os.environ.get("FORCE_REFRESH", "1") == "1", help="Force price download on first build")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    start_dt = datetime.fromisoformat(args.start)

    rebuild_dashboard(start_dt, args.warmup, args.step, args.force_refresh)

    scheduler_thread = threading.Thread(
        target=_scheduler_loop,
        args=(args.interval, start_dt, args.warmup, args.step, args.force_refresh),
        daemon=True,
    )
    scheduler_thread.start()

    serve_http(args.port)


if __name__ == "__main__":
    main()
