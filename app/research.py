"""Serve the nightly-built Bokeh research dashboard (all 6 tabs) at /research.

The heavy backtest + Bokeh render happens in CI (daily-deploy); the live service just serves the
artifact and exposes a small parquet-derived summary. Includes the DEMO / not-investment-advice framing.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

from src.config import DEMO_DISCLAIMER

router = APIRouter(tags=["research"])

_RESEARCH_HTML = Path(os.environ.get("RAAAL_RESEARCH_HTML", "reports/regime_dashboard.html"))

_PLACEHOLDER = f"""<!doctype html><html><head><meta charset="utf-8"><title>RAAAL Research</title></head>
<body style="font-family:system-ui;max-width:720px;margin:60px auto;color:#222">
<h1>Research dashboard</h1>
<p style="background:#fff8e1;border-left:4px solid #ffc107;padding:12px 16px;border-radius:5px">
<b>DEMO — not investment advice.</b> {DEMO_DISCLAIMER}</p>
<p>The nightly Bokeh dashboard (regime bands, strategy lab, salience, FOMO/FOBI) has not been built yet.
Run <code>python -m src.history</code> then <code>python -m src.visualization.bokeh_app</code>, or wait
for the daily CI to publish it here.</p>
</body></html>"""


@router.get("/research", response_class=HTMLResponse)
def research():
    if _RESEARCH_HTML.exists():
        return FileResponse(str(_RESEARCH_HTML), media_type="text/html")
    return HTMLResponse(_PLACEHOLDER)
