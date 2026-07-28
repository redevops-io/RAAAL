"""investment-agent — the live Agentic Investment OS service (FastAPI).

Guarded so the app always serves /health even if an optional dependency is missing (mirrors the
agentic_os apps). Paper-trading-first: no broker client / order router is imported anywhere in this
package, so there is no code path to a real venue.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config import DEMO_DISCLAIMER

PORT = int(os.environ.get("PORT", "8250"))

app = FastAPI(title="investment-agent (RAAAL Agentic Investment OS)")

# guarded router mount: a broken optional import degrades to /health, never a crash
_ROUTERS_OK = True
_ROUTER_ERR = ""
try:
    from .api_investment import router as investment_router
    from .console import router as console_router
    from .research import router as research_router
    app.include_router(investment_router)
    app.include_router(research_router)
    app.include_router(console_router)   # serves the operating console at /
except Exception as exc:  # noqa: BLE001
    _ROUTERS_OK = False
    _ROUTER_ERR = str(exc)[:200]


@app.get("/health")
def health():
    return {"ok": True, "service": "investment-agent", "routers": _ROUTERS_OK,
            "error": _ROUTER_ERR, "mode": "paper", "disclaimer": DEMO_DISCLAIMER}


@app.get("/info")
def info():
    return JSONResponse({
        "service": "investment-agent",
        "disclaimer": DEMO_DISCLAIMER,
        "console": "/", "api": "/api/investment/...", "research": "/research", "health": "/health",
        "paper_only": True, "external_execution_path": False,
    })


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
