"""The live Agentic Investment OS service (FastAPI).

Wraps RAAAL's deterministic engine with the runtime stack over HTTP. Paper-trading-first: this package
imports NO broker client and NO order router — there is no code path from here to a real venue. The only
state-mutating endpoint is mission approval, which applies a PAPER order to the local state store.
"""
from __future__ import annotations
