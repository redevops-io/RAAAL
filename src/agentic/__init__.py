"""The agentic operating layer around RAAAL's deterministic engine.

This package wraps RAAAL (the quantitative authority) with the ReDevOps runtime
stack: Discovery Runtime (detectors + proposals), the Decision Planner (strategy
SELECTION over the approved registry — never fabrication), Context Runtime
(evidence bundles), Mission Runtime (objective-compare graph + approval), and a
safe learning selector.

Design rules (see src/agentic/selection.py and the strategy registry in
src/strategies.py):
  * RAAAL strategies produce candidate portfolios; the runtime only selects among them.
  * No allocation is ever produced outside a registered strategy `implementation`.
  * Hard mandate constraints are applied before any selection or learning.
  * Paper-trading-first: the live path has no code route to a real venue.

`agentic_os` (the shared runtime package) is imported lazily inside the modules
that need it, so `src.agentic.selection` stays importable without it.
"""
from __future__ import annotations
