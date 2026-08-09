"""The execution boundary: a compiled plan in, a figure and its evidence out.

    runtime artifact ─┐
                      ├─► execute_compiled_plan ─► figure + evidence
    legacy artifact ──┘

**One implementation, two callers.** The pilot path does not get its own copy of
Quantify's execution logic — a second copy would drift, and the drift would show
up as two users getting different numbers from the same plan.

**Legacy execution may be reused; legacy interpretation may not be
reintroduced.** That is the rule this module exists to make structural.
`execute_compiled_plan` takes a `ScenarioSpecification` — something already
compiled — and there is no path through it that turns text into a plan.
`compile_scenario` and `compile_draft` are not imported here and
`test_pilot_route` proves the pilot branch never reaches them.

**Why `stated_text` is still a parameter, and why that is not reinterpretation.**
The coverage gate reads the user's own words to check that every declared
element was actually executed — the gate that caught three prompts returning an
identical $103,393 while each quietly dropped a different declared element. It
*verifies*; it never produces a value. And for a runtime plan it is deliberately
an independent witness: the reader said what the sentence meant, and coverage
asks the raw text whether anything the reader found was then lost on the way to
a figure. Deriving the check from the reader's own output instead would be
asking one witness to confirm itself.

**Why this file is not called `execution.py`.** That name was taken —
`workspace/execution.py` states what an accepted worksheet proposal will
execute — and the first version of this module overwrote it. Nothing warned:
the write reported success, the module imported, and the failure surfaced as a
collection error in a test file whose subject had silently ceased to exist.
Second occurrence of that class in this project.

**Where the body lives.** `_run` and its helpers are still defined in
`routes.py`. Moving them is a mechanical change this module deliberately does
not bundle with wiring the pilot: the boundary is what the pilot needs, and a
four-hundred-line move in the same commit as a new execution path would make a
regression in either indistinguishable from the other. The import is one-way —
this module reaches into routes, nothing reaches back — so the move is a
later edit to one file.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def execute_compiled_plan(scenario, access, *,
                          scope: Optional[Dict[str, Any]] = None,
                          stated_text: str = "") -> Dict[str, Any]:
    """Simulate a compiled scenario and its benchmarks under one set of flows.

    `access` is the `MarketDataAccess`, never a bare frame: the record of which
    data produced a figure has to travel with the data, because a caller
    attaching provenance afterwards is a caller that can forget — and a run it
    forgot on looks exactly like one it did not.

    Returns the run dict the worksheet renders, including `unavailable` and
    `strategy_not_executed` when the engine refuses. A refusal is a result: the
    figure is absent and the reason is the engine's own.
    """
    from .routes import _run

    return _run(scenario, access, scope, stated_text=stated_text)


def market_data_for(scenario, *, context: str, plan_id: str = ""):
    """The prices this plan needs, with the provenance of where they came from.

    Wrapped for the same reason as above: one accessor, so both paths ask the
    same question and a change to how data is resolved cannot reach one and
    miss the other.
    """
    from .routes import _market_data

    return _market_data(context, plan_id=plan_id, scenario=scenario)
