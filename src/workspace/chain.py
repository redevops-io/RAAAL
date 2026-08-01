"""The private artifact chain, composed from the public chain primitives.

A scenario's chain is not a methodology's chain — it starts at a question
somebody asked rather than at a published rule, and it ends at a disclosure
rather than at publication. But the *rendering* must be identical, so this
reuses `Link`, `State`, `Domain` and `ChainState` rather than inventing a
parallel vocabulary. A private plan showing a differently-shaped status row
would teach users that the two halves of the product mean different things by
the same symbols.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..web.chain import Adversity, ChainState, Domain, Link, State

#: Canonical order for a scenario. Position is stable across every plan, for the
#: same reason the public one is: column-scanning only works if a step is always
#: in the same slot.
SCENARIO_CHAIN_ORDER: Sequence[tuple] = (
    ("Intent", Domain.REASONING),
    ("Understanding", Domain.REASONING),
    ("Scenario", Domain.EXECUTION),
    ("Flows", Domain.EXECUTION),
    ("Rule", Domain.EXECUTION),
    ("Simulation", Domain.EXECUTION),
    ("Benchmarks", Domain.JUDGMENT),
    ("Trials", Domain.JUDGMENT),
    ("Disclosure", Domain.JUDGMENT),
)


def _link(step: str, state: State, value: str, **kw) -> Link:
    domain = next(d for s, d in SCENARIO_CHAIN_ORDER if s == step)
    return Link(step=step, domain=domain, state=state, value=value, **kw)


def build_scenario_chain(
    *,
    subject: str,
    scenario=None,
    intent=None,
    result=None,
    benchmarks: Sequence[Any] = (),
    comparability=None,
    saved: bool = False,
) -> ChainState:
    """One payload, rendered by the same glyph and table the library uses."""
    links: List[Link] = []

    if intent is None:
        links.append(_link("Intent", State.ABSENT, "described directly",
                           summary="no candidate set was generated"))
    else:
        links.append(_link(
            "Intent", State.OK, intent.artifact_id,
            summary=intent.disclosure(),
            adversity=Adversity.ADVISORY if intent.is_a_search else Adversity.NONE,
        ))

    if scenario is None:
        links.append(_link("Understanding", State.UNKNOWN, "not compiled"))
    else:
        p = scenario.provenance
        open_items = len(p.unresolved) + len(p.unconfirmed) + len(p.open_contradictions)
        links.append(_link(
            "Understanding",
            State.BLOCK if p.open_contradictions else
            (State.WARN if open_items else State.OK),
            (f"{open_items} to confirm" if open_items else "fully confirmed"),
            summary=(
                f"{len(p.stated)} stated · {len(p.inferred)} inferred · "
                f"{len(p.unresolved)} open"
            ),
            adversity=(
                Adversity.BLOCKING if p.open_contradictions
                else (Adversity.ADVISORY if open_items else Adversity.NONE)
            ),
        ))
        links.append(_link(
            "Scenario",
            State.BLOCK if not scenario.is_runnable else
            (State.OK if saved else State.WARN),
            scenario.artifact_id,
            summary=("saved" if saved else "provisional — not yet saved"),
            adversity=(Adversity.BLOCKING if not scenario.is_runnable
                       else Adversity.NONE),
        ))
        links.append(_link(
            "Flows", State.OK, f"{scenario.flow_schedule.cadence}",
            summary=f"schedule {scenario.flow_schedule.schedule_hash[:12]}…",
        ))
        links.append(_link(
            "Rule",
            State.OK if scenario.event_program else State.ABSENT,
            f"{len(scenario.event_program)} step(s)",
            summary=f"rule {scenario.rule_hash[:12]}…",
        ))

    if result is None:
        links.append(_link("Simulation", State.UNKNOWN, "not run"))
    else:
        links.append(_link(
            "Simulation", State.OK,
            f"{len(result.path.value)} sessions",
            summary=("time-weighted and money-weighted reported separately"),
        ))

    comparable = [b for b in benchmarks if getattr(b, "comparable", False)]
    incomparable = [b for b in benchmarks if not getattr(b, "comparable", True)]
    links.append(_link(
        "Benchmarks",
        State.WARN if incomparable else (State.OK if comparable else State.ABSENT),
        f"{len(comparable)} comparable",
        summary=(f"{len(incomparable)} could not receive the same contributions"
                 if incomparable else "all received identical contributions"),
        adversity=Adversity.ADVISORY if incomparable else Adversity.NONE,
    ))

    trials = getattr(intent, "trials_incurred", 0) if intent else 0
    links.append(_link(
        "Trials", State.WARN if trials > 1 else State.OK,
        f"{max(trials, 1)} attempt(s)",
        summary=("alternatives chosen by result count as attempts"
                 if trials > 1 else "no selection penalty applies"),
        adversity=Adversity.ADVISORY if trials > 1 else Adversity.NONE,
    ))

    if comparability is None:
        links.append(_link("Disclosure", State.UNKNOWN, "—"))
    else:
        links.append(_link(
            "Disclosure",
            State.OK if comparability.attribution_isolated else State.WARN,
            comparability.comparison_class.value.replace("_", " ").lower(),
            summary=comparability.required_disclosure,
            adversity=(Adversity.NONE if comparability.attribution_isolated
                       else Adversity.ADVISORY),
        ))

    return ChainState(subject=subject, links=tuple(links))
