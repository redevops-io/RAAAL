"""Whether the production solver conforms to the frozen MWR contract.

    docs/Measures.md            what a money-weighted return means
    Returns/MoneyWeighted  the reporting contract, proven
    this module            whether `accounting.money_weighted_return` obeys it

The separation is the point. The solver may search; it does not get to define
what a valid answer is. So nothing here calls the solver to find out what MWR
means — it takes the definition as given and asks whether what came back is
admissible and unique.

**What this can and cannot establish.** Root *existence* is decidable by
sampling: two sign changes in the present-value curve prove two roots, and that
is enough to say a series is `NON_UNIQUE`. Root *uniqueness* is not — a grid
that finds one crossing has found one crossing, and calling that a proof of
uniqueness would be the same defect as a solver reporting the first root it
happens to reach. `NO_EVIDENCE_OF_NON_UNIQUENESS` says exactly what it means,
and is deliberately clumsy so nobody quotes it as `UNIQUE`.

**Tolerances live here and nowhere above.** `docs/Measures.md` has no epsilon in it
and must not acquire one: a financial definition with a tolerance is a
definition that changes when somebody tunes a solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

#: Bumped when `docs/Measures.md` changes in substance. A certification names the
#: contract it was checked against, because "the solver was verified" is not a
#: fact unless it says verified against what.
CONTRACT_VERSION = "quantify/mwr-contract@1"

#: The implementation this module certifies. Read from the engine rather than
#: restated, so a solver that changed under an unchanged name cannot carry an
#: old certification.
SOLVER = "mission.accounting.money_weighted_return"

#: Only the implementation boundary has one.
ROOT_TOLERANCE = 1e-9

RATE = "RATE"
NO_SOLUTION = "NO_SOLUTION"
NON_UNIQUE = "NON_UNIQUE"
INSUFFICIENT_CASH_FLOWS = "INSUFFICIENT_CASH_FLOWS"

#: Not `UNIQUE`. Sampling can prove a second root exists and cannot prove one
#: does not, and a name that blurred the two would be the whole defect again.
NO_EVIDENCE_OF_NON_UNIQUENESS = "NO_EVIDENCE_OF_NON_UNIQUENESS"


@dataclass(frozen=True)
class Certification:
    """What was checked, against which contract, and what it found."""

    verdict: str
    contract_version: str = CONTRACT_VERSION
    solver: str = SOLVER
    tolerance: float = ROOT_TOLERANCE
    roots_found: Sequence[float] = field(default_factory=tuple)
    detail: str = ""

    @property
    def reportable(self) -> bool:
        """Whether a number may be published for this series."""
        return self.verdict == RATE

    def to_json(self) -> dict:
        return {"verdict": self.verdict,
                "contract_version": self.contract_version,
                "solver": self.solver, "tolerance": self.tolerance,
                "roots_found": [float(r) for r in self.roots_found],
                "detail": self.detail}


def npv(amounts: Sequence[float], sessions: Sequence[float],
        horizon: float, terminal: float, rate: float) -> float:
    """The engine's own present-value form, restated once.

    Written here rather than imported because the question is whether two
    independent statements of the same equation agree. Importing the solver's
    inner function would compare it against itself.
    """
    total = 0.0
    for amount, session in zip(amounts, sessions):
        total += amount * (1.0 + rate) ** (horizon - session)
    return total - terminal


def admissible_roots(amounts: Sequence[float], sessions: Sequence[float],
                     horizon: float, terminal: float, *,
                     lowest: float = -0.99, highest: float = 100.0,
                     steps: int = 20_000) -> Sequence[float]:
    """Every rate above -1 where the present value crosses zero, by sampling.

    A grid, and its limits are stated rather than hidden: a root inside a
    narrower interval than the step size is missed, and this cannot see beyond
    `highest`. What it is used for survives that — finding a *second* root is
    positive evidence, and missing one only ever makes this report less
    certainty than the truth, never more.
    """
    found = []
    width = (highest - lowest) / steps
    previous_rate = lowest
    previous = npv(amounts, sessions, horizon, terminal, previous_rate)
    for step in range(1, steps + 1):
        rate = lowest + step * width
        value = npv(amounts, sessions, horizon, terminal, rate)
        if abs(value) < ROOT_TOLERANCE:
            found.append(rate)
        elif previous * value < 0:
            low, high = previous_rate, rate
            for _ in range(80):
                mid = (low + high) / 2.0
                if npv(amounts, sessions, horizon, terminal, low) * \
                        npv(amounts, sessions, horizon, terminal, mid) <= 0:
                    high = mid
                else:
                    low = mid
            found.append((low + high) / 2.0)
        previous_rate, previous = rate, value

    # Collapse near-duplicates: a crossing detected twice is one root.
    distinct = []
    for rate in sorted(found):
        if not distinct or abs(rate - distinct[-1]) > 1e-6:
            distinct.append(rate)
    return tuple(distinct)


def certify(amounts: Sequence[float], sessions: Sequence[float],
            horizon: float, terminal: float,
            solver_result: Optional[float]) -> Certification:
    """Whether the solver's answer is one the contract permits.

    Four outcomes, and the one that matters is `NON_UNIQUE`: a returned number
    that satisfies the equation is still not reportable when a second rate
    satisfies it too. That is the case the solver's own uniqueness argument
    assumes away — Descartes' rule gives one positive root for a series of
    contributions plus a terminal value, and nothing checks that the series
    has that shape.
    """
    signs = {amount > 0 for amount in amounts if amount != 0}
    if terminal <= 0 or not amounts or len(signs) == 0:
        return Certification(
            verdict=INSUFFICIENT_CASH_FLOWS,
            detail="no flows, or no terminal value to recover")

    roots = admissible_roots(amounts, sessions, horizon, terminal)

    if len(roots) > 1:
        return Certification(
            verdict=NON_UNIQUE, roots_found=roots,
            detail=(f"{len(roots)} admissible rates satisfy this series; "
                    "publishing any of them would report a number the data "
                    "does not determine"))

    if not roots:
        return Certification(
            verdict=NO_SOLUTION,
            detail="no rate above -1 sets the present value to zero")

    if solver_result is None:
        return Certification(
            verdict=NO_SOLUTION, roots_found=roots,
            detail="a root exists and the solver did not find it")

    if abs(solver_result - roots[0]) > 1e-6:
        return Certification(
            verdict=NON_UNIQUE, roots_found=tuple(roots) + (solver_result,),
            detail="the solver returned a rate this scan did not find")

    return Certification(verdict=RATE, roots_found=roots,
                         detail=NO_EVIDENCE_OF_NON_UNIQUENESS)


def conformance_of(amounts, sessions, horizon, terminal, solver_result) -> dict:
    """A record fit to hang off a result in the UI.

    Carries the contract version, the solver, and the tolerance, so a figure
    can say which formal reporting contract its number was checked against
    rather than merely that somebody once checked something.
    """
    certification = certify(amounts, sessions, horizon, terminal,
                            solver_result)
    return {**certification.to_json(),
            "solver_returned": (None if solver_result is None
                                else float(solver_result))}
