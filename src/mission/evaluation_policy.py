"""How a figure is measured, as against what the plan does.

Two sets of conventions and they are not the same set, which is why they are
two objects:

    StrategySpec.conventions   decide what the plan *does* — which calendar a
                               contribution lands on, how a non-session date
                               rolls, what frequency the cadence is, when a
                               trade settles, what currency the amounts are

    EvaluationPolicy           decides how the result is *measured* — how a
                               rate is compounded, what a year is when a figure
                               is annualised, and the date the valuation is
                               made as of

The split matters for the identity. Two people running the same strategy under
different measurement conventions are running one strategy and getting two
figures, and a `strategy_hash` that moved with the annualisation basis would
say they were different plans. So the spec hash covers the first set and the
result cites the second separately — which is also what lets a comparison say
"same plan, different measurement" instead of just "different".

**The evaluator infers none of these.** Every value is supplied. A convention
the evaluator picked for itself is one nobody can check a figure against, and
`evaluation_date` is the sharp case: QuantLib keeps a global
`Settings.instance().evaluationDate` that defaults to today, so an evaluator
reading it would produce a result that depended on when it ran rather than on
what it ran on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

#: The shape of this contract, in the hashed body for the same reason
#: `SPEC_VERSION` is.
POLICY_VERSION = "quantify-evaluation-policy@1"


@dataclass(frozen=True)
class EvaluationPolicy:
    """The measurement conventions a run was computed under."""

    compounding: str
    annualisation: str
    sessions_per_year: str
    evaluation_date: str
    data_policy: str
    """Which data this run was permitted to reach — `SYNTHETIC_ONLY` or a
    licensed tier. Not a QuantLib convention, and it belongs here anyway: it
    changes what a figure is about, and a result that did not say which would
    let a synthetic number be read as a market one."""

    models_settlement: bool = False
    """Whether the settlement lag the specification names is actually applied.

    False, and stated. This build simulates on session closes and does not move
    cash a day later. A convention named but not honoured is only safe when the
    record says so — unnamed and unhonoured is how somebody assumes it was
    handled, which is the `dividend_policy` failure exactly."""

    version: str = POLICY_VERSION

    def to_json(self) -> Dict[str, Any]:
        return {"compounding": self.compounding,
                "annualisation": self.annualisation,
                "sessions_per_year": self.sessions_per_year,
                "evaluation_date": self.evaluation_date,
                "data_policy": self.data_policy,
                "models_settlement": self.models_settlement,
                "version": self.version}


def declared_policy(*, data_policy: str, as_of: Optional[Any] = None
                    ) -> EvaluationPolicy:
    """This build's measurement conventions, from the one place that names them.

    `as_of` is passed in rather than read from QuantLib's global settings. A
    result whose evaluation date came from a library default would change with
    the clock, and two runs of one specification would stop agreeing overnight
    — the reproducibility defect, arriving through a convention nobody thought
    of as an input.
    """
    from .conventions import (ANNUALISATION, COMPOUNDING, evaluation_date)

    return EvaluationPolicy(
        compounding=COMPOUNDING,
        annualisation=ANNUALISATION.name,
        sessions_per_year=str(ANNUALISATION.per_year),
        evaluation_date=evaluation_date(as_of),
        data_policy=data_policy,
        models_settlement=False)
