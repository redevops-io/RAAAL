"""Compiler defaults, as a versioned artifact rather than as prompt text.

A default buried in a system prompt is the purest form of the failure this
platform exists to prevent: a choice that moves a published number, made by
nobody, recorded nowhere, and changeable without a version bump. Every erratum in
the library was one of those.

So the defaults are an artifact with an id, a hash and a rationale per entry:

    compiler-defaults/us-equity-scenario@1

`why` is not documentation. A user confirming nine inferences needs to know which
ones move the number, and an inference presented without its consequence gets
waved through.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Default:
    field: str
    value: str
    why: str
    alternatives: Sequence[str] = ()
    changes_result: bool = True
    """False only where the choice is presentational. Almost nothing here is."""

    def to_json(self) -> Dict[str, Any]:
        return {"field": self.field, "value": self.value, "why": self.why,
                "alternatives": list(self.alternatives),
                "changes_result": self.changes_result}


@dataclass(frozen=True)
class DefaultSet:
    name: str
    version: int
    defaults: Mapping[str, Default]

    @property
    def artifact_id(self) -> str:
        return f"compiler-defaults/{self.name}@{self.version}"

    def get(self, field: str) -> Optional[Default]:
        return self.defaults.get(field)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps({k: v.to_json() for k, v in sorted(self.defaults.items())},
                       sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_json(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "defaults": {k: v.to_json() for k, v in sorted(self.defaults.items())},
        }


def _d(field, value, why, alternatives=()) -> Default:
    return Default(field=field, value=value, why=why, alternatives=alternatives)


US_EQUITY_SCENARIO = DefaultSet(
    name="us-equity-scenario",
    version=1,
    defaults={d.field: d for d in (
        _d("moving_average_kind", "simple",
           "'200DMA' most commonly means a simple average; an exponential one "
           "reacts faster and would trigger on different days.",
           ("simple", "exponential")),
        _d("signal_price", "close",
           "A close is observable at the end of the session; an intraday touch "
           "is not knowable until after it has happened.",
           ("close", "intraday_low")),
        _d("execution_timing", "next_session_open",
           "A signal formed from today's close cannot be traded at today's "
           "close. Filling on the price that generated the signal is the single "
           "most common way a backtest overstates a rule.",
           ("next_session_open", "next_session_close")),
        _d("contribution_day_rule", "first_session_of_period",
           "'Every month' does not name a day. The first trading session is the "
           "usual reading of a paycheque landing at the start of the month.",
           ("first_session_of_period", "last_session_of_period",
            "calendar_first_rolled_forward")),
        _d("dividends", "reinvested",
           "Most brokerage accounts reinvest by default, and holding dividends "
           "as cash produces a materially lower result over long horizons.",
           ("reinvested", "held_as_cash")),
        _d("weighting", "equal_weight_at_purchase",
           "'Equally' most often means equal dollars at the moment of buying. "
           "Maintaining equal weight thereafter requires selling what rose.",
           ("equal_weight_at_purchase", "equal_weight_maintained")),
        _d("cash_policy", "idle",
           "Uninvested cash earning nothing is the conservative reading and the "
           "one that does not credit the plan with a yield it never chose.",
           ("idle", "money_market")),
        _d("tax_treatment", "NONE_APPLIED",
           "Applying tax requires a jurisdiction, account type and lot method "
           "that nobody has stated. Reporting pre-tax and saying so is honest; "
           "assuming long-term capital gains is not.",
           ("NONE_APPLIED",)),
        _d("fractional_shares", "allowed",
           "'Buy $2,000 of' implies a notional order. Whole-share rounding "
           "leaves residual cash and changes the path.",
           ("allowed", "whole_shares_only")),
    )},
)

#: The set a scenario compiles against unless one is named. Pinned, not "latest":
#: recompiling the same words a year later must produce the same scenario or the
#: artifact was never immutable.
DEFAULT_SET = US_EQUITY_SCENARIO
