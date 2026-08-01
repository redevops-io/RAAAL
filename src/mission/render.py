"""A Mission, said back in words.

Three purposes, declared on the output rather than inferred from it:

    SPECIFICATION   complete and unambiguous; expected to round-trip exactly
    SUMMARY         ordinary prose for a person; may omit detail
    EXPLANATION     why the plan is what it is; not a specification at all

The declaration is the point. A concise summary that looks like prose is exactly
what someone will paste back in later expecting identical behaviour, and without
a stated purpose nothing stops a UI blurb becoming an authoritative export. Only
`SPECIFICATION` claims losslessness, and only it is held to it.

The renderer emits phrasings the recognisers actually match. That is not a
convenience: a specification the compiler cannot read back is not a
specification, and writing it in prettier language nobody parses would make the
round-trip benchmark pass on text no user could ever have written.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class Purpose(str, Enum):
    SPECIFICATION = "SPECIFICATION"
    SUMMARY = "SUMMARY"
    EXPLANATION = "EXPLANATION"

    @property
    def claims_lossless(self) -> bool:
        return self is Purpose.SPECIFICATION


@dataclass(frozen=True)
class Rendered:
    purpose: Purpose
    text: str
    #: Fields the renderer knowingly left out. Empty for a SPECIFICATION by
    #: construction — anything omitted there is a defect, not a choice.
    omitted: Sequence[str] = ()

    def to_json(self) -> Dict[str, Any]:
        return {"purpose": self.purpose.value, "text": self.text,
                "omitted": list(self.omitted),
                "claims_lossless": self.purpose.claims_lossless}


_CADENCE_WORDS = {
    "monthly": "every month", "quarterly": "every quarter",
    "annual": "every year", "weekly": "every week",
    "biweekly": "every other week", "payroll": "every payday",
    "daily": "every day", "once": "as a lump sum",
}
_DAY_RULE_WORDS = {
    "first_session_of_period": "on the first trading day of the period",
    "calendar_first_rolled_forward": "on the first calendar day of the month",
}
_DIVIDEND_WORDS = {
    "reinvested": "reinvesting the dividends",
    "held_as_cash": "holding the dividends as cash",
}
_WEIGHTING_WORDS = {
    "equal_weight_at_purchase": "buying equal dollars at each purchase",
    "equal_weight_maintained": "rebalancing them back to equal weights",
}
_TRIGGER_WORDS = {
    "persistent_condition": "Whenever SPY is below its {estimator} 200 day "
                            "moving average",
    "crossing_event": "On the day SPY crosses below its {estimator} 200 day "
                      "moving average",
}


def _tidy(text: str) -> str:
    """Collapse the gap an omitted estimator leaves behind."""
    return " ".join(text.split())
_FUNDING_WORDS = {
    "contribution": "out of that contribution",
    "additional_cash": "with additional cash",
}
_TAX_WORDS = {
    "NONE_APPLIED": "", "ROTH": "in my Roth IRA",
    "TRADITIONAL": "in my traditional IRA", "TAXABLE": "in my taxable account",
}


def _assets(assets: Sequence[str]) -> str:
    assets = list(assets)
    if not assets:
        return "the instruments named"
    if len(assets) == 1:
        return assets[0]
    if len(assets) == 2:
        return f"{assets[0]} and {assets[1]}"
    return ", ".join(assets[:-1]) + f" and {assets[-1]}"


def _estimator(scenario) -> str:
    for step in scenario.event_program:
        if isinstance(step, dict) and step.get("estimator"):
            return step["estimator"]
    return "simple"


def _trigger(scenario) -> Optional[str]:
    for step in scenario.event_program:
        if isinstance(step, dict) and step.get("semantics"):
            return step["semantics"]
    return None


def specification(scenario) -> Rendered:
    """Every *stated* field, in language the recognisers read.

    Clause order follows the compiler's own reading order rather than what
    sounds best. The benchmark's job is to prove nothing is lost, and prose that
    reads well while dropping the funding source would prove the opposite.

    Inferred fields are deliberately left unsaid. A specification that writes
    them out reproduces the values and destroys the provenance: what the system
    supplied comes back as something the user stated, and the confirmation
    screen then asks them to confirm nothing. The round-trip benchmark caught
    exactly that — identical values, a different `content_hash`, because
    `weighting` moved from inferred to stated.

    Leaving them unsaid is what makes the round trip faithful: they are inferred
    again, to the same value, from the same versioned default set.
    """
    inferred = {i.field for i in scenario.provenance.inferred}
    unresolved = {u.field for u in scenario.provenance.unresolved}
    #: Never say what the user did not. An inferred value restated becomes a
    #: stated one; an unresolved question answered becomes a decision they never
    #: made. Both reproduce the values and destroy the provenance.
    unsaid = inferred | unresolved
    flows = scenario.flow_schedule
    holdings = scenario.holdings_policy
    allocation = scenario.allocation_rule

    clauses: List[str] = [
        f"I put ${flows.amount:,.0f} into {_assets(allocation.assets)}",
    ]
    if "cadence" not in unsaid:
        clauses.append(_CADENCE_WORDS.get(flows.cadence, flows.cadence))
    account = _TAX_WORDS.get(scenario.tax_treatment, "")
    if account:
        clauses.append(account)
    if "contribution_day_rule" not in unsaid:
        clauses.append(_DAY_RULE_WORDS.get(flows.day_rule, flows.day_rule))
    if "dividends" not in unsaid:
        clauses.append(_DIVIDEND_WORDS.get(holdings.dividend_policy,
                                           holdings.dividend_policy))

    # A weighting the user stated is written out whatever the holding count.
    # Guarding on `len(assets) > 1` dropped a stated "keep them at equal
    # weights" on a single holding, which then re-compiled to the default — the
    # rule silently changed on a round trip. The guard belongs on *inferred*
    # weightings, where the clause would be noise the compiler ignores anyway.
    if "weighting" not in unsaid:
        clauses.append(_WEIGHTING_WORDS.get(allocation.weighting,
                                            allocation.weighting))

    text = ", ".join(c for c in clauses if c)
    if not holdings.sells_allowed:
        text += ", and I never sell"
    text += "."

    trigger = _trigger(scenario)
    if not trigger and "trigger_semantics" in unresolved:
        # The description mentioned a market condition and never said how it
        # behaves, so the compiler asked and built no event program. Dropping
        # the mention makes the regenerated text stop asking — the open question
        # is answered by omission, which is the one outcome a specification must
        # never produce. Said back without semantics, so it is asked again.
        # The funding source travels with the mention, not with the trigger.
        # It was only written inside the resolved-trigger branch, so a stated
        # "with additional cash" was dropped whenever the semantics were still
        # open — and came back as the contribution default, which invests a
        # different amount of money.
        funding = ("" if "funding_source" in inferred
                   else _FUNDING_WORDS.get(flows.funding_source, ""))
        text += (" I want to buy more when it drops"
                 + (f" {funding}" if funding else "") + ".")
    if trigger:
        # The estimator is only named when the user named it. Writing an
        # inferred "simple" into the sentence turns a default into a decision.
        estimator = ("" if "moving_average_kind" in unsaid
                     else _estimator(scenario))
        opening = _TRIGGER_WORDS[trigger].format(estimator=estimator)
        funding = ("" if "funding_source" in inferred
                   else _FUNDING_WORDS.get(flows.funding_source, ""))
        text += (f" {opening} I buy more of {_assets(allocation.assets)}"
                 + (f" {funding}" if funding else "") + ".")
    # A stated funding source with neither a resolved trigger nor an open one
    # has no clause to live in, and was silently dropped. It gets its own
    # sentence rather than being lost: it decides how much money the plan
    # invests, which is not a detail a specification may omit.
    if (not trigger and "trigger_semantics" not in unresolved
            and "funding_source" not in inferred
            and flows.funding_source != "contribution"):
        text += f" I fund that {_FUNDING_WORDS[flows.funding_source]}."

    return Rendered(purpose=Purpose.SPECIFICATION, text=_tidy(text))


def summary(scenario) -> Rendered:
    """Ordinary prose. Shorter, and honest about what it drops.

    Deliberately omits the details a person reading a plan card does not want,
    and names them, so nothing downstream can mistake this for a complete
    statement of the plan.
    """
    flows = scenario.flow_schedule
    allocation = scenario.allocation_rule
    cadence = _CADENCE_WORDS.get(flows.cadence, flows.cadence)
    text = (f"${flows.amount:,.0f} into {_assets(allocation.assets)} {cadence}"
            + (", never selling." if not scenario.holdings_policy.sells_allowed
               else "."))
    omitted = ["day_rule", "dividend_policy", "tax_treatment"]
    if len(allocation.assets) > 1:
        omitted.append("weighting")
    if _trigger(scenario):
        omitted += ["trigger_semantics", "moving_average_estimator",
                    "funding_source"]
    return Rendered(purpose=Purpose.SUMMARY, text=text, omitted=tuple(omitted))


def explanation(scenario) -> Rendered:
    """Why the plan is what it is. Never a specification, and says so."""
    flows = scenario.flow_schedule
    lines = [
        f"This plan contributes ${flows.amount:,.0f} {_CADENCE_WORDS.get(flows.cadence, flows.cadence)}.",
        f"Dividends are {scenario.holdings_policy.dividend_policy.replace('_', ' ')}, "
        "which changes how the position compounds over time.",
    ]
    if _trigger(scenario):
        lines.append(
            f"The buying rule fires as a {_trigger(scenario).replace('_', ' ')}, "
            f"funded {flows.funding_source.replace('_', ' ')} — taken from the "
            "contribution the plan invests the same total, and as additional "
            "cash it invests more.")
    return Rendered(purpose=Purpose.EXPLANATION, text=" ".join(lines),
                    omitted=("this is not a specification and will not "
                             "round-trip",))


def render(scenario, purpose: Purpose = Purpose.SPECIFICATION) -> Rendered:
    return {Purpose.SPECIFICATION: specification,
            Purpose.SUMMARY: summary,
            Purpose.EXPLANATION: explanation}[purpose](scenario)
