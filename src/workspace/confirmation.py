"""The confirmation screen, prepared as data.

The one question this screen answers is **"did Quantify understand my plan?"**
Everything else is subordinate to that, including the machinery that makes the
answer trustworthy.

The shape follows what the corpus measured rather than what the architecture
finds interesting:

    69.3% of complete descriptions have nothing left to ask   -> one screen
    30.7% need one or two focused questions                   -> ask, then confirm
     0.0% need three or more

So the majority must not be walked through a wizard designed for the minority.
A stated field is shown as a count that expands; an *inferred* one is the thing
a user has to look at, because it is the only category where the system decided
something on their behalf.

This module computes; the template arranges. A page-level test asserts the
template contains no semantic logic, because a screen that recalculates what the
compiler already decided is a second implementation of the same rules and the
copy in the template is the one that drifts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..mission.scenario import UNSIMULATED

#: What each account type means to the engine. Account type now decides tax
#: treatment, contribution limits, withdrawal constraints and benchmark
#: comparability, so "Roth IRA" needs to say what it buys the user and what it
#: does not — a one-line inference card cannot carry that.
ACCOUNT_CONTEXT: Mapping[str, Dict[str, Any]] = {
    "TAXABLE": {
        "label": "Taxable brokerage account",
        "modelled": ("dividends and gains are taxable in the year realized",
                     "no contribution limit",
                     "no withdrawal restriction"),
        "not_modelled": ("specific federal or state tax rates",
                         "long- versus short-term gain treatment",
                         "wash-sale rules"),
    },
    "ROTH": {
        "label": "Roth IRA",
        "modelled": ("qualified growth is not taxed",
                     "annual contribution limit applies",
                     "withdrawal restrictions before retirement age"),
        "not_modelled": ("income-based contribution phase-out",
                         "early-withdrawal exceptions",
                         "state-specific treatment"),
    },
    "ROTH_401K": {
        "label": "Roth 401(k)",
        "modelled": ("qualified growth is not taxed",
                     "employee deferral limit applies",
                     "withdrawal restrictions before retirement age"),
        "not_modelled": ("employer match", "plan-specific rules",
                         "state-specific treatment"),
    },
    "TRADITIONAL_IRA": {
        "label": "Traditional IRA",
        "modelled": ("growth is tax-deferred",
                     "annual contribution limit applies",
                     "withdrawals are taxed as income"),
        "not_modelled": ("deduction eligibility", "required minimum distributions",
                         "state-specific treatment"),
    },
    "TRADITIONAL_401K": {
        "label": "401(k)",
        "modelled": ("growth is tax-deferred",
                     "employee deferral limit applies",
                     "withdrawals are taxed as income"),
        "not_modelled": ("employer match", "required minimum distributions",
                         "plan-specific rules"),
    },
}

#: Closed vocabularies, so a question with known answers offers them instead of
#: a free-text box. A text field where the options are finite invites a phrasing
#: the compiler then has to re-read, and re-reading is where meaning is lost.
CHOICES: Mapping[str, Sequence[Dict[str, str]]] = {
    "account_type": (
        {"value": "TAXABLE", "label": "Taxable brokerage account"},
        {"value": "TRADITIONAL_IRA", "label": "Traditional IRA"},
        {"value": "ROTH", "label": "Roth IRA"},
        {"value": "TRADITIONAL_401K", "label": "401(k)"},
        {"value": "ROTH_401K", "label": "Roth 401(k)"},
    ),
    "trigger_semantics": (
        {"value": "persistent_condition",
         "label": "Every day the condition holds"},
        {"value": "crossing_event",
         "label": "Only on the day it first becomes true"},
    ),
    "funding_source": (
        {"value": "contribution", "label": "Out of the regular contribution"},
        {"value": "additional_cash", "label": "Additional money on top"},
    ),
    "weighting": (
        {"value": "equal_weight_at_purchase",
         "label": "Equal dollars at each purchase"},
        {"value": "equal_weight_maintained",
         "label": "Keep the positions equal over time"},
    ),
    "dividends": (
        {"value": "reinvested", "label": "Reinvest them"},
        {"value": "held_as_cash", "label": "Hold them as cash"},
    ),
    "contribution_day_rule": (
        {"value": "first_session_of_period",
         "label": "First trading day of the period"},
        {"value": "calendar_first_rolled_forward",
         "label": "First calendar day of the month"},
    ),
    "moving_average_kind": (
        {"value": "simple", "label": "Simple moving average"},
        {"value": "exponential", "label": "Exponential moving average"},
    ),
}

#: Accounts the compiler recognises as a *phrase* but cannot simulate. Routed
#: rather than refused: a donor-advised fund is a real thing a user has, and
#: "not supported" with no next step is a dead end.
UNSUPPORTED_ACCOUNTS = {
    "inherited ira": "Inherited IRAs have their own distribution schedule.",
    "daf": "Donor-advised funds have their own contribution and grant rules.",
    "retirement accounts": "Say which one — the tax treatment differs by account.",
    "cash savings": "Cash savings is where money comes from, not where it is "
                    "invested. Which account does it go into?",
}

#: Plain-language renderings for the summary line. Ordered as a reader checks
#: them: what, how much, how often, where, then the details.
_SUMMARY_ORDER = ("holdings", "amount", "cadence", "account", "funding",
                  "dividends", "execution", "trigger", "weighting")


@dataclass(frozen=True)
class Inference:
    field: str
    value: str
    label: str
    why: str
    confirmed: bool
    choices: Sequence[Dict[str, str]] = ()


@dataclass(frozen=True)
class Question:
    field: str
    question: str
    why_it_matters: str
    choices: Sequence[Dict[str, str]] = ()
    routing: str = ""
    """Where to send someone whose answer the engine cannot model."""


@dataclass(frozen=True)
class NotSimulated:
    field: str
    declared: str
    why: str


@dataclass
class ConfirmationView:
    """Everything the screen shows, decided here rather than in the template."""

    headline: str
    summary: Sequence[Dict[str, str]] = ()
    stated_count: int = 0
    stated_detail: Sequence[str] = ()
    inferences: Sequence[Inference] = ()
    questions: Sequence[Question] = ()
    conflicts: Sequence[Dict[str, Any]] = ()
    not_simulated: Sequence[NotSimulated] = ()
    account: Optional[Dict[str, Any]] = None
    over_limit: Optional[Dict[str, Any]] = None
    """A stated contribution the stated account does not permit.

    Blocking, and not a `conflict`: nothing here is ambiguous. Both facts were
    read correctly and the plan they describe cannot be executed, so the honest
    move is to refuse and let the user say which of the two they meant."""

    better_route: Optional[Dict[str, str]] = None
    defaults_ref: str = ""
    """Which versioned default set supplied the inferences. Dropped in the
    first draft of this screen — it is the difference between "we guessed" and
    "a published, versioned default decided this, and here is its id"."""

    can_run: bool = False
    can_save: bool = False

    @property
    def path(self) -> str:
        """Which of the two flows this description is on.

        Named rather than inferred from counts in the template, so the layout
        cannot disagree with the decision.
        """
        if self.conflicts or self.over_limit:
            return "BLOCKED"
        if self.questions:
            return "CLARIFY"
        return "FAST"

    @property
    def question_count(self) -> int:
        return len(self.questions)

    def to_json(self) -> Dict[str, Any]:
        return {
            "headline": self.headline, "path": self.path,
            "summary": [dict(s) for s in self.summary],
            "stated_count": self.stated_count,
            "inferences": [i.__dict__ for i in self.inferences],
            "questions": [q.__dict__ for q in self.questions],
            "conflicts": [dict(c) for c in self.conflicts],
            "not_simulated": [n.__dict__ for n in self.not_simulated],
            "account": self.account, "over_limit": self.over_limit,
            "defaults_ref": self.defaults_ref,
            "can_run": self.can_run, "can_save": self.can_save,
        }


_CADENCE_LABEL = {
    "monthly": "every month", "quarterly": "every quarter",
    "annual": "every year", "weekly": "every week",
    "biweekly": "every other week", "payroll": "every payday",
    "daily": "every day", "once": "as a lump sum",
}
_DAY_LABEL = {
    "first_session_of_period": "first eligible trading session",
    "calendar_first_rolled_forward": "first calendar day, rolled forward",
}
_FUNDING_LABEL = {
    "contribution": "included in the regular contribution",
    "additional_cash": "additional money on top of the contribution",
}
_DIVIDEND_LABEL = {"reinvested": "reinvested",
                   "held_as_cash": "held as cash"}
_WEIGHTING_LABEL = {
    "equal_weight_at_purchase": "equal dollars at each purchase",
    "equal_weight_maintained": "positions kept equal over time",
}
_TRIGGER_LABEL = {
    "persistent_condition": "buys on every day the condition holds",
    "crossing_event": "buys only on the day the condition first becomes true",
}

FIELD_LABEL = {
    "account_type": "Account", "trigger_semantics": "When the rule fires",
    "funding_source": "Where the extra money comes from",
    "weighting": "How the holdings are weighted",
    "dividends": "Dividends", "contribution_day_rule": "Contribution date",
    "moving_average_kind": "Moving average", "cadence": "How often",
    "amount": "How much", "execution_timing": "When orders execute",
    "benchmark_set": "What to compare against",
    "starting_capital": "Starting capital",
}


def _summary_rows(scenario) -> List[Dict[str, str]]:
    """The compiled plan in plain language, in the order a reader checks it."""
    if scenario is None:
        return []
    flows = scenario.flow_schedule
    holdings = scenario.holdings_policy
    allocation = scenario.allocation_rule

    rows: List[Dict[str, str]] = []

    def add(key: str, label: str, value: Optional[str]) -> None:
        if value:
            rows.append({"key": key, "label": label, "value": value})

    assets = ", ".join(allocation.assets) or "not yet identified"
    add("holdings", "Invest in", assets)
    if flows.amount:
        add("amount", "Amount", f"${flows.amount:,.0f}")
    add("cadence", "How often", _CADENCE_LABEL.get(flows.cadence, flows.cadence))
    account = ACCOUNT_CONTEXT.get(scenario.tax_treatment)
    add("account", "Account", account["label"] if account else None)
    add("funding", "Funding", _FUNDING_LABEL.get(flows.funding_source))
    add("dividends", "Dividends", _DIVIDEND_LABEL.get(holdings.dividend_policy))
    add("execution", "Execution", _DAY_LABEL.get(flows.day_rule))
    if len(allocation.assets) > 1:
        add("weighting", "Weighting", _WEIGHTING_LABEL.get(allocation.weighting))
    for step in scenario.event_program:
        if isinstance(step, dict) and step.get("semantics"):
            add("trigger", "Rule", _TRIGGER_LABEL.get(step["semantics"]))
    if not holdings.sells_allowed:
        rows.append({"key": "selling", "label": "Selling", "value": "never"})
    return sorted(rows, key=lambda r: _SUMMARY_ORDER.index(r["key"])
                  if r["key"] in _SUMMARY_ORDER else len(_SUMMARY_ORDER))


def _routing_for(question_field: str, text: str) -> str:
    """A next step for an account the engine cannot model.

    "Not supported" with nowhere to go is a dead end, and a donor-advised fund
    is a real thing a user has.
    """
    if question_field != "account_type":
        return ""
    lowered = text.lower()
    for phrase, note in UNSUPPORTED_ACCOUNTS.items():
        if phrase in lowered:
            return note
    return ""


#: Contributions per year, by declared cadence. A cadence absent from this map
#: yields no annual figure and therefore no limit check — an unknown cadence
#: must not be guessed at, because guessing low would clear a plan that is over
#: the limit and guessing high would refuse one that is not.
_PER_YEAR = {"monthly": 12, "quarterly": 4, "annual": 1, "yearly": 1,
             "weekly": 52, "biweekly": 26, "payroll": 26, "once": 1,
             "one_off": 1}


def _annual_contribution(schedule) -> Optional[float]:
    per_year = _PER_YEAR.get(getattr(schedule, "cadence", ""))
    if per_year is None or not getattr(schedule, "amount", 0):
        return None
    return float(schedule.amount) * per_year


def _over_limit(scenario) -> Optional[Dict[str, Any]]:
    """Whether the stated contribution exceeds what the stated account permits.

    Checked here, before the run. Discovered afterwards it is a number the user
    has already been shown and already believes.
    """
    import datetime as _dt

    from ..runtime import AccountRuntime
    from ..runtime.account_limits import LimitState
    from .account_support import LABELS
    from .environment import ACCOUNT_KINDS

    kind = ACCOUNT_KINDS.get(getattr(scenario, "tax_treatment", ""))
    annual = _annual_contribution(getattr(scenario, "flow_schedule", None))
    if kind is None or annual is None:
        return None

    year = _dt.date.today().year
    decision = AccountRuntime(name=f"account/{kind.value.lower()}", version=1,
                              account_kind=kind).cap_contribution(annual, year=year)
    if decision.within_limit:
        return None

    limit = decision.limit
    return {
        "requested": decision.requested,
        "permitted": decision.permitted,
        "refused": decision.refused,
        "year": year,
        "account_label": LABELS.get(scenario.tax_treatment,
                                    scenario.tax_treatment),
        "detail": (
            f"This plan contributes ${decision.requested:,.0f} a year, and the "
            f"{year} limit for this account is ${decision.permitted:,.0f}. "
            f"It is over by ${decision.refused:,.0f}."),
        # The figure doing the refusing is named, with its own reliability. A
        # plan refused by an unchecked number should say so in the same breath.
        "limit_is_verified": limit.state is LimitState.VERIFIED,
        "caveat": limit.why_not_enforced,
        "choices": [
            {"value": "reduce",
             "label": f"Contribute ${decision.permitted:,.0f} a year instead"},
            {"value": "change_account",
             "label": "This is a different kind of account"},
            {"value": "split",
             "label": "Some of it goes somewhere else"},
        ],
    }


def build(result, *, text: str = "") -> ConfirmationView:
    """Prepare the confirmation screen from a compiled result."""
    scenario = result.scenario
    summary = _summary_rows(scenario)

    inferences = [
        Inference(field=i.field, value=i.value,
                  label=FIELD_LABEL.get(i.field, i.field.replace("_", " ")),
                  why=i.why, confirmed=i.confirmed,
                  choices=CHOICES.get(i.field, ()))
        for i in result.inferred
    ]
    questions = [
        Question(field=u.field, question=u.question,
                 why_it_matters=u.why_it_matters,
                 choices=CHOICES.get(u.field, ()),
                 routing=_routing_for(u.field, text))
        for u in result.unresolved
    ]
    conflicts = [
        {"detail": c.detail, "between": list(c.between), "resolved": c.resolved}
        for c in result.contradictions
    ]

    # Shown *before* the run, not only in the modelling scope afterwards. A
    # choice the user made and the engine cannot honour is something they need
    # to know while they can still change their mind about running it.
    not_simulated: List[NotSimulated] = []
    if scenario is not None:
        declared = {"dividend_policy": scenario.holdings_policy.dividend_policy}
        for name, value in declared.items():
            if name in UNSIMULATED:
                not_simulated.append(NotSimulated(
                    field=name, declared=value, why=UNSIMULATED[name]))

    account = None
    if scenario is not None:
        from .account_support import support_for

        context = ACCOUNT_CONTEXT.get(scenario.tax_treatment)
        if context:
            # Three claims, never one. "Account matched" read as "account
            # modelled completely" is the assumption a single badge invites.
            account = {**context, "value": scenario.tax_treatment,
                       "support": support_for(scenario.tax_treatment).to_json()}

    over_limit = _over_limit(scenario) if scenario is not None else None

    return ConfirmationView(
        headline=("This plan contributes more than the account allows"
                  if over_limit else
                  "Here is what we understood" if not conflicts
                  else "These instructions conflict"),
        summary=summary,
        stated_count=len(result.stated),
        stated_detail=tuple(result.stated),
        inferences=tuple(inferences),
        questions=tuple(questions),
        conflicts=tuple(conflicts),
        not_simulated=tuple(not_simulated),
        account=account,
        over_limit=over_limit,
        better_route=(result.confirmation().get("a_better_route")),
        defaults_ref=result.defaults_ref,
        # An over-limit plan cannot run. Showing the refusal and leaving the
        # button live would make the warning advisory, and the figure it
        # produced would be one the account does not permit.
        can_run=result.can_simulate and over_limit is None,
        can_save=result.can_save,
    )
