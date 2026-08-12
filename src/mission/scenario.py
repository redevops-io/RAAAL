"""ScenarioSpecification — the layer between what someone said and what runs.

Compiling an Intent straight into a Methodology would be a category error: the
personal event program is not an investment methodology. "Buy $2,000 of these six
names when SPY closes below its 200-day average, and never sell" contains a
market rule, a money path, and an evaluation choice, tangled together. Only the
first is a methodology; the second is nobody's research and the third is a
protocol.

    Intent
      ↓
    ScenarioSpecification        the whole thing, with provenance
      ↓
    Methodology + EvaluationProtocol + FlowSchedule

Keeping them separate is what lets the same rule be tested under someone else's
contributions without claiming the two people had the same financial outcome —
and it is what makes `extract_rule` a clean seam rather than a redaction pass.
"""
from __future__ import annotations

import re

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .spec import FlowSchedule, MISSION_SPEC_VERSION, Objective, Provenance


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


#: A leading article, stripped only where the registry has no opinion.
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.I)


def execution_subject(asset: str) -> str:
    """What a holding *is*, for the purpose of plan identity only.

    `VerifiedIntent` keeps what the person said and `canonical_form()` keeps
    what the plan holds — both stay verbatim. This is the third thing: the
    subject two requests must share before they may be called the same plan.

    Resolved phrases collapse to the registry's own identity, so `S&P 500`,
    `the S&P` and `SPX` are one subject; the resolver already does that and
    this reuses it rather than re-deciding. Unresolved phrases fall back to a
    canonical spelling with a leading article removed — safe *here* precisely
    because the phrase did not resolve, since anything the registry can name
    has already been collapsed above. Stripping articles globally would be
    wrong, because a real product name can contain one, and a real product
    name resolves.

    Never used for what executes. The first version of this put the
    canonicalisation in `canonical_form()`, which is consumed as data by
    `representation.py` and `to_json()` — so `SPY` became `spy`, the engine
    found no prices for it, and sixteen tests failed on missing signals. The
    lesson is in the name: this is the *identity* of a holding, not the
    holding.
    """
    try:
        from .resolver import resolve

        found = resolve(asset)
        concept = getattr(found, "concept_id", None)
        if concept:
            return f"concept:{concept}"
        instrument = getattr(found, "instrument", None)
        symbol = getattr(instrument, "symbol", None)
        if symbol:
            return f"instrument:{symbol}"
    except Exception:                                          # noqa: BLE001
        # A registry that will not load leaves every phrase on the fallback,
        # which keeps identity deterministic and article-insensitive — a
        # narrower answer than resolution gives, not a wrong one.
        pass
    return _LEADING_ARTICLE.sub("", " ".join(str(asset).split())).lower()


@dataclass(frozen=True)
class AllocationRule:
    """What the money buys, and how it is divided.

    `weighting` distinguishes the two readings of "equally" that a user will not:
    equal dollars at purchase never needs a sale, equal weight thereafter needs
    one on every drift. Conflating them is how "never sell" and "equally" end up
    silently contradicting each other.
    """

    assets: Sequence[str]
    weighting: str = "equal_weight_at_purchase"

    def canonical_form(self) -> Dict[str, Any]:
        """The holdings as written. Data, and consumed as such."""
        return {"assets": sorted(self.assets), "weighting": self.weighting}

    def execution_form(self) -> Dict[str, Any]:
        """The holdings as subjects. Identity, and consumed only as such."""
        return {"assets": sorted(execution_subject(a) for a in self.assets),
                "weighting": self.weighting}


@dataclass(frozen=True)
class HoldingsPolicy:
    sells_allowed: bool = True
    rebalancing_allowed: bool = True

    dividend_policy: str = "reinvested"
    """`reinvested` or `held_as_cash`. Part of the *rule*, not the money path:
    reinvesting compounds the position while holding as cash does not, and over
    a long horizon the two are materially different strategies.

    It reached the scenario only after the stability benchmark found it missing.
    The compiler recognised the phrase, the confirmation screen quoted it back
    under "you stated", and the compiled scenario contained no trace of it — so
    "reinvest the dividends" and "hold the dividends as cash" produced an
    identical `content_hash`. Recognised, confirmed, and unrepresented.

    See `UNSIMULATED` below: representing it is not the same as honouring it."""

    def canonical_form(self) -> Dict[str, Any]:
        return {"sells_allowed": self.sells_allowed,
                "rebalancing_allowed": self.rebalancing_allowed,
                "dividend_policy": self.dividend_policy}


#: Declared behaviours the engine records but does not simulate, and why. A
#: declaration with no realization is the failure this project exists to close,
#: so where one cannot be honoured yet it is named here rather than left to look
#: enforced. Every entry must appear in a result's modelling scope.
UNSIMULATED = {
    "event_program": (
        "The engine replays a flow schedule against a buy-and-hold program. "
        "The declared conditional rule — a trigger, its condition and the "
        "purchase it causes — is recorded, hashed into the methodology and "
        "shown on the page, and is not executed. A figure produced beside it "
        "is the figure for holding the instruments, not for following the "
        "rule, and reads as though the rule had been evaluated."
    ),
    "dividend_policy": (
        "The engine runs on price series only, so dividends are neither paid "
        "nor reinvested. The choice is recorded and distinguishes two "
        "strategies from each other; it does not yet change a simulated "
        "figure, and any result reading as though it did would be wrong."
    ),
}


@dataclass(frozen=True)
class BenchmarkSet:
    """Declared before results, generated by a published rule.

    `generated_by_rule` names a public policy artifact rather than describing one
    inline, so the set a user was shown can be reconstructed later — the same
    reason a run records its policy id rather than its verdict alone.
    """

    generated_by_rule: str
    members: Sequence[str] = ()
    ordering: str = "unordered"

    def canonical_form(self) -> Dict[str, Any]:
        return {"generated_by_rule": self.generated_by_rule,
                "members": sorted(self.members), "ordering": self.ordering}


@dataclass(frozen=True)
class ScenarioSpecification:
    """Everything the user described, separated into what each layer owns."""

    name: str
    version: int
    objective: Objective
    event_program: Sequence[Dict[str, Any]]
    flow_schedule: FlowSchedule
    allocation_rule: AllocationRule
    holdings_policy: HoldingsPolicy = field(default_factory=HoldingsPolicy)
    cash_policy_ref: str = ""
    benchmark_set: Optional[BenchmarkSet] = None
    provenance: Provenance = field(default_factory=Provenance)
    intent_ref: Optional[str] = None
    pending_template: Optional[str] = None
    """A life-event template that will supply the flows once filled in. While it
    is pending, an empty flow schedule is incomplete rather than contradictory —
    complaining that there is no money to invest would be describing the wrong
    problem, and would send the reader to fix something that is not broken."""

    tax_treatment: str = "NONE_APPLIED"
    spec_version: str = MISSION_SPEC_VERSION

    funding: Optional[Any] = None
    """How money arrives: a `funding.Scheduled` or a `funding.EventTriggered`.

    **The authority.** `cadence` and a conditional purchase rule were both
    answers to *when does money appear*, stored in different places, and the
    engine consumed only the first — so "buy $1,000 every time SPY crosses
    below its 200-day average" compiled to one contribution and returned a
    buy-and-hold figure.

    A sum type has nowhere to state both. `flow_schedule` is retained as the
    projection every benchmark, hash and comparability dimension already reads,
    and the compiler is the only thing that builds either: for an event-triggered
    policy it writes a schedule that states no cadence and no amount, so the two
    cannot disagree. `self_conflicts` refuses a scenario where they do.

    Optional because scenarios stored before this existed have none, and a
    default `Scheduled` reconstructed from the schedule would be this build
    asserting what an older one meant.
    """

    @property
    def is_event_funded(self) -> bool:
        from .funding import EventTriggered

        return isinstance(self.funding, EventTriggered)

    @property
    def concept_id(self) -> str:
        return f"scenario/{self.name}"

    @property
    def artifact_id(self) -> str:
        return f"scenario/{self.name}@{self.version}"

    def self_conflicts(self) -> List[str]:
        """Contradictions the structure exposes that prose hides.

        The compiler should already have caught these from the text. This catches
        them from the compiled form, which is the version that actually runs — and
        a check that only reads the input trusts the compiler to have been right.
        """
        conflicts: List[str] = []
        maintains_weight = self.allocation_rule.weighting == "equal_weight_maintained"

        # These two are one conflict seen from two sides, and a confirmation
        # screen listing it twice reads as carelessness. The selling form is
        # reported because it names the mechanism: what the plan would have to
        # do that the user forbade.
        if maintains_weight and not self.holdings_policy.sells_allowed:
            conflicts.append(
                "holdings_policy forbids selling while allocation_rule maintains "
                "equal weight; maintaining a weight requires selling what rose"
            )
        elif maintains_weight and not self.holdings_policy.rebalancing_allowed:
            conflicts.append(
                "allocation_rule maintains equal weight while rebalancing is "
                "forbidden"
            )
        if (self.flow_schedule.amount <= 0
                and self.flow_schedule.starting_capital <= 0
                and not self.pending_template
                # An event-funded plan states no cadence and no amount on the
                # schedule *by construction* — the money arrives when the
                # trigger fires, and how much is on the funding policy. Read
                # without this, it looks like a plan with no money at all.
                and not self.is_event_funded):
            conflicts.append(
                "no contributions and no starting capital: there is no money to "
                "invest, so every figure would be undefined rather than zero"
            )

        # The contradiction the sum type exists to prevent, checked on the
        # compiled form rather than trusted to the compiler.
        #
        # `funding` is the authority and `flow_schedule` is its projection. If a
        # scenario ever states both an event trigger and a cadence, two things
        # answer "when does money arrive" and the engine will consume one of
        # them — which is precisely the state that produced a buy-and-hold
        # figure under a conditional rule.
        if self.is_event_funded and (
                self.flow_schedule.amount > 0
                or self.flow_schedule.cadence not in ("", "event_triggered")):
            conflicts.append(
                "funding is event-triggered while flow_schedule also states a "
                f"cadence ({self.flow_schedule.cadence!r}) and an amount "
                f"({self.flow_schedule.amount}); money cannot arrive on a "
                "schedule and on a trigger, and whichever the engine reads "
                "would silently become the plan"
            )
        return conflicts

    @property
    def is_runnable(self) -> bool:
        return not self.self_conflicts()

    # ---- the split into the three artifacts that execute ------------------

    def methodology_part(self) -> Dict[str, Any]:
        """The market rule alone — impersonal, and the only part that could ever
        become public research."""
        return {
            "pipeline": [step.get("action", step.get("observe", "step"))
                         for step in self.event_program],
            "event_program": list(self.event_program),
            "allocation_rule": self.allocation_rule.canonical_form(),
            "holdings_policy": self.holdings_policy.canonical_form(),
        }

    def protocol_part(self) -> Dict[str, Any]:
        """How it is evaluated. Shared with the library's protocols by design."""
        return {"cash_policy_ref": self.cash_policy_ref,
                "tax_treatment": self.tax_treatment,
                "benchmark_set": (self.benchmark_set.canonical_form()
                                  if self.benchmark_set else None)}

    def flow_part(self) -> Dict[str, Any]:
        """The money path. Never public, and never part of the rule.

        The funding policy travels here because it *is* the money path — and
        because a stored plan that lost it would be reopened as a plan with no
        rule. The first version of this omitted it, and the reopened plan
        showed the Deployment 1 refusal: the page had no way to know the rule
        had ever executed, which is the read-path failure the whole ledger
        exists to make impossible.
        """
        body = dict(self.flow_schedule.canonical_form())
        if self.funding is not None:
            body["funding"] = self.funding.to_json()
        return body

    def execution_form(self) -> Dict[str, Any]:
        """`canonical_form()` with holdings replaced by their subjects.

        The one thing plan identity is computed from. Everything else is
        shared, so the two forms cannot drift apart in any respect except the
        one this exists to canonicalise.
        """
        form = self.canonical_form()
        form["methodology"] = {
            **form["methodology"],
            "allocation_rule": self.allocation_rule.execution_form(),
        }
        return form

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "objective": self.objective.value,
            "methodology": self.methodology_part(),
            "protocol": self.protocol_part(),
            "flows": self.flow_part(),
            "inferred": sorted(
                ({"field": i.field, "value": i.value}
                 for i in self.provenance.inferred),
                key=lambda d: d["field"],
            ),
        }

    def semantic_form(self) -> Dict[str, Any]:
        """Everything that can change a result, and nothing else.

        `content_hash` covers the compiled artifact. This covers the *inputs
        that decide it*, including several the canonical form does not carry:
        the funding policy, the watched instrument, the resolved identities and
        the time-window instruction.

        It exists because the preview and the save route compiled the same
        description under different inputs and persisted a different plan. The
        save path omitted `priceable`, so no funding policy could be built, and
        every stored plan was written unable to execute while the page showed
        one that ran. A digest over the compiled body alone did not notice:
        both bodies were internally consistent.

        Deliberately excludes timestamps, database ids, render order and
        display wording. A gate that tripped on those would be turned off.
        """
        provenance = self.provenance
        return {
            "held_assets": sorted(self.allocation_rule.assets),
            "weighting": self.allocation_rule.weighting,
            "funding": (self.funding.to_json()
                        if self.funding is not None else None),
            "flows": self.flow_schedule.canonical_form(),
            "event_program": list(self.event_program),
            "holdings_policy": self.holdings_policy.canonical_form(),
            "tax_treatment": self.tax_treatment,
            "asset_resolutions": sorted(
                (one.observed_phrase, one.chosen_instrument_id,
                 one.registry_digest)
                for one in (provenance.asset_resolutions or ())),
            "time_window": (provenance.time_window.to_json()
                            if getattr(provenance.time_window, "to_json", None)
                            else None),
            "exclusions": sorted(one.item for one in (provenance.excluded or ())),
            "amendments": sorted(
                (one.question_id, one.answer)
                for one in (provenance.amended or ())),
            "inferred": sorted(
                (one.field, one.value) for one in provenance.inferred),
            "benchmark_set": (self.benchmark_set.canonical_form()
                              if self.benchmark_set else None),
            "spec_version": self.spec_version,
        }

    @property
    def semantic_digest(self) -> str:
        return _hash(self.semantic_form())

    @property
    def content_hash(self) -> str:
        return _hash(self.canonical_form())

    @property
    def rule_hash(self) -> str:
        """Identity of the market rule alone.

        Two people running the same rule on different salaries share this and
        differ in `content_hash`, which is exactly the distinction the comparison
        classes turn on.
        """
        return _hash(self.methodology_part())

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "artifact_id": self.artifact_id,
            "concept_id": self.concept_id,
            "content_hash": self.content_hash,
            "rule_hash": self.rule_hash,
            "intent_ref": self.intent_ref,
            "provenance": self.provenance.to_json(),
            "self_conflicts": self.self_conflicts(),
            "is_runnable": self.is_runnable,
        }
