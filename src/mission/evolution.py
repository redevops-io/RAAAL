"""Three operations that look alike and must never be confused.

    replay          run the Mission that was saved, as it was saved
    reinterpret     compile the original words again, under today's compiler
    migrate         adopt the new interpretation, deliberately and on the record

A plan saved last year and opened today is a *replay*. Recompiling its text
under a compiler that has since learned to represent three more fields is a
different Mission, however faithfully it reads the same sentence — and showing
its figures under the old plan's name would rewrite history silently.

That is not hypothetical here. Three fields became canonical in one afternoon:

    dividend_policy          reinvesting compounds; holding as cash does not
    moving_average_estimator simple and exponential cross at different times
    funding_source           contribution invests the same total; extra cash more

Every plan saved before those changes has a stored scenario that lacks them.
Recompiling gives a different content hash, a different schedule hash and
possibly a different benchmark comparability verdict. All three are correct
answers to different questions.

The rule:

    historical replay must be stable across compiler upgrades
    reinterpretation may differ, and when it does the difference is typed
    migration is proposed, never performed silently
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Bumped whenever the compiler's canonical output can change for unchanged
#: input. Stamped into a saved plan so a later reader can tell whether a
#: difference is drift or an upgrade.
COMPILER_VERSION = "3"

#: What each version learned to represent. The record that turns "the hashes
#: differ" into "here is which change did it".
COMPILER_CHANGELOG: Mapping[str, Sequence[str]] = {
    "2": ("dividend_policy",),
    "3": ("moving_average_estimator", "funding_source"),
}


#: Where each field introduced by a compiler version lives in the stored form.
#: Used to reconstruct what an older compiler would have written, so a benchmark
#: can exercise a real migration instead of comparing today's output to itself.
FIELD_PATHS: Mapping[str, str] = {
    "dividend_policy": "methodology.holdings_policy.dividend_policy",
    "funding_source": "flows.funding_source",
    "moving_average_estimator": "methodology.event_program",
}


#: Stored bodies carry what was inferred, not the sentence explaining it. Shown
#: verbatim rather than reconstructed: inventing the rationale a user actually
#: read would be worse than admitting it was not kept.
REASON_NOT_STORED = (
    "The reason shown when this plan was saved was not stored with it."
)


class MigrationRequired(RuntimeError):
    """A stored plan cannot be replayed as saved."""


def as_compiled_by(stored: Mapping[str, Any], version: str) -> Dict[str, Any]:
    """What `version` of the compiler would have stored for this scenario.

    Reconstructed by removing every field introduced after it, driven by the
    changelog so the two cannot drift apart. Without this, a benchmark
    "comparing compiler versions" compiles both sides with today's compiler and
    finds no difference — measuring nothing while appearing to pass.
    """
    import copy

    out = copy.deepcopy(dict(stored))
    for introduced, fields in COMPILER_CHANGELOG.items():
        if introduced <= version:
            continue
        for name in fields:
            path = FIELD_PATHS.get(name)
            if not path:
                continue
            node: Any = out
            parts = path.split(".")
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node, Mapping) else None
                if node is None:
                    break
            if not isinstance(node, dict):
                continue
            leaf = parts[-1]
            if leaf == "event_program" and isinstance(node.get(leaf), list):
                node[leaf] = [{k: v for k, v in step.items()
                               if k != "estimator"} if isinstance(step, dict)
                              else step for step in node[leaf]]
            else:
                node.pop(leaf, None)
    return out


@dataclass(frozen=True)
class FieldChange:
    path: str
    before: Any
    after: Any

    @property
    def kind(self) -> str:
        if self.before is None:
            return "ADDED"
        if self.after is None:
            return "REMOVED"
        return "CHANGED"

    def __str__(self) -> str:
        if self.kind == "ADDED":
            return f"{self.path}: now represented as {self.after!r}"
        if self.kind == "REMOVED":
            return f"{self.path}: no longer represented (was {self.before!r})"
        return f"{self.path}: {self.before!r} -> {self.after!r}"


@dataclass
class SemanticDiff:
    """What changed between a stored Mission and a fresh interpretation.

    Typed rather than a text diff, because the consequences differ. A new
    represented field means the old plan was silent on something; a changed rule
    identity means it is no longer the same strategy; a changed schedule
    identity means benchmark comparisons that were flow-matched may not be.
    """

    stored_compiler: str
    current_compiler: str
    changes: List[FieldChange] = field(default_factory=list)

    rule_identity_changed: bool = False
    schedule_identity_changed: bool = False
    content_identity_changed: bool = False

    added_questions: List[str] = field(default_factory=list)
    resolved_questions: List[str] = field(default_factory=list)
    simulation_support_changed: List[str] = field(default_factory=list)

    @property
    def added(self) -> List[FieldChange]:
        return [c for c in self.changes if c.kind == "ADDED"]

    @property
    def removed(self) -> List[FieldChange]:
        return [c for c in self.changes if c.kind == "REMOVED"]

    @property
    def is_empty(self) -> bool:
        return not (self.changes or self.added_questions
                    or self.resolved_questions)

    @property
    def affects_comparability(self) -> bool:
        """Whether a benchmark comparison made under the old plan still holds.

        The schedule hash is what flow-matched comparison turns on, so a change
        there invalidates the matching even when the rule is untouched.
        """
        return self.schedule_identity_changed or self.rule_identity_changed

    def explain(self) -> List[str]:
        """Plain sentences, in the order a reader needs them."""
        out = []
        if self.stored_compiler != self.current_compiler:
            learned = []
            for version, fields in sorted(COMPILER_CHANGELOG.items()):
                if self.stored_compiler < version <= self.current_compiler:
                    learned.extend(fields)
            out.append(
                f"This plan was compiled by version {self.stored_compiler}; "
                f"the current compiler is version {self.current_compiler}"
                + (f", which learned to represent {', '.join(sorted(learned))}."
                   if learned else "."))
        out += [str(c) for c in self.changes]
        if self.rule_identity_changed:
            out.append("The market rule now has a different identity, so this "
                       "is not the same strategy the saved result measured.")
        if self.schedule_identity_changed:
            out.append("The contribution schedule now has a different identity. "
                       "Benchmarks matched to the old flows may no longer be "
                       "matched to these.")
        for question in self.added_questions:
            out.append(f"The current compiler asks about {question}, which the "
                       "saved plan never settled.")
        for question in self.resolved_questions:
            out.append(f"{question} was an open question and is now settled.")
        return out

    def to_json(self) -> Dict[str, Any]:
        return {
            "stored_compiler": self.stored_compiler,
            "current_compiler": self.current_compiler,
            "changes": [{"path": c.path, "kind": c.kind,
                         "before": c.before, "after": c.after}
                        for c in self.changes],
            "rule_identity_changed": self.rule_identity_changed,
            "schedule_identity_changed": self.schedule_identity_changed,
            "content_identity_changed": self.content_identity_changed,
            "affects_comparability": self.affects_comparability,
            "added_questions": list(self.added_questions),
            "resolved_questions": list(self.resolved_questions),
            "explanation": self.explain(),
        }


def _flatten(node: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(node, Mapping):
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, (list, tuple)):
        # Lists are compared whole. An element-wise diff of an event program
        # reports a reordering as four changes, which reads as four problems.
        out[prefix] = list(node)
    else:
        out[prefix] = node
    return out


#: Never part of a semantic difference. A plan renamed or re-saved is the same
#: plan, and reporting the name as a change buries the ones that matter.
_IGNORED = {"name", "version", "artifact_id", "content_hash", "rule_hash",
            "spec_version"}


def diff_stored_against(stored: Mapping[str, Any], current,
                        *, stored_compiler: str = "1",
                        current_compiler: str = COMPILER_VERSION,
                        stored_unresolved: Sequence[str] = (),
                        current_unresolved: Sequence[str] = ()) -> SemanticDiff:
    """Compare a stored canonical form against a freshly compiled scenario."""
    before = {k: v for k, v in _flatten(_canonical_of(stored)).items()
              if k.split(".")[-1] not in _IGNORED}
    after = {k: v for k, v in _flatten(current.canonical_form()).items()
             if k.split(".")[-1] not in _IGNORED}

    diff = SemanticDiff(stored_compiler=stored_compiler,
                        current_compiler=current_compiler)
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path), after.get(path)
        if old != new:
            diff.changes.append(FieldChange(path=path, before=old, after=new))

    stored_rule = (stored.get("methodology") or {})
    diff.rule_identity_changed = stored_rule != current.methodology_part()
    diff.schedule_identity_changed = (
        (stored.get("flows") or {}) != current.flow_part())
    diff.content_identity_changed = bool(diff.changes)

    diff.added_questions = sorted(set(current_unresolved) - set(stored_unresolved))
    diff.resolved_questions = sorted(set(stored_unresolved) - set(current_unresolved))
    return diff


def _canonical_of(stored: Mapping[str, Any]) -> Dict[str, Any]:
    """The canonical subset of a stored `to_json()` payload."""
    return {k: stored[k] for k in ("objective", "methodology", "protocol",
                                   "flows", "inferred") if k in stored}


def _window_from(provenance):
    """The stored temporal instruction, or None.

    A `provenance@1` body has no `time_window` key, and None is the correct
    answer there — the plan predates the field, so nothing was recorded and
    nothing may be assumed.
    """
    from .time_window import TimeWindow

    if not isinstance(provenance, dict):
        return None
    return TimeWindow.from_json(provenance.get("time_window"))


def _exclusions_from(provenance):
    """What the user chose to proceed without, read back as structure.

    Restored because the coverage gate consults it: without these a replayed
    plan looks like one that declared nothing it could not model, and the
    disclosure that narrowed its scope disappears from the page.
    """
    from .spec import ScenarioExclusion

    if not isinstance(provenance, dict):
        return ()
    return tuple(
        ScenarioExclusion(item=str(one.get("item", "")),
                          reason=str(one.get("reason", "")),
                          decision=str(one.get("decision")
                                       or "PROCEED_WITHOUT_MODELLING"),
                          acknowledged_at=str(one.get("acknowledged_at", "")))
        for one in (provenance.get("excluded") or ())
        if isinstance(one, dict) and one.get("item"))


def _funding_from(body):
    """Rebuild the funding policy a plan was saved with.

    `None` for bodies written before it existed, which is the honest answer:
    reconstructing a `Scheduled` from the schedule would be this build asserting
    what an older one meant, and reconstructing an `EventTriggered` would be
    worse — it would claim a rule executed on a plan whose figure came from
    buy-and-hold.
    """
    if not isinstance(body, Mapping) or not body.get("kind"):
        return None

    from decimal import Decimal

    from .funding import (
        Estimator,
        EventTriggered,
        ExecutionTiming,
        FundingKind,
        Scheduled,
        Trigger,
    )
    from .signals import SignalKind

    if body["kind"] == FundingKind.EVENT_TRIGGERED.value:
        trigger = body.get("trigger") or {}
        return EventTriggered(
            trigger=Trigger(
                subject=trigger.get("subject", ""),
                window=int(trigger.get("window", 0)),
                estimator=Estimator(trigger.get("estimator", "simple")),
                kind=SignalKind(trigger.get(
                    "kind", SignalKind.CROSSED_BELOW_MOVING_AVERAGE.value))),
            amount=Decimal(str(body.get("amount", "0"))),
            execution_timing=ExecutionTiming(
                body.get("execution_timing",
                         ExecutionTiming.NEXT_SESSION_OPEN.value)),
            starting_capital=Decimal(str(body.get("starting_capital", "0"))))

    return Scheduled(
        cadence=body.get("cadence", "once"),
        amount=Decimal(str(body.get("amount", "0"))),
        day_rule=body.get("day_rule", "first_session_of_period"),
        starting_capital=Decimal(str(body.get("starting_capital", "0"))))


def rebuild_scenario(stored: Mapping[str, Any]):
    """Rebuild a `ScenarioSpecification` from a stored canonical body.

    Returns `None` when the body is too old to rebuild faithfully, so a caller
    can say the replay is approximate rather than quietly serving a fresh
    interpretation under an old plan's name.
    """
    from .scenario import (AllocationRule, BenchmarkSet, HoldingsPolicy,
                           ScenarioSpecification)
    from .spec import FlowSchedule, Inference, Objective, Provenance

    methodology = stored.get("methodology") or {}
    flows = stored.get("flows") or {}
    protocol = stored.get("protocol") or {}
    if not methodology or not flows:
        return None

    allocation = methodology.get("allocation_rule") or {}
    holdings = methodology.get("holdings_policy") or {}
    benchmark = protocol.get("benchmark_set")

    try:
        return ScenarioSpecification(
            name=stored.get("name", "plan"),
            version=int(stored.get("version", 1)),
            objective=Objective(stored.get("objective", "REPLAY")),
            event_program=list(methodology.get("event_program") or []),
            flow_schedule=FlowSchedule(
                cadence=flows.get("cadence", "once"),
                amount=float(flows.get("amount", 0.0)),
                day_rule=flows.get("day_rule", "first_session_of_period"),
                inflation_adjusted=bool(flows.get("inflation_adjusted", False)),
                starting_capital=float(flows.get("starting_capital", 0.0)),
                # Absent in bodies written before the field was carried. The
                # default matches what those plans actually simulated.
                funding_source=flows.get("funding_source", "contribution"),
            ),
            allocation_rule=AllocationRule(
                assets=tuple(allocation.get("assets") or ()),
                weighting=allocation.get("weighting", "equal_weight_at_purchase"),
            ),
            holdings_policy=HoldingsPolicy(
                sells_allowed=bool(holdings.get("sells_allowed", True)),
                rebalancing_allowed=bool(holdings.get("rebalancing_allowed", True)),
                dividend_policy=holdings.get("dividend_policy", "reinvested"),
            ),
            benchmark_set=(BenchmarkSet(
                generated_by_rule=benchmark.get("generated_by_rule", ""),
                members=tuple(benchmark.get("members") or ()),
                ordering=benchmark.get("ordering", "unordered"),
            ) if benchmark else None),
            tax_treatment=protocol.get("tax_treatment", "NONE_APPLIED"),
            cash_policy_ref=protocol.get("cash_policy_ref", ""),
            funding=_funding_from(flows.get("funding")),
            # Inferred values participate in the content hash, so a rebuild
            # that drops them is not the same scenario. Only field and value are
            # stored — the rationale shown at the time is not — so a replayed
            # plan can say *what* was inferred and not *why*. Recorded as a
            # limitation rather than reconstructed, because inventing the
            # sentence a user actually read would be worse than omitting it.
            # The time window and the exclusions come back too. Rebuilding
            # only `inferred` meant a reopened plan ran with no window at all:
            # the stored body held "over the past five years", the plan page
            # rebuilt from it, and `_resolve_window` found nothing to slice
            # by, so the figure covered the whole snapshot. That is F1 again,
            # on the reopen path, for plans that recorded the window
            # correctly.
            #
            # Both are restored rather than re-derived. Neither is a
            # presentation artifact — they were written as structure by the
            # compiler and by the user's own acknowledgement — and re-reading
            # the description here would substitute today's reading for the
            # one the owner confirmed.
            provenance=Provenance(
                inferred=tuple(
                    Inference(field=entry.get("field", ""),
                              value=entry.get("value", ""),
                              why=REASON_NOT_STORED, confirmed=True)
                    for entry in (stored.get("inferred") or [])),
                excluded=_exclusions_from(stored.get("provenance")),
                time_window=_window_from(stored.get("provenance"))),
        )
    except (TypeError, ValueError, KeyError):
        return None


@dataclass(frozen=True)
class MigrationProposal:
    """An explanation and an offer. Never an action.

    Adopting a new interpretation changes what a saved plan means, and the
    person who saved it is the only one who can agree to that. A system that
    migrates silently has rewritten a record the user believed was theirs.
    """

    plan_id: str
    diff: SemanticDiff
    recommended: bool

    @property
    def required(self) -> bool:
        """Whether the stored plan can still be replayed at all.

        Distinct from *recommended*. A plan whose stored form still runs may
        keep running for ever; the new interpretation is an option, not a debt.
        """
        return False

    def to_json(self) -> Dict[str, Any]:
        return {"plan_id": self.plan_id, "recommended": self.recommended,
                "required": self.required, "diff": self.diff.to_json()}


def propose_migration(plan_id: str, diff: SemanticDiff) -> Optional[MigrationProposal]:
    if diff.is_empty:
        return None
    return MigrationProposal(plan_id=plan_id, diff=diff,
                             recommended=diff.affects_comparability)
