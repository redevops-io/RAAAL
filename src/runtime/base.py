"""One lifecycle for every runtime a Mission executes inside.

The pattern this codebase keeps repeating: a configuration value turns out to
affect comparability, reproducibility or interpretation, and becomes a
first-class artifact.

    execution_lag=1        →  EvaluationProtocol
    252 trading days       →  Calendar
    Sharpe > 0.9           →  StatisticalPolicy
    "NONE_APPLIED"         →  TaxRuntime

Each of those was built independently, and each invented slightly different
rules for hashing, versioning and compatibility. Left alone, the next four would
too, and the cost only shows up later — as two runtimes that disagree about what
"the same version" means, discovered by a comparison that should have been
refused.

So the lifecycle is defined once here and the differences live only in what each
runtime declares.

**`content_hash` and `compatibility_hash` are deliberately different.** Identity
changes when anything changes, including prose. Comparability changes only when
something changes that could move a number. A calendar version that fixes a typo
in its description is a new artifact whose results remain comparable with the
old one — and a system with only one hash has to choose between lying about
identity and refusing a valid comparison.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Sequence


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=str).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RuntimeAssumption:
    """Something a runtime asserts, and the mechanism that performs it."""

    name: str
    statement: str
    realized_by: str
    risk: str = ""
    citation: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "statement": self.statement,
                "realized_by": self.realized_by, "risk": self.risk,
                "citation": self.citation}


class Exclusion(str, Enum):
    """Why something is not modelled. Three very different statements.

    Listing them together as "not modelled" makes a correct treatment look like
    a gap. Capital gains are not taxed inside a 401(k) during accumulation
    because they are not taxable — that is the right answer, and presenting it
    beside "we did not implement state tax" invites a reader to discount both.
    """

    NOT_APPLICABLE = "NOT_APPLICABLE"
    """It does not arise here. Nothing is missing."""

    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    """It arises and is deliberately not modelled. A real gap, stated."""

    UNRESOLVED = "UNRESOLVED"
    """It arises, and could be modelled, but an input nobody supplied is needed.
    Answerable by asking rather than by building."""


@dataclass(frozen=True)
class RuntimeLimitation:
    """Something a runtime does not do, and which kind of not-doing it is.

    `reason` is the runtime's own reading in isolation. Whether it is right
    often depends on the composed environment — which is what
    `ExecutionEnvironment.scope()` refines.
    """

    name: str
    statement: str
    reason: "Exclusion" = None  # type: ignore[assignment]
    applicable_unless: Sequence[str] = ()
    """Environment conditions under which this becomes NOT_APPLICABLE, expressed
    as `kind:predicate` — e.g. `account:tax_deferred`. Declared here so the
    refinement is a property of the runtime rather than a special case buried in
    the environment."""

    def __post_init__(self) -> None:
        if self.reason is None:
            object.__setattr__(self, "reason", Exclusion.OUT_OF_SCOPE)

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "statement": self.statement,
                "reason": self.reason.value,
                "applicable_unless": list(self.applicable_unless)}


class RuntimeArtifact(ABC):
    """The shared surface every runtime presents.

    Subclasses supply `declared_form` and, where the two differ,
    `comparable_form`. Everything else — identity, hashing, realization
    checking, serialisation — is inherited, so a new runtime cannot quietly
    invent its own versioning semantics.
    """

    kind: ClassVar[str] = "runtime"

    undefined_without: ClassVar[Sequence[str]] = ()
    """Runtime kinds whose absence leaves this one's declarations meaningless.

    Not a construction order and not a software dependency. A tax runtime saying
    "gains are not taxed" is a correct statement in a 401(k) and an admission of
    incompleteness in a taxable account — the sentence has no truth value until
    an account runtime is present. `undefined_without` says that, where
    `requires` said "instantiate this first", which is a different and less
    useful claim."""

    interpreted_with: ClassVar[Sequence[str]] = ()
    """Kinds that sharpen this one's meaning without being necessary for it."""

    affects_causal_isolation: ClassVar[Sequence[str]] = ()
    """The subset of `interpreted_with` whose difference defeats an isolation
    claim about this runtime.

    Not every interpretation relation is causal. An account may *refuse* a flow
    without changing what the flow means, so a differing account does not make
    "only the flow schedule differs" a false statement. An account that changes
    how gains are treated does. Listing the causal ones explicitly keeps the
    derived comparison dependency from overreaching into every relation a
    runtime happens to mention."""

    @classmethod
    def causal_dependencies(cls) -> tuple:
        """What must be equal for an isolation claim about this runtime to hold.

        Derived from the runtime's own declarations, so the comparison registry
        stops carrying a second copy of the same fact. Two declarations of one
        thing drift; this is the same reason `ISOLATION_DIMENSIONS` stopped being
        a hand-written tuple.
        """
        causal = tuple(k for k in cls.affects_causal_isolation
                       if k in cls.interpreted_with)
        return tuple(cls.undefined_without) + causal

    name: str
    version: int

    # ---- identity ---------------------------------------------------------

    @property
    def artifact_id(self) -> str:
        return f"{self.kind}/{self.name}@{self.version}"

    @property
    def concept_id(self) -> str:
        """The runtime *family*: one semantic runtime continuing through time.

        `calendar/nyse` is the same runtime whether its coverage horizon ends in
        2026 or 2030. Extending coverage mints a version and changes nothing
        about what the runtime means, so results across the two remain
        comparable and the family is what a reader is actually thinking of.

        Three levels, matching methodology/version/run exactly:

            calendar/nyse        family     — the semantic runtime
            calendar/nyse@2      version    — a specific declaration
            RealizedData(...)    instance   — what a run actually received
        """
        return f"{self.kind}/{self.name}"

    @property
    def family(self) -> str:
        return self.concept_id

    def same_family_as(self, other: "RuntimeArtifact") -> bool:
        return self.family == other.family

    @abstractmethod
    def declared_form(self) -> Dict[str, Any]:
        """Everything this runtime declares. Any change mints a new version."""

    def comparable_form(self) -> Dict[str, Any]:
        """The subset that could move a number.

        Defaults to the whole declaration, which is the safe direction: a
        runtime that has not thought about the distinction refuses comparisons
        it might have allowed, rather than allowing ones it should have refused.
        """
        return self.declared_form()

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.declared_form())

    @property
    def compatibility_hash(self) -> str:
        return canonical_hash(self.comparable_form())

    def is_comparable_with(self, other: "RuntimeArtifact") -> bool:
        return (self.kind == other.kind
                and self.compatibility_hash == other.compatibility_hash)

    # ---- what it claims, and whether the claim is real --------------------

    @property
    def assumptions(self) -> Sequence[RuntimeAssumption]:
        return ()

    @property
    def limitations(self) -> Sequence[RuntimeLimitation]:
        return ()

    def realization_checks(self) -> List[str]:
        """Names every assumption says performs it."""
        return [a.realized_by for a in self.assumptions]

    def unrealized(self, implemented: Sequence[str]) -> List[str]:
        """Assumptions whose named mechanism does not exist.

        The same check the methodology verifier runs. A runtime declaring a
        behaviour it does not perform is the failure mode this whole system is
        built against, and a runtime is a more convincing place to hide one than
        a methodology because nobody reads it.
        """
        available = set(implemented)
        return [a.name for a in self.assumptions if a.realized_by not in available]

    def scope(self) -> Dict[str, Any]:
        """What is modelled beside what is not. Travels with results."""
        return {
            "modelled": [a.to_json() for a in self.assumptions],
            "not_modelled": [l.to_json() for l in self.limitations],
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "concept_id": self.concept_id,
            "kind": self.kind,
            "content_hash": self.content_hash,
            "compatibility_hash": self.compatibility_hash,
            "declared": self.declared_form(),
            "scope": self.scope(),
        }


class MissingRuntime(KeyError):
    """A required runtime was not supplied.

    Raised rather than defaulted. Every default in this system's history became
    an erratum, and a defaulted *runtime* would be an erratum affecting every
    result produced under it.
    """


class Severity(str, Enum):
    INVALID = "INVALID"
    """The composition cannot produce a meaningful result."""

    SUSPECT = "SUSPECT"
    """It will produce a result, and the result is probably not what was meant."""


@dataclass(frozen=True)
class CompositionConflict:
    """Two individually valid runtimes that do not make sense together.

    The next defect class after "behaviour without declaration" and "declaration
    without realization": every runtime passes its own checks and the environment
    they compose into is still wrong. Nothing at the runtime level can catch it,
    because no runtime can see the others.
    """

    code: str
    runtimes: Sequence[str]
    detail: str
    severity: Severity = Severity.INVALID
    category: "RuleCategory" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.category is None:
            object.__setattr__(self, "category", RuleCategory.SEMANTICS)

    def to_json(self) -> Dict[str, Any]:
        return {"code": self.code, "runtimes": list(self.runtimes),
                "detail": self.detail, "severity": self.severity.value,
                "category": self.category.value}


class RuleCategory(str, Enum):
    """What kind of assumption a rule is about.

    Grouping matters at the point a reader asks *why* an environment is invalid.
    "Three conflicts" is a number; "your temporal assumptions disagree" is an
    answer, and it points at the two runtimes to look at.
    """

    SEMANTICS = "SEMANTICS"
    TEMPORAL = "TEMPORAL"
    TAX = "TAX"
    ACCOUNT = "ACCOUNT"
    CORPORATE_ACTION = "CORPORATE_ACTION"
    DATA = "DATA"
    EXECUTION = "EXECUTION"


CompositionCheck = Callable[["ExecutionEnvironment"], List[CompositionConflict]]


@dataclass(frozen=True)
class CompositionRuleSpec:
    """A cross-runtime rule, addressable like everything else here.

    The rules are as much a part of the declared system as the runtimes they
    check. Leaving them as anonymous functions would make "what does this
    platform know composes badly?" a question answerable only by reading source.
    """

    id: str
    category: "RuleCategory"
    severity: "Severity"
    affects: Sequence[str]
    description: str
    check: CompositionCheck

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "category": self.category.value,
                "severity": self.severity.value, "affects": list(self.affects),
                "description": self.description}


#: Every registered rule. Enumerable on purpose.
REGISTERED_RULES: List[CompositionRuleSpec] = []


def composition_rule(*, id: str, category: "RuleCategory", severity: "Severity",
                     affects: Sequence[str], description: str):
    """Register a cross-runtime rule with the metadata that makes it queryable."""
    def decorate(fn: CompositionCheck) -> CompositionCheck:
        REGISTERED_RULES.append(CompositionRuleSpec(
            id=id, category=category, severity=severity,
            affects=tuple(affects), description=description, check=fn))
        return fn
    return decorate


@dataclass(frozen=True)
class ExecutionEnvironment:
    """The composed environment a Mission executes inside.

    A result does not come from a methodology. It comes from a methodology
    evaluated under a protocol, on a calendar, taxed a particular way, in a
    particular kind of account, against a particular vintage of market data. The
    environment names all of it, and its hash is what a run cites.
    """

    runtimes: Mapping[str, RuntimeArtifact]

    #: Kinds a Mission cannot execute without. Anything absent raises rather
    #: than falling back, because a silently defaulted runtime produces a number
    #: nobody chose the conditions for.
    REQUIRED: ClassVar[Sequence[str]] = ("protocol", "calendar")

    def __post_init__(self) -> None:
        missing = [k for k in self.REQUIRED if k not in self.runtimes]
        if missing:
            raise MissingRuntime(
                f"execution environment is missing {missing}. Every runtime must "
                "be named: a defaulted one produces a number nobody chose the "
                "conditions for"
            )
        for kind, runtime in self.runtimes.items():
            if runtime.kind != kind:
                raise ValueError(
                    f"runtime filed under {kind!r} declares itself {runtime.kind!r}"
                )

    def get(self, kind: str) -> RuntimeArtifact:
        if kind not in self.runtimes:
            raise MissingRuntime(f"no {kind} runtime in this environment")
        return self.runtimes[kind]

    @property
    def environment_hash(self) -> str:
        """Identity of the whole environment. What a run cites."""
        return canonical_hash(
            {kind: rt.content_hash for kind, rt in sorted(self.runtimes.items())})

    @property
    def compatibility_hash(self) -> str:
        return canonical_hash(
            {kind: rt.compatibility_hash
             for kind, rt in sorted(self.runtimes.items())})

    def differences(self, other: "ExecutionEnvironment") -> List[str]:
        """Runtime kinds that would change a number between the two."""
        kinds = sorted(set(self.runtimes) | set(other.runtimes))
        out: List[str] = []
        for kind in kinds:
            mine, theirs = self.runtimes.get(kind), other.runtimes.get(kind)
            if mine is None or theirs is None:
                out.append(kind)
            elif mine.compatibility_hash != theirs.compatibility_hash:
                out.append(kind)
        return out

    def isolation(self, other: "ExecutionEnvironment") -> Optional[str]:
        """The single runtime two environments differ in, if there is exactly one.

        This is what makes an explanation mechanical rather than narrative:
        "same methodology, same protocol, same calendar, different tax runtime"
        yields `tax`, and the comparison can say what it isolates instead of
        listing what it does not.
        """
        differing = self.differences(other)
        return differing[0] if len(differing) == 1 else None

    def unrealized(self, implemented: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
        """Every runtime's declarations that nothing performs."""
        out: Dict[str, List[str]] = {}
        for kind, runtime in self.runtimes.items():
            gaps = runtime.unrealized(implemented.get(kind, ()))
            if gaps:
                out[kind] = gaps
        return out

    # ---- composition ------------------------------------------------------

    def missing_dependencies(self) -> Dict[str, List[str]]:
        """Runtimes whose declared dependencies are absent."""
        out: Dict[str, List[str]] = {}
        for kind, runtime in self.runtimes.items():
            gaps = [d for d in runtime.undefined_without
                    if d not in self.runtimes]
            if gaps:
                out[kind] = gaps
        return out

    def validate_composition(self) -> List[CompositionConflict]:
        """Whether these runtimes make sense together.

        `is_comparable_with` asks whether two versions of one runtime agree.
        This asks the question no runtime can answer about itself.
        """
        conflicts: List[CompositionConflict] = []
        for kind, gaps in sorted(self.missing_dependencies().items()):
            conflicts.append(CompositionConflict(
                code="UNDEFINED_WITHOUT", runtimes=(kind, *gaps),
                detail=(f"the {kind} runtime's declarations have no truth value "
                        f"without {', '.join(gaps)}: the same statement means "
                        f"different things depending on them"),
                category=RuleCategory.SEMANTICS,
            ))
        for spec in REGISTERED_RULES:
            conflicts.extend(spec.check(self))
        return conflicts

    def conflicts_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Grouped, because "why is this invalid?" is a question about kinds."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        for conflict in self.validate_composition():
            out.setdefault(conflict.category.value, []).append(conflict.to_json())
        return out

    @property
    def is_valid(self) -> bool:
        return not any(c.severity is Severity.INVALID
                       for c in self.validate_composition())

    def satisfies(self, condition: str) -> bool:
        """Evaluate a `kind:predicate` condition against this environment.

        Predicates are properties on the runtime itself, so a runtime declaring
        `account:tax_deferred` is asking a question the account runtime already
        knows how to answer, rather than the environment inspecting its fields.
        """
        kind, _, predicate = condition.partition(":")
        runtime = self.runtimes.get(kind)
        return bool(runtime is not None and getattr(runtime, predicate, False))

    def scope(self) -> Dict[str, Any]:
        """The whole environment's modelled/not-modelled statement, merged.

        A reader needs one list, not seven — and the list is *refined* by the
        composition. Capital gains going untaxed is a gap in a taxable account
        and the correct treatment inside a 401(k), and the same runtime produces
        both statements. Only the environment can tell them apart.
        """
        modelled: List[Dict[str, Any]] = []
        not_modelled: List[Dict[str, Any]] = []
        for kind, runtime in sorted(self.runtimes.items()):
            for m in runtime.scope()["modelled"]:
                modelled.append({**m, "runtime": kind})
            for limitation in runtime.limitations:
                entry = limitation.to_json()
                if any(self.satisfies(c) for c in limitation.applicable_unless):
                    entry["reason"] = Exclusion.NOT_APPLICABLE.value
                    entry["refined_by"] = [
                        c for c in limitation.applicable_unless if self.satisfies(c)]
                not_modelled.append({**entry, "runtime": kind})

        by_reason: Dict[str, int] = {}
        for entry in not_modelled:
            by_reason[entry["reason"]] = by_reason.get(entry["reason"], 0) + 1

        return {
            "modelled": modelled,
            "not_modelled": not_modelled,
            "by_reason": by_reason,
            "note": (
                "Everything under 'modelled' is performed by a named, versioned "
                "runtime. Under 'not modelled', NOT_APPLICABLE means it does not "
                "arise in this environment, OUT_OF_SCOPE means it does and is "
                "deliberately not modelled, and UNRESOLVED means an input nobody "
                "supplied is needed. Treating all three as the same omission is "
                "how a correct treatment gets read as a gap."
            ),
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "environment_hash": self.environment_hash,
            "compatibility_hash": self.compatibility_hash,
            "runtimes": {kind: rt.to_json()
                         for kind, rt in sorted(self.runtimes.items())},
            "scope": self.scope(),
            "composition_conflicts": [c.to_json()
                                      for c in self.validate_composition()],
            "is_valid": self.is_valid,
        }


# --- cross-runtime rules --------------------------------------------------
#
# Each names the pair it is about. They live here rather than inside a runtime
# because no runtime can see the others, which is exactly why this defect class
# survives runtime-level checking.

@composition_rule(
    id="GAINS_TAXED_IN_SHELTER", category=RuleCategory.TAX,
    severity=Severity.INVALID, affects=("tax", "account"),
    description="A tax runtime taxing realized gains annually inside an account "
                "that shelters them.")
def _gains_taxed_inside_a_shelter(env: "ExecutionEnvironment") -> List[CompositionConflict]:
    tax, account = env.runtimes.get("tax"), env.runtimes.get("account")
    if tax is None or account is None:
        return []
    if getattr(tax, "models_capital_gains", False) and \
            getattr(account, "tax_deferred", False):
        return [CompositionConflict(
            code="GAINS_TAXED_IN_SHELTER", runtimes=("tax", "account"),
            category=RuleCategory.TAX,
            detail=("the tax runtime taxes realized gains annually while the "
                    "account runtime shelters them. Both are individually "
                    "correct; together they tax something that is not taxable, "
                    "and every rebalancing decision under them is distorted"),
        )]
    return []


@composition_rule(
    id="SESSION_ALIGNMENT_MISMATCH", category=RuleCategory.TEMPORAL,
    severity=Severity.INVALID, affects=("calendar", "market_data"),
    description="Data aligned to one trading calendar evaluated under another.")
def _calendar_and_data_disagree(env: "ExecutionEnvironment") -> List[CompositionConflict]:
    calendar, data = env.runtimes.get("calendar"), env.runtimes.get("market_data")
    if calendar is None or data is None:
        return []
    declared = getattr(data, "session_alignment", None)
    expected = getattr(calendar, "name", None)
    if declared and expected and declared != expected:
        return [CompositionConflict(
            code="SESSION_ALIGNMENT_MISMATCH", runtimes=("calendar", "market_data"),
            category=RuleCategory.TEMPORAL,
            detail=(f"the calendar expects {expected} sessions and the data is "
                    f"aligned to {declared}. Sessions one side has and the other "
                    f"does not become padding, which is the defect that inflated "
                    f"annualized figures by 31%"),
        )]
    return []


@composition_rule(
    id="FLOW_KIND_UNSUPPORTED_BY_ACCOUNT", category=RuleCategory.ACCOUNT,
    severity=Severity.INVALID, affects=("flow", "account"),
    description="A flow the account cannot receive — employer shares into an "
                "account with no mechanism for them.")
def _account_cannot_receive_flow(env: "ExecutionEnvironment") -> List[CompositionConflict]:
    flow, account = env.runtimes.get("flow"), env.runtimes.get("account")
    if flow is None or account is None:
        return []
    kinds = {getattr(k, "value", k) for k in getattr(flow, "supported_kinds", ())}
    if "RSU_VEST" in kinds and not getattr(account, "accepts_employer_shares", True):
        return [CompositionConflict(
            code="FLOW_KIND_UNSUPPORTED_BY_ACCOUNT", runtimes=("flow", "account"),
            category=RuleCategory.ACCOUNT,
            detail=("the flow runtime delivers employer shares and the account "
                    "cannot receive them. The vests would silently land nowhere, "
                    "and the plan would appear to work while contributing nothing"),
        )]
    return []


@composition_rule(
    id="ADJUSTMENT_POLICY_CONFLICT", category=RuleCategory.CORPORATE_ACTION,
    severity=Severity.INVALID, affects=("corporate_action", "market_data"),
    description="Corporate actions applied to prices already adjusted for them.")
def _corporate_actions_need_unadjusted(env: "ExecutionEnvironment") -> List[CompositionConflict]:
    actions, data = env.runtimes.get("corporate_action"), env.runtimes.get("market_data")
    if actions is None or data is None:
        return []
    if getattr(actions, "requires_unadjusted", False) and \
            getattr(data, "adjustment_policy", "") == "adjusted_only":
        return [CompositionConflict(
            code="ADJUSTMENT_POLICY_CONFLICT",
            runtimes=("corporate_action", "market_data"),
            category=RuleCategory.CORPORATE_ACTION,
            detail=("the corporate-action runtime applies splits and dividends "
                    "itself, and the data is already adjusted for them. Applying "
                    "both counts every action twice"),
        )]
    return []
