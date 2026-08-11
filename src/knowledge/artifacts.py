"""Reasoning as versioned artifacts.

Every release has moved one implicit thing into an addressable object:
methodology, evaluation protocol, statistics, publication policy, trading
calendar. This one moves the reasoning.

Three artifacts, each independently addressable::

    claim/hrp-reduces-concentration@1
    assumption/sample-covariance@1
    evidence/lopez-de-prado-2016@1

**Methodologies reference claims; they do not contain them.** A prose
``claim_used`` string cannot be supported by two methodologies, contradicted by a
replication, or superseded without editing every file that quotes it. An
addressable claim can.

**Evidence declares its stance toward a claim, not the other way round.** Adding
a contradicting paper must not require editing the claim it contradicts —
otherwise the act of recording disagreement is gated on the claim's owner.

**Assumptions get equal status to claims**, because the project's own history
says they should: the covariance estimator, the trading calendar, the execution
lag, the turnover precedence rule and the annualization basis were each a defect,
and each was an undeclared assumption rather than a wrong claim or missing
evidence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

KNOWLEDGE_SPEC_VERSION = "0.1"


def _hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    """Evidence supports it and nothing credible contradicts it."""

    CONTESTED = "CONTESTED"
    """Supporting and contradicting evidence both exist. Not a defect — a state."""

    QUALIFIED = "QUALIFIED"
    """Holds only within stated bounds; evidence narrows its applicability."""

    REFUTED = "REFUTED"
    """Contradicting evidence prevails."""

    SUPERSEDED = "SUPERSEDED"
    """Replaced by a later claim version."""

    UNASSESSED = "UNASSESSED"


class Stance(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    QUALIFIES = "QUALIFIES"
    """Neither supports nor refutes — narrows where the claim holds."""

    UNRELATED = "UNRELATED"


class EvidenceKind(str, Enum):
    PAPER = "PAPER"
    REPLICATION = "REPLICATION"
    PLATFORM_RESULT = "PLATFORM_RESULT"
    """A result produced by this platform — a run, an assessment, an erratum."""

    ERRATUM = "ERRATUM"
    FORWARD_RECORD = "FORWARD_RECORD"


class AssumptionKind(str, Enum):
    ESTIMATION = "ESTIMATION"        # e.g. sample vs exponential covariance
    MARKET_STRUCTURE = "MARKET_STRUCTURE"  # e.g. which sessions exist
    EXECUTION = "EXECUTION"          # e.g. one-session lag, cost level
    CONSTRAINT_POLICY = "CONSTRAINT_POLICY"  # e.g. hard bounds beat soft turnover
    STATISTICAL = "STATISTICAL"      # e.g. annualization basis, IID under the null
    DATA = "DATA"                    # e.g. adjusted close is point-in-time


@dataclass(frozen=True)
class Realization:
    """Where an assumption actually takes effect.

    The link that turns "which methodologies depend on assumption X?" from a
    grep for a string into a graph query.
    """

    artifact_kind: str      # methodology | protocol | calendar | statistics
    field: str              # dotted path, e.g. params.covariance_estimator
    value: Any = None

    def to_json(self) -> Dict[str, Any]:
        return {"artifact_kind": self.artifact_kind, "field": self.field, "value": self.value}


@dataclass(frozen=True)
class Assumption:
    """A declared belief that must hold for a result to mean what it says."""

    name: str
    version: int
    statement: str
    kind: AssumptionKind
    realized_by: Sequence[Realization] = ()
    risk: str = ""
    """What goes wrong if it is false. Populated from experience where possible."""

    validated_by: Sequence[str] = ()
    """Protocol or test references that actually check it, rather than assert it."""

    history: Sequence[str] = ()
    superseded_by: Optional[str] = None
    spec_version: str = KNOWLEDGE_SPEC_VERSION

    @property
    def concept_id(self) -> str:
        return f"assumption/{self.name}"

    @property
    def artifact_id(self) -> str:
        return f"assumption/{self.name}@{self.version}"

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "statement": self.statement,
            "kind": self.kind.value,
            "realized_by": sorted(
                (r.to_json() for r in self.realized_by),
                key=lambda r: (r["artifact_kind"], r["field"]),
            ),
            "validated_by": sorted(self.validated_by),
            "superseded_by": self.superseded_by,
        }

    @property
    def content_hash(self) -> str:
        return _hash(self.canonical_form())

    @property
    def is_validated(self) -> bool:
        """An assumption nothing checks is an assertion, not a control."""
        return bool(self.validated_by)

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "artifact_id": self.artifact_id,
            "concept_id": self.concept_id,
            "content_hash": self.content_hash,
            "risk": self.risk,
            "history": list(self.history),
            "is_validated": self.is_validated,
        }


@dataclass(frozen=True)
class Claim:
    """An addressable assertion about how markets or methods behave."""

    name: str
    version: int
    statement: str
    scope: str = ""
    """Where the claim is asserted to hold. A claim without scope is unfalsifiable."""

    depends_on: Sequence[str] = ()
    """Assumption references this claim rests on."""

    superseded_by: Optional[str] = None
    derived_from: Optional[str] = None
    change_rationale: str = ""
    spec_version: str = KNOWLEDGE_SPEC_VERSION

    @property
    def concept_id(self) -> str:
        return f"claim/{self.name}"

    @property
    def artifact_id(self) -> str:
        return f"claim/{self.name}@{self.version}"

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "statement": self.statement,
            "scope": self.scope,
            "depends_on": sorted(self.depends_on),
            "superseded_by": self.superseded_by,
        }

    @property
    def content_hash(self) -> str:
        return _hash(self.canonical_form())

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "artifact_id": self.artifact_id,
            "concept_id": self.concept_id,
            "content_hash": self.content_hash,
            "derived_from": self.derived_from,
            "change_rationale": self.change_rationale,
        }


@dataclass(frozen=True)
class Evidence:
    """A source, and the stance it takes toward a claim.

    Stance lives here rather than on the claim so that recording disagreement
    never requires editing the thing being disagreed with.
    """

    name: str
    version: int
    kind: EvidenceKind
    about: str
    """The claim reference this evidence bears on."""

    stance: Stance
    summary: str
    identifier: str = ""
    """External identifier where one exists — doi:, arxiv:, or an internal id."""

    strength: str = "moderate"       # strong | moderate | weak | anecdotal
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    """Bi-temporal validity. Evidence is invalidated, never deleted."""

    produced_by: Sequence[str] = ()
    """Run, erratum or assessment references, when generated by this platform."""

    spec_version: str = KNOWLEDGE_SPEC_VERSION

    @property
    def artifact_id(self) -> str:
        return f"evidence/{self.name}@{self.version}"

    @property
    def is_current(self) -> bool:
        return self.valid_to is None

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "kind": self.kind.value,
            "about": self.about,
            "stance": self.stance.value,
            "identifier": self.identifier,
            "strength": self.strength,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }

    @property
    def content_hash(self) -> str:
        return _hash(self.canonical_form())

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "summary": self.summary,
            "produced_by": list(self.produced_by),
            "is_current": self.is_current,
        }


class FindingStatus(str, Enum):
    """The lifecycle of a *conclusion*, not of the inquiry that produced it.

    Those are different things, which is why `Investigation` exists separately:
    an inquiry can end without ever producing a conclusion, and this enum has no
    way to say that.
    """

    OPEN = "OPEN"
    """Provisional. The conclusion is stated but not yet settled."""

    CONCLUDED = "CONCLUDED"
    """Settled, with the resolution recorded."""

    SUPERSEDED = "SUPERSEDED"
    """A later finding replaced this conclusion."""

    WITHDRAWN = "WITHDRAWN"
    """The conclusion did not survive scrutiny. Retained, not deleted."""


class ImpactRelation(str, Enum):
    """How a finding bears on another artifact."""

    REFUTES = "REFUTES"
    QUALIFIES = "QUALIFIES"
    MOTIVATED = "MOTIVATED"
    """The finding caused this artifact to be created."""

    CORRECTED = "CORRECTED"
    """The finding produced a correction to this artifact."""

    INTRODUCED = "INTRODUCED"
    """The finding forced this assumption to be declared."""

    INVALIDATES_RESULTS_OF = "INVALIDATES_RESULTS_OF"


@dataclass(frozen=True)
class Impact:
    """A typed edge from a finding to something it affects."""

    target: str
    relation: ImpactRelation
    detail: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "relation": self.relation.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Finding:
    """The conclusion of an investigation.

    Neither a claim nor evidence. A claim is an assertion about the world; a piece
    of evidence bears on one claim; a finding is what someone concluded after
    synthesising several pieces of evidence — and that conclusion typically
    touches several claims, methodologies and assumptions at once.

    Before this artifact existed, that synthesis lived in a methodology's
    ``change_rationale`` prose and in erratum documents. It was therefore not
    queryable, not reviewable as a unit, and not something Discovery Runtime could
    produce or Mission Runtime could route.

    Findings are the natural output of an investigation and the natural input to a
    review: they say *what we concluded*, *what it rests on*, and *what has to
    change as a result*.
    """

    name: str
    version: int
    statement: str
    status: FindingStatus
    supported_by: Sequence[str] = ()
    """Evidence references the conclusion synthesises. A finding with no evidence
    is an opinion."""

    impacts: Sequence[Impact] = ()
    resolution: str = ""
    """What was actually done. An investigation that concluded and changed nothing
    should say so explicitly."""

    opened_at: Optional[str] = None
    concluded_at: Optional[str] = None
    superseded_by: Optional[str] = None
    spec_version: str = KNOWLEDGE_SPEC_VERSION

    @property
    def concept_id(self) -> str:
        return f"finding/{self.name}"

    @property
    def artifact_id(self) -> str:
        return f"finding/{self.name}@{self.version}"

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "statement": self.statement,
            "status": self.status.value,
            "supported_by": sorted(self.supported_by),
            "impacts": sorted(
                (i.to_json() for i in self.impacts),
                key=lambda i: (i["target"], i["relation"]),
            ),
            "superseded_by": self.superseded_by,
        }

    @property
    def content_hash(self) -> str:
        return _hash(self.canonical_form())

    @property
    def is_evidenced(self) -> bool:
        return bool(self.supported_by)

    def targets(self, prefix: str) -> List[str]:
        """Impacted artifacts of one kind, e.g. ``methodology`` or ``claim``."""
        return [i.target for i in self.impacts if i.target.startswith(f"{prefix}/")]

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "artifact_id": self.artifact_id,
            "concept_id": self.concept_id,
            "content_hash": self.content_hash,
            "resolution": self.resolution,
            "opened_at": self.opened_at,
            "concluded_at": self.concluded_at,
            "is_evidenced": self.is_evidenced,
        }


class InvestigationOutcome(str, Enum):
    """How an inquiry ended — including the ways that produce no finding.

    This enum is the reason `Investigation` had to become its own artifact. A
    finding can only record that something *was* concluded; there was no way to
    say an inquiry ran carefully and concluded that the effect is not there, or
    that the evidence could not settle it, or that the work simply stopped.

    Those three are not the same, and flattening them is how a research record
    acquires survivorship bias: only the inquiries that found something leave a
    trace, and the library then reads as though every question ever asked was
    answered affirmatively.
    """

    PENDING = "PENDING"
    """Still open. No outcome yet — not the same as no result."""

    FINDING_RECORDED = "FINDING_RECORDED"
    """Concluded, and the conclusion is a finding artifact."""

    NO_EFFECT_FOUND = "NO_EFFECT_FOUND"
    """Concluded: the hypothesis did not hold. A null result, deliberately
    first-class. This is the outcome most likely to go unrecorded and the one
    whose absence most distorts a trial count."""

    INCONCLUSIVE = "INCONCLUSIVE"
    """Concluded: the evidence available could not settle the question. Distinct
    from a null result — "we could not tell" is not "there is nothing there"."""

    ABANDONED = "ABANDONED"
    """Closed without concluding. Honest about effort spent and stopped."""

    @property
    def is_closed(self) -> bool:
        return self is not InvestigationOutcome.PENDING

    @property
    def is_null_result(self) -> bool:
        """Concluded, carefully, without a finding to show for it."""
        return self in {InvestigationOutcome.NO_EFFECT_FOUND,
                        InvestigationOutcome.INCONCLUSIVE}


#: Outcomes that must name at least one finding, and outcomes that must not. An
#: investigation claiming a finding it does not cite is unverifiable; one
#: claiming a null result while citing a finding has mislabelled its own outcome.
_REQUIRES_FINDING = {InvestigationOutcome.FINDING_RECORDED}
_FORBIDS_FINDING = {InvestigationOutcome.NO_EFFECT_FOUND,
                    InvestigationOutcome.INCONCLUSIVE}


@dataclass(frozen=True)
class Investigation:
    """A question someone asked, and how asking it ended.

    The artifact that exists *before* there is anything to conclude. A finding is
    an answer; an investigation is the question plus the work — which means it
    can be recorded the moment the question is asked, and can close without ever
    producing an answer.

    Two failure modes this makes visible, both of which the platform's own
    history contains:

    - **Silent abandonment.** Configurations were tried and dropped without
      record. Every dropped configuration is still a trial for deflation
      purposes, so an unrecorded abandonment quietly inflates every DSR in the
      lineage. `trials_examined` is what stops that.
    - **Conclusions with no inquiry.** A finding whose investigation was never
      recorded asserts that someone did the work without saying what work.
    """

    name: str
    version: int
    question: str
    """The question as asked, not as answered. Stated in the interrogative so a
    reader can tell whether the outcome actually addresses it."""

    outcome: InvestigationOutcome
    motivation: str = ""
    """Why it was opened — the observation, complaint or anomaly that triggered
    it. Without this, an investigation reads as curiosity rather than cause."""

    examined: Sequence[str] = ()
    """Artifacts inspected: methodologies, runs, evidence, assumptions. What a
    reader needs to judge whether the inquiry was thorough enough for its
    outcome to mean anything."""

    findings: Sequence[str] = ()
    """Finding references produced. Empty for a null, inconclusive or abandoned
    inquiry — which is the whole point of the type."""

    trials_examined: int = 0
    """Configurations evaluated during the inquiry.

    Counted whether or not they produced anything, because deflation does not
    care how a trial ended. This is the direct, mechanical connection between
    honest record-keeping and an honest Sharpe ratio."""

    resolution: str = ""
    """What was actually done. An investigation that concluded and changed
    nothing must say so rather than trailing off."""

    opened_at: Optional[str] = None
    closed_at: Optional[str] = None
    superseded_by: Optional[str] = None
    spec_version: str = KNOWLEDGE_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.outcome in _REQUIRES_FINDING and not self.findings:
            raise ValueError(
                f"{self.artifact_id}: outcome {self.outcome.value} claims a finding "
                "but cites none — an uncited conclusion cannot be reviewed"
            )
        if self.outcome in _FORBIDS_FINDING and self.findings:
            raise ValueError(
                f"{self.artifact_id}: outcome {self.outcome.value} cites "
                f"{len(self.findings)} finding(s). If the inquiry produced a "
                "finding, its outcome is FINDING_RECORDED; a null result that "
                "concluded something has mislabelled itself"
            )
        if self.outcome.is_null_result and not self.examined:
            raise ValueError(
                f"{self.artifact_id}: a null result must say what it examined. "
                "'We looked and found nothing' without naming what was looked at "
                "is an assertion, not a result"
            )
        if self.outcome.is_closed and not self.closed_at:
            raise ValueError(
                f"{self.artifact_id}: outcome {self.outcome.value} is a closed "
                "state but no closed_at is recorded"
            )

    @property
    def concept_id(self) -> str:
        return f"investigation/{self.name}"

    @property
    def artifact_id(self) -> str:
        return f"investigation/{self.name}@{self.version}"

    @property
    def is_open(self) -> bool:
        return self.outcome is InvestigationOutcome.PENDING

    @property
    def produced_nothing(self) -> bool:
        """Closed without a finding. Presentable, not a failure."""
        return self.outcome.is_closed and not self.findings

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "question": self.question,
            "outcome": self.outcome.value,
            "examined": sorted(self.examined),
            "findings": sorted(self.findings),
            "trials_examined": self.trials_examined,
            "superseded_by": self.superseded_by,
        }

    @property
    def content_hash(self) -> str:
        return _hash(self.canonical_form())

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "artifact_id": self.artifact_id,
            "concept_id": self.concept_id,
            "content_hash": self.content_hash,
            "motivation": self.motivation,
            "resolution": self.resolution,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "is_open": self.is_open,
            "produced_nothing": self.produced_nothing,
        }
