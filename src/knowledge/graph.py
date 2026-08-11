"""Queries over the knowledge graph.

The point of making reasoning addressable is the questions it lets you ask.
Before this, "which methodologies depend on the sample-covariance assumption?"
was a grep for the string ``EWMA`` across YAML — which finds the ones that
mention it and misses the ones that inherit it.

Claim status is **derived from evidence**, never stored. Storing it would let the
recorded status drift from the evidence that justifies it, which is the same
class of error as storing a content hash next to the content.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .artifacts import Claim, ClaimStatus, Evidence, Stance


@dataclass
class ClaimAssessment:
    """A claim's standing, derived from the evidence that bears on it."""

    claim: Claim
    supporting: List[Evidence] = field(default_factory=list)
    contradicting: List[Evidence] = field(default_factory=list)
    qualifying: List[Evidence] = field(default_factory=list)
    status: ClaimStatus = ClaimStatus.UNASSESSED

    def to_json(self) -> Dict[str, Any]:
        return {
            "claim": self.claim.to_json(),
            "status": self.status.value,
            "supporting": [e.to_json() for e in self.supporting],
            "contradicting": [e.to_json() for e in self.contradicting],
            "qualifying": [e.to_json() for e in self.qualifying],
            "note": (
                "Status is derived from current evidence, not stored. Contested is a "
                "state, not a defect — it records that the question is open."
            ),
        }


_STRENGTH_ORDER = {"strong": 3, "moderate": 2, "weak": 1, "anecdotal": 0}


def assess_claim(claim: Claim, evidence: Sequence[Evidence]) -> ClaimAssessment:
    """Derive a claim's status from the evidence currently bearing on it.

    A claim with contradicting evidence is CONTESTED rather than REFUTED unless
    the contradiction is both current and at least as strong as the support —
    a single weak dissent should not overturn a well-supported claim, and a
    strong replication failure should not be filed as a footnote.
    """
    # Both conditions are required. Written as a nested expression rather than
    # `A and B or C`, which parses as `(A and B) or C` and silently let
    # invalidated evidence keep counting.
    relevant = [
        e for e in evidence
        if e.is_current and _references(e.about, claim.concept_id, claim.artifact_id)
    ]

    supporting = [e for e in relevant if e.stance is Stance.SUPPORTS]
    contradicting = [e for e in relevant if e.stance is Stance.CONTRADICTS]
    qualifying = [e for e in relevant if e.stance is Stance.QUALIFIES]

    if claim.superseded_by:
        status = ClaimStatus.SUPERSEDED
    elif not relevant:
        status = ClaimStatus.UNASSESSED
    elif contradicting:
        best_against = max(_STRENGTH_ORDER.get(e.strength, 0) for e in contradicting)
        best_for = max(
            (_STRENGTH_ORDER.get(e.strength, 0) for e in supporting), default=-1
        )
        status = ClaimStatus.REFUTED if best_against > best_for else ClaimStatus.CONTESTED
    elif qualifying:
        status = ClaimStatus.QUALIFIED
    else:
        status = ClaimStatus.SUPPORTED

    return ClaimAssessment(
        claim=claim,
        supporting=supporting,
        contradicting=contradicting,
        qualifying=qualifying,
        status=status,
    )


def _references(value: str, concept_id: str, artifact_id: str) -> bool:
    """Whether a reference points at this artifact, pinned or unpinned."""
    return value in (concept_id, artifact_id)


class KnowledgeGraph:
    """Read-only views across methodologies, claims, assumptions and evidence."""

    def __init__(self, methodologies, claims, assumptions, evidence, findings=(),
                 investigations=()) -> None:
        self.methodologies = list(methodologies)
        self.claims = list(claims)
        self.assumptions = list(assumptions)
        self.evidence = list(evidence)
        self.findings = list(findings)
        self.investigations = list(investigations)

    # ---- the query the grep could not answer ------------------------------

    def methodologies_depending_on_assumption(self, reference: str) -> List[Any]:
        """Every methodology that rests on an assumption, directly or via a claim.

        Indirect dependency matters: a methodology that references a claim which
        itself depends on an assumption inherits that dependency, and a search
        that only looked at the methodology's own list would miss it.
        """
        target = self._resolve_assumption(reference)
        if target is None:
            return []

        direct = {
            m.version_id for m in self.methodologies
            if any(
                _references(a, target.concept_id, target.artifact_id)
                for a in getattr(m, "assumptions_ref", ())
            )
        }

        via_claims = set()
        for claim in self.claims:
            if not any(
                _references(a, target.concept_id, target.artifact_id)
                for a in claim.depends_on
            ):
                continue
            for m in self.methodologies:
                if any(
                    _references(c, claim.concept_id, claim.artifact_id)
                    for c in getattr(m, "claims_ref", ())
                ):
                    via_claims.add(m.version_id)

        affected = direct | via_claims
        return [m for m in self.methodologies if m.version_id in affected]

    def methodologies_referencing_claim(self, reference: str) -> List[Any]:
        target = self._resolve_claim(reference)
        if target is None:
            return []
        return [
            m for m in self.methodologies
            if any(
                _references(c, target.concept_id, target.artifact_id)
                for c in getattr(m, "claims_ref", ())
            )
        ]

    def claims_for_methodology(self, methodology) -> List[ClaimAssessment]:
        out = []
        for reference in getattr(methodology, "claims_ref", ()):
            claim = self._resolve_claim(reference)
            if claim is not None:
                out.append(assess_claim(claim, self.evidence))
        return out

    def assumptions_for_methodology(self, methodology) -> List[Any]:
        """Assumptions the methodology declares, plus those inherited via claims."""
        seen: Dict[str, Any] = {}

        for reference in getattr(methodology, "assumptions_ref", ()):
            a = self._resolve_assumption(reference)
            if a:
                seen[a.artifact_id] = a

        for reference in getattr(methodology, "claims_ref", ()):
            claim = self._resolve_claim(reference)
            if claim is None:
                continue
            for inherited in claim.depends_on:
                a = self._resolve_assumption(inherited)
                if a:
                    seen.setdefault(a.artifact_id, a)

        return sorted(seen.values(), key=lambda a: a.artifact_id)

    def unvalidated_assumptions(self) -> List[Any]:
        """Assumptions nothing checks.

        Every defect in the project's erratum history was one of these before it
        was found, so this list is the standing register of where the next one
        is most likely to be.
        """
        return [a for a in self.assumptions if not a.is_validated]

    def contested_claims(self) -> List[ClaimAssessment]:
        assessed = [assess_claim(c, self.evidence) for c in self.claims]
        return [
            a for a in assessed
            if a.status in {ClaimStatus.CONTESTED, ClaimStatus.REFUTED, ClaimStatus.QUALIFIED}
        ]

    def impact_of_claim_change(self, reference: str) -> Dict[str, Any]:
        """What would need review if this claim's status changed.

        The query Discovery Runtime needs: new evidence arrives, and the question
        is which published work rests on the claim it bears on.
        """
        claim = self._resolve_claim(reference)
        if claim is None:
            return {"claim": reference, "found": False}

        affected = self.methodologies_referencing_claim(reference)
        return {
            "claim": claim.artifact_id,
            "found": True,
            "status": assess_claim(claim, self.evidence).status.value,
            "affected_methodologies": [m.version_id for m in affected],
            "affected_count": len(affected),
            "assumptions_inherited": list(claim.depends_on),
        }

    # ---- resolution -------------------------------------------------------

    def _resolve_claim(self, reference: str) -> Optional[Claim]:
        return self._resolve(reference, self.claims, "claim")

    def _resolve_assumption(self, reference: str):
        return self._resolve(reference, self.assumptions, "assumption")

    @staticmethod
    def _resolve(reference: str, pool: Sequence[Any], prefix: str):
        ref = reference.removeprefix(f"{prefix}/")
        if "@" in ref:
            name, _, version = ref.partition("@")
            for item in pool:
                if item.name == name and item.version == int(version):
                    return item
            return None
        matches = [item for item in pool if item.name == ref]
        return max(matches, key=lambda i: i.version) if matches else None


    # ---- findings ---------------------------------------------------------

    def findings_affecting(self, reference: str) -> List[Any]:
        """Every finding whose conclusion touches this artifact.

        The question a reviewer asks arriving at a methodology page: *what has
        already been concluded about this?* Before findings existed, the answer
        lived in prose scattered across change rationales and erratum documents.
        """
        return [
            f for f in self.findings
            if any(_matches_reference(i.target, reference) for i in f.impacts)
        ]

    def provisional_findings(self) -> List[Any]:
        """Conclusions stated but not yet settled."""
        from .artifacts import FindingStatus

        return [f for f in self.findings if f.status is FindingStatus.OPEN]

    def unevidenced_findings(self) -> List[Any]:
        """A finding with no supporting evidence is an opinion."""
        return [f for f in self.findings if not f.is_evidenced]

    def finding_provenance(self, finding) -> Dict[str, Any]:
        """Resolve a finding's evidence and impacted artifacts into a summary."""
        evidence = [
            e for e in self.evidence
            if any(_matches_reference(e.artifact_id, ref) for ref in finding.supported_by)
        ]
        return {
            "finding": finding.to_json(),
            "evidence": [e.to_json() for e in evidence],
            "impacted_claims": finding.targets("claim"),
            "impacted_methodologies": finding.targets("methodology"),
            "impacted_assumptions": finding.targets("assumption"),
            "impacted_errata": finding.targets("erratum"),
        }


    # ---- investigations ---------------------------------------------------

    def open_inquiries(self) -> List[Any]:
        """Questions currently being asked. Not findings — questions."""
        return [i for i in self.investigations if i.is_open]

    def null_results(self) -> List[Any]:
        """Inquiries that concluded without a finding.

        Surfaced deliberately and prominently. A library that shows only the
        investigations that found something reports a filtered history, and the
        filter runs in exactly the direction that flatters the platform.
        """
        return [i for i in self.investigations if i.produced_nothing]

    def investigation_for_finding(self, reference: str) -> Optional[Any]:
        """The inquiry that produced a conclusion, if one was recorded."""
        for investigation in self.investigations:
            if any(_matches_reference(f, reference) for f in investigation.findings):
                return investigation
        return None

    def unattributed_findings(self) -> List[Any]:
        """Conclusions with no recorded inquiry behind them.

        The mirror of `null_results`: one hides work that produced nothing, this
        hides the work behind something that was produced. Both are declarations
        without a realization, which the platform treats as a defect everywhere
        else and should treat as one here.
        """
        return [
            f for f in self.findings
            if self.investigation_for_finding(f.artifact_id) is None
        ]

    def recorded_trials(self, *, prefix: str = "methodology/") -> Dict[str, int]:
        """Trials each methodology accrued through recorded investigations.

        This is the mechanical link between honest record-keeping and an honest
        Sharpe ratio: deflation counts every configuration that was tried,
        including the ones that were tried and dropped. An investigation that
        closes with NO_EFFECT_FOUND still spent its trials.
        """
        counts: Dict[str, int] = {}
        for investigation in self.investigations:
            if not investigation.trials_examined:
                continue
            for reference in investigation.examined:
                if reference.startswith(prefix):
                    counts[reference] = counts.get(reference, 0) + investigation.trials_examined
        return counts

    def investigation_provenance(self, investigation) -> Dict[str, Any]:
        """An inquiry with its findings and examined artifacts resolved."""
        findings = [
            f for f in self.findings
            if any(_matches_reference(f.artifact_id, ref) for ref in investigation.findings)
        ]
        return {
            "investigation": investigation.to_json(),
            "findings": [f.to_json() for f in findings],
            "examined_methodologies": [
                r for r in investigation.examined if r.startswith("methodology/")
            ],
            "examined_evidence": [
                r for r in investigation.examined if r.startswith("evidence/")
            ],
            "examined_assumptions": [
                r for r in investigation.examined if r.startswith("assumption/")
            ],
        }


def _matches_reference(target: str, reference: str) -> bool:
    """Whether two references point at the same artifact, pinned or not."""
    if target == reference:
        return True
    base_t = target.split("@")[0]
    base_r = reference.split("@")[0]
    return base_t == base_r and ("@" not in target or "@" not in reference)
