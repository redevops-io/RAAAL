"""The line between the public library and a private workspace.

The GitHub analogy is the right one and its failure mode is the instructive
part: forks leak. What keeps this boundary intact is not that the two live in
different UIs but that references may only run in one direction.

    private ──may reference──▶ public      a plan cites methodology/hrp@3
    public  ──may reference──▶ private     never

That single rule carries the legal position. The public library stays impersonal
because nothing in it can point at anything personal, and the invariant is
checkable rather than reviewed.

The optional contribution path needs its own treatment. A Mission cannot be
"promoted" to the library, because a Mission is a person's financial situation
and a methodology is a rule. What can cross is the **rule with the person
removed** — and the removal has to be enumerated rather than trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Sequence


class Visibility(str, Enum):
    PUBLIC_LIBRARY = "PUBLIC_LIBRARY"
    """Impersonal, published, citable. Methodologies, protocols, claims,
    evidence, findings, investigations, errata."""

    PRIVATE_WORKSPACE = "PRIVATE_WORKSPACE"
    """One person's. Intents, missions, their runs, their tracking."""


#: Artifact kinds and where they live. A kind absent from here has no declared
#: visibility, which `visibility_of` treats as an error rather than a default —
#: a new artifact type defaulting to public is the one mistake this file exists
#: to prevent.
VISIBILITY: Dict[str, Visibility] = {
    "methodology": Visibility.PUBLIC_LIBRARY,
    "protocol": Visibility.PUBLIC_LIBRARY,
    "calendar": Visibility.PUBLIC_LIBRARY,
    "stat-policy": Visibility.PUBLIC_LIBRARY,
    "claim": Visibility.PUBLIC_LIBRARY,
    "assumption": Visibility.PUBLIC_LIBRARY,
    "evidence": Visibility.PUBLIC_LIBRARY,
    "finding": Visibility.PUBLIC_LIBRARY,
    "investigation": Visibility.PUBLIC_LIBRARY,
    "erratum": Visibility.PUBLIC_LIBRARY,
    # Runtimes declare policy, never a person. A tax runtime says what the rules
    # are; the account balance it is applied to lives in the workspace.
    "tax": Visibility.PUBLIC_LIBRARY,
    "account": Visibility.PUBLIC_LIBRARY,
    "market_data": Visibility.PUBLIC_LIBRARY,
    "flow": Visibility.PUBLIC_LIBRARY,
    "corporate_action": Visibility.PUBLIC_LIBRARY,
    "intent": Visibility.PRIVATE_WORKSPACE,
    "mission": Visibility.PRIVATE_WORKSPACE,
    "plan-run": Visibility.PRIVATE_WORKSPACE,
}


class UndeclaredVisibility(KeyError):
    """An artifact kind with no declared side of the boundary."""


class BoundaryViolation(ValueError):
    """A public artifact tried to reference a private one."""


def visibility_of(reference: str) -> Visibility:
    kind = reference.split("/")[0]
    if kind not in VISIBILITY:
        raise UndeclaredVisibility(
            f"artifact kind {kind!r} has no declared visibility. Every kind must "
            "state which side of the public/private boundary it lives on before "
            "it can be referenced — defaulting to public is how the boundary is "
            "lost quietly"
        )
    return VISIBILITY[kind]


def check_reference(source: str, target: str) -> None:
    """Refuse a reference that would carry private data into public view."""
    if (visibility_of(source) is Visibility.PUBLIC_LIBRARY
            and visibility_of(target) is Visibility.PRIVATE_WORKSPACE):
        raise BoundaryViolation(
            f"{source} may not reference {target}: a public artifact citing a "
            "private one makes the library personal, which is the property the "
            "publisher's position depends on"
        )


def check_all(source: str, targets: Sequence[str]) -> None:
    for target in targets:
        check_reference(source, target)


#: Everything that must be removed before a rule extracted from a Mission can be
#: proposed to the library. Enumerated rather than trusted, because "we strip
#: personal data" is a claim and this is a list.
PERSONAL_FIELDS = (
    "flows",
    "starting_capital",
    "tax_treatment",
    "intent_ref",
    "title",
    "provenance",
)

#: Keys that may never appear anywhere in an extracted rule, at any depth. The
#: field-level strip above removes the places personal data is *supposed* to
#: live; this catches it where it is not supposed to be, which is where it will
#: actually be. A rule carrying `employer` in a nested condition passes a
#: field-name check and fails this one.
PROHIBITED_KEYS = frozenset({
    "income", "salary", "account_value", "account_values", "balance",
    "employer", "company", "vesting_schedule", "vesting", "grant",
    "tax_rate", "withholding", "age", "date_of_birth", "retirement_age",
    "contribution", "contribution_amount", "holdings", "positions",
    "user", "user_id", "email", "name_of_investor", "identity",
    "run_id", "run_ids", "mission_ref", "intent_ref", "plan_id",
})

#: Value patterns that indicate a private reference regardless of the key it
#: hides behind.
_PRIVATE_PREFIXES = ("mission/", "intent/", "plan-run/")


class PrivacyLeak(ValueError):
    """Personal data survived extraction."""


def scan_for_personal_data(payload: Any, path: str = "rule") -> List[str]:
    """Walk an extracted rule and report every personal key or private reference.

    Deterministic and exhaustive rather than sampled. A privacy check that
    inspects the top level only is the check that gets passed by the payload that
    matters.
    """
    leaks: List[str] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            here = f"{path}.{key}"
            if str(key).lower() in PROHIBITED_KEYS:
                leaks.append(f"{here}: prohibited key {key!r}")
            leaks.extend(scan_for_personal_data(value, here))
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            leaks.extend(scan_for_personal_data(item, f"{path}[{i}]"))
    elif isinstance(payload, str):
        if any(payload.startswith(prefix) for prefix in _PRIVATE_PREFIXES):
            leaks.append(f"{path}: reference to a private artifact {payload!r}")

    return leaks


@dataclass(frozen=True)
class Extraction:
    """What could be proposed to the library, and what had to come off first."""

    rule: Dict[str, Any]
    stripped: Sequence[str]
    blockers: Sequence[str]
    leaks: Sequence[str] = ()

    @property
    def proposable(self) -> bool:
        return not self.blockers and not self.leaks

    def verify(self) -> None:
        """Raise unless the rule is clean. The gate before public authoring."""
        if self.leaks:
            raise PrivacyLeak(
                "extracted rule still carries personal data: "
                + "; ".join(self.leaks)
            )

    source_scenario: str = ""
    rule_hash: str = ""

    def report(self) -> Dict[str, Any]:
        """A retained record, whether or not the extraction succeeded.

        A failed extraction is deleted at everyone's peril: it is the clearest
        available evidence of a compiler or schema defect, and the case where
        personal data reached a field it was never supposed to occupy is exactly
        the one worth keeping.
        """
        return {
            "source_scenario": self.source_scenario,
            "rule_hash": self.rule_hash,
            "fields_removed": list(self.stripped),
            "values_removed": [l for l in self.leaks if "prohibited key" in l],
            "prohibited_references_found": [
                l for l in self.leaks if "private artifact" in l
            ],
            "public_boundary_check": "PASS" if not self.leaks else "FAIL",
            "eligible_for_authoring": self.proposable,
            "blockers": list(self.blockers),
        }

    def to_json(self) -> Dict[str, Any]:
        return {"rule": self.rule, "stripped": list(self.stripped),
                "blockers": list(self.blockers), "leaks": list(self.leaks),
                "proposable": self.proposable,
                "extraction_report": self.report()}


def extract_rule(mission) -> Extraction:
    """Separate the rule from the person, for the optional contribution path.

    This is deliberately not called `promote`. A Mission does not become a
    methodology — a *rule* is lifted out of it and must then be authored as one,
    with its own claims, assumptions and evidence. What a person did with their
    salary is not research, and the fact that it worked for them is the weakest
    possible evidence that it works.
    """
    blockers: List[str] = []
    if not mission.provenance.is_complete:
        blockers.append(
            "the plan still has unconfirmed inferences or open questions, so it "
            "is not yet a rule anyone stated"
        )
    if not mission.events:
        blockers.append(
            "the plan has no event program — there is no rule here, only a "
            "contribution schedule"
        )

    rule = {
        "events": list(mission.events),
        "constraints": sorted(mission.constraints),
        "objective": mission.objective.value,
    }
    return Extraction(
        rule=rule,
        stripped=PERSONAL_FIELDS,
        blockers=tuple(blockers),
        leaks=tuple(scan_for_personal_data(rule)),
        source_scenario=getattr(mission, "artifact_id", ""),
        rule_hash=_hash_rule(rule),
    )


def _hash_rule(rule: Dict[str, Any]) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(rule, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
