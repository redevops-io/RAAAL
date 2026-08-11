"""On-disk registries for claims, assumptions and evidence.

Same shape as every other registry: one YAML per version, versions coexisting,
filename and content must agree about identity.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type

import yaml

from .artifacts import (
    Assumption,
    AssumptionKind,
    Claim,
    Evidence,
    EvidenceKind,
    Finding,
    FindingStatus,
    Impact,
    ImpactRelation,
    Investigation,
    InvestigationOutcome,
    Realization,
    Stance,
)

_FILENAME = re.compile(r"^(?P<name>[a-z0-9][a-z0-9_-]*)@(?P<version>\d+)\.ya?ml$")


class _BaseRegistry:
    root: Path
    prefix: str

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _iter_files(self) -> Iterator[Path]:
        if not self.root.exists():
            return
        for path in sorted(self.root.iterdir()):
            if _FILENAME.match(path.name):
                yield path

    def _parse(self, payload: Dict[str, Any]):
        raise NotImplementedError

    def _load_file(self, path: Path):
        payload = yaml.safe_load(path.read_text()) or {}
        item = self._parse(payload)
        match = _FILENAME.match(path.name)
        assert match
        if match.group("name") != item.name or int(match.group("version")) != item.version:
            raise ValueError(
                f"{path.name} declares {item.artifact_id} — filename and content "
                "disagree about identity"
            )
        return item

    def load_all(self) -> List[Any]:
        return [self._load_file(p) for p in self._iter_files()]

    def versions(self, name: str) -> List[Any]:
        return sorted((i for i in self.load_all() if i.name == name), key=lambda i: i.version)

    def get(self, name: str, version: Optional[int] = None):
        candidates = self.versions(name)
        if not candidates:
            raise KeyError(f"no {self.prefix} named {name!r} in {self.root}")
        if version is None:
            return candidates[-1]
        for item in candidates:
            if item.version == version:
                return item
        raise KeyError(
            f"{name}@{version} not found; available: {[i.version for i in candidates]}"
        )

    def resolve(self, reference: str):
        ref = reference.removeprefix(f"{self.prefix}/")
        if "@" in ref:
            name, _, version = ref.partition("@")
            return self.get(name, int(version))
        return self.get(ref)

    def names(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for item in self.load_all():
            out.setdefault(item.name, []).append(item.version)
        return {k: sorted(v) for k, v in sorted(out.items())}


class ClaimRegistry(_BaseRegistry):
    prefix = "claim"

    def __init__(self, root: Path | str = "claims") -> None:
        super().__init__(root)

    def _parse(self, payload: Dict[str, Any]) -> Claim:
        return Claim(
            name=payload["name"],
            version=int(payload["version"]),
            statement=payload["statement"],
            scope=payload.get("scope", ""),
            depends_on=tuple(payload.get("depends_on", ())),
            superseded_by=payload.get("superseded_by"),
            derived_from=payload.get("derived_from"),
            change_rationale=payload.get("change_rationale", ""),
        )


class AssumptionRegistry(_BaseRegistry):
    prefix = "assumption"

    def __init__(self, root: Path | str = "assumptions") -> None:
        super().__init__(root)

    def _parse(self, payload: Dict[str, Any]) -> Assumption:
        return Assumption(
            name=payload["name"],
            version=int(payload["version"]),
            statement=payload["statement"],
            kind=AssumptionKind(payload["kind"]),
            realized_by=tuple(
                Realization(
                    artifact_kind=r["artifact_kind"],
                    field=r["field"],
                    value=r.get("value"),
                )
                for r in payload.get("realized_by", [])
            ),
            risk=payload.get("risk", ""),
            validated_by=tuple(payload.get("validated_by", ())),
            history=tuple(payload.get("history", ())),
            superseded_by=payload.get("superseded_by"),
        )


class EvidenceRegistry(_BaseRegistry):
    prefix = "evidence"

    def __init__(self, root: Path | str = "evidence") -> None:
        super().__init__(root)

    def _parse(self, payload: Dict[str, Any]) -> Evidence:
        return Evidence(
            name=payload["name"],
            version=int(payload["version"]),
            kind=EvidenceKind(payload["kind"]),
            about=payload["about"],
            stance=Stance(payload["stance"]),
            summary=payload.get("summary", ""),
            identifier=payload.get("identifier", ""),
            strength=payload.get("strength", "moderate"),
            valid_from=payload.get("valid_from"),
            valid_to=payload.get("valid_to"),
            produced_by=tuple(payload.get("produced_by", ())),
        )


class FindingRegistry(_BaseRegistry):
    prefix = "finding"

    def __init__(self, root: Path | str = "findings") -> None:
        super().__init__(root)

    def _parse(self, payload: Dict[str, Any]) -> Finding:
        return Finding(
            name=payload["name"],
            version=int(payload["version"]),
            statement=payload["statement"],
            status=FindingStatus(payload.get("status", "OPEN")),
            supported_by=tuple(payload.get("supported_by", ())),
            impacts=tuple(
                Impact(
                    target=i["target"],
                    relation=ImpactRelation(i["relation"]),
                    detail=i.get("detail", ""),
                )
                for i in payload.get("impacts", [])
            ),
            resolution=payload.get("resolution", ""),
            opened_at=payload.get("opened_at"),
            concluded_at=payload.get("concluded_at"),
            superseded_by=payload.get("superseded_by"),
        )


class InvestigationRegistry(_BaseRegistry):
    """Inquiries on disk, including the ones that produced nothing.

    A registry that only held successful investigations would be a marketing
    artifact. The validation lives on `Investigation` itself, so a malformed
    outcome fails at load rather than at render.
    """

    prefix = "investigation"

    def __init__(self, root: Path | str = "investigations") -> None:
        super().__init__(root)

    def _parse(self, payload: Dict[str, Any]) -> Investigation:
        return Investigation(
            name=payload["name"],
            version=int(payload["version"]),
            question=payload["question"],
            outcome=InvestigationOutcome(payload.get("outcome", "PENDING")),
            motivation=payload.get("motivation", ""),
            examined=tuple(payload.get("examined", ())),
            findings=tuple(payload.get("findings", ())),
            trials_examined=int(payload.get("trials_examined", 0)),
            resolution=payload.get("resolution", ""),
            opened_at=payload.get("opened_at"),
            closed_at=payload.get("closed_at"),
            superseded_by=payload.get("superseded_by"),
        )
