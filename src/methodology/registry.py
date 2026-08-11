"""On-disk methodology registry.

Methodologies live as YAML under `methodologies/`, one file per version, named
`<concept>@<version>.yaml`. Versions coexist rather than replacing each other —
that is the whole point of a version id, and it is what lets a consumer pin.

dbt's own guidance is that two or three live versions is the practical ceiling
before sunsetting; `deprecation_date` is how a version is retired without being
deleted.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

from .spec import Methodology, from_dict

DEFAULT_ROOT = Path("methodologies")
_FILENAME = re.compile(r"^(?P<concept>[a-z0-9][a-z0-9_-]*)@(?P<version>\d+)\.ya?ml$")


class MethodologyRegistry:
    """Loads and resolves methodology versions from a directory."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    # ---- loading ----------------------------------------------------------

    def _iter_files(self) -> Iterator[Path]:
        if not self.root.exists():
            return
        for path in sorted(self.root.iterdir()):
            if _FILENAME.match(path.name):
                yield path

    def load_all(self) -> List[Methodology]:
        return [self._load_file(p) for p in self._iter_files()]

    def _load_file(self, path: Path) -> Methodology:
        payload = yaml.safe_load(path.read_text()) or {}
        methodology = from_dict(payload)

        # The filename is metadata too; a mismatch means one of them is a lie.
        match = _FILENAME.match(path.name)
        assert match  # guarded by _iter_files
        if (
            match.group("concept") != methodology.concept
            or int(match.group("version")) != methodology.version
        ):
            raise ValueError(
                f"{path.name} declares {methodology.version_id} — filename and "
                "content disagree about identity"
            )
        return methodology

    # ---- resolution -------------------------------------------------------

    def versions(self, concept: str) -> List[Methodology]:
        """All versions of a concept, oldest first."""
        return sorted(
            (m for m in self.load_all() if m.concept == concept),
            key=lambda m: m.version,
        )

    def get(self, concept: str, version: Optional[int] = None) -> Methodology:
        """Resolve a concept to a version.

        `version=None` means "latest", mirroring dbt's unpinned `ref()`. Pinning
        is always available and is what a published result MUST record.
        """
        candidates = self.versions(concept)
        if not candidates:
            raise KeyError(f"no methodology named {concept!r} in {self.root}")
        if version is None:
            return candidates[-1]
        for m in candidates:
            if m.version == version:
                return m
        available = [m.version for m in candidates]
        raise KeyError(f"{concept}@{version} not found; available: {available}")

    def resolve(self, reference: str) -> Methodology:
        """Resolve ``concept`` or ``concept@version`` or a full version id."""
        ref = reference.removeprefix("methodology/")
        if "@" in ref:
            concept, _, version = ref.partition("@")
            return self.get(concept, int(version))
        return self.get(ref)

    def concepts(self) -> Dict[str, List[int]]:
        """Concept id -> available version numbers."""
        out: Dict[str, List[int]] = {}
        for m in self.load_all():
            out.setdefault(m.concept, []).append(m.version)
        return {k: sorted(v) for k, v in sorted(out.items())}

    # ---- writing ----------------------------------------------------------

    def save(self, methodology: Methodology, *, overwrite: bool = False) -> Path:
        """Write a version to disk.

        Refuses to overwrite by default: a published version is immutable, and
        silently rewriting one breaks every result that cites it.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{methodology.concept}@{methodology.version}.yaml"
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path.name} exists — published versions are immutable. "
                "Use revise() to mint the next version."
            )
        payload = methodology.to_json()
        # Derived fields are recomputed on load; storing them invites drift.
        for derived in ("concept_id", "version_id", "content_hash"):
            payload.pop(derived, None)
        path.write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=True))
        return path
