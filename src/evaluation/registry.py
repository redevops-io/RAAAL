"""On-disk registry for evaluation protocols.

Mirrors `methodology.registry`: one YAML per version under `protocols/`, named
`<name>@<version>.yaml`, versions coexisting rather than replacing each other.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

from .protocol import EvaluationProtocol, from_dict

DEFAULT_ROOT = Path("protocols")
_FILENAME = re.compile(r"^(?P<name>[a-z0-9][a-z0-9_-]*)@(?P<version>\d+)\.ya?ml$")


class ProtocolRegistry:
    """Loads and resolves evaluation-protocol versions from a directory."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def _iter_files(self) -> Iterator[Path]:
        if not self.root.exists():
            return
        for path in sorted(self.root.iterdir()):
            if _FILENAME.match(path.name):
                yield path

    def _load_file(self, path: Path) -> EvaluationProtocol:
        payload = yaml.safe_load(path.read_text()) or {}
        protocol = from_dict(payload)
        match = _FILENAME.match(path.name)
        assert match
        if (
            match.group("name") != protocol.name
            or int(match.group("version")) != protocol.version
        ):
            raise ValueError(
                f"{path.name} declares {protocol.protocol_id} — filename and "
                "content disagree about identity"
            )
        return protocol

    def load_all(self) -> List[EvaluationProtocol]:
        return [self._load_file(p) for p in self._iter_files()]

    def versions(self, name: str) -> List[EvaluationProtocol]:
        return sorted(
            (p for p in self.load_all() if p.name == name), key=lambda p: p.version
        )

    def get(self, name: str, version: Optional[int] = None) -> EvaluationProtocol:
        candidates = self.versions(name)
        if not candidates:
            raise KeyError(f"no evaluation protocol named {name!r} in {self.root}")
        if version is None:
            return candidates[-1]
        for p in candidates:
            if p.version == version:
                return p
        raise KeyError(
            f"{name}@{version} not found; available: {[p.version for p in candidates]}"
        )

    def resolve(self, reference: str) -> EvaluationProtocol:
        ref = reference.removeprefix("protocol/")
        if "@" in ref:
            name, _, version = ref.partition("@")
            return self.get(name, int(version))
        return self.get(ref)

    def names(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for p in self.load_all():
            out.setdefault(p.name, []).append(p.version)
        return {k: sorted(v) for k, v in sorted(out.items())}
