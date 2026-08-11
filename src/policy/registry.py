"""On-disk registry for statistical policies.

Same shape as the methodology and protocol registries: one YAML per version under
`policies/`, versions coexisting. A published result cites the policy version it
was judged under, so raising a threshold later does not retroactively invalidate
past verdicts — it produces a new policy version and, if desired, a re-evaluation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

from .statistical_policy import Requirement, Severity, StatisticalPolicy

DEFAULT_ROOT = Path("policies")
_FILENAME = re.compile(r"^(?P<name>[a-z0-9][a-z0-9_-]*)@(?P<version>\d+)\.ya?ml$")


class PolicyRegistry:
    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def _iter_files(self) -> Iterator[Path]:
        if not self.root.exists():
            return
        for path in sorted(self.root.iterdir()):
            if _FILENAME.match(path.name):
                yield path

    def _load_file(self, path: Path) -> StatisticalPolicy:
        payload = yaml.safe_load(path.read_text()) or {}
        # `policies/` is loaded wholesale by filename shape, so any YAML dropped
        # here that happens to be called `something@1.yaml` is read as a policy.
        # One was, and the bare `KeyError: 'name'` surfaced as a 500 on every
        # page with no indication of which file caused it. Skipping silently
        # would be worse — a real policy with a typo would vanish and results
        # would be judged under a registry that quietly lost a member — so this
        # still fails, but says what it was reading and what was missing.
        missing = [key for key in ("name", "version") if key not in payload]
        if missing:
            raise ValueError(
                f"{path}: not a statistical policy — missing {', '.join(missing)}. "
                "Only statistical policies belong in the policy registry."
            )
        requirements = tuple(
            Requirement(
                code=r["code"],
                description=r.get("description", ""),
                severity=Severity(r.get("severity", "WARN")),
                threshold=r.get("threshold"),
                comparison=r.get("comparison", "gte"),
            )
            for r in payload.get("requirements", [])
        )
        policy = StatisticalPolicy(
            name=payload["name"],
            version=int(payload["version"]),
            title=payload.get("title", payload["name"]),
            requirements=requirements,
            rationale=payload.get("rationale", ""),
        )
        match = _FILENAME.match(path.name)
        assert match
        if match.group("name") != policy.name or int(match.group("version")) != policy.version:
            raise ValueError(
                f"{path.name} declares {policy.policy_id} — filename and content "
                "disagree about identity"
            )
        return policy

    def load_all(self) -> List[StatisticalPolicy]:
        return [self._load_file(p) for p in self._iter_files()]

    def versions(self, name: str) -> List[StatisticalPolicy]:
        return sorted((p for p in self.load_all() if p.name == name), key=lambda p: p.version)

    def get(self, name: str, version: Optional[int] = None) -> StatisticalPolicy:
        candidates = self.versions(name)
        if not candidates:
            raise KeyError(f"no statistical policy named {name!r} in {self.root}")
        if version is None:
            return candidates[-1]
        for p in candidates:
            if p.version == version:
                return p
        raise KeyError(
            f"{name}@{version} not found; available: {[p.version for p in candidates]}"
        )

    def resolve(self, reference: str) -> StatisticalPolicy:
        ref = reference.removeprefix("stat-policy/")
        if "@" in ref:
            name, _, version = ref.partition("@")
            return self.get(name, int(version))
        return self.get(ref)

    def names(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for p in self.load_all():
            out.setdefault(p.name, []).append(p.version)
        return {k: sorted(v) for k, v in sorted(out.items())}
