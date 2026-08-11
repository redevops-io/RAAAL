"""Declaration verification — every declaration must identify its realization.

A declarative system has two failure modes:

1. **Behaviour without declaration** — a hidden choice that moves a published
   number while living nowhere in an artifact. Releases 1–5 were largely about
   eliminating this one.
2. **Declaration without behaviour** — a declared rule that appears to take
   effect and does not. Arguably worse, because declaring it *creates* the belief
   that it is checked.

This module addresses the second. It resolves each rule's `enforced_by` path
against the methodology and checks the property the rule asserts, so a rule that
drifts from the field realizing it fails loudly instead of silently disagreeing.

Nothing here performs allocation work. Contracts execute, rules verify, policies
interpret — three separate responsibilities that had blurred together.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .spec import Methodology

_EXPECTATION = re.compile(r"^\s*(<=|>=|==|<|>)\s*(.+?)\s*$")


class UnresolvableRealization(ValueError):
    """Raised when a declaration names a realization that does not exist."""


@dataclass(frozen=True)
class VerificationResult:
    declaration_id: str
    kind: str                 # rule | universe_filter
    enforced_by: str
    expected: str
    observed: Any
    passed: bool
    detail: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "declaration_id": self.declaration_id,
            "kind": self.kind,
            "enforced_by": self.enforced_by,
            "expected": self.expected,
            "observed": self.observed,
            "passed": self.passed,
            "detail": self.detail,
        }


def resolve(methodology: Methodology, path: str) -> Any:
    """Resolve a dotted realization path against a methodology.

    Supported roots:

    * ``contract.<field>[.<key>]`` — output-contract fields, including nested
      mappings such as ``weight_bounds.max``
    * ``params.<name>`` — a declared parameter's value
    * ``pipeline.<step>`` — whether a pipeline stage is present
    * ``fallback_chain.<name>`` — whether a fallback is declared

    An unresolvable path is an error, not a ``None``. A rule pointing at a field
    that does not exist is exactly the drift this module exists to catch.
    """
    head, _, rest = path.partition(".")

    if head == "params":
        param = methodology.params.get(rest)
        if param is None:
            raise UnresolvableRealization(f"no parameter named {rest!r}")
        return param.value

    if head == "pipeline":
        return rest in methodology.pipeline

    if head == "fallback_chain":
        return rest in methodology.fallback_chain

    if head == "contract":
        target: Any = methodology.contract
        for part in rest.split("."):
            if isinstance(target, dict):
                if part not in target:
                    raise UnresolvableRealization(f"contract has no key {part!r} in {path!r}")
                target = target[part]
            else:
                if not hasattr(target, part):
                    raise UnresolvableRealization(f"contract has no field {part!r} in {path!r}")
                target = getattr(target, part)
        return target

    raise UnresolvableRealization(
        f"unsupported realization root {head!r} in {path!r}; expected one of "
        "contract, params, pipeline, fallback_chain"
    )


def _check(observed: Any, expected: str) -> tuple[bool, str]:
    if expected.strip() == "present":
        passed = bool(observed) if isinstance(observed, bool) else observed is not None
        return passed, ("present" if passed else "absent")

    match = _EXPECTATION.match(expected)
    if not match:
        return False, f"malformed expectation {expected!r}"

    operator, raw = match.groups()

    # Try numeric first; fall back to string equality for values like `sample`.
    try:
        wanted: Any = float(raw)
        actual: Any = float(observed)
        numeric = True
    except (TypeError, ValueError):
        wanted, actual, numeric = raw, str(observed), False

    if not numeric and operator != "==":
        return False, f"operator {operator} is not defined for non-numeric {raw!r}"

    outcome = {
        "<=": lambda: actual <= wanted,
        ">=": lambda: actual >= wanted,
        "==": lambda: actual == wanted,
        "<": lambda: actual < wanted,
        ">": lambda: actual > wanted,
    }[operator]()

    return outcome, f"{observed} {operator if outcome else 'not ' + operator} {raw}"


def verify(methodology: Methodology) -> List[VerificationResult]:
    """Check every declaration against the field that realizes it."""
    results: List[VerificationResult] = []

    for rule in methodology.rules:
        try:
            observed = resolve(methodology, rule.enforced_by)
        except UnresolvableRealization as exc:
            results.append(
                VerificationResult(
                    declaration_id=rule.id, kind="rule", enforced_by=rule.enforced_by,
                    expected=rule.expected, observed=None, passed=False,
                    detail=f"unresolvable realization: {exc}",
                )
            )
            continue

        passed, detail = _check(observed, rule.expected)
        results.append(
            VerificationResult(
                declaration_id=rule.id, kind="rule", enforced_by=rule.enforced_by,
                expected=rule.expected, observed=observed, passed=passed, detail=detail,
            )
        )

    for filt in methodology.universe_filters:
        try:
            observed = resolve(methodology, filt.enforced_by)
        except UnresolvableRealization as exc:
            results.append(
                VerificationResult(
                    declaration_id=filt.id, kind="universe_filter",
                    enforced_by=filt.enforced_by, expected="present", observed=None,
                    passed=False, detail=f"unresolvable realization: {exc}",
                )
            )
            continue

        passed, detail = _check(observed, "present")
        results.append(
            VerificationResult(
                declaration_id=filt.id, kind="universe_filter",
                enforced_by=filt.enforced_by, expected="present", observed=observed,
                passed=passed, detail=detail,
            )
        )

    return results


def unrealized_declarations(methodology: Methodology) -> List[VerificationResult]:
    """Declarations that do not hold. Empty is the only acceptable state.

    A non-empty result means the methodology asserts something its own fields do
    not support — the second failure mode of a declarative system.
    """
    return [r for r in verify(methodology) if not r.passed]
