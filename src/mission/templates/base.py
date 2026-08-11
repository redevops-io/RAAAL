"""Life-event templates: the part that cannot be generated.

A template encodes knowledge that is specific, checkable, and wrong in expensive
ways if guessed — vesting cliffs, blackout windows, withholding rates, exercise
mechanics. That is exactly why it is the durable part of the product and exactly
why it needs the same discipline as everything else here.

`citations` is what makes a template defensible rather than plausible. A template
that cites the actual withholding rule is auditable; one that does not is a guess
in a nice interface, and the interface makes it *more* convincing, not less.

Every assumption declares what realizes it, so the same verifier that catches a
methodology declaring a rule its executor ignores catches a template declaring a
behaviour its event program does not implement.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


class InputKind(str, Enum):
    DATE = "DATE"
    DATE_LIST = "DATE_LIST"
    SHARES = "SHARES"
    MONEY = "MONEY"
    RATE = "RATE"
    TICKER = "TICKER"
    CHOICE = "CHOICE"
    WINDOW_LIST = "WINDOW_LIST"


@dataclass(frozen=True)
class TemplateInput:
    """One thing the user must supply, typed and with its unit stated.

    `unit` is not cosmetic. "0.22" and "22" are the same withholding rate typed
    two ways and differ by a factor of a hundred in the answer, and a field that
    does not say which it wants will receive both.
    """

    name: str
    kind: InputKind
    label: str
    unit: str = ""
    required: bool = True
    choices: Sequence[str] = ()
    default: Any = None
    why_it_matters: str = ""

    def validate(self, value: Any) -> List[str]:
        problems: List[str] = []
        if value is None or value == "" or value == []:
            if self.required and self.default is None:
                problems.append(f"{self.name} is required: {self.label}")
            return problems

        if self.kind is InputKind.RATE:
            rate = float(value)
            if not 0.0 <= rate <= 1.0:
                problems.append(
                    f"{self.name} must be a fraction between 0 and 1 (got {rate}). "
                    f"A rate typed as a percentage is a hundredfold error in the "
                    f"answer, so it is refused rather than guessed at."
                )
        if self.kind is InputKind.SHARES and float(value) <= 0:
            problems.append(f"{self.name} must be positive")
        if self.kind is InputKind.MONEY and float(value) < 0:
            problems.append(f"{self.name} may not be negative")
        if self.kind is InputKind.CHOICE and value not in self.choices:
            problems.append(
                f"{self.name} must be one of {sorted(self.choices)}, got {value!r}")
        return problems

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind.value, "label": self.label,
                "unit": self.unit, "required": self.required,
                "choices": list(self.choices), "default": self.default,
                "why_it_matters": self.why_it_matters}


@dataclass(frozen=True)
class TemplateCitation:
    """Where a rule in this template comes from."""

    identifier: str
    title: str
    supports: str
    effective_from: str = ""
    url: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"identifier": self.identifier, "title": self.title,
                "supports": self.supports, "effective_from": self.effective_from,
                "url": self.url}


@dataclass(frozen=True)
class TemplateAssumption:
    """A declared behaviour, and the thing that actually performs it."""

    name: str
    statement: str
    realized_by: str
    """Where in the event program this happens. A declaration with no named
    realization is the defect the contract verifier exists to catch."""

    risk: str = ""
    citation: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "statement": self.statement,
                "realized_by": self.realized_by, "risk": self.risk,
                "citation": self.citation}


@dataclass(frozen=True)
class TemplateLimitation:
    """Something this template does not model, stated rather than discovered."""

    name: str
    statement: str

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "statement": self.statement}


@dataclass(frozen=True)
class MissionTemplate:
    """A reusable life-event workflow."""

    name: str
    version: int
    title: str
    question: str
    inputs: Sequence[TemplateInput]
    assumptions: Sequence[TemplateAssumption] = ()
    citations: Sequence[TemplateCitation] = ()
    limitations: Sequence[TemplateLimitation] = ()

    @property
    def artifact_id(self) -> str:
        return f"template/{self.name}@{self.version}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps({
            "name": self.name, "version": self.version,
            "inputs": [i.to_json() for i in self.inputs],
            "assumptions": [a.to_json() for a in self.assumptions],
        }, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    def validate(self, values: Mapping[str, Any]) -> List[str]:
        problems: List[str] = []
        for spec in self.inputs:
            problems.extend(spec.validate(values.get(spec.name)))
        unknown = set(values) - {i.name for i in self.inputs}
        if unknown:
            problems.append(
                f"unrecognised input(s) {sorted(unknown)}. A template that "
                "silently accepts fields it does not use will appear to honour "
                "them"
            )
        return problems

    def unrealized_assumptions(self, implemented: Sequence[str]) -> List[str]:
        """Assumptions whose named realization does not exist.

        The same check the methodology verifier runs, pointed at templates:
        declaring a behaviour and not implementing it is the failure mode, and
        the interface makes an unimplemented declaration *more* convincing.
        """
        available = set(implemented)
        return [a.name for a in self.assumptions if a.realized_by not in available]

    def modelling_scope(self) -> Dict[str, Any]:
        """What this template models and what it deliberately does not.

        Rendered side by side rather than as a footnote. Very few products in
        this category distinguish "we simulate the employer withholding shares"
        from "we compute your tax", and a reader who cannot see the line will
        assume the more flattering side of it.
        """
        return {
            "modelled": [
                {"name": a.name, "statement": a.statement,
                 "realized_by": a.realized_by, "citation": a.citation,
                 "risk": a.risk}
                for a in self.assumptions
            ],
            "not_modelled": [
                {"name": l.name, "reason": l.statement} for l in self.limitations
            ],
            "note": (
                "Everything on the left is mechanical: it is what happens to the "
                "award, and this template performs it. Everything on the right "
                "depends on facts about you that have not been stated, and "
                "assuming them would be worse than reporting without them."
            ),
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "modelling_scope": self.modelling_scope(),
            "title": self.title,
            "question": self.question,
            "inputs": [i.to_json() for i in self.inputs],
            "assumptions": [a.to_json() for a in self.assumptions],
            "citations": [c.to_json() for c in self.citations],
            "limitations": [l.to_json() for l in self.limitations],
        }
