"""What the user declared about a vest, before anything is computed.

    description -> stage-1 recognition -> template hint -> RSUDeclaration
                -> confirmation card -> run

**Declarations only.** Delivered shares, proceeds, projected concentration,
realized weights and expected performance are execution outputs and are refused
here by name. A confirmation screen showing a computed figure is showing an
answer to a question the user has not yet agreed to ask, and the number would
carry none of the caveats a real result carries.

**Version pins travel with the declaration.** The card and the worksheet are
both projections and neither computes, so the only way they can disagree about
one run is if the declarations moved between confirming and executing. That is a
refusal, not a silent refresh: a plan confirmed under one set of rules and run
under another was never confirmed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..runtime.base import canonical_hash

TEMPLATE_HINT = "rsu-vesting"


class Destination(str, Enum):
    """Where a confirmation field goes. Every field has exactly one.

    A field with no destination is copy — text a user reads and agrees to that
    reaches nothing. This is the confirmation-layer form of recognition without
    representation, which is the defect this whole project began with.
    """

    ENGINE_INPUT = "ENGINE_INPUT"
    RUNTIME_DECLARATION = "RUNTIME_DECLARATION"
    UNRESOLVED_QUESTION = "UNRESOLVED_QUESTION"
    NOT_MODELLED = "NOT_MODELLED"


#: Names that only exist after execution. Present on a declaration, they would
#: put a computed answer on a screen that precedes the computation.
FORBIDDEN_ON_DECLARATION = (
    "delivered_shares", "shares_delivered", "proceeds", "net_proceeds",
    "projected_concentration", "realized_concentration", "realized_weights",
    "expected_return", "expected_performance", "terminal_value",
)


@dataclass(frozen=True)
class DeclarationVersions:
    """The rule set a plan was confirmed under."""

    template_version: str = ""
    rsu_runtime_version: str = ""
    account_runtime_version: str = ""
    tax_runtime_version: str = ""
    corporate_action_runtime_version: str = ""
    scope_schema_version: str = ""

    @property
    def pin(self) -> str:
        return canonical_hash({f.name: getattr(self, f.name)
                               for f in fields(self)})

    def to_json(self) -> Dict[str, Any]:
        return {**{f.name: getattr(self, f.name) for f in fields(self)},
                "pin": self.pin}


class DeclarationVersionMismatch(RuntimeError):
    """Confirmed under one declaration set, attempted under another.

    Refused rather than refreshed. Re-reading the plan against newer rules would
    execute something the user never agreed to, and the screen that got their
    agreement would no longer describe it.
    """


@dataclass(frozen=True)
class RSUDeclaration:
    """The typed handoff. Every field is something the user or a runtime said."""

    grant_identity: Optional[str] = None
    employer_ticker: Optional[str] = None
    vest_schedule: Optional[Sequence[str]] = None
    gross_shares: Optional[float] = None
    gross_value: Optional[float] = None
    withholding_method: Optional[str] = None
    withholding_rate: Optional[float] = None
    corporate_action_ref: Optional[str] = None
    disposition_policy: Optional[str] = None
    blackout_schedule: Optional[Sequence[tuple]] = None
    allocation_policy: Optional[Any] = None
    concentration_cap: Optional[float] = None
    account_destination: Optional[str] = None
    tax_runtime_ref: Optional[str] = None
    account_runtime_ref: Optional[str] = None
    market_data_ref: Optional[str] = None

    versions: DeclarationVersions = field(default_factory=DeclarationVersions)

    def unresolved(self) -> Sequence[str]:
        """Declared fields with no value. Each becomes a question, not a
        default."""
        return tuple(f.name for f in fields(self)
                     if f.name != "versions" and getattr(self, f.name) is None)

    def to_json(self) -> Dict[str, Any]:
        payload = {f.name: getattr(self, f.name) for f in fields(self)
                   if f.name != "versions"}
        return {**payload, "versions": self.versions.to_json(),
                "unresolved": list(self.unresolved())}


def check_versions(confirmed: DeclarationVersions,
                   executing: DeclarationVersions) -> None:
    """Refuse a run whose rules moved since the user confirmed."""
    if confirmed.pin != executing.pin:
        moved = [f.name for f in fields(confirmed)
                 if getattr(confirmed, f.name) != getattr(executing, f.name)]
        raise DeclarationVersionMismatch(
            "DECLARATION_VERSION_MISMATCH: this plan was confirmed under a "
            f"different rule set ({', '.join(moved)} changed). Re-confirm it "
            "rather than running an interpretation nobody agreed to")
