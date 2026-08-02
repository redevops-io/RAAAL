"""What each stored record class is, who owns it, and when it goes.

Three data systems with genuinely different policies, kept apart because one
policy over all three would be wrong for at least two of them:

    workspace store   the user's durable research record
    trace store       operational telemetry, expendable and short-lived
    market data       governed by licence, not by user ownership

**The classification is checked against the schema, not trusted.** The registry
below is a declaration; `tests/test_retention.py` enumerates the tables SQLite
actually reports and fails when one is unclassified. Parametrising that test
from the registry would let a new table pass by never appearing — the same hole
the comparison-profile and diagnostic-destination guards had to close.

**"No longer visible" is not deletion.** A record hidden by a query predicate is
still a record. Deletion here enumerates every table, removes the rows, and then
verifies none remain; anything weaker is a claim the system cannot support.

**Indirect ownership is where a cascade forgets.** `plan_run` carries no owner
column and is reachable only through its plan, so a deletion written around
`WHERE owner = ?` would leave every run behind while reporting success.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

#: Bumped when a policy changes. Recorded on every deletion receipt so a receipt
#: says which rules it was carried out under.
RETENTION_POLICY_VERSION = "retention/workspace@1"


class DataClass(str, Enum):
    PERSONAL_RECORD = "PERSONAL_RECORD"
    """The user's own declarations and results. Deleted with the account."""

    PERSONAL_TELEMETRY = "PERSONAL_TELEMETRY"
    """Operational, derived from personal activity. Expires on a schedule."""

    SHARED_PUBLIC = "SHARED_PUBLIC"
    """Methodologies, runtime declarations, synthetic datasets. Survives a
    user's deletion; a personal *reference* to one does not."""

    LICENSED_SOURCE = "LICENSED_SOURCE"
    """Vendor data. Governed by the agreement, never by user ownership."""


class OwnerScope(str, Enum):
    DIRECT = "DIRECT"
    """Carries an `owner` column."""

    INDIRECT = "INDIRECT"
    """Reachable only through a parent. The case a cascade forgets."""

    UNSCOPED = "UNSCOPED"


class DeletionBehaviour(str, Enum):
    DELETE_WITH_OWNER = "DELETE_WITH_OWNER"
    EXPIRE_ON_SCHEDULE = "EXPIRE_ON_SCHEDULE"
    RETAIN = "RETAIN"


@dataclass(frozen=True)
class RecordClass:
    """One table's classification. Every field is required to be stated."""

    table: str
    data_class: DataClass
    owner_scope: OwnerScope
    retention_policy: str
    deletion_behaviour: DeletionBehaviour
    export_behaviour: str
    contains_sensitive_financial_data: bool
    contains_model_content: bool
    owner_column: Optional[str] = None
    reached_through: Optional[str] = None
    """For INDIRECT scope: the join that finds these rows. Named because a
    deletion that assumes a cascade deletes nothing and reports success."""

    sensitive_fields: Sequence[str] = ()

    def to_json(self) -> Dict[str, Any]:
        return {"table": self.table, "data_class": self.data_class.value,
                "owner_scope": self.owner_scope.value,
                "retention_policy": self.retention_policy,
                "deletion_behaviour": self.deletion_behaviour.value,
                "export_behaviour": self.export_behaviour,
                "contains_sensitive_financial_data":
                    self.contains_sensitive_financial_data,
                "contains_model_content": self.contains_model_content,
                "owner_column": self.owner_column,
                "reached_through": self.reached_through,
                "sensitive_fields": list(self.sensitive_fields)}


ACTIVE_ACCOUNT = "retained while the account is active"

#: Every workspace table. A table absent here fails the inventory test, which
#: reads the schema rather than this dictionary.
WORKSPACE_RECORDS: Mapping[str, RecordClass] = {
    one.table: one for one in (
        RecordClass(
            table="plan", data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=True,
            sensitive_fields=("stated_text", "scenario", "parse")),
        RecordClass(
            table="plan_run", data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.INDIRECT,
            reached_through="plan_run.plan_id -> plan.plan_id -> plan.owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=False,
            sensitive_fields=("result", "comparison")),
        RecordClass(
            table="proposal", data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=False),
        RecordClass(
            table="observation", data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=False),
        RecordClass(
            table="worksheet", data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=False),
        RecordClass(
            table="worksheet_intent", data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=True,
            sensitive_fields=("instruction", "structured_request")),
        RecordClass(
            table="worksheet_proposal", data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=False),
        RecordClass(
            table="planned_event", data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=False,
            sensitive_fields=("grant_ref", "asset", "expected_quantity",
                              "expected_value")),
        RecordClass(
            table="observed_event", data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=False,
            sensitive_fields=("asset", "quantity", "value", "evidence_refs")),
        RecordClass(
            table="event_reconciliation",
            data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=False),
        RecordClass(
            table="confirmation_event",
            data_class=DataClass.PERSONAL_TELEMETRY,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=False,
            sensitive_fields=("original_value", "final_value")),
    )
}

#: Categories of sensitive financial content, named so a classification can be
#: read against something. None of this is a name or a national identifier, and
#: all of it is sensitive.
SENSITIVE_CATEGORIES: Sequence[str] = (
    "employer name or ticker", "compensation and vest schedule",
    "contribution amounts", "account type", "holdings",
    "tax assumptions", "raw user instructions", "evidence references",
    "model prompts and responses",
)


def unclassified(tables: Sequence[str]) -> Sequence[str]:
    """Tables the schema reports and this registry does not classify."""
    return tuple(sorted(set(tables) - set(WORKSPACE_RECORDS)))


def owner_scoped_tables() -> Sequence[RecordClass]:
    """Everything a user deletion must reach, direct and indirect."""
    return tuple(one for one in WORKSPACE_RECORDS.values()
                 if one.deletion_behaviour is DeletionBehaviour.DELETE_WITH_OWNER)
