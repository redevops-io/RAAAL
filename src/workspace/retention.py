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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
class OwnershipPath:
    """How to reach an owner from a table, as data rather than prose.

    Executable metadata, not documentation. Deletion, export, tenant isolation
    and any later auditing consume the same graph, so a table whose ownership is
    described in a comment and special-cased in one function is a table the
    other three will get wrong.

    **The join must span the parent's whole key.** This originally took a single
    `local_key` and `parent_key`, which was correct while parent tables were
    identified by one column. Once tenant identity entered every key, a join on
    the id alone matched *both* tenants' parents — and a deletion for one owner
    removed the other's child rows while reporting the right counts. Nothing
    about that failure looks wrong from inside: the query is valid, it returns
    rows, and they are the wrong rows.

        plan_run.(owner, plan_id) -> plan.(owner, plan_id) -> plan.owner
    """

    local_key: Tuple[str, ...]
    parent_table: str
    parent_key: Tuple[str, ...]
    parent_owner_column: str

    def __post_init__(self) -> None:
        # Accept a bare string for either key and normalize, so a
        # single-column path stays readable at the declaration site.
        if isinstance(self.local_key, str):
            object.__setattr__(self, "local_key", (self.local_key,))
        if isinstance(self.parent_key, str):
            object.__setattr__(self, "parent_key", (self.parent_key,))
        if len(self.local_key) != len(self.parent_key):
            raise ValueError(
                f"{self.parent_table}: the join has {len(self.local_key)} "
                f"local column(s) and {len(self.parent_key)} parent column(s). "
                "A partial join matches more parents than it should, and the "
                "extra ones belong to other tenants.")

    def describe(self) -> str:
        pairs = ", ".join(f"{local} -> {self.parent_table}.{parent}"
                          for local, parent
                          in zip(self.local_key, self.parent_key))
        return f"{pairs} -> {self.parent_table}.{self.parent_owner_column}"

    def _on(self, table: str) -> str:
        return " AND ".join(
            f"{self.parent_table}.{parent} = {table}.{local}"
            for local, parent in zip(self.local_key, self.parent_key))

    def select(self, table: str) -> str:
        return (f"SELECT {table}.* FROM {table} JOIN {self.parent_table} "
                f"ON {self._on(table)} "
                f"WHERE {self.parent_table}.{self.parent_owner_column} = ?")

    def delete(self, table: str) -> str:
        columns = ", ".join(self.local_key)
        parents = ", ".join(self.parent_key)
        return (f"DELETE FROM {table} WHERE ({columns}) IN "
                f"(SELECT {parents} FROM {self.parent_table} "
                f"WHERE {self.parent_owner_column} = ?)")

    def to_json(self) -> Dict[str, Any]:
        return {"local_key": list(self.local_key),
                "parent_table": self.parent_table,
                "parent_key": list(self.parent_key),
                "parent_owner_column": self.parent_owner_column,
                "describes": self.describe()}


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
    ownership_path: Optional[OwnershipPath] = None
    """For INDIRECT scope: the join that finds these rows, as data.

    A deletion that assumes a cascade deletes nothing and reports success, and
    an ownership path written in a comment is one every consumer re-derives
    differently."""

    sensitive_fields: Sequence[str] = ()

    @property
    def reached_through(self) -> Optional[str]:
        return self.ownership_path.describe() if self.ownership_path else None

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
                "ownership_path": (self.ownership_path.to_json()
                                   if self.ownership_path else None),
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
            # Directly scoped since the ownership migration. It used to be
            # reachable only through its plan, which made every ownership
            # question a join and left deletion one forgotten cascade away from
            # keeping every run while reporting success. `OwnershipPath` keeps
            # its coverage in `tests/ownership_fixture.py` rather than by
            # holding a production table in a weaker shape.
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            contains_sensitive_financial_data=True,
            contains_model_content=False,
            sensitive_fields=("result", "comparison")),
        RecordClass(
            table="plan_migration", data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            # It carries a compiled scenario, which is derived from the user's
            # own description of their finances.
            contains_sensitive_financial_data=True,
            contains_model_content=False,
            sensitive_fields=("scenario",)),
        RecordClass(
            table="run_invalidation", data_class=DataClass.PERSONAL_RECORD,
            # PERSONAL_RECORD, like the run it names. It holds no content the
            # user wrote — a run id, a classification and a sentence written by
            # us — but it exists because this user ran something, and a record
            # that survives the deletion of the run it refers to would be a
            # dangling statement about a person whose data is gone.
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export",
            # The reason names what was wrong with an engine, never what the
            # user described. A withdrawal notice that quoted the plan would
            # reproduce the content it exists to stop being trusted.
            contains_sensitive_financial_data=False,
            contains_model_content=False,
            sensitive_fields=()),
        RecordClass(
            table="market_data_access_event",
            # PERSONAL_RECORD despite holding no personal content. It holds
            # digests, column names, row counts and timestamps — nothing the
            # user wrote. What makes it personal is that it exists *because*
            # this user ran something, and when it was resolved says so.
            # Classifying it SHARED_PUBLIC because the market data behind it is
            # shared would be exactly the reasoning the retention rule forbids:
            # a personal record is not retained merely because its subject is
            # shared.
            data_class=DataClass.PERSONAL_RECORD,
            owner_scope=OwnerScope.DIRECT, owner_column="owner",
            retention_policy=ACTIVE_ACCOUNT,
            deletion_behaviour=DeletionBehaviour.DELETE_WITH_OWNER,
            export_behaviour="included in a workspace export, so an exported "
                             "run stays verifiable",
            contains_sensitive_financial_data=False,
            contains_model_content=False),
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


def paths_not_spanning_their_parent_key(
        primary_keys: Mapping[str, Sequence[str]]) -> Sequence[str]:
    """Ownership paths that join on less than their parent's whole key.

    Takes the primary keys from the *database*, so this compares a declaration
    against the schema rather than against another declaration.

    A path joining on a subset matches every parent row sharing that subset —
    which, since tenant identity entered every key, means every tenant's. The
    resulting deletion is a valid query returning rows, so it reports plausible
    counts while removing another tenant's records. That is not hypothetical:
    the indirect fixture's original single-column path did exactly this.
    """
    wrong = []
    for record in WORKSPACE_RECORDS.values():
        path = record.ownership_path
        if path is None:
            continue
        expected = tuple(primary_keys.get(path.parent_table, ()))
        if expected and set(path.parent_key) != set(expected):
            wrong.append(
                f"{record.table}: joins {path.parent_table} on "
                f"{tuple(path.parent_key)} but its key is {expected}")
    return tuple(wrong)


def unclassified(tables: Sequence[str]) -> Sequence[str]:
    """Tables the schema reports and this registry does not classify."""
    return tuple(sorted(set(tables) - set(WORKSPACE_RECORDS)))


def owner_scoped_tables() -> Sequence[RecordClass]:
    """Everything a user deletion must reach, direct and indirect."""
    return tuple(one for one in WORKSPACE_RECORDS.values()
                 if one.deletion_behaviour is DeletionBehaviour.DELETE_WITH_OWNER)
