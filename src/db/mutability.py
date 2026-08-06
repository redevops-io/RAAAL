"""Which tables hold bodies that may never be rewritten, and what enforces it.

Several store methods document immutability — "immutable from here", "revisions
are never edited", "a plan revisited next year must show the result it actually
got". Those are claims about what the code will not do. This module is what
makes them checkable against the statements PostgreSQL actually receives.

    IMMUTABLE_ARTIFACT   the body is written once; a second write with a
                         different body is a conflict, never an overwrite
    MUTABLE_LIFECYCLE    an immutable body with a status beside it; the status
                         moves, the body does not
    MUTABLE_PROJECTION   derived or operational; re-derivable, so replacing it
                         loses nothing that cannot be recomputed

**Classified per column, not per table.** A worksheet proposal has an immutable
reviewed diff and a status that has to move from PROPOSED to ACCEPTED. Calling
the whole table mutable because one field changes would leave the diff
unprotected, which is the field that matters.

**The check reads captured statements, not source.** Nine prose-matching
failures in this codebase have all had the same shape: a pattern found in a
comment, a docstring or a variable name rather than in what ran. And the
statement that reaches PostgreSQL is not the one written in the store —
`INSERT OR REPLACE` is rewritten to `ON CONFLICT`, and it is the rewrite that
either overwrites a body or does not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


class TableClass(str, Enum):
    IMMUTABLE_ARTIFACT = "IMMUTABLE_ARTIFACT"
    MUTABLE_LIFECYCLE = "MUTABLE_LIFECYCLE"
    MUTABLE_PROJECTION = "MUTABLE_PROJECTION"


@dataclass(frozen=True)
class Mutability:
    """One table's write policy, with the columns it protects."""

    table: str
    kind: TableClass
    immutable_columns: Tuple[str, ...] = ()
    """Columns no UPDATE may assign and no conflict action may overwrite."""

    rationale: str = ""


#: The body columns of an artifact: its payload, and the hash taken over it.
#: Updating either is how a stored claim stops meaning what it said.
_BODY = ("payload", "content_hash")

#: Every table, classified. `tests/test_immutability.py` compares this against
#: the tables PostgreSQL reports and fails on one that is missing, so a new
#: table is unclassified rather than silently mutable.
TABLE_MUTABILITY: Mapping[str, Mutability] = {
    one.table: one for one in (
        Mutability(
            table="worksheet", kind=TableClass.IMMUTABLE_ARTIFACT,
            immutable_columns=("payload", "canonical_hash"),
            rationale="One row per revision. An UPDATE that changed a revision "
                      "would erase the history revisions exist to keep."),
        Mutability(
            table="planned_event", kind=TableClass.IMMUTABLE_ARTIFACT,
            immutable_columns=_BODY,
            rationale="An expectation that changed after the fact is not an "
                      "expectation."),
        Mutability(
            table="observed_event", kind=TableClass.IMMUTABLE_ARTIFACT,
            immutable_columns=_BODY,
            rationale="A correction is a new row that supersedes. Overwriting "
                      "the first erases that a correction happened, which is "
                      "part of the audit trail rather than noise in it."),
        Mutability(
            table="plan_run", kind=TableClass.IMMUTABLE_ARTIFACT,
            immutable_columns=("result", "comparison"),
            rationale="A run records the verdict a plan actually got. A plan "
                      "revisited next year must show that figure, not one "
                      "recomputed against rules that have since moved."),
        Mutability(
            table="plan_migration", kind=TableClass.IMMUTABLE_ARTIFACT,
            immutable_columns=("scenario", "content_hash", "from_compiler",
                               "to_compiler", "from_engine", "to_engine",
                               "authorized_by", "migrated_at", "old_run"),
            rationale="An authorisation that happened at a moment, and the "
                      "interpretation it authorised. Editing the scenario "
                      "would change what the owner agreed to after they "
                      "agreed to it; editing `authorized_by` or `migrated_at` "
                      "would move the consent. `new_run` is writable once, "
                      "because the record is created before the run it "
                      "names — the alternative is a run citing a migration "
                      "that does not exist yet."),
        Mutability(
            table="run_invalidation", kind=TableClass.IMMUTABLE_ARTIFACT,
            immutable_columns=("classification", "reason", "engine_version",
                               "invalidated_at"),
            rationale="A withdrawal is a historical statement: on this date we "
                      "determined this figure must not be read as a result. "
                      "Classifying it MUTABLE_LIFECYCLE was the lazy answer — "
                      "it protected no column, so the classification enforced "
                      "nothing. Overwriting `invalidated_at` on a re-run of "
                      "the sweep would move the date users were first told, "
                      "and rewriting the reason would let a withdrawal be "
                      "quietly softened. A revised judgement is a new "
                      "classification value, not an edit to the old one."),
        Mutability(
            table="market_data_access_event", kind=TableClass.IMMUTABLE_ARTIFACT,
            immutable_columns=("frame_digest", "provenance_digest",
                               "selected_columns", "row_count", "snapshot_id",
                               "access_decision", "accessed_at", "run_id",
                               "content_hash"),
            rationale="A historical fact about a delivery that happened. There "
                      "is no state it can legitimately move to, and every "
                      "column is the evidence — a mutable field here would be "
                      "a field that can be edited to make a run verify."),
        Mutability(
            table="plan", kind=TableClass.IMMUTABLE_ARTIFACT,
            immutable_columns=("scenario", "parse", "content_hash"),
            rationale="The compiled scenario and the stage 1 parse are pinned. "
                      "Recompiling a saved plan against a changed model would "
                      "silently alter a plan the user already confirmed."),
        Mutability(
            table="worksheet_proposal", kind=TableClass.MUTABLE_LIFECYCLE,
            immutable_columns=("payload", "source_revision"),
            rationale="The reviewed diff never changes; the outcome beside it "
                      "moves from PROPOSED exactly once."),
        Mutability(
            table="worksheet_intent", kind=TableClass.MUTABLE_LIFECYCLE,
            immutable_columns=("structured_request", "instruction_hash",
                               "chain_hash", "sequence"),
            rationale="The classification and its chain link are what a trial "
                      "total is derived from. The proposal reference and "
                      "status are filled in afterwards."),
        Mutability(
            table="proposal", kind=TableClass.MUTABLE_LIFECYCLE,
            immutable_columns=("payload",),
            rationale="A forward-tracking proposal's body is what was offered; "
                      "its status records what became of it."),
        Mutability(
            table="event_reconciliation", kind=TableClass.MUTABLE_PROJECTION,
            rationale="Derived from the planned and observed events, and "
                      "re-derivable from them. Replacing it loses nothing that "
                      "cannot be recomputed, and `reconciliation_view.verify` "
                      "is what checks a stored conclusion still follows."),
        Mutability(
            table="observation", kind=TableClass.MUTABLE_PROJECTION,
            rationale="A mission observation record, superseded by re-reading "
                      "its source."),
        Mutability(
            table="confirmation_event", kind=TableClass.MUTABLE_PROJECTION,
            rationale="Confirmation-screen telemetry. Expendable by policy."),
    )
}


# --------------------------------------------------------------------------
# statement analysis


_UPDATE = re.compile(r"^\s*UPDATE\s+(\w+)\s+SET\s+(.*?)(?:\s+WHERE\s|\s*$)",
                     re.IGNORECASE | re.DOTALL)
_INSERT = re.compile(r"^\s*INSERT\s+INTO\s+(\w+)", re.IGNORECASE)
_CONFLICT_ACTION = re.compile(
    r"ON\s+CONFLICT\s*\([^)]*\)\s*DO\s+UPDATE\s+SET\s+(.*?)(?:\s+WHERE\s|\s*$)",
    re.IGNORECASE | re.DOTALL)
_ASSIGNED = re.compile(r"(\w+)\s*=")

#: JSONB operations that edit a document in place. None of them belongs
#: anywhere near an immutable body — a new revision or a new row is the only
#: way an artifact's contents may change.
_IN_PLACE_JSON = (
    ("jsonb_set", re.compile(r"\bjsonb_set\s*\(", re.IGNORECASE)),
    ("jsonb_insert", re.compile(r"\bjsonb_insert\s*\(", re.IGNORECASE)),
    ("|| (concatenation)", re.compile(r"(payload|result|comparison|scenario)"
                                      r"\s*\|\|", re.IGNORECASE)),
    ("- (key removal)", re.compile(r"(payload|result|comparison|scenario)"
                                   r"\s*-\s*'", re.IGNORECASE)),
    ("#- (path removal)", re.compile(r"#-", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Violation:
    statement: str
    table: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover - diagnostics
        return f"{self.table}: {self.reason}\n    {self.statement.strip()[:160]}"


def _assigned_columns(clause: str) -> List[str]:
    return [name.lower() for name in _ASSIGNED.findall(clause)
            if name.lower() != "excluded"]


def inspect_statement(sql: str) -> List[Violation]:
    """Violations in one statement, as PostgreSQL would receive it."""
    found: List[Violation] = []

    update = _UPDATE.search(sql)
    if update:
        table, clause = update.group(1).lower(), update.group(2)
        policy = TABLE_MUTABILITY.get(table)
        if policy is not None:
            for column in _assigned_columns(clause):
                if column in policy.immutable_columns:
                    found.append(Violation(
                        sql, table,
                        f"UPDATE assigns {column!r}, which is an immutable "
                        f"body column ({policy.rationale})"))

    insert = _INSERT.search(sql)
    if insert:
        table = insert.group(1).lower()
        policy = TABLE_MUTABILITY.get(table)
        action = _CONFLICT_ACTION.search(sql)
        if policy is not None and action is not None:
            for column in _assigned_columns(action.group(1)):
                if column in policy.immutable_columns:
                    found.append(Violation(
                        sql, table,
                        f"ON CONFLICT DO UPDATE overwrites {column!r} — a "
                        "second write with a different body must be a conflict, "
                        f"not a replacement ({policy.rationale})"))

    for name, pattern in _IN_PLACE_JSON:
        if pattern.search(sql):
            table = (update.group(1).lower() if update
                     else insert.group(1).lower() if insert else "?")
            if TABLE_MUTABILITY.get(table) and \
                    TABLE_MUTABILITY[table].immutable_columns:
                found.append(Violation(
                    sql, table,
                    f"edits a JSON document in place with {name}; an immutable "
                    "body changes by new revision or new row only"))
    return found


def violations(statements: Iterable[str]) -> List[Violation]:
    """Every violation across a captured set of statements."""
    found: List[Violation] = []
    for statement in statements:
        found.extend(inspect_statement(statement))
    return found
