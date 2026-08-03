"""The workspace schema, defined once for every dialect that runs it.

Previously the schema was a SQLite DDL string, and three repair routines
(`_add_missing_columns`, `_relax_not_null`, `_widen_primary_keys`) rebuilt tables
on open to catch up databases created before a change. Those routines are not a
migration system: they have no ordering, no recorded state, no downgrade, and no
way to answer "which schema is this database at". `QUANTIFY_MIGRATION_HEAD` was
a required deployment fact with nothing to produce it.

This module is the single source of truth. Alembic generates migrations from it,
both dialects render DDL from it, and `tests/test_migration_parity.py` fails when
a migration and this metadata disagree — so the model and the migrations cannot
drift apart silently, which is the failure that makes a migration system worse
than none.

**The rationale comments moved with the columns.** Each one records a decision
that was expensive to reach — owner in the primary key, `trial_effect` nullable,
`effective_date` apart from `observed_at`. A schema rewrite that dropped them
would leave the constraints in place and the reasons nowhere, and the next
person to touch them would have only the shape to go on.
"""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

from .types import DecimalText, JsonText
from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    text,
)

#: Bumped when the schema's *shape* changes in a way a reader must know about.
#: Distinct from the Alembic revision, which identifies a migration; this
#: identifies the model those migrations are converging on.
SCHEMA_VERSION = "workspace-schema@1"

metadata = MetaData()


plan = Table(
    "plan", metadata,
    Column("plan_id", Text, primary_key=True),
    Column("owner", Text, nullable=False),
    Column("title", Text, nullable=False),
    Column("scenario", JsonText, nullable=False),
    Column("intent", Text),
    Column("stated_text", Text, nullable=False),
    Column("saved_at", Text, nullable=False),
    Column("rule_hash", Text, nullable=False),
    Column("content_hash", Text, nullable=False),
    # The stage 1 parse this plan was compiled from. Pinned rather than
    # re-derived: stage 1 may involve a language model, and recompiling a saved
    # plan against a model that has since changed would silently alter a plan
    # the user already read and confirmed.
    Column("parse", JsonText),
)

proposal = Table(
    "proposal", metadata,
    Column("proposal_id", Text, primary_key=True),
    Column("plan_id", Text, nullable=False),
    Column("owner", Text, nullable=False),
    Column("payload", JsonText, nullable=False),
    Column("generated_at", Text, nullable=False),
    Column("status", Text, nullable=False),
)

observation = Table(
    "observation", metadata,
    Column("observation_id", Text, primary_key=True),
    Column("plan_id", Text, nullable=False),
    Column("owner", Text, nullable=False),
    Column("observed_at", Text, nullable=False),
    Column("payload", JsonText, nullable=False),
)

# What was asked, before anything changed. Durable on its own, not folded into
# the proposal it produced.
#
# The planner's protections are history-dependent: "add 63-day volatility" is
# analytical the first time and parameter tuning the fourth, and the repetition
# signature is what stops repeated tuning hiding behind rephrasing. Held only in
# a request, that history dies between calls — a user could try three windows
# across three requests and each would arrive looking like the first. Trial
# accounting that resets is worse than none, because it reports a small number
# rather than no number.
worksheet_intent = Table(
    "worksheet_intent", metadata,
    Column("intent_id", Text, primary_key=True),
    Column("worksheet_id", Text, nullable=False),
    Column("owner", Text, nullable=False),
    Column("source_revision", Integer, nullable=False),
    Column("sequence", Integer, nullable=False),
    # Nullable on purpose. The durable semantic record is the structured intent
    # and `instruction_hash`; the raw sentence may carry holdings, salary or
    # employer detail and is subject to a stricter retention policy than the
    # classification derived from it.
    Column("instruction", Text),
    Column("instruction_hash", Text, nullable=False),
    Column("structured_request", JsonText, nullable=False),
    Column("edit_effect", Text, nullable=False),
    Column("selection_basis", Text, nullable=False),
    Column("repetition_signature", Text, nullable=False),
    Column("related_prior", JsonText, nullable=False),
    Column("results_visible", Integer, nullable=False),
    Column("alternatives", Integer, nullable=False),
    # Nullable deliberately. NULL means the planner could not read the
    # instruction, which is not the same as it costing nothing: an unclassified
    # request may have asked for one chart or for forty.
    Column("trial_effect", Integer),
    Column("planner_version", Text, nullable=False),
    # Each row chains to its predecessor. Editing a prior intent's
    # classification, or deleting one from the middle, breaks every successor's
    # hash — so a trial total derived from a doctored chain is detectably
    # derived from a doctored chain rather than quietly smaller.
    Column("chain_hash", Text, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("proposal_id", Text),
    Column("status", Text, nullable=False),
    # A reference into operational telemetry, which expires on its own
    # schedule. Nullable, and nothing may require it to resolve: a trace that
    # has aged out must not make a stored intent unreadable.
    Column("trace_id", Text),
)

# Ordering within a worksheet must be unique and gapless. Two intents claiming
# one position make the chain ambiguous, and an ambiguous chain cannot support
# a trial total anyone should rely on.
Index("worksheet_intent_sequence", worksheet_intent.c.worksheet_id,
      worksheet_intent.c.owner, worksheet_intent.c.sequence, unique=True)

# Worksheet proposals are immutable. Acceptance creates new artifacts and
# records the outcome here; it never rewrites the diff that was reviewed.
#
# Named apart from `proposal`, which is the mission forward-tracking artifact.
# `CREATE TABLE IF NOT EXISTS` on a name already taken is a silent no-op, so
# the columns would simply not have existed.
worksheet_proposal = Table(
    "worksheet_proposal", metadata,
    Column("proposal_id", Text, primary_key=True),
    Column("owner", Text, nullable=False),
    Column("worksheet_id", Text, nullable=False),
    Column("source_revision", Integer, nullable=False),
    Column("status", Text, nullable=False),
    Column("payload", JsonText, nullable=False),
    Column("created_at", Text, nullable=False),
    Column("resolved_at", Text),
    Column("actor", Text),
    Column("result_revision", Integer),
    Column("result_runs", JsonText),
    Column("trace_id", Text),
)

# One row per worksheet *revision*. Revisions are never edited, so the primary
# key spans the id and the revision: an UPDATE that lost a revision would erase
# the history that revisions exist to keep.
worksheet = Table(
    "worksheet", metadata,
    # Owner is part of the identity, not a filter applied afterwards.
    #
    # Keyed on (worksheet_id, revision) alone, a second owner could not create
    # a worksheet whose id another tenant already held: the write was refused,
    # and the refusal answered a question the requester was not entitled to
    # ask. Reads were correctly scoped, so nothing leaked on the way out — the
    # oracle was on the way in.
    Column("owner", Text, primary_key=True),
    Column("worksheet_id", Text, primary_key=True),
    Column("revision", Integer, primary_key=True),
    Column("payload", JsonText, nullable=False),
    Column("canonical_hash", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)

# Forward tracking, as three independent records. Inputs and conclusions are
# stored separately so the conclusion can be re-derived and compared, the same
# two-layer check the result context uses for presentability.
#
# Owner is in every key from the start. These rows carry employer names, grant
# references and compensation quantities, so a cross-tenant existence leak here
# is more sensitive than the worksheet-id one that prompted the rule.
planned_event = Table(
    "planned_event", metadata,
    Column("owner", Text, primary_key=True),
    Column("worksheet_id", Text, primary_key=True),
    Column("planned_event_id", Text, primary_key=True),
    Column("plan_revision", Integer, nullable=False),
    Column("grant_ref", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("expected_effective_date", Text, nullable=False),
    Column("asset", Text),
    Column("expected_quantity", DecimalText),
    Column("expected_value", DecimalText),
    Column("payload", JsonText, nullable=False),
    Column("matching_policy_version", Text, nullable=False),
    Column("source_ref", Text),
    Column("content_hash", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)

# `effective_date` and `observed_at` stay apart all the way through storage. A
# vest reported in July may have settled in June, and collapsing them would
# make an on-time vest look late for as long as the record survives.
observed_event = Table(
    "observed_event", metadata,
    Column("owner", Text, primary_key=True),
    Column("worksheet_id", Text, primary_key=True),
    Column("observed_event_id", Text, primary_key=True),
    Column("kind", Text, nullable=False),
    Column("effective_date", Text, nullable=False),
    Column("observed_at", Text, nullable=False),
    Column("asset", Text),
    Column("quantity", DecimalText),
    Column("value", DecimalText),
    Column("payload", JsonText, nullable=False),
    Column("evidence_refs", JsonText, nullable=False, server_default=text("'[]'")),
    Column("source", Text, nullable=False),
    Column("supersedes", Text),
    Column("content_hash", Text, nullable=False),
    Column("created_at", Text, nullable=False),
)

event_reconciliation = Table(
    "event_reconciliation", metadata,
    Column("owner", Text, primary_key=True),
    Column("worksheet_id", Text, primary_key=True),
    Column("reconciliation_id", Text, primary_key=True),
    Column("planned_event_id", Text),
    # Nullable: pending, overdue and confirmed-missing rows have no
    # observation, and a placeholder would read as one.
    Column("observed_event_id", Text),
    Column("status", Text, nullable=False),
    Column("payload", JsonText, nullable=False),
    Column("matching_policy_version", Text, nullable=False),
    Column("superseded_by", Text),
    Column("content_hash", Text, nullable=False),
    Column("derived_at", Text, nullable=False),
)

Index("reconciliation_worksheet", event_reconciliation.c.owner,
      event_reconciliation.c.worksheet_id, event_reconciliation.c.derived_at)

# Confirmation-screen telemetry. Structure now, conclusions later: intent
# cannot be inferred without users, but the first sessions are the ones worth
# measuring and they only happen once.
confirmation_event = Table(
    "confirmation_event", metadata,
    Column("event_id", Text, primary_key=True),
    Column("owner", Text, nullable=False),
    Column("occurred_at", Text, nullable=False),
    Column("kind", Text, nullable=False),
    Column("path", Text),
    Column("field", Text),
    Column("provenance", Text),
    Column("original_value", Text),
    Column("final_value", Text),
    Column("reason", Text),
    Column("compiler_version", Text),
    Column("defaults_ref", Text),
)

plan_run = Table(
    "plan_run", metadata,
    Column("run_id", Text, primary_key=True),
    Column("plan_id", Text, nullable=False),
    Column("ran_at", Text, nullable=False),
    Column("result", JsonText, nullable=False),
    Column("comparison", JsonText, nullable=False),
    # `plan_run` carries no owner column and is reachable only through its
    # plan. `src/workspace/retention.py` declares that ownership path so
    # deletion and export both find these rows; the foreign key is what makes
    # the path true rather than merely asserted.
)


def primary_key_columns(table_name: str) -> Tuple[str, ...]:
    """The primary key of a table, from the model rather than from a guess.

    The upsert translation needs a conflict target, and a hand-maintained list
    of conflict targets is a second schema that drifts from the first.
    """
    table = metadata.tables[table_name]
    return tuple(column.name for column in table.primary_key.columns)


def table_names() -> Tuple[str, ...]:
    return tuple(sorted(metadata.tables))
